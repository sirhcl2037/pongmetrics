"""Cache sparse reviewed image features, then adapt only TrackNet's final 1x1 head.

The backbone is frozen. Sampling follows a recorded default-tracker input trace.
Only small feature vectors are retained, not whole clips or dense feature maps.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as functional

from neural_candidates import StreamingDetector
from evaluate_manual_labels import evaluate


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample_features(features, heatmap, label, origin, rng):
    height, width = heatmap.shape
    mask = np.ones((height, width), dtype=bool)
    positive = []
    targets = []
    if label['visibility'] == 'uncertain':
        return None
    if label['visibility'] == 'visible':
        point = label['position']['value']
        x, y = point['x']*512-origin[0], point['y']*288-origin[1]
        if not (0 <= x < width and 0 <= y < height):
            return None
        yy, xx = np.ogrid[:height, :width]
        mask[(xx-x)**2+(yy-y)**2 <= 12**2] = False
        # Clicks on tiny blurred balls are approximate. Anchor the positive patch
        # to the strongest real response in a bounded neighborhood of the click.
        x0, x1 = max(0, int(x)-8), min(width, int(x)+9)
        y0, y1 = max(0, int(y)-8), min(height, int(y)+9)
        patch = heatmap[y0:y1, x0:x1]
        py, px = np.unravel_index(int(np.argmax(patch)), patch.shape)
        px, py = px+x0, py+y0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if 0 <= px+dx < width and 0 <= py+dy < height:
                    positive.append((py+dy)*width+px+dx)
                    targets.append(float(np.exp(-(dx*dx+dy*dy)/4)))
    available = np.flatnonzero(mask.ravel())
    random = rng.choice(available, min(64, len(available)), replace=False)
    hard = available[np.argsort(heatmap.ravel()[available])[-16:]]
    negative = np.unique(np.concatenate([random, hard]))
    indices = np.array([*positive, *negative], dtype=np.int64)
    values = features.reshape(features.shape[0], -1)[:, indices].T.copy()
    target = np.array([*targets, *np.zeros(len(negative))], dtype=np.float32)
    weights = np.where(target > 0, 4., 1.).astype(np.float32)
    return values, target, weights


def cache(args):
    began = time.perf_counter()
    labels = json.loads(args.labels.read_text(encoding='utf-8'))
    trace = json.loads(args.trace.read_text(encoding='utf-8'))
    detector = StreamingDetector(args.video)
    for key in ['name', 'size', 'width', 'height']:
        if detector.video[key] != labels['video'][key] or detector.video[key] != trace['video'][key]:
            raise ValueError(f'Source {key} mismatch')
    if labels['fps'] != detector.fps or trace['fps'] != detector.fps:
        raise ValueError('FPS mismatch')
    annotated = {round(row['timestamp']*detector.fps):row for row in labels['labels']}
    rows = {row['frame']:row for row in trace['trackingFrames']}
    if len(annotated) != len(labels['labels']):
        raise ValueError('Duplicate annotated frame')
    captured = {}
    def before(_, inputs):
        captured['features'] = inputs[0][0].detach()
    def after(_, inputs, output):
        captured['heatmap'] = torch.sigmoid(output[0, -1].detach())
    pre = detector.model.predictor.register_forward_pre_hook(before)
    post = detector.model.predictor.register_forward_hook(after)
    rng = np.random.default_rng(42)
    values, targets, weights = [], [], []
    used_frames = []
    try:
        for frame, timestamp, history, _ in detector.frames():
            if frame not in annotated or frame not in rows:
                continue
            center = None
            origin = [0, 0]
            row = rows[frame]
            if row.get('search') == 'crop':
                prediction = row['prediction']
                center = np.clip([prediction['x'], prediction['y']], 0, 1)
                origin = np.clip(np.rint(center*[512, 288]-[128, 72]), [0, 0], [256, 144])
            detector.detect(history, center)
            result = sample_features(captured['features'].numpy(), captured['heatmap'].numpy(), annotated[frame], origin, rng)
            if result:
                features, target, weight = result
                values.append(features.astype(np.float16))
                targets.append(target)
                weights.append(weight)
                used_frames.append(frame)
            captured.clear()
            if frame % 25 == 0:
                print(f'{frame} frames sampled; {sum(len(t) for t in targets)} sparse vectors', flush=True)
    finally:
        pre.remove()
        post.remove()
    if not values:
        raise ValueError('No reviewed frames matched the training trace')
    metadata = dict(format='pongmetrics-head-features', version=1, video=detector.video,
                    labelSha256=sha(args.labels), traceSha256=sha(args.trace), baseWeightsSha256=detector.weights_hash,
                    usedFrames=used_frames, seconds=time.perf_counter()-began,
                    sampling=dict(seed=42, clickRadiusModelPixels=8, negativeExclusionRadius=12,
                                  positivePatch=3, randomNegatives=64, hardNegatives=16))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, features=np.concatenate(values), targets=np.concatenate(targets),
                        weights=np.concatenate(weights), baseWeight=detector.model.predictor.weight[-1, :, 0, 0].detach().numpy(),
                        baseBias=detector.model.predictor.bias[-1].detach().numpy(), metadata=json.dumps(metadata))
    print(json.dumps({**metadata, 'usedFrames':len(used_frames)}, indent=2), flush=True)


def mine(args):
    """Add hard negative pixels from mistakes on the training rally only."""
    labels = json.loads(args.labels.read_text(encoding='utf-8'))
    trace = json.loads(args.trace.read_text(encoding='utf-8'))
    adapter = json.loads(args.adapter.read_text(encoding='utf-8'))
    with np.load(args.features) as data:
        arrays = {key:data[key].copy() for key in data.files if key != 'metadata'}
        metadata = json.loads(str(data['metadata']))
    if metadata['labelSha256'] != sha(args.labels) or adapter['training']['labelSha256'] != sha(args.labels):
        raise ValueError('Hard-negative mining must use the training labels, never validation labels')
    if trace.get('model', {}).get('headAdapterSha256') != sha(args.adapter):
        raise ValueError('Mining trace must come from the supplied adapter')
    failures = evaluate(trace, labels, 0)['referenceFailures']
    selected = {row['frame'] for row in failures if row['reason'] == 'false-positive'}
    if not selected:
        raise ValueError('No false-positive training frames to mine')
    detector = StreamingDetector(args.video)
    for key in ['name','size','width','height']:
        if detector.video[key] != labels['video'][key]:
            raise ValueError('Mining source video mismatch')
    if detector.weights_hash != adapter['baseWeightsSha256']:
        raise ValueError('Original checkpoint mismatch')
    rows = {row['frame']:row for row in trace['trackingFrames']}
    captured = {}
    def capture(_, inputs):
        captured['features'] = inputs[0][0].detach()
    hook = detector.model.predictor.register_forward_pre_hook(capture)
    extra = []
    try:
        for frame, _, history, _ in detector.frames():
            if frame not in selected:
                continue
            row = rows[frame]
            point = row.get('prediction')
            center = np.clip([point['x'],point['y']],0,1) if row.get('search') == 'crop' else None
            detector.detect(history,center)
            features = captured['features'].numpy().reshape(64,-1)
            logits = np.asarray(adapter['weights'])@features+adapter['bias']
            hardest = np.argpartition(logits,-512)[-512:]
            extra.append(features[:,hardest].T.astype(np.float16))
            captured.clear()
    finally:
        hook.remove()
    added = np.concatenate(extra)
    arrays['features'] = np.concatenate([arrays['features'], added])
    arrays['targets'] = np.concatenate([arrays['targets'], np.zeros(len(added),dtype=np.float32)])
    arrays['weights'] = np.concatenate([arrays['weights'], np.full(len(added),4,dtype=np.float32)])
    metadata['hardNegativeMining'] = dict(frames=sorted(selected), vectors=len(added), adapterSha256=sha(args.adapter),
                                        traceSha256=sha(args.trace), weight=4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays, metadata=json.dumps(metadata))
    print(json.dumps(metadata['hardNegativeMining'],indent=2))


def train(args):
    torch.set_num_threads(4)
    torch.manual_seed(42)
    with np.load(args.features) as data:
        features = torch.tensor(data['features'].astype(np.float32))
        targets = torch.tensor(data['targets'])
        importance = torch.tensor(data['weights'])
        base_weight = torch.tensor(data['baseWeight'])
        base_bias = torch.tensor(data['baseBias'])
        metadata = json.loads(str(data['metadata']))
    if args.mined_negative_weight is not None:
        mining = metadata.get('hardNegativeMining')
        if not mining or not 0 < args.mined_negative_weight <= 10:
            raise ValueError('A positive mined-negative weight requires a mined cache')
        importance[-mining['vectors']:] = args.mined_negative_weight
        mining['weight'] = args.mined_negative_weight
    mean = features.mean(0)
    scale = features.std(0).clamp_min(1e-3)
    inputs = (features-mean)/scale
    if getattr(args,'residual',False):
        base_logits = features@base_weight+base_bias
        first = torch.nn.Linear(features.shape[1],8)
        last = torch.nn.Linear(8,1)
        torch.nn.init.normal_(first.weight, std=.05)
        torch.nn.init.zeros_(first.bias)
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)
        parameters = [*first.parameters(),*last.parameters()]
        optimizer = torch.optim.Adam(parameters,lr=.01)
        for epoch in range(200):
            optimizer.zero_grad()
            delta = 3*torch.tanh(last(torch.relu(first(inputs))).squeeze(1)/3)
            loss = (functional.binary_cross_entropy_with_logits(base_logits+delta,targets,reduction='none')*importance).sum()/importance.sum()
            loss = loss+.02*delta.square().mean()+.0001*sum(p.square().mean() for p in parameters)
            loss.backward()
            optimizer.step()
            if epoch%50 == 0:
                print(f'Residual epoch {epoch}: {float(loss.detach()):.5f}',flush=True)
        first_weight = first.weight.detach()/scale
        first_bias = first.bias.detach()-(first_weight*mean).sum(1)
        adapter = dict(format='pongmetrics-residual-head-adapter',version=1,
                       baseWeightsSha256=metadata['baseWeightsSha256'],weights=base_weight.tolist(),bias=float(base_bias),
                       hiddenWeights=first_weight.tolist(),hiddenBias=first_bias.tolist(),
                       outputWeights=last.weight.detach().flatten().tolist(),outputBias=float(last.bias.detach()[0]),residualCap=3,
                       training=metadata,trainingFeatureSha256=sha(args.features),
                       finalTrainingObjective=float(loss.detach()),
                       note='Frozen backbone and base classifier, eight-unit residual head. Experimental until cross-rally validation.')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(adapter,indent=2),encoding='utf-8')
        return
    initial_weight = base_weight*scale
    initial_bias = base_bias+(base_weight*mean).sum()
    weight = torch.nn.Parameter(initial_weight.clone())
    bias = torch.nn.Parameter(initial_bias.clone())
    optimizer = torch.optim.LBFGS([weight, bias], max_iter=100, line_search_fn='strong_wolfe')
    def objective():
        logits = inputs@weight+bias
        loss = (functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')*importance).sum()/importance.sum()
        return loss+args.anchor*((weight-initial_weight).square().mean()+(bias-initial_bias).square())
    before = float(objective().detach())
    def closure():
        optimizer.zero_grad()
        loss = objective()
        loss.backward()
        return loss
    optimizer.step(closure)
    actual_weight = weight.detach()/scale
    actual_bias = bias.detach()-(actual_weight*mean).sum()
    adapter = dict(format='pongmetrics-linear-head-adapter', version=1,
                   baseWeightsSha256=metadata['baseWeightsSha256'], weights=actual_weight.tolist(), bias=float(actual_bias),
                   training=metadata, objectiveBefore=before, objectiveAfter=float(objective().detach()), anchor=args.anchor,
                   trainingFeatureSha256=sha(args.features),
                   note='Frozen backbone; last output channel only. Requires separate-rally validation before use.')
    if not np.isfinite(adapter['weights']).all() or not np.isfinite(adapter['bias']):
        raise ValueError('Non-finite trained head')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(adapter, indent=2), encoding='utf-8')
    print(json.dumps({key:adapter[key] for key in ['objectiveBefore','objectiveAfter','anchor']}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    extract = commands.add_parser('cache')
    extract.add_argument('--video', type=Path, required=True)
    extract.add_argument('--labels', type=Path, required=True)
    extract.add_argument('--trace', type=Path, required=True)
    extract.add_argument('--output', type=Path, required=True)
    fit = commands.add_parser('train')
    fit.add_argument('--features', type=Path, required=True)
    fit.add_argument('--anchor', type=float, default=.05)
    fit.add_argument('--mined-negative-weight', type=float)
    fit.add_argument('--residual', action='store_true')
    fit.add_argument('--output', type=Path, required=True)
    negatives = commands.add_parser('mine')
    for name in ['video','labels','trace','adapter','features','output']:
        negatives.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'cache':
        cache(args)
    elif args.command == 'mine':
        mine(args)
    else:
        if args.anchor <= 0:
            parser.error('Positive regularization is required')
        train(args)


if __name__ == '__main__':
    main()
