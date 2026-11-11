"""Verify recorded ball outputs before preferring a local ONNX runtime."""
import argparse
import hashlib
import json
from pathlib import Path

from neural_candidates import ROOT
from onnx_ball import bundle_path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(reference, candidate):
    if reference['video'] != candidate['video'] or reference['fps'] != candidate['fps']:
        raise ValueError('Comparison videos and frame rates must match')
    settings = ('candidateFloor', 'weakPeaks', 'nativeZoom', 'strongScore', 'acquisitionScore',
                'backgroundSamples', 'temporalOrder', 'analyzedStart', 'analyzedEnd')
    if any(reference.get('model', {}).get(key) != candidate.get('model', {}).get(key) for key in settings):
        raise ValueError('Detection or tracking settings changed')
    original = reference['ballObservations']
    updated = candidate['ballObservations']
    if len(original) != len(updated):
        raise ValueError('Observation count changed')
    maximum_error = 0.
    for before, after in zip(original, updated):
        if before['timestamp'] != after['timestamp'] or before['position']['value'] != after['position']['value']:
            raise ValueError('Observation timestamp or center changed')
        a, b = before['position']['confidence'], after['position']['confidence']
        maximum_error = max(maximum_error, abs(a - b))
        if abs(a - b) > 1e-5 or any((a >= step/20) != (b >= step/20) for step in range(21)):
            raise ValueError('Confidence changed beyond tolerance or crossed a UI score step')
    fields = ('frame', 'timestamp', 'position', 'reason', 'search', 'trackId', 'sceneCut')
    before_rows, after_rows = reference['trackingFrames'], candidate['trackingFrames']
    if len(before_rows) != len(after_rows):
        raise ValueError('Processed frame count changed')
    if any(any(a.get(key) != b.get(key) for key in fields) for a, b in zip(before_rows, after_rows)):
        raise ValueError('Tracking decisions or search schedule changed')
    return {'frames':len(before_rows), 'observations':len(original), 'maxConfidenceDifference':maximum_error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair', nargs=2, type=Path, action='append', required=True, metavar=('REFERENCE', 'CANDIDATE'))
    parser.add_argument('--reference-adapter', type=Path, required=True)
    parser.add_argument('--candidate-adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prefer-onnx', action='store_true')
    args = parser.parse_args()
    reference_adapter = json.loads(args.reference_adapter.read_bytes())
    candidate_adapter = json.loads(args.candidate_adapter.read_bytes())
    for key in ('format', 'baseWeightsSha256', 'weights', 'bias'):
        if reference_adapter[key] != candidate_adapter[key]:
            parser.error('Head adapter coefficients must be identical')
    results = []
    runtimes = set()
    for reference_path, candidate_path in args.pair:
        reference = json.loads(reference_path.read_bytes())
        candidate = json.loads(candidate_path.read_bytes())
        model = candidate['model']
        if reference['model']['weightsSha256'] != model['weightsSha256']:
            parser.error('Original checkpoint changed')
        if reference['model']['headAdapterSha256'] != sha(args.reference_adapter) or model['headAdapterSha256'] != sha(args.candidate_adapter):
            parser.error('Adapters do not match the recorded runs')
        if model.get('inferenceBackend') != 'onnx':
            parser.error('Candidate must use ONNX inference')
        metrics = compare(reference, candidate)
        runtimes.add((model['weightsSha256'], model['headAdapterSha256'], model['cpuThreads']))
        results.append({'reference':str(reference_path), 'candidate':str(candidate_path),
                        'referenceSha256':sha(reference_path), 'candidateSha256':sha(candidate_path), **metrics})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')
    if args.prefer_onnx:
        import onnxruntime as ort
        if len(runtimes) != 1:
            parser.error('All validated pairs must use the same model and thread count')
        weights, adapter, threads = runtimes.pop()
        graph = bundle_path(ROOT, weights, adapter)
        manifest_path = graph.with_suffix('.json')
        manifest = json.loads(manifest_path.read_bytes())
        if sha(graph) != manifest['onnxSha256']:
            parser.error('ONNX graph hash changed')
        manifest.update(preferredRuntime=ort.__version__, preferredThreads=threads,
                        validationSha256=sha(args.output))
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
