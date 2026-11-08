"""Evaluate labeled localization, or explicitly measure model agreement only."""
import argparse
import json
from pathlib import Path

import numpy as np


def index_observations(data):
    fps = data['fps']
    result = {}
    for o in data['ballObservations']:
        key = round(o['timestamp']*fps)
        if abs(o['timestamp']*fps-key) > .05:
            raise ValueError('This evaluation uses CFR frame labels; irregular timestamps require timestamp labels')
        if key in result:
            raise ValueError('Duplicate observation frame; CFR evaluation requires unique frame keys')
        result[key] = o['position']['value']
    return result


def continuity(data):
    keys = sorted(index_observations(data))
    runs = []
    for k in keys:
        if not runs or k != runs[-1][-1]+1:
            runs.append([k])
        else:
            runs[-1].append(k)
    return dict(observations=len(keys), contiguousRuns=len(runs),
                longestRunFrames=max(map(len, runs), default=0),
                meanRunFrames=float(np.mean(list(map(len, runs)))) if runs else 0,
                gapFramesBetweenObservations=sum(b-a-1 for a, b in zip(keys, keys[1:])),
                note='Continuity measures coverage, not accuracy or correct identity.')


def evaluate(data, labels, tolerance=5.):
    if data['video'] != labels['video'] or data['fps'] != labels['fps']:
        raise ValueError('Video metadata and FPS must match the evaluation labels')
    predictions = index_observations(data)
    tp = fp = fn = tn = 0
    errors, failures = [], []
    seen = set()
    for label in labels['frames']:
        frame = label['frame']
        if frame in seen:
            raise ValueError('Duplicate label frame')
        seen.add(frame)
        if label['status'] == 'uncertain':
            continue
        point = predictions.get(frame)
        if label['status'] == 'visible':
            target = label['position']
            distance = None if point is None else float(np.linalg.norm(
                (np.array([point['x'], point['y']])-np.array([target['x'], target['y']]))*[640, 360]))
            if distance is not None:
                errors.append(distance)
            if distance is not None and distance <= tolerance:
                tp += 1
            else:
                fn += 1
                fp += point is not None
                failures.append(dict(frame=frame, reason='missed' if point is None else 'wrong-location', pixels=distance))
        elif label['status'] == 'absent':
            if point is None:
                tn += 1
            else:
                fp += 1
                failures.append(dict(frame=frame, reason='false-positive'))
        else:
            raise ValueError('Labels must be visible, absent, or uncertain')
    return dict(truePositives=tp, falsePositives=fp, falseNegatives=fn, trueNegatives=tn,
                precision=tp/(tp+fp) if tp+fp else None, recall=tp/(tp+fn) if tp+fn else None,
                tolerancePixelsAt640x360=tolerance, evaluatedFrames=tp+fn+tn+sum(f['reason']=='false-positive' for f in failures),
                medianLocalizationErrorPixels=float(np.median(errors)) if errors else None,
                failures=failures, continuity=continuity(data),
                identityConsistency='Not measurable without labeled distractors/identity sequences; one-ball tracklets are not identity ground truth.')


def agreement(reference, data, tolerance=5.):
    if reference['video'] != data['video'] or reference['fps'] != data['fps']:
        raise ValueError('Reference must use the same video and FPS')
    a, b = index_observations(reference), index_observations(data)
    distances = {k: float(np.linalg.norm((np.array([a[k]['x'], a[k]['y']])-np.array([b[k]['x'], b[k]['y']]))*[640, 360])) for k in a.keys() & b.keys()}
    return dict(matched=sum(d <= tolerance for d in distances.values()), referenceObservations=len(a),
                missingFrames=sorted(a.keys()-b.keys()), extraFrames=sorted(b.keys()-a.keys()),
                disagreeingFrames={k:d for k,d in distances.items() if d > tolerance},
                note='Agreement with a detector is not labeled precision/recall.')


def baseline(cache):
    observations = []
    for row in cache['frames']:
        strong = [c for c in row['candidates'] if c['threshold'] == .5]
        if len(strong) == 1:
            c = strong[0]
            observations.append(dict(timestamp=row['timestamp'], position=dict(value=dict(x=c['x'], y=c['y']), source='automatic', confidence=c['score'])))
    return dict(format='pongmetrics-ball-detections', version=1, coordinateSystem='normalized-video-image',
                video=cache['video'], fps=cache['fps'], ballObservations=observations)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('detections', type=Path)
    p.add_argument('--labels', type=Path)
    p.add_argument('--reference', type=Path)
    p.add_argument('--baseline-output', type=Path, help='Input is a candidate cache; export original single-component rule')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    data = json.loads(a.detections.read_text(encoding='utf8'))
    if a.baseline_output:
        data = baseline(data)
        a.baseline_output.write_text(json.dumps(data, indent=2), encoding='utf8')
    report = dict(continuity=continuity(data))
    if a.labels:
        report['labeled'] = evaluate(data, json.loads(a.labels.read_text(encoding='utf8')))
    if a.reference:
        report['agreement'] = agreement(json.loads(a.reference.read_text(encoding='utf8')), data)
    if a.output:
        a.output.write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
