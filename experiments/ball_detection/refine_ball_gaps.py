"""Recover short ball gaps using agreeing forward and reverse heatmap evidence."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from temporal_track import SCALE, xy


MAX_MISSING_FRAMES = 4
ANCHOR_SCORE = .4
PRIMARY_SCORE = .25
SUPPORT_SCORE = .1
AGREEMENT_RADIUS = 3  # Pixels at TrackNet's 512x288 resolution.
TRACK_RADIUS = 8
IMAGE_EDGE_MARGIN = 4


def confirmed_peak(evidence, expected):
    choices = []
    for direction, other in [('forward', 'reverse'), ('reverse', 'forward')]:
        support = [(xy(candidate), candidate['score']) for candidate in evidence.get(other, [])
                   if candidate['score'] >= SUPPORT_SCORE]
        for candidate in evidence.get(direction, []):
            if candidate['score'] < PRIMARY_SCORE:
                continue
            point = xy(candidate)
            error = float(np.linalg.norm(point - expected))
            if error > TRACK_RADIUS:
                continue
            corroboration = max((score for position, score in support
                                if np.linalg.norm(position - point) <= AGREEMENT_RADIUS), default=0.)
            if corroboration >= SUPPORT_SCORE:
                choices.append((error, -candidate['score'], -corroboration, direction, candidate))
    if not choices:
        return None
    error, _, support, direction, peak = min(choices, key=lambda choice: choice[:4])
    return {**peak, 'direction':direction, 'oppositeScore':-support, 'trackResidualPixels':error}


def image_edge_peak(evidence, expected):
    choices = []
    for candidate in evidence.get('uncutCrop', []):
        if candidate['score'] < ANCHOR_SCORE:
            continue
        point = xy(candidate)
        if np.min(np.minimum(point, SCALE - point)) > IMAGE_EDGE_MARGIN:
            continue
        residual = float(np.linalg.norm(point - expected))
        if residual > TRACK_RADIUS:
            continue
        support = [max((other['score'] for other in evidence.get(direction, [])
                        if np.linalg.norm(xy(other) - point) <= AGREEMENT_RADIUS), default=0.)
                   for direction in ('forward', 'reverse')]
        if min(support) >= SUPPORT_SCORE:
            choices.append((residual, candidate, min(support)))
    if not choices:
        return None
    residual, candidate, support = min(choices, key=lambda choice: (choice[0], -choice[1]['score']))
    return {**candidate, 'direction':'crop', 'oppositeScore':support, 'trackResidualPixels':residual}


def refine(baseline, probe):
    if baseline['video'] != probe['video'] or baseline['fps'] != probe['fps']:
        raise ValueError('Gap probe and baseline must describe the same video')
    result = copy.deepcopy(baseline)
    rows = result['trackingFrames']
    evidence = {row['frame']:row for row in probe['frames']}
    fps = baseline['fps']
    anchors = [index for index, row in enumerate(rows)
               if row['position'] is not None and row['confidence'] >= ANCHOR_SCORE]
    additions = []

    def add_observation(row, peak, track_id, reason):
        row.update(position={'x':peak['x'], 'y':peak['y']}, confidence=peak['score'], kind='observed',
                   reason=reason, gapEvidence=peak, trackId=track_id)
        additions.append({'timestamp':row['timestamp'], 'position':{'value':row['position'],
                          'source':'automatic', 'confidence':row['confidence']}})

    for start, end in zip(anchors, anchors[1:]):
        if not 1 < end - start <= MAX_MISSING_FRAMES + 1:
            continue
        left, right = rows[start], rows[end]
        duration = right['timestamp'] - left['timestamp']
        if not 0 < duration * fps <= end - start + .1:
            continue
        interval = rows[start:end+1]
        if any(current['sceneCut'] or current['frame'] != previous['frame'] + 1 or
               not .5 <= (current['timestamp'] - previous['timestamp']) * fps <= 1.5
               for previous, current in zip(interval, interval[1:])):
            continue
        a, b = xy(left['position']), xy(right['position'])
        for row in rows[start+1:end]:
            if row['position'] is not None or row['frame'] not in evidence:
                continue
            observed = evidence[row['frame']]
            if abs(observed['timestamp'] - row['timestamp']) > 1e-6:
                raise ValueError('Probe timestamp does not match the missing frame')
            fraction = (row['timestamp'] - left['timestamp']) / duration
            peak = confirmed_peak(observed, (1-fraction)*a + fraction*b)
            if peak is None:
                continue
            track_id = left['trackId'] if left['trackId'] == right['trackId'] else None
            add_observation(row, peak, track_id, 'confirmed-forward-reverse-peak')

    # At the physical image boundary, the guided crop can discard a real peak.
    # Require the previous original observation and corroboration in both full-frame passes.
    for previous, row in zip(baseline['trackingFrames'], rows[1:]):
        if (row['position'] is not None or row.get('search') != 'crop' or not row.get('prediction') or
                row['frame'] not in evidence or previous['position'] is None or previous['confidence'] < ANCHOR_SCORE or
                row['sceneCut'] or row['frame'] != previous['frame'] + 1 or
                not .5 <= (row['timestamp'] - previous['timestamp']) * fps <= 1.5):
            continue
        observed = evidence[row['frame']]
        if abs(observed['timestamp'] - row['timestamp']) > 1e-6:
            raise ValueError('Probe timestamp does not match the missing frame')
        peak = image_edge_peak(observed, xy(row['prediction']))
        if peak is not None:
            add_observation(row, peak, previous['trackId'], 'confirmed-image-edge-peak')
    result['ballObservations'] = sorted(result['ballObservations'] + additions, key=lambda observation: observation['timestamp'])
    result['summary'].update(observations=len(result['ballObservations']), gapRecoveries=len(additions), gapProbe=probe['summary'])
    result['model']['gapRefinement'] = {'method':'forward-reverse-heatmap-agreement-v1',
                                       'maxMissingFrames':MAX_MISSING_FRAMES, 'minimumAnchorScore':ANCHOR_SCORE,
                                       'primaryScore':PRIMARY_SCORE, 'supportScore':SUPPORT_SCORE,
                                       'agreementRadiusModelPixels':AGREEMENT_RADIUS, 'trackRadiusModelPixels':TRACK_RADIUS,
                                       'imageEdgeMarginModelPixels':IMAGE_EDGE_MARGIN}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('probe', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    baseline = args.baseline.read_bytes()
    probe = json.loads(args.probe.read_text(encoding='utf-8'))
    if (probe.get('format') != 'pongmetrics-ball-gap-probe' or probe.get('version') != 1 or
            probe.get('baselineSha256') != hashlib.sha256(baseline).hexdigest()):
        parser.error('Probe was not generated for this exact baseline')
    result = refine(json.loads(baseline), probe)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
