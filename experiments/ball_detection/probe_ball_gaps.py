"""Probe missing TrackNet frames with both temporal directions, without using labels."""
import argparse
from collections import deque
from contextlib import closing
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from neural_candidates import StreamingDetector
from refine_ball_gaps import refinement_targets


def probe(detector, rows, *, all_nearby=False):
    # Include short holes, starts and ends, and any nearby labelled negatives
    # incidentally covered by that rule. No ground truth chooses probe frames.
    observed = {row['frame'] for row in rows if row['position'] is not None}
    missing = {row['frame']:row for row in rows if row['position'] is None and
               any(row['frame'] + offset in observed for offset in range(-4, 5))}
    gaps, edges = refinement_targets(rows, detector.fps)
    if not all_nearby:
        missing = {row['frame']:row for row in rows if row['frame'] in gaps or row['frame'] in edges}
    if not missing:
        return []
    buffered = deque()
    results = []

    def candidates(history, center=None):
        return detector.detect(history, center, probe=True)

    def inspect():
        frame, timestamp, history, _ = buffered[0]
        saved = missing.get(frame)
        if saved is None:
            return
        if abs(saved['timestamp'] - timestamp) > 1e-6:
            raise ValueError('Source frame timestamps disagree with baseline')
        entry = {'frame':frame, 'timestamp':timestamp}
        future = [history[-1]]
        for next_frame, next_time, next_history, cut in list(buffered)[1:]:
            expected_time = timestamp + len(future) / detector.fps
            if cut or next_frame != frame + len(future) or abs(next_time - expected_time) > .5 / detector.fps:
                break
            future.append(next_history[-1])
        if len(future) == 4 or all_nearby:
            entry['forward'] = candidates(history)
        if len(future) == 4:
            entry['reverse'] = candidates(future[::-1])
        if (all_nearby or frame in edges) and 'forward' in entry and saved.get('search') == 'crop' and saved.get('prediction'):
            point = saved['prediction']
            entry['uncutCrop'] = candidates(history, np.clip([point['x'], point['y']], 0, 1))
        results.append(entry)
        if len(results) % 20 == 0:
            print(f'{len(results)}/{len(missing)} gap frames probed', flush=True)

    with closing(detector.frames()) as frames:
        for row in frames:
            buffered.append(row)
            if len(buffered) == 4:
                inspect()
                if len(results) == len(missing):
                    return results
                buffered.popleft()
    while buffered:
        inspect()
        buffered.popleft()
    if len(results) != len(missing):
        raise ValueError('Missing decoded probe frames')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('--head-adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=('auto', 'torch', 'onnx'), default='auto')
    parser.add_argument('--all-nearby', action='store_true', help='Reproduce the broader probe schedule for paired comparisons')
    args = parser.parse_args()
    baseline_bytes = args.baseline.read_bytes()
    baseline = json.loads(baseline_bytes)
    model = baseline['model']
    if model.get('nativeZoom') or model.get('analyzedStart') != 0:
        parser.error('Probe requires a zero-start baseline with standard crop geometry')
    cv2.setNumThreads(1)
    began = time.perf_counter()
    detector = StreamingDetector(args.video, end=model['analyzedEnd'], background_samples=model['backgroundSamples'],
                                 candidate_floor=model['candidateFloor'], head_adapter=args.head_adapter, backend=args.backend)
    if detector.video != baseline['video'] or detector.fps != baseline['fps']:
        parser.error('Video or frame rate does not match baseline')
    if detector.weights_hash != model['weightsSha256'] or detector.head_adapter_hash != model['headAdapterSha256']:
        parser.error('Checkpoint and adapter must match the baseline')
    frames = probe(detector, baseline['trackingFrames'], all_nearby=args.all_nearby)
    summary = {'frames':len(frames), 'calls':detector.calls, 'inferenceSeconds':detector.inference_seconds,
               'totalSeconds':time.perf_counter()-began, 'backend':detector.backend, 'cpuThreads':detector.cpu_threads}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {'format':'pongmetrics-ball-gap-probe', 'version':1,
               'baselineSha256':hashlib.sha256(baseline_bytes).hexdigest(),
               'video':detector.video, 'fps':detector.fps, 'summary':summary, 'frames':frames}
    args.output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
