"""Probe missing TrackNet frames with both temporal directions, without using labels."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from neural_candidates import StreamingDetector, extract_candidates, extract_weak_peaks


def probe(detector, rows):
    # Include short holes, starts and ends, and any nearby labelled negatives
    # incidentally covered by that rule. No ground truth chooses probe frames.
    observed = {row['frame'] for row in rows if row['position'] is not None}
    missing = {row['frame']:row for row in rows if row['position'] is None and
               any(row['frame'] + offset in observed for offset in range(-4, 5))}
    buffered = deque()
    results = []
    captured = None

    def capture(_, __, output):
        nonlocal captured
        captured = output[0][0, 0].numpy()

    def candidates(history, center=None):
        detector.detect(history, center)
        offset = (0, 0)
        if center is not None:
            offset = np.clip(np.rint(np.asarray(center) * [512, 288] - [128, 72]), [0, 0], [256, 144]).astype(int)
        return extract_candidates(captured, offset, detector.candidate_floor) + extract_weak_peaks(captured, offset)

    def inspect():
        frame, timestamp, history, _ = buffered[0]
        saved = missing.get(frame)
        if saved is None:
            return
        if abs(saved['timestamp'] - timestamp) > 1e-6:
            raise ValueError('Source frame timestamps disagree with baseline')
        entry = {'frame':frame, 'timestamp':timestamp, 'forward':candidates(history)}
        future = [history[-1]]
        for next_frame, next_time, next_history, cut in list(buffered)[1:]:
            expected_time = timestamp + len(future) / detector.fps
            if cut or next_frame != frame + len(future) or abs(next_time - expected_time) > .5 / detector.fps:
                break
            future.append(next_history[-1])
        if len(future) == 4:
            entry['reverse'] = candidates(future[::-1])
        if saved.get('search') == 'crop' and saved.get('prediction'):
            point = saved['prediction']
            entry['uncutCrop'] = candidates(history, np.clip([point['x'], point['y']], 0, 1))
        results.append(entry)
        if len(results) % 20 == 0:
            print(f'{len(results)}/{len(missing)} gap frames probed', flush=True)

    handle = detector.model.register_forward_hook(capture)
    try:
        for row in detector.frames():
            buffered.append(row)
            if len(buffered) == 4:
                inspect()
                buffered.popleft()
        while buffered:
            inspect()
            buffered.popleft()
    finally:
        handle.remove()
    if len(results) != len(missing):
        raise ValueError('Missing decoded probe frames')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('--head-adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    baseline_bytes = args.baseline.read_bytes()
    baseline = json.loads(baseline_bytes)
    model = baseline['model']
    if model.get('nativeZoom') or model.get('analyzedStart') != 0:
        parser.error('Probe requires a zero-start baseline with standard crop geometry')
    cv2.setNumThreads(1)
    detector = StreamingDetector(args.video, end=model['analyzedEnd'], background_samples=model['backgroundSamples'],
                                 candidate_floor=model['candidateFloor'], head_adapter=args.head_adapter)
    if detector.video != baseline['video'] or detector.fps != baseline['fps']:
        parser.error('Video or frame rate does not match baseline')
    if detector.weights_hash != model['weightsSha256'] or detector.head_adapter_hash != model['headAdapterSha256']:
        parser.error('Checkpoint and adapter must match the baseline')
    frames = probe(detector, baseline['trackingFrames'])
    summary = {'frames':len(frames), 'calls':detector.calls, 'inferenceSeconds':detector.inference_seconds}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {'format':'pongmetrics-ball-gap-probe', 'version':1,
               'baselineSha256':hashlib.sha256(baseline_bytes).hexdigest(),
               'video':detector.video, 'fps':detector.fps, 'summary':summary, 'frames':frames}
    args.output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
