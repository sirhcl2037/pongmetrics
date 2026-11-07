"""Streaming TrackNet candidates. Cache image evidence separately from tracking."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


def temporal_indices(index):
    return [max(0, j) for j in range(index - 3, index + 1)]


def extract_candidates(heatmap, offset=(0, 0), floor=.15):
    """Keep strong components unchanged, plus weaker independent components.

    Peaks are neural evidence, not probabilities. No roundness/color gate: the
    ball can be a streak or overlap skin. A weak component containing a strong
    one is not a second observation.
    """
    result = []
    for threshold in (.5, floor):
        count, labels, stats, centers = cv2.connectedComponentsWithStats(
            (heatmap >= threshold).astype(np.uint8))
        for k in range(1, count):
            x, y, w, h, area = stats[k]
            pixels = labels[y:y+h, x:x+w] == k
            values = heatmap[y:y+h, x:x+w][pixels]
            peak = float(values.max())
            if threshold < .5 and peak >= .5:
                continue
            point = centers[k] + offset
            result.append({'x': float(point[0]/512), 'y': float(point[1]/288),
                           'score': peak, 'area': int(area), 'threshold': threshold})
    return sorted(result, key=lambda c: c['score'], reverse=True)


class StreamingDetector:
    """Four source frames and a bounded sampled background; no clip-sized array."""
    def __init__(self, video, end=float('inf'), background_samples=24):
        began = time.perf_counter()
        self.video_path = Path(video)
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise ValueError('Cannot open video')
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(self.fps) or not 0 < self.fps <= 240:
            cap.release()
            raise ValueError('Unsupported FPS')
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video = dict(name=Path(video).name, size=Path(video).stat().st_size,
                          width=int(cap.get(3)), height=int(cap.get(4)), duration=count/self.fps)
        self.end = min(end, self.video['duration'])
        # Decode sequentially for exact parity with detect.py. Retain at most N
        # samples; 0 explicitly requests its historical, unbounded full median.
        last = max(0, int(np.ceil(self.end*self.fps))-1)
        indices = set(np.linspace(0, last, min(background_samples, last+1), dtype=int)) if background_samples else None
        samples = []
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok or cap.get(cv2.CAP_PROP_POS_MSEC)/1000 >= self.end:
                break
            if indices is None or i in indices:
                samples.append(cv2.resize(frame, (512, 288)))
            i += 1
        cap.release()
        if not samples:
            raise ValueError('No background frames')
        self.background_sample_count = len(samples)
        self.background = np.median(np.stack(samples), axis=0).astype(np.float32)/255
        del samples
        import torch
        from detect import TrackNet
        from mmengine.logging.history_buffer import HistoryBuffer
        torch.set_num_threads(4)
        with torch.serialization.safe_globals([HistoryBuffer, np.ndarray, np.dtype,
                (np._core.multiarray.scalar, 'numpy.core.multiarray.scalar'),
                (np._core.multiarray._reconstruct, 'numpy.core.multiarray._reconstruct'),
                type(np.dtype('float64')), type(np.dtype('int64')), getattr]):
            state = torch.load(ROOT/'balltrack_best.pth', map_location='cpu', weights_only=True)
        self.model = TrackNet(in_dim=15, out_dim=4, last_only=True).eval()
        self.model.load_state_dict({k.removeprefix('module.'): v for k, v in state.get('state_dict', state).items()})
        self.weights_hash = hashlib.sha256((ROOT/'balltrack_best.pth').read_bytes()).hexdigest()
        self.startup_seconds = time.perf_counter()-began
        self.calls = self.crop_calls = self.full_frame_calls = 0
        self.inference_seconds = 0.

    def frames(self):
        cap = cv2.VideoCapture(str(self.video_path))
        previous_time = -1.
        history = deque(maxlen=4)
        previous_small = None
        i = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                if timestamp >= self.end:
                    break
                if timestamp <= previous_time:
                    raise ValueError('Non-monotonic decoder timestamps; refusing misaligned tracking')
                previous_time = timestamp
                resized = cv2.resize(frame, (512, 288))
                small = cv2.resize(resized, (64, 36)).astype(np.float32)/255
                cut = previous_small is not None and float(np.mean(np.abs(small-previous_small))) > .22
                if cut:
                    history.clear()
                previous_small = small
                if not history:
                    history.extend([resized]*3)
                history.append(resized)
                yield i, timestamp, list(history), cut
                i += 1
        finally:
            cap.release()

    def detect(self, history, center=None):
        import torch
        left, top, width, height = 0, 0, 512, 288
        if center is not None:
            width, height = 256, 144
            left, top = np.clip(np.rint(np.asarray(center)*[512, 288]-[128, 72]), [0, 0], [256, 144]).astype(int)
            self.crop_calls += 1
        else:
            self.full_frame_calls += 1
        region = np.s_[top:top+height, left:left+width]
        sequence = [self.background[region], *[f[region].astype(np.float32)/255 for f in history]]
        tensor = torch.from_numpy(np.stack(sequence).transpose(0, 3, 1, 2).reshape(1, 15, height, width))
        began = time.perf_counter()
        with torch.inference_mode():
            heatmap = self.model(frames=tensor)[0][0, 0].numpy()
        self.inference_seconds += time.perf_counter()-began
        self.calls += 1
        candidates = extract_candidates(heatmap, (left, top))
        if center is not None:
            candidates = [c for c in candidates if left+4 <= c['x']*512 < left+width-4
                          and top+4 <= c['y']*288 < top+height-4]
        return candidates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('video', type=Path)
    p.add_argument('--end', type=float, default=8.28)
    p.add_argument('--background-samples', type=int, default=24)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.end <= 0 or a.background_samples < 0:
        p.error('Invalid end or sample count')
    began = time.perf_counter()
    detector = StreamingDetector(a.video, a.end, a.background_samples)
    rows = []
    for i, timestamp, history, cut in detector.frames():
        rows.append(dict(frame=i, timestamp=timestamp, candidates=detector.detect(history), sceneCut=cut))
        if i % 25 == 0:
            print(f'{i} frames cached', flush=True)
    payload = dict(format='pongmetrics-neural-candidates', version=1, video=detector.video,
                   fps=detector.fps, model=dict(name='RacketVision TrackNetV3', weightsSha256=detector.weights_hash,
                   backgroundSamples=detector.background_sample_count, candidateFloor=.15,
                   temporalOrder='background,BGR[i-3],BGR[i-2],BGR[i-1],BGR[i]; repeat first'),
                   summary=dict(frames=len(rows), heavyCalls=detector.calls,
                   startupSeconds=detector.startup_seconds, inferenceSeconds=detector.inference_seconds,
                   totalSeconds=time.perf_counter()-began), frames=rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2), encoding='utf8')
    print(json.dumps(payload['summary']))


if __name__ == '__main__':
    main()
