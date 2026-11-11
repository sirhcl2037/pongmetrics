"""Streaming TrackNet candidates. Cache image evidence separately from tracking."""
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / 'balltrack_best.pth'


def load_tracknet(weights_hash, head_adapter, threads=4):
    import torch
    from detect import TrackNet
    from mmengine.logging.history_buffer import HistoryBuffer

    torch.set_num_threads(threads)
    with torch.serialization.safe_globals([HistoryBuffer, np.ndarray, np.dtype,
            (np._core.multiarray.scalar, 'numpy.core.multiarray.scalar'),
            (np._core.multiarray._reconstruct, 'numpy.core.multiarray._reconstruct'),
            type(np.dtype('float64')), type(np.dtype('int64')), getattr]):
        state = torch.load(WEIGHTS, map_location='cpu', weights_only=True)
    model = TrackNet(in_dim=15, out_dim=4, last_only=True).eval()
    model.load_state_dict({k.removeprefix('module.'): v for k, v in state.get('state_dict', state).items()})
    if head_adapter is not None:
        from head_adapter import apply_head_adapter
        apply_head_adapter(model, head_adapter, weights_hash)
    return model


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


def extract_weak_peaks(heatmap, offset=(0, 0), floor=.01):
    """Experimental isolated maxima, avoiding centroids of diffuse weak regions."""
    maxima = cv2.dilate(heatmap, np.ones((7, 7), dtype=np.uint8))
    mask = ((heatmap == maxima) & (heatmap >= floor) & (heatmap < .5)).astype(np.uint8)
    count, components, _, centers = cv2.connectedComponentsWithStats(mask)
    result = []
    for index in range(1, count):
        x, y = np.rint(centers[index]).astype(int)
        score = float(heatmap[components == index].max())
        result.append(dict(x=float((x + offset[0])/512), y=float((y + offset[1])/288),
                           score=score, area=1, threshold=floor, evidence='local-maximum'))
    return sorted(result, key=lambda c: c['score'], reverse=True)[:100]


def sample_background(capture, end, indices, native_zoom, scan_to_end):
    """Decode sequentially, retrieving BGR pixels only for sampled frames."""
    samples, zoom_samples = [], []
    last_timestamp = None
    index = 0
    try:
        while capture.grab():
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if timestamp >= end and not scan_to_end:
                break
            last_timestamp = timestamp
            if timestamp < end and (indices is None or index in indices):
                ok, frame = capture.retrieve()
                if not ok:
                    raise ValueError('Could not retrieve a background sample')
                samples.append(cv2.resize(frame, (512, 288)))
                if native_zoom:
                    zoom_samples.append(cv2.resize(frame, (1024, 576)))
            index += 1
    finally:
        capture.release()
    return samples, zoom_samples, last_timestamp


class StreamingDetector:
    """Four source frames and a bounded sampled background; no clip-sized array."""
    def __init__(self, video, end=float('inf'), background_samples=24, candidate_floor=.15, weak_peaks=False,
                 native_zoom=False, head_adapter=None, *, backend='torch', cpu_threads=None):
        if not 0 < candidate_floor < .5:
            raise ValueError('Candidate floor must be between zero and .5')
        if backend not in ('auto', 'torch', 'onnx'):
            raise ValueError('Unsupported inference backend')
        if cpu_threads is not None and (not isinstance(cpu_threads, int) or cpu_threads < 1):
            raise ValueError('CPU thread count must be a positive integer')
        self.candidate_floor = candidate_floor
        self.weak_peaks = weak_peaks
        self.native_zoom = native_zoom
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
        samples, zoom_samples, last_timestamp = sample_background(cap, self.end, indices, native_zoom, scan_to_end=True)
        # MKV frame-count/duration headers can underreport decodable footage.
        if last_timestamp is not None:
            self.video['duration'] = round(last_timestamp + 1/self.fps, 6)
            corrected_end = min(end, self.video['duration'])
            if abs(corrected_end - self.end) > .5/self.fps:
                self.end = corrected_end
                last = max(0, int(np.ceil(self.end*self.fps))-1)
                indices = set(np.linspace(0, last, min(background_samples, last+1), dtype=int)) if background_samples else None
                cap = cv2.VideoCapture(str(video))
                samples, zoom_samples, _ = sample_background(cap, self.end, indices, native_zoom, scan_to_end=False)
        if not samples:
            raise ValueError('No background frames')
        self.background_sample_count = len(samples)
        self.background = np.median(np.stack(samples), axis=0).astype(np.float32)/255
        del samples
        if native_zoom:
            self.zoom_background = np.median(np.stack(zoom_samples), axis=0).astype(np.float32)/255
            del zoom_samples
        self.weights_hash = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
        self.head_adapter_hash = hashlib.sha256(Path(head_adapter).read_bytes()).hexdigest() if head_adapter else None
        self.onnx_session = None
        self.backend = 'torch'
        self.cpu_threads = cpu_threads or 4
        if backend in ('auto', 'onnx'):
            from onnx_ball import bundle_path, load_session
            graph = bundle_path(ROOT, self.weights_hash, self.head_adapter_hash)
            onnx_threads = cpu_threads or min(8, os.cpu_count() or 1)
            if graph.is_file() or backend == 'onnx':
                try:
                    self.onnx_session = load_session(graph, self.weights_hash, self.head_adapter_hash,
                                                     onnx_threads, require_preference=backend == 'auto')
                except (ImportError, OSError, ValueError) as error:
                    if backend == 'onnx':
                        raise ValueError(f'Cannot load the ball ONNX export: {error}') from error
                    print(f'Using PyTorch for ball detection: {error}', file=sys.stderr)
                if self.onnx_session is not None:
                    self.backend = 'onnx'
                    self.cpu_threads = onnx_threads
        if self.onnx_session is None:
            self.model = load_tracknet(self.weights_hash, head_adapter, self.cpu_threads)
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
                source = frame if getattr(self, 'native_zoom', False) else resized
                if not history:
                    history.extend([source]*3)
                history.append(source)
                yield i, timestamp, list(history), cut
                i += 1
        finally:
            cap.release()

    def detect(self, history, center=None):
        if self.native_zoom:
            if center is not None:
                return self.detect_native_crop(history, center)
            history = [cv2.resize(frame, (512, 288)) for frame in history]
        left, top, width, height = 0, 0, 512, 288
        if center is not None:
            width, height = 256, 144
            left, top = np.clip(np.rint(np.asarray(center)*[512, 288]-[128, 72]), [0, 0], [256, 144]).astype(int)
            self.crop_calls += 1
        else:
            self.full_frame_calls += 1
        region = np.s_[top:top+height, left:left+width]
        sequence = [self.background[region], *[f[region].astype(np.float32)/255 for f in history]]
        frames = np.stack(sequence).transpose(0, 3, 1, 2).reshape(1, 15, height, width)
        heatmap = self.infer_heatmap(frames)
        candidates = extract_candidates(heatmap, (left, top), self.candidate_floor)
        if self.weak_peaks:
            candidates.extend(extract_weak_peaks(heatmap, (left, top)))
        if center is not None:
            candidates = [c for c in candidates if left+4 <= c['x']*512 < left+width-4
                          and top+4 <= c['y']*288 < top+height-4]
        return candidates

    def detect_native_crop(self, history, center):
        """Crop half the source width/height, then resize to the full model input."""
        source_height, source_width = history[-1].shape[:2]
        width, height = source_width//2, source_height//2
        left = int(np.clip(round(center[0]*source_width-width/2), 0, source_width-width))
        top = int(np.clip(round(center[1]*source_height-height/2), 0, source_height-height))
        crops = [cv2.resize(frame[top:top+height, left:left+width], (512, 288)).astype(np.float32)/255 for frame in history]
        # The sampled median is retained at 2x model resolution, avoiding 24 full 4K backgrounds.
        x0, y0 = round(left/source_width*1024), round(top/source_height*576)
        background = self.zoom_background[y0:y0+288, x0:x0+512]
        frames = np.stack([background, *crops]).transpose(0, 3, 1, 2).reshape(1, 15, 288, 512)
        heatmap = self.infer_heatmap(frames)
        self.crop_calls += 1
        candidates = extract_candidates(heatmap, floor=self.candidate_floor)
        result = []
        for candidate in candidates:
            if not 4/512 <= candidate['x'] < 508/512 or not 4/288 <= candidate['y'] < 284/288:
                continue
            result.append({**candidate, 'x':(left+candidate['x']*width)/source_width,
                           'y':(top+candidate['y']*height)/source_height})
        return result

    def infer_heatmap(self, frames):
        began = time.perf_counter()
        if self.onnx_session is not None:
            heatmap = self.onnx_session.run(['heatmap'], {'frames':frames})[0][0, 0]
        else:
            import torch
            with torch.inference_mode():
                heatmap = self.model(frames=torch.from_numpy(frames))[0][0, 0].numpy()
        self.inference_seconds += time.perf_counter() - began
        self.calls += 1
        return heatmap


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
