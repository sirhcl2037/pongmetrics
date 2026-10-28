"""Offline RacketVision experiment. No interpolation, inpainting, or web service."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor/RacketVision/source/BallTrack'))
from model.tracknet_v3 import TrackNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video', type=Path)
    parser.add_argument('--weights', type=Path, default=ROOT / 'balltrack_best.pth')
    parser.add_argument('--output', type=Path, default=ROOT / 'output')
    parser.add_argument('--threshold', type=float, default=.5)
    parser.add_argument('--end', type=float, default=8.28, help='End of fixed-camera segment, seconds')
    args = parser.parse_args()
    if not 0 < args.threshold <= 1 or args.end <= 0:
        parser.error('threshold must be (0,1] and end must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError('Cannot open video')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(3)), int(cap.get(4))
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps
    frames, timestamps = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
        if timestamp >= args.end:
            break
        frames.append(cv2.resize(frame, (512,288)))
        timestamps.append(timestamp)
    cap.release()
    if not frames or not all(b > a for a,b in zip(timestamps,timestamps[1:])):
        raise RuntimeError('No frames or non-monotonic decoder timestamps')
    # Preserve OpenCV BGR channel order, as in the training dataset.
    background = np.median(np.stack(frames), axis=0).astype(np.float32)/255
    frames = np.asarray(frames, dtype=np.float32)/255
    # Official checkpoint includes MMEngine training history; allow only its
    # known numeric containers while retaining PyTorch's restricted loader.
    from mmengine.logging.history_buffer import HistoryBuffer
    with torch.serialization.safe_globals([HistoryBuffer, np.ndarray, np.dtype, (np._core.multiarray.scalar, 'numpy.core.multiarray.scalar'), (np._core.multiarray._reconstruct, 'numpy.core.multiarray._reconstruct'), type(np.dtype('float64')), type(np.dtype('int64')), getattr]):
        state = torch.load(args.weights, map_location='cpu', weights_only=True)
    state = state.get('state_dict',state)
    state = {k.removeprefix('module.'):v for k,v in state.items()}
    model = TrackNet(in_dim=15,out_dim=4,last_only=True).eval()
    model.load_state_dict(state)
    observations, audit = [], []
    started = time.perf_counter()
    for i, timestamp in enumerate(timestamps):
        # Training dataset uses [i-3,i-2,i-1,i], replicating frame zero at start.
        sequence = [background, *[frames[max(0,j)] for j in range(i-3,i+1)]]
        tensor = torch.from_numpy(np.stack(sequence).transpose(0,3,1,2).reshape(1,15,288,512))
        with torch.inference_mode():
            hm = model(frames=tensor)[0][0,0].numpy()
        count, labels, stats, centers = cv2.connectedComponentsWithStats((hm >= args.threshold).astype(np.uint8))
        candidates = []
        for k in range(1,count):
            score = float(hm[labels == k].max())
            candidates.append((score,k))
        candidates.sort(reverse=True)
        accepted = len(candidates) == 1
        point = None
        confidence = float(hm.max())
        if accepted:
            _, k = candidates[0]
            x,y = centers[k]
            point = {'x':float(x/512), 'y':float(y/288)}
            observations.append({'timestamp':timestamp,'position':{'value':point,'source':'automatic','confidence':confidence}})
        audit.append({'frame':i,'timestamp':timestamp,'accepted':accepted,'confidence':confidence,'position':point,'reason':'detected' if accepted else 'ambiguous' if candidates else 'below-threshold'})
        if i%25 == 0:
            print(f'{i}/{len(frames)} frames; {len(observations)} detections', flush=True)
    video = dict(name=args.video.name,size=args.video.stat().st_size,width=width,height=height,duration=duration)
    info = dict(name='RacketVision TrackNetV3',source='https://github.com/OrcustD/RacketVision',revision='c44af2a08524d3cb54d818f19686f4cdea4d2793',weightsSha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),threshold=args.threshold,confidenceMeaning='uncalibrated heatmap peak',interpolation=False)
    payload = dict(format='pongmetrics-ball-detections',version=1,video=video,fps=fps,coordinateSystem='normalized-video-image',model=info,ballObservations=observations)
    (args.output/'detections.json').write_text(json.dumps(payload,indent=2),encoding='utf8')
    (args.output/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    # Review contact sheets show predicted dots on actual decoded frames.
    cap = cv2.VideoCapture(str(args.video))
    tiles = []
    for i, item in enumerate(audit):
        ok, frame = cap.read()
        if not ok: break
        if i % 5: continue
        frame = cv2.resize(frame,(640,360))
        if item['position']:
            p = item['position']; cv2.circle(frame,(round(p['x']*640),round(p['y']*360)),8,(0,255,255),1)
        cv2.putText(frame,f"f{i} {item['timestamp']:.2f}s {item['reason']} {item['confidence']:.2f}",(10,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
        tiles.append(frame)
    cap.release()
    for page in range(0,len(tiles),12):
        group = tiles[page:page+12]
        group += [np.zeros_like(tiles[0])]*(12-len(group))
        sheet = np.vstack([np.hstack(group[r:r+3]) for r in range(0,12,3)])
        cv2.imwrite(str(args.output/f'review-{page//12}.jpg'),sheet)
    print(json.dumps(dict(frames=len(frames),accepted=len(observations),gaps=len(frames)-len(observations),seconds=time.perf_counter()-started)),flush=True)


if __name__ == '__main__':
    main()
