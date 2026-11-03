"""Offline person detection + ByteTrack. Boxes and IDs, not pose or named identity."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT/'output'/'yolo-settings'))
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')


class CachedPlayerTracks:
    """Replay saved observations for ball experiments, without loading YOLO."""
    def __init__(self, path, video=None):
        data = json.loads(Path(path).read_text(encoding='utf8'))
        if data.get('format') != 'pongmetrics-player-tracks' or data.get('version') != 1:
            raise ValueError('Unsupported player track file')
        self.fps = data['fps']
        if not isinstance(self.fps,(int,float)) or not np.isfinite(self.fps) or not 0 < self.fps <= 240:
            raise ValueError('Invalid player track FPS')
        if video is not None and (data['video']['name'] != Path(video).name or data['video']['size'] != Path(video).stat().st_size):
            raise ValueError('Player tracks belong to a different video')
        self.rows = {}
        for row in data['frames']:
            key = round(row['timestamp']*self.fps)
            if key in self.rows:
                raise ValueError('Duplicate player frame')
            self.rows[key] = row

    def at(self, timestamp):
        # Missing frames remain missing; never carry a stale box through a gap.
        return self.rows.get(round(timestamp*self.fps))


class PersonTracker:
    """One instance per continuous shot; IDs reset when the camera cuts."""
    def __init__(self, weights, imgsz=416, confidence=.1, roi=(.15,.05,.85,.95), tracker=str(ROOT/'players-bytetrack.yaml')):
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(4)
        self.model = YOLO(str(weights))
        self.imgsz, self.confidence, self.roi, self.tracker = imgsz, confidence, roi, tracker
        self.identities = {}

    def update(self, frame):
        result = self.model.track(frame, persist=True, tracker=self.tracker, classes=[0],
                                  imgsz=self.imgsz, conf=self.confidence, iou=.45, device='cpu', verbose=False)[0]
        people = []
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            height, width = frame.shape[:2]
            for box, confidence, track_id in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.id.cpu().numpy()):
                x0,y0,x1,y1 = np.clip(box/[width,height,width,height],0,1)
                cx, cy = float((x0+x1)/2), float((y0+y1)/2)
                rx0,ry0,rx1,ry1 = self.roi
                if not (rx0 <= cx <= rx1 and ry0 <= cy <= ry1):
                    continue
                people.append({'trackId':int(track_id), 'confidence':float(confidence),
                    'box':{'x':float(x0),'y':float(y0),'width':float(x1-x0),'height':float(y1-y0)},
                    'center':{'x':cx,'y':cy},
                    'groundProxy':{'x':cx,'y':float(y1)}})
        # Bind only an unambiguous first pair. These labels describe their initial
        # screen positions; a new ByteTrack ID is never silently assigned a name.
        if not self.identities and len(people) == 2:
            pair = sorted(people,key=lambda p:p['center']['y'])
            self.identities = {pair[0]['trackId']:'far-player',pair[1]['trackId']:'near-player'}
        for person in people:
            person['playerId'] = self.identities.get(person['trackId'])
        return {'people':people, 'missingPlayers':sorted(set(self.identities.values())-{p['playerId'] for p in people}),
                'needsReview':len(people) != 2 or any(p['playerId'] is None for p in people)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('video', type=Path)
    p.add_argument('--weights', type=Path, default=ROOT/'output'/'models'/'yolo11n.pt')
    p.add_argument('--imgsz', type=int, default=416)
    p.add_argument('--stride', type=int, default=1, help='Detect on every Nth source frame, starting with the first frame in the segment')
    p.add_argument('--confidence', type=float, default=.1, help='Low threshold retains ByteTrack rescue detections')
    p.add_argument('--tracker', default=str(ROOT/'players-bytetrack.yaml'), help='Built-in name or custom tracker YAML')
    p.add_argument('--roi', nargs=4, type=float, default=(.15,.05,.85,.95), metavar=('LEFT','TOP','RIGHT','BOTTOM'), help='Normalized box-center eligibility region; tune to exclude officials/spectators')
    p.add_argument('--start', type=float, default=2)
    p.add_argument('--end', type=float, default=8.28)
    p.add_argument('--output', type=Path, default=ROOT/'output'/'players.json')
    p.add_argument('--review-video', type=Path, help='Optional annotated MP4; never repeats inference')
    p.add_argument('--no-review-sheet', action='store_true', help='Skip the sampled contact sheet')
    a = p.parse_args()
    if a.stride < 1:
        p.error('--stride must be at least 1')
    if not 0 <= a.start < a.end or a.imgsz < 64 or a.imgsz%32 or not 0 < a.confidence <= 1:
        p.error('Invalid time interval, image size (multiple of 32), or confidence')
    if not (0 <= a.roi[0] < a.roi[2] <= 1 and 0 <= a.roi[1] < a.roi[3] <= 1):
        p.error('Invalid ROI')
    if not a.weights.is_file():
        p.error('Weights missing; download yolo11n.pt to --weights first (see README).')
    began = time.perf_counter()
    tracker = PersonTracker(a.weights,a.imgsz,a.confidence,a.roi,a.tracker)
    startup = time.perf_counter()-began
    cap = cv2.VideoCapture(str(a.video))
    if not cap.isOpened():
        raise RuntimeError('Cannot open video')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not 0 < fps <= 240:
        raise RuntimeError('Unsupported FPS')
    video = {'name':a.video.name,'size':a.video.stat().st_size,'width':int(cap.get(3)),
             'height':int(cap.get(4)),'duration':cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps}
    rows, tiles, writer = [], [], None
    segment_frames = 0
    processing = 0.
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            if timestamp < a.start: continue
            if timestamp >= a.end: break
            sample = segment_frames % a.stride == 0
            segment_frames += 1
            if not sample:
                continue
            tick = time.perf_counter()
            row = tracker.update(frame)
            processing += time.perf_counter()-tick
            row['timestamp'] = timestamp
            rows.append(row)
            make_tile = not a.no_review_sheet and (len(rows)-1)%25 == 0
            if a.review_video or make_tile:
                preview = cv2.resize(frame,(960,540))
                for person in row['people']:
                    box = person['box']; color = (0,220,0) if person['playerId']=='near-player' else (255,200,0)
                    x,y,w,h = round(box['x']*960),round(box['y']*540),round(box['width']*960),round(box['height']*540)
                    cv2.rectangle(preview,(x,y),(x+w,y+h),color,2)
                    cv2.putText(preview,f"{person['playerId'] or 'unassigned'} #{person['trackId']}",(x,max(20,y-7)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
                cv2.putText(preview,f'{timestamp:.2f}s',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2)
                if make_tile: tiles.append(preview)
                if a.review_video:
                    if writer is None:
                        a.review_video.parent.mkdir(parents=True,exist_ok=True)
                        writer = cv2.VideoWriter(str(a.review_video),cv2.VideoWriter_fourcc(*'mp4v'),fps/a.stride,(960,540))
                        if not writer.isOpened(): raise RuntimeError('Cannot create review video')
                    writer.write(preview)
    finally:
        cap.release()
        if writer is not None: writer.release()
    counts = {name:sum(any(p['playerId']==name for p in r['people']) for r in rows) for name in ['near-player','far-player']}
    summary = {'frames':len(rows),'framesWithBothPlayers':sum(not r['needsReview'] for r in rows),
               'sourceFrames':segment_frames,'stride':a.stride,
               'observedFramesByPlayer':counts,'trackIds':sorted({p['trackId'] for r in rows for p in r['people']}),
               'startupSeconds':startup,'detectionAndTrackingSeconds':processing,'totalSeconds':time.perf_counter()-began}
    import ultralytics
    payload = {'format':'pongmetrics-player-tracks','version':1,'coordinateSystem':'normalized-video-image',
        'video':video,'fps':fps,'model':{'name':a.weights.name,'sha256':hashlib.sha256(a.weights.read_bytes()).hexdigest(),
        'ultralyticsVersion':ultralytics.__version__,'tracker':a.tracker,'imgsz':a.imgsz,'confidence':a.confidence,'roi':a.roi,'stride':a.stride},
        'identityMeaning':'Initial near/far positions within this shot; not verified names or cross-cut identity',
        'summary':summary,'frames':rows}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(payload,indent=2),encoding='utf8')
    if tiles:
        if len(tiles)%2: tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(a.output.with_suffix('.jpg')),np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,len(tiles),2)]))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
