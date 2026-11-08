"""Overlay actual observations, candidate evidence, and separate predictions."""
import argparse
from collections import deque
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('video', type=Path)
    p.add_argument('detections', type=Path)
    p.add_argument('--frames', help='Comma-separated source frame indices; defaults to recovered weak evidence')
    p.add_argument('--output', type=Path, required=True, help='Contact sheet JPG/PNG')
    p.add_argument('--review-video', type=Path)
    p.add_argument('--max-tiles', type=int, default=24, help='Bound contact-sheet memory on long videos')
    a = p.parse_args()
    if a.max_tiles < 1:
        p.error('--max-tiles must be positive')
    data = json.loads(a.detections.read_text(encoding='utf8'))
    if a.video.name != data['video']['name'] or a.video.stat().st_size != data['video']['size']:
        p.error('Review video does not match detections')
    rows = {r['frame']: r for r in data.get('trackingFrames', data.get('frames', []))}
    chosen = set(map(int, a.frames.split(','))) if a.frames else {i for i,r in rows.items() if 'weak-evidence' in r.get('reason', '')}
    if len(chosen) > a.max_tiles:
        ordered = sorted(chosen)
        chosen = {ordered[i] for i in np.linspace(0, len(ordered)-1, a.max_tiles, dtype=int)}
    cap = cv2.VideoCapture(str(a.video))
    if a.review_video:
        a.review_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(a.review_video), cv2.VideoWriter_fourcc(*'mp4v'), data['fps'], (960, 540)) if a.review_video else None
    if writer is not None and not writer.isOpened():
        cap.release()
        raise ValueError('Cannot create review video')
    tiles = []
    trail = deque(maxlen=6)
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i not in chosen and writer is None:
                trail.clear()
                i += 1
                continue
            image = cv2.resize(frame, (960, 540))
            row = rows.get(i, {})
            def point(p):
                return round(p['x']*960), round(p['y']*540)
            for c in row.get('candidates', []):
                cv2.circle(image, point(c), 7, (150, 150, 150), 1)
                cv2.putText(image, f"{c['score']:.2f}", point(c), cv2.FONT_HERSHEY_SIMPLEX, .35, (200, 200, 200), 1)
            prediction = row.get('prediction')
            if prediction:
                cv2.drawMarker(image, point(prediction), (255, 170, 0), cv2.MARKER_CROSS, 12, 1)
                radius = row.get('uncertaintyRadiusPixels')
                if radius:
                    cv2.circle(image, point(prediction), round(radius*960/512), (140, 80, 0), 1)
            if row.get('position'):
                trail.append(point(row['position']))
                if len(trail) > 1:
                    cv2.polylines(image, [np.asarray(trail, dtype=np.int32)], False, (0, 210, 210), 1)
                cv2.circle(image, point(row['position']), 9, (0, 255, 255), 2)
            else:
                trail.clear()
            cv2.putText(image, f"f{i} {row.get('timestamp', i/data['fps']):.2f}s {row.get('reason', '')}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 1)
            cv2.putText(image, 'yellow: observation | blue cross/ring: prediction/search | gray: candidates', (10, 520), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            if i in chosen:
                tiles.append(image)
            if writer:
                writer.write(image)
            i += 1
    finally:
        cap.release()
        if writer:
            writer.release()
    if tiles:
        if len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        a.output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(a.output), np.vstack([np.hstack(tiles[i:i+2]) for i in range(0, len(tiles), 2)]))


if __name__ == '__main__':
    main()
