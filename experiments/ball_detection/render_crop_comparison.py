"""Render synchronized baseline/native-zoom observations on identical source frames."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('zoom', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--labels', type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding='utf-8'))
    zoom = json.loads(args.zoom.read_text(encoding='utf-8'))
    if baseline['video'] != zoom['video'] or args.video.stat().st_size != baseline['video']['size']:
        parser.error('Comparison requires the same source video')
    fps = baseline['fps']
    records = [{row['frame']: row for row in data['trackingFrames']} for data in [baseline, zoom]]
    frames = sorted(set(records[0]) & set(records[1]))
    if not frames:
        parser.error('No shared frames')
    labels = {}
    if args.labels:
        data = json.loads(args.labels.read_text(encoding='utf-8'))
        for label in data['labels']:
            if label['visibility'] == 'visible' and abs(label['timestamp']*fps-round(label['timestamp']*fps)) < .001:
                labels[round(label['timestamp']*fps)-1] = label['position']['value']
    capture = cv2.VideoCapture(str(args.video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 360))
    if not writer.isOpened():
        capture.release()
        raise ValueError('Cannot create review video')
    differences = []
    for index in frames:
        a, b = records[0][index]['position'], records[1][index]['position']
        if bool(a) != bool(b) or (a and b and np.hypot(a['x']-b['x'], a['y']-b['y']) > .01):
            differences.append(index)
    # Contact sheets emphasize disagreements; the video retains every shared frame.
    selected = {frames[0]}
    pool = differences or frames
    selected.update(pool[round(i*(len(pool)-1)/2)] for i in range(3))
    tiles = []
    try:
        for index in range(frames[0], frames[-1]+1):
            ok, image = capture.read()
            if not ok:
                break
            if index not in records[0] or index not in records[1]:
                continue
            small = cv2.resize(image, (640, 360))
            panels = []
            for title, rows in zip(['Current tracker', 'Original-image crop'], records):
                panel = small.copy()
                row = rows[index]
                point = row['position']
                if point:
                    location = (round(point['x']*640), round(point['y']*360))
                    cv2.circle(panel, location, 6, (0, 0, 0), 3)
                    cv2.circle(panel, location, 4, (0, 255, 255), 2)
                if index in labels:
                    target = labels[index]
                    cv2.drawMarker(panel, (round(target['x']*640), round(target['y']*360)), (0, 255, 0), cv2.MARKER_CROSS, 12, 1)
                cv2.rectangle(panel, (0, 0), (640, 45), (0, 0, 0), -1)
                cv2.putText(panel, f'{title} | {index/fps:.2f}s | '+('observed' if point else 'gap'),
                            (8, 19), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
                cv2.putText(panel, 'Yellow: detector. Green: human label.' if labels else 'Yellow: detector. No human labels for accuracy.',
                            (8, 38), cv2.FONT_HERSHEY_SIMPLEX, .4, (220, 220, 220), 1)
                panels.append(panel)
            comparison = cv2.hconcat(panels)
            writer.write(comparison)
            if index in selected:
                tiles.append(comparison)
    finally:
        capture.release()
        writer.release()
    if tiles:
        cv2.imwrite(str(args.output.with_suffix('.jpg')), np.vstack(tiles))


if __name__ == '__main__':
    main()
