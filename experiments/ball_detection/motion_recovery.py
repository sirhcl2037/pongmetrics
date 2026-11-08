"""Experimental image-evidence recovery in short gaps of a neural track."""
import argparse
import copy
import json
import time
from pathlib import Path

import cv2
import numpy as np


def image_candidates(previous, current, following):
    gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    saturation = cv2.cvtColor(current, cv2.COLOR_BGR2HSV)[:, :, 1]
    motion = np.minimum(cv2.absdiff(gray, cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)),
                        cv2.absdiff(gray, cv2.cvtColor(following, cv2.COLOR_BGR2GRAY)))
    contrast = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    mask = ((gray >= 110) & (saturation < 90) & (motion > 10) & (contrast > 15)).astype(np.uint8)
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    return [dict(x=float(centers[i][0]/640), y=float(centers[i][1]/360))
            for i in range(1, count) if 2 <= stats[i, cv2.CC_STAT_AREA] <= 150
            and stats[i, cv2.CC_STAT_WIDTH] <= 40 and stats[i, cv2.CC_STAT_HEIGHT] <= 40]


def recover(video, detections):
    began = time.perf_counter()
    output = copy.deepcopy(detections)
    fps = detections['fps']
    observations = {round(row['timestamp']*fps): row for row in output['ballObservations']}
    candidates = {}
    capture = cv2.VideoCapture(str(video))
    def read():
        ok, image = capture.read()
        return cv2.resize(image, (640, 360)) if ok else None
    previous = current = read()
    following = read()
    index = 0
    while current is not None and following is not None:
        candidates[index] = image_candidates(previous, current, following)
        previous, current, following = current, following, read()
        index += 1
    capture.release()
    anchors = sorted(frame for frame, row in observations.items() if row['position'].get('confidence', 0) >= .5)
    additions = []
    for left, right in zip(anchors, anchors[1:]):
        if not 1 < right-left <= 6:
            continue
        if any(row.get('sceneCut') for row in detections.get('trackingFrames', [])
               if left < round(row['timestamp']*fps) <= right):
            continue
        a = observations[left]['position']['value']
        b = observations[right]['position']['value']
        pa, pb = np.array([a['x'], a['y']])*[640, 360], np.array([b['x'], b['y']])*[640, 360]
        for frame in range(left+1, right):
            if frame in observations:
                continue
            expected = pa+(pb-pa)*(frame-left)/(right-left)
            ranked = sorted([(float(np.linalg.norm(np.array([point['x'], point['y']])*[640, 360]-expected)), point)
                             for point in candidates.get(frame, [])], key=lambda item: item[0])
            if not ranked or ranked[0][0] > 12 or (len(ranked) > 1 and ranked[1][0]-ranked[0][0] < 4):
                continue
            additions.append(dict(timestamp=frame/fps, position=dict(value=ranked[0][1], source='automatic', confidence=.15)))
    output['ballObservations'] = sorted([*output['ballObservations'], *additions], key=lambda row: row['timestamp'])
    output['model']['motionRecovery'] = 'experimental bright moving components with two-sided anchors'
    # These experimental additions have separate provenance; existing diagnostics remain neural-only.
    output['motionRecoveries'] = additions
    output['summary']['motionRecoverySeconds'] = time.perf_counter()-began
    output['summary']['motionRecoveries'] = len(additions)
    output['summary']['observations'] = len(output['ballObservations'])
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('detections', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    detections = json.loads(args.detections.read_text(encoding='utf-8'))
    if args.video.stat().st_size != detections['video']['size']:
        parser.error('Video size mismatch')
    output = recover(args.video, detections)
    args.output.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output['summary'], indent=2))


if __name__ == '__main__':
    main()
