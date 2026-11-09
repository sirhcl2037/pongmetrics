"""Render representative evaluation failures against reviewed ball centers."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np

from evaluate_manual_labels import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('detections', type=Path)
    parser.add_argument('labels', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    detections = json.loads(args.detections.read_text(encoding='utf-8'))
    labels = json.loads(args.labels.read_text(encoding='utf-8'))
    metrics = evaluate(detections, labels, 0)
    groups = defaultdict(list)
    for failure in metrics['referenceFailures']:
        groups[failure['reason']].append(failure['frame'])
    selected = {}
    for reason, frames in groups.items():
        for index in np.linspace(0, len(frames)-1, min(4, len(frames)), dtype=int):
            selected[frames[index]] = reason
    if not selected:
        print('No failures at the 20-pixel reference tolerance.')
        return
    fps = detections['fps']
    targets = {round(row['timestamp']*fps):row for row in labels['labels']}
    proposals = {round(row['timestamp']*fps):row['position']['value'] for row in detections['ballObservations']}
    capture = cv2.VideoCapture(str(args.video))
    panels = []
    frame = 0
    try:
        while frame <= max(selected):
            ok, image = capture.read()
            if not ok:
                break
            if frame in selected:
                panel = cv2.resize(image, (640, 360))
                label = targets[frame]
                proposal = proposals.get(frame)
                target = label['position']['value'] if label.get('position') else None
                if target:
                    cv2.drawMarker(panel, (round(target['x']*640), round(target['y']*360)), (0,255,0), cv2.MARKER_CROSS, 16, 2)
                if proposal:
                    cv2.circle(panel, (round(proposal['x']*640), round(proposal['y']*360)), 7, (0,0,0), 3)
                    cv2.circle(panel, (round(proposal['x']*640), round(proposal['y']*360)), 5, (0,255,255), 2)
                cv2.rectangle(panel, (0,0), (640,48), (0,0,0), -1)
                cv2.putText(panel, f"{frame} | {frame/fps:.2f}s | {selected[frame]} | {label['visibility']}",
                            (7,19), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
                cv2.putText(panel, 'Green: reviewed center. Yellow: detector proposal.', (7,39), cv2.FONT_HERSHEY_SIMPLEX, .4, (220,220,220), 1)
                panels.append(panel)
            frame += 1
    finally:
        capture.release()
    if panels:
        if len(panels)%2:
            panels.append(np.zeros_like(panels[0]))
        sheet = np.vstack([np.hstack(panels[i:i+2]) for i in range(0,len(panels),2)])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.output), sheet)
    print(json.dumps({'selectedFrames':selected,'referenceFailureCounts':{key:len(value) for key,value in groups.items()}}))


if __name__ == '__main__':
    main()
