"""Audit saved final detections without inference or changes to reviewed labels."""
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from evaluate_manual_labels import evaluate

ROOT = Path(__file__).resolve().parent


def distance(a, b):
    return math.hypot((a['x']-b['x'])*1280, (a['y']-b['y'])*720)


def diagnose(track, labels, offset):
    metrics = evaluate(track, labels, offset)
    targets = {round(r['timestamp']*track['fps'])+offset: r for r in labels['labels']
               if abs(r['timestamp']*track['fps']-round(r['timestamp']*track['fps'])) <= .001}
    rows = {r['frame']: r for r in track['trackingFrames']}
    result = []
    for failure in metrics['referenceFailures']:
        frame = failure['frame']
        row, label = rows[frame], targets[frame]
        target = label['position']['value']
        ranked = sorted(row['candidates'], key=lambda c: distance(c, target))
        near = [c for c in ranked if distance(c, target) <= 20]
        category = ('center-error' if failure['reason'] != 'missing' else
                    'no-retained-candidates' if not ranked else
                    'candidates-only-away-from-label' if not near else
                    'nearby-evidence-below-follow-cutoff' if max(c['score'] for c in near) < track['model']['strongScore'] else
                    'nearby-evidence-rejected-by-tracking')
        entry = dict(**failure, timestamp=row['timestamp'], labelTimestamp=label['timestamp'],
                     target=target, proposal=row['position'], category=category, search=row['search'],
                     prediction=row['prediction'], gateRadiusModelPixels=row['uncertaintyRadiusPixels'],
                     sceneCut=row['sceneCut'], candidateCount=len(ranked), nearbyCandidates=near,
                     nearestCandidate=ranked[0] if ranked else None,
                     nearestDistanceReferencePixels=distance(ranked[0], target) if ranked else None)
        if row['search'] == 'crop' and row['prediction']:
            center = np.clip([row['prediction']['x'], row['prediction']['y']], 0, 1)
            left, top = np.clip(np.rint(center*[512,288]-[128,72]), [0,0], [256,144])
            entry['targetInsideRetainedCrop'] = bool(left+4 <= target['x']*512 < left+252 and top+4 <= target['y']*288 < top+140)
        result.append(entry)
    return dict(counts=dict(Counter(r['category'] for r in result)),
                searches=dict(Counter(r['search'] for r in result if r['reason']=='missing')),
                failures=result)


def render(video, failures, destination):
    """Decode sequentially: MKV frame-count/seek metadata is unreliable."""
    requested = {r['frame']: r for r in failures}
    cap = cv2.VideoCapture(str(video))
    tiles = []
    index = 0
    try:
        while index <= max(requested):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'Video ended before requested frame {max(requested)}')
            if index in requested:
                r = requested[index]
                h, w = frame.shape[:2]
                # Same 320x180 reference field of view across both resolutions.
                cx, cy = round(r['target']['x']*w), round(r['target']['y']*h)
                cw, ch = round(w/4), round(h/4)
                x = max(0, min(w-cw, cx-cw//2))
                y = max(0, min(h-ch, cy-ch//2))
                tile = cv2.resize(frame[y:y+ch,x:x+cw], (480,270))
                def point(p):
                    return round((p['x']*w-x)*480/cw), round((p['y']*h-y)*270/ch)
                cv2.drawMarker(tile, point(r['target']), (0,255,0), cv2.MARKER_CROSS, 18, 1)
                for c in r['nearbyCandidates']:
                    cv2.circle(tile, point(c), 9, (255,255,0), 1)
                if r['proposal']:
                    cv2.circle(tile, point(r['proposal']), 6, (0,255,255), 2)
                tile = cv2.copyMakeBorder(tile, 48,0,0,0, cv2.BORDER_CONSTANT)
                for j, text in enumerate([f"frame {index} | {r['timestamp']:.2f}s | {r['search']}", r['category']]):
                    cv2.putText(tile, text, (6,18+j*21), cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
                tiles.append(tile)
            index += 1
    finally:
        cap.release()
    while len(tiles)%4:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(destination), np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)]))


def main():
    out = ROOT/'output'
    downloads = Path('C:/Users/liang/Downloads')
    cases = [('720p','Wang Chuqin vs Sora Matsushima Match Highlights #ITTFWorldCup 2026.mp4','.ball-labels.json',-1),
             ('4k','LJK vs Hugo.mkv','.ball-labels (1).json',0)]
    report = {}
    for name, filename, suffix, offset in cases:
        track = json.loads((out/f'head-final-{name}-v1.json').read_text())
        labels = json.loads((downloads/(filename+suffix)).read_text(encoding='utf-8'))
        report[name] = diagnose(track,labels,offset)
        render(downloads/filename,report[name]['failures'],out/f'missed-diagnosis-{name}.jpg')
        print(name,report[name]['counts'],report[name]['searches'],flush=True)
    (out/'missed-diagnosis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
