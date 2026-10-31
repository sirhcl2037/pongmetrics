"""Experimental color/motion tracker. Predictions are never exported as detections."""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def classify_ball(hsv, position):
    """Classify once from bright pixels close to the trusted initial seed."""
    x, y = np.clip(np.round(position), [0, 0], [hsv.shape[1]-1, hsv.shape[0]-1]).astype(int)
    patch = hsv[max(0,y-3):y+4, max(0,x-3):x+4].reshape(-1, 3)
    bright = patch[patch[:,2] >= max(150, float(np.percentile(patch[:,2], 80)))]
    if not len(bright) or float(np.median(bright[:,1])) > 100:
        raise ValueError('Initial seed does not contain a clear white ball; choose a clearer seed. Orange tracking is not supported yet.')
    return {'color':'white', 'maxSaturation':min(105, max(55, int(np.percentile(bright[:,1], 80))+25)), 'minValue':max(140, min(190, int(np.median(bright[:,2]))-40)), 'source':'initial-seed', 'locked':True}


def select_candidates(hsv, gray, previous, predicted, radius, appearance):
    cx, cy = np.clip(predicted, [0,0], [gray.shape[1]-1,gray.shape[0]-1]).astype(int)
    x0, x1 = max(0,cx-radius), min(gray.shape[1],cx+radius+1)
    y0, y1 = max(0,cy-radius), min(gray.shape[0],cy+radius+1)
    roi = hsv[y0:y1,x0:x1]
    local = gray[y0:y1,x0:x1]
    delta = cv2.absdiff(local, previous[y0:y1,x0:x1])
    # A white blur may be elongated, but must remain brighter than its surroundings.
    contrast = cv2.subtract(local, cv2.morphologyEx(local, cv2.MORPH_OPEN, np.ones((9,9),np.uint8)))
    mask = cv2.inRange(roi, (0,0,appearance['minValue']), (179,appearance['maxSaturation'],255))
    mask &= ((delta > 10) & (contrast > 8)).astype(np.uint8)*255
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    candidates = []
    for k in range(1,count):
        area = int(stats[k,cv2.CC_STAT_AREA])
        width, height = stats[k,2:4]
        if not 1 <= area <= 40 or max(width,height) > 20:
            continue
        pixels = labels == k
        point = centers[k]+[x0,y0]
        distance = float(np.linalg.norm(point-predicted))
        proximity = np.exp(-distance/max(radius*.5,1))
        whiteness = 1-float(roi[:,:,1][pixels].mean())/255
        prominence = min(1,float(contrast[pixels].mean())/45)
        motion = min(1,float(delta[pixels].mean())/45)
        score = float(.4*proximity+.2*whiteness+.25*prominence+.15*motion)
        candidates.append((score,point))
    return sorted(candidates,key=lambda c:c[0],reverse=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('video', type=Path)
    p.add_argument('--seed', nargs=3, type=float, required=True, metavar=('SECONDS', 'X', 'Y'), help='Clear ball seed; x/y normalized to image')
    p.add_argument('--end', type=float, default=8.28)
    p.add_argument('--heavy', action='store_true', help='Use real TrackNet recovery and periodic verification')
    p.add_argument('--verify-every', type=int, default=5, help='Maximum interval between heavy checks; 1 gives baseline detection output on every frame after the seed')
    p.add_argument('--output', type=Path, default=Path(__file__).parent/'output'/'fast-track.json')
    a = p.parse_args()
    start, x, y = a.seed
    if a.verify_every < 1:
        p.error('verify-every must be positive')
    if not 0 <= x <= 1 or not 0 <= y <= 1 or start < 0 or a.end <= start:
        p.error('Invalid seed or end time')
    began = time.perf_counter()
    detector = None
    if a.heavy:
        from heavy_detector import HeavyDetector
        detector = HeavyDetector(a.video, a.end)
    cap = cv2.VideoCapture(str(a.video))
    if not cap.isOpened():
        raise RuntimeError('Cannot open video')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not 0 < fps <= 240:
        raise RuntimeError('Unsupported FPS')
    size = (640, 360)
    position = np.array([x*size[0], y*size[1]], dtype=float)
    velocity = np.zeros(2)
    previous = None
    appearance = None
    misses = 0
    records = []
    elapsed = 0.0
    last_observed = position.copy()
    last_observed_time = start
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
        if timestamp < start:
            continue
        if timestamp >= a.end:
            break
        tick = time.perf_counter()
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if appearance is None:
            appearance = classify_ball(hsv, position)
        predicted = position + velocity
        radius = min(140, 24 + misses*12 + int(np.linalg.norm(velocity)*.5))
        candidates = select_candidates(hsv, gray, previous, predicted, radius, appearance) if previous is not None and misses <= 5 else []
        # A close runner-up means appearance is ambiguous: retain a prediction.
        detected = bool(candidates) and candidates[0][0] > .85 and (len(candidates) == 1 or candidates[0][0]-candidates[1][0] > .12)
        visual_candidate = candidates[0] if detected else None
        heavy_checked = detector is not None and previous is not None and (not detected or len(records) % a.verify_every == 0)
        heavy_result = detector.detect(timestamp) if heavy_checked else None
        if heavy_checked:
            detected = heavy_result is not None
            if detected:
                candidates = [(heavy_result[1], heavy_result[0]*size)]
            elif visual_candidate is not None:
                # Retain image evidence in heavy-detector gaps for manual review.
                # Do not feed an unverified recovery back into the motion state.
                candidates = [visual_candidate]
        if previous is None:
            kind, confidence = 'seed', 1.0
        elif detected:
            point = candidates[0][1]
            interval = max(1, (timestamp-last_observed_time)*fps)
            velocity = .3*velocity + .7*(point-last_observed)/interval
            last_observed, last_observed_time = point.copy(), timestamp
            position = point
            misses = 0
            kind, confidence = 'detected', candidates[0][0]
        else:
            position = predicted
            misses += 1
            velocity *= .95
            kind, confidence = ('predicted' if misses <= 5 else 'lost'), float(np.exp(-misses/2))
        record = {'timestamp': timestamp, 'position': {'x': float(position[0]/640), 'y': float(position[1]/360)}, 'kind': kind, 'method': 'heavy' if heavy_checked else 'color-motion', 'confidence': confidence, 'uncertaintyRadiusPixels': radius, 'needsDetector': misses > 5}
        record['reviewRequired'] = kind == 'detected' and not heavy_checked
        if heavy_checked and heavy_result is None and visual_candidate is not None:
            score, point = visual_candidate
            record['recoveryCandidate'] = {'position':{'x':float(point[0]/640),'y':float(point[1]/360)},'score':score,'reviewRequired':True}
        records.append(record)
        previous = gray
        elapsed += time.perf_counter()-tick
    cap.release()
    total = time.perf_counter()-began
    summary = {'frames': len(records), 'detected': sum(r['kind']=='detected' for r in records), 'predicted': sum(r['kind']=='predicted' for r in records), 'lost': sum(r['kind']=='lost' for r in records), 'trackingSeconds': elapsed, 'trackingFps': len(records)/elapsed if elapsed else 0, 'decodeAndTrackingSeconds': total}
    summary['heavyCalls'] = detector.calls if detector else 0
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({'format':'pongmetrics-fast-track-experiment', 'version':1, 'fps':fps, 'workingSize':size, 'ballAppearance':appearance, 'summary':summary, 'frames':records}, indent=2), encoding='utf8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
