"""Compare hybrid observations with saved TrackNet outputs, not ground truth."""
import argparse
import json
import numpy as np


def compare(reference, hybrid):
    fps = hybrid['fps']
    index = {round(o['timestamp']*fps):o['position']['value'] for o in reference['ballObservations']}
    differences, missed, extra = [], [], []
    scale = np.array(hybrid['workingSize'])
    for row in hybrid['frames']:
        if row['kind'] == 'seed':
            continue
        baseline = index.get(round(row['timestamp']*fps))
        observed = row['kind'] == 'detected'
        if baseline is not None and observed:
            point = row['position']
            differences.append(float(np.linalg.norm((np.array([point['x'],point['y']])-np.array([baseline['x'],baseline['y']]))*scale)))
        elif baseline is not None:
            missed.append(row['timestamp'])
        elif observed:
            extra.append(row['timestamp'])
    return {'sharedDetections':len(differences),'missedBaselineTimestamps':missed,'extraDetectionTimestamps':extra,'medianDifferencePixels':float(np.median(differences)) if differences else None,'maxDifferencePixels':max(differences) if differences else None,'note':'Agreement with model outputs is not labeled accuracy; seed excluded.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference')
    p.add_argument('hybrid')
    a = p.parse_args()
    with open(a.reference,encoding='utf8') as f:
        reference=json.load(f)
    with open(a.hybrid,encoding='utf8') as f:
        hybrid=json.load(f)
    print(json.dumps(compare(reference,hybrid),indent=2))
