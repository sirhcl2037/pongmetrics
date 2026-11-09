"""Evaluate browser labels without modifying them; report alignment separately."""
import argparse
import json
import math
import statistics
from pathlib import Path


def evaluate(detections, labels, offset):
    for key in ('name', 'size', 'width', 'height'):
        if detections['video'][key] != labels['video'][key]:
            raise ValueError(f'Video {key} mismatch')
    fps = labels['fps']
    if detections.get('fps') != fps:
        raise ValueError('FPS mismatch')
    processed = ({round(row['timestamp'] * fps) for row in detections['trackingFrames']}
                 if 'trackingFrames' in detections else None)
    predictions = {round(row['timestamp'] * fps): row['position']['value']
                   for row in detections['ballObservations']}
    visible = missing = negatives = false_positives = uncertain = 0
    off_grid = 0
    unprocessed = 0
    failures = []
    reference_failures = []
    errors = []
    reference_errors = []
    seen = set()
    width, height = labels['video']['width'], labels['video']['height']
    for row in labels['labels']:
        timestamp = row['timestamp']
        if timestamp in seen:
            raise ValueError('Duplicate label timestamp')
        seen.add(timestamp)
        if abs(timestamp * fps - round(timestamp * fps)) > .001:
            off_grid += 1
            continue
        frame = round(timestamp * fps) + offset
        if processed is not None and frame not in processed:
            unprocessed += 1
            continue
        prediction = predictions.get(frame)
        status = row['visibility']
        if status == 'uncertain':
            uncertain += 1
        elif status == 'visible':
            visible += 1
            if prediction is None:
                missing += 1
                failures.append(dict(frame=frame, reason='missing'))
                reference_failures.append(dict(frame=frame, reason='missing'))
            else:
                target = row['position']['value']
                errors.append(math.hypot((prediction['x'] - target['x']) * width,
                                         (prediction['y'] - target['y']) * height))
                reference_errors.append(math.hypot((prediction['x'] - target['x']) * 1280,
                                                   (prediction['y'] - target['y']) * 720))
                if errors[-1] > 20:
                    failures.append(dict(frame=frame, reason='wrong-location', pixels=errors[-1]))
                if reference_errors[-1] > 20:
                    reference_failures.append(dict(frame=frame, reason='outside-center-tolerance', referencePixels=reference_errors[-1]))
        elif status in ('occluded', 'out-of-frame'):
            negatives += 1
            false_positives += prediction is not None
            if prediction is not None:
                failures.append(dict(frame=frame, reason='false-positive'))
                reference_failures.append(dict(frame=frame, reason='false-positive'))
        else:
            raise ValueError(f'Unknown visibility {status}')
    correct = sum(error <= 20 for error in reference_errors)
    incorrect = len(reference_errors)-correct+false_positives
    return dict(predictionFrameOffset=offset, visibleFrames=visible, missingDetections=missing,
                proposalCoverage=len(errors)/visible if visible else None,
                withinPixels={str(t): sum(error <= t for error in errors) for t in (5, 10, 20)},
                medianErrorPixels=statistics.median(errors) if errors else None,
                negativeFrames=negatives, negativeFalsePositives=false_positives,
                uncertainFramesExcluded=uncertain, offGridLabelsExcluded=off_grid,
                unprocessedLabelsExcluded=unprocessed, failures=failures,
                referenceResolution=[1280,720],
                withinReferencePixels={str(t):sum(error <= t for error in reference_errors) for t in (5,10,20,30)},
                referencePrecision=correct/(correct+incorrect) if correct+incorrect else None,
                referenceRecall=correct/visible if visible else None,
                referenceFailures=reference_failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('detections', type=Path)
    parser.add_argument('labels', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frame-offset', type=int, default=0,
                        help='Explicit reviewed alignment adjustment, in detector frames')
    args = parser.parse_args()
    detections = json.loads(args.detections.read_text(encoding='utf-8'))
    labels = json.loads(args.labels.read_text(encoding='utf-8'))
    report = dict(
        recordedTimestampEvaluation=evaluate(detections, labels, 0),
        selectedAlignmentEvaluation=evaluate(detections, labels, args.frame_offset),
        alignmentDiagnostics=[evaluate(detections, labels, shift) for shift in range(-2, 3)],
        note='Offsets are diagnostics, not corrected ground truth. Confirm alignment visually before tuning. '
             'Missing predictions include footage the detector did not process. '
             'Pixel errors use original video resolution; reference metrics scale positions to 1280x720 for cross-resolution comparison. '
             'Accepted detector-assisted positions are review-approved, not independent manual center measurements.',
        runtime=detections.get('summary', {}).get('sourceRun', {}))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
