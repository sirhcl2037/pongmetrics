"""Read original OpenTTGames annotations and score source-frame-aligned segments."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile


def read_annotations(archive):
    with zipfile.ZipFile(archive) as source:
        balls = {int(k): v for k, v in json.loads(source.read('ball_markup.json')).items()}
        events = {int(k): v for k, v in json.loads(source.read('events_markup.json')).items()}
    for frame, point in balls.items():
        if frame < 0 or set(point) != {'x', 'y'} or not all(math.isfinite(v) for v in point.values()):
            raise ValueError('Invalid OpenTT ball annotation')
    return balls, events


def evaluate(detections, balls, events, start_frame, end_frame, source_fps):
    if not 0 <= start_frame < end_frame or not 0 < source_fps <= 240:
        raise ValueError('Invalid evaluation range or FPS')
    if abs(detections['fps'] - source_fps) > 1e-6:
        raise ValueError('Evaluation requires source-rate predictions')
    processed = set()
    for row in detections['trackingFrames']:
        index = round(row['timestamp'] * source_fps)
        frame = start_frame + index
        if row.get('frame', index) != index or frame in processed or not start_frame <= frame < end_frame:
            raise ValueError('Tracking rows do not match the source frame grid')
        processed.add(frame)
    predictions = {}
    for row in detections['ballObservations']:
        frame = start_frame + round(row['timestamp'] * source_fps)
        if frame not in processed or frame in predictions:
            raise ValueError('Ball prediction does not match a unique tracking frame')
        predictions[frame] = row['position']['value']
    width, height = detections['video']['width'], detections['video']['height']
    visible = absent = missed = false_positive = unprocessed = ambiguous = 0
    errors, failures = [], []
    for frame, point in sorted(balls.items()):
        if not start_frame <= frame < end_frame:
            continue
        if frame not in processed:
            unprocessed += 1
            continue
        prediction = predictions.get(frame)
        if point == {'x': -1, 'y': -1}:
            absent += 1
            if prediction:
                false_positive += 1
                failures.append({'frame': frame, 'reason': 'prediction-on-labelled-absence'})
        elif min(point.values()) < 0:
            ambiguous += 1
        else:
            if not (0 <= point['x'] < width and 0 <= point['y'] < height):
                raise ValueError('Ball label exceeds source dimensions')
            visible += 1
            if prediction is None:
                missed += 1
                failures.append({'frame': frame, 'reason': 'missing'})
                continue
            error = math.hypot((prediction['x'] - point['x'] / width) * 1280,
                               (prediction['y'] - point['y'] / height) * 720)
            errors.append(error)
            if error > 20:
                failures.append({'frame': frame, 'reason': 'outside-20-reference-pixels', 'error': error})
    return {'sourceRange': [start_frame, end_frame], 'visibleLabels': visible, 'absenceLabels': absent,
            'unprocessedLabels': unprocessed, 'ambiguousLabelsExcluded': ambiguous, 'unlabelledFrames': sum(
                frame not in balls for frame in processed if start_frame <= frame < end_frame),
            'missing': missed, 'predictionsOnAbsence': false_positive,
            'withinReferencePixels': {str(limit): sum(e <= limit for e in errors) for limit in (5, 10, 20)},
            'medianReferenceErrorPixels': statistics.median(errors) if errors else None,
            'maxReferenceErrorPixels': max(errors, default=None),
            'recallAt20ReferencePixels': sum(e <= 20 for e in errors) / visible if visible else None,
            'events': [{'sourceFrame': frame, 'timestamp': (frame-start_frame)/source_fps, 'label': label}
                       for frame, label in sorted(events.items()) if start_frame <= frame < end_frame],
            'failures': failures, 'runtime': detections.get('summary', {})}


def prepare(video, archive, output, start, end):
    import cv2
    source = cv2.VideoCapture(str(video))
    fps, count = source.get(5), int(source.get(7))
    width, height = int(source.get(3)), int(source.get(4))
    if not source.isOpened() or not 0 <= start < end <= count:
        source.release()
        raise ValueError('Invalid source or segment bounds')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        source.release()
        raise ValueError('Output already exists')
    source.set(cv2.CAP_PROP_POS_FRAMES, start)
    pending = output.with_suffix('.pending.avi')
    writer = cv2.VideoWriter(str(pending), cv2.VideoWriter_fourcc(*'FFV1'), fps, (width, height))
    if not writer.isOpened():
        source.release()
        raise ValueError('Cannot open lossless segment writer')
    sample_hashes = {}
    try:
        for index in range(start, end):
            ok, frame = source.read()
            if not ok or round(source.get(1)) != index + 1:
                raise ValueError('Source frame alignment failed')
            writer.write(frame)
            if index in (start, (start+end)//2, end-1):
                sample_hashes[index-start] = hashlib.sha256(frame.tobytes()).hexdigest()
    finally:
        source.release()
        writer.release()
    check = cv2.VideoCapture(str(pending))
    try:
        for frame_index, digest in sample_hashes.items():
            check.set(1, frame_index)
            ok, frame = check.read()
            if not ok or hashlib.sha256(frame.tobytes()).hexdigest() != digest:
                raise ValueError('Lossless segment verification failed')
    finally:
        check.release()
    metadata = {'sourceVideo': video.name, 'sourceSize': video.stat().st_size, 'sourceRange': [start, end],
                'fps': fps, 'width': width, 'height': height, 'decodedSampleHashes': sample_hashes,
                'archiveSha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                'annotationConvention': 'zero-based source frame, pixels, (-1,-1) absent; missing keys unlabelled'}
    output.with_suffix('.source.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    pending.replace(output)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--video', type=Path)
    parser.add_argument('--detections', type=Path)
    parser.add_argument('--start-frame', type=int, required=True)
    parser.add_argument('--end-frame', type=int, required=True)
    parser.add_argument('--fps', type=float, default=120)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if bool(args.video) == bool(args.detections):
        parser.error('Choose video extraction or detection evaluation')
    balls, events = read_annotations(args.annotations)
    if args.video:
        result = prepare(args.video, args.annotations, args.output, args.start_frame, args.end_frame)
    else:
        detections = json.loads(args.detections.read_text(encoding='utf-8'))
        source_path = args.detections.parent / Path(detections['video']['name']).with_suffix('.source.json')
        metadata = json.loads(source_path.read_text(encoding='utf-8'))
        if (metadata['sourceRange'] != [args.start_frame, args.end_frame] or
                metadata['archiveSha256'] != hashlib.sha256(args.annotations.read_bytes()).hexdigest() or
                metadata['fps'] != args.fps or
                metadata['width'] != detections['video']['width'] or
                metadata['height'] != detections['video']['height']):
            raise ValueError('Detection segment and source annotation metadata mismatch')
        result = evaluate(detections, balls, events, args.start_frame, args.end_frame, args.fps)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('failures', 'events')}, indent=2))


if __name__ == '__main__':
    main()
