"""Inspect 2D paddle detections from a local YOLO ONNX export on a short rally."""

import argparse
import ast
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort


def wrist_support(box, people, width, height):
    """Return an ID only when exactly one tracked person's wrists support a proposal."""
    x, y, w, h = box
    supported = set()
    for person in people:
        radius = .15 * person['box']['height'] * height
        for joint in person.get('landmarks', []):
            point = joint.get('position')
            if not joint['name'].endswith('-wrist') or not point or point.get('confidence', 0) < .35:
                continue
            u, v = point['value']['x'], point['value']['y']
            gap = math.hypot(max(x-u, 0, u-x-w)*width, max(y-v, 0, v-y-h)*height)
            if gap <= radius:
                supported.add(person['trackId'])
    return next(iter(supported)) if len(supported) == 1 else None


class PaddleDetector:
    def __init__(self, path, score, threads):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        names = ast.literal_eval(self.session.get_modelmeta().custom_metadata_map.get('names', '{}'))
        names = list(names.values()) if isinstance(names, dict) and list(names) == list(range(len(names))) else names
        if (len(inputs) != 1 or len(outputs) != 1 or inputs[0].type != 'tensor(float)' or
                not isinstance(names, list) or not all(isinstance(name, str) for name in names)):
            raise ValueError('Expected a raw, floating-point YOLO detection export with ordered class names')
        self.classes = [i for i, name in enumerate(names) if 'racket' in name.lower() or 'paddle' in name.lower()]
        if not self.classes:
            raise ValueError(f'No racket class in model metadata: {names}')
        self.names = names
        self.input = inputs[0].name
        batch, channels, self.height, self.width = inputs[0].shape
        if batch != 1 or channels != 3 or not all(isinstance(n, int) and n > 0 for n in (self.height, self.width)):
            raise ValueError('Expected a fixed-size 1×3×height×width tensor')
        self.score = score

    def detect(self, frame):
        height, width = frame.shape[:2]
        scale = min(self.width / width, self.height / height)
        resized_width, resized_height = round(width * scale), round(height * scale)
        left, top = (self.width - resized_width) // 2, (self.height - resized_height) // 2
        canvas = np.full((self.height, self.width, 3), 114, dtype=np.uint8)
        canvas[top:top+resized_height, left:left+resized_width] = cv2.resize(frame, (resized_width, resized_height))
        tensor = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
        output, = self.session.run(None, {self.input: tensor})
        if output.ndim != 3 or output.shape[:2] != (1, 4 + len(self.names)):
            raise ValueError('Expected raw YOLO boxes and class scores')
        rows = output[0].T
        labels = np.argmax(rows[:, 4:], axis=1)
        scores = rows[np.arange(len(rows)), 4+labels]
        valid = np.isin(labels, self.classes) & (scores >= self.score) & np.isfinite(rows).all(axis=1)
        boxes, kept_scores = [], []
        for (cx, cy, w, h), score in zip(rows[valid, :4], scores[valid]):
            x1, x2 = np.clip((np.array([cx-w/2, cx+w/2]) - left) / scale, 0, width)
            y1, y2 = np.clip((np.array([cy-h/2, cy+h/2]) - top) / scale, 0, height)
            if x2 > x1 and y2 > y1:
                boxes.append([float(x1), float(y1), float(x2-x1), float(y2-y1)])
                kept_scores.append(float(score))
        return [{'box': [x/width, y/height, w/width, h/height], 'score': kept_scores[i]}
                for i in cv2.dnn.NMSBoxes(boxes, kept_scores, self.score, .45)
                for x, y, w, h in [boxes[i]]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--session', type=Path, help='Optional matching session with ball tracks and shot tags')
    parser.add_argument('--player-tracks', type=Path, help='Optional matching player tracks with wrist landmarks')
    parser.add_argument('--start', type=float, required=True)
    parser.add_argument('--end', type=float, required=True)
    parser.add_argument('--score', type=float, default=.25)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True, help='Output directory for JSON and review video')
    args = parser.parse_args()
    if not 0 <= args.start < args.end < float('inf') or not 0 < args.score < 1 or args.threads < 1:
        parser.error('Choose a finite time range, score between 0 and 1, and positive thread count')
    cv2.setNumThreads(1)
    detector = PaddleDetector(args.model, args.score, args.threads)
    capture = cv2.VideoCapture(str(args.video))
    writer = None
    rows = []
    elapsed = 0.
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        width, height = (int(capture.get(prop)) for prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
        if not capture.isOpened() or not 0 < fps <= 240 or min(width, height) <= 0:
            raise ValueError('Cannot read video metadata')
        video = dict(name=args.video.name, size=args.video.stat().st_size, width=width, height=height)
        ball, tags, people_by_frame = {}, {}, {}
        if args.session:
            session = json.loads(args.session.read_text(encoding='utf-8'))
            if (any(session['video'].get(key) != value for key, value in video.items()) or
                    abs(session['fps'] - fps) > .001):
                raise ValueError('Session metadata must match the input video')
            tags = {round(s['time'] * fps): s for s in session['shots']}
            ball = {round(b['timestamp'] * fps): b['position']['value'] for b in session['ballObservations']}
        if args.player_tracks:
            track = json.loads(args.player_tracks.read_text(encoding='utf-8'))
            if (track.get('format') != 'pongmetrics-player-tracks' or
                    track.get('coordinateSystem') != 'normalized-video-image' or
                    any(track['video'].get(key) != value for key, value in video.items()) or
                    abs(track['fps'] - fps) > .001):
                raise ValueError('Player-track metadata must match the input video')
            for row in track['frames']:
                n = round(row['timestamp'] * fps)
                if n in people_by_frame or abs(row['timestamp'] * fps - n) > .25:
                    raise ValueError('Player-track timestamps must align with unique video frames')
                people_by_frame[n] = row['people']
        args.output.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.output / 'racket-review.webm'), cv2.VideoWriter_fourcc(*'VP80'), fps/2, (width, height))
        if not writer.isOpened():
            raise RuntimeError('Could not open the review video writer')
        frame_number = -1
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if timestamp >= args.end:
                break
            if timestamp + 1e-6 < args.start:
                continue
            if ((rows and timestamp <= rows[-1]['timestamp']) or
                    ((args.session or args.player_tracks) and abs(timestamp * fps - frame_number) > .25)):
                raise ValueError('Source timestamps do not align with the review frames')
            began = time.perf_counter()
            paddles = detector.detect(frame)
            elapsed += time.perf_counter()-began
            ball_point = ball.get(frame_number)
            tag = tags.get(frame_number)
            rows.append(dict(frame=frame_number, timestamp=timestamp, paddles=paddles, ball=ball_point,
                             taggedPlayer=tag.get('playerId') if tag else None))
            for paddle in paddles:
                x, y, w, h = paddle['box']
                track_id = wrist_support(paddle['box'], people_by_frame.get(frame_number, []), width, height)
                paddle['wristSupportedTrackId'] = track_id
                if args.player_tracks and track_id is None:
                    continue
                cv2.rectangle(frame, (round(x*width), round(y*height)), (round((x+w)*width), round((y+h)*height)), (255, 224, 54), 2)
                if track_id is not None:
                    cv2.putText(frame, str(track_id), (round(x*width), max(12, round(y*height)-4)),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 224, 54), 1, cv2.LINE_AA)
            if ball_point:
                cv2.circle(frame, (round(ball_point['x']*width), round(ball_point['y']*height)), 6, (0, 255, 255), 2)
            box_label = 'paddle near wrist' if args.player_tracks else 'paddle proposals'
            label = f'{timestamp:.2f}s  |  Cyan: {box_label}  |  Yellow: ball track'
            if tag:
                label += '  |  Tagged shot'
            cv2.rectangle(frame, (0, 0), (width, 34), (24, 21, 20), -1)
            cv2.putText(frame, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, .55, (245, 245, 245), 1, cv2.LINE_AA)
            writer.write(frame)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if not rows:
        raise ValueError('No decoded frames in the requested interval')
    summary = {'frames': len(rows), 'framesWithPaddleProposals': sum(bool(r['paddles']) for r in rows),
               'paddleProposals': sum(len(r['paddles']) for r in rows), 'detectionSeconds': elapsed}
    if args.player_tracks:
        counts = [sum(p['wristSupportedTrackId'] is not None for p in row['paddles']) for row in rows]
        summary.update(wristSupportedProposals=sum(counts), framesWithWristSupport=sum(count > 0 for count in counts))
    payload = {'video': video, 'fps': fps, 'coordinateSystem': 'normalized-video-image', 'boxFormat': 'xywh',
               'model': args.model.name, 'classes': detector.names, 'scoreThreshold': args.score, 'summary': summary, 'frames': rows}
    (args.output / 'racket-review.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
