"""Associate neural image evidence; predictions never become observations."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

SCALE = np.array([512., 288.])


def xy(candidate):
    return np.array([candidate['x'], candidate['y']])*SCALE


class BallTracker:
    def __init__(self, fps, strong_score=.5, acquisition_score=None):
        if not 0 < strong_score <= 1:
            raise ValueError('Association score must be between zero and one')
        self.fps = fps
        self.strong_score = strong_score
        self.acquisition_score = strong_score if acquisition_score is None else acquisition_score
        if not strong_score <= self.acquisition_score <= 1:
            raise ValueError('New tracks require at least the association confidence')
        self.position = None
        self.velocity = np.zeros(2)  # Pixels per second, not per decoded frame.
        self.last_time = None
        self.previous_time = None
        self.pending = None
        self.segment = 0
        self.rows = []

    def guidance(self, timestamp):
        if self.position is None or timestamp-self.last_time > .32:
            return None
        dt = timestamp-self.last_time
        # Unseen contacts can reverse velocity. Limit extrapolation to 2 frames.
        prediction = self.position+self.velocity*min(dt, 2/self.fps)
        radius = min(100., 20.+dt*self.fps*10.+np.linalg.norm(self.velocity)/self.fps*.5)
        return prediction, radius

    def update(self, frame, timestamp, candidates, scene_cut=False):
        if not np.isfinite(timestamp) or (self.previous_time is not None and timestamp <= self.previous_time):
            raise ValueError('Tracking requires strictly increasing decoder timestamps')
        if self.previous_time is not None and timestamp-self.previous_time > 2.1/self.fps:
            scene_cut = True  # Discontinuous input must not inherit stale momentum.
        self.previous_time = timestamp
        if scene_cut:
            self.position, self.last_time, self.pending = None, None, None
            self.velocity[:] = 0
        for c in candidates:
            if not all(np.isfinite(c[k]) for k in ('x', 'y', 'score')) or not (0 <= c['x'] <= 1 and 0 <= c['y'] <= 1 and 0 <= c['score'] <= 1):
                raise ValueError('Invalid neural candidate')
        guidance = self.guidance(timestamp)
        row = dict(frame=frame, timestamp=timestamp, position=None, kind='lost',
                   confidence=0., reason='no-associated-evidence', candidates=candidates,
                   prediction=None, uncertaintyRadiusPixels=None, sceneCut=scene_cut,
                   trackId=None)
        if guidance is not None:
            prediction, radius = guidance
            row.update(prediction=dict(x=float(prediction[0]/512), y=float(prediction[1]/288)),
                       uncertaintyRadiusPixels=radius, kind='occluded')
        self.rows.append(row)
        strong = [c for c in candidates if c['score'] >= self.strong_score]
        selected = None
        if guidance is not None:
            ranked = sorted([(np.linalg.norm(xy(c)-prediction)/radius-.2*c['score'], c)
                             for c in strong if np.linalg.norm(xy(c)-prediction) <= radius], key=lambda v: v[0])
            if ranked and (len(ranked) == 1 or ranked[1][0]-ranked[0][0] >= .15):
                selected = ranked[0][1]
        if selected is not None:
            self._accept(row, selected, 'associated')
            self.pending = None
        else:
            # Global search is available on every frame. Never let an old crop
            # or old velocity prevent reacquisition. Require a short tracklet.
            acquisition = [c for c in strong if c['score'] >= self.acquisition_score]
            unique = acquisition[0] if len(acquisition) == 1 else None
            pending = self.pending
            if unique is not None and pending is not None:
                old, old_row = pending
                step = (timestamp-old_row['timestamp'])*self.fps
                distance = np.linalg.norm(xy(unique)-xy(old))
                if .5 <= step <= 1.5 and distance <= 90*step:
                    self.position = None
                    self.segment += 1
                    self._accept(old_row, old, 'reacquired-tracklet')
                    self._accept(row, unique, 'reacquired-tracklet')
                    self.pending = None
                    return row
            self.pending = (unique, row) if unique is not None else None
        return row

    def _accept(self, row, candidate, reason):
        point = xy(candidate)
        if self.position is not None:
            dt = row['timestamp']-self.last_time
            if dt > 0:
                # Do not average a contact reversal with stale incoming motion.
                instantaneous = (point-self.position)/dt
                reversal = float(np.dot(instantaneous, self.velocity)) < 0
                self.velocity = instantaneous if reversal else .75*instantaneous+.25*self.velocity
        else:
            self.velocity[:] = 0
            if self.segment == 0:
                self.segment = 1
        self.position, self.last_time = point, row['timestamp']
        row.update(position=dict(x=candidate['x'], y=candidate['y']), kind='observed',
                   confidence=candidate['score'], reason=reason, trackId=self.segment)

    def finish(self):
        # Offline two-sided confirmation: weak peaks must be supported on BOTH
        # sides by strong observations, at most 3 intervening frames apart.
        # Coordinates always come from a real heatmap component, never the line.
        observed = [i for i, r in enumerate(self.rows) if r['position'] is not None]
        # Offline backward confirmation can recover the first blurred frame of
        # a reacquired track, using TWO subsequent strong observations. Do not
        # bootstrap more weak frames from another weak frame.
        for i in observed:
            if i == 0 or i+1 >= len(self.rows):
                continue
            weak, a, b = self.rows[i-1:i+2]
            if weak['position'] is not None or a['confidence'] < self.strong_score or b['confidence'] < self.strong_score or b['trackId'] != a['trackId']:
                continue
            if a['sceneCut'] or b['sceneCut'] or b['timestamp']-weak['timestamp'] > 2.1/self.fps:
                continue
            expected = xy(a['position'])-(xy(b['position'])-xy(a['position']))*(a['timestamp']-weak['timestamp'])/(b['timestamp']-a['timestamp'])
            ranked = sorted([(np.linalg.norm(xy(c)-expected), c) for c in weak['candidates'] if c['score'] >= .2], key=lambda v: v[0])
            if ranked and ranked[0][0] <= 8 and (len(ranked) == 1 or ranked[1][0]-ranked[0][0] > 6):
                c = ranked[0][1]
                weak.update(position=dict(x=c['x'], y=c['y']), kind='observed', confidence=c['score'],
                            reason='backward-confirmed-weak-evidence', trackId=a['trackId'])
        for left, right in zip(observed, observed[1:]):
            if not 1 < right-left <= 4:
                continue
            a, b = self.rows[left], self.rows[right]
            if a['trackId'] != b['trackId'] or any(r['sceneCut'] for r in self.rows[left+1:right+1]):
                continue
            duration = b['timestamp']-a['timestamp']
            if duration*self.fps > 4.1:
                continue
            pa, pb = xy(a['position']), xy(b['position'])
            for row in self.rows[left+1:right]:
                t = (row['timestamp']-a['timestamp'])/duration
                expected = pa*(1-t)+pb*t
                ranked = sorted([(np.linalg.norm(xy(c)-expected), c) for c in row['candidates']
                                 if c['score'] >= .15], key=lambda v: v[0])
                # Small residual for blur; reject competing nearby peaks.
                if ranked and ranked[0][0] <= 12 and (len(ranked) == 1 or ranked[1][0]-ranked[0][0] > 6):
                    c = ranked[0][1]
                    row.update(position=dict(x=c['x'], y=c['y']), kind='observed',
                               confidence=c['score'], reason='two-sided-weak-evidence', trackId=a['trackId'])
        return self.rows


def make_payload(cache, rows, seconds):
    observations = [dict(timestamp=r['timestamp'], position=dict(value=r['position'],
                    source='automatic', confidence=r['confidence'])) for r in rows if r['position'] is not None]
    return dict(format='pongmetrics-ball-detections', version=1,
                coordinateSystem='normalized-video-image', video=cache['video'], fps=cache['fps'],
                model={**cache.get('model', {}), 'tracker': 'temporal-neural-v1',
                       'confidenceMeaning': 'uncalibrated heatmap peak; weak evidence uses future confirmation',
                       'interpolation': False},
                summary=dict(frames=len(rows), observations=len(observations),
                             weakRecoveries=sum('weak-evidence' in r['reason'] for r in rows),
                             tracklets=len({r['trackId'] for r in rows if r['trackId'] is not None}),
                             associationSeconds=seconds, sourceRun=cache.get('summary', {})),
                ballObservations=observations, trackingFrames=rows)


def recover_peak_gaps(rows, fps):
    """Experimental offline confirmation of real maxima between strong anchors."""
    anchors = [i for i, row in enumerate(rows) if row['position'] is not None and row['confidence'] >= .5]
    for left, right in zip(anchors, anchors[1:]):
        a, b = rows[left], rows[right]
        duration = b['timestamp'] - a['timestamp']
        if not 1 < right-left <= 6 or duration * fps > 6.1:
            continue
        if any(row['sceneCut'] for row in rows[left+1:right+1]):
            continue
        for row in rows[left+1:right]:
            if row['position'] is not None:
                continue
            fraction = (row['timestamp'] - a['timestamp']) / duration
            expected = xy(a['position']) * (1-fraction) + xy(b['position']) * fraction
            ranked = sorted([(float(np.linalg.norm(xy(c)-expected)), c)
                             for c in row['candidates'] if c.get('evidence') == 'local-maximum' and c['score'] >= .03],
                            key=lambda item: item[0])
            if not ranked or ranked[0][0] > 8:
                continue
            if len(ranked) > 1 and ranked[1][0]-ranked[0][0] <= 4:
                continue
            candidate = ranked[0][1]
            row.update(position=dict(x=candidate['x'], y=candidate['y']), kind='observed',
                       confidence=candidate['score'], reason='two-sided-local-peak-weak-evidence', trackId=a['trackId'])
    return rows


def replay(cache):
    began = time.perf_counter()
    if cache.get('format') != 'pongmetrics-neural-candidates' or not 0 < cache['fps'] <= 240:
        raise ValueError('Unsupported candidate cache')
    tracker = BallTracker(cache['fps'], cache.get('model', {}).get('strongScore', .5), cache.get('model', {}).get('acquisitionScore'))
    previous_frame = -1
    for row in cache['frames']:
        if not isinstance(row['frame'], int) or row['frame'] <= previous_frame:
            raise ValueError('Candidate frame indices must strictly increase')
        previous_frame = row['frame']
        original = [c for c in row['candidates'] if c.get('evidence') != 'local-maximum']
        tracker.update(row['frame'], row['timestamp'], original, row.get('sceneCut', False))
    rows = tracker.finish()
    if cache.get('model', {}).get('weakPeaks'):
        for row, source in zip(rows, cache['frames']):
            row['candidates'] = source['candidates']
        rows = recover_peak_gaps(rows, cache['fps'])
    return make_payload(cache, rows, time.perf_counter()-began)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--candidates', type=Path, help='Replay raw candidates without neural inference')
    source.add_argument('--video', type=Path, help='Run streaming TrackNet on a local video')
    p.add_argument('--end', type=float, default=float('inf'))
    p.add_argument('--start', type=float, default=0)
    p.add_argument('--native-zoom', action='store_true', help='Experimental original-image quarter-area crops resized to 512x288')
    model = p.add_mutually_exclusive_group()
    model.add_argument('--head-adapter', type=Path, help='Explicit checkpoint-bound adapter; otherwise use the validated local default when present')
    model.add_argument('--base-model', action='store_true', help='Use original pretrained weights and original tracking defaults')
    p.add_argument('--strong-score', type=float, help='Override association cutoff for uncalibrated heatmap peaks')
    p.add_argument('--acquisition-score', type=float, help='Confidence required to start a new track; defaults to association cutoff')
    p.add_argument('--background-samples', type=int, default=24)
    p.add_argument('--candidate-floor', type=float, default=.15)
    p.add_argument('--weak-peaks', action='store_true', help='Experimental additional local maxima; requires validation')
    p.add_argument('--full-frame', action='store_true', help='Disable guided crops for controlled comparisons')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--save-candidates', type=Path)
    a = p.parse_args()
    if a.start < 0 or a.end <= a.start or a.background_samples < 0 or not 0 < a.candidate_floor < .5:
        p.error('Invalid end or sample count')
    if a.candidates:
        if a.head_adapter or a.base_model:
            p.error('Cached evidence already belongs to a model; model selection requires --video')
        cache = json.loads(a.candidates.read_text(encoding='utf8'))
        if a.strong_score is not None:
            cache.setdefault('model', {})['strongScore'] = a.strong_score
        if a.acquisition_score is not None:
            cache.setdefault('model', {})['acquisitionScore'] = a.acquisition_score
        payload = replay(cache)
    else:
        from neural_candidates import StreamingDetector
        began = time.perf_counter()
        from head_adapter import model_settings
        try:
            adapter, association, acquisition, adapter_name = model_settings(a.head_adapter,a.base_model,a.strong_score,a.acquisition_score)
        except (ValueError,OSError) as error:
            p.error(str(error))
        if a.native_zoom and a.weak_peaks:
            p.error('Compare native zoom independently of weak-peak recovery')
        detector = StreamingDetector(a.video, a.end, a.background_samples, a.candidate_floor, a.weak_peaks, a.native_zoom, adapter)
        tracker = BallTracker(detector.fps, association, acquisition)
        source_rows = []
        failures = 0
        rescue_calls = 0
        for i, timestamp, history, cut in detector.frames():
            if timestamp < a.start:
                continue
            guidance = None if cut else tracker.guidance(timestamp)
            center = None
            # Reacquire globally after ONE failed crop, and every half second.
            # Crop covers both sides of motion reversal at the original scale.
            if not a.full_frame and guidance and failures == 0 and i % max(1, round(detector.fps/2)):
                center = np.clip(guidance[0]/SCALE, 0, 1)
            candidates = detector.detect(history, center)
            rescued = False
            if center is not None and guidance and not any(c['score'] >= association for c in candidates):
                # A weak peak near the prediction can lose confidence when the
                # crop removes context. Retry THIS frame before discarding it.
                if any(c['score'] >= .15 and np.linalg.norm(xy(c)-guidance[0]) <= guidance[1] for c in candidates):
                    candidates = detector.detect(history)
                    rescue_calls += 1
                    rescued = True
            original = [c for c in candidates if c.get('evidence') != 'local-maximum']
            row = tracker.update(i, timestamp, original, cut)
            row['search'] = 'full-frame-rescue' if rescued else 'crop' if center is not None else 'full-frame'
            failures = 0 if row['position'] else failures+1
            source_rows.append(dict(frame=i, timestamp=timestamp, candidates=candidates, sceneCut=cut, search=row['search']))
            if i % 25 == 0:
                print(f'{i} frames tracked', flush=True)
        cache = dict(format='pongmetrics-neural-candidates', version=1, video=detector.video,
                     fps=detector.fps, frames=source_rows,
                     model=dict(name='RacketVision TrackNetV3', weightsSha256=detector.weights_hash,
                                backgroundSamples=detector.background_sample_count, candidateFloor=a.candidate_floor, weakPeaks=a.weak_peaks,
                                nativeZoom=a.native_zoom, headAdapterSha256=detector.head_adapter_hash, headAdapterName=adapter_name, strongScore=association,
                                acquisitionScore=tracker.acquisition_score,
                                analyzedStart=a.start, analyzedEnd=a.end if np.isfinite(a.end) else detector.end,
                                temporalOrder='background,BGR[i-3],BGR[i-2],BGR[i-1],BGR[i]; repeat first'),
                     summary=dict(heavyCalls=detector.calls, cropCalls=detector.crop_calls,
                                  fullFrameCalls=detector.full_frame_calls, sameFrameRescueCalls=rescue_calls, startupSeconds=detector.startup_seconds,
                                  inferenceSeconds=detector.inference_seconds, totalSeconds=time.perf_counter()-began))
        rows = tracker.finish()
        if a.weak_peaks:
            for row, source in zip(rows, source_rows):
                row['candidates'] = source['candidates']
            rows = recover_peak_gaps(rows, detector.fps)
        payload = make_payload(cache, rows, time.perf_counter()-began-detector.startup_seconds-detector.inference_seconds)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2), encoding='utf8')
    if a.save_candidates:
        a.save_candidates.parent.mkdir(parents=True, exist_ok=True)
        a.save_candidates.write_text(json.dumps(cache, indent=2), encoding='utf8')
    print(json.dumps(payload['summary'], indent=2))


if __name__ == '__main__':
    main()
