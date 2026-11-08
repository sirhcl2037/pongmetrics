import unittest
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from neural_candidates import extract_candidates, temporal_indices, StreamingDetector
from temporal_track import BallTracker, replay
from evaluate_tracking import evaluate, baseline, agreement


def candidate(x, y=.5, score=.9):
    return dict(x=x, y=y, score=score, threshold=.5 if score >= .5 else .15, area=10)


def track(sequence, times=None, cuts=()):
    tracker = BallTracker(25)
    for i, candidates in enumerate(sequence):
        tracker.update(i, times[i] if times else i/25, candidates, i in cuts)
    return tracker, tracker.finish()


class TemporalTrackingTests(unittest.TestCase):
    def test_occlusion_never_emits_prediction_as_observation(self):
        _, rows = track([[candidate(.3)], [candidate(.31)]]+[[]]*25)
        self.assertIsNotNone(rows[0]['position'])
        self.assertIsNotNone(rows[2]['prediction'])
        self.assertTrue(all(r['position'] is None for r in rows[2:]))
        self.assertIsNone(rows[-1]['prediction'])

    def test_weak_evidence_requires_two_sided_support(self):
        _, rows = track([[candidate(.3)], [candidate(.31)], [candidate(.32, score=.3)], [candidate(.33)]])
        self.assertEqual(rows[2]['reason'], 'two-sided-weak-evidence')
        self.assertEqual(rows[2]['position']['x'], .32)
        _, trailing = track([[candidate(.3)], [candidate(.31)], [candidate(.32, score=.3)]])
        self.assertIsNone(trailing[2]['position'])

    def test_backward_recovery_uses_two_future_strong_frames(self):
        _, rows = track([[candidate(.3, score=.3)], [candidate(.31)], [candidate(.32)]])
        self.assertEqual(rows[0]['reason'], 'backward-confirmed-weak-evidence')
        _, single = track([[candidate(.3, score=.3)], [candidate(.31)]])
        self.assertIsNone(single[0]['position'])

    def test_weak_distractor_and_ambiguous_peaks_are_rejected(self):
        for proposed in [[candidate(.7, score=.3)], [candidate(.32, score=.3), candidate(.321, score=.35)]]:
            _, rows = track([[candidate(.3)], [candidate(.31)], proposed, [candidate(.33)]])
            self.assertIsNone(rows[2]['position'])

    def test_global_reacquisition_resets_velocity_after_long_loss(self):
        tracker, rows = track([[candidate(.3)], [candidate(.31)]]+[[]]*15+[[candidate(.8)], [candidate(.79)]])
        self.assertEqual(rows[-2]['reason'], 'reacquired-tracklet')
        self.assertEqual(rows[-1]['trackId'], 2)
        self.assertLess(tracker.velocity[0], 0)

    def test_isolated_far_false_positive_does_not_move_track(self):
        _, rows = track([[candidate(.3)], [candidate(.31)], [candidate(.9)], [candidate(.33)]])
        self.assertIsNone(rows[2]['position'])
        self.assertEqual(rows[3]['position']['x'], .33)

    def test_motion_association_chooses_ball_over_higher_score_distractor(self):
        _, rows = track([[candidate(.3)], [candidate(.31)], [candidate(.32, score=.6), candidate(.9, score=.99)]])
        self.assertEqual(rows[2]['position']['x'], .32)

    def test_contact_reversal_does_not_permanently_extrapolate_away(self):
        tracker, rows = track([[candidate(x)] for x in [.3, .33, .36, .33, .30, .27]])
        self.assertTrue(all(r['position'] is not None for r in rows))
        self.assertLess(tracker.velocity[0], 0)

    def test_cuts_and_discontinuities_do_not_bridge_weak_evidence(self):
        for times, cuts in [(None, (2,)), ([0, .04, 2., 2.04], ())]:
            _, rows = track([[candidate(.3)], [candidate(.31)], [candidate(.32, score=.3)], [candidate(.33)]], times, cuts)
            self.assertIsNone(rows[2]['position'])
            self.assertIsNone(rows[3]['position'])

    def test_duplicate_and_invalid_timestamps_rejected(self):
        with self.assertRaises(ValueError):
            track([[], []], [0, 0])
        with self.assertRaises(ValueError):
            track([[candidate(float('nan'))]])

    def test_heatmap_components_keep_current_frame_centroid_and_blur(self):
        hm = np.zeros((288, 512), dtype=np.float32)
        hm[50:52, 100:120] = .3
        hm[90:93, 200:203] = .9
        candidates = extract_candidates(hm)
        self.assertEqual(len(candidates), 2)
        self.assertAlmostEqual(candidates[0]['x'], 201/512)
        self.assertAlmostEqual(candidates[1]['x'], 109.5/512)
        self.assertEqual(temporal_indices(0), [0, 0, 0, 0])
        self.assertEqual(temporal_indices(5), [2, 3, 4, 5])

    def test_decoder_history_is_bounded_and_current_frame_included(self):
        class Capture:
            def __init__(self): self.i = 0; self.released = False
            def read(self):
                if self.i == 100: return False, None
                image = np.full((12, 20, 3), self.i, dtype=np.uint8)
                self.i += 1
                return True, image
            def get(self, _): return (self.i-1)*40
            def release(self): self.released = True
        detector = object.__new__(StreamingDetector)
        detector.video_path, detector.end = 'fake', 4.
        cap = Capture()
        with patch('neural_candidates.cv2.VideoCapture', return_value=cap):
            for i, timestamp, history, _ in detector.frames():
                self.assertEqual(len(history), 4)
                self.assertEqual([int(im[0, 0, 0]) for im in history], temporal_indices(i))
                self.assertAlmostEqual(timestamp, i*.04)
        self.assertTrue(cap.released)

    def test_evaluation_counts_wrong_locations_and_unlabeled_frames_honestly(self):
        video = dict(name='fake')
        data = dict(video=video, fps=25, ballObservations=[dict(timestamp=i/25, position=dict(value=dict(x=x, y=.5))) for i,x in [(0,.3),(1,.9),(3,.5),(4,.5)]])
        labels = dict(video=video, fps=25, frames=[dict(frame=i,status='visible',position=dict(x=.3,y=.5)) for i in range(3)]+[dict(frame=3,status='absent'),dict(frame=4,status='uncertain')])
        metrics = evaluate(data, labels)
        self.assertEqual((metrics['truePositives'],metrics['falsePositives'],metrics['falseNegatives']), (1,2,2))
        self.assertAlmostEqual(metrics['precision'], 1/3)
        self.assertAlmostEqual(metrics['recall'], 1/3)

    def test_replay_long_duration_reacquires_late_without_call_budget(self):
        frames = [dict(frame=i,timestamp=i/25,candidates=[]) for i in range(20000)]
        for i in [0,1,19998,19999]: frames[i]['candidates'] = [candidate(.3)]
        cache = dict(format='pongmetrics-neural-candidates', fps=25,video={},frames=frames)
        result = replay(cache)
        self.assertEqual(len(result['ballObservations']), 4)
        self.assertEqual(result['summary']['tracklets'], 2)
        self.assertEqual(len(baseline(cache)['ballObservations']), 4)

    def test_recorded_rally_retains_baseline_and_recovers_labeled_misses(self):
        fixtures = Path(__file__).parent/'fixtures'
        cache = json.loads((fixtures/'rally-candidates.json').read_text(encoding='utf8'))
        labels = json.loads((fixtures/'rally-labels.json').read_text(encoding='utf8'))
        result = replay(cache)
        self.assertEqual(agreement(baseline(cache), result)['matched'], 141)
        self.assertEqual(len(result['ballObservations']), 152)
        self.assertEqual(evaluate(baseline(cache), labels)['truePositives'], 10)
        self.assertEqual(evaluate(result, labels)['truePositives'], 12)


if __name__ == '__main__':
    unittest.main()
