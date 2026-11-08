import copy
import unittest

import numpy as np

from neural_candidates import extract_weak_peaks
from temporal_track import recover_peak_gaps


class PeakRecoveryTests(unittest.TestCase):
    def rows(self):
        return [dict(timestamp=i/25, position=({'x':x, 'y':.5} if i != 1 else None),
                     confidence=.9 if i != 1 else 0, sceneCut=False, trackId=1,
                     candidates=[]) for i, x in enumerate([.3, .31, .32])]

    def test_empty_frame_remains_missing(self):
        rows = self.rows()
        recover_peak_gaps(rows, 25)
        self.assertIsNone(rows[1]['position'])

    def test_recovers_actual_peak_and_stops_at_scene_cut(self):
        rows = self.rows()
        peak = dict(x=.31, y=.5, score=.1, evidence='local-maximum')
        rows[1]['candidates'] = [peak]
        cut = copy.deepcopy(rows)
        cut[1]['sceneCut'] = True
        recover_peak_gaps(rows, 25)
        self.assertEqual(rows[1]['position'], dict(x=.31, y=.5))
        recover_peak_gaps(cut, 25)
        self.assertIsNone(cut[1]['position'])

    def test_two_competing_peaks_remain_ambiguous(self):
        rows = self.rows()
        rows[1]['candidates'] = [dict(x=x, y=.5, score=.1, evidence='local-maximum') for x in [.31, .311]]
        recover_peak_gaps(rows, 25)
        self.assertIsNone(rows[1]['position'])

    def test_local_maxima_do_not_use_diffuse_blob_centroid(self):
        heatmap = np.full((288, 512), .02, dtype=np.float32)
        heatmap[100, 120] = .08
        heatmap[100, 180] = .12
        peaks = extract_weak_peaks(heatmap, floor=.03)
        self.assertEqual(len(peaks), 2)
        self.assertEqual(peaks[0]['x'], 180/512)
        self.assertEqual(peaks[1]['x'], 120/512)


if __name__ == '__main__':
    unittest.main()
