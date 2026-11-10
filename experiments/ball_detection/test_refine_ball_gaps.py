import copy
import unittest

from refine_ball_gaps import refine


class GapRefinementTest(unittest.TestCase):
    def setUp(self):
        self.baseline = {'video':{'name':'rally.mp4'}, 'fps':25, 'model':{}, 'summary':{},
                         'trackingFrames':[], 'ballObservations':[]}
        for frame, position in enumerate([{'x':.2, 'y':.5}, None, {'x':.3, 'y':.5}]):
            row = {'frame':frame, 'timestamp':frame/25, 'position':position,
                   'confidence':.8 if position else 0, 'sceneCut':False, 'trackId':1 if position else None}
            self.baseline['trackingFrames'].append(row)
            if position:
                self.baseline['ballObservations'].append({'timestamp':row['timestamp'],
                    'position':{'value':position, 'source':'automatic', 'confidence':.8}})
        self.probe = {'video':self.baseline['video'], 'fps':25, 'summary':{},
                      'frames':[{'frame':1, 'timestamp':.04, 'forward':[{'x':.254,'y':.5,'score':.13}],
                                 'reverse':[{'x':.256,'y':.5,'score':.66}]}]}

    def test_recovery_uses_actual_peak_and_preserves_original_observations(self):
        original = copy.deepcopy(self.baseline)
        refined = refine(self.baseline, self.probe)
        self.assertEqual(self.baseline, original)
        self.assertEqual(refined['summary']['gapRecoveries'], 1)
        self.assertEqual(refined['ballObservations'][::2], original['ballObservations'])
        self.assertEqual(refined['ballObservations'][1]['position']['value'], {'x':.256,'y':.5})
        self.assertEqual(refined['trackingFrames'][1]['gapEvidence']['direction'], 'reverse')

    def test_disagreement_or_scene_cut_leaves_gap_empty(self):
        self.baseline['trackingFrames'][2]['sceneCut'] = True
        self.assertEqual(refine(self.baseline, self.probe)['summary']['gapRecoveries'], 0)
        self.baseline['trackingFrames'][2]['sceneCut'] = False
        self.probe['frames'][0]['forward'][0]['x'] = .22
        self.assertEqual(refine(self.baseline, self.probe)['summary']['gapRecoveries'], 0)

    def test_one_sided_or_far_peaks_leave_gap_empty(self):
        self.probe['frames'][0]['forward'] = []
        self.assertEqual(refine(self.baseline, self.probe)['summary']['gapRecoveries'], 0)
        self.probe['frames'][0]['forward'] = [{'x':.29,'y':.5,'score':.13}]
        self.probe['frames'][0]['reverse'][0]['x'] = .29
        self.assertEqual(refine(self.baseline, self.probe)['summary']['gapRecoveries'], 0)

    def test_image_boundary_requires_crop_peak_and_two_supporting_passes(self):
        self.baseline['trackingFrames'].pop()
        self.baseline['ballObservations'].pop()
        self.baseline['trackingFrames'][1].update(search='crop', prediction={'x':.998,'y':.5})
        peak = {'x':508/512,'y':.5,'score':.58}
        self.probe['frames'][0].update(uncutCrop=[peak], forward=[{**peak,'score':.15}], reverse=[{**peak,'score':.2}])
        refined = refine(self.baseline, self.probe)
        self.assertEqual(refined['trackingFrames'][1]['reason'], 'confirmed-image-edge-peak')
        self.assertEqual(refined['ballObservations'][-1]['position']['value'], {'x':508/512,'y':.5})
        self.probe['frames'][0]['reverse'] = []
        self.assertEqual(refine(self.baseline, self.probe)['summary']['gapRecoveries'], 0)
