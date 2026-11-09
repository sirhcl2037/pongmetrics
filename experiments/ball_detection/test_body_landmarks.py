import unittest

from player_track import normalized_landmarks


class BodyLandmarksTest(unittest.TestCase):
    def test_normalization_and_missing_joints(self):
        points = [[640,360,.8] for _ in range(17)]
        points[0] = [0,0,0]
        points[1] = [1400,1,.9]
        points[2] = [float('nan'),1,.9]
        joints = normalized_landmarks(points,1280,720)
        self.assertEqual(len(joints),17)
        self.assertTrue(all(joints[i]['position'] is None for i in range(3)))
        self.assertEqual(joints[9]['name'],'left-wrist')
        self.assertEqual(joints[9]['position']['value'],{'x':.5,'y':.5})
        self.assertEqual(joints[9]['position']['source'],'automatic')
        self.assertEqual(joints[9]['position']['confidence'],.8)

    def test_wrong_skeleton_rejected(self):
        with self.assertRaises(ValueError):
            normalized_landmarks([[1,1,1]],1280,720)
