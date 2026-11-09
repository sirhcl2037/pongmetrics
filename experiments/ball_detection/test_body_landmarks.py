import unittest

from player_track import normalized_landmarks
from compare_pose_crops import lift_landmarks, iou
import numpy as np


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

    def test_crop_points_lift_to_original_video_without_resurrecting_missing_joints(self):
        points=np.array([[20,30,.8]]*17)
        points[0]=[0,0,0]
        joints=lift_landmarks(points,100,200,1000,500)
        self.assertIsNone(joints[0]['position'])
        self.assertEqual(joints[9]['position']['value'],{'x':.12,'y':.46})
        self.assertEqual(points[9].tolist(),[20,30,.8])

    def test_crop_person_box_matching(self):
        self.assertEqual(iou(np.array([0,0,10,10]),np.array([0,0,10,10])),1)
        self.assertEqual(iou(np.array([0,0,10,10]),np.array([20,20,30,30])),0)
