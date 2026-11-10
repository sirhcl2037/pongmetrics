import unittest
import numpy as np

from rtm_pose import prepare_box,decode_simcc


class RTMPoseGeometryTest(unittest.TestCase):
    def test_affine_center_and_inverse_coordinates(self):
        center,scale,matrix=prepare_box([100,200,300,600])
        np.testing.assert_allclose(matrix@np.r_[center,1],[96,128],atol=1e-5)
        self.assertAlmostEqual(scale[0]/scale[1],192/256)
        x=np.zeros((1,17,384));y=np.zeros((1,17,512))
        x[:,:,192]=.8;y[:,:,256]=.6
        points,scores=decode_simcc(x,y,center,scale)
        np.testing.assert_allclose(points,np.tile(center,(17,1)))
        np.testing.assert_allclose(scores,.6)
        x[:,:,192]=0
        self.assertTrue(np.isnan(decode_simcc(x,y,center,scale)[0]).all())

    def test_bad_box_and_skeleton_rejected(self):
        for box in [[0,0,0,2],[1,1,0,2],[0,float('nan'),2,3]]:
            with self.assertRaises(ValueError): prepare_box(box)
        with self.assertRaises(ValueError):
            decode_simcc(np.zeros((1,16,384)),np.zeros((1,16,512)),[0,0],[1,1])
