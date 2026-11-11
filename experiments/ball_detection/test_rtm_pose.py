import unittest
import numpy as np

from rtm_pose import average_flipped_simcc, decode_simcc, prepare_box


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

    def test_high_resolution_simcc_uses_model_input_dimensions(self):
        input_size = (288, 384)
        center, scale, matrix = prepare_box([100, 200, 300, 600], input_size)
        np.testing.assert_allclose(matrix @ np.r_[center, 1], [144, 192], atol=1e-5)
        x = np.zeros((1, 17, 576))
        y = np.zeros((1, 17, 768))
        x[:, :, 288] = .8
        y[:, :, 384] = .6
        points, scores = decode_simcc(x, y, center, scale, input_size)
        np.testing.assert_allclose(points, np.tile(center, (17, 1)))
        np.testing.assert_allclose(scores, .6)
        with self.assertRaises(ValueError):
            decode_simcc(x, y, center, scale)

    def test_flip_averaging_restores_left_wrist_and_horizontal_bins(self):
        x = np.zeros((1, 17, 384))
        y = np.zeros((1, 17, 512))
        flipped_x = np.zeros_like(x)
        flipped_y = np.zeros_like(y)
        x[0, 9, 64] = .4
        y[0, 9, 123] = .4
        flipped_x[0, 10, 383 - 64] = .8
        flipped_y[0, 10, 123] = .8
        averaged_x, averaged_y = average_flipped_simcc((x, y), (flipped_x, flipped_y))
        self.assertAlmostEqual(averaged_x[0, 9, 64], .6)
        self.assertAlmostEqual(averaged_y[0, 9, 123], .6)
        self.assertEqual(np.count_nonzero(averaged_x), 1)
        self.assertEqual(np.count_nonzero(averaged_y), 1)
