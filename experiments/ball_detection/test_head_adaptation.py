import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import contextlib
import io

import numpy as np
import torch

from adapt_ball_head import sample_features, train
from head_adapter import apply_head_adapter, model_settings
from temporal_track import BallTracker


class HeadAdaptationTests(unittest.TestCase):
    def test_default_adapter_configuration_and_explicit_baseline(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory)/'default.json'
            path.write_text(json.dumps(dict(name='test-head',tracking=dict(associationScore=.4,acquisitionScore=.6))))
            selected, association, acquisition, name = model_settings(default_path=path)
            self.assertEqual((selected,association,acquisition,name),(path,.4,.6,'test-head'))
            self.assertEqual(model_settings(base_model=True,default_path=path),(None,.5,.5,None))
            with self.assertRaises(ValueError):
                model_settings(default_path=path,strong_score=.9)

    def test_restart_confirmation_is_stricter_than_existing_track_association(self):
        def point(x,score):
            return dict(x=x,y=.5,score=score)
        tracker = BallTracker(25,strong_score=.5,acquisition_score=.6)
        tracker.update(0,0,[point(.3,.9)])
        tracker.update(1,.04,[point(.31,.9)])
        row = tracker.update(2,.08,[point(.32,.55)])
        self.assertIsNotNone(row['position'])
        for frame in range(3,18):
            tracker.update(frame,frame/25,[])
        for frame,score in zip(range(18,21),[.51,.56,.74]):
            tracker.update(frame,frame/25,[point(.7,score)])
        self.assertTrue(all(row['position'] is None for row in tracker.rows[18:21]))

    def test_invalid_residual_is_rejected_without_mutating_base_head(self):
        class Model:
            predictor = torch.nn.Conv2d(3,4,1)
        model = Model()
        original = model.predictor.weight.detach().clone()
        data = dict(format='pongmetrics-residual-head-adapter',version=1,baseWeightsSha256='base',weights=[1,2,3],bias=.4,
                    hiddenWeights=[[1,2]],hiddenBias=[0],outputWeights=[0],outputBias=0,residualCap=3)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory)/'adapter.json';path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                apply_head_adapter(model,path,'base')
        torch.testing.assert_close(model.predictor.weight,original)

    def test_training_applies_requested_mined_negative_weight(self):
        rng = np.random.default_rng(1)
        features = rng.normal(size=(32,3)).astype(np.float32)
        metadata = dict(baseWeightsSha256='base', hardNegativeMining=dict(vectors=4,weight=4))
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            source = Path(directory)/'features.npz'
            output = Path(directory)/'adapter.json'
            np.savez(source,features=features,targets=(features[:,0]>0).astype(np.float32),weights=np.ones(32,dtype=np.float32),
                     baseWeight=np.zeros(3,dtype=np.float32),baseBias=np.array(0,dtype=np.float32),metadata=json.dumps(metadata))
            with contextlib.redirect_stdout(io.StringIO()):
                train(SimpleNamespace(features=source,output=output,anchor=.05,mined_negative_weight=.5))
            self.assertEqual(json.loads(output.read_text())['training']['hardNegativeMining']['weight'],.5)

    def test_uncertain_labels_are_excluded_and_negatives_have_zero_targets(self):
        features = np.ones((4, 24, 32), dtype=np.float32)
        heatmap = np.zeros((24, 32), dtype=np.float32)
        self.assertIsNone(sample_features(features, heatmap, {'visibility':'uncertain'}, [0,0], np.random.default_rng(1)))
        result = sample_features(features, heatmap, {'visibility':'occluded'}, [0,0], np.random.default_rng(1))
        self.assertEqual(result[0].shape[1],4)
        self.assertTrue(np.all(result[1] == 0))

    def test_click_uncertainty_uses_real_local_peak_and_background_stays_separate(self):
        features = np.zeros((2, 40, 50), dtype=np.float32)
        features[0, :, :] = np.arange(50)
        features[1, :, :] = np.arange(40)[:, None]
        heatmap = np.zeros((40, 50), dtype=np.float32)
        heatmap[20, 22] = .1
        label = {'visibility':'visible','position':{'value':{'x':20/512,'y':20/288}}}
        x, target, _ = sample_features(features, heatmap, label, [0,0], np.random.default_rng(1))
        np.testing.assert_array_equal(x[np.argmax(target)], [22,20])
        self.assertTrue(np.all(np.sum((x[target==0]-[20,20])**2,axis=1)>12**2))

    def test_adapter_only_changes_final_output_channel_and_checks_checkpoint(self):
        class Model:
            predictor = torch.nn.Conv2d(3,4,1)
        model = Model()
        original = model.predictor.weight.detach().clone()
        data = dict(format='pongmetrics-linear-head-adapter',version=1,baseWeightsSha256='base',weights=[1,2,3],bias=.4)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory)/'adapter.json'
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                apply_head_adapter(model,path,'different')
            apply_head_adapter(model,path,'base')
        torch.testing.assert_close(model.predictor.weight[:-1],original[:-1])
        torch.testing.assert_close(model.predictor.weight[-1,:,0,0],torch.tensor([1.,2.,3.]))


if __name__ == '__main__':
    unittest.main()
