import hashlib
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from onnx_ball import FORMAT, load_session
from verify_ball_backend import compare
from neural_candidates import sample_background
import numpy as np


class OnnxBundleTest(unittest.TestCase):
    def test_output_gate_rejects_changed_centers_settings_and_score_thresholds(self):
        reference = {'video':{'name':'test.mp4'}, 'fps':25,
                     'ballObservations':[{'timestamp':.04,'position':{'value':{'x':.2,'y':.3},'confidence':.5}}],
                     'trackingFrames':[{'frame':1,'timestamp':.04,'position':{'x':.2,'y':.3}}]}
        updated = copy.deepcopy(reference)
        updated['ballObservations'][0]['position']['confidence'] += 1e-7
        self.assertEqual(compare(reference, updated)['observations'], 1)
        updated['ballObservations'][0]['position']['value']['x'] += .001
        with self.assertRaises(ValueError): compare(reference, updated)
        updated = copy.deepcopy(reference)
        updated['ballObservations'][0]['position']['confidence'] -= 1e-7
        with self.assertRaises(ValueError): compare(reference, updated)
        updated = copy.deepcopy(reference)
        updated['model'] = {'candidateFloor':.01}
        with self.assertRaises(ValueError): compare(reference, updated)

    def test_auto_requires_matching_runtime_threads_and_graph(self):
        runtime = SimpleNamespace(__version__='test-runtime', SessionOptions=Mock(), InferenceSession=Mock())
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, onnxruntime=runtime):
            graph = Path(folder) / 'model.onnx'
            graph.write_bytes(b'exported graph')
            metadata = graph.with_suffix('.json')
            manifest = {'format':FORMAT, 'weightsSha256':'checkpoint', 'headAdapterSha256':'head',
                        'onnxSha256':hashlib.sha256(graph.read_bytes()).hexdigest()}
            metadata.write_text(json.dumps(manifest), encoding='utf-8')
            self.assertIsNone(load_session(graph, 'checkpoint', 'head', 8, require_preference=True))
            runtime.InferenceSession.assert_not_called()
            manifest.update(preferredRuntime='test-runtime', preferredThreads=8)
            metadata.write_text(json.dumps(manifest), encoding='utf-8')
            self.assertIsNone(load_session(graph, 'checkpoint', 'head', 4, require_preference=True))
            self.assertIs(load_session(graph, 'checkpoint', 'head', 8, require_preference=True),
                          runtime.InferenceSession.return_value)
            with self.assertRaises(ValueError):
                load_session(graph, 'checkpoint', 'another head', 8, require_preference=True)
            graph.write_bytes(b'wrong graph')
            with self.assertRaises(ValueError):
                load_session(graph, 'checkpoint', 'head', 8, require_preference=True)

    def test_background_scan_retrieves_only_selected_frames(self):
        class Capture:
            def __init__(self):
                self.index = -1
                self.retrieved = []
                self.released = False

            def grab(self):
                self.index += 1
                return self.index < 5

            def get(self, _):
                return self.index * 40

            def retrieve(self):
                self.retrieved.append(self.index)
                return True, np.full((12,20,3), self.index, np.uint8)

            def release(self):
                self.released = True

        capture = Capture()
        samples, zoom, timestamp = sample_background(capture, .14, {1,3}, True, scan_to_end=True)
        self.assertEqual(capture.retrieved, [1,3])
        self.assertEqual([int(image[0,0,0]) for image in samples], [1,3])
        self.assertEqual([image.shape for image in zoom], [(576,1024,3)]*2)
        self.assertEqual(timestamp, .16)
        self.assertTrue(capture.released)
