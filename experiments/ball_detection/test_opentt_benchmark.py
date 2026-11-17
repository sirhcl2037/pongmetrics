import unittest

from opentt_benchmark import evaluate


class OpenTTBenchmarkTest(unittest.TestCase):
    def test_sparse_labels_absence_and_source_frame_offset(self):
        data = {'fps':120, 'video':{'width':1920, 'height':1080},
                'trackingFrames':[{'timestamp':i/120} for i in range(5)],
                'ballObservations':[{'timestamp':i/120, 'position':{'value':{'x':.5, 'y':.5}}}
                                    for i in (0, 1, 2, 3)]}
        labels = {100:{'x':960, 'y':540}, 102:{'x':-1, 'y':-1},
                  103:{'x':-1, 'y':540}, 104:{'x':960, 'y':540}, 105:{'x':960, 'y':540}}
        result = evaluate(data, labels, {102:'bounce', 105:'net'}, 100, 106, 120)
        self.assertEqual(result['visibleLabels'], 2)
        self.assertEqual(result['missing'], 1)
        self.assertEqual(result['unlabelledFrames'], 1)
        self.assertEqual(result['predictionsOnAbsence'], 1)
        self.assertEqual(result['ambiguousLabelsExcluded'], 1)
        self.assertEqual(result['unprocessedLabels'], 1)
        self.assertEqual(result['withinReferencePixels']['20'], 1)
        self.assertEqual(result['events'][0]['timestamp'], 2/120)
        self.assertEqual(labels[103], {'x':-1, 'y':540})

    def test_wrong_fps_rejected(self):
        with self.assertRaises(ValueError):
            evaluate({'fps':30}, {}, {}, 0, 10, 120)
