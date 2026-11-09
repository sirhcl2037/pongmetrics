import unittest

from evaluate_manual_labels import evaluate


class ManualEvaluationTests(unittest.TestCase):
    def test_offset_excludes_unprocessed_and_ambiguous_labels(self):
        video = dict(name='clip', size=10, width=1280, height=720)
        detections = dict(video=video, fps=25,
                          trackingFrames=[dict(timestamp=0)],
                          ballObservations=[dict(timestamp=0, position=dict(value=dict(x=.5, y=.5)))])
        labels = dict(video=video, fps=25, labels=[
            dict(timestamp=.04, visibility='visible', position=dict(value=dict(x=.5, y=.5))),
            dict(timestamp=.06, visibility='visible', position=dict(value=dict(x=.5, y=.5))),
            dict(timestamp=.08, visibility='out-of-frame', position=None)])
        result = evaluate(detections, labels, -1)
        self.assertEqual(result['withinPixels']['20'], 1)
        self.assertEqual(result['offGridLabelsExcluded'], 1)
        self.assertEqual(result['unprocessedLabelsExcluded'], 1)
        self.assertEqual(result['negativeFrames'], 0)

    def test_false_positive_and_localization_error_use_original_pixels(self):
        video = dict(name='clip', size=10, width=1280, height=720)
        detections = dict(video=video, fps=25, ballObservations=[
            dict(timestamp=0, position=dict(value=dict(x=.51, y=.5))),
            dict(timestamp=.04, position=dict(value=dict(x=.5, y=.5)))])
        labels = dict(video=video, fps=25, labels=[
            dict(timestamp=0, visibility='visible', position=dict(value=dict(x=.5, y=.5))),
            dict(timestamp=.04, visibility='occluded', position=None)])
        result = evaluate(detections, labels, 0)
        self.assertEqual(result['withinPixels']['10'], 0)
        self.assertEqual(result['withinPixels']['20'], 1)
        self.assertAlmostEqual(result['medianErrorPixels'], 12.8)
        self.assertEqual(result['negativeFalsePositives'], 1)
        self.assertEqual(result['referenceFailures'], [dict(frame=1, reason='false-positive')])

    def test_reference_tolerance_scales_with_video_resolution(self):
        video = dict(name='clip',size=10,width=3840,height=2160)
        detections = dict(video=video,fps=50,ballObservations=[
            dict(timestamp=0,position=dict(value=dict(x=.51,y=.5)))])
        labels = dict(video=video,fps=50,labels=[
            dict(timestamp=0,visibility='visible',position=dict(value=dict(x=.5,y=.5)))])
        result = evaluate(detections,labels,0)
        self.assertEqual(result['withinPixels']['20'],0)
        self.assertEqual(result['withinReferencePixels']['20'],1)
        self.assertEqual(result['referenceRecall'],1)
        self.assertEqual(result['referencePrecision'],1)


if __name__ == '__main__':
    unittest.main()
