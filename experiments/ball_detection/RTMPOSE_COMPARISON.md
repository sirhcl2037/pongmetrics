# RTMPose optional pipeline — 2026-11-09

A follow-up compares a higher-resolution model and a mirrored-crop pass on matched source boxes. See [POSE_REFINEMENT_TRIAL.md](POSE_REFINEMENT_TRIAL.md).

## Result and usage

An additional **Generate body landmarks · RTMPose** button now runs YOLO11n person tracking and then RTMPose-m on original-source player regions. It uses an isolated CPU ONNX environment. Existing YOLO pose generation remains available. Both formats import through **Import player tracks**; player names still use Locate clicks. The static production app only displays imports; local generation remains a Vite development/preview capability.

The comparison shows a useful qualitative improvement on some difficult arms, especially the near player's swinging arm at 3.00 s in the 720p clip and the raised arm at 6.30 s in the 4K clip. It does not establish state-of-the-art performance or a measured joint-accuracy gain. At 3.00 s the improved wrist remains inward of some blurred hand pixels. Hidden anatomy remains inferred and can be wrong. Pose quality alone does not establish correctness of technique or power generation.

RTMPose's raw SimCC strengths differ from YOLO keypoint scores. They are stored as `rawScore`, with a clipped [0,1] value in `position.confidence` for existing schema compatibility. Neither is a visibility probability. The common .35 drawing gate is a display heuristic, not a calibrated cross-model accuracy threshold. Missing/out-of-image or nonpositive-score joints remain null; joints are not interpolated or temporally smoothed.

## Experiment

Used the official OpenMMLab RTMPose-m body7 256×192 ONNX export. Pre/postprocessing follows the [official ONNX example](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose/examples/onnxruntime): source-image affine person crop, aspect-preserving 1.25 box padding, BGR mean/std normalization, SimCC argmax divided by two, then inverse scaling back to source coordinates. Our implementation only needs NumPy, OpenCV, and ONNX Runtime; no MMPose/MMCV dependency changes are made to the working ball environment.

Model SHA256: `5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c`. Model provenance stays in exported JSON alongside the original detector metadata. Saved tracking IDs and timestamps are preserved during refinement. This is pose refinement, not cross-cut re-identification.

- 720p: 12 selected frames across the four previously audited contact neighborhoods, two players per frame, 24 calls. Pose calls: .415 s, startup .661 s, total 1.615 s.
- 4K: five previously compared frames at 6.10/6.30/6.50/6.70/6.90 s, two players per frame, ten calls. Pose calls: .246 s, startup .387 s, total 5.057 s including sequential 4K decoding to the sample.
- Full 720p refinement: 219 frames and 426 saved people, 9.14 s pose calls / 10.37 s total. This excludes original person/YOLO-pose generation. Some runtime jobs overlapped, so this is not a controlled throughput benchmark.
- Full 4K person detection: 764 decoded frames, 19.26 s detector/tracker / 30.39 s total. Header originally reported 550 frames; corrected decoded duration is retained. Extra people and track changes exist; initial near/far labels are not verified identity.
- Full 4K RTMPose refinement: 764 frames, 1,529 saved people, 35.20 s pose calls / 45.14 s total including source decoding. Together with the preceding person pass this is roughly 75.5 s in these runs. This is offline processing, not established real-time performance.
- A three-frame upload exercised the new optional endpoint: HTTP 200, three frames, six person observations with 17 joints each, RTMPose model metadata. Missing-Origin request returned 403. Test server closed afterward.
- Replacing full 720p YOLO pose with RTMPose did not change the seven existing player-axis shot candidates on this development rally. Better-looking arms have not yet improved measured shot coverage; contact-label ground truth is still needed.

For full-rally output timings, see the JSON summaries, not paper FPS. The baseline comparisons change pose model and person-region input at the same time; they evaluate a practical pipeline swap, not an isolated architecture ablation. Human landmark ground truth is not available, so numerical joint error is not reported. Existing manually labelled ball data is unchanged.

## Setup

The current machine is set up. To reproduce from the repository root:

```powershell
experiments/ball_detection/.venv/Scripts/python.exe -m venv experiments/ball_detection/output/pose-env
experiments/ball_detection/output/pose-env/Scripts/python.exe -m pip install -r experiments/ball_detection/requirements-pose.txt
```

Download the model ZIP from [OpenMMLab](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip), save its single `end2end.onnx` as `output/models/rtmpose-m.onnx`, and verify the SHA above. Weights, environments and generated data remain ignored by Git. Optional `PONGMETRICS_POSE_PYTHON` overrides the pose worker executable; existing `PONGMETRICS_PYTHON` still controls person/ball generation.

Refine existing tracks without repeating person detection:

```powershell
experiments/ball_detection/output/pose-env/Scripts/python.exe experiments/ball_detection/refine_pose.py 'path/to/video.mp4' 'path/to/player-tracks.json' --output experiments/ball_detection/output/refined-player-tracks.json
```

Add `--times ... --track-ids ... --comparison-dir ...` for a sparse diagnostic run. Sparse outputs are only visible at sampled frames. The full 720p and 4K outputs are `output/body-landmarks-rtmpose-720p.json` and `output/body-landmarks-rtmpose-4k.json`. Comparison sheets are in `output/rtmpose-720p-comparison/` and `output/rtmpose-4k-comparison/` (left YOLO full frame, right RTMPose source region).

Unit tests verify affine center mapping, aspect ratio, inverse SimCC coordinates, invalid/missing joints, and malformed inputs. The existing frontend import parser also accepts these real outputs. No automatic coaching claims or forced joint visibility classifications are added.
