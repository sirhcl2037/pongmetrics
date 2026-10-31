# Offline ball detection experiment

## Fast tracking prototype

`fast_track.py` is a separate, experimental color/motion tracker using OpenCV and NumPy. The initial clear seed classifies white-ball appearance once; thresholds are locked for the video and stored in the output. Orange seeds are unsupported. Supply a clear ball seed as seconds and normalized x/y:

```powershell
experiments/ball_detection/.venv/Scripts/python.exe experiments/ball_detection/fast_track.py 'path/to/clip.mp4' --seed 2.0 .5308287 .4364107
```

It searches near a constant-velocity prediction for small moving white components scored by proximity, saturation, local brightness contrast, and motion, rejects ambiguous candidates, and stores one record per frame. Records distinguish `seed`, `detected`, `predicted`, and `lost`; match scores are heuristic, not calibrated correctness. After five missed frames, `needsDetector` signals that a heavy detector should reinitialize position and velocity. With --heavy, real TrackNet inference runs whenever the cheap match is uncertain and at least every --verify-every frames (default 5). It resets motion from recovered observations. Use --verify-every 1 for heavy verification of every frame after the seed. Strong visual candidates in heavy-detector gaps are retained as recoveryCandidate with reviewRequired, separately from predictions and accepted observations. Its experimental JSON is deliberately separate from the app's detection import so predictions cannot appear as confirmed observations. Working coordinates use a 640x360 image; estimates can move outside the image during loss.

On the existing validation clip, seeded from the saved TrackNet position at 2.0 seconds and ending at 8.28, a quick run processed 157 frames in 0.558 seconds including decoding and startup work (0.210 seconds inside tracking/preprocessing). It accepted 10 visual candidates, predicted 40 frames, and marked 106 lost; the remaining frame was the seed. This demonstrates cheap computation, **not useful tracking accuracy**. Candidate acceptance is not ground truth. Next work: visually label representative frames, tune color/blur candidate extraction and direction-change handling, then connect sparse detector recovery. No speed comparison with neural inference is an accuracy-equivalent comparison.

The browser stays a local video viewer. Python runs a real pretrained [RacketVision TrackNetV3](https://github.com/OrcustD/RacketVision) model; it writes JSON that PongMetrics imports. It is not a backend dependency. Model/source downloads and the Python environment are ignored by Git.

## Reproduce (Python 3.12, PowerShell, repository root)

Hybrid recovery: add `--heavy` to the fast tracker command. Compare its experimental output with `python experiments/ball_detection/compare_track.py experiments/ball_detection/output/detections.json experiments/ball_detection/output/hybrid-track.json` (set `--output` to that hybrid path). The detector uses the fixed segment's median background as in the original experiment, so this remains offline processing.

Hybrid check, 2.0–8.28 seconds: 157 frames, 119 heavy calls instead of 156 after the seed, 84.95 seconds including detector initialization/background preparation. It retained all 106 baseline detections after the seed, with a maximum coordinate difference of 2.89 pixels at 640x360. The extra cheap candidate at 6.84 seconds lands near the far player's face and is unverified; it must not be counted as improved recall. Cheap detections carry `reviewRequired`. No matched all-heavy timing run or labeled accuracy benchmark was performed. More true ball observations, especially in existing heavy-detector gaps, remain the goal; extra candidates and predicted positions do not establish that goal.

```powershell
python -m venv experiments/ball_detection/.venv
experiments/ball_detection/.venv/Scripts/python.exe -m pip install -r experiments/ball_detection/requirements.txt
experiments/ball_detection/.venv/Scripts/python.exe experiments/ball_detection/setup.py
experiments/ball_detection/.venv/Scripts/python.exe experiments/ball_detection/detect.py 'path/to/clip.mp4' --end 8.28
experiments/ball_detection/.venv/Scripts/python.exe experiments/ball_detection/render_review.py 'path/to/clip.mp4'
```

`--end` limits analysis/background estimation to a fixed-camera rally before the test clip's close-up cut. Change it for other clips. Input is kept locally. The script stores the checkpoint SHA-256 in its output. The source is pinned; the publisher's checkpoint URL uses main, so compare that hash when reproducing an existing run.

See [VALIDATION.md](VALIDATION.md) for the actual run and limitations. `render_review.py` creates a silent GIF directly from the saved audit and decoded video for visual review without repeating inference.

## Data and timing

Input is BGR 512×288, median background plus four frames (15 channels). Training uses the current frame and three preceding frames, repeating the first frame at the start. We follow that loader, rather than the repository inference helper's exclusive range ending one frame earlier. Model outputs are decoded as thresholded connected components. Exactly one candidate is accepted; no candidates or multiple candidates yield a gap. Scores are heatmap peaks, **not calibrated probabilities of correctness**. No interpolation, tracking heuristics, trajectory prediction, or InpaintNet fills gaps.

`output/detections.json` contains normalized image positions using `Observation<Point>`, actual decoder timestamps in seconds, FPS, matching video metadata, and model provenance. `audit.json` includes every analyzed frame and the reason for missing observations. Review sheets overlay sampled detections on the decoded video. Sampled sheets support qualitative inspection; they do not establish dataset precision/recall. Unreviewed frames are not ground truth.

Load the original clip in PongMetrics, choose **Import ball detections**, and select the JSON. The dot updates on decoded video frame callbacks. A short trail joins only consecutive confident detections, and vanishes at a missing frame. Raise **Min score** to hide weak detections; lowering it cannot recover candidates already rejected by Python. Normal session export/import retains these observations and their FPS. Nothing converts these image positions to 3D.

## ONNX investigation

The neural network is standard convolution/BatchNorm/ReLU/MaxPool/nearest Resize/Concat/Sigmoid, with input `[1,15,288,512]` and final heatmap `[1,1,288,512]`. A small wrapper around `_forward(x)[:, -1:]` can avoid MMEngine's training interface during export. The PyTorch state file is about 46.6 MB; browser activation memory and latency still need measurement.

After accuracy is accepted, export with [torch.onnx.export](https://docs.pytorch.org/docs/stable/onnx.html), compare heatmaps and decoded centers against PyTorch on these same frames, then benchmark [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) WebGPU and WASM. Browser preprocessing must reproduce BGR ordering, background and temporal alignment. Operator support is provider-specific. Do not change precision/quantize until position and gap parity have been measured. No ONNX runtime, model download, or inference workload has been added to the production bundle yet.

White-only revision: synthetic checks passed for moving white selection, orange rejection, and stationary rejection. On the same 157-frame segment it accepted one candidate, predicted five frames, then marked 150 lost (0.310 seconds including decoding). This conservative result is not an accuracy improvement. After loss it waits for detector recovery instead of choosing arbitrary candidates from a growing region; this earlier result used color-only mode.
