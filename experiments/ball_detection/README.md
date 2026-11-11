# Offline ball detection experiment

## Current temporal neural tracker

See [INFERENCE_SPEED.md](INFERENCE_SPEED.md) for the float32 ONNX CPU path, paired speed measurements, and complete-clip output comparisons.

The validated local default now uses the small, data-trained `review-head-v1` adapter with separate follow/restart confidence cutoffs. See [HEAD_ADAPTATION.md](HEAD_ADAPTATION.md) for training provenance, cross-rally development results, rejected trials, and limitations. Use `temporal_track.py --base-model --video ...` to reproduce the original pretrained detector. No extra neural-network layers are enabled in the default.

See [TRACKING.md](TRACKING.md) for the streaming neural tracker, cached regression fixture, evaluation commands, measured results, and limitations. Localhost now has **Generate ball tracks** and a separate prediction/search overlay. The original detector and color experiments below remain available as references; their historical timings and defaults do not describe the new pipeline.

Player detection/tracking is now available as a separate reusable offline pipeline: see [PLAYERS.md](PLAYERS.md) for YOLO11n + ByteTrack setup, cached outputs, review video, tuning, and measured results.

## Fast tracking prototype

### Background-relative color experiment

Latest performance/debug pass: mixture candidates now require at least 3 pixels and are rejected when surrounded by substantial foreground color changes inconsistent with white. This addresses small highlights on foreground objects where the stored background is invalid. Color projection coefficients are cached between heavy calibrations, square roots are evaluated only for surviving components, and mixture tracking skips full-frame HSV/Lab conversion after seed initialization. Exact projection-equivalence checks passed.

The heavy detector stores decoded frames as bytes (one quarter of float32 storage), converts only the requested temporal crops to float32, and estimates its background from at most 24 sampled frames. This changes the background input relative to the original full-frame median; use `--heavy-background-samples 0` to restore that median. It still decodes and retains all frames up to `--end`, so memory/decode cost remains dependent on clip duration.

Real 157-frame mixture run with 20 heavy calls (17 crops, 3 full frames): **18.31 seconds total**, including **4.37 seconds heavy startup**, **0.35 seconds mixture background preparation**, and **13.16 seconds tracking/inference**. It accepted 37 detections, including 24 cheap detections. Of 106 baseline detections, 36 were matched within 3.08 pixels and 70 were missed; one extra detection remains unverified. The preceding version with full-median preparation took 34.36 seconds in this session and produced the same detection counts with nearly identical baseline agreement. These are individual runs, not controlled repeated benchmarks or a guaranteed deadline. Results: `output/mixture-fast-live.json`. The mode remains opt-in because coverage is still inadequate.

After foreground rejection, cached replay recovered 45 baseline detections within 2.53 pixels, with one additional unverified detection, versus the previous 89-pixel drift. Cached full-frame results differ from real crop inference; use the real-run figures above for the end-to-end result. Historical mixture results below describe the earlier implementation.

Add `--background-color` with `--heavy` or `--cached-heavy-json`. This opt-in mode samples up to 12 frames over the requested segment to estimate per-pixel median background and median absolute color variation. Real inference reuses the detector's decoded frames for sampling. This is offline preprocessing and uses future frames; it assumes a fixed camera. Measured preprocessing on the short cached clip was 0.51 seconds, not a guaranteed two-second bound.

Matching uses linearized RGB/BGR, projecting each pixel's change from the background onto the direction toward white. It checks mixture fraction, reconstruction error, background stability, component size, frame-to-frame brightening, and location. Unchanging bright markings have zero mixture fraction; near-white backgrounds and unstable regions are excluded. A background hidden by stationary people can still be estimated incorrectly.

Heavy detections with peak at least `.65` and compatible image evidence provide calibration patches. Peaks are heuristic scores, not probabilities. Bright neutral cores refine the white reference; dim/blurred patches retain the white reference and record mixture evidence. Cheap detections never train the appearance model. `backgroundColorModel` in the output stores sample patches, background patches, color-change vectors, mixture fractions, bounds, and the learned white reference. Classification stays white. Seed position/time remain required; without a heavy source the existing color tracker is used.

Cached 157-frame replay: 20 lookups, 12 appearance samples, 26 cheap detections, 40 accepted detections, 1.71 seconds total. Of these, 35 occurred at baseline detection timestamps, but maximum position disagreement was **89.35 pixels**, and 5 occurred in baseline gaps. Timestamp overlap is not correct recovery. This drift is why the mode is optional and the prior tracker remains the default. No real-inference timing or labeled accuracy improvement has been established for this mode. Mixture math and unchanged-background checks passed. Replay output: `output/mixture-trial.json`.

Optional contrast-first experiment: add `--contrast-components` to form connected regions from local brightness contrast before checking their overall color, motion, and shape. Regions can be compact or elongated (area up to 120 pixels, extent up to 32 pixels), with covariance-based width/length limits and a minimum bounding-box occupancy. At least half the region must satisfy the brightness/saturation evidence, and the existing 70% Lab agreement gate remains. This is deliberately opt-in: on the 157-frame cached replay with 20 heavy lookups it recovered 23/106 baseline detections with 8 cheap detections and no extra detections, versus 29/106 with 19 cheap detections and 5 unverified extras for the default. Agreement is not ground-truth accuracy. Cached replay took 0.86 seconds; this does not measure real inference runtime. Results are in `output/contrast-components-trial.json`. The result does not justify switching the default.

Heavy recovery now defaults to a 256x144 crop in the model's 512x288 image space, centered on the predicted ball. All four frames and the background share the same crop; coordinates are mapped back to the full image. The downloaded weights are unchanged. This uses one-quarter of the full-frame input pixels without shrinking the ball; actual speed and accuracy are unmeasured. After two failed heavy checks, recovery uses the full frame. Cheap misses alone do not trigger full-frame inference; crops center on the last observation. Each attempt consumes one call from the existing budget. `--full-frame-heavy` disables cropping for reference runs. Summary fields `heavyCropCalls` and `heavyFullFrameCalls` count real inference attempts; cached full-frame observations cannot evaluate crop accuracy. No tests were run for this change.

Latest optimization pass (not tested): candidate components are formed before the Lab and motion gates, requiring at least 70% of component pixels to match the locked seed chroma. Signed brightness change rejects vacated bright spots (mean change must exceed 3 and at least 15% of pixels must brighten by more than 10). Component statistics use bounding-box slices instead of rescanning the full search area for each candidate. These gates can also reject real balls with little brightness change; accuracy needs evaluation.

Heavy calls now default to a maximum of 20 per run, including cached lookups. Repeated heavy misses use retry intervals of 2, 4, 8, then 16 frames. Successful recovery clears backoff; recovery after more than five lost frames clears stale velocity. Budget exhaustion can reduce recall and leaves subsequent cheap detections unverified. This enforces a cost cap; it does not demonstrate baseline accuracy within that cap. For an unlimited reference replay on every post-seed frame use `--verify-every 1 --max-heavy-calls 0`. Cached replay imports no PyTorch/model code, but still decodes the source video. Historical measurements below predate this pass.

`fast_track.py` is a separate, experimental color/motion tracker using OpenCV and NumPy. The initial clear seed classifies white-ball appearance once. It locks HSV brightness/saturation bounds and an 8-unit CIE Lab chroma radius around the seed patch median; at least 70% of each candidate's pixels must fall within that chroma radius. Orange seeds are unsupported. Supply a clear ball seed as seconds and normalized x/y:

```powershell
experiments/ball_detection/.venv/Scripts/python.exe experiments/ball_detection/fast_track.py 'path/to/clip.mp4' --seed 2.0 .5308287 .4364107
```

The command above uses only the cheap tracker. To reuse saved heavy detections without running the neural network:

```powershell
experiments/ball_detection/.venv/Scripts/python.exe experiments/ball_detection/fast_track.py 'path/to/clip.mp4' --seed 2.0 .5308287 .4364107 --end 8.28 --cached-heavy-json experiments/ball_detection/output/detections.json --output experiments/ball_detection/output/cached-track.json
```

Use the same source video and timeline as the saved detections. This cache contains heavy observation positions, not decoded frames or precomputed color features. Frames missing from the saved observations return a detector gap. Choose either `--cached-heavy-json` or `--heavy`.

| Option | Default | Behavior |
| --- | --- | --- |
| `--max-heavy-calls` | `20` | Caps inference calls or cached lookups; `0` allows unlimited calls. |
| `--verify-every` | `5` | Verification interval while tracking, subject to budget and retry backoff; `1` disables backoff. |
| `--min-color-score` | `0.75` | Candidate score must exceed this threshold. |
| `--ambiguity-margin` | `0.08` | Best candidate must beat the runner-up by more than this margin. |

The seed consumes no heavy call. Output summary fields `heavyCalls`, `maxHeavyCalls`, and `heavyBudgetExhausted` expose budget use. Once the budget runs out, the tracker can continue cheap tracking. After more than five misses, wide-search candidates are retained for review but cannot reinitialize the tracker without heavy confirmation. A cap of 20 does not guarantee recovery throughout a clip.

It searches near a constant-velocity prediction for small moving white components scored by proximity, saturation, seed chroma, local brightness contrast, and motion, rejects ambiguous candidates, and stores one record per frame. Records distinguish `seed`, `detected`, `predicted`, and `lost`; match scores are heuristic, not calibrated correctness. After five missed frames, `needsDetector` signals that a heavy detector should reinitialize position and velocity. With `--heavy`, real TrackNet inference handles uncertainty and periodic verification subject to the budget and retry backoff above. Strong visual candidates in heavy-detector gaps are retained as `recoveryCandidate` with `reviewRequired`, separately from predictions and accepted observations. Its experimental JSON is deliberately separate from the app's detection import so predictions cannot appear as confirmed observations. Working coordinates use a 640x360 image; estimates can move outside the image during loss.

On the existing validation clip, seeded from the saved TrackNet position at 2.0 seconds and ending at 8.28, a quick run processed 157 frames in 0.558 seconds including decoding and startup work (0.210 seconds inside tracking/preprocessing). It accepted 10 visual candidates, predicted 40 frames, and marked 106 lost; the remaining frame was the seed. This demonstrates cheap computation, **not useful tracking accuracy**. Candidate acceptance is not ground truth. Next work: visually label representative frames, tune color/blur candidate extraction and direction-change handling. No speed comparison with neural inference is an accuracy-equivalent comparison.

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

## Diagnosis and budgeted replay (2026-11-01)

The previous `.93` score threshold allowed only 3 nearby candidates above threshold in a replay guided by saved detections. Of 106 saved post-seed locations, 51 had nearby components surviving the appearance/motion gates; 45 lacked a nearby HSV/contrast component, 8 failed Lab, 1 failed motion, and 1 failed size. Many blurred ball patches exceeded the locked saturation limit. These are comparisons to model positions, not ground-truth labels. Diagnostic frames and filter counts are cached locally under `output/diagnostic-frames.npz` and `output/filter-diagnosis.json`.

Current defaults use score `.75`, margin `.08`, and a 48-pixel initial search radius. After two misses the bounded search centers on the last observation rather than indefinitely extrapolating velocity. Searches after prolonged loss produce review candidates only; allowing them to reset tracking introduced drift exceeding 100 pixels in an intermediate cached replay and was rejected.

Heavy attempts are spaced by at least `ceil(segmentFrames / maxHeavyCalls)` when a finite budget is configured. On this rally that is 8 frames, preventing repeated successful recoveries from consuming all calls in the first second. Unlimited mode retains the previous cadence/backoff behavior.

A real 2.0–8.28-second run completed in **13.24 seconds** including detector initialization, decoding, and tracking. It used **20 heavy calls** (19 crops, 1 full frame), distributed through 8.16 seconds, and accepted **34 detections**, including **19 cheap detections**. Of 106 saved baseline detections, 29 were recovered within 1.72 pixels and 77 were missed; 5 additional detections remain unverified. The previous budgeted run took 12.05 seconds, accepted 18 detections with no cheap detections, and exhausted its calls at 2.96 seconds. Runtime is below the 20-second target on this clip; detection coverage remains inadequate. Results: `output/debug-live-trial.json`. No claim of ground-truth accuracy or performance on longer videos is established.

## ONNX investigation

The neural network is standard convolution/BatchNorm/ReLU/MaxPool/nearest Resize/Concat/Sigmoid, with input `[1,15,288,512]` and final heatmap `[1,1,288,512]`. A small wrapper around `_forward(x)[:, -1:]` can avoid MMEngine's training interface during export. The PyTorch state file is about 46.6 MB; browser activation memory and latency still need measurement.

After accuracy is accepted, export with [torch.onnx.export](https://docs.pytorch.org/docs/stable/onnx.html), compare heatmaps and decoded centers against PyTorch on these same frames, then benchmark [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) WebGPU and WASM. Browser preprocessing must reproduce BGR ordering, background and temporal alignment. Operator support is provider-specific. Do not change precision/quantize until position and gap parity have been measured. No ONNX runtime, model download, or inference workload has been added to the production bundle yet.

White-only revision: synthetic checks passed for moving white selection, orange rejection, and stationary rejection. On the same 157-frame segment it accepted one candidate, predicted five frames, then marked 150 lost (0.310 seconds including decoding). This conservative result is not an accuracy improvement. After loss it waits for detector recovery instead of choosing arbitrary candidates from a growing region; this earlier result used color-only mode.

The false candidate shown near the far player's face scored 0.90 under the HSV-only filter. Its seed chroma was Lab a/b=(128,122); candidate pixels were around (134,132), outside radius 8. A targeted replay of frame 6.84 s rejected that candidate. A subsequent cached-heavy replay of the 157-frame segment, using `--min-color-score .8 --ambiguity-margin .08` before the latest optimization pass, accepted 29 cheap detections and made 127 cached lookups (77 detections, 50 gaps). All 106 accepted post-seed detections matched saved heavy detections within 2.89 pixels; there were no additional accepted detections in heavy gaps. This measures agreement with the saved model, not ground-truth accuracy. These counts used the previous unlimited fallback behavior and do not describe the current 20-call default. No tests or inference runs were performed for the latest optimization pass.
