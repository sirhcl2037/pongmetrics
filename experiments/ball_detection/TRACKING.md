# Temporal neural ball tracking

The current implementation keeps RacketVision TrackNetV3 and its downloaded weights. It adds temporal association, weak-evidence recovery, streaming decoding, guided crops with full-frame recovery, application generation, and inspection/evaluation tools. It requires no paid API or hosted service. The earlier color tracker is still experimental; its pre-existing uncommitted contact changes were preserved.

## What the inspection established

- `detect.py` uses the training loader's current-frame alignment: median background, BGR frames `[i-3,i-2,i-1,i]`, repeating frame zero at the beginning. The last output channel is for `i`. The vendored inference helper ends its range at `i-1`; copying that helper would introduce a frame delay. The training configuration uses `last_only=True`, so the other output channels must not be treated as trained past-frame detections.
- The saved baseline has 207 frames, 141 observations and 66 below-threshold gaps. There are no ambiguous multi-component frames on this particular clip, so improved ambiguity handling is covered by synthetic tests rather than a measured real-clip benefit.
- The color experiment's 20-call cap sacrifices most coverage. No such lifetime call cap exists in the new tracker: neural inference runs on each frame and recovery remains available late in the video.
- Historical `HeavyDetector` retains all decoded frames; `detect.py` retains float32 frames. The new `StreamingDetector` retains four uint8 temporal frames, one background, and transient tensors. Default background construction retains at most 24 sampled uint8 frames, then releases them. Sampling still scans the file once before tracking. JSON observations and diagnostic records still grow with duration; total process memory is not strictly constant.
- Broad person boxes did not establish paddle contact in earlier experiments. The new tracker handles reversal without excluding candidates inside players or relying on those boxes. Player tracking remains an independently generated overlay.

## Association and recovery

`neural_candidates.py` keeps connected components at .5 plus independent weaker components at .15. Strong-component centroids match the original detector exactly. It retains blurred/streak-shaped neural evidence without hard white-color or roundness gates. Peaks are uncalibrated scores.

`BallTracker` in `temporal_track.py` uses pixels per second and actual decoder timestamps. It ranks strong candidates by distance from a velocity prediction and score, rejecting close competing alternatives. Search radius grows with missed time and speed; extrapolation is capped at two frames. A reversal replaces stale incoming velocity. After .32 seconds without observations, motion guidance expires. Two consecutive globally consistent strong observations can reacquire anywhere, resetting momentum. Isolated distant candidates cannot restart the track. Scene cuts detected by a coarse image-difference heuristic and input discontinuities reset history; this is not a complete shot-boundary detector.

Offline confirmation recovers weak *observed heatmap components*: either between strong observations separated by at most three missing frames (12 model-pixel residual gate), or immediately before two consecutive strong observations (8-pixel backward-prediction gate, score at least .2). Competitors must be sufficiently separated. Weak observations cannot bootstrap a chain of weak extrapolations. This uses future evidence and is not a causal realtime tracker.

The live runner usually uses a 256x144 crop at original model scale, centered on motion guidance. It checks the full frame every half second and after a failed crop. A nearby weak crop peak triggers a full-frame retry on the **same frame**, because missing crop context can suppress a real detection. All temporal inputs and background use the same crop. No model replacement or retraining occurred.

Missing observations have `position: null`; predictions and search radii live only in `trackingFrames`. Only observed component coordinates enter `ballObservations`, imports, or session exports. A search radius is a heuristic gate, not a calibrated statistical uncertainty interval.

## Measured development results (2026-11-04)

Same local 1280x720, 25 FPS, 450268-byte match clip; analysis 0–8.28 seconds, 207 frames. Existing weights SHA-256: `00d707b9db7a49561c411e4765956e79bcd7c7e20c7a0a535073440b3e972342`.

| Measurement | Original single-component rule | Temporal, full-frame cached evidence | Temporal, guided crop + rescue |
| --- | ---: | ---: | ---: |
| Observations | 141 | 152 | 153 |
| Original detections retained within 5 px at 640x360 | 141 | 141 | 141 |
| Longest contiguous observation run | 22 frames | 29 frames | 29 frames |
| Gaps between first and last observation | 66 | 55 | 54 |
| Heavy calls | 207 | 207 | 212 |
| Full-frame / crop calls | 207 / 0 | 207 / 0 | 85 / 127 |
| Measured neural inference | 112.98 s | same cached evidence | 77.88 s |
| Capture/run including startup and decoding | 124.38 s | replay adds ~.003 s | 83.10 s, plus latest cached association ~.003 s |

The capture reproduces every original saved baseline coordinate exactly. Full-frame uses the historical full median; guided mode uses a 24-sample median, so the runtime/coverage difference combines background and crop changes. These are individual CPU runs, not a controlled repeated speed benchmark. The guided path is about 33% shorter end to end in these runs; it does **not** meet a 20-second target. An intermediate guided run without same-frame rescue lost baseline frame 186 (7.44 s); the final rescue restored it with five extra inference calls across the clip.

`fixtures/rally-candidates.json` contains the full-frame raw evidence, enabling fast reproduction without neural inference or video. `fixtures/rally-labels.json` records 28 visually inspected frames: 12 clear visible-ball labels and 16 uncertain cases excluded from accuracy calculations. Labels were estimated by Codex from decoded images, with neighboring-frame inspection at contacts. The sample includes systematic frames and selected failures; it is a small development set, not independent human validation.

On those **12 clear visible frames**, baseline localization recall within 5 pixels at 640x360 is **10/12**; both final temporal modes recover **12/12**, including the white blur at frame 170 and visible ball at frame 200. No wrong locations occur among these evaluated positives. There are no labeled negative frames, so the returned precision of 1.0 does not establish general false-positive performance. Of the 12 additional guided observations, two have clear visual labels and ten remain unverified. Do not call all additions true positives. Ball identity consistency cannot be measured on this single-ball clip without labeled distractor/identity sequences. Four trajectory IDs represent reacquired tracklets, not four different balls.

## Reproduce from the repository root (PowerShell)

Use the environment and weights setup in [README.md](README.md). Set the video path once:

```powershell
$py = 'experiments/ball_detection/.venv/Scripts/python.exe'
$clip = 'C:/Users/liang/Downloads/Wang Chuqin vs Sora Matsushima Match Highlights #ITTFWorldCup 2026.mp4'
```

Fast replay and evaluation, using committed evidence:

```powershell
& $py experiments/ball_detection/temporal_track.py --candidates experiments/ball_detection/fixtures/rally-candidates.json --output experiments/ball_detection/output/temporal-full.json
& $py experiments/ball_detection/evaluate_tracking.py experiments/ball_detection/fixtures/rally-candidates.json --baseline-output experiments/ball_detection/output/temporal-baseline.json --labels experiments/ball_detection/fixtures/rally-labels.json
& $py experiments/ball_detection/evaluate_tracking.py experiments/ball_detection/output/temporal-full.json --reference experiments/ball_detection/output/temporal-baseline.json --labels experiments/ball_detection/fixtures/rally-labels.json
```

Reproduce neural capture with the original full-median preprocessing (explicit `0` retains all background frames and uses more RAM):

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '4'
& $py experiments/ball_detection/neural_candidates.py $clip --end 8.28 --background-samples 0 --output experiments/ball_detection/output/temporal-candidates.json
```

Run the final guided path, then replay saved evidence for further experiments:

```powershell
& $py experiments/ball_detection/temporal_track.py --video $clip --end 8.28 --save-candidates experiments/ball_detection/output/temporal-rescue-candidates.json --output experiments/ball_detection/output/temporal-rescue.json
& $py experiments/ball_detection/temporal_track.py --candidates experiments/ball_detection/output/temporal-rescue-candidates.json --output experiments/ball_detection/output/temporal-rescue.json
& $py experiments/ball_detection/evaluate_tracking.py experiments/ball_detection/output/temporal-rescue.json --reference experiments/ball_detection/output/temporal-baseline.json --labels experiments/ball_detection/fixtures/rally-labels.json --output experiments/ball_detection/output/temporal-rescue-evaluation.json
& $py experiments/ball_detection/review_tracking.py $clip experiments/ball_detection/output/temporal-rescue.json --output experiments/ball_detection/output/temporal-review.jpg --review-video experiments/ball_detection/output/temporal-review.mp4
```

Add `--full-frame` to disable guided crops. Without `--end`, the runner analyzes the whole file. Use a continuous fixed-camera shot; global background estimation across multiple shots remains a limitation. `--frames 170,200` selects inspection frames; contact sheets are limited to 24 sampled tiles by default. Yellow means actual observed evidence; blue cross/ring means prediction/search gate; gray means raw candidates. The review video is silent and intended for CFR inspection. Evaluation rejects irregular timestamps rather than silently mapping them to incorrect frame labels.

Tests and frontend checks:

```powershell
& $py -m unittest discover -s experiments/ball_detection -p 'test_*.py' -v
npm --prefix frontend test
npm --prefix frontend run build
npm --prefix frontend run lint
```

The 15 Python tests cover current-frame history, bounded decoding, ambiguity, distant false positives, motion reversal, occlusion, weak evidence, cuts, late reacquisition after a synthetic 13-minute gap, metric accounting, and the recorded rally regression. Six frontend tests cover import/session/overlay behavior and long-observation-list lookup. Build and lint pass. A real local upload to `/api/ball-tracks` returned HTTP 200 for the existing 25-frame smoke clip, producing 20 observations and 25 diagnostic rows accepted by the frontend parsers. That run took 20.79 seconds with 27 heavy calls. This is endpoint/parser verification, not a complete automated browser interaction test.

## Application use and remaining work

Run `npm --prefix frontend run dev`, upload a continuous shot, wait for automatic player generation, then choose **Generate ball tracks**. The local middleware allows one analysis job at a time and supports cancellation. Child Python BLAS/OpenMP thread counts are limited. Imported temporal JSON works on the static deployment as well. **Show predictions and search radius** is optional and separate from the solid ball overlay. Generated observations survive session export; diagnostics remain available in the original tracking output. Generation selects minimum score .15 so recovered weak evidence is visible; raise it to hide low peaks.

Long-match image buffering is fixed structurally and late recovery is tested synthetically. Long-match accuracy, negative-frame precision, changing cameras, lighting drift, orange balls, player identity changes, and variable-rate evaluation remain unvalidated. Forty-eight baseline frames have no neural component above .15; association alone cannot recover evidence the detector does not supply. The highest-impact next validation is an independent, densely labeled continuous rally including negative/occluded/contact frames and distractors, followed by evaluating whether more spatial context or a better detector is needed. A larger unlabeled detection count would not settle that question.
