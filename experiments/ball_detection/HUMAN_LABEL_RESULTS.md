# Human-label development experiments, 2026-11-08

The user supplied 219 timestamped labels for the 1280×720, 25 FPS, 8.76-second match clip. Keep the original export unchanged. The exported clicks align to the preceding decoded frame: use an explicit prediction-frame offset of -1. A visual spot check at exported frame 70 matches decoded frame 69. On the historical tracker, this adjustment reduces median error from 18.6 pixels to 1.5 pixels.

One label at 8.30 seconds is between the 25 FPS frame timestamps and duplicates the approximate frame index of the 8.32-second label. Exclude that ambiguous record rather than double counting it. This leaves 181 visible frames, 33 negative frames, and four uncertain frames. Exclude uncertain labels and frames the detector did not process. Errors below use original-resolution pixels.

## Measurements

| Experiment | Visible detections within 20 px | Detections on negative frames | Runtime |
| --- | --- | --- | --- |
| Historical guided tracker, first 207 frames | 152/180 | 0/22 | 83.10 s historical run |
| Current guided control, all 219 frames | 154/181 | 2/33 | 146.39 s |
| Lower connected-component threshold to .01 | 147/181 | 2/33 | 116.00 s |
| Isolated weak peaks with two-sided recovery | 160/181 | 5/33 | 133.14 s inference/capture plus ~.005 s replay |
| Image-motion recovery on historical track | 156/180 | 1/22 | Historical inference plus .63 s recovery |
| Exploratory beam search on peak cache | 165/181 | 18/33 | Reuses peak inference; offline association additional |

These are single sequential CPU runs, not repeated controlled speed benchmarks. The historical run excludes the close-up and constructs its background from the shorter segment. The full-clip control and peak trial analyze identical footage with the same background sampling and model-call counts. The measured peak trial was shorter than its control, but run-to-run CPU variation prevents claiming a reliable speedup.

The weak-peak cache contains a candidate within 20 pixels on 178/181 visible frames. This is an oracle candidate ceiling, not tracker accuracy: it uses the human labels to choose among peaks. Candidate selection remains difficult. Lowering the whole-region threshold merges weak evidence into diffuse blobs. A permissive motion path increases false positives substantially. A channels-last CPU tensor-layout microbenchmark did not improve speed.

## Decision

Keep current browser generation defaults. The new `--weak-peaks` and `--candidate-floor` arguments are experiment controls. No tested change met the requested approximately 99% visible-frame recovery with preserved false-positive performance and runtime. Empty evidence stays empty; recovered positions come from heatmap maxima or moving image components.

The next substantive approach is model adaptation on motion-blurred/contact examples, with separate continuous exchanges reserved for validation. The supplied clip is now a development set because these experiments used its labels. A 99% development-set result would need independent validation. Benchmark any adapted model at the existing input size and architecture before considering a larger network.

## Reproduce

From the repository root in PowerShell, using the existing Python environment and downloaded weights:

```powershell
$py = 'experiments/ball_detection/.venv/Scripts/python.exe'
$clip = 'C:/Users/liang/Downloads/Wang Chuqin vs Sora Matsushima Match Highlights #ITTFWorldCup 2026.mp4'
$labels = "$clip.ball-labels.json"
& $py experiments/ball_detection/temporal_track.py --video $clip --output experiments/ball_detection/output/current-control.json --save-candidates experiments/ball_detection/output/current-control-candidates.json
& $py experiments/ball_detection/temporal_track.py --video $clip --weak-peaks --output experiments/ball_detection/output/peak-tracks.json --save-candidates experiments/ball_detection/output/peak-candidates.json
& $py experiments/ball_detection/temporal_track.py --candidates experiments/ball_detection/output/peak-candidates.json --output experiments/ball_detection/output/peak-recovered.json
& $py experiments/ball_detection/evaluate_manual_labels.py experiments/ball_detection/output/peak-recovered.json $labels --frame-offset -1 --output experiments/ball_detection/output/human-label-evaluation.json
& $py experiments/ball_detection/motion_recovery.py $clip experiments/ball_detection/output/current-control.json --output experiments/ball_detection/output/motion-control.json
```

Generated reports/caches remain in the ignored output directory. The original human dataset remains in Downloads. The exploratory beam search was an ad hoc negative experiment; its output is recorded locally but it is not a supported tracker option.
