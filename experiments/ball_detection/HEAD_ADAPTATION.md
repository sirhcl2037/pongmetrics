# Review-trained detector update, 2026-11-08

`temporal_track.py --video ...` now selects `adapters/review-head-v1.json` by default. The localhost generation endpoint calls this same CLI. Use `--base-model` for the original checkpoint and original tracking defaults; use `--head-adapter PATH` for an explicit experimental adapter. Direct `StreamingDetector` construction retains the original model unless an adapter is explicitly supplied, which keeps feature extraction and experiments reproducible.

## Training and model selection

The CNN backbone and input resolution remain unchanged. We fitted 64 weights and one bias in the existing final 1×1 output channel. All earlier layers and the other output channels remain unchanged. The original checkpoint is never overwritten. The adapter checks the original checkpoint SHA-256 before applying its parameters.

Training used only `LJK vs Hugo.mkv.ball-labels (1).json`: 764 reviewed frames, including 504 visible-ball frames and 260 negative frames. The training-label SHA-256 is `b32f5881c6ac3089524e9a44ec2b54df860fac0336aa284cf9cdee852299a5a2`. The 720p Wang/Matsushima rally contributed no gradient-training examples. It was used for validation and model/policy selection, so it is a development validation set, not an untouched final test set.

We replayed the recorded original-tracker input geometry, extracting sparse 64-channel feature vectors from the frozen network. The 65,641 sampled vectors compress to 3.68 MiB. Full feature maps and video frames are not retained. Positive patches are anchored to actual heatmap responses within eight model pixels of the human click, accommodating approximate centers. Background sampling excludes a 12-pixel radius around the marked point and includes random and hard negative samples. The fixed seed is 42. A regularized weighted binary cross-entropy objective is optimized with LBFGS, anchoring the head to its original parameters. This is an experiment for approximate reviewed labels, not exact ball-shape supervision.

The promoted configuration uses a .4 cutoff when following an established track and .6 for confirming two consecutive observations to start/restart a track. Position and motion evidence still gate association. Missing-frame recovery still requires actual image evidence. Starting a new track is intentionally more conservative than following an existing one. Raw heatmap scores remain uncalibrated strengths, not probabilities.

The original learned adapter improved the untrained 720p rally, but at a shared .5 cutoff introduced three false restarts on the training clip. Strong hard-negative retraining regressed 720p localization to 147/181, a lighter pass to 152/181, and an experimental eight-unit residual head to 144/181. Those variants were rejected. Uniformly lowering the cutoff also introduced false detections. Separating follow and restart thresholds retained the useful observations without those false restarts. Final results below come from fresh live inference, not only replay of fixed crops.

## Final end-to-end measurements

Centers are scored within 20 pixels at a 1280×720 reference resolution (60 original pixels for the 4K clip). Presence coverage and center localization are separate metrics.

| Metric | Original model | Promoted adapter + confirmation policy |
| --- | --- | --- |
| 720p centers within tolerance | 154/181 | 157/181 |
| 720p proposals on negative frames | 2/33 | 0/33 |
| 4K centers within tolerance | 473/504 | 479/504 |
| 4K visible frames with a proposal | 476/504 | 482/504 |
| 4K proposals on negative frames | 0/260 | 0/260 |
| 720p total model calls | 225 | 231 |
| 4K total model calls | 774 | 768 |

The 4K presence coverage rises from 94.44% to 95.63%; its missing visible-ball frames drop from 28 to 22. Center tolerance recall rises from 93.85% to 95.04%. The older 720p export uses its reviewed -1-frame alignment adjustment; the new 4K export uses zero offset. Uncertain/off-grid/unprocessed labels are excluded by the evaluator.

The final runs took 83.74 s for 720p and 270.45 s for 4K, versus earlier baseline runs of 146.39 s and 739.37 s. Host timing varied substantially between runs; those numbers do not establish a reliable speedup. There are no added network layers or larger inputs in the promoted variant. Counting a 256×144 crop as one quarter of a full-frame pass, aggregate CNN work across both clips is approximately 1% lower, with the older clip about 2.3% higher and the 4K clip about 1.9% lower. Runtime should be treated as approximately the same computational budget, with normal per-clip variation.

The 4K labels include 350 accepted earlier detector centers. They are human-reviewed, but not independent manually redrawn coordinates, so center-error statistics are partly detector-assisted. Only two rallies have been used; a third fresh rally is needed for an independent generalization check. This update does not achieve 99% recovery.

## Reproduction

Use the existing Python environment and original downloaded checkpoint. Training and all generated data remain local.

```powershell
$py = 'experiments/ball_detection/.venv/Scripts/python.exe'
$video = 'C:/Users/liang/Downloads/LJK vs Hugo.mkv'
$labels = 'C:/Users/liang/Downloads/LJK vs Hugo.mkv.ball-labels (1).json'
$old = 'C:/Users/liang/Downloads/Wang Chuqin vs Sora Matsushima Match Highlights #ITTFWorldCup 2026.mp4'
$out = 'experiments/ball_detection/output'
& $py experiments/ball_detection/temporal_track.py --base-model --video $video --output "$out/crop-comparison/4k-full-baseline.json"
& $py experiments/ball_detection/adapt_ball_head.py cache --video $video --labels $labels --trace "$out/crop-comparison/4k-full-baseline.json" --output "$out/head-adaptation-4k.npz"
& $py experiments/ball_detection/adapt_ball_head.py train --features "$out/head-adaptation-4k.npz" --output "$out/head-adapter-4k-v1.json"
& $py experiments/ball_detection/temporal_track.py --video $old --head-adapter "$out/head-adapter-4k-v1.json" --strong-score .4 --acquisition-score .6 --output "$out/head-final-720p-v1.json"
& $py experiments/ball_detection/temporal_track.py --video $video --head-adapter "$out/head-adapter-4k-v1.json" --strong-score .4 --acquisition-score .6 --output "$out/head-final-4k-v1.json"
```

The promoted manifest adds the validated tracking settings and provenance to the first learned adapter; its learned coefficients are identical to that adapter. The failed residual/head-mining variants remain opt-in experiment tools, not production defaults. No training data, dense caches, or original videos are committed. Existing reviewed labels and previous tracks are preserved. Detection exports include the selected adapter name/hash and thresholds.

Generated final tracks are `output/head-final-720p-v1.json` and `output/head-final-4k-v1.json`; `output/head-final-validation-v1.json` records the paired baseline/candidate metrics and provenance. Replaying `--candidates` uses the recorded model's evidence and permits explicit score overrides; it cannot change the model that produced that evidence.
