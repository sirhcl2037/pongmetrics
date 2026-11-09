# Full-clip baseline, 2026-11-08

Source: `LJK vs Hugo.mkv`, 3840×2160, 50 FPS. The container reports 550 frames/11 seconds, but sequential decoding confirms 764 frames with presentation timestamps 0–15.26 seconds (15.28-second frame span). The corrected loader processed all 764 frames in one continuous default-tracker run. No native-zoom or weak-peak experiment flags were enabled.

Ground truth: `LJK vs Hugo.mkv.ball-labels (1).json`, SHA-256 `b32f5881c6ac3089524e9a44ec2b54df860fac0336aa284cf9cdee852299a5a2`. It contains 504 visible labels, 255 out-of-frame labels, and five occluded labels. All 764 timestamps were evaluated with zero alignment offset and no excluded/unprocessed frames. The original label file is unchanged and a copy is retained in the ignored review-backup directory.

## Results

| Measurement | Result |
| --- | --- |
| Visible frames with a proposal | 476/504 = 94.44% |
| Visible frames without a proposal | 28 |
| Predictions on out-of-frame/occluded frames | 0/260 |
| Centers within 20 reference pixels | 473/504 = 93.85% |
| Proposals outside that center tolerance | 3 |
| Total observations | 476 |
| Model calls | 774 (450 crops, 324 full-frame) |
| Full runtime, one CPU run | 739.37 seconds = 12m 19s |
| Inference / startup / other work | 692.72 s / 29.75 s / 16.89 s |

Reference coordinates are scaled to 1280×720. A 20-reference-pixel tolerance is equivalent to 60 original pixels in this 4K video, allowing comparison with the earlier 720p threshold. At the stricter 20-original-pixel threshold, 468 visible frames pass. Coverage and center localization are deliberately reported separately: a center correction does not automatically mean that the detector selected the wrong object.

The labels include 350 accepted proposals, 154 manual corrections/placements, and 260 visibility decisions. Accepted proposals are human-reviewed but their centers were copied from the earlier detector output. Center-error statistics on this development set are therefore not an independent estimate of manually drawn center accuracy. These results are specific to this clip and do not establish general precision/recall on unseen matches.

## Misses to investigate

The 28 misses occur in 14 short runs, at most four consecutive frames (80 ms):

`219–221, 425–426, 468, 483–485, 522–525, 552, 576, 599–601, 603–605, 611, 639, 651–652, 705–706, 727`.

Of those missing frames, 25 have no retained candidate, two have candidates elsewhere, and one (frame 522) has a nearby candidate with peak score .451 that association rejected. “No retained candidate” refers to the configured .15 extraction floor and existing crop-boundary filtering; it does not establish that the raw heatmap contains no lower-score evidence. Threshold/association changes alone cannot recover all these misses from the saved candidate cache. Representative examples include service/contact, blur, and a ball at the image edge; categories need inspection before model changes.

The three center-tolerance failures are frames 553, 577, and 578, with approximately 76.3, 63.8, and 63.2 original-pixel offsets. They are retained as localization errors, not asserted to be wrong-object detections.

## Artifacts and reproduction

All generated data is in `experiments/ball_detection/output/crop-comparison/`:

- `4k-full-baseline.json`: complete timestamped detections and diagnostics, importable with the original video.
- `4k-full-candidates.json`: retained evidence for cached association experiments.
- `4k-full-baseline-evaluation.json`: scoring, reviewed-label provenance/hash, failure frames, candidate diagnosis, and runtime.
- `4k-full-baseline-errors.jpg`: selected disagreement examples, green reviewed centers and yellow detector observations. Selection is diagnostic rather than representative sampling.

```powershell
$py = 'experiments/ball_detection/.venv/Scripts/python.exe'
$clip = 'C:/Users/liang/Downloads/LJK vs Hugo.mkv'
$labels = 'C:/Users/liang/Downloads/LJK vs Hugo.mkv.ball-labels (1).json'
$out = 'experiments/ball_detection/output/crop-comparison'
& $py experiments/ball_detection/temporal_track.py --base-model --video $clip --output "$out/4k-full-baseline.json" --save-candidates "$out/4k-full-candidates.json"
& $py experiments/ball_detection/evaluate_manual_labels.py "$out/4k-full-baseline.json" $labels --frame-offset 0 --output "$out/4k-full-baseline-evaluation.json"
& $py experiments/ball_detection/review_manual_errors.py $clip "$out/4k-full-baseline.json" $labels --output "$out/4k-full-baseline-errors.jpg"
```

The evaluation CLI regenerates its standard report; the saved development report additionally records the dataset hash and candidate diagnosis above. The earlier partial track and the reviewed labels were not overwritten.
