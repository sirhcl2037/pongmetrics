# Practical upgrade path and research

The pipeline is not research-limited yet. Current human pose is a nano YOLO baseline; ball adaptation only changed the 65 parameters in the final heatmap channel; shot spotting is a hand-built five-observation reversal heuristic. They are usable experiment baselines, not demonstrated state-of-the-art coaching components. The ball detector does already use weights/code distributed with the recent RacketVision benchmark, so "replace everything with a newer paper" is not a sufficient strategy.

## Implemented first upgrade

RTMPose-m on source player regions is now optional, with a separate ONNX CPU environment and actual results on both user clips. See [RTMPOSE_COMPARISON.md](RTMPOSE_COMPARISON.md). Some problematic arms improve qualitatively, with low measured pose-call cost, while inferred hidden joints remain uncertain. This is enough to make an option reviewable, not enough to replace the default or claim a joint-error percentage.

## Next experiments, in order

1. Reuse the existing labelled ball frames for detector evaluation and the existing hard pose images for model comparison. Add a fresh continuous rally for final validation; both existing rallies influenced development. Only ask for manual pose correction on disagreement examples that affect a planned metric.
2. Establish contact-event performance using the shot review workflow. Keep identity mistakes separate from temporal misses. If coverage stalls because five-point ball reversal windows are absent, add image-based contact evidence and short-sequence features; increasingly elaborate angle thresholds cannot recover absent input observations.
3. For ball misses, compare raw heatmaps on contact/blur/advertising examples before changing the model. The saved-candidate diagnosis found 37/46 missed frames without a nearby retained candidate. If visual evidence is inadequate, fine-tune a late decoder block with normal, difficult and negative sequences, source-scale/blur augmentation, approximate-center targets, and held-out-rally validation. Do not train only on mistakes or infer an exact blurred-ball center.
4. Benchmark a temporal ball alternative if suitable pretrained weights are available. Treat partially/fully occluded inferred positions separately from image-supported detections in downstream use. Include all out-of-frame/background frames and false restarts, not only frames where the benchmark assumes a ball exists.
5. Optimize the selected pipeline: reuse player boxes, cache observations, batch independent pose crops where supported, avoid repeating detectors, and benchmark end-to-end CPU/GPU backends before smaller inputs or quantization. A stronger offline model can generate review proposals, but model agreement is not human ground truth. Skip frames only if the feedback metric tolerates the lost contact timing.

## Papers worth reading

| Paper | Why it matters here | Limit before adopting |
| --- | --- | --- |
| [RTMPose (2023)](https://arxiv.org/abs/2303.07399) | Practical pose accuracy/deployment trade-off; implemented first. | COCO benchmark accuracy/FPS is not table-tennis wrist accuracy or this PC's end-to-end speed. |
| [ViTPose (2022)](https://arxiv.org/abs/2204.12484) | Stronger accuracy-oriented pose family for an offline reference comparison. | Larger checkpoints/compute; reference predictions still need checking. |
| [TOTNet (2025)](https://arxiv.org/abs/2508.09650) | Temporal convolutions, occlusion augmentation and visibility-weighted loss; includes occlusion-rich table tennis. | Official README documents training, but I did not find a ready pretrained-checkpoint link there. Reported results do not establish our negative-frame precision or 25/50-FPS performance. |
| [TrackNetV4 (2024 preprint / 2025 implementation)](https://arxiv.org/abs/2409.14543) | Motion-attention/frame-difference enhancement to the TrackNet family. | Paper abstract evaluates tennis/shuttlecock; table-tennis transfer must be tested. [Official code](https://github.com/TrackNetV4/TrackNetV4) is TensorFlow. |
| [TTNet (2020)](https://arxiv.org/abs/2004.09927) | Coarse-to-fine ball detection plus learned short-video event spotting. | Events in this work are bounces/net hits, not our racket-contact/serve categories; dataset is 120 FPS. Useful architecture ideas, not an automatic drop-in solution. |
| [RacketVision (AAAI 2026)](https://arxiv.org/abs/2511.17045) | Paired ball/racket annotations and attention-based fusion; already the source of our TrackNet implementation/weights. | Its forecasting results do not establish our hit identification or coaching quality. Racket tracking is additional work. |

We should attempt new architecture research only after a reproducible existing-model comparison shows a specific failure that source resolution, temporal context, targeted adaptation, or better event modelling cannot address. For PongMetrics the endpoint is trustworthy player feedback on most shots, not a leaderboard pose score.
