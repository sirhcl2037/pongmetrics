# OpenTTGames game_5 and high frame rate gap recovery

The source was the 1920×1080, 120 fps `game_5.mp4` and matching annotation archive from [OpenTTGames](https://lab.osai.ai/). The archive SHA-256 was `7f5dfedcb2e381a4ec9809418f346260135e3ec39b56f64ca5cd1325967a3671`. Video, annotations, and generated tracks remain local.

Two 520-frame excerpts were selected from the supplied event timestamps before running the detector. Source intervals are half-open. The zero-based timestamps were checked against decoded images around four labels in each segment, comparing shifts from -2 to +2 frames. The lossless copies and adjacent source manifests are in `output/opentt-game5`. Their first, middle, and last decoded frames matched the source hashes. These excerpts are sufficient for the recorded experiment.

| Segment | Source frames | Time |
| --- | --- | --- |
| Development | 5220–5740 | 43.500–47.833 s |
| Reserved | 64540–65060 | 537.833–542.167 s |

The existing forward and reverse refinement used a four-frame limit for a gap between strong observations. At 120 fps this excluded the weak run already observed in `game_4` and a six-frame gap at a racket contact in `game_5`. The revised limit is the larger of four frames and 50 ms of source frames rounded down. At 120 fps it allows six frames. The same image evidence, agreement and scene-cut checks apply. The probe now closes the decoder as soon as it has completed its last target.

The comparison uses the same baseline detections, image input, checkpoint, adapter, and ONNX runtime for both gap limits. The revised pass recovered all three labelled misses in the previously reviewed `game_4` excerpt (36934–36936), plus one visually supported unlabelled position at 36937. Recovery used eight additional neural calls. This known failure motivated the change.

On the `game_5` development excerpt, the wider window recovered one visible unlabelled position at 5667 before racket contact. Both limits found all 230 visible labels within five reference pixels. The revised probe used 12 calls, taking 9.3 seconds with the existing model loaded. All eight detections on frames marked absent aligned with a visible ball in the sampled images.

On the reserved `game_5` excerpt, both limits found all 217 labelled centres within five reference pixels, recovered the same two unlabelled positions during the toss, and produced identical ball tracks. Both probes made seven additional neural calls and ran in about 13 seconds after model initialization. The seven reserved predictions on labelled absence align with a visible ball in the source frames.

On each `game_5` excerpt, both passes matched all six reviewed returns to the correct hitters and missed the serve. Neither pass produced an extra suggestion in either review interval. Each interval included seven bounce and seven net labels. The added development ball point narrowed the bracket for contact at 5670 from frames 5666–5673 to 5667–5673 and moved the estimate one frame later. The reserved contact output was identical. Serves remain an accuracy target for the contact detector.

## Reproduce

Follow the preprocessing and scoring commands in [OPENTT_GAME4.md](OPENTT_GAME4.md) with the source ranges above and `game_5.zip`. To run the updated refinement, add `--refine-gaps` to the default `temporal_track.py` command. The probe can also reuse an exact saved baseline:

```powershell
$exp = 'experiments/ball_detection'
$py = "$exp/.venv/Scripts/python.exe"
$out = "$exp/output/opentt-game5"
& $py "$exp/probe_ball_gaps.py" "$out/reserved.avi" "$out/reserved-baseline.json" --head-adapter "$exp/adapters/review-head-v1.json" --backend onnx --output "$out/reserved-probe.json"
& $py "$exp/refine_ball_gaps.py" "$out/reserved-baseline.json" "$out/reserved-probe.json" --output "$out/reserved-refined.json"
```

Contact references were picked independently from decoded frames. Timestamps are approximate within two frames. The development review covers 5220–5708 with contacts at 5244 (left serve), 5322 (right), 5397 (left), 5465 (right), 5535 (left), 5602 (right), and 5670 (left). The reserved review covers 64540–65060 with contacts at 64628 (left serve), 64702 (right), 64747 (left), 64796 (right), 64849 (left), 64906 (right), and 64948 (left).
