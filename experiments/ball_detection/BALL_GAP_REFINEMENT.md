# Ball gap refinement

Enable **Recover short ball gaps** before choosing **Generate ball tracks** on localhost, or pass `--refine-gaps` to `temporal_track.py --video ...`. The optional pass uses the selected detector runtime, including the validated local ONNX export. It confirms observed heatmap peaks from both directions and adds them to the track. Static deployments can import the resulting JSON.

The selective pass retains all eight additional reviewed ball positions from the original broad probe on the two development clips.

| Clip | Within 20 reference pixels before → after | Missing visible frames before → after | Detections on labelled empty frames before → after |
| --- | ---: | ---: | ---: |
| 720p Wang–Matsushima | 157/181 → 162/181 | 24 → 19 | 0/33 → 0/33 |
| 4K Liang–Calderano | 479/504 → 482/504 | 22 → 19 | 0/260 → 0/260 |

Reference pixels are scaled to 1280×720. The 720p label export uses the previously reviewed −1-frame evaluation adjustment. The 4K evaluation uses zero offset. The five new 720p points are within five reference pixels of the labels. The 4K additions at frames 525, 577 and 727 are about 5.3, 17.4 and 8.1 reference pixels from the reviewed centers. Both clips contributed to development. Measuring transfer requires an independent clip.

## Evidence and cost

`probe_ball_gaps.py` takes an existing detector export and checks the source metadata, checkpoint, head adapter, and timestamps against that export. It schedules inference only for short gaps bounded by strong observations and crop failures near a physical image edge. It uses the same geometry and continuity checks as the refinement step. The forward pass uses the normal four-frame history. The reverse pass uses three subsequent frames and ends on the same target frame. Future frames stay in the same continuous scene. The edge check examines crop candidates before the crop margin removes them.

`refine_ball_gaps.py` recovers a point only inside a short gap between existing anchors, with an additional rule for the physical image edge. In a gap of at most four frames, a candidate must score at least .25, have a candidate scoring at least .10 within three model pixels in the opposite temporal direction, and be within eight model pixels of the line between observed anchors. The line filters candidates. The stored location comes from the selected image candidate. The anchors come from the original export. The passes share model weights and source images, and their agreement measures stability across temporal inputs.

At 4K frame 727 (14.54 s), the guided crop responded at .578 near the right edge, but the crop margin removed it. The offline refinement requires a preceding strong observation, agreement with its motion prediction, and supporting peaks from both full-frame passes. It accepts the measured crop candidate within four model pixels of the physical image edge.

| Clip | Additional model calls, broad → selective | Additional elapsed time, broad → selective |
| --- | ---: | ---: |
| 720p | 105 → 52 | 49.0 → 27.2 s |
| 4K | 68 → 37 | 41.3 → 28.2 s |

Each pair used the same baseline export, video, model and eight-thread ONNX Runtime session. The probe time includes the second decode and excludes model setup. The broad and selective passes emitted identical observations within each pair. Centers and timestamps also matched the earlier PyTorch refinement. Each figure is from one run on this machine. The optional pass adds time to standard tracking.

The probe also found a strong reverse response at 4K frame 222 before the first track anchor. The short-gap rule leaves this frame open. A separate comparison would be needed to validate starting a track from that kind of evidence.

## Reproduce

For standalone comparison, use the original video, a zero-start export with standard crop geometry, and the exact adapter used in that export. The probe ties its output to the baseline bytes. Run the same command again with `--all-nearby` and a different output path to reproduce the broad schedule.

```powershell
$experiment = 'experiments/ball_detection'
$python = "$experiment/.venv/Scripts/python.exe"
& $python "$experiment/probe_ball_gaps.py" 'C:/Users/liang/Downloads/LJK vs Hugo.mkv' "$experiment/output/ball-speed/onnx-4k.json" --head-adapter "$experiment/adapters/review-head-v1.json" --backend onnx --output "$experiment/output/gap-probe-selective-4k.json"
& $python "$experiment/refine_ball_gaps.py" "$experiment/output/ball-speed/onnx-4k.json" "$experiment/output/gap-probe-selective-4k.json" --output "$experiment/output/gap-selective-4k.json"
```

`output/head-gap-refined-720p.json` and `output/head-gap-refined-4k.json` are complete tracks for **Import ball detections** with the matching clip. `output/gap-refinement-evaluation.json` contains the paired metrics and selected evidence. The browser import parser accepted all 219 and 764 tracking rows. Ball-only contact suggestions stayed at seven on 720p and went from 12 to 13 on 4K. Measuring contact accuracy requires independently reviewed contacts.
