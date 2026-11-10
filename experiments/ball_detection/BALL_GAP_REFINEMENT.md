# Ball gap refinement

Replaying the adapted TrackNet detector around short gaps recovered eight additional reviewed ball positions on the two development clips. The pass uses heatmap candidates from both directions through the video and adds positions to an exported track for review.

| Clip | Within 20 reference pixels before → after | Missing visible frames before → after | Detections on labelled empty frames before → after |
| --- | ---: | ---: | ---: |
| 720p Wang–Matsushima | 157/181 → 162/181 | 24 → 19 | 0/33 → 0/33 |
| 4K Liang–Calderano | 479/504 → 482/504 | 22 → 19 | 0/260 → 0/260 |

Reference pixels are scaled to 1280×720. The 720p label export uses the previously reviewed −1-frame evaluation adjustment. The 4K evaluation uses zero offset. The five new 720p points are within five reference pixels of the labels. The 4K additions at frames 525, 577 and 727 are about 5.3, 17.4 and 8.1 reference pixels from the reviewed centers. Both clips contributed to development. Measuring transfer requires an independent clip.

## Evidence and cost

`probe_ball_gaps.py` takes an existing detector export and runs on missing frames within four frames of an existing observation. It checks the source metadata, checkpoint, head adapter, and timestamps against that export. The forward pass uses the normal four-frame history. The reverse pass uses three subsequent frames and ends on the same target frame. Future frames are bounded to the same continuous scene. The probe also records candidates before the crop margin rejects them on missing guided-crop frames.

`refine_ball_gaps.py` recovers a point only inside a short gap between existing anchors, with an additional rule for the physical image edge. In a gap of at most four frames, a candidate must score at least .25, have a candidate scoring at least .10 within three model pixels in the opposite temporal direction, and be within eight model pixels of the line between observed anchors. The line filters candidates. The stored location comes from the selected image candidate. The anchors come from the original export. The passes share model weights and source images, and their agreement measures stability across temporal inputs.

At 4K frame 727 (14.54 s), the guided crop responded at .578 near the right edge, but the crop margin removed it. The offline refinement requires a preceding strong observation, agreement with its motion prediction, and supporting peaks from both full-frame passes. It accepts the measured crop candidate within four model pixels of the physical image edge.

| Clip | Probed frames | Additional model calls | Time inside additional model calls |
| --- | ---: | ---: | ---: |
| 720p | 49 | 105 | 182.81 s |
| 4K | 30 | 68 | 113.32 s |

These are CPU measurements from the individual probe runs. They exclude video decoding and model setup. The cached results support fast replay. The additional inference cost makes this an offline experiment. Further work can reuse forward peaks from the original pass and run reverse inference only for gaps that meet the anchor checks.

The probe also found a strong reverse response at 4K frame 222 before the first track anchor. The short-gap rule leaves this frame open. A separate comparison would be needed to validate starting a track from that kind of evidence.

## Reproduce

Use the original video, the corresponding `head-final-*-v1.json` export, and the original `output/head-adapter-4k-v1.json` file whose hash matches that export. The promoted adapter contains the same learned coefficients but different manifest metadata. The probe ties its output to the exact baseline bytes so a cache cannot accidentally be paired with a different run.

```powershell
$experiment = 'experiments/ball_detection'
$python = "$experiment/.venv/Scripts/python.exe"
& $python "$experiment/probe_ball_gaps.py" 'C:/Users/liang/Downloads/LJK vs Hugo.mkv' "$experiment/output/head-final-4k-v1.json" --head-adapter "$experiment/output/head-adapter-4k-v1.json" --output "$experiment/output/gap-probe-4k.json"
& $python "$experiment/refine_ball_gaps.py" "$experiment/output/head-final-4k-v1.json" "$experiment/output/gap-probe-4k.json" --output "$experiment/output/head-gap-refined-4k.json"
```

`output/head-gap-refined-720p.json` and `output/head-gap-refined-4k.json` are complete tracks for **Import ball detections** with the matching clip. `output/gap-refinement-evaluation.json` contains the paired metrics and selected evidence. The browser import parser accepted all 219 and 764 tracking rows. Ball-only contact suggestions stayed at seven on 720p and went from 12 to 13 on 4K. Measuring contact accuracy requires independently reviewed contacts.
