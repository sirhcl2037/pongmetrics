# Original-image crop comparison, 2026-11-08

Compared the existing guided tracker with a quarter-area source crop: half the source width and height, cropped before resizing to the full 512×288 model input. The current tracker first resizes the whole frame to 512×288, then normally feeds a 256×144 tracking crop. Both modes retain full-frame reacquisition, periodic global checks, identical weights, score thresholds, and temporal association. Their guidance paths can diverge as detections change.

The native crop doubles linear ball sampling relative to the existing tracking crop, but its network input has four times the area. Four native source frames are retained, not the whole video. Background sampling is bounded to 24 images at 1024×576 for the native mode, alongside the existing 512×288 background. The higher-resolution background is cropped consistently with the four source images.

## Segments and results

- Previous video: 1280×720, 25 FPS, first 4 seconds (100 frames).
- New `LJK vs Hugo.mkv`: 3840×2160, 50 FPS, 6–10 seconds (200 frames). The initial close-up is excluded from inference. Background estimation still samples the video prefix through the segment end in both modes.
- Sequential CPU runs on the same machine. These are individual runs, not a repeated performance benchmark. Startup, decode/preprocessing, inference, and association are included.

| Video / measurement | Current tracker | Original-image crop |
| --- | --- | --- |
| 720p, within 20 source pixels of human labels | 77/86 visible frames | 76/86 visible frames |
| 720p, predictions on labeled negative frames | 0/13 | 0/13 |
| 720p, median localization error on detected visible frames | 1.35 px | 1.15 px |
| 720p, total runtime | 101.19 s | 173.29 s |
| 4K, frames with observations (not accuracy) | 194/200 | 185/200 |
| 4K, tracklets | 1 | 3 |
| 4K, total runtime | 127.81 s | 399.03 s |

The old labels use the previously reviewed -1 frame offset. Unprocessed, uncertain, and off-grid labels are excluded. The native 720p run recovered no additional visible frame within the 20-pixel tolerance, introduced a wrong-location observation on frame 43 (~24.9 pixels from the click), and lost the previously detected frame 67.

There are no human labels for the new 4K segment, so its observation counts cannot establish precision or recall. The two source videos also differ in camera view, motion, compression, and FPS. Cross-video differences cannot be attributed exclusively to source resolution.

## Interpretation

The original-image crops visibly retain more detail, especially on the 4K source, but this version does not improve the labeled 720p result and is slower on both segments. New 4K tracking also produces fewer observations, more reacquisitions, and 19 same-frame rescue calls versus one for the control. Lost context or changed apparent ball scale are plausible explanations, not isolated causal findings. Keep the app's existing default. The crop mode is exposed only as `--native-zoom` for further experiments. Crop/scale augmentation during model adaptation is a possible next test.

## Reproduce and inspect

```powershell
$py = 'experiments/ball_detection/.venv/Scripts/python.exe'
$old = 'C:/Users/liang/Downloads/Wang Chuqin vs Sora Matsushima Match Highlights #ITTFWorldCup 2026.mp4'
$new = 'C:/Users/liang/Downloads/LJK vs Hugo.mkv'
& $py experiments/ball_detection/compare_native_crops.py --old-video $old --new-video $new --labels "$old.ball-labels.json" --output-dir experiments/ball_detection/output/crop-comparison
```

Generated files in `output/crop-comparison/`:

- `comparison.json` has evaluation details, per-run timings, and model-call counts.
- `720p-baseline.json`, `720p-native-zoom.json`, `4k-baseline.json`, `4k-native-zoom.json` can be imported with their original source video.
- `720p-comparison.mp4` and `4k-comparison.mp4` show synchronized source frames, current tracker on the left and native crop on the right. Yellow marks observed detections; green crosses are the old human labels when supplied. Gaps remain empty.
- Matching JPG contact sheets emphasize disagreement frames, rather than serving as unbiased accuracy samples.
- `input-detail-comparison.jpg` illustrates the same field of view at current crop sampling (displayed at 2×) versus original-image cropping. It illustrates input detail, not detector accuracy.

Use `render_crop_comparison.py SOURCE BASELINE_JSON NATIVE_JSON --output VIDEO.mp4`, optionally with `--labels` for the old clip, to regenerate the review videos/contact sheets. The source videos and original labels are unchanged.
