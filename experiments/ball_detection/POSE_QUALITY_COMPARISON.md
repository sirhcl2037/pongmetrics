# Source-resolution pose comparison — 2026-11-09

Tested the higher-quality `LJK vs Hugo.mkv` (3840×2160) at 6.10, 6.30, 6.50, 6.70 and 6.90 seconds, for the two playing people (saved track IDs 1 and 2). The referee was excluded explicitly. This is five selected frames / ten person observations, not a general pose benchmark.

Baseline: saved whole-frame YOLO11n-pose results at inference size 640. Experiment: crop each player from the original decoded 4K image, with 25% padding on every side, and pass that crop to the same checkpoint at size 640. This preserves more player detail than resizing the entire 3840-wide frame first. Cropping also changes surrounding context and apparent person scale, so observed changes cannot be attributed exclusively to pixel resolution.

The crop detection must overlap the saved target box at IoU >= .25; unmatched crops export null landmarks. Track identity comes from the saved observation, not a new tracker. Joint coordinates are lifted back to normalized original-video coordinates; missing model joints remain null. Checkpoint SHA is verified against the baseline. Original data and app defaults remain unchanged.

## Qualitative result

All ten crops matched their target people. The ten additional calls took 4.26 seconds including first-call overhead, excluding model construction and source decoding. This does not establish a full-pipeline runtime comparison: the baseline was a previous 50-frame tracking run, whereas these are ten crop predictions.

- **6.10:** crop arm positions differ modestly; both modes broadly follow the players. This is not a dramatic improvement.
- **6.30:** for track 2 (Liang), the raised left arm behind/alongside the head is represented more plausibly by the crop. The full-frame result places that arm much closer to the torso. The hitting arm looks plausible in both. Some limb/body points elsewhere still shift.
- **6.50:** both modes capture much of the upper-body posture. Wrist positions remain approximate, particularly where the hands overlap. No clear universal winner.
- **6.70:** track 1's occluded bent leg is estimated in empty space in both modes. Cropping does not reveal anatomy hidden by the table. High scores cannot establish that these points are correct.
- **6.90:** upper-body arms generally look plausible, with changed wrist/elbow positions. Neither output establishes exact contact location. Motion blur persists in the original 4K frame.

The 4K source is visually clearer than the previously audited 720p clip, but those are different players, camera angles, movement, and encoding. That comparison does not isolate source resolution as the cause. A controlled resolution test would downsample this same 4K source and use identical crops and inference settings; that was not run here.

**Decision:** source-player crops are a promising optional refinement, especially for visible arms, but do not yet warrant replacing the default. Blur and occlusion remain independent problems. Use this output for visual review, not validated biomechanics. Next useful evidence is manual marking of visible joints in these exact frames, including unavailable labels for hidden/ambiguous joints, before evaluating a broader crop mode.

## Reproduce and inspect

```powershell
experiments/ball_detection/.venv/Scripts/python.exe experiments/ball_detection/compare_pose_crops.py 'C:/Users/liang/Downloads/LJK vs Hugo.mkv' experiments/ball_detection/output/pose-smoke-4k.json --times 6.10 6.30 6.50 6.70 6.90 --track-ids 1 2 --output experiments/ball_detection/output/pose-quality-4k
```

Ignored output contains `comparison-{305,315,325,335,345}.jpg` (left = original whole-frame pose; right = crop pose, both drawn over the same source pixels), `comparison.json` with both sets of observations, and importable `player-tracks.json` with five sparse crop-pose frames. The latter parses successfully in the browser's player-track importer. Its overlay will only appear at those five sampled frames; it is not a continuous track.

Unit tests cover coordinate lifting, preserving unavailable joints and input arrays, and target-box matching. No further weights were downloaded, no training was run, and no frontend behavior changed.
