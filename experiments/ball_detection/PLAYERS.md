# Player tracking

Offline YOLO11n person detector + ByteTrack, with one detector call per frame. `PersonTracker` in `player_track.py` is reusable in another Python pipeline; settings and weights can be swapped independently. This is an experimental Python output, not yet a player overlay/import feature in the browser and not yet connected to ball-crop selection.

## Setup

Run from the repository root using the existing Python 3.12 environment:

```powershell
experiments/ball_detection/.venv/Scripts/python.exe -m pip install -r experiments/ball_detection/requirements-players.txt
$env:YOLO_CONFIG_DIR = Join-Path (Get-Location) 'experiments/ball_detection/output/yolo-settings'
experiments/ball_detection/.venv/Scripts/python.exe -c "from ultralytics import YOLO; YOLO('experiments/ball_detection/output/models/yolo11n.pt')"
```

Weights are downloaded from the official Ultralytics assets release and kept in ignored experiment output. No training or video upload is needed. See [YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/), [tracking documentation](https://docs.ultralytics.com/modes/track/), and the provider's [license](https://www.ultralytics.com/license).

## Run and inspect

```powershell
experiments/ball_detection/.venv/Scripts/python.exe experiments/ball_detection/player_track.py 'path/to/clip.mp4' --start 2 --end 8.28 --review-video experiments/ball_detection/output/players-review.mp4
```

Outputs: `output/players.json` (every processed frame), `output/players.jpg` (sampled contact sheet), and the optional silent review video. JSON includes video metadata, weight SHA-256, package version, detector/tracker settings, and timings. `CachedPlayerTracks(path, video=source_path).at(timestamp)` reuses saved tracks without loading the detector; absent frames return `None`. Reuse these observations for ball-tracking experiments instead of repeating neural detection.

The playing-area filter checks box centers within normalized `--roi LEFT TOP RIGHT BOTTOM` (default `.15 .05 .85 .95`). Tune it for another camera angle to exclude officials and spectators. It filters output eligibility; the detector still sees the full frame. When exactly two eligible people first appear, the upper one is labeled `far-player` and the lower one `near-player`. These initial labels stay attached to their track IDs; they are not verified player names. New IDs are unassigned and flagged for review rather than silently inheriting identity. `missingPlayers` and `needsReview` expose gaps and ambiguity. No positions are interpolated or carried through missing frames.

Coordinates are normalized to the original video. `box` gives x/y/width/height; `center` is the box center. `groundProxy` is the box bottom center, **not measured feet or a metric court position**. Arms/rackets can extend outside the box. Person boxes should inform ball tracking, not exclude ball detections inside them.

## Tuning

- `--weights`: another local YOLO detection checkpoint; default `yolo11n.pt`.
- `--imgsz`: maximum inference dimension, rounded internally for aspect ratio; default 416, must be a multiple of 32. Try 512 if distant players are missed.
- `--confidence`: default `.1`, keeping lower-score detections for ByteTrack association; model scores are uncalibrated.
- `--tracker`: default `players-bytetrack.yaml`; edit the separate file or provide another built-in/custom YAML.
- `--roi`, `--start`, `--end`, `--output`: video segment and output controls. A tracker instance is for one continuous shot; split camera cuts and side changes explicitly.

The custom ByteTrack configuration disables score fusion, raises association cost allowance to `.9`, and requires `.5` confidence to start new tracks. Detector duplicate suppression uses IoU `.45`. Initial defaults fragmented the distant player's ID; these settings fixed that on this sample. They are not guaranteed on other footage or doubles.

## Validation on the existing rally

Wang Chuqin vs Sora Matsushima, 2.0–8.28 seconds, 157 frames at 25 FPS:

- Both players observed with their original IDs in **157/157 frames**; exactly two eligible IDs (1 and 2).
- Player-only processing: **10.65 seconds total**, including 1.97 seconds initialization and optional review encoding; 7.20 seconds detection/tracking (about 46 ms/frame).
- Sampled overlays visually inspected: boxes follow both players, with officials excluded. All-frame checks confirmed normalized bounds, presence of both IDs, and consistent near/far order.
- This is continuity/qualitative validation, not labeled box accuracy, cross-cut re-identification, or combined player+ball runtime. Every-frame heavy ball detection plus this detector has not been benchmarked together; the combined 20-second goal remains unproven.
