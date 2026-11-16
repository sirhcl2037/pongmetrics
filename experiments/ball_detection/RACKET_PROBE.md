# Paddle detection probe

`probe_rackets.py` runs a local YOLO ONNX detector on a short interval and saves the raw paddle boxes and a half-speed review video. A matching session adds ball positions and existing shot tags. Matching player tracks let the probe highlight only boxes near a wrist of exactly one tracked person. The JSON retains every proposal for comparison. Wrist proximity supports a candidate player ID. It does not establish visibility, persistent paddle identity, or contact.

## Demo result

Using the authors' LATTE-MV paddle detector, exported locally at 640 pixels, the 4.32–7.44 s demo interval produced 255 box proposals over 78 frames in 7.0 seconds of detection on an eight-thread CPU run. Wrist proximity retained 136 proposals on 75 frames. It removed a visible shoe false positive at 5.88 s and kept the far player's box that overlaps the ball at 5.04 s. At 6.28 s neither player's wrist supported a box. The figures count proposals, and the demo tags are approximate. Accuracy and added contact recall need separately labeled footage.

Audio can refine the timing of a contact candidate supported by video. Audio alone must never add or confirm a return, assign the hitter, or override contradictory visual evidence. An audio peak may trigger a closer look at the video.

- **Visible ball:** Check its position and motion against the player's racket or reachable swing area at that time. A ball crossing the middle of the table while both players are clearly out of reach contradicts a paddle hit. A screen position over the table can also be an airborne ball, so use the trajectory and player context to distinguish a bounce from a hit.
- **Ball briefly hidden:** A missing detection does not rule out contact. A ball entering a plausible occlusion near the player's swing and reappearing on a compatible outgoing path can support an inferred hit. Use synchronized audio to narrow the time within that gap when available. Keep the inferred contact distinct from a directly observed contact and preserve the gap in the measured ball track.
- **Insufficient evidence:** If the surrounding ball track and swing cannot support a plausible hitter and time window, leave the event unconfirmed.

The source MP4 has an audio stream and the demo WebM is silent. This probe evaluates video only. Compare visual-only and visual-plus-audio contact timing on the same labeled windows, with table bounces and background impacts as negatives. Track missed contacts and false positives alongside any timing improvement.

## Reproduce

The model came from the [LATTE-MV authors' PaddleTracker](https://github.com/sastry-group/LATTE-MV/tree/bcada1f744af3e1c39fd5adb5ee5bb534f0d1ba2/PaddleTracker). The exact input was `models/yolov8s-oiv7-finetuned.pt`, Git blob `bd9ed7613e7f0217a1d5a3f260f716d9536de1cd`. It was exported locally with restricted checkpoint loading, Ultralytics 8.4.198, 640×640, float32, opset 17, and NMS disabled in the graph. Keep the checkpoint and exports local to the experiment. Commercial weights clearance has not been established.

```powershell
$env:ULTRALYTICS_SAFE_LOAD = 'true'
$env:TORCH_FORCE_WEIGHTS_ONLY_LOAD = '1'
$exp = 'experiments/ball_detection'
$py = "$exp/.venv/Scripts/python.exe"
& $py -c "from ultralytics import YOLO; YOLO('experiments/ball_detection/output/models/latte-mv-paddle-research.pt').export(format='onnx', imgsz=640, device='cpu', simplify=False, nms=False, opset=17)"
& $py "$exp/probe_rackets.py" 'path/to/match.mp4' --model "$exp/output/models/latte-mv-paddle-research.onnx" --start 4.32 --end 7.44 --session 'path/to/session.json' --player-tracks 'path/to/player-tracks.json' --output "$exp/output/racket-probe"
```
