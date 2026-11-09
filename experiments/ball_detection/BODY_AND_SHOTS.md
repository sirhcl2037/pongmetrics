# Body landmarks and experimental shot candidates

Update: shot suggestions now score reversals along the current player axis (method `player-axis-ball-reversal-v2`). See [POSE_CONTACT_AUDIT.md](POSE_CONTACT_AUDIT.md) for implementation, observed pose errors around contacts, and limits on coaching use. The v1 counts below document the initial run.

## Use in PongMetrics

1. Load a video on localhost. Click **Generate body landmarks**. This runs pretrained YOLO11n-pose + ByteTrack in the existing local Python worker, with 640-pixel inference. It also produces player boxes; a separate person-detection pass is unnecessary.
2. Use **Player overlay** and **Body landmarks** to display skeletons. Joints and bones require joint scores of at least .35; missing joints and missing frames stay hidden. Scores are uncalibrated model strengths.
3. Use the existing player names and Locate clicks to associate tracked people. On side-on footage the referee may also be detected; no automatic name is assigned unless the initial pair is unambiguous. Once any players are mapped, shot attribution considers only mapped tracks. A new track ID does not inherit a name.
4. **Export player tracks & landmarks** saves original model metadata, keypoint scores, timestamps, and identity IDs. Import the file through **Import player tracks** on localhost or the static site. These data are separate from session and ball-label exports.
5. Import or generate ball detections, then click **Find shot candidates**. Jump to each suggestion, inspect the clip, and use the existing shot controls to tag it. Dismiss incorrect suggestions and **Export candidates** to keep that review state. Dismissals are held in memory until exported; they do not enter the ball-label backup. Candidate import is not implemented.

The static deployment displays imported landmarks and computes shot candidates entirely in the browser. Local Python generation is a development-only endpoint; no server or model weights are added to the static production bundle.

## Pose setup and data

Uses the existing pinned `requirements-players.txt` environment. Download the additional official checkpoint once (about 6 MB) from the repository root:

```powershell
experiments/ball_detection/.venv/Scripts/python.exe -c "import urllib.request; urllib.request.urlretrieve('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt','experiments/ball_detection/output/models/yolo11n-pose.pt')"
experiments/ball_detection/.venv/Scripts/python.exe experiments/ball_detection/player_track.py 'path/to/video.mp4' --pose --imgsz 640 --start 0 --end inf --output experiments/ball_detection/output/body-landmarks.json
```

Weights remain in ignored output. JSON retains the existing `pongmetrics-player-tracks` v1 schema with an optional ordered `landmarks` array on each person: 17 COCO joint names, each with `position: {value: {x,y}, source: 'automatic', confidence}` or null. Coordinates are normalized to the original video. Invalid, unavailable, and out-of-image joints become null; low-score valid observations remain in exported data. Array entries align with their person box in the tracked model result. [Ultralytics pose output/keypoint documentation](https://docs.ultralytics.com/tasks/pose/) defines the landmark order; the existing model-provider license applies.

These are 2D image estimates. They do not measure joint depth, force, power, racket angle, or correctness of technique. Use continuous camera views; automatic cross-cut identity is not implemented.

## Shot heuristic

`frontend/src/shotDetection.ts` is a pure function independent of UI and Python. It examines five observed ball positions, requiring eligible scores and no inter-observation gap longer than 2.1 source frames. Displacement on each side must exceed four reference pixels at 1280×720, with a 2D turn above approximately 75.5 degrees. It supports horizontal/side-on and vertical/end-on views. Nearby turns within .16 seconds are consolidated by turn angle.

At the candidate timestamp, current-frame player evidence can add a track ID: a confident wrist within 100 reference pixels, or a box within 35 pixels. Multiple eligible people leave identity unknown. Wrist distance is a loose contact cue, not measured racket contact. Absent human observations remain absent; boxes/joints are never carried through gaps. These heuristic tolerances are initial defaults, not validated contact thresholds.

Candidates explicitly carry `needsReview: true` and an evidence label (`ball-reversal`, `player-proximity`, or `wrist-proximity`). They are separate from `Session.shots`. No forehand/backhand or stroke type is invented. Bounces, localization jitter, scene changes, and long gaps can cause false candidates or missed shots. Scene-cut filtering and direction-dependent uncertainty are future work. Counts below are not measured precision or recall and do not establish the 95% analyzable-shot goal.

## Validation on local clips

- 720p pose sample 2–4 s: 50 decoded frames; both initial player IDs present in 50/50; sampled skeleton overlays inspected.
- Entire 720p clip: 219 decoded frames, initial two identities both present in 208/219; three total IDs including an unassigned extra person. Single CPU run: 12.23 s detection/tracking, 14.62 s total. No pose ground truth exists, so these are continuity counts, not joint accuracy.
- 4K side-on sample 6–7 s: 50 frames, three tracked people including the referee. Initial automatic near/far identities remain unassigned; sampled skeletons inspected and both players had 15 confident joints in the first sampled frame. Player Locate clicks are needed for names.
- Saved adapted ball runs at minimum score .15 produced seven 720p candidates with pose support (six wrist, one box), and twelve 4K ball-only candidates. Full 4K pose inference was intentionally not run; its single-second sample is not a full-rally landmark export.
- Local endpoint smoke test uploaded a three-frame clip, received HTTP 200, parsed three frames and two people per frame with 17 joints each. Requests without Origin returned 403. Test server was shut down afterward.
- Frontend tests cover normalized pose round-trips, invalid joints, missing frames, both reversal axes, straight/stationary motion, long gaps, weak observations, mapped-track filtering, and ambiguous people. Python tests cover normalized landmark conversion and missing points.

Generated artifacts in ignored `output`: `body-landmarks-720p.json` / `.jpg`, `pose-smoke-720p.json` / `.jpg`, `pose-smoke-4k.json` / `.jpg`, `shot-candidates-720p.json`, and `shot-candidates-4k.json`.

## Feedback categories next

Reuse timestamped player/ball evidence across **Footwork & Recovery**, **Stroke Mechanics**, and **Tactical Choices**, with each future suggestion linked to a clip, measurements, and uncertainty. This change establishes pose and event evidence; it does not yet generate coaching judgments. Verify contacts manually before using them to evaluate body sequencing or tactics.
