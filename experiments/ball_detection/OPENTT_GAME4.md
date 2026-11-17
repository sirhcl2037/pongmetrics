# OpenTTGames game_4 experiment

This experiment uses the local game_4 video and annotation archive from [OSAI OpenTTGames](https://lab.osai.ai/). The source license is CC BY-NC-SA 4.0. Video, labels, extracted frames, and generated model outputs stay in local ignored storage. Existing PongMetrics human labels remain intact.

## Data and split

The downloaded source is 1920×1080 at 120 fps, 63,120 frames, 526 seconds, and 3,947,371,986 bytes. Annotation archive SHA-256 is `cfa9b066a177f3d822460963b52bf454ca877435e5f454f0e512343f61d2c8cb`.

The split was recorded before model experiments. Source frame intervals are half-open.

| Purpose | Source frames | Time |
| --- | --- | --- |
| Development | 1200–2160 | 10–18 seconds |
| Reserved evaluation | 36000–36960 | 300–308 seconds |

These are segments of one recording. A final claim about generalization needs other matches and camera views. The base TrackNet checkpoint's training overlap with OpenTTGames has not been established.

The video and labels were checked visually at source frames 1278, 1298, 1302, 1327, 1352, and 1397, comparing offsets from -2 through +2 frames. Zero-based source-frame indexing aligns the labels with the ball. Extraction uses FFV1 and verifies identical decoded pixels on the first, middle, and last segment frames.

## Annotation interpretation

`ball_markup.json` contains 7,073 sparse records. Scoring uses the original pixel coordinates and reports 5, 10, and 20 pixel tolerances at the existing 1280×720 reference resolution. `(-1,-1)` is explicit absence. Frames absent from the JSON are unlabelled. Two mixed records, frame 13095 with only x=-1 and frame 50076 with only y=-1, are excluded as ambiguous when they enter an evaluation range.

`events_markup.json` contains 231 bounce, 187 net, and 176 empty-event records. Sampled net events show travel above the net. Each supplied event is retained verbatim. An empty-event label supplies no racket-contact ground truth. Bounce/net proximity can identify suggestions to inspect, but a nearby actual racket contact must be checked separately.

The independently frame-reviewed development contact interval is 1200–1680. Approximate source frames are 1273 (right, serve), 1374 (left), 1443 (right), and 1519 (left). Visual uncertainty brackets are 1272–1275, 1373–1375, 1441–1445, and 1518–1521. These are assistant visual annotations. They have not been verified by the user. Evaluate one-to-one contact matching with the existing ±.12-second tolerance and report hitter identity separately.

## Reproduce

Run from the repository root:

```powershell
$py = 'experiments/ball_detection/.venv/Scripts/python.exe'
$exp = 'experiments/ball_detection'
$out = "$exp/output/opentt-game4"
& $py "$exp/opentt_benchmark.py" --video "$env:USERPROFILE/Downloads/game_4.mp4" --annotations "$env:USERPROFILE/Downloads/game_4.zip" --start-frame 1200 --end-frame 2160 --output "$out/development.avi"
& $py "$exp/temporal_track.py" --video "$out/development.avi" --backend onnx --cpu-threads 8 --output "$out/dev-baseline.json" --save-candidates "$out/dev-baseline-candidates.json"
& $py "$exp/opentt_benchmark.py" --annotations "$env:USERPROFILE/Downloads/game_4.zip" --detections "$out/dev-baseline.json" --start-frame 1200 --end-frame 2160 --output "$out/dev-baseline-evaluation.json"
```

The evaluator checks the adjacent segment source metadata and annotation archive hash. Predictions use segment-relative timestamps. The report restores original source-frame indices for errors and event inspection. Partial inference is exposed by the unprocessed-label count.

To compare the production default with sparse idle search on the reserved segment, extract frames 36000–36960 to `reserved.avi`. Run `temporal_track.py` twice on that AVI, passing `--sparse-idle-search` on the second run. Score both outputs with `opentt_benchmark.py` using the reserved source range. Run the trials sequentially on the same machine and use the same backend and thread count.

Generate player tracks once with `player_track.py reserved.avi --start 0 --end inf --no-review-sheet --output reserved-players.json`. For the contact comparison, the review JSON takes a half-open source-frame `interval`, the two mapped IDs under `trackIds` keyed by `left` and `right`, and a `contacts` array of records with `frame` and `side`. Pass each ball track and its evaluation JSON with the same player tracks and review JSON:

```powershell
node --experimental-strip-types "$exp/compare_opentt_contacts.mjs" "$out/reserved-baseline.json" "$out/reserved-players.json" "$out/reserved-baseline-evaluation.json" "$out/reserved-contacts.json"
```

The contact comparison reports non-racket events within 50 ms of unmatched suggestions. The manual review interval defines the denominator for racket-contact scoring.

## Development findings

The current detector at 120 fps found all 138 labelled ball centres within five reference pixels. It returned observations at both frames marked absent. The ball is visible at the predicted position in the original frames 1311 and 1312, as well as the surrounding frames, so the scored disagreements remain in the report.

The current shot detector matched all four reviewed contacts within 120 ms with the correct hitter and no extra suggestions. It used ball observations on either side of a six-frame gap to suggest the right player's return near frame 1443. The mean absolute timing difference against the visual references was 11 ms.

Full-frame inference took most of the development baseline runtime. At the default setting the detector continued searching every frame once it lost the ball. A cached development experiment suggested that searching every other idle frame would avoid 282 calls and preserve labelled accuracy. It removed the visible, unlabelled ball observation at frame 1217, immediately before reacquisition.

The opt-in `--sparse-idle-search` flag applies to video with at least 100 fps. It retains the original video frame grid and neural history, and runs on every frame during active tracking, pending acquisition, and detected cuts. Skipped frames retain explicit missing evidence. Ball positions still come from image inference.

The live development run used the same 960 frames, checkpoint, adapter, float32 ONNX graph, and eight inference threads. Runtime went from 379.9 to 327.5 seconds, and neural calls went from 972 to 690. The flag preserved all 138 labelled detections within five reference pixels and all four confirmed contact suggestions. Across the full segment it removed one visible unlabelled observation, at frame 1217. This makes the flag suitable for a speed versus coverage choice during local analysis.

A small development probe compared the model's original consecutive frame input with a four-frame stride, and with a gamma adjustment for this darker recording. The original history detected 46 of 46 sampled visible positions within 20 reference pixels. The stride detected 45, and the gamma adjustment added no detections. The final trial keeps the original input.

The reserved contact review covers source frames 36650–36954. The visual references are frame 36730 (left, serve), 36812 (right), and 36893 (left), with uncertainty brackets of 36729–36732, 36811–36813, and 36892–36895. The review was recorded before opening the reserved model outputs.

The reserved baseline found 114 of 117 labelled ball centres within five reference pixels. At the three missed frames, 36934–36936, the ball remains visible against the floor, and the model produced consecutive candidates near the labelled centres with scores of 0.242, 0.265, and 0.268. The association stage left the frames empty. Evaluate the existing short gap refinement against a new reserved segment with similar weak evidence before changing association thresholds. Frame 36953 is marked absent, but the prediction aligns with the visible ball in the decoded image.

| Reserved ball tracking, 960 frames | Baseline | Sparse idle search |
| --- | ---: | ---: |
| Runtime | 559.0 s | 333.2 s |
| Neural calls | 971 | 632 |
| Labelled centres within five reference pixels | 114/117 | 114/117 |
| Predictions on labelled absence | 1/1 | 1/1 |
| Observations across all frames | 280 | 278 |

The two runs used the same excerpt, model, and inference settings apart from the idle search flag. The sparse pass omitted two visible, unlabelled toss frames, 36676 and 36677, and slightly shifted two centres during reacquisition. The decoded frame grid matched, and predictions agreed on all labelled frames. The single sequential pair measures runtime on this host and clip.

Both reserved runs matched both returns at their reviewed frames and assigned the correct hitters. Both missed the serve at 36730, where the descending toss turned into a faster ball travelling to the right. There were no unmatched suggestions in the reviewed interval, which included four labelled bounces and three labelled net events. The left and right mappings were track IDs 3 and 2 in the reserved run, and 2 and 1 in development. The two clips share the same match and camera, and the contact sample is small.
