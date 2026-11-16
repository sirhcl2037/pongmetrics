# Shot-level analysis review

**Advanced tools → Shot-level analysis review** measures contact detection and usable evidence against every manually tagged contact in a selected interval. It uses existing ball detections, player tracks, Locate mappings, and manual tags. **Review suggested contacts** below the video is the candidate inspection view.

## Review workflow

1. Load a video and import/generate ball observations and player tracks. Locate/name both players.
2. Choose a continuous fixed-camera interval containing roughly 10–20 contacts. Mark **every actual contact**, including detector misses, with the existing FH/BH and player controls. Pause and step to contact; choose Other if stroke subtype is unclear. Automatic source annotations are not benchmark references.
3. Export the session JSON to save manual tags and player mappings. Player/pose tracks retain their separate export.
4. Enter the interval bounds in Shot-level analysis review. Confirm that every contact was tagged. This confirmation is deliberately not assumed from a few annotations and resets when tags or bounds change. It is not saved across reloads.
5. Inspect recall, precision, missed contacts, unmatched suggestions, actor attribution, ball evidence and coarse screen movement. Jump to a manual contact to inspect its limitations. Export the shot analysis review, which includes references, observations, source player-track JSON, player mappings, settings, and the report for reproducibility. Dismissed candidate UI suggestions still participate in detector scoring; dismissing an error must not silently improve precision.

### Paired development audit

The following two broadcast excerpts were inspected frame by frame around visible swings and trajectory changes. The timestamps are approximate, especially where the racket or ball is occluded. These clips and their ball labels informed detector development. Coverage and timing still need validation on a held-out recording.

| Inputs at minimum ball score .15 | Detector | Matched within ±.12 s | Extra suggestions | Correct hitter among matches |
| --- | --- | --- | --- | --- |
| 720p, 25 fps, RTMPose tracks | v2 | 6/13 | 1 | 6/6 |
| Same 720p inputs | v3 | 13/13 | 1 | 13/13 |
| 4K, 50 fps, narrow RTMPose parent tracks | v2 | 10/11 | 0 | 8/10 |
| Same 4K inputs | v3 | 11/11 | 0 | 9/11 |
| 4K, full-width YOLO11n boxes, same ball track | v3 | 11/11 | 0 | 11/11 |

Eight newly matched contacts were inferred from short gaps. The 720p extra suggestion at 6.96 s is a net-area event with no visible player contact. Starting the review at .5 minimum ball score instead reduces the 720p v3 result to 10/13. The 4K full-width run keeps 11/11 at .5.

The audit used the local `output/gap-720p-selective-onnx.json` and `output/gap-4k-selective-onnx.json` tracks and the `output/body-landmarks-rtmpose-{720p,4k}.json` imports. The 4K full-width run used YOLO11n, 416 input size and the default full-width ROI. Reference contacts are listed as time and mapped track ID:

- 720p, 1.52–7.15 s: `1.72/1, 2.14/2, 2.54/1, 3.00/2, 3.36/1, 3.75/2, 4.14/1, 4.56/2, 5.04/1, 5.48/2, 5.90/1, 6.32/2, 6.74/1`.
- 4K, 6.00–11.24 s: `6.11/3, 6.77/4, 7.12/3, 7.54/4, 7.88/3, 8.36/4, 8.74/3, 9.20/4, 9.69/3, 10.15/4, 11.03/3`.

Replay a saved complete review with the current detector and an optional baseline module. The report uses the same observations, player mappings, minimum score, interval, and matching tolerance for both:

```powershell
$exp = 'experiments/ball_detection'
git show f7d2d98:frontend/src/shotDetection.ts | Set-Content -Encoding utf8 "$exp/output/shotDetection-v2.ts"
node --experimental-strip-types "$exp/compare_shot_detectors.mjs" 'path/to/clip.shot-analysis-review.json' "$exp/output/shotDetection-v2.ts"
```

### Contact differences in v3

Visible ball turns beyond both located players' estimated reach are rejected when both boxes are present. A wrist estimate can strengthen support within plausible reach. A high wrist score by itself establishes neither a hitter nor joint visibility.

When both track IDs are located, two observed flight segments can suggest a candidate within a ball gap up to 0.2 s. The projected flight segments must come near each other at the same time, with exactly one mapped player within estimated reach. When both boxes are present the motion must also reverse along the player axis. Candidates carry the bounding timestamps and estimated fit error. The ball track retains its original samples and gaps. These estimates need video review and do not create shot tags.

## Matching and evidence

`frontend/src/shotAnalysis.ts` is independent of React. Ordered dynamic programming produces a one-to-one contact assignment, maximizing the number of matches within ±.12 s and then minimizing total absolute timing error. A nearest-first greedy strategy can consume the only eligible prediction for another shot. Candidates within .12 s outside interval edges can match an in-interval reference; unmatched outside-edge candidates are excluded from false positives. Empty denominators return null, displayed as a dash. These are contact-event metrics, not wing/stroke-classification metrics.

Reference actor identity requires a player ID assigned manually plus a current saved Locate mapping. Automatically inferred reference identities are not accepted as actor ground truth. Each match records correct/incorrect/unknown attribution; absent/ambiguous player mappings and missing candidate identities stay unknown.

Ball context means at least two eligible **distinct source frames** in each .12-second neighborhood before and after contact. The report scans this evidence in timestamp order. In a paired CPU microbenchmark with 16,200 observations and 1,300 references assembled by repeating the 720p excerpt, the median evidence run went from 23.0 ms to 1.2 ms across 12 alternating runs with the same per-shot evidence counts. This measurement excludes neural inference and candidate matching. Ball context is not a check of coordinate accuracy or placement.

With an explicitly confirmed fixed camera, player-box center displacement is measured for roughly .3 seconds after each manual contact, normalized by that player's contact-frame box width. It requires the same mapped track at both endpoints, at least 80% observed frames, no missing run longer than two frames, and an interval long enough to include the window. Missing data produces null rather than interpolated movement. Human-tagged shots can have usable movement evidence even when the automatic contact detector missed them.

This is screen displacement in box widths, not meters, true velocity, feet motion, readiness, or recovery quality. Box deformation from arm motion and perspective can change the result. The code does not discard inferred joints merely because they could be occluded, nor does it assume those joints are accurate. This first movement measure uses boxes; pose remains visual evidence pending validation.

## Feedback categories

- **Footwork & Recovery:** an evidence preview showing post-contact screen movement and missing-data reasons.
- **Stroke Mechanics:** visual pose review remains available; detailed joint-angle/power feedback needs validation.
- **Tactical Choices:** requires verified ball outcomes and intended targets before recommendations.

The 95% analyzable-shot objective remains a product target. None of the counts here is labelled as validated coaching coverage. Later, specific feedback metrics should define their own success criteria and be checked against human judgments.

Tests cover non-greedy one-to-one matching, duplicate suggestions, partial/empty reference data, interval boundary matching, wrong/correct actor assignment, missing people, manually unassigned tracks, fixed-camera gating, shortened/gappy movement windows, and separation of manual contact ground truth from automatic suggestions. Existing pose/trail/calibration tests remain in the suite.
