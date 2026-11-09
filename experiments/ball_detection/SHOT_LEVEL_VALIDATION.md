# Shot-level analysis review

The next benchmark is useful evidence per manually verified shot, rather than frame detection coverage. The UI now includes **Shot-level analysis review** below shot candidates. It reuses existing ball detections, player tracks/pose, player Locate mappings, manual shot tags, and session exports. No inference, training, new dependency, or backend is added.

## Review workflow

1. Load a video and import/generate ball observations and player tracks. Locate/name both players.
2. Choose a continuous fixed-camera interval containing roughly 10–20 contacts. Mark **every actual contact**, including detector misses, with the existing FH/BH and player controls. Pause and step to contact; choose Other if stroke subtype is unclear. Automatic source annotations are not benchmark references.
3. Export the session JSON to save manual tags and player mappings. Player/pose tracks retain their separate export.
4. Enter the interval bounds in Shot-level analysis review. Confirm that every contact was tagged. This confirmation is deliberately not assumed from a few annotations and resets when tags or bounds change. It is not saved across reloads.
5. Inspect recall, precision, missed contacts, unmatched suggestions, actor attribution, ball evidence and coarse screen movement. Jump to a manual contact to inspect its limitations. Export the shot analysis review, which includes references, observations, source player-track JSON, player mappings, settings, and the report for reproducibility. Dismissed candidate UI suggestions still participate in detector scoring; dismissing an error must not silently improve precision.

No real shot-level precision/recall is claimed yet: existing human ball-position labels are not complete contact ground truth. The candidate-contact neighborhoods inspected for pose are also not an independently verified contact dataset. The user-reviewed interval is needed before publishing those measurements.

## Matching and evidence

`frontend/src/shotAnalysis.ts` is independent of React. Ordered dynamic programming produces a one-to-one contact assignment, maximizing the number of matches within ±.12 s and then minimizing total absolute timing error. A nearest-first greedy strategy can consume the only eligible prediction for another shot. Candidates within .12 s outside interval edges can match an in-interval reference; unmatched outside-edge candidates are excluded from false positives. Empty denominators return null, displayed as a dash. These are contact-event metrics, not wing/stroke-classification metrics.

Reference actor identity requires a player ID assigned manually plus a current saved Locate mapping. Automatically inferred reference identities are not accepted as actor ground truth. Each match records correct/incorrect/unknown attribution; absent/ambiguous player mappings and missing candidate identities stay unknown.

Ball context currently means at least two eligible **distinct source frames** in each .12-second neighborhood before and after contact. It describes nearby evidence availability only: ball coordinates themselves are not verified by this check, and it does not establish a complete trajectory or placement analysis.

With an explicitly confirmed fixed camera, player-box center displacement is measured for roughly .3 seconds after each manual contact, normalized by that player's contact-frame box width. It requires the same mapped track at both endpoints, at least 80% observed frames, no missing run longer than two frames, and an interval long enough to include the window. Missing data produces null rather than interpolated movement. Human-tagged shots can have usable movement evidence even when the automatic contact detector missed them.

This is screen displacement in box widths, not meters, true velocity, feet motion, readiness, or recovery quality. Box deformation from arm motion and perspective can change the result. The code does not discard inferred joints merely because they could be occluded, nor does it assume those joints are accurate. This first movement measure uses boxes; pose remains visual evidence pending validation.

## Feedback categories

- **Footwork & Recovery:** an evidence preview showing post-contact screen movement and missing-data reasons.
- **Stroke Mechanics:** visual pose review remains available; detailed joint-angle/power feedback needs validation.
- **Tactical Choices:** requires verified ball outcomes and intended targets before recommendations.

The 95% analyzable-shot objective remains a product target. None of the counts here is labelled as validated coaching coverage. Later, specific feedback metrics should define their own success criteria and be checked against human judgments.

Tests cover non-greedy one-to-one matching, duplicate suggestions, partial/empty reference data, interval boundary matching, wrong/correct actor assignment, missing people, manually unassigned tracks, fixed-camera gating, shortened/gappy movement windows, and separation of manual contact ground truth from automatic suggestions. Existing pose/trail/calibration tests remain in the suite.
