# Pose audit and player-axis shot evidence — 2026-11-09

## Player-axis implementation

Shot candidates now include `playerAxis` (current image-space endpoints, signed incoming/outgoing displacement projections, alignment, and reversal status) and `contactScore` (a heuristic ranking, not probability). The axis connects confident hip midpoints, falling back to box centers. All calculations scale normalized coordinates to a 1280×720 reference so portrait/diagonal axes are not distorted by normalized x/y units.

Exactly two eligible people must be present at the candidate frame, separated by at least 50 reference pixels. If player Locate mappings exist, only mapped tracks are eligible; locating both players excludes an official. Extra, missing, coincident, or ambiguous people leave the axis unavailable. No stale person coordinates are carried through gaps. The axis direction is arbitrary: reversing its endpoints changes both projection signs and leaves reversal/alignment unchanged.

Alignment is the smaller absolute cosine of incoming/outgoing movement with the player axis. Reversal requires opposite projection signs and at least two reference pixels of longitudinal movement on each side. Score is `.8 × alignment` for longitudinal reversals plus `.2` for unambiguous player proximity. It ranks turns within the existing .16-second consolidation window. Cross-axis and ball-only candidates remain visible, labelled with weaker/unavailable axis support. A .55 alignment threshold controls only the UI support label, not candidate inclusion. The full evidence is exported with method `player-axis-ball-reversal-v2`.

On the saved 720p rally, using current-frame tracked IDs 1 and 2, seven suggestions remain. Axis alignments range .599–.910; all reverse longitudinally. Consolidation shifts three suggestions by one frame compared with v1 (3.00→2.96, 3.72→3.68, 5.04→5.08). Those shifts are changes, not demonstrated timing improvements. No manually verified contact labels exist yet; this is not evidence of precision/recall or 95% shot coverage. Projective perspective, lateral hits, bounces and localization errors still require review.

## Qualitative pose audit

Inspected four candidate-contact neighborhoods: 2.56, 3.00, 5.04 and 6.72 seconds, each with its preceding and following frame, for both players (24 player-frames). Original 720p source pixels were cropped before display enlargement; the model was not rerun. This is a targeted qualitative audit, not a randomly sampled or manually landmarked accuracy benchmark. Candidate times are not independently verified contacts.

Findings from the generated sheets:

- Shoulders, torso/hip regions, and coarse stance generally follow the visible body across these selected frames. This supports exploratory position/recovery visualization, subject to later quantitative validation.
- At 2.52–2.60 s the near player's fast right-hand estimates appear displaced toward the forearm while scores remain about .85–.88. The far player's wrist estimates look more plausible in these frames. Neither judgment establishes exact center error without manual landmarks.
- At 2.96–3.04 s the near player makes a large blurred arm movement. At 3.00 s wrist points sit inward on the forearms rather than at the visible blurred hand extents. Scores are .74 (left) and .48 (right). A single score cutoff would not reliably remove both errors. This can distort elbow angles and apparent arm sequencing.
- At 5.00–5.08 s the far player's extended arm is visible but the wrist estimate appears to lag/inset relative to the hand, despite scores around .98–.99. The near player's legs/torso show a broad lunge plausibly, but wrist positions remain approximate.
- At 6.68–6.76 s torso/stance estimates appear plausible; exact contact-hand position is ambiguous in blur/overlap. Far-player lower legs also cross the table occlusion in these views. Model-inferred landmarks behind an obstruction are not independently observed anatomy even at high scores.

**Decision:** use pose for coarse movement visualization and weak contact-proximity evidence. Detailed wrist/elbow angles, exact contact position, power generation, and correctness judgments need stronger validation. Keep body confidence as uncalibrated score; do not present a skeleton as measured biomechanics. The new UI text explicitly notes that high-score blurred/occluded joints can be misplaced.

Next accuracy experiment should hand-label visible shoulders, elbows, wrists and hips on a small contact set, mark genuinely unresolvable joints as unavailable, and compare before/after a source-resolution player-crop pose pass. That crop pass is an experiment to measure, not an assumed improvement. No new pose model, training, or full 4K pass was run for this audit.

## Artifacts and reproduction

```powershell
experiments/ball_detection/.venv/Scripts/python.exe experiments/ball_detection/audit_pose_contacts.py 'C:/Users/liang/Downloads/Wang Chuqin vs Sora Matsushima Match Highlights #ITTFWorldCup 2026.mp4' experiments/ball_detection/output/body-landmarks-720p.json --times 2.56 3.00 5.04 6.72 --output experiments/ball_detection/output/pose-contact-audit
```

Ignored output includes `pose-contact-audit/contact-{64,75,126,168}.jpg` and `scores.json`; v2 shot evidence is in `shot-candidates-720p-axis-v2.json`. Audited point indices: 5/6 shoulders, 7/8 elbows, 9/10 wrists, 11/12 hips. Screenshots are visual inspection aids; no hand-entered anatomical ground truth was fabricated.

Tests cover horizontal, vertical and diagonal player axes, transverse turns, missing/extra/coincident people, restricted mapped IDs, stale human frames, and the existing ball-gap and pose-validation cases. Frontend build and lint pass. This audit adds no changes to the original reviewed ball labels or checkpoint weights.
