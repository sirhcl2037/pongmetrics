# PongMetrics

A web-based table tennis analytics platform designed to analyze match footage, identify player tendencies, and generate detailed performance profiles.

## Planned Features

- **Video Analysis:** Analyze uploaded table tennis match recordings.
- **PongDNA:** Visualize player styles across forehand–backhand and spin–speed dimensions.
- **Player Statistics:** Track shot selection, consistency, and tactical patterns.
- **Machine Learning:** Explore shot prediction, playing-style classification, and skill estimation.

## Tech Stack

**Frontend:** React, TypeScript  
**Backend:** Python, FastAPI  
**Analytics & ML:** NumPy, scikit-learn

## Status

In development.

Local video annotation and offline player/ball tracking are implemented. See [temporal ball tracking](experiments/ball_detection/TRACKING.md) and [player tracking](experiments/ball_detection/PLAYERS.md) for setup, measured results, and reproducible commands. Tracking generation runs locally through Vite; the deployed static app can import generated files.

## Review a match

Upload a match and use **Mark start**, **Mark end**, and **Save rally** below the video. Select a rally to jump to it, replay the marked range, and open any tagged shots inside it. Record the point winner and one training focus or cue. The focus totals show which themes recur across the rallies you reviewed.

Use **Export JSON** to save the review alongside the original video. Upload that video and **Import JSON** to resume. Rally review works with the video alone.

## Review recovery

Start with 10–20 points from a fixed view with your feet visible. Name both players, tag their contacts, and save each rally. In **Recovery review**, select yourself and a shot. **Replay recovery** plays from your contact through the next tagged opponent contact; **Opponent contact** jumps to the decision frame. Record **Ready**, **Late**, or **Unclear** (with a reason). Ready means balanced and prepared to move at the opponent’s contact; late means visibly still recovering.

Judgments are manual, even when the contact tags came from an automatic source. Missing, unassigned, repeated-player, or simultaneous contacts never form a guessed recovery window. Tag corrections invalidate affected judgments without erasing their saved snapshots. The summary excludes unclear and unreviewed cases from the late percentage and reports missing contacts separately. It covers tagged shots in saved rallies, not all returns in the video.

Jump to late examples across rallies and save the suggested practice cue when late recovery recurs. **Compare with another session** opens a previous export without replacing your work; select the same person explicitly. Use comparable footage and the same judgment definition. **Export JSON** preserves judgments and practice notes; export before reloading or changing videos. Older session files remain supported.
