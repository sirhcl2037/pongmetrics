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

Upload a match and use **Mark start**, **Mark end**, and **Save rally** below the video. Select **Tag shots** to name the players and mark both players' contacts. Use **Rally setup and notes** to add another rally, adjust a range, or record a practice cue.

Use **Export JSON** to save the review alongside the original video. Upload that video and **Import JSON** to resume. Rally review works with the video alone.

## Review recovery

Start with one point from a fixed view with your feet visible. Name both players, tag their contacts, and save the rally. In **Coach**, select yourself and click **Review next recovery** to play the next unreviewed recovery across saved rallies. Replay runs at half speed and freezes at the next tagged opponent contact. Record **Ready** if you were balanced and prepared to move, **Late** if you were still recovering, or **Unclear** if you cannot tell. Start with 10–20 points.

If you have ball and player tracks, locate both players and use **Review suggested contacts** to jump to possible contacts before tagging. Suggestions across gaps include the time range bracketed by observed ball positions.

Judgments are manual, even when the contact tags came from an automatic source. Missing, unassigned, repeated-player, or simultaneous contacts never form a guessed recovery window. Tag corrections invalidate affected judgments without erasing their saved snapshots. The summary excludes unclear and unreviewed cases from the late percentage and reports missing contacts separately. It covers tagged shots in saved rallies, not all returns in the video.

Jump to late examples across rallies and save the suggested practice cue when late recovery recurs. **Compare with another session** opens a previous export without replacing your work; select the same person explicitly. Use comparable footage and the same judgment definition. **Export JSON** preserves judgments and practice notes; export before reloading or changing videos. Older session files remain supported.
