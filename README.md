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
