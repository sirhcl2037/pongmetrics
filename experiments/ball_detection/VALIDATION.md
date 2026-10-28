# Match clip validation

Video: `Wang Chuqin vs Sora Matsushima Match Highlights #ITTFWorldCup 2026.mp4`, 450268 bytes, 1280×720, 25 FPS, 8.76 s. Analysis stops at 8.28 s before the close-up cut. Median background uses only that segment.

Official RacketVision checkpoint SHA-256: `00d707b9db7a49561c411e4765956e79bcd7c7e20c7a0a535073440b3e972342`.

At heatmap threshold 0.5, the CPU experiment produced **141 accepted observations, 66 gaps, 207 frames** (68.1% coverage). Inference and review-sheet generation took about 168 s on this machine, roughly 1.23 analyzed frames/s. This is offline inference, not a realtime browser benchmark.

Qualitative review inspected all four generated contact sheets, sampled every fifth frame (42 frames). Detections align with the visible ball through successive exchanges, including the table/court positions at 2.0, 2.4, 2.8, 3.2, 4.0, 4.4, 4.8, 5.4 and 6.2 seconds. Visible gaps occur at e.g. 1.2, 1.4, 3.4, 4.2, 5.2, 5.8 and 6.0 s. Fast contact/occlusion frames remain difficult. In particular, contact-adjacent detections can overlap a racket or hand: accepted does not mean ground-truth correct. No dense hand-labeled evaluation was performed; coverage is not precision or recall. All frames and scores remain in `output/audit.json` for further review.

Browser integration was checked using this actual JSON and original clip: import succeeds, the dot coordinates at 2.0 s match the exported observation, seeking to the missing 1.4 s frame hides the dot, raising minimum score to 1 hides it, and playback advances the dot through multiple distinct coordinates. Mobile layout has no horizontal overflow and no runtime errors were observed. Tests also cover gaps, confidence thresholds, mismatched videos and duplicate timestamps.

The qualitative milestone is demonstrated, with gaps preserved. A proper labeled validation set and localization-error/false-positive measurements should precede using this as training ground truth or treating confidence as calibrated accuracy.

## ONNX decision

Investigated the model graph and official PyTorch/ONNX Runtime Web documentation. The operations are plausible export candidates; deployment has not been validated. First export a tensor-only wrapper and establish PyTorch/ONNX heatmap and centroid parity. Then benchmark WebGPU with these exact inputs and preprocessing, plus a WASM fallback. With ~46.6 MB weights and the measured CPU throughput, realtime browser performance cannot be assumed. Keep Python offline and the browser JSON-only until these checks pass. No ONNX dependency was added to the app.
