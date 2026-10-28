# Offline ball detection experiment

The browser stays a local video viewer. Python runs a real pretrained [RacketVision TrackNetV3](https://github.com/OrcustD/RacketVision) model; it writes JSON that PongMetrics imports. It is not a backend dependency. Model/source downloads and the Python environment are ignored by Git.

## Reproduce (Python 3.12, PowerShell, repository root)

```powershell
python -m venv experiments/ball_detection/.venv
experiments/ball_detection/.venv/Scripts/python.exe -m pip install -r experiments/ball_detection/requirements.txt
experiments/ball_detection/.venv/Scripts/python.exe experiments/ball_detection/setup.py
experiments/ball_detection/.venv/Scripts/python.exe experiments/ball_detection/detect.py 'path/to/clip.mp4' --end 8.28
experiments/ball_detection/.venv/Scripts/python.exe experiments/ball_detection/render_review.py 'path/to/clip.mp4'
```

`--end` limits analysis/background estimation to a fixed-camera rally before the test clip's close-up cut. Change it for other clips. Input is kept locally. The script stores the checkpoint SHA-256 in its output. The source is pinned; the publisher's checkpoint URL uses main, so compare that hash when reproducing an existing run.

See [VALIDATION.md](VALIDATION.md) for the actual run and limitations. `render_review.py` creates a silent GIF directly from the saved audit and decoded video for visual review without repeating inference.

## Data and timing

Input is BGR 512×288, median background plus four frames (15 channels). Training uses the current frame and three preceding frames, repeating the first frame at the start. We follow that loader, rather than the repository inference helper's exclusive range ending one frame earlier. Model outputs are decoded as thresholded connected components. Exactly one candidate is accepted; no candidates or multiple candidates yield a gap. Scores are heatmap peaks, **not calibrated probabilities of correctness**. No interpolation, tracking heuristics, trajectory prediction, or InpaintNet fills gaps.

`output/detections.json` contains normalized image positions using `Observation<Point>`, actual decoder timestamps in seconds, FPS, matching video metadata, and model provenance. `audit.json` includes every analyzed frame and the reason for missing observations. Review sheets overlay sampled detections on the decoded video. Sampled sheets support qualitative inspection; they do not establish dataset precision/recall. Unreviewed frames are not ground truth.

Load the original clip in PongMetrics, choose **Import ball detections**, and select the JSON. The dot updates on decoded video frame callbacks. A short trail joins only consecutive confident detections, and vanishes at a missing frame. Raise **Min score** to hide weak detections; lowering it cannot recover candidates already rejected by Python. Normal session export/import retains these observations and their FPS. Nothing converts these image positions to 3D.

## ONNX investigation

The neural network is standard convolution/BatchNorm/ReLU/MaxPool/nearest Resize/Concat/Sigmoid, with input `[1,15,288,512]` and final heatmap `[1,1,288,512]`. A small wrapper around `_forward(x)[:, -1:]` can avoid MMEngine's training interface during export. The PyTorch state file is about 46.6 MB; browser activation memory and latency still need measurement.

After accuracy is accepted, export with [torch.onnx.export](https://docs.pytorch.org/docs/stable/onnx.html), compare heatmaps and decoded centers against PyTorch on these same frames, then benchmark [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) WebGPU and WASM. Browser preprocessing must reproduce BGR ordering, background and temporal alignment. Operator support is provider-specific. Do not change precision/quantize until position and gap parity have been measured. No ONNX runtime, model download, or inference workload has been added to the production bundle yet.
