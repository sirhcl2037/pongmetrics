# Ball inference speed

The local tracker can use a float32 ONNX CPU export of the selected checkpoint and head adapter. `temporal_track.py` defaults to `--backend auto`. A matching local validation manifest selects ONNX. Other installations use PyTorch until their export is validated. `--backend torch` reproduces the eager runtime and `--backend onnx` explicitly selects an export for comparison.

The current machine uses ONNX Runtime 1.23.2 with eight inference threads. The standard detector previously used four PyTorch threads. Exported tracks record the runtime and thread count.

## Measurements

Paired runs of the same 30 frames from the 720p clip, covering 2.0–3.2 seconds, produced 28 identical observation centers and timestamps with identical tracking decisions.

| Stage | PyTorch, four threads | ONNX, eight threads |
| --- | --- | --- |
| Initialization | 13.08 s | 1.13 s |
| Neural inference | 23.70 s | 11.00 s |
| Total tracking process | 37.30 s | 12.67 s |

This single comparison measured a 2.94× overall speedup and 2.16× inference speedup. Timing excludes interpreter startup. Lazy PyTorch imports contribute to the shorter initialization. These measurements depend on host load and clip content.

A separate warmed microbenchmark used four real frames at both supported input sizes. Median full-frame inference fell from 1804 to 783 ms, and crop inference from 546 to 216 ms. The largest heatmap difference was 1.77e-6, with identical threshold masks at 0.15 and 0.5.

Background preparation now calls OpenCV `grab()` sequentially and retrieves BGR images only for selected median samples. Paired preparation of both standard and native-zoom backgrounds fell from 28.07 to 14.04 seconds on the 4K clip. All sampled image hashes and decoded durations matched. The sequential scan still corrects the MKV's underreported duration.

## Accuracy checks

Both complete reviewed clips were compared against the stored PyTorch outputs. Across 983 decoded frames, all 639 emitted observations retained exactly the same centers and timestamps. Tracking gaps, search schedules, and track IDs matched. The maximum confidence difference was 1.05e-6, and no UI confidence step changed.

These results cover the two reviewed clips. Full-clip 4K timing is excluded because the laptop slept during that run. Raw measurements and track comparisons are stored locally under `output/ball-speed/`.

## Reproduce

Run from `experiments/ball_detection` with the existing ball environment and checkpoint installed.

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-ball-onnx.txt
.venv/Scripts/python.exe export_ball_onnx.py
.venv/Scripts/python.exe temporal_track.py --video 'path/to/clip.mp4' --backend torch --output output/reference.json
.venv/Scripts/python.exe temporal_track.py --video 'path/to/clip.mp4' --backend onnx --output output/onnx.json
.venv/Scripts/python.exe verify_ball_backend.py --pair output/reference.json output/onnx.json --reference-adapter adapters/review-head-v1.json --candidate-adapter adapters/review-head-v1.json --output output/parity.json --prefer-onnx
```

Repeat `--pair` to validate multiple clips before selection. The checker requires matching model settings and adapter coefficients, exact observation centers and timestamps, identical tracking decisions, and confidence agreement within 1e-5 without crossing UI steps. Local selection is bound to checkpoint, adapter, graph checksum, runtime version, and thread count. Re-exporting clears the selection until validation runs again. Model exports, environments, and measurement outputs remain local ignored files.
