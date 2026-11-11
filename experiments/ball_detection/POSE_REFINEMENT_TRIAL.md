# Body landmark refinement trial

This trial compares the current RTMPose-M at 256×192, RTMPose-M with a mirrored crop pass, and RTMPose-L at 384×288. All three use the same saved person boxes, source pixels, crop padding, and track IDs. The larger model changes both network capacity and input resolution. The mirrored pass uses the smaller model twice, swaps anatomical left and right back, reverses the horizontal score bins, and averages the score distributions before decoding. [Official RTMPose exports](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose), [MMPose flip implementation](https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/utils/tta.py).

## Results

Reviewed six player-frames from each of the 720p and 4K clips. On unannotated crops, the assistant marked approximate anatomical centers for 54 visible elbows, wrists, and ankles. Eight joints were marked hidden and ten ambiguous. Candidate predictions were scored after the marks were recorded. This targeted development sample uses approximate visual references from a single reviewer.

| Variant | Visible within 5% of box height | Wrists within 5% | Ankles within 5% | Hidden joints drawn at the 0.35 score gate | Median CPU time per person |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTMPose-M | 48/54 | 14/16 | 20/20 | 8/8 | 39.5 ms |
| RTMPose-M with mirrored crop | 49/54 | 15/16 | 20/20 | 8/8 | 81.5 ms |
| RTMPose-L at 384×288 | 49/54 | 15/16 | 20/20 | 8/8 | 179.3 ms |

The only change to the 5% pass count was a borderline left wrist at 6.70 s in the 4K clip. Median error across joints changed by less than half a percent of box height within either clip. On the 720p subset the existing RTMPose-M placed five of five visible wrists within the cutoff, compared with one of five for whole-frame YOLO11n pose. That comparison includes the benefit of source-resolution player cropping. Timing is from a separate matched, warmed test on 12 person crops, with four CPU threads for each variant. Full-clip 4K timing was invalidated by laptop sleep.

The current RTMPose-M option remains the practical choice for full-rally review. Future comparisons should measure accuracy on the player's own camera angle, separately score hidden and ambiguous joints, and measure usable evidence per shot. Footwork and post-shot recovery can start with visible ankle tracks in a fixed view. Contact-time wrist analysis needs a visibility check and a review path for ambiguous frames before it can support coaching cues.

## Run a comparison

From `experiments/ball_detection`, reuse an existing matching player-tracks JSON. Omit `--times` for a continuous full-clip import, or specify contact neighborhoods for a quick sparse comparison.

```powershell
output/pose-env/Scripts/python.exe refine_pose.py 'clip.mp4' 'player-tracks.json' --flip-test --output output/pose-m-mirrored.json
output/pose-env/Scripts/python.exe refine_pose.py 'clip.mp4' 'player-tracks.json' --weights output/models/rtmpose-l-384.onnx --output output/pose-l.json
```

For RTMPose-L, use `end2end.onnx` from the official [384×288 model archive](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504.zip). The graph used here has SHA-256 `62425affc6b9edba8f8a65f55f5e278fec5d905d0c90578e0f8503bdf3954587`. Full RTMPose-L outputs for both clips passed the app's **Import player tracks** parser with all 983 source-frame rows, boxes, and track IDs preserved. Local experiment files, reference crops and marks, scoring script, and timing results are under `output/pose-resolution-study/`.
