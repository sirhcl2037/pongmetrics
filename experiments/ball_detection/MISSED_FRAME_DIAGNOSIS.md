# Missed-frame diagnosis — 2026-11-09

Audited the final `head-final-*-v1.json` runs using saved candidates and sequentially decoded source images. No neural inference, training, threshold changes, or label edits were performed. The 720p labels retain the reviewed -1-frame adjustment; 4K uses zero. Distance tolerance is 20 pixels at 1280×720 (60 source pixels at 4K).

| Evidence on missing visible frames | 720p | 4K | Total |
| --- | ---: | ---: | ---: |
| No retained candidates anywhere | 11 | 7 | 18 |
| Candidates exist, all outside label tolerance | 10 | 9 | 19 |
| Nearby evidence, all below .4 following cutoff | 3 | 4 | 7 |
| Nearby evidence at/above .4, rejected by tracking | 0 | 2 | 2 |
| Missing total | 24 | 22 | 46 |

The three additional 4K center errors are frames 523, 553, 651. There are no 720p center-tolerance failures among its retained observations. These categories describe saved evidence, not proven root causes. Candidate extraction uses a .15 floor, component centroids, and crop-border rejection; absence from the cache does not prove absence of a raw heatmap response. Frame 577's nearest candidate is 20.94 reference pixels away, just outside tolerance, so the 19 distant-candidate cases are not all necessarily wrong objects.

## Concrete findings

- **Most misses need better image evidence or extraction:** 37/46 have no saved candidate within label tolerance. 33/46 misses already used full-frame search or a full-frame rescue (720p 19, 4K 14); simply increasing crop size cannot explain or solve most failures.
- **One image-edge filtering issue:** 4K frame 727 (14.54 s) has its label outside the retained crop interior. The crop reaches the physical right image edge, but the unconditional four-model-pixel border exclusion rejects that edge too. All other 12 missing crop frames have their labels inside the retained interior. This is a concrete exclusion condition, not proof that the network emitted a recoverable candidate there. A small follow-up should exempt physical image boundaries and validate that frame with a targeted inference probe.
- **Weak evidence:** 720p frames 85, 187, 199; 4K 220, 221, 484, 578. These contain nearby candidates but none reach .4. Uniform threshold reduction previously introduced false detections; prefer bounded confirmation using actual neighboring image evidence, evaluated against negative frames.
- **Conservative restart at 4K 222:** nearby peaks score .564 and .534, below the .6 acquisition cutoff; there is no established track. This explains why .4 following eligibility alone does not produce an observation. This policy also prevented earlier logo false restarts.
- **Motion reversal/restart at 4K 579:** the .644 nearby candidate lies outside the stale motion gate and becomes a pending restart. Frame 580 scores .590, below .6, breaking the consecutive acquisition requirement. The offline pass recovers 580 but leaves 579 missing. Investigate bounded backward confirmation here before globally relaxing acquisition.
- **Visual patterns:** the contact sheets show many failures near hands/paddles and overlapping player bodies, elongated motion blur, and bright floor advertising. Examples: 720p 43–47 and 84–87; 4K 483–485, 552–553, 599–604. 4K 523/651 center failures occur against bright advertising; 553 has a long blurred streak. These are qualitative observations, not independently scored causal classes. Several labels near bodies are visually ambiguous at these display sizes; retain approximate-center tolerance.

## Recommended order

1. Fix the physical-edge border exclusion and investigate evidence-based backward confirmation at 579. These are small, specific tracking/extraction experiments; compare all existing negative frames before promotion.
2. For the dominant remaining failures, inspect raw heatmaps on a small selection of contact/blur/advertising frames to distinguish sub-.15 response from centroid/extraction problems. This requires only selected inference, not another full model sweep.
3. If raw responses are inadequate, fine-tune a final decoder block using these difficult sequences plus normal and negative examples. Keep an unseen rally for evaluation. Treat accepted detector centers as approximate reviewed supervision.
4. Optimize runtime after selecting the accuracy change; preserve explicit gaps without image evidence.

## Reproduce

Run `experiments/ball_detection/.venv/Scripts/python.exe experiments/ball_detection/diagnose_misses.py` from the repository root. It reads the original Downloads videos/labels and saved final runs. Generated outputs (ignored by Git):

- `output/missed-diagnosis.json`: all 49 failure records, scores, search modes, predictions and crop checks.
- `output/missed-diagnosis-720p.jpg`: all 24 missing frames.
- `output/missed-diagnosis-4k.jpg`: all 22 missing frames plus three center errors.

Green crosses mark reviewed centers, cyan circles mark candidates within tolerance, yellow circles mark final observations. The crops use the same reference field of view across resolutions; enlarged 720p images do not add source detail.
