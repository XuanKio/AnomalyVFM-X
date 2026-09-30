# Single-image AI tiling comparison

Scope: implement a reusable full-image + overlapping tile inference method and a reproducible CLI comparison on exactly the first AI-generated aluminum crack image. Existing web/CLI defaults and dirty user files remain intact except documented additions. No training, downloads, publishing or commits needed for this local experiment. Workflow uses local execution with no Plane integration.

Project: Python/PyTorch local anomaly demo. Reuse demo_fast loading/configuration, existing BF16 checkpoint, and demo_visuals fixed color rendering. Contract: float anomaly maps, image score distinct from segmentation, scores not probabilities. Runtime Python 3.10 / torch 2.10.0 / torchvision 0.25.0 already installed.

1. Freeze input and experiment configuration before inference: input image SHA-256, input 672, source tile 672, overlap .25, mean kernel 5 for both methods, global weight .25 and tile weight .75, fixed display threshold .5. One first generated image, no parameter search.
2. Implement demo_tiling.py with full coverage tile coordinates including small/rectangular images, positive feather weights, sequential inference and CPU accumulation. Baseline map matches web math on original-sized images; global score retained separately. Return baseline, tiled and fused float maps plus metadata. Preserve aspect handling of each baseline/crop transform, no change to checkpoint.
3. Implement compare_tiling.py runner using cached model, warm-up, repeated baseline and tiled comparison timing, source-coordinate maps, fair fixed-scale whole-image and zoom panels, metadata/report and local HTML. Pre-annotated rectangle is only a rough human location reference, never pixel GT. Report point maximum in/out rectangle and mask coverage descriptively, not F1/IoU/accuracy.
4. Tests: coverage, coordinates, feather invariance, constant field reconstruction, identical-map fusion, bounds/input validation; run existing tests and CLI smoke. Review new code. Do not change parameters after seeing the prediction to improve the headline.
5. Summarize actual observed outcome including failure or regression and latency tradeoff. Update docs/DEMO.md and README for the new opt-in CLI. Keep assets in ignored outputs.

Acceptance: reproducible baseline/proposed artifacts on the identical SHA-256 image, no GT leakage into inference, finite aligned maps, same color scale and threshold, all tests pass, actual real-model run completes on local GPU, conclusion does not generalize from n=1.

Plan review: inspected root/docs constraints and current Runtime.predict math; chosen additive CLI experiment avoids touching web API and existing defaults. Root owns comparison runner/docs/output; implementer owns demo_tiling.py and tests/test_demo_tiling.py. Independent review before final delivery.
