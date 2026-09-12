# AV Perception: nuScenes Pipeline

Applying autonomous-vehicle perception techniques to the nuScenes dataset, with a focus on multi-camera sensor fusion and 3D scene understanding.

## Status: Work in Progress

Working against `nuscenes-mini` (v1.0) in `nuscenes.ipynb`. Schema exploration is done and single-machine ingestion is underway: sample data is walked per scene, per-sensor timestamp skew against the sample keyframe is computed and thresholded (43.5ms), and the synced records are written out to JSON, with skew-distribution plots (histogram, scatter, per-camera box plot) to sanity-check the sync. Ray-based parallelization and the pixel→ego→world projection are next.

## Tech Stack

- Python
- nuScenes devkit
- NumPy / SciPy (spatial transforms)
- Pandas / Plotly (skew analysis + visualization)
- Ray (planned)

## Roadmap

- [x] **Phase 1 — Foundations:** studied the math this project depends on — camera calibration / projective geometry (how a 3D point maps to a pixel) and SE(3) rigid-body transforms (representing and composing 3D poses/rotations)
- [ ] **Phase 2 — AV data ingestion + fusion** (in progress)
  - [x] Explore the nuScenes schema via `nuscenes-devkit` (scenes, samples, sensor calibration, ego pose)
  - [x] Single-machine ingestion: sync the 6 camera streams by nearest timestamp, threshold + visualize skew, extract ego pose + calibration to JSON
  - [ ] Ray-parallelized ingestion, with 1-worker vs. N-worker throughput benchmark
  - [ ] Pixel → ego → world SE(3) projection, validated by reprojecting ground-truth 3D boxes and measuring IoU
  - [ ] BEV visualization of fused points
  - [ ] *(optional)* CAN bus EKF pose estimate vs. nuScenes ground-truth pose
- [ ] **Phase 3 — Scene graph + VLA ablation + eval:** Grounding DINO + SAM instance masks, 3D scene graph, LightEMMA baseline vs. scene-graph-context comparison
- [ ] **Phase 4 — ROS2 + Nav2 fundamentals**
- [ ] **Phase 5 — Home graph + Nav2 execution**
- [ ] **Phase 6 — Flywheel + polish**
- [ ] **Phase 7 — Closed-loop extension** (optional)

This is an active personal project, part of a broader self-directed study in autonomous systems perception.
