# AV Perception: nuScenes Pipeline

Applying autonomous-vehicle perception techniques to the nuScenes dataset, with a focus on multi-camera sensor fusion and 3D scene understanding.

## Status: Work in Progress

Schema exploration, single-machine ingestion, Ray-parallelized ingestion, and the pixel→ego→world SE(3) projection (validated by reprojecting ground-truth 3D boxes and measuring IoU) are all done — see the Roadmap below for details, and [docs/index.html](docs/index.html) for the benchmark and correctness results. BEV visualization of fused points is next.

## Tech Stack

- Python
- nuScenes devkit
- NumPy / SciPy (spatial transforms)
- Pandas / Plotly (skew analysis + visualization)
- Ray (planned)

## Roadmap

- [x] **Phase 1 — Foundations:** studied the math this project depends on before writing any pipeline
  code — camera calibration / projective geometry (how a 3D point maps to a pixel) and SE(3)
  rigid-body transforms (representing and composing 3D poses/rotations)
- [ ] **Phase 2 — AV data ingestion + fusion** (in progress)
  - [x] **Schema exploration:** walked the [nuScenes schema](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md)
    via `nuscenes-devkit` (scenes, samples, sensor calibration, ego pose) to understand what's
    available and how the tables relate before deciding what to extract
  - [x] **Single-machine ingestion:** per sample, synced the 6 camera streams to the reference
    keyframe by nearest timestamp, thresholded (43.5ms) and visualized the resulting skew
    (histogram, scatter, per-camera box plot) to sanity-check the sync, and wrote the synced
    records plus ego pose + calibration out to JSON
  - [x] **Ray-parallelized ingestion:** re-ran that ingestion under Ray across a sweep of worker
    counts (1/2/4/8) with repeated timed runs, to measure actual throughput gains and where the
    fixed per-worker cost (interpreter startup, imports, `ray.init()`) stops paying off — see
    `benchmark.py` and the published report's Performance section
  - [x] **Pixel → ego → world SE(3) projection, validated:** implemented the transform chain
    (and its inverse, pixel → 3D) that moves a point between camera, ego, and world space, then
    validated it by reprojecting every ground-truth 3D box back into each camera it's visible in
    and comparing to nuScenes' own 2D box via IOU — see `evaluate_projections.py`,
    `visualize_projections.py`, and the published report's Correctness section
  - [ ] **BEV visualization of fused points** (next): load the actual LiDAR point clouds (ingestion
    currently only records `num_lidar_pts` per annotation, not the points themselves), project
    them into the already-validated ego/world frame alongside the camera-derived 3D boxes, and
    render the combined result as a top-down (bird's-eye-view) plot per sample — reusing the
    projection chain from the phase above as the thing being sanity-checked visually, this time
    across sensors instead of within one camera
  - [ ] *(optional)* CAN bus EKF pose estimate vs. nuScenes ground-truth pose
- [ ] **Phase 3 — Scene graph + VLA ablation + eval:** Grounding DINO + SAM instance masks, 3D scene graph, LightEMMA baseline vs. scene-graph-context comparison
- [ ] **Phase 4 — ROS2 + Nav2 fundamentals**
- [ ] **Phase 5 — Home graph + Nav2 execution**
- [ ] **Phase 6 — Flywheel + polish**
- [ ] **Phase 7 — Closed-loop extension** (optional)

This is an active personal project, part of a broader self-directed study in autonomous systems perception.
