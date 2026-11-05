<!--
DRAFT (Claude): the "Key ideas" and "What took longest to click" sections are
written in your voice from what the code shows you understood -- rewrite
them in your own words; that's what makes this page worth reading. In
particular, "What took longest to click" is a guess based on where the code
has the most careful comments, not something you told me.
-->
# Learning log

What I studied for each phase, the ideas I took from it, and where those ideas show up in
the code. The [build log](build.html) covers the same ground from the other direction:
what I built, and what building it taught me.
{: .lede }

## Phase 1: Foundations <span class="badge done">Complete</span>

Before writing any pipeline code: the geometry every later stage depends on: how a 3D
point becomes a pixel, and how to represent and compose 3D poses.

### Studied

**Camera intrinsics/extrinsics and projective geometry**: Cyrill Stachniss,
[*Photogrammetry I & II*](https://www.youtube.com/playlist?list=PLgnQpQtFTOGRYjqjdZxTEQPZuFHQa7O7Y)
(University of Bonn)

- [Camera Parameters – Extrinsics and Intrinsics](https://www.youtube.com/watch?v=uHApDqH-8UE)
- [Camera Calibration using Zhang's Method](https://www.youtube.com/watch?v=-9He7Nu3u8s)
- [Direct Linear Transform for Camera Calibration and Localization](https://www.youtube.com/watch?v=3NcQbZu6xt8)
- [Intrinsic Camera Calibration – 5 Minutes with Cyrill](https://www.youtube.com/watch?v=26nV4oDLiqc)

**SE(3) rigid-body transforms**: Lynch & Park, *Modern Robotics*,
[Chapter 3](https://modernrobotics.northwestern.edu/chapters/chapter3/) (Northwestern)

**Hands-on:** rotation and rigid-transform math in Python with
[`scipy.spatial.transform`](https://docs.scipy.org/doc/scipy/reference/spatial.transform.html)

### Key ideas

- **A camera is two transforms.** The extrinsics are a rigid-body transform (rotation +
  translation) taking a point into the camera's own frame; the intrinsics matrix \(K\) then
  maps that camera-frame point onto the image plane. Only the second step involves the
  perspective divide; the first is pure SE(3).
- **Projection only works in front of the camera.** After the extrinsic transform, a point
  with \(z \le 0\) is behind the image plane, and dividing by \(z\) gives a meaningless (or
  mirrored) pixel. Those points have to be filtered out *before* the divide, not after.
- **Frame order is the whole game.** nuScenes stores every pose as "this frame, expressed in
  its parent": sensor in ego, ego in world. Going the other way (world → ego → camera, to
  reproject an annotation) means applying the *inverse* of each, in reverse order. Getting one
  inverse wrong still produces plausible-looking numbers.
- **Rotations have conventions that silently disagree.** nuScenes writes quaternions
  scalar-first \((w, x, y, z)\); `scipy` defaults to scalar-last. Reading one as the other
  doesn't crash; it just produces a different, wrong rotation.

### Where it shows up

- [`transform.py`](https://github.com/Atmika293/av-scene-graph-planning/blob/main/src/av_data_pipeline/transform.py)
  chains global → ego → camera → pixel with `RigidTransform`, reads every nuScenes quaternion with
  `scalar_first=True`, and drops points with \(z \le 0\) before multiplying by \(K\).
- [`check_orientation.py`](https://github.com/Atmika293/av-scene-graph-planning/blob/main/scripts/validation/check_orientation.py)
  exists because of the last two points: it checks the rotation handling against how the car
  actually moved, rather than trusting it.

### What took longest to click

Keeping track of which direction each stored transform points. Writing it out as
"*A*-in-*B*" for every pose, and checking each stage of the chain separately against the
devkit's own intermediate output, is what made it manageable.

## Phase 2: Ingestion + fusion <span class="badge progress">In progress</span>

Phase 2 is mostly building (see the [build log](build.html)); the learning here is the
working references used along the way, plus a reading track on BEV *networks*, the learned
counterpart to the geometric BEV built in this phase.

### Working references

- [`nuscenes-devkit` tutorial](https://github.com/nutonomy/nuscenes-devkit) and the
  [nuScenes schema](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md)
  (worked through on `v1.0-mini` before deciding what to extract)
- Ray docs: [Tasks](https://docs.ray.io/en/latest/ray-core/tasks.html),
  [Actors](https://docs.ray.io/en/latest/ray-core/actors.html),
  [Objects](https://docs.ray.io/en/latest/ray-core/objects.html), the
  [design patterns and anti-patterns index](https://docs.ray.io/en/latest/ray-core/patterns/index.html)
  (especially [passing the same large argument by value](https://docs.ray.io/en/latest/ray-core/patterns/pass-large-arg-by-value.html)
  and [limiting pending tasks with `ray.wait`](https://docs.ray.io/en/latest/ray-core/patterns/limit-pending-tasks.html)),
  and [A Gentle Introduction to Ray Core by Example](https://docs.ray.io/en/latest/ray-core/examples/gentle_walkthrough.html)

### Key ideas

- **A Ray task runs in its own process, not a thread.** Nothing in the driver, not even an
  already-loaded `nusc`, is visible inside a task; only what's explicitly passed in (and
  round-tripped through serialization) exists on the worker side. Most of my early design
  mistakes (shipping a global reference, assuming one task could see what another had just
  written) turned out to be this one fact showing up in different disguises.
- **The object store's "zero-copy" guarantee is narrower than it sounds.** `ray.put()` avoids
  re-serializing the same argument for every task that uses it, but each worker still unpickles
  its own full copy unless the data is numpy- or Arrow-backed. A large nested Python object
  (a loaded `nusc`, say) gets copied per worker either way; `ray.put()` saves redundant
  *serialization*, not redundant *memory*.

- **Granularity is a real tradeoff, not "more parallel is always better."** Too fine (one task
  per sample, with ~40 samples per scene) pays fixed per-task scheduling overhead for too
  little work each; too coarse (a single-threaded driver doing all the extraction itself)
  leaves the part that's actually worth parallelizing (the geometry and the writes) stuck
  waiting behind it. Batching scenes landed in between.
  
- **Shared mutable state needs an actor, and an actor is a new bottleneck.** Plain tasks are
  stateless, so a second task can't see what a first task wrote into some shared buffer (that needs an actor). But then every producer serializes behind that one actor's inbox, which
  just moves the bottleneck rather than removing it. Giving each task full ownership of its own
  output (build and write its own rows, no coordination) sidesteps the problem instead.

- **Aggregating a cell means deciding what "one value" should mean, first.** When several
  lidar points land in the same raster cell, they could be repeated measurements of the same
  physical spot (a multi-return artifact; average them, since that reduces measurement
  noise) or they could be different surfaces that happen to project into the same cell (e.g. a
  tree branch over a road. Averaging blends two unrelated realities into a number that
  describes neither). The fix wasn't picking a better tie-break rule; it was checking, from
  the actual returns, which of those two situations was really happening before deciding
  whether to average at all.
- **A color scale is a choice, not a given.** Mapping a skewed, long-tailed value (lidar
  intensity here) onto a fixed range by its literal min/max lets a handful of rare outliers
  dictate the whole scale, compressing the typical values (the ones that actually carry the
  scene's structure) into a narrow, almost-indistinguishable band. Clipping at a percentile
  instead (letting the rare extremes simply saturate) trades fine detail among the outliers
  for real contrast across the data that matters. Remote-sensing imagery calls this a
  "percentile stretch."

### Where it shows up

- `partition.py`'s manifest pattern: each batch's scenes are resolved into a plain-dict
  manifest (pure list indexing, since `NuScenes.__init__` already loads every table eagerly)
  *in the driver*, and only that resolved manifest (never `nusc` itself) crosses into the
  two Ray tasks it's handed to.
- [`bev/raster.py`](https://github.com/Atmika293/av-scene-graph-planning/blob/main/src/bev/raster.py)'s
  `build_raster` averages a cell's nearest-to-ego points only when they're tied at the
  *exact* same distance (a real multi-return case, confirmed against actual data), not
  whenever multiple points simply happen to share a cell.
- [`build_bev_report.py`](https://github.com/Atmika293/av-scene-graph-planning/blob/main/scripts/report/build_bev_report.py)'s
  `raster_intensity_traces`/`scatter_3d_intensity_traces` clip the color scale at the 95th
  percentile of each sample's actual intensity values, rather than the sensor's theoretical
  0–255 range.

### Reading: BEV networks

The BEV built in this phase is *geometric*: LiDAR already says where every point is in 3D,
so making a top-down view is a matter of projecting points and colouring them from the
cameras. BEV *networks* solve the hard version: building a top-down representation from
cameras alone, where depth has to be learned rather than measured. This is the dominant
approach in current AV perception, so it's worth knowing where my geometric version sits
relative to it.

| Paper | Status |
|---|---|
| **Lift-Splat-Shoot**, Philion & Fidler, [ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590188.pdf) | <span class="badge progress">Reading</span> |
| **BEVFormer**, Li et al., ECCV 2022 | <span class="badge todo">Up next</span> |
| **BEVFusion**, Liang et al., NeurIPS 2022 | <span class="badge todo">Up next</span> |
| **Survey: Delving into the Devils of Bird's-eye-view Perception**, T-PAMI 2023, [arXiv 2209.05324](https://arxiv.org/abs/2209.05324) | <span class="badge todo">Up next</span> |

<!-- DRAFT (Claude): placeholder notes on LSS -- replace with your own once you've finished it. -->
#### Notes so far: Lift-Splat-Shoot

- **Lift:** each pixel is lifted into a frustum of points along its ray (one per discrete
  depth bin), weighted by a predicted depth distribution. Where my pipeline gets depth exactly
  from LiDAR, LSS has to *predict* it, and hedges across depths when unsure.
- **Splat:** the lifted features from all six cameras are sum-pooled into a shared BEV grid
  of pillars: the learned counterpart of my raster's binning step.
- **Shoot:** planning by scoring a set of template trajectories against the BEV cost map.
  Relevant later, in Phase 3's trajectory evaluation.
