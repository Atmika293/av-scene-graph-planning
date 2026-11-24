<!--
DRAFT (Claude): the "What I learned" paragraphs are written from the commit
messages and the code's own comments -- they're the most personal part of
this page, so rewrite them in your own words. Facts (what was built, what
broke, what the checks found) are taken from commits/docstrings; numbers are
{{stat:...}} directives; dates are {{dates:...}} directives computed from the
feat:/fix: commits touching each milestone's own files (see report/markup.py).
-->
# Build log

Each milestone in the order it was built: the goal, the approach, how I checked it, and what
it taught me. Dates come from the commit history.
{: .lede }

## 1 · Schema exploration and camera sync <span class="badge done">Done</span> {: #schema }
{{dates: nuscenes.ipynb}} · [notebook](https://github.com/Atmika293/av-scene-graph-planning/blob/main/nuscenes.ipynb)
{: .meta }
<div class="tags"><span class="tag">nuScenes devkit</span> <span class="tag">Pandas</span> <span class="tag">Matplotlib</span></div>

**Goal.** Understand what nuScenes actually contains, and how its tables relate, before
deciding what to extract.

**Approach.** Worked through the devkit tutorial on `v1.0-mini` (scenes, samples,
sample_data, calibrated sensors, ego pose, annotations, instances, categories), then added my
own extraction on top: walk each scene's samples, pull every sensor's data with its
calibration, and measure each camera's timestamp skew against the sample's keyframe.

**How I checked it.** Thresholded skew at 43.5 ms and plotted it three ways (a histogram, a
per-sample scatter with the threshold line, and a per-camera box plot) to see whether
misalignment was rare noise or one camera being systematically late.

**What I learned.** A nuScenes "sample" is a keyframe, not a moment where every sensor fired
together: each camera's frame is the nearest one to the keyframe, so "synced" always means
"synced within some tolerance", and that tolerance is a choice worth measuring before
building on it.

## 2 · Ingestion module and Parquet tables <span class="badge done">Done</span> {: #ingestion }
<!-- Hand-dated: later fix:/feat: commits to ingestion.py and the ingestion checks (the
duplicate last sample, the whole-export check) were made for the full-dataset run. -->
Sep 13 – Nov 1, 2026 · [checks](validation.html#ingestion)
{: .meta }
<div class="tags"><span class="tag">PyArrow</span> <span class="tag">Schema design</span></div>

**Goal.** Turn the notebook's extraction into a module that writes typed Parquet tables
(samples, sensor data, annotations, projected 2D boxes).

**Approach.** Moved the logic out of the notebook, then replaced every literal table/field
name with a shared key vocabulary (`nuscenes.py`, `ingestion_keys.py`) so ingestion, transform
and table-building all read the same names. Row builders flatten each sample's aggregate record
into rows validated against explicit `pyarrow` schemas.

**How I checked it.** Three scripts, each against something independent of the code under
test: sample traversal against nuScenes' own table scan, every ingested field against the raw
devkit record ({{stat:ingestion_keys.verified|,}} values), and every table row against the
schema plus a small re-indexer written separately from the row builders
({{stat:table_rows.verified|,}} values).

**What broke.** Running the checks found real bugs: a key renamed in one place but not its
callers (it crashed `verify_ingestion_keys.py`), schema literals that weren't valid
`pyarrow` schemas, annotation rows built from the wrong column map, and a call to a function
that didn't exist. All were fixed in one pass, and the checks have passed since.

**What I learned.** The bugs that matter in data plumbing are wiring bugs: the right logic
fed the wrong record. They look fine until something independent reads the same source.
That's why the table check uses its own re-indexer rather than the code it's checking: reusing
it would have hidden exactly these bugs.

## 3 · Projection: global → ego → camera → pixel <span class="badge done">Done</span> {: #projection }
{{dates: src/av_scene_graph_planning/transform.py scripts/export_2d_annotations_debug.py scripts/validation/export_2d_annotations_debug.py scripts/evaluate_projections.py scripts/validation/evaluate_projections.py scripts/visualize_projections.py scripts/viz/visualize_projections.py scripts/build_correctness_report.py scripts/report/build_correctness_report.py}} · [checks](validation.html#projection) · [analysis](result-projection-accuracy.html)
{: .meta }
<div class="tags"><span class="tag">SE(3) / SciPy</span> <span class="tag">Computer vision</span></div>

**Goal.** Project every annotated 3D box into each camera that sees it, and prove the
projection is right.

**Approach.** An SE(3) chain built on `scipy`'s `RigidTransform`: box corners from global to
ego to camera frame, an in-front-of-camera filter, the intrinsics and perspective divide, then
clipping the projected hull to the image canvas.

**How I checked it.** nuScenes ships a script that does the same projection, but it only
outputs final boxes, so I forked it to also output the corners after every intermediate
stage, and diffed my pipeline against it stage by stage. A disagreement now points at the
exact transform that caused it. Then the end result: {{stat:projection.iou_count|,}} boxes,
lowest IoU {{stat:projection.iou_min|.4f}}, with {{stat:projection.missing}} boxes missing and
{{stat:projection.extra}} invented.

**What I learned.** "Matches the reference" is only as strong as the reference: this proves my
code agrees with the devkit's implementation, not with an independent ground truth (nuScenes
has no separately annotated 2D boxes). And where IoU dips at all, it's on tiny or edge-clipped
boxes, where a sub-pixel difference is a large fraction of the box. That's why the
minimum-area cutoff for the image gallery is a setting (`config.toml`), not a constant.

## 4 · Ray-parallelized ingestion <span class="badge done">Done</span> {: #ray }
<!-- Hand-dated: later feat: commits to partition.py/benchmark.py (the task-layout
experiment, see experiments.md) aren't part of this milestone. -->
Oct 9 – Nov 1, 2026 · [benchmark on v1.0-mini](result-ray-scaling.html)
{: .meta }
<div class="tags"><span class="tag">Ray</span> <span class="tag">Performance benchmarking</span></div>

**Goal.** Find out whether parallelizing ingestion actually helps, and by how much.

**Approach.** A Ray driver (`partition.py`) that batches scenes, resolves each batch into a
plain-dict manifest *in the driver* (`nusc.get()` is cheap list indexing, since
`NuScenes.__init__` already loads every table eagerly), `ray.put()`s that manifest once per
batch, and farms it out to two independent table-writing tasks per batch, with per-task
timing (queue wait, compute, disk write). A benchmark harness reruns it at
{{stat:benchmark.worker_counts}} workers with repeated timed runs and records whole-machine
CPU utilization.

**Result.** The best run was {{stat:benchmark.best_speedup|.2f}}× faster than one worker, at
{{stat:benchmark.best_workers}} workers; {{stat:benchmark.max_workers}} workers was *slower*
than {{stat:benchmark.best_workers}}. At {{stat:benchmark.best_workers}} workers,
{{stat:benchmark.best_startup_fraction|.0%}} of each run is fixed startup cost (interpreter,
imports, `ray.init()`, loading the devkit's tables), which no number of workers can shrink.

**What I learned.**

- *On a dataset this small, parallelism is capped by its fixed costs before it is capped by
  cores.* I also caught a misleading chart of my own: a numeric worker-count axis invented an
  intermediate tick ("6") with no run behind it. Worker counts are sweep values, not a
  continuous range, so the axis is now categorical.
- *A Ray task is a separate process, not a thread.* Nothing in the driver (not even an
  already-loaded `nusc`) is visible inside a task; only what's explicitly passed in exists on
  the worker side. That one fact ruled out most of the designs I tried first: shipping `nusc`
  itself to every worker, a global buffer a second task could read from, a single `nusc` shared
  across per-sample tasks. The manifest pattern (resolve everything to a plain dict in the
  driver, hand *that* to the tasks) exists because of it.
- *Not every independent step deserves its own task.* I started with one task per output
  table (three for the core tables, one for the projected bboxes), since each write is
  genuinely independent of the others. Writing a table is cheap reshaping, though, so
  splitting the three core writes further only paid scheduling and deserialization overhead
  per task without adding real parallel compute. Consolidating them into one task, alongside
  the bbox task, is why there are two tasks per batch, not four. Measured later, the
  difference is at most {{stat:task_layout.diff_pct_max|.1%}}:
  [Experiments](experiment-ray-task-layout.html).

## 5 · Geometric BEV fusion <span class="badge done">Done</span> {: #bev }
Nov 1 – Nov 4, 2026 · [checks](validation.html#bev) · [interactive views](bev.html)
{: .meta }
<div class="tags"><span class="tag">OpenCV</span> <span class="tag">Plotly</span> <span class="tag">3D geometry</span></div>

**Goal.** Fuse LiDAR and cameras into one top-down view: every LiDAR point, coloured by the
camera that sees it, with annotated boxes overlaid.

**Approach.** Load each sample's LiDAR sweep into the ego frame; project the points into every
camera with the already-validated projection chain and sample each point's colour
(bilinear, via `cv2.remap`); bring the annotation boxes into the same frame; then render as a
scatter, a multi-channel raster (colour, height, intensity), and 3D views.

**How I checked it.** Six checks, again each against something independent. The strongest
is that the number of my LiDAR points inside each annotated box matches nuScenes' own
`num_lidar_pts` for {{stat:box_containment.verified}} / {{stat:box_containment.total}} boxes.
The others check the ego-frame axes against how the car actually moved, the LiDAR file's
columns against the devkit's loader, colour sampling against hand-computed interpolation, and
the raster against a deliberately naive reimplementation.

**What I learned.**

- *Rendering libraries have opinions.* Plotly's image trace draws row 0 at the top, the
  opposite of this project's forward-is-up convention. That would have silently flipped the
  whole view front-to-back; I only trusted the fix after rendering both variants and looking.
- *Averaging is a modelling decision.* Several LiDAR points can land in one raster cell at
  different heights (a branch over a road), so each cell keeps the surface nearest the sensor
  rather than blending unrelated ones. Exact ties *are* averaged: on real data they turned out
  to be multiple returns from one spot, not different surfaces.
- *Colour is a constraint, not decoration.* With up to ten categories on screen, no
  colour-blind-safe palette held up past seven distinct colours, so the six most common
  categories get their own colour and the rest share one.

## 6 · Multisweep BEV <span class="badge done">Done</span> {: #multisweep }
<!-- Hand-dated: the multisweep work started Nov 4 but was first committed on
Nov 16, so its commit dates alone would show a single day. -->
Nov 4 – Nov 16, 2026 · [checks](validation.html#multisweep) · [interactive views](bev.html)
{: .meta }
<div class="tags"><span class="tag">nuScenes devkit</span> <span class="tag">3D geometry</span> <span class="tag">Plotly</span></div>

**Goal.** One LiDAR sweep is sparse away from the car and leaves shadows behind anything in
the way. Accumulate the sweeps from the half second before a keyframe into one BEV, to fill in
more of the ground, without blurring what each raster cell means.

**Approach.** Walk back from a keyframe through `LIDAR_TOP`'s previous-sweep links to build a
window of sweeps. Match each sweep to every camera's frame nearest to it in time, within a
maximum gap, and colour each sweep from its own matched images. Move every sweep through the
global frame into the keyframe's ego frame, rasterize each on the same grid, and merge them
newest first: each cell takes all of its channels from the newest sweep that reached it. On
the [BEV page](bev.html), a Sweeps menu switches the viewer between the keyframe alone and the
merged window, with an extra view showing how old each cell's data is.

**How I checked it.** Four checks, each against something independent:

- *Alignment* against the devkit's own sweep accumulator,
  `LidarPointCloud.from_file_multisweep`: {{stat:multisweep.aligned_points|,}} /
  {{stat:multisweep.points|,}} points within a millimetre of it, with the same intensity
  (largest error {{stat:multisweep.max_error_mm|.3f}} mm).
- *Camera matching* against a brute-force scan of nuScenes' `sample_data` table:
  {{stat:camera_matching.verified|,}} / {{stat:camera_matching.total|,}} sweep and camera pairs.
- *Distance to the sensor* against the devkit's single-sweep loader:
  {{stat:multisweep.distance_ok|,}} / {{stat:multisweep.distance_total|,}} points.
- *The merge* against a naive per-cell loop: {{stat:multisweep.merge_ok|,}} /
  {{stat:multisweep.merge_total|,}} cells.

The alignment, camera-matching and merge checks also run self-tests: deliberately wrong inputs
that they have to report, so a pass means something.

**What I learned.**

- *"The ego frame" is many frames.* Every sweep and every camera image has its own, wherever
  the car was at that instant. Even "nearest the sensor" changes meaning: it has to be measured
  from the sweep's own origin, where the LiDAR actually was, not from the keyframe's.
- *A merge rule decides what a cell means.* Taking each cell whole from one sweep keeps its
  height, intensity and colour from one observation. The cost is visible on the page: where the
  newest sweep had no matched camera, the cell stays uncoloured even if an older sweep saw it.
- *Alignment only fixes the car's own motion.* The static world lines up across sweeps;
  anything else that moved leaves a trail, and only the keyframe has boxes to explain it.

## 7 · Experiments: testing the Ray task-layout claim <span class="badge done">Done</span> {: #experiments }
<!-- Hand-dated: the later feat: commit giving benchmark_task_layout.py one results file per
dataset and machine was made for the full-dataset run. -->
Nov 16, 2026 · [experiment](experiment-ray-task-layout.html)
{: .meta }
<div class="tags"><span class="tag">Ray</span> <span class="tag">Experiment design</span></div>

**Goal.** Test a claim this log already makes. Entry 4 says merging the three core-table writes
into one Ray task removed per-task overhead, but I never timed the four-task layout before
switching.

**Approach.** Bring the four-task layout back behind a config switch, so both layouts share
every other code path, and run both through the main benchmark's harness. The two layouts'
runs alternate, with the order swapped each repeat, and the metric is time after startup,
since startup costs the same under both. Each task's own timing is recorded, so the result
can be split by task type from the same runs.

**Result.** Two tasks per batch was faster by at most {{stat:task_layout.diff_pct_max|.1%}}:
consistently at {{stat:task_layout.faster_workers}} workers, within noise at
{{stat:task_layout.noise_workers}}. Bbox projection is {{stat:task_layout.bbox_share_min|.0%}}
to {{stat:task_layout.bbox_share_max|.0%}} of all task compute, so the core-table layout was
never where the time went. The full write-up is on the new
[Experiments](experiments.html) page.

**What I learned.**

- *A plausible explanation isn't a measurement.* The reasoning in entry 4 was right about the
  direction, but it read as if it explained the design, and the effect turned out to be about
  one percent.
- *Measure the change, not just the outcome.* Splitting compute by task type, from the same
  runs, is what showed where the time actually goes: bbox projection, which is the next thing
  worth measuring.

## 8 · Full-dataset ingestion run <span class="badge done">Done</span> {: #full-dataset }
<!-- Hand-dated: from the first commit preparing the Runpod run to the sweeps themselves. -->
Nov 20 – Nov 23, 2026 · [benchmark on v1.0-trainval](result-ray-scaling-trainval.html)
{: .meta }

**Goal.** Find out whether the parallelism measured on `v1.0-mini`
({{stat:benchmark.best_speedup|.2f}}× at {{stat:benchmark.best_workers}} workers) holds on
the full nuScenes dataset, or whether a dataset orders of magnitude larger surfaces a
different bottleneck (disk I/O, worker memory, or scheduling overhead) that `v1.0-mini`
was too small to expose.

**Approach.** Run the same Ray-parallelized ingestion pipeline (`partition.py`) against the
full dataset on Runpod, repeating the worker-count sweep `benchmark.py` already runs for
mini. Mini was swept again on the same pod, so the comparison changes only the dataset.

**Result.** On a {{stat:benchmark_trainval.cpus}}-CPU pod, trainval ran
{{stat:benchmark_trainval.best_speedup|.2f}}× faster at {{stat:benchmark_trainval.best_workers}}
workers than at one, against {{stat:benchmark_trainval.mini_best_speedup|.2f}}× (at
{{stat:benchmark_trainval.mini_best_workers}} workers) for mini on the same machine: a dataset
this size gives the workers enough to do. The limit is the fixed startup, mostly loading the
devkit's tables, which is {{stat:benchmark_trainval.startup_share_max|.0%}} of a
{{stat:benchmark_trainval.max_workers}}-worker run. Leaving startup out, the rest of the run
was {{stat:benchmark_trainval.work_speedup|.2f}}× faster. Disk I/O was not a bottleneck: Parquet
writes averaged at most {{stat:benchmark_trainval.write_ms_max|.0f}} ms per task at any worker
count.

**How I checked it.** The export from the sweep's last run
{{stat:benchmark_trainval.export_check}}: every devkit record present once, nothing extra,
every row attached to the right sample.
