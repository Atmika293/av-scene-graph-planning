<!--
DRAFT (Claude): the opening paragraph speaks for you -- check the tone and
the claim about where you're coming from before publishing. Every number
below is a {{stat:...}} directive, filled in from reports/ at build time.
-->
# AV Perception: nuScenes Pipeline

A self-directed project building an autonomous-vehicle perception pipeline on the
[nuScenes](https://www.nuscenes.org/) dataset, with each stage validated against an independent
reference before the next one builds on it. The learning and build logs document the reading,
decisions and mistakes along the way.
{: .lede }

[View the code on GitHub →](https://github.com/Atmika293/av-scene-graph-planning)

<div class="skill-pills">
<span class="pill">Ray</span>
<span class="pill">PyArrow</span>
<span class="pill">SE(3) / SciPy</span>
<span class="pill">OpenCV</span>
<span class="pill">Plotly</span>
</div>

<div class="progress-strip" aria-label="Roadmap progress: phase 2 of 7 in progress">
<div class="done">1 · Foundations</div>
<div class="current">2 · Ingestion + fusion</div>
<div>3 · Scene graph + VLA</div>
<div>4 · ROS2 + Nav2</div>
<div>5 · SLAM / SfM</div>
<div>6 · Home graph</div>
<div>7 · Flywheel</div>
</div>

**Now:** geometric bird's-eye-view fusion of LiDAR points with camera colour: built and
validated, with the interactive views published. **Reading:** Lift-Splat-Shoot, the first
of the learned BEV papers. See the [roadmap](roadmap.html) for what comes next.

<div class="stats">
<a class="stat" href="results.html#performance"><div class="value">{{stat:benchmark.best_speedup|.2f}}×</div><div class="label">best Ray speedup ({{stat:benchmark.best_workers}} workers), and why it stops there</div></a>
<a class="stat" href="bev.html"><img class="thumb" src="bev_thumbnail.png" alt="Top-down view of LiDAR points coloured by camera, with annotated boxes"><div class="label">LiDAR + camera fusion in bird's-eye view: explore the interactive views</div></a>
</div>

## Explore

<div class="cards">
<a class="card" href="learning.html"><strong>Learning log</strong><span>What I studied for each phase, the ideas I took from it, and where they show up in the code.</span></a>
<a class="card" href="build.html"><strong>Build log</strong><span>Each milestone: goal, approach, how I checked it, what I learned.</span></a>
<a class="card" href="validation.html"><strong>Validation</strong><span>The chain of checks behind every stage, each against an independent reference, with its latest result.</span></a>
<a class="card" href="results.html"><strong>Results</strong><span>Ray scaling benchmark and projection-accuracy analysis.</span></a>
<a class="card" href="bev.html"><strong>BEV fusion</strong><span>Interactive top-down and 3D views of LiDAR + camera fusion, with annotated boxes.</span></a>
<a class="card" href="roadmap.html"><strong>Roadmap</strong><span>The seven-phase plan, the optional extensions, and where this project is in it.</span></a>
</div>
