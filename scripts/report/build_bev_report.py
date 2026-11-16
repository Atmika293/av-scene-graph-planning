"""Builds docs/_bev_section.html (the published BEV page's content) and
docs/bev_data/<sample_token>.json (one file per sample, fetched on demand).

Same generator role as build_correctness_report.py, for a new page
(bev.html, via src/report/site.py's PAGES list) instead of a new section of
the existing one.

Four views so far -- raster color (default), colored scatter, raster
height, and raster intensity -- all built from the same validated
pipelines scripts/validation/ already checked (bev.raster.build_raster,
GeometricBEVPipeline.project_to_bev). The height/intensity views needed no
new validation: build_raster already returns those channels, already
checked by check_raster.py's 5 checks plus its naive-implementation
cross-check -- this just stops discarding data that was already being
computed. The remaining design-spec views (height-colored scatter,
3D-by-RGB, 3D-by-intensity) aren't here yet: each gets its own reports/
validation first, same as the others did, before joining this page.

Multisweep: a "Sweeps" menu switches the same viewer between the keyframe's
own sweep and a merged window of it plus the sweeps before it
(bev.multisweep, checked by check_multisweep.py and check_camera_matching.py;
see multisweep_bev_data). So both modes show the same moment, each scene's
sample is the window's reference keyframe, not the scene's first sample
(which has no earlier sweeps). Multisweep mode offers raster views only,
plus a source-sweep-age raster, since a window has ~10x the points.

Data loading: switching VIEW is instant and client-side, since a sample's
JSON carries every view's data together -- only switching SAMPLE triggers a
fetch (docs/bev_data/<token>.json, a relative path that works the same
locally and on GitHub Pages, since bev.html sits in docs/ too). Point
coordinates are rounded to centimetres (2 decimal places) to keep the
per-sample JSON smaller; raster cells aren't rounded further since
cell_size already bounds their resolution.

Raster orientation: go.Image reverses its y-axis by default (row 0 renders
at the TOP), the opposite of this project's forward-is-up convention
(matplotlib's origin="lower", used throughout scripts/validation/
check_raster.py and scripts/viz/render_bev.py) -- confirmed by actually
rendering both variants via kaleido, not just reading the docs text, since
this would have been a silent, easy-to-miss bug (the whole view flipped
forward/backward) rather than an obvious one. Fixed by giving both axes an
explicit range in bev_layout (also confirmed via kaleido to override
go.Image's default the same way an explicit autorange=True did).

Box footprints (SampleBoxProjection, already validated by
check_box_containment.py's N_LIDAR_PTS count match) are overlaid on both
views as closed-loop outlines, colored by category -- the 6 most frequent
categories across these samples, counted at build time
(assign_category_colors), get CATEGORY_PALETTE's hand-tuned, vivid colors,
verified with the dataviz skill's validator (see CATEGORY_PALETTE's own
comment for how), folding rarer categories into a shared neutral
"Other" (OTHER_CATEGORY_COLOR), since 9-10 simultaneously visible
categories exceeds what any CVD-safe categorical scheme supports at once --
confirmed empirically (an 8th slot consistently broke validation regardless
of placement), not just assumed from the palette's own documented 3-slot
guidance for the muted default palette.

Run with: python scripts/report/build_bev_report.py (from the repo root,
after ingesting/having local access to the configured nuScenes dataset).
"""

import json
import os
from collections import Counter

import numpy as np
import plotly.graph_objects as go
from nuscenes.nuscenes import NuScenes

from av_data_pipeline.config import NUSCENES_VERSION, NUSCENES_DATAROOT
from av_data_pipeline.ingestion import extract_sample_data_from_sample
from av_data_pipeline.ingestion_keys import SAMPLE_INGESTION_KEYS, SAMPLE_ANNOTATION_INGESTION_KEYS
from bev.fusion import GeometricBEVPipeline
from bev.multisweep import WINDOW_LENGTH, assemble_window, project_window_to_bev
from bev.raster import build_raster, merge_rasters, to_plot_xy
from bev.sample import SampleBoxProjection

REPORT_DIR = "docs"
DATA_DIR = os.path.join(REPORT_DIR, "bev_data")
SECTION_FILE = os.path.join(REPORT_DIR, "_bev_section.html")
THUMBNAIL_FILE = os.path.join(REPORT_DIR, "bev_thumbnail.png")  # the home page's BEV card
NUM_SCENES = 3  # one sample from each of the first N scenes, for variety
GRID_HALF_EXTENT_METERS = 25.0  # same as the validated scripts/validation/check_raster.py
CELL_SIZE_METERS = 0.25
COORD_DECIMALS = 2  # centimetre precision
# "colored" points use real, uncontrolled camera-sampled RGB -- often muted
# grays/earth tones (asphalt, concrete, buildings) -- so a gray sentinel for
# "uncovered"/"uncolored" risks blending in rather than reading as synthetic.
# Using the project's validated categorical palette's blue (dataviz skill,
# references/palette.md slot 1): strongly saturated, rare in road-scene
# photography, and clearly separated from the magenta (#ff00ff) already used
# for box footprint overlays in render_bev.py. One RGB source of truth, used
# both as a hex string (scatter marker color) and as an (r, g, b) triplet
# (raster cell color) -- shared across both views, not two colors that could
# drift apart.
UNCOVERED_COLOR_RGB = [42, 120, 214]
UNCOVERED_COLOR = "#{:02x}{:02x}{:02x}".format(*UNCOVERED_COLOR_RGB)
RASTER_EMPTY_COLOR = [255, 255, 255]  # blends with the page's white background
RASTER_UNCOLORED_COLOR = UNCOVERED_COLOR_RGB  # occupied but no camera coverage -- same meaning as scatter's "uncovered"

# SampleAnnotation.__extract_corners orders a box's 8 corners as top/bottom
# pairs; indices 0, 2, 4, 6 are its top face, tracing the footprint
# rectangle in perimeter order once projected to the ego x-y plane -- same
# convention as scripts/viz/render_bev.py's FOOTPRINT_CORNER_ORDER. Repeating
# index 0 closes the loop for plotting.
FOOTPRINT_CORNER_ORDER = [0, 2, 4, 6, 0]
# Six colour slots for box categories, given in this order to the 6 most
# frequent categories across the page's samples, counted at build time
# (assign_category_colors), so the list follows the samples when they change.
# Deliberately vivid colors -- not the project's validated categorical
# palette's own (muted) values, by request. Hand-tuned per-hue and verified
# with the dataviz skill's validator (scripts/validate_palette.js) against
# UNCOVERED_COLOR (blue) and each other, --pairs all (every box is visible
# simultaneously, not sequentially): PASS at 7 slots (blue + these 6), with
# the worst CVD pair in the 6-8 band, legal because the legend names every
# category. What made 7 pass was widening the LIGHTNESS gap between
# same-family colors, since CVD collapses hue discrimination but not
# lightness. A 7th category slot consistently broke validation regardless of
# placement, so everything past the top 6 folds into OTHER_CATEGORY_COLOR.
CATEGORY_PALETTE = ["#f2930d", "#e10e20", "#20b691", "#7e0f95", "#0b8447", "#d788ca"]
# Plain-English names for nuScenes' 23 annotation categories, for the page's
# box-colour list (the legend keeps the dataset's own names).
CATEGORY_READABLE_NAMES = {
    "animal": "Animal",
    "human.pedestrian.adult": "Adult pedestrian",
    "human.pedestrian.child": "Child pedestrian",
    "human.pedestrian.construction_worker": "Construction worker",
    "human.pedestrian.personal_mobility": "Person on a scooter, skateboard, etc.",
    "human.pedestrian.police_officer": "Police officer",
    "human.pedestrian.stroller": "Stroller",
    "human.pedestrian.wheelchair": "Wheelchair",
    "movable_object.barrier": "Barrier",
    "movable_object.debris": "Debris",
    "movable_object.pushable_pullable": "Cart, trolley or bin",
    "movable_object.trafficcone": "Traffic cone",
    "static_object.bicycle_rack": "Bicycle rack",
    "vehicle.bicycle": "Bicycle",
    "vehicle.bus.bendy": "Articulated bus",
    "vehicle.bus.rigid": "Bus",
    "vehicle.car": "Car",
    "vehicle.construction": "Construction vehicle",
    "vehicle.emergency.ambulance": "Ambulance",
    "vehicle.emergency.police": "Police car",
    "vehicle.motorcycle": "Motorcycle",
    "vehicle.trailer": "Trailer",
    "vehicle.truck": "Truck",
}
# Category -> colour, filled by assign_category_colors() before anything is drawn.
CATEGORY_COLORS: dict[str, str] = {}
# Not a muted gray: that has the exact same blend-with-photographic-colors
# problem UNCOVERED_COLOR was chosen to avoid. All 8 validated categorical
# slots are already spoken for (blue + the 6 category slots above), so this
# reuses the palette's own primary ink instead of inventing an ungoverned
# 9th hue -- near-black is about as rare in muted camera photography as
# saturated blue is, for the same reason.
OTHER_CATEGORY_COLOR = "#0b0b0b"
# Top ~45% of the right margin, stacked above where bev_layout's legend
# starts (y=0.52) -- a colorbar and the category legend otherwise both
# default to the same right-edge position and overlap.
COLORBAR_POSITION = {"y": 1, "yanchor": "top", "len": 0.45}


def nan_grid_to_json(grid: np.array, decimals: int) -> list:
    """Convert a 2D float array (NaN = no data) into a JSON-safe nested list.

    JSON has no NaN; go.Heatmap treats a null z-cell as "no data" (rendered
    distinctly from any real colorscale value, confirmed empirically), so
    None is the correct target, not a sentinel number like -1.

    Args:
        grid: 2D array, NaN where there's no data.
        decimals: Rounding precision for real values.

    Returns:
        Nested list (rows of cells), each cell a rounded float or None.
    """
    rounded = np.round(grid, decimals).astype(object)
    rounded[np.isnan(grid)] = None
    return rounded.tolist()


def grid_percentile(grid: list, percentile: float) -> float:
    """Percentile of a nan_grid_to_json-shaped grid's real (non-None) cells.

    Args:
        grid: Nested list as produced by nan_grid_to_json.
        percentile: 0-100.

    Returns:
        The percentile value over every non-None cell.
    """
    values = [v for row in grid for v in row if v is not None]
    return float(np.percentile(values, percentile))


def sample_bev_data(sample_data: dict) -> dict:
    """Build one sample's scatter and raster data, ready for JSON + Plotly.

    Args:
        sample_data: One sample's sample_data dict (as built by
            av_data_pipeline.ingestion.extract_sample_data_from_sample).

    Returns:
        A dict with "scatter" ({"colored": {x, y, color}, "uncovered": {x, y}}),
        "raster" ({z, x0, y0, dx, dy}), and "boxes" (a list of
        {category, x, y} footprint outlines), ready to embed as JSON.
    """
    pipeline = GeometricBEVPipeline(sample_data)
    ego_points, colors = pipeline.project_to_bev()
    intensity = pipeline.lidar_projection.intensity

    plot_x, plot_y = to_plot_xy(ego_points)
    valid = np.all(colors != -1, axis=-1)

    colored_hex = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in colors[valid]]
    scatter = {
        "colored": {
            "x": np.round(plot_x[valid], COORD_DECIMALS).tolist(),
            "y": np.round(plot_y[valid], COORD_DECIMALS).tolist(),
            # z (height) only used by the 3D views -- same points, same
            # colored/uncovered split as the 2D scatter, just with a third
            # coordinate added rather than a separate duplicated point set.
            "z": np.round(ego_points[valid, 2], COORD_DECIMALS).tolist(),
            "color": colored_hex,
        },
        "uncovered": {
            "x": np.round(plot_x[~valid], COORD_DECIMALS).tolist(),
            "y": np.round(plot_y[~valid], COORD_DECIMALS).tolist(),
            "z": np.round(ego_points[~valid, 2], COORD_DECIMALS).tolist(),
        },
        # All points, not split by camera coverage -- intensity doesn't care
        # whether a camera saw the point (same reasoning as the raster
        # intensity channel). Duplicates x/y/z already present above, split
        # differently; kept simple over byte-minimal.
        "intensity": {
            "x": np.round(plot_x, COORD_DECIMALS).tolist(),
            "y": np.round(plot_y, COORD_DECIMALS).tolist(),
            "z": np.round(ego_points[:, 2], COORD_DECIMALS).tolist(),
            "value": np.round(intensity, 1).tolist(),
        },
    }

    annotations = sample_data[SAMPLE_INGESTION_KEYS['annotations']]
    ego_boxes = SampleBoxProjection(sample_data).project_to_ego()
    boxes = []
    for ann_token, corners in ego_boxes.items():
        category = annotations[ann_token][SAMPLE_ANNOTATION_INGESTION_KEYS['category']]
        footprint_x, footprint_y = to_plot_xy(corners[FOOTPRINT_CORNER_ORDER])
        boxes.append({
            "category": category,
            "x": np.round(footprint_x, COORD_DECIMALS).tolist(),
            "y": np.round(footprint_y, COORD_DECIMALS).tolist(),
        })

    raster = build_raster(ego_points, colors, intensity, GRID_HALF_EXTENT_METERS, CELL_SIZE_METERS)
    return {"scatter": scatter, "raster": raster_payload(raster), "boxes": boxes}


def raster_payload(raster: dict) -> dict:
    """A build_raster/merge_rasters result as JSON-ready display data: RGB image, height, intensity.

    Args:
        raster: The raster to display.

    Returns:
        {z (display RGB), x0, y0, dx, dy, height, intensity}, ready to embed as JSON.
    """
    # Flatten raster's three-way occupied/colored/uncolored distinction into
    # two flat RGB fallbacks (empty cells blend into the page background,
    # occupied-but-uncolored cells read as a visible gray) -- simpler than
    # an RGBA image for this first version; see the module docstring.
    display_color = np.where(
        raster["colored_mask"][:, :, None], raster["color"],
        np.where(raster["occupied_mask"][:, :, None], RASTER_UNCOLORED_COLOR, RASTER_EMPTY_COLOR)
    ).astype(np.uint8)
    left, right, bottom, top = raster["extent"]
    return {
        "z": display_color.tolist(),
        "x0": left,
        "y0": bottom,
        "dx": CELL_SIZE_METERS,
        "dy": CELL_SIZE_METERS,
        # NaN isn't valid JSON; go.Heatmap treats null z-cells as "no data"
        # (distinct from any real colorscale value), confirmed empirically,
        # so NaN -> None here is both necessary and the right semantic match.
        "height": nan_grid_to_json(raster["height"], decimals=2),
        "intensity": nan_grid_to_json(raster["intensity"], decimals=1),
    }


def multisweep_bev_data(nusc: NuScenes, sweep_tokens: list[str], camera_groups: dict) -> dict:
    """Build the merged multisweep raster for one window, ready for JSON + Plotly.

    Each sweep is coloured by its own nearest cameras and rasterized on the
    same grid as the single-sweep view, then merged newest first
    (bev.raster.merge_rasters), all checked by check_multisweep.py and
    check_camera_matching.py. Only raster data: a window has ~10x a sweep's
    points, too many to ship as per-point scatter data, while a raster is
    the same size however many sweeps feed it.

    Args:
        nusc: NuScenes devkit instance.
        sweep_tokens: The window's LIDAR_TOP tokens, newest (the reference keyframe) first.
        camera_groups: Sweep token -> {camera channel: matched camera token}
            (see bev.multisweep.assemble_window).

    Returns:
        {"raster": raster_payload(merged) plus "age" (ms before the reference
        sweep that each cell's data came from, None where empty),
        "sweeps": window length, "span_ms": oldest sweep's age,
        "occupied_gain": merged occupied cells / the reference sweep's alone}.
    """
    sweeps = project_window_to_bev(nusc, sweep_tokens, camera_groups)
    rasters = [build_raster(sweep["points"], sweep["colors"], sweep["intensity"],
                            GRID_HALF_EXTENT_METERS, CELL_SIZE_METERS, distance=sweep["distance"])
               for sweep in sweeps]
    merged = merge_rasters(rasters)

    timestamps = [nusc.get("sample_data", token)["timestamp"] for token in sweep_tokens]
    ages_ms = np.array([(timestamps[0] - timestamp) / 1000 for timestamp in timestamps])
    age = np.where(merged["source"] >= 0, ages_ms[np.maximum(merged["source"], 0)], np.nan)

    payload = raster_payload(merged)
    payload["age"] = nan_grid_to_json(age, decimals=0)
    return {
        "raster": payload,
        "sweeps": len(sweeps),
        "span_ms": round(float(ages_ms[-1])),
        "occupied_gain": round(float(merged["occupied_mask"].sum() / max(rasters[0]["occupied_mask"].sum(), 1)), 2),
    }


def bev_layout() -> dict:
    """Shared layout for both views -- equal aspect, forward-is-up, zoomed to
    the ego's neighborhood by default.

    An explicit range (rather than letting Plotly autofit to the data) keeps
    the scatter view's default zoom level reasonable: raw lidar range has a
    long thin tail out past 100m (see bev.raster's module docstring/the
    measurements behind GRID_HALF_EXTENT_METERS), so autofit-to-data would
    zoom out to fit a handful of far outlier points, leaving most of the
    chart as empty space around the actually-dense central area. Reuses the
    same ±GRID_HALF_EXTENT_METERS bound the raster grid is already built to,
    for visual consistency between views -- still fully interactive, a
    reader can zoom/pan out from here, this only sets the initial view.

    Returns:
        A layout dict for go.Figure.update_layout / embedding in JS.
    """
    axis_range = [-GRID_HALF_EXTENT_METERS, GRID_HALF_EXTENT_METERS]
    return {
        "height": 600,
        "margin": {"t": 30},
        "xaxis": {"title": "left <- ego y -> right (flipped)", "scaleanchor": "y", "range": axis_range},
        # An explicit ascending range also overrides go.Image's default
        # reversed y-axis (see the module docstring) -- confirmed empirically
        # for this mechanism specifically, not just carried over from the
        # earlier autorange=True fix.
        "yaxis": {"title": "ego x (forward)", "range": axis_range},
        # Stacked with COLORBAR_POSITION (top ~45% of the right margin) so
        # the two never overlap on the height/intensity views -- harmless on
        # scatter/raster-color, which have no colorbar competing for space.
        "legend": {"y": 0.52, "yanchor": "top"},
        # Only used by the 3D views -- confirmed empirically (kaleido) that
        # an unused "scene" key alongside 2D xaxis/yaxis is harmless for 2D
        # traces, so one shared layout serves every view rather than
        # dispatching between a 2D and 3D layout per view. Scatter3d
        # defaults z to "up" already (also confirmed empirically -- unlike
        # go.Image, no reversal fix needed here), so this just sets range,
        # titles, and an initial camera angle (tuned by eye, not a
        # validated convention, same as render_bev_3d.py's ELEV/AZIM).
        "scene": {
            "xaxis": {"title": "left <- ego y -> right (flipped)", "range": axis_range},
            "yaxis": {"title": "ego x (forward)", "range": axis_range},
            "zaxis": {"title": "ego z (up)"},
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.4, "y": -1.4, "z": 1.0}},
        },
    }


def coverage_legend_entry(is_3d: bool = False) -> "go.Scatter":
    """A fixed-size, legend-only marker explaining the "no camera coverage" blue.

    Shared so the legend looks identical across views -- scatter's real
    uncovered-point markers are tiny (size 2, matching tens of thousands of
    plotted points), which would make a barely-visible legend swatch if
    reused directly for the legend the way go.Scattergl normally does it;
    go.Image doesn't produce a legend entry at all on its own. One shared,
    deliberately larger (size 8) dummy marker (x/y[/z] are None, so nothing
    is actually drawn) covers every case identically instead of each view
    improvising its own.

    Args:
        is_3d: True to build a go.Scatter3d instead of go.Scatter. Matters
            even though this trace draws nothing: a 2D go.Scatter mixed into
            an otherwise-all-3D figure makes Plotly allocate and render a
            full (empty, confusing) 2D axis frame around the 3D scene --
            confirmed empirically, not a theoretical concern -- since a
            genuine 2D trace now exists needing that axis system, unlike an
            unused layout key alone.
    """
    if is_3d:
        return go.Scatter3d(x=[None], y=[None], z=[None], mode="markers",
                             marker=dict(size=8, color=UNCOVERED_COLOR),
                             name="no camera coverage")
    return go.Scatter(x=[None], y=[None], mode="markers",
                       marker=dict(size=8, color=UNCOVERED_COLOR),
                       name="no camera coverage")


def scatter_traces(data: dict) -> list:
    """Build the two Scattergl traces (uncovered, colored) for one sample's data."""
    return [
        go.Scattergl(x=data["scatter"]["uncovered"]["x"], y=data["scatter"]["uncovered"]["y"],
                     mode="markers", marker=dict(size=2, color=UNCOVERED_COLOR, opacity=0.6),
                     name="no camera coverage", showlegend=False),
        coverage_legend_entry(),
        # No legend entry: unlike "no camera coverage", there's no single
        # color this swatch could represent -- each point keeps its own
        # real sampled color, so a legend dot here wouldn't mean anything.
        go.Scattergl(x=data["scatter"]["colored"]["x"], y=data["scatter"]["colored"]["y"],
                     mode="markers", marker=dict(size=2, color=data["scatter"]["colored"]["color"]),
                     name="camera-colored", showlegend=False),
    ]


def scatter_3d_traces(data: dict) -> list:
    """Build the two Scatter3d traces (uncovered, colored) for one sample's data.

    No box_traces() here: box footprints are flat 2D line traces (type
    "scatter", the default xaxis/yaxis), not 3D -- overlaying them on a 3D
    scene would draw them on an unrelated, unaligned coordinate system
    rather than actually appearing at the boxes' real 3D position. 3D box
    wireframes are a real feature, just not part of this promotion --
    reports/validation.py's build only checked point rendering.
    """
    c = data["scatter"]["colored"]
    u = data["scatter"]["uncovered"]
    return [
        go.Scatter3d(x=u["x"], y=u["y"], z=u["z"], mode="markers",
                     marker=dict(size=2, color=UNCOVERED_COLOR, opacity=0.6),
                     name="no camera coverage", showlegend=False),
        coverage_legend_entry(is_3d=True),
        go.Scatter3d(x=c["x"], y=c["y"], z=c["z"], mode="markers",
                     marker=dict(size=2, color=c["color"]),
                     name="camera-colored", showlegend=False),
    ]


def scatter_3d_intensity_traces(data: dict) -> list:
    """Build a Scatter3d trace colored by lidar return intensity (viridis).

    Clips at the 95th percentile, same reasoning and same measured
    distribution shape as raster_intensity_traces (point-level intensity
    has the same long thin tail: median ~12, p95 ~65, max 255 -- confirmed
    on scripts/viz/render_bev_3d.py's sanity-check render before this
    promotion, not assumed to carry over unchanged).
    """
    i = data["scatter"]["intensity"]
    zmax = float(np.percentile(i["value"], 95))
    return [go.Scatter3d(x=i["x"], y=i["y"], z=i["z"], mode="markers",
                          marker=dict(size=2, color=i["value"], colorscale="Viridis",
                                      cmin=0, cmax=zmax,
                                      colorbar=dict(title="intensity", **COLORBAR_POSITION)))]


def raster_traces(data: dict) -> list:
    """Build the raster's Image trace, plus a legend-only marker explaining its blue cells.

    go.Image doesn't produce its own legend entry the way Scattergl does, so
    without this, the blue "no camera coverage" cells would be visible in
    the image but unexplained -- the scatter view's legend covers the same
    color, but switching views shouldn't drop that explanation.
    """
    r = data["raster"]
    image_trace = go.Image(z=r["z"], x0=r["x0"], y0=r["y0"], dx=r["dx"], dy=r["dy"])
    return [image_trace, coverage_legend_entry()]


def raster_height_traces(data: dict) -> list:
    """Build a Heatmap trace of the raster's height channel (meters, viridis).

    No coverage_legend_entry here: "no camera coverage" is meaningless for
    this channel -- a point can have a real height whether or not any
    camera saw it, so there's no blue region to explain.
    """
    r = data["raster"]
    return [go.Heatmap(z=r["height"], x0=r["x0"], y0=r["y0"], dx=r["dx"], dy=r["dy"],
                        colorscale="Viridis",
                        colorbar=dict(title="height (m)", **COLORBAR_POSITION))]


def raster_intensity_traces(data: dict) -> list:
    """Build a Heatmap trace of the raster's intensity channel (viridis).

    Clips the color scale at the 95th percentile rather than the real
    (auto-scaled) max: measured on real raster cells, intensity's median is
    ~11 and its p95 is ~49, but a single outlier cell can reach 239 --
    auto-scaling to that actual max would stretch nearly all the color
    range across a handful of outlier cells, compressing the 95% of cells
    that carry the real structure into a narrow dark band (confirmed on the
    3D point-level view first, same root cause here). Height doesn't get
    the same treatment: measured median ~0, p95 ~5.1, max ~7.0 -- no
    comparable outlier gap, so plain auto-scaling is already fine there.
    """
    r = data["raster"]
    zmax = grid_percentile(r["intensity"], 95)
    return [go.Heatmap(z=r["intensity"], x0=r["x0"], y0=r["y0"], dx=r["dx"], dy=r["dy"],
                        colorscale="Viridis", zmin=0, zmax=zmax,
                        colorbar=dict(title="intensity", **COLORBAR_POSITION))]


def box_traces(data: dict) -> list:
    """Build one line trace per box footprint, with one legend entry per category.

    Plotly gives every trace its own legend row by default; many boxes
    sharing a category would otherwise repeat that category's label once per
    box. legendgroup + showlegend=False on every occurrence after the first
    collapses each category to a single legend entry instead.
    """
    traces = []
    seen_categories = set()
    for box in data["boxes"]:
        category = box["category"]
        color = CATEGORY_COLORS.get(category, OTHER_CATEGORY_COLOR)
        show_legend = category not in seen_categories
        seen_categories.add(category)
        traces.append(go.Scatter(x=box["x"], y=box["y"], mode="lines",
                                  line=dict(color=color, width=1.5),
                                  name=category, legendgroup=category, showlegend=show_legend))
    return traces


def assign_category_colors(samples: list[dict]) -> list[str]:
    """Give CATEGORY_PALETTE's slots to the most frequent box categories across the samples.

    Ties break alphabetically so the assignment is stable between builds.
    Fills CATEGORY_COLORS in place (box_traces, write_thumbnail and the
    page's JS all read it) and returns the chosen categories, most frequent first.
    """
    counts = Counter(box["category"] for sample in samples for box in sample["data"]["boxes"])
    ranked = sorted(counts, key=lambda category: (-counts[category], category))
    chosen = ranked[:len(CATEGORY_PALETTE)]
    CATEGORY_COLORS.clear()
    CATEGORY_COLORS.update(zip(chosen, CATEGORY_PALETTE))
    return chosen


def write_thumbnail(data: dict, path: str) -> None:
    """Render a small, axis-free version of the scatter view (points + box footprints) as a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3, 3), dpi=160)
    u, c = data["scatter"]["uncovered"], data["scatter"]["colored"]
    ax.scatter(u["x"], u["y"], s=1.5, c=UNCOVERED_COLOR, alpha=0.6, linewidths=0)
    ax.scatter(c["x"], c["y"], s=1.5, c=c["color"], linewidths=0)
    for box in data["boxes"]:
        ax.plot(box["x"], box["y"], lw=1.2,
                color=CATEGORY_COLORS.get(box["category"], OTHER_CATEGORY_COLOR))
    ax.set_xlim(-GRID_HALF_EXTENT_METERS, GRID_HALF_EXTENT_METERS)
    ax.set_ylim(-GRID_HALF_EXTENT_METERS, GRID_HALF_EXTENT_METERS)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATAROOT, verbose=True)

    os.makedirs(DATA_DIR, exist_ok=True)

    samples = []
    for scene in nusc.scene[:NUM_SCENES]:
        # The multisweep window's reference keyframe, so both Sweeps modes
        # show the same moment, ego frame and boxes (a scene's first sample
        # has no earlier sweeps to accumulate).
        sweep_tokens, camera_groups = assemble_window(nusc, scene["token"])
        sample_token = nusc.get("sample_data", sweep_tokens[0])["sample_token"]
        sample_data = extract_sample_data_from_sample(nusc, sample_token)

        data = sample_bev_data(sample_data)
        data["multisweep"] = multisweep_bev_data(nusc, sweep_tokens, camera_groups)
        out_path = os.path.join(DATA_DIR, f"{sample_token}.json")
        with open(out_path, "w") as outfile:
            json.dump(data, outfile)
        size_kb = os.path.getsize(out_path) / 1024
        print(f"Wrote {out_path} ({size_kb:.0f} KB)")

        samples.append({"token": sample_token, "scene_name": scene["name"], "data": data})

    colored_categories = assign_category_colors(samples)
    category_items = "\n".join(
        f'<li><span style="display:inline-block;width:0.9em;height:0.9em;border-radius:2px;'
        f'vertical-align:-0.1em;margin-right:6px;background:{CATEGORY_COLORS[category]}"></span>'
        f'{CATEGORY_READABLE_NAMES.get(category, category)} ({category})</li>'
        for category in colored_categories)
    default_sample = samples[0]
    write_thumbnail(default_sample["data"], THUMBNAIL_FILE)
    print(f"Wrote {THUMBNAIL_FILE}")
    default_fig = go.Figure(raster_traces(default_sample["data"]) + box_traces(default_sample["data"]))
    default_fig.update_layout(**bev_layout())
    initial_html = default_fig.to_html(full_html=False, include_plotlyjs="cdn", div_id="bev-chart")

    sample_options = "\n".join(
        f'<option value="{s["token"]}"{" selected" if s is default_sample else ""}>{s["scene_name"]}</option>'
        for s in samples
    )
    # Only the default sample is embedded inline (for a fast first paint,
    # matching what fig.to_html() already does for the chart itself) -- the
    # other samples are genuinely fetched on demand when selected, not
    # pre-loaded, or there'd be no point in bev_data/*.json existing at all.
    data_cache_entries = f'"{default_sample["token"]}": {json.dumps(default_sample["data"])}'

    section_html = f"""<section id="bev">
<h2>BEV: lidar + camera fusion</h2>
<div class="prose">
<h3>Reading the chart</h3>
<ul>
<li><strong>Points:</strong> each LiDAR point takes the colour of the camera that sees it
(<a href="https://github.com/Atmika293/av-scene-graph-planning/blob/main/src/bev/fusion.py">bev/fusion.py</a>), then is either plotted directly
(<strong>scatter</strong>) or binned into a fixed-size grid (<strong>raster</strong>,
<a href="https://github.com/Atmika293/av-scene-graph-planning/blob/main/src/bev/raster.py">bev/raster.py</a>, checked by
<a href="https://github.com/Atmika293/av-scene-graph-planning/blob/main/scripts/validation/check_raster.py">check_raster.py</a>).</li>
<li><strong>Blue:</strong> LiDAR returns no camera saw. <strong>White:</strong> cells with no
LiDAR return at all.</li>
<li><strong>Boxes:</strong> annotated box footprints
(<a href="https://github.com/Atmika293/av-scene-graph-planning/blob/main/src/bev/sample.py">bev/sample.py</a>), each in its category's colour, per the legend.</li>
</ul>
<h3>The Sweeps menu</h3>
<ul>
<li><strong>1 (keyframe):</strong> the keyframe's own LiDAR sweep.</li>
<li><strong>Multisweep:</strong> adds the sweeps just before it. Each is coloured by the camera
frames nearest to it in time and moved into the keyframe's ego frame, then all are merged on the
same grid, each cell keeping the newest sweep that reached it
(<a href="https://github.com/Atmika293/av-scene-graph-planning/blob/main/src/bev/multisweep.py">bev/multisweep.py</a>, checked in the
<a href="validation.html#multisweep">multisweep checks</a>).</li>
<li><strong>Trails:</strong> boxes are the keyframe's, so anything that moved during the window
leaves a trail behind its box.</li>
<li><strong>Blue bands:</strong> the newest sweep to reach those cells had no camera frame close
enough in time for that direction. The merge keeps that sweep's uncoloured point rather than an
older, coloured one.</li>
<li><strong>Views:</strong> raster views only, plus <strong>source sweep age</strong>: how long
before the keyframe each cell's data was captured.</li>
</ul>
</div>
<div style="margin:12px 0;">
<label>View: <select id="bev-view-select">
<option value="raster" selected>Raster (color)</option>
<option value="scatter">Scatter (color)</option>
<option value="raster_height">Raster (height)</option>
<option value="raster_intensity">Raster (intensity)</option>
<option value="scatter_3d">Scatter 3D (color)</option>
<option value="scatter_3d_intensity">Scatter 3D (intensity)</option>
<option value="raster_age" disabled>Raster (source sweep age)</option>
</select></label>
&nbsp;&nbsp;
<label>Sweeps: <select id="bev-sweeps-select">
<option value="single" selected>1 (keyframe)</option>
<option value="multi">Multisweep</option>
</select></label>
&nbsp;&nbsp;
<label>Sample: <select id="bev-sample-select">
{sample_options}
</select></label>
</div>
<p class="chart-note" id="bev-caption"></p>
{initial_html}
<script>
(function() {{
  var BEV_DATA_CACHE = {{
{data_cache_entries}
  }};
  var BEV_LAYOUT = {json.dumps(bev_layout())};
  var CATEGORY_COLORS = {json.dumps(CATEGORY_COLORS)};
  var OTHER_CATEGORY_COLOR = "{OTHER_CATEGORY_COLOR}";
  var COLORBAR_POSITION = {json.dumps(COLORBAR_POSITION)};

  // Shared by both views so the legend looks identical regardless of which
  // is active -- scatter's real uncovered-point markers are tiny (size 2,
  // matching tens of thousands of plotted points), a barely-visible legend
  // swatch if reused directly; go.Image doesn't produce a legend entry at
  // all on its own. One shared, deliberately larger (size 8) dummy marker
  // (x/y are null, so nothing is actually drawn) covers both identically.
  // is3d matters even though this trace draws nothing: a 2D "scatter" trace
  // mixed into an otherwise-all-3D figure makes Plotly allocate and render
  // a full (empty, confusing) 2D axis frame around the 3D scene --
  // confirmed empirically, not a theoretical concern.
  function coverageLegendEntry(is3d) {{
    return {{x: [null], y: [null], z: is3d ? [null] : undefined,
             mode: "markers", type: is3d ? "scatter3d" : "scatter",
             marker: {{size: 8, color: "{UNCOVERED_COLOR}"}}, name: "no camera coverage"}};
  }}
  function scatterTraces(data) {{
    return [
      {{x: data.scatter.uncovered.x, y: data.scatter.uncovered.y, mode: "markers", type: "scattergl",
        marker: {{size: 2, color: "{UNCOVERED_COLOR}", opacity: 0.6}}, name: "no camera coverage",
        showlegend: false}},
      coverageLegendEntry(),
      {{x: data.scatter.colored.x, y: data.scatter.colored.y, mode: "markers", type: "scattergl",
        marker: {{size: 2, color: data.scatter.colored.color}}, name: "camera-colored", showlegend: false}}
    ];
  }}
  function rasterTraces(data) {{
    var r = data.raster;
    return [
      {{type: "image", z: r.z, x0: r.x0, y0: r.y0, dx: r.dx, dy: r.dy}},
      coverageLegendEntry()
    ];
  }}
  // No coverageLegendEntry() for height/intensity: "no camera coverage" is
  // meaningless for these channels -- a point has a real height/intensity
  // whether or not a camera saw it, so there's no blue region to explain.
  function rasterHeightTraces(data) {{
    var r = data.raster;
    var colorbar = Object.assign({{title: {{text: "height (m)"}}}}, COLORBAR_POSITION);
    return [{{type: "heatmap", z: r.height, x0: r.x0, y0: r.y0, dx: r.dx, dy: r.dy,
              colorscale: "Viridis", colorbar: colorbar}}];
  }}
  function percentileOf(values, p) {{
    var sorted = values.slice().sort(function(a, b) {{ return a - b; }});
    var idx = Math.floor(p / 100 * (sorted.length - 1));
    return sorted[idx];
  }}
  function gridPercentile(grid, p) {{
    var values = [];
    for (var i = 0; i < grid.length; i++) {{
      for (var j = 0; j < grid[i].length; j++) {{
        if (grid[i][j] !== null) values.push(grid[i][j]);
      }}
    }}
    return percentileOf(values, p);
  }}
  function rasterIntensityTraces(data) {{
    var r = data.raster;
    var colorbar = Object.assign({{title: {{text: "intensity"}}}}, COLORBAR_POSITION);
    // Clips at the 95th percentile rather than the real max -- see
    // raster_intensity_traces' docstring (build_bev_report.py) for why.
    var zmax = gridPercentile(r.intensity, 95);
    return [{{type: "heatmap", z: r.intensity, x0: r.x0, y0: r.y0, dx: r.dx, dy: r.dy,
              colorscale: "Viridis", zmin: 0, zmax: zmax, colorbar: colorbar}}];
  }}
  function rasterAgeTraces(data) {{
    var r = data.raster;
    var colorbar = Object.assign({{title: {{text: "ms before keyframe"}}}}, COLORBAR_POSITION);
    return [{{type: "heatmap", z: r.age, x0: r.x0, y0: r.y0, dx: r.dx, dy: r.dy,
              colorscale: "Viridis", reversescale: true, zmin: 0, colorbar: colorbar}}];
  }}
  function boxTraces(data) {{
    var seen = {{}};
    return data.boxes.map(function(box) {{
      var color = CATEGORY_COLORS[box.category] || OTHER_CATEGORY_COLOR;
      var showLegend = !seen[box.category];
      seen[box.category] = true;
      return {{x: box.x, y: box.y, mode: "lines", type: "scatter",
               line: {{color: color, width: 1.5}},
               name: box.category, legendgroup: box.category, showlegend: showLegend}};
    }});
  }}
  // No boxTraces() on either 3D view: box footprints are flat 2D lines on
  // the default xaxis/yaxis, not the 3D scene -- overlaying them would draw
  // on an unrelated, unaligned coordinate system. 3D box wireframes are a
  // real feature, just not part of this promotion (see scatter_3d_traces'
  // Python docstring).
  function scatter3dTraces(data) {{
    var c = data.scatter.colored, u = data.scatter.uncovered;
    return [
      {{x: u.x, y: u.y, z: u.z, mode: "markers", type: "scatter3d",
        marker: {{size: 2, color: "{UNCOVERED_COLOR}", opacity: 0.6}}, name: "no camera coverage",
        showlegend: false}},
      coverageLegendEntry(true),
      {{x: c.x, y: c.y, z: c.z, mode: "markers", type: "scatter3d",
        marker: {{size: 2, color: c.color}}, name: "camera-colored", showlegend: false}}
    ];
  }}
  function scatter3dIntensityTraces(data) {{
    var i = data.scatter.intensity;
    var colorbar = Object.assign({{title: {{text: "intensity"}}}}, COLORBAR_POSITION);
    var zmax = percentileOf(i.value, 95);
    return [{{x: i.x, y: i.y, z: i.z, mode: "markers", type: "scatter3d",
              marker: {{size: 2, color: i.value, colorscale: "Viridis", cmin: 0, cmax: zmax,
                        colorbar: colorbar}}}}];
  }}

  var VIEW_TRACE_BUILDERS = {{
    raster: rasterTraces,
    scatter: scatterTraces,
    raster_height: rasterHeightTraces,
    raster_intensity: rasterIntensityTraces,
    scatter_3d: scatter3dTraces,
    scatter_3d_intensity: scatter3dIntensityTraces,
    raster_age: rasterAgeTraces
  }};
  var VIEWS_WITH_BOXES = {{raster: true, scatter: true, raster_height: true, raster_intensity: true,
                          raster_age: true}};
  // Multisweep ships raster data only (see multisweep_bev_data), and source
  // age only means something when there's more than one sweep.
  var VIEWS_BY_SWEEPS = {{
    single: ["raster", "scatter", "raster_height", "raster_intensity", "scatter_3d", "scatter_3d_intensity"],
    multi: ["raster", "raster_height", "raster_intensity", "raster_age"]
  }};

  // Multisweep mode swaps in the merged raster; boxes stay the keyframe's.
  function sweepsData(data, sweeps) {{
    return sweeps === "multi" ? {{raster: data.multisweep.raster, boxes: data.boxes}} : data;
  }}
  function caption(data, sweeps) {{
    if (sweeps !== "multi") return "";
    var m = data.multisweep;
    return m.sweeps + " sweeps over the " + m.span_ms + " ms before the keyframe: " +
           m.occupied_gain.toFixed(2) + "\u00d7 the occupied cells of the keyframe sweep alone.";
  }}
  function syncViewOptions(sweeps) {{
    var select = document.getElementById("bev-view-select");
    var allowed = VIEWS_BY_SWEEPS[sweeps];
    for (var i = 0; i < select.options.length; i++) {{
      select.options[i].disabled = allowed.indexOf(select.options[i].value) < 0;
    }}
    if (allowed.indexOf(select.value) < 0) select.value = "raster";
  }}

  function bevRender(data, view, sweeps) {{
    document.getElementById("bev-caption").textContent = caption(data, sweeps);
    data = sweepsData(data, sweeps);
    var baseTraces = VIEW_TRACE_BUILDERS[view](data);
    var traces = VIEWS_WITH_BOXES[view] ? baseTraces.concat(boxTraces(data)) : baseTraces;
    Plotly.react("bev-chart", traces, BEV_LAYOUT);
  }}

  function bevOnChange() {{
    var sweeps = document.getElementById("bev-sweeps-select").value;
    syncViewOptions(sweeps);
    var view = document.getElementById("bev-view-select").value;
    var token = document.getElementById("bev-sample-select").value;
    if (BEV_DATA_CACHE[token]) {{
      bevRender(BEV_DATA_CACHE[token], view, sweeps);
    }} else {{
      fetch("bev_data/" + token + ".json")
        .then(function(resp) {{ return resp.json(); }})
        .then(function(data) {{ BEV_DATA_CACHE[token] = data; bevRender(data, view, sweeps); }});
    }}
  }}
  document.getElementById("bev-sweeps-select").addEventListener("change", bevOnChange);

  document.getElementById("bev-view-select").addEventListener("change", bevOnChange);
  document.getElementById("bev-sample-select").addEventListener("change", bevOnChange);
}})();
</script>
<div class="prose">
<h3>What each view shows</h3>
<p>All views are in the keyframe's ego frame. The 2D ones are top-down, the car at
the origin facing up the chart, within {GRID_HALF_EXTENT_METERS:.0f} m of it on each side.</p>
<ul>
<li><strong>Raster (color):</strong> the points binned into a grid of {CELL_SIZE_METERS} m cells. Each
cell shows the camera colour of the point nearest the sensor that landed in it, so it reads like a
top-down photo. Blue cells had LiDAR returns but no camera view; white cells had no return.
Box footprints are drawn on top.</li>
<li><strong>Scatter (color):</strong> every LiDAR point plotted on its own, with the same camera colours
and boxes. Nothing is merged into cells, so it shows the true point density: dense near the car,
sparse rings further out.</li>
<li><strong>Raster (height):</strong> the same grid, coloured by the height in metres of each cell's
nearest point, measured from the ego frame's origin near road level. The road is flat and dark;
cars, walls and vegetation stand out brighter.</li>
<li><strong>Raster (intensity):</strong> the same grid, coloured by how strongly the surface reflected
the laser. Lane paint, signs and number plates are typically bright, asphalt dark. The scale stops short of
the few brightest cells so they don't flatten the rest.</li>
<li><strong>Scatter 3D (color)</strong> and <strong>Scatter 3D (intensity):</strong> every point at its
real height, with camera colour or intensity; drag to rotate. Boxes aren't drawn here, because
they are flat footprints. Single sweep only: the 3D views plot every point, and a multisweep window
has about {WINDOW_LENGTH} times as many, which would make each sample's download several times larger
and the 3D view slow to rotate. The rasters stay the same size however many sweeps feed them,
so multisweep offers rasters only.</li>
<li><strong>Raster (source sweep age):</strong> multisweep only. How many milliseconds before the
keyframe each cell's point was captured: the keyframe's own sweep is 0, older sweeps are larger.
It shows which gaps in the keyframe's own sweep the older sweeps filled.</li>
</ul>
<h3>Box colours</h3>
<p>The {len(colored_categories)} most frequent categories across these
samples, counted each time the page is built, get their own colour:</p>
<ul>
{category_items}
</ul>
<p>The colours were checked to stay distinguishable from each other and from the blue "no camera
coverage", including for colour-blind viewers. More colours failed that check, so every other
category is drawn in near-black, though each still gets its own legend entry.</p>
</div>
</section>
"""

    with open(SECTION_FILE, "w") as outfile:
        outfile.write(section_html)
    print(f"Wrote {SECTION_FILE}")
