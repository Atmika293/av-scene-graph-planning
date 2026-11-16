"""Render a multisweep BEV raster next to the single reference sweep it extends.

For each of the first NUM_SCENES scenes, builds the sweep window from
bev.multisweep (the reference keyframe sweep plus the sweeps before it, each
coloured by its own nearest-in-time camera frames), rasterizes each sweep on
one shared grid, merges them newest first with bev.raster.merge_rasters, and
draws four panels:

  1. the reference sweep alone (what the existing single-sample BEV shows),
  2. the merged window, same colour scheme, so the coverage gain is direct,
  3. the merged window's height channel,
  4. which sweep each merged cell came from, as its time before the reference,
     so the newest-first merge can be checked by eye: cells near the vehicle
     should be the reference's, older sweeps should only appear where newer
     ones had no point.

In both colour panels, coloured cells show their real camera colour (often
greys: asphalt, concrete, shadowed foliage), white is an empty cell, and blue
an occupied cell no camera coloured, so gaps in coverage are visible rather
than blending in with genuinely grey ground. The
reference keyframe's annotated boxes are overlaid as footprints on every
panel. Moving objects aren't removed from older sweeps, so a vehicle that
moved during the window can leave a trail of older positions in panel 2.

This is a debug/sanity-check render like render_bev.py: nothing here is
scored against a reference.

Run with: python scripts/viz/render_multisweep.py (from the repo root).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from nuscenes.nuscenes import NuScenes

from av_data_pipeline.config import NUSCENES_VERSION, NUSCENES_DATAROOT
from av_data_pipeline.ingestion import extract_sample_data_from_sample
from bev.multisweep import assemble_window, project_window_to_bev
from bev.raster import build_raster, merge_rasters, to_plot_xy
from bev.sample import SampleBoxProjection

IMAGE_DIR = "reports/multisweep_images"
NUM_SCENES = 3
GRID_HALF_EXTENT_METERS = 25.0  # same grid as the validated scripts/validation/check_raster.py
CELL_SIZE_METERS = 0.25
BOX_COLOR = "#ff00ff"
# Same blue as scripts/report/build_bev_report.py's RASTER_UNCOLORED_COLOR (see
# its comment for why not grey), as an RGB fraction for the RGBA image.
UNCOLORED_CELL_RGB = np.array([42, 120, 214]) / 255.0

# Same as render_bev.py: corners 0, 2, 4, 6 are a box's top face in perimeter
# order (see SampleAnnotation's corner layout); repeating 0 closes the outline.
FOOTPRINT_CORNER_ORDER = [0, 2, 4, 6, 0]


def color_image(raster: dict) -> np.array:
    """RGBA image of a raster's colour channel: white empty, blue uncoloured, else the cell's colour.

    Args:
        raster: A build_raster or merge_rasters result.

    Returns:
        (H, W, 4) float RGBA image, for imshow with origin="lower".
    """
    rgba = np.ones((*raster["occupied_mask"].shape, 4))
    rgba[raster["occupied_mask"], :3] = UNCOLORED_CELL_RGB
    rgba[raster["colored_mask"], :3] = raster["color"][raster["colored_mask"]] / 255.0
    return rgba


def camera_colour_examples(colors: np.array, count: int = 5) -> list[np.array]:
    """A few real camera colours from this image, dark to light, for the legend swatch.

    A single fixed swatch can't stand for "each point's own pixel colour", and
    an empty outline reads as "nothing here". Picking colours at evenly spaced
    brightness percentiles shows what camera-coloured points actually look
    like in this image (mostly greys, on road scenes).

    Args:
        colors: Camera-coloured RGB values (M, 3), 0-255.
        count: How many example colours to return.

    Returns:
        count RGB colours as 0-1 floats, darkest first.
    """
    if len(colors) == 0:
        return [np.zeros(3)]
    brightness = colors @ np.array([0.299, 0.587, 0.114])
    order = np.argsort(brightness)
    picks = order[(np.linspace(0.05, 0.95, count) * (len(order) - 1)).astype(int)]
    return [colors[i] / 255.0 for i in picks]


def draw_boxes(ax: "plt.Axes", boxes: dict) -> None:
    """Overlay ego-frame box footprints (SampleBoxProjection.project_to_ego) on a plot-frame axis."""
    for corners in boxes.values():
        footprint_x, footprint_y = to_plot_xy(corners[FOOTPRINT_CORNER_ORDER])
        ax.plot(footprint_x, footprint_y, c=BOX_COLOR, linewidth=0.8)


def render_multisweep(reference: dict, merged: dict, sweep_ages_ms: list[float],
                      boxes: dict, title: str) -> "plt.Figure":
    """Four-panel comparison of the reference sweep's raster and the merged window.

    Args:
        reference: The reference sweep's own build_raster result.
        merged: merge_rasters' result for the whole window.
        sweep_ages_ms: Each sweep's time before the reference sweep, in
            milliseconds, indexed like merged["source"].
        boxes: The reference keyframe's ego-frame box corners.
        title: Figure title.

    Returns:
        The rendered matplotlib Figure.
    """
    extent = merged["extent"]
    viridis = plt.get_cmap("viridis").copy()
    viridis.set_bad(color="white")

    age = np.full(merged["source"].shape, np.nan)
    occupied = merged["source"] >= 0
    age[occupied] = np.array(sweep_ages_ms)[merged["source"][occupied]]

    fig, axes = plt.subplots(2, 2, figsize=(13, 13))
    axes = axes.ravel()

    # imshow with origin="lower" puts row 0 at the bottom, matching
    # build_raster's row = f(plot_y), so forward stays up (see check_raster.py).
    axes[0].imshow(color_image(reference), origin="lower", extent=extent)
    axes[0].set_title(f"reference sweep only: {reference['occupied_mask'].sum():,} cells, "
                      f"{reference['colored_mask'].sum():,} coloured")

    axes[1].imshow(color_image(merged), origin="lower", extent=extent)
    axes[1].set_title(f"merged {len(sweep_ages_ms)} sweeps: {merged['occupied_mask'].sum():,} cells, "
                      f"{merged['colored_mask'].sum():,} coloured")

    height = axes[2].imshow(np.ma.masked_invalid(merged["height"]), origin="lower", extent=extent, cmap=viridis)
    axes[2].set_title("merged height (ego z, m)")
    fig.colorbar(height, ax=axes[2], fraction=0.046)

    source = axes[3].imshow(np.ma.masked_invalid(age), origin="lower", extent=extent,
                            cmap=viridis.reversed(), vmin=0, vmax=max(sweep_ages_ms) or 1)
    axes[3].set_title("source sweep: ms before the reference")
    fig.colorbar(source, ax=axes[3], fraction=0.046)

    # The camera-colour entry is a row of real example colours from the merged
    # raster: one fixed swatch can't stand for "each cell's own pixel colour".
    examples = tuple(Patch(facecolor=colour)
                     for colour in camera_colour_examples(merged["color"][merged["colored_mask"]].astype(float)))
    color_legend_handles = [examples, Patch(facecolor=UNCOLORED_CELL_RGB),
                            Patch(facecolor="white", edgecolor="#999999"),
                            Line2D([], [], color=BOX_COLOR, linewidth=1)]
    color_legend_labels = ["camera colour (examples from this image)", "occupied, no camera coverage",
                           "empty", "annotation (reference keyframe)"]
    for ax in axes[:2]:
        ax.legend(handles=color_legend_handles, labels=color_legend_labels, loc="lower left",
                  fontsize=8, framealpha=0.9, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)})

    for ax in axes:
        draw_boxes(ax, boxes)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel("left <- ego y -> right (flipped)")
        ax.set_ylabel("ego x (forward)")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATAROOT, verbose=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    for scene in nusc.scene[:NUM_SCENES]:
        lidar_sweep_tokens, lidar_camera_groups = assemble_window(nusc, scene["token"])
        sweeps = project_window_to_bev(nusc, lidar_sweep_tokens, lidar_camera_groups)

        rasters = [build_raster(sweep["points"], sweep["colors"], sweep["intensity"],
                                GRID_HALF_EXTENT_METERS, CELL_SIZE_METERS, distance=sweep["distance"])
                   for sweep in sweeps]
        merged = merge_rasters(rasters)

        timestamps = [nusc.get("sample_data", token)["timestamp"] for token in lidar_sweep_tokens]
        sweep_ages_ms = [(timestamps[0] - timestamp) / 1000 for timestamp in timestamps]

        # The reference sweep is a keyframe, so its sample has annotations;
        # that sample's LIDAR_TOP record is the reference sweep itself, so
        # SampleBoxProjection puts the boxes in the same ego frame as the raster.
        reference_record = nusc.get("sample_data", lidar_sweep_tokens[0])
        boxes = SampleBoxProjection(
            extract_sample_data_from_sample(nusc, reference_record["sample_token"])).project_to_ego()

        fig = render_multisweep(rasters[0], merged, sweep_ages_ms, boxes,
                                title=f"Multisweep BEV -- {scene['name']}, reference {lidar_sweep_tokens[0]}")
        out_path = os.path.join(IMAGE_DIR, f"{scene['name']}.png")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)

        gain = merged["occupied_mask"].sum() / max(rasters[0]["occupied_mask"].sum(), 1)
        print(f"Wrote {out_path} ({len(sweeps)} sweeps over {sweep_ages_ms[-1]:.0f} ms, "
              f"{gain:.2f}x the reference sweep's occupied cells, {len(boxes)} boxes)")
