"""Render GeometricBEVPipeline's output as top-down (bird's-eye-view) PNGs.

For each of the first NUM_SAMPLES samples in the first scene, builds a
colored point cloud via bev.fusion.project_scene_to_bev (LiDAR points in the
ego frame, colored by whichever camera saw each one -- see bev/fusion.py) and
scatter-plots it: forward (ego +x) oriented up, left (ego +y) oriented left,
points no camera covered drawn in blue so coverage gaps are visible rather
than just missing. Each sample's annotated boxes (via
bev.sample.SampleBoxProjection) are overlaid as footprint outlines, so
box/point alignment can be checked by eye.

This is a debug/sanity-check render, not yet a report phase (see the
README's roadmap) -- there's no ground truth to score it against the way
evaluate_projections.py scores the camera projection, so this is for
visually confirming the lidar/camera fusion looks right before it becomes one.

Run with: python scripts/viz/render_bev.py (from the repo root, after
ingesting/having local access to the configured nuScenes dataset).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
import numpy as np
from nuscenes.nuscenes import NuScenes

from av_data_pipeline.config import NUSCENES_VERSION, NUSCENES_DATAROOT
from av_data_pipeline.ingestion import build_scene_data
from av_data_pipeline.ingestion_keys import SAMPLE_INGESTION_KEYS
from bev.fusion import project_scene_to_bev
from bev.sample import SampleBoxProjection

IMAGE_DIR = "reports/bev_images"
NUM_SAMPLES = 3
MARKER_SIZE = 1
BOX_COLOR = "#ff00ff"
# Same blue as scripts/report/build_bev_report.py's UNCOVERED_COLOR (see its
# comment): camera-sampled colours are often greys, so a grey "no coverage"
# colour would blend in with genuinely grey points (asphalt, concrete).
UNCOVERED_COLOR = "#2a78d6"

# SampleAnnotation.__extract_corners (av_data_pipeline.transform) orders a
# box's 8 corners as top/bottom pairs; indices 0, 2, 4, 6 are its top face,
# which traces the box's footprint rectangle in perimeter order once
# projected to the ego x-y plane. Repeating index 0 closes the loop for plotting.
FOOTPRINT_CORNER_ORDER = [0, 2, 4, 6, 0]


def to_plot_xy(ego_xy: np.array) -> tuple[np.array, np.array]:
    """Map ego-frame (x, y) to plot (horizontal, vertical) so forward is up.

    Ego frame: x = forward, y = left (see av_data_pipeline.transform's
    module docstring / the correctness report). Plotting -y on the
    horizontal axis and x on the vertical axis puts "forward" at the top
    of the image and "left" on the left, instead of the arbitrary
    orientation the raw ego-frame axes would otherwise plot as.
    """
    return -ego_xy[:, 1], ego_xy[:, 0]


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


def render_bev(ego_points: np.array, colors: np.array, boxes: dict, title: str) -> "plt.Figure":
    """Scatter-plot one sample's colored point cloud, with box footprints overlaid.

    Args:
        ego_points: LiDAR points (N, 3), in the ego frame (see
            LidarProjection.lidar_to_ego).
        colors: Each point's RGB color (N, 3), or [-1, -1, -1] where no
            camera covered it (see GeometricBEVPipeline.project_to_bev).
        boxes: Mapping from annotation token to that annotation's ego-frame
            box corners (8, 3), as returned by SampleBoxProjection.project_to_ego.
        title: Plot title.

    Returns:
        The rendered matplotlib Figure.
    """
    valid = np.all(colors != -1, axis=-1)
    plot_x, plot_y = to_plot_xy(ego_points)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(plot_x[~valid], plot_y[~valid], s=MARKER_SIZE, c=UNCOVERED_COLOR)
    ax.scatter(plot_x[valid], plot_y[valid], s=MARKER_SIZE, c=colors[valid].astype(np.uint8) / 255.0)

    for corners in boxes.values():
        footprint_x, footprint_y = to_plot_xy(corners[FOOTPRINT_CORNER_ORDER])
        ax.plot(footprint_x, footprint_y, c=BOX_COLOR, linewidth=1)

    ax.set_aspect("equal")
    ax.set_xlabel("left <- ego y -> right (flipped)")
    ax.set_ylabel("ego x (forward)")
    # Built by hand rather than from the scatters: matplotlib takes a
    # per-point-coloured scatter's legend swatch from its first point, so the
    # "colored" swatch would be an arbitrary (often white) pixel colour that
    # changes from image to image. Instead it's a row of real example colours.
    examples = tuple(Line2D([], [], linestyle="", marker="o", markersize=7, color=colour)
                     for colour in camera_colour_examples(colors[valid]))
    ax.legend(handles=[
        examples,
        Line2D([], [], linestyle="", marker="o", markersize=7, color=UNCOVERED_COLOR),
        Line2D([], [], color=BOX_COLOR, linewidth=1),
    ], labels=["camera-coloured (examples from this image)", "no camera coverage", "annotation"],
       handler_map={tuple: HandlerTuple(ndivide=None, pad=0)})
    ax.set_title(title)

    return fig


if __name__ == "__main__":
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATAROOT, verbose=True)

    scene_token = nusc.scene[0]["token"]
    scene_data = build_scene_data(nusc, scene_token)[:NUM_SAMPLES]
    bev_results = project_scene_to_bev(scene_data)

    os.makedirs(IMAGE_DIR, exist_ok=True)

    for sample_data, (ego_points, colors) in zip(scene_data, bev_results):
        sample_token = sample_data[SAMPLE_INGESTION_KEYS['token']]
        boxes = SampleBoxProjection(sample_data).project_to_ego()
        fig = render_bev(ego_points, colors, boxes, title=f"BEV render -- sample {sample_token}")

        out_path = os.path.join(IMAGE_DIR, f"{sample_token}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        valid_fraction = np.mean(np.all(colors != -1, axis=-1))
        print(f"Wrote {out_path} ({valid_fraction:.1%} of {ego_points.shape[0]} points colored, {len(boxes)} boxes)")
