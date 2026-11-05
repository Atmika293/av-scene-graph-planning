"""Render GeometricBEVPipeline's output as 3D scatter PNGs -- a sanity-check
render for the Scatter3d (RGB) and 3D-by-intensity views, before either
becomes an interactive trace on docs/bev.html.

The underlying data (ego_points, colors, intensity) is already validated --
same GeometricBEVPipeline.project_to_bev output the published 2D scatter
uses, and intensity already cross-checked against the devkit's own loader
(scripts/validation/check_intensity.py). What's new, and what this script
exists to sanity-check by eye, is the 3D rendering itself: whether height
actually renders as "up" (the same category of bug caught empirically for
go.Image's default axis reversal -- see build_bev_report.py's module
docstring), whether the initial camera angle makes the scene legible, and
(for the intensity view) whether the colormap/normalization choice actually
shows useful contrast rather than washing everything into a narrow band.

Axis mapping: matplotlib's Axes3D renders Z vertically by convention, so
Z = ego z (height) directly matches "height renders as up" -- no axis
reinterpretation needed there, unlike go.Image. X/Y reuse the same
ground-plane mapping as every other view in this project (plot_x = -ego_y,
plot_y = ego_x), so a reader already familiar with the 2D scatter/raster
isn't learning a new convention for the ground plane, only gaining a third
(height) axis.

Points are limited to GRID_HALF_EXTENT_METERS (same bound as the published
raster/scatter's default zoom) for the same reason established there: raw
lidar range has a long thin tail past 100m that would otherwise dominate
the plot's scale and leave the actually-dense center looking sparse.

This is a debug/sanity-check render, not a report phase -- there's no
ground truth to score a 3D view against, so this is for visually confirming
the rendering looks right before any of it becomes an interactive page.

Run with: python scripts/viz/render_bev_3d.py (from the repo root, after
ingesting/having local access to the configured nuScenes dataset).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the 3d projection
import numpy as np
from nuscenes.nuscenes import NuScenes

from av_data_pipeline.config import NUSCENES_VERSION, NUSCENES_DATAROOT
from av_data_pipeline.ingestion import build_scene_data
from av_data_pipeline.ingestion_keys import SAMPLE_INGESTION_KEYS
from bev.fusion import GeometricBEVPipeline

IMAGE_DIR = "reports/bev_3d_images"
NUM_SAMPLES = 3
MARKER_SIZE = 2
GRID_HALF_EXTENT_METERS = 25.0  # same bound as the published raster/scatter's default zoom
UNCOVERED_COLOR = "#2a78d6"  # matches build_bev_report.py's UNCOVERED_COLOR, for consistency
ELEV, AZIM = 25, -50  # initial camera angle; tuned by eye, not a validated convention


def to_plot_xy(ego_xy: np.array) -> tuple[np.array, np.array]:
    """Map ego-frame (x, y) to plot (horizontal, vertical) so forward is up.

    Same convention as every other view in this project: ego x = forward,
    y = left, so plotting -y horizontally and x vertically puts forward
    toward the top/back of the scene.
    """
    return -ego_xy[:, 1], ego_xy[:, 0]


def render_bev_3d_rgb(ego_points: np.array, colors: np.array, title: str) -> "plt.Figure":
    """3D scatter colored by real camera-sampled RGB, uncovered points in blue.

    Args:
        ego_points: Lidar points (N, 3), ego frame.
        colors: Each point's RGB color (N, 3), or [-1, -1, -1] if no camera
            covered it (see GeometricBEVPipeline.project_to_bev).
        title: Plot title.

    Returns:
        The rendered matplotlib Figure.
    """
    plot_x, plot_y = to_plot_xy(ego_points)
    in_range = np.hypot(plot_x, plot_y) <= GRID_HALF_EXTENT_METERS
    valid = np.all(colors != -1, axis=-1) & in_range
    uncovered = ~np.all(colors != -1, axis=-1) & in_range

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(plot_x[uncovered], plot_y[uncovered], ego_points[uncovered, 2],
               s=MARKER_SIZE, c=UNCOVERED_COLOR, alpha=0.5, label="no camera coverage")
    ax.scatter(plot_x[valid], plot_y[valid], ego_points[valid, 2],
               s=MARKER_SIZE, c=colors[valid].astype(np.uint8) / 255.0, label="camera-colored")

    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xlabel("left <- ego y -> right (flipped)")
    ax.set_ylabel("ego x (forward)")
    ax.set_zlabel("ego z (up)")
    ax.legend(markerscale=5)
    ax.set_title(title)

    return fig


def render_bev_3d_intensity(ego_points: np.array, intensity: np.array, title: str) -> "plt.Figure":
    """3D scatter colored by lidar return intensity (viridis).

    Args:
        ego_points: Lidar points (N, 3), ego frame.
        intensity: Each point's lidar return intensity (N,), 0-255.
        title: Plot title.

    Returns:
        The rendered matplotlib Figure.
    """
    plot_x, plot_y = to_plot_xy(ego_points)
    in_range = np.hypot(plot_x, plot_y) <= GRID_HALF_EXTENT_METERS

    # A fixed 0-255 range wastes almost all visual contrast on rare bright
    # outliers: measured on real data, ~80% of points are below 30 and the
    # median is ~12, with a long thin tail out to 255 (confirmed, not
    # assumed). Clipping at the 95th percentile keeps the bulk of real
    # variation visible; the rare brighter returns simply clip to the top
    # color instead of needing the full range to accommodate them.
    vmax = np.percentile(intensity[in_range], 95)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    scatter = ax.scatter(plot_x[in_range], plot_y[in_range], ego_points[in_range, 2],
                          s=MARKER_SIZE, c=intensity[in_range], cmap="viridis", vmin=0, vmax=vmax)
    fig.colorbar(scatter, ax=ax, label="intensity", fraction=0.03, pad=0.1)

    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xlabel("left <- ego y -> right (flipped)")
    ax.set_ylabel("ego x (forward)")
    ax.set_zlabel("ego z (up)")
    ax.set_title(title)

    return fig


if __name__ == "__main__":
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATAROOT, verbose=True)

    scene_token = nusc.scene[0]["token"]
    scene_data = build_scene_data(nusc, scene_token)[:NUM_SAMPLES]

    os.makedirs(IMAGE_DIR, exist_ok=True)

    for sample_data in scene_data:
        sample_token = sample_data[SAMPLE_INGESTION_KEYS['token']]

        pipeline = GeometricBEVPipeline(sample_data)
        ego_points, colors = pipeline.project_to_bev()
        intensity = pipeline.lidar_projection.intensity

        fig_rgb = render_bev_3d_rgb(ego_points, colors, title=f"BEV 3D (RGB) -- sample {sample_token}")
        rgb_path = os.path.join(IMAGE_DIR, f"{sample_token}_rgb.png")
        fig_rgb.savefig(rgb_path, dpi=150)
        plt.close(fig_rgb)

        fig_intensity = render_bev_3d_intensity(ego_points, intensity,
                                                 title=f"BEV 3D (intensity) -- sample {sample_token}")
        intensity_path = os.path.join(IMAGE_DIR, f"{sample_token}_intensity.png")
        fig_intensity.savefig(intensity_path, dpi=150)
        plt.close(fig_intensity)

        print(f"Wrote {rgb_path}, {intensity_path}")
