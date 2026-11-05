"""Bins a sample's colored, ego-frame point cloud into a multi-channel BEV raster.

Multi-channel: unlike the scatter views, each grid cell holds several
aggregate values rather than a single continuous point -- color (R, G, B),
height (z), and lidar return intensity. Grid bounds and cell size are
configurable: a finer cell size looks sparser/noisier since point density
doesn't scale with cell count, a coarser one loses spatial detail.

Cell selection: multiple lidar points can land in the same (x, y) cell at
very different heights (a tree branch over a road, a car's roof vs. its
wheel well) -- averaging them would blend unrelated surfaces into a
meaningless value, so each occupied cell keeps only its nearest-to-ego
points: the surface actually nearest the sensor, rather than an arbitrary
top/bottom convention. Within that nearest-to-ego group, points can still
be exactly tied in distance -- a real lidar multi-return artifact (multiple
beams reporting the same physical spot with different intensity readings),
confirmed against real data, not floating-point noise -- and those ARE
averaged together, since they're repeated measurements of one spot rather
than different surfaces.

Two distinct kinds of "no data" are tracked separately: occupied_mask (did
any lidar point land in this cell at all) vs. colored_mask (did this cell's
nearest-to-ego group have at least one camera-colored member -- same
"uncovered" meaning as the scatter view's sentinel, just per-cell instead of
per-point). A cell can be occupied but uncolored; it can't be colored
without being occupied.

This is reused by both scripts/validation/check_raster.py (which validates
it, including an independent per-point-loop cross-check in
build_raster_naive) and scripts/report/build_bev_report.py (which publishes
it), so both run the exact same validated code rather than two copies that
could drift apart.
"""

import numpy as np


def to_plot_xy(ego_xy: np.array) -> tuple[np.array, np.array]:
    """Map ego-frame (x, y) to plot (horizontal, vertical) so forward is up.

    Same convention as render_bev.py's to_plot_xy: ego x = forward, y =
    left, so plotting -y horizontally and x vertically puts forward at the
    top of the image.
    """
    return -ego_xy[:, 1], ego_xy[:, 0]


def build_raster(ego_points: np.array, colors: np.array, intensity: np.array,
                  half_extent: float, cell_size: float) -> dict:
    """Bin a sample's colored, ego-frame point cloud into a multi-channel raster.

    Args:
        ego_points: Lidar points (N, 3), ego frame.
        colors: Each point's RGB color (N, 3), or [-1, -1, -1] if no camera
            covered it (see GeometricBEVPipeline.project_to_bev).
        intensity: Each point's lidar return intensity (N,).
        half_extent: Grid half-width/height in meters (grid covers
            [-half_extent, +half_extent] on both plot axes).
        cell_size: Grid cell size in meters.

    Returns:
        A dict with:
          occupied_mask (H, W) bool -- did any lidar point land in this cell.
          colored_mask (H, W) bool -- did this cell have at least one
              camera-colored point among its nearest-to-ego group (implies
              occupied_mask).
          color (H, W, 3) int -- average color over the nearest-to-ego
              group's colored members, 0 where not colored_mask.
          height (H, W) float -- average ego-frame z over the nearest-to-ego
              group, NaN where not occupied_mask.
          intensity (H, W) float -- average intensity over the same group,
              NaN where not occupied_mask.
          extent -- (left, right, bottom, top) in meters, for plt.imshow.
    """
    plot_x, plot_y = to_plot_xy(ego_points)

    grid_size = int(round(2 * half_extent / cell_size))
    col = np.floor((plot_x + half_extent) / cell_size).astype(int)
    row = np.floor((plot_y + half_extent) / cell_size).astype(int)
    in_bounds = (col >= 0) & (col < grid_size) & (row >= 0) & (row < grid_size)

    rows, cols = row[in_bounds], col[in_bounds]
    pt_colors = colors[in_bounds]
    pt_heights = ego_points[in_bounds, 2]
    pt_intensity = intensity[in_bounds]
    # pt_colors carries GeometricBEVPipeline's own per-point [-1,-1,-1]
    # sentinel (no camera covered that point).
    pt_colored = np.all(pt_colors != -1, axis=-1)
    distance = np.linalg.norm(ego_points[in_bounds], axis=1)

    num_cells = grid_size * grid_size
    cell_id = rows * grid_size + cols

    # Nearest-to-ego group: real lidar sweeps can return multiple points at
    # the exact same position (an exact distance tie -- a genuine multi-return
    # artifact, not floating-point noise, confirmed against real data). Rather
    # than picking one of them arbitrarily, average every point tied for
    # nearest in its cell -- they're repeated measurements of one physical
    # spot, not different surfaces (unlike points at different positions/
    # heights sharing a cell, which still shouldn't be blended -- see the
    # module docstring). A cell with no tie just averages a group of size 1,
    # so this needs no special case for the overwhelmingly common case.
    min_dist_per_cell = np.full(num_cells, np.inf)
    np.minimum.at(min_dist_per_cell, cell_id, distance)
    is_nearest = distance == min_dist_per_cell[cell_id]

    def scatter_mean(values: np.array, mask: np.array) -> tuple[np.array, np.array]:
        """Per-cell mean of values[mask], grouped by cell_id[mask]."""
        total = np.zeros((num_cells,) + values.shape[1:])
        count = np.zeros(num_cells)
        np.add.at(total, cell_id[mask], values[mask])
        np.add.at(count, cell_id[mask], 1)
        has_data = count > 0
        mean = np.full_like(total, np.nan, dtype=float)
        mean[has_data] = total[has_data] / count[has_data, None] if total.ndim > 1 else total[has_data] / count[has_data]
        return mean, has_data

    height_mean, occupied_flat = scatter_mean(pt_heights, is_nearest)
    intensity_mean, _ = scatter_mean(pt_intensity, is_nearest)
    color_mean, colored_flat = scatter_mean(pt_colors.astype(float), is_nearest & pt_colored)

    occupied_mask = occupied_flat.reshape(grid_size, grid_size)
    colored_mask = colored_flat.reshape(grid_size, grid_size)
    height = height_mean.reshape(grid_size, grid_size)
    intensity_channel = intensity_mean.reshape(grid_size, grid_size)
    # Cells with no colored member get 0 (never read -- always gated behind
    # colored_mask); color is uint8, so NaN (from scatter_mean's default)
    # can't be stored directly.
    color = np.where(colored_flat[:, None], np.round(color_mean), 0).astype(np.uint8).reshape(grid_size, grid_size, 3)

    return {
        "occupied_mask": occupied_mask,
        "colored_mask": colored_mask,
        "color": color,
        "height": height,
        "intensity": intensity_channel,
        "extent": (-half_extent, half_extent, -half_extent, half_extent),
    }
