"""Build the correctness section of the published report.

Combines two sources into one HTML fragment, docs/_correctness_section.html
(not a full page -- see build_report.py, which will combine this with the
performance section into one docs/index.html):

  - reports/evaluate_projections_data.jsonl (every annotation/camera pair
    evaluate_projections.py compared, see that script): an IOU-vs-area
    scatter (full range), an IOU-vs-area box plot (binned, zoomed), an
    IOU-vs-category box plot, and an IOU-vs-camera-channel box plot, built
    from the *full*, unfiltered set of records -- including the tiny,
    edge-clipped slivers visualize_projections.py's MIN_BBOX_AREA floor
    excludes from image selection. Showing them here is deliberate: the
    scatter is what makes visible *why* that floor exists (low-area boxes
    correlate with lower, though still very high, IOU -- see
    build_iou_vs_area_chart's note for the actual numbers).
  - reports/selected_projections.json (visualize_projections.py's 5
    worst-IOU + 5 best-IOU diverse picks, with their rendered images): laid
    out as a two-column captioned image gallery.

Math notation in the chart-note explanations (the IOU and bbox-area
formulas) is rendered via KaTeX -- see av_data_pipeline.report_html.

Run with: python scripts/build_correctness_report.py (from the repo root,
after both evaluate_projections.py and visualize_projections.py have run).
"""

import json
import math
import os
import statistics

import plotly.graph_objects as go

from av_data_pipeline.report_html import KATEX_CDN_HTML

DATA_FILE = "reports/evaluate_projections_data.jsonl"
SELECTED_FILE = "reports/selected_projections.json"
REPORT_DIR = "docs"
OUTPUT_FILE = os.path.join(REPORT_DIR, "_correctness_section.html")

# A single, consistent hue for every chart: none of them encode category
# identity through color (the box plots already encode it through x
# position), so a per-series color cycle would be noise, not signal.
MARK_COLOR = "#4C78A8"

# Headroom added on each side of the data-driven IOU zoom range (see
# compute_iou_zoom_range) as a fraction of the actual min-max span, so
# points at the extremes aren't drawn flush against the plot border.
ZOOM_PADDING_FRACTION = 0.1


def _add_box_trace(fig: go.Figure, values: list[float], name: str) -> None:
    """Add one box trace with whiskers at the true min/max, not Plotly's default 1.5*IQR fence.

    This dataset's IQR at any of these groupings (category, camera, area
    bin) is ~1e-6 -- about 300x smaller than the ~3e-4 range the zoomed
    y-axis actually shows. Plotly's default box statistics flag anything
    outside 1.5*IQR as an "outlier" and draw it as an individual point
    instead of folding it into the box/whiskers; with an IQR this tiny,
    that was most of each group's tail (up to ~15% of its points), so what
    rendered was a box collapsed to a sliver plus a cloud of individual
    dots spread across the whole visible range -- indistinguishable from a
    scatter plot. Passing precomputed quartiles with the fences pinned to
    the actual min/max removes the outlier-point mode entirely: every
    point is represented by the box or its whiskers, and the chart reads
    as an actual box-and-whisker shape.

    Args:
        fig: Figure to add the trace to.
        values: The group's raw values (e.g. one category's IOUs).
        name: The box's label (x-axis tick / legend-less identifier).
    """
    values = sorted(values)
    n = len(values)
    q1 = statistics.median(values[: n // 2]) if n >= 2 else values[0]
    q3 = statistics.median(values[(n + 1) // 2:]) if n >= 2 else values[0]
    fig.add_trace(go.Box(
        x=[name], q1=[q1], median=[statistics.median(values)], q3=[q3],
        lowerfence=[values[0]], upperfence=[values[-1]],
        name=name, marker=dict(color=MARK_COLOR), line=dict(color=MARK_COLOR),
    ))


def compute_iou_zoom_range(records: list[dict]) -> list[float]:
    """Compute a zoomed IOU axis range from the data itself, not a guess.

    A fixed guess like [0.9, 1.05] can be wildly wider than where the data
    actually falls -- on this dataset every IOU is within [0.9997, 1.0000],
    a span of ~0.0003, so a [0.9, 1.05] range would still render as "every
    point sits on the IOU=1 line". Computing the range from the real
    min/max instead makes the zoom adapt to however tight or wide the
    actual spread is, on this dataset or a future one.

    Args:
        records: Every (annotation, camera) record.

    Returns:
        [min - padding, max + padding], Plotly's yaxis.range format.
    """
    ious = [r["iou"] for r in records]
    lo, hi = min(ious), max(ious)
    padding = (hi - lo) * ZOOM_PADDING_FRACTION
    return [lo - padding, hi + padding]


def load_projection_records(data_file: str) -> list[dict]:
    """Load every record written by evaluate_projections.py.

    Args:
        data_file: Path to the JSONL file (see evaluate_projections.py).

    Returns:
        One dict per (annotation, camera) pair.
    """
    records = []
    with open(data_file) as fp:
        for line in fp:
            records.append(json.loads(line))

    return records


def build_iou_vs_area_chart(records: list[dict]) -> go.Figure:
    """Scatter: IOU vs. projected bbox area, log-scaled, over every record.

    Args:
        records: Every (annotation, camera) record.

    Returns:
        The scatter figure.
    """
    fig = go.Figure(go.Scatter(
        x=[r["area"] for r in records],
        y=[r["iou"] for r in records],
        mode="markers",
        marker=dict(size=4, color=MARK_COLOR, opacity=0.35, line=dict(width=0)),
    ))
    fig.update_xaxes(type="log", title="Projected bbox area (px^2, log scale)")
    fig.update_yaxes(title="IOU", range=[0, 1.02])
    fig.update_layout(title="IOU vs. projected bbox area (every annotation/camera pair, full range)",
                      showlegend=False)
    return fig


AREA_BIN_COUNT = 8


def compute_log_area_bins(records: list[dict], bin_count: int = AREA_BIN_COUNT) -> list[float]:
    """Compute log-spaced bbox-area bin edges spanning the data's actual area range.

    Log-spaced (not equal-width) bins, since area itself spans single-digit
    slivers to frame-filling boxes -- equal-width bins would put almost
    every record in the first one.

    Args:
        records: Every (annotation, camera) record.
        bin_count: Number of bins to divide the area range into.

    Returns:
        bin_count + 1 edges, from the smallest to the largest area.
    """
    areas = [r["area"] for r in records if r["area"] > 0]
    log_lo, log_hi = math.log10(min(areas)), math.log10(max(areas))
    step = (log_hi - log_lo) / bin_count
    return [10 ** (log_lo + i * step) for i in range(bin_count + 1)]


def build_iou_vs_area_binned_chart(records: list[dict], zoom_range: list[float]) -> go.Figure:
    """Box plot: IOU distribution per log-spaced bbox-area bin, zoomed to the data's actual IOU range.

    A separate chart from build_iou_vs_area_chart, not a zoomed copy of it:
    a zoomed *scatter* of this data is just a dense horizontal smear (every
    point's IOU is within a few thousandths of 1.0, and there are ~17.7k of
    them), which hides exactly the small-vs-large-object pattern this chart
    exists to show. Binning into log-spaced area buckets and drawing one box
    per bucket -- the same approach as the category/camera charts -- makes
    that pattern legible instead.

    Args:
        records: Every (annotation, camera) record.
        zoom_range: [min, max] y-axis range, see compute_iou_zoom_range.

    Returns:
        The binned box-plot figure, one box per area bin, smallest to largest.
    """
    edges = compute_log_area_bins(records)

    fig = go.Figure()
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        is_last_bin = i == len(edges) - 2
        ious = [r["iou"] for r in records
               if ((lo <= r["area"] <= hi) if is_last_bin else (lo <= r["area"] < hi))]
        if not ious:
            continue
        _add_box_trace(fig, ious, f"{lo:,.0f}–{hi:,.0f}")

    fig.update_layout(title="IOU distribution by projected bbox area (log-spaced bins, zoomed to the data's actual range)",
                      yaxis_title="IOU", xaxis_title="Projected bbox area (px^2)", showlegend=False)
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(range=zoom_range)
    return fig


def build_iou_vs_category_chart(records: list[dict], zoom_range: list[float]) -> go.Figure:
    """Box plot: IOU distribution per category, over every record.

    Args:
        records: Every (annotation, camera) record.
        zoom_range: [min, max] y-axis range, see compute_iou_zoom_range.

    Returns:
        The box-plot figure, one box per category in alphabetical order
        (which also groups categories by their shared dot-prefix, e.g.
        every "vehicle.*" together).
    """
    categories = sorted({r["category"] for r in records})

    fig = go.Figure()
    for category in categories:
        ious = [r["iou"] for r in records if r["category"] == category]
        _add_box_trace(fig, ious, category)

    fig.update_layout(title="IOU distribution by category (zoomed to the data's actual range)",
                      yaxis_title="IOU", xaxis_title="Category", showlegend=False)
    fig.update_xaxes(tickangle=45)
    # Box/whisker statistics (including the min/max fences -- see
    # _add_box_trace) are computed from the full data regardless of the
    # visible range, so only a whisker's tip gets visually clipped here,
    # not the median/quartiles that are this chart's main signal.
    fig.update_yaxes(range=zoom_range)
    return fig


def build_iou_vs_camera_chart(records: list[dict], zoom_range: list[float]) -> go.Figure:
    """Box plot: IOU distribution per camera channel, over every record.

    Args:
        records: Every (annotation, camera) record.
        zoom_range: [min, max] y-axis range, see compute_iou_zoom_range.

    Returns:
        The box-plot figure, one box per camera channel, alphabetically
        ordered (CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT, CAM_FRONT,
        CAM_FRONT_LEFT, CAM_FRONT_RIGHT).
    """
    channels = sorted({r["camera_channel"] for r in records})

    fig = go.Figure()
    for channel in channels:
        ious = [r["iou"] for r in records if r["camera_channel"] == channel]
        _add_box_trace(fig, ious, channel)

    fig.update_layout(title="IOU distribution by camera channel (zoomed to the data's actual range)",
                      yaxis_title="IOU", xaxis_title="Camera channel", showlegend=False)
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(range=zoom_range)
    return fig


def build_stats_grid_html(records: list[dict]) -> str:
    """A small grid of aggregate IOU stat tiles across every record.

    Args:
        records: Every (annotation, camera) record.

    Returns:
        An HTML snippet: one tile per stat.
    """
    ious = [r["iou"] for r in records]
    stats = [
        ("Pairs compared", f"{len(records):,}"),
        ("Mean IOU", f"{statistics.mean(ious):.6f}"),
        ("Median IOU", f"{statistics.median(ious):.6f}"),
        ("Min IOU", f"{min(ious):.6f}"),
    ]

    tiles = "\n".join(
        "<div style=\"padding:12px 20px; border:1px solid #ccc; border-radius:6px; "
        "text-align:center; min-width:120px;\">"
        f"<div style=\"font-size:0.85em; color:#666;\">{label}</div>"
        f"<div style=\"font-size:1.4em; font-weight:600;\">{value}</div></div>"
        for label, value in stats
    )
    return f"<div style=\"display:flex; gap:12px; flex-wrap:wrap; margin:12px 0;\">{tiles}</div>"


def _build_gallery_card_html(r: dict) -> str:
    """One image + caption for the gallery grid.

    Image paths in `r` are relative to the repo root (as written by
    visualize_projections.py, e.g. "docs/correctness_images/..."); this
    fragment lives at OUTPUT_FILE (also under REPORT_DIR), so the path is
    re-rooted relative to REPORT_DIR for the <img> tag to resolve.

    Args:
        r: One record from reports/selected_projections.json.

    Returns:
        An HTML snippet: one <figure>.
    """
    image_rel = os.path.relpath(r["image"], REPORT_DIR)
    return (
        "<figure style=\"margin:0 0 16px 0; text-align:center;\">"
        f"<img src=\"{image_rel}\" style=\"max-width:100%; display:block; margin:0 auto;\">"
        f"<figcaption>{r['category']} &middot; IOU={r['iou']:.6f}</figcaption>"
        "</figure>"
    )


def build_gallery_html(selected: list[dict]) -> str:
    """Two-column image gallery: worst 5 in one column, best 5 in the other.

    Args:
        selected: Records from reports/selected_projections.json, each
            with "group" ("worst"/"best") and "image".

    Returns:
        An HTML snippet: a two-column grid, each column titled and sorted
        worst-to-least-worst or best-to-least-best.
    """
    worst = sorted((r for r in selected if r["group"] == "worst"), key=lambda r: r["iou"])
    best = sorted((r for r in selected if r["group"] == "best"), key=lambda r: -r["iou"])

    worst_cards = "\n".join(_build_gallery_card_html(r) for r in worst)
    best_cards = "\n".join(_build_gallery_card_html(r) for r in best)

    return (
        "<div style=\"display:grid; grid-template-columns: 1fr 1fr; gap:24px;\">\n"
        f"<div><h4>Worst 5 (lowest IOU)</h4>{worst_cards}</div>\n"
        f"<div><h4>Best 5 (highest IOU)</h4>{best_cards}</div>\n"
        "</div>"
    )


def build_report(records: list[dict], selected: list[dict]) -> None:
    """Write the correctness section's charts, summary, and gallery as an HTML fragment.

    Args:
        records: Every (annotation, camera) record (for the charts).
        selected: The worst-5/best-5 picks (for the gallery).
    """
    os.makedirs(REPORT_DIR, exist_ok=True)

    zoom_range = compute_iou_zoom_range(records)
    ious = [r["iou"] for r in records]
    zoom_lo, zoom_hi = min(ious), max(ious)

    charts = [
        ("IOU vs. projected bbox area (full range)",
         build_iou_vs_area_chart(records),
         "Every annotation/camera pair compared, full IOU range. Each point is one "
         "pair; x-axis is our computed bbox's area in pixels (log scale, since areas "
         "span single-digit slivers to frame-filling boxes). On this dataset every "
         "point already sits within \\([0.9997, 1.0000]\\) of IOU regardless of area "
         "-- see the zoomed chart below for the real spread -- but low-area points "
         "still correlate with the lower end of that narrow band: IOU is more "
         "sensitive to a fixed absolute coordinate difference once a box's area is "
         "small, so low-area points are partly a measurement-sensitivity artifact, "
         "not purely evidence of a worse projection. That sensitivity is why "
         "visualize_projections.py's worst/best image selection excludes boxes below "
         "a minimum area."),
        ("IOU distribution by projected bbox area (binned)",
         build_iou_vs_area_binned_chart(records, zoom_range),
         f"Same data as the chart above, but grouped into {AREA_BIN_COUNT} log-spaced "
         f"bbox-area bins and zoomed to the data's actual IOU range "
         f"(\\([{zoom_lo:.6f}, {zoom_hi:.6f}]\\), not a guessed window). A zoomed "
         "*scatter* of this data would just be a dense horizontal smear -- every "
         "pair's IOU differs from every other's by at most a few thousandths, and "
         "there are ~17.7k of them -- so binning is what makes the small-vs-large "
         "pattern legible: a visibly lower median or wider spread in the small-area "
         "bins (left) than the large-area ones (right) is the area-sensitivity effect "
         "described above; a flat median across bins would mean area isn't actually "
         "driving the differences."),
        ("IOU distribution by category",
         build_iou_vs_category_chart(records, zoom_range),
         "IOU distribution per nuScenes category (every pair in that category, not "
         "just the 10 in the gallery below), zoomed to the same data-driven range. A "
         "category with a visibly lower median or wider spread than the rest would "
         "suggest the pipeline handles that category's typical size, shape, or "
         "occlusion pattern less well than others; points outside the visible range "
         "are still included in the box/median statistics, only clipped from view."),
        ("IOU distribution by camera channel",
         build_iou_vs_camera_chart(records, zoom_range),
         "IOU distribution per camera channel. Six cameras share the same "
         "intrinsics/extrinsics pipeline code, so a channel with a systematically "
         "lower median or wider spread than the others would point at a "
         "channel-specific issue -- e.g. a calibration or distortion-handling bug "
         "specific to that camera -- rather than something wrong with the shared "
         "projection logic itself."),
    ]
    chart_html = "\n".join(
        f"<h3>{heading}</h3>\n<p class=\"chart-note\">{note}</p>\n" +
        fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False)
        for i, (heading, fig, note) in enumerate(charts)
    )

    iou_formula_html = (
        "<p class=\"chart-note\">Every chart below is built on IOU "
        "(intersection-over-union) between our computed 2D bbox and the nuScenes "
        "reference bbox: \\[\\text{IOU} = \\frac{|\\text{Our box} \\cap "
        "\\text{Reference box}|}{|\\text{Our box} \\cup \\text{Reference box}|}\\] "
        "1 means the two boxes match exactly; 0 means they don't overlap at all. "
        "Area (used on the x-axis below) is computed the same way for a single "
        "clipped box: \\[\\text{Area} = \\max\\big(0,\\ (x_{max}-x_{min}) "
        "\\cdot (y_{max}-y_{min})\\big)\\]</p>"
    )

    section_html = (
        "<section id=\"correctness\">\n"
        f"{KATEX_CDN_HTML}\n"
        "<h2>Correctness: projection accuracy vs. reference</h2>\n"
        f"{build_stats_grid_html(records)}\n"
        f"{iou_formula_html}\n"
        f"{chart_html}\n"
        "<h3>Sample localizations</h3>\n"
        "<p class=\"chart-note\">10 annotation/camera pairs rendered with both boxes "
        "drawn (red: ours, green: reference) -- the 5 lowest-IOU and 5 highest-IOU "
        "pairs among eligible (non-degenerate) boxes, each deduplicated by physical "
        "object and spread across categories. See the stat grid and charts above for "
        "the quantitative picture; these are a qualitative, representative sample of "
        "what localizations actually look like across the quality spectrum.</p>\n"
        f"{build_gallery_html(selected)}\n"
        "</section>\n"
    )

    with open(OUTPUT_FILE, "w") as outfile:
        outfile.write(section_html)

    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    records = load_projection_records(DATA_FILE)
    with open(SELECTED_FILE) as fp:
        selected = json.load(fp)

    build_report(records, selected)
