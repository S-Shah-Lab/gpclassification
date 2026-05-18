"""
Plot cross-validation metrics from one or more GP spatial-filter experiment runs.
Individual-canvas edition: every metric panel is rendered on its own figure
and saved as a separate file.

Each path supplied via ``--roots`` must point to a directory that contains
``nf_<k>/fold_<i>/[run_<timestamp>/]run_log.json`` sub-trees — identical to the
original ``plot_metric_…`` script.

Key differences from the original
----------------------------------
- **One figure per metric panel** — each plot is a self-contained, publication-
  ready canvas with its own title, legend, and axis labels.
- **Whisker error bars** with capped stems replace shaded bands; both SEM and STD
  are supported via ``--err``.  Individual fold values can still be overlaid as a
  jittered strip with ``--show-folds``.
- **Chance-level reference lines** on accuracy and AUC-ROC panels (0.5,
  dashed in a neutral grey), clearly annotated.
- **Scalable legend** placed outside the plot area (right side) so up to 10
  overlapping series never obscure the data.  Labels wrap automatically.
- **Refined typography and layout**: generous padding, no top/right spines, subtle
  background grid lines, careful font hierarchy, metadata chips in the header.
- **Save directory**: ``--save-dir`` writes one file per panel using the metric name
  as the filename (e.g. ``accuracy_test.png``).

All data-loading utilities are identical to the original script.

Typical usage
-------------
::

    python plot_metric_bci_competition_III_header_chips.py \\
        --roots \\
            .../no_align/ard_False/kernel_rbf/spatialFilter_trainable \\
            .../riemann_align/ard_False/kernel_rbf/spatialFilter_trainable \\
        --nfs 1 2 4 8 \\
        --save-dir ./plots/comparison \\
        --no-show
"""

from __future__ import annotations

import os
import json
import argparse
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------------
# Global aesthetic configuration
# ---------------------------------------------------------------------------

# Curated 10-colour palette: colourblind-safe, high contrast on white.
# Ten entries cover the expected crowded-comparison case without cycling.
_PALETTE: List[str] = [
    "#4477AA",   # steel blue
    "#EE6677",   # coral red
    "#228833",   # forest green
    "#CCBB44",   # warm gold
    "#66CCEE",   # sky blue
    "#AA3377",   # plum
    "#BBBBBB",   # neutral grey
    "#000000",   # black
    "#EE7733",   # orange
    "#009988",   # teal
]

_MARKERS: List[str] = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">"]
_LINESTYLES: List[str] = ["-"]

# Metric display metadata ─────────────────────────────────────────────────────
_METRIC_KEYS: List[str] = [
    "nlml",
    "nlpd_train", "acc_train",  "brier_train",  "aucroc_train",
    "nlpd_val",   "acc_val",    "brier_val",    "aucroc_val",
    "nlpd_test",  "acc_test",   "brier_test",   "aucroc_test",
]

_BOUNDED_ROOTS = {"acc", "brier", "aucroc"}

# Panel catalogue: (row, col) → (display title, metric_key)
# Columns: 0 = train | 1 = val | 2 = test
_ALL_PANELS: Dict[Tuple[int, int], Tuple[str, str]] = {
    (0, 0): ("Training NLML",              "nlml"),
    (0, 1): ("Validation NLPD",            "nlpd_val"),
    (0, 2): ("Test NLPD",                  "nlpd_test"),
    (1, 0): ("Accuracy  —  train",         "acc_train"),
    (1, 1): ("Accuracy  —  val",           "acc_val"),
    (1, 2): ("Accuracy  —  test",          "acc_test"),
    (2, 0): ("Brier score  —  train",      "brier_train"),
    (2, 1): ("Brier score  —  val",        "brier_val"),
    (2, 2): ("Brier score  —  test",       "brier_test"),
    (3, 0): ("AUC-ROC  —  train",          "aucroc_train"),
    (3, 1): ("AUC-ROC  —  val",            "aucroc_val"),
    (3, 2): ("AUC-ROC  —  test",           "aucroc_test"),
}

# For bounded metrics: display a "chance" reference line at this y value.
# None → no reference line.
_CHANCE_LEVEL: Dict[str, Optional[float]] = {
    "acc":    0.5,
    "aucroc": 0.5,
    "brier":  None,
}

# y-axis label text (shared across all splits of the same root metric)
_YLABELS: Dict[str, str] = {
    "nlml":   "Negative log marginal likelihood",
    "nlpd":   "Negative log predictive density",
    "acc":    "Accuracy",
    "brier":  "Brier score",
    "aucroc": "AUC-ROC",
}

# Human-readable translations for path segment tokens
_SEGMENT_LABELS: Dict[str, str] = {
    "no_align":                "No alignment",
    "euclidean_align":         "Euclidean alignment",
    "riemann_align":           "Riemannian alignment",
    "ard_True":                "ARD",
    "ard_False":               "No ARD",
    "kernel_rbf":              "RBF kernel",
    "kernel_linear":           "Linear kernel",
    "spatialFilter_trainable": "W trainable",
    "spatialFilter_fixed":     "W fixed (CSP)",
    "shuffle_True":            "Shuffled",
    "shuffle_False":           "",
}

# Split badge styling — coloured pill rendered next to the metric title
_SPLIT_STYLE: Dict[str, Dict[str, str]] = {
    "train": {"bg": "#DDE8F8", "fg": "#1B488C", "icon": "◆"},
    "val":   {"bg": "#FFF0C4", "fg": "#7A5200", "icon": "◇"},
    "test":  {"bg": "#D6F0D8", "fg": "#175E1C", "icon": "★"},
}


# ---------------------------------------------------------------------------
# Data loading utilities  (identical to the original script)
# ---------------------------------------------------------------------------

def _resolve_best_log(data: dict) -> dict:
    """Return the IterLog entry at best_iter, falling back to the last entry."""
    best_iter = data["meta"].get("best_iter", None)
    if best_iter is None:
        return data["logs"][-1]
    for entry in data["logs"]:
        if entry["step"] == best_iter:
            return entry
    return data["logs"][-1]


def get_result_dict(root: str) -> Dict[int, Dict[str, list]]:
    """
    Walk *root* for ``nf_*/fold_*/[run_*/]run_log.json`` and return::

        {nf_int: {metric_key: [fold_val, ...]}}

    Handles both the flat layout (``run_log.json`` directly in ``fold_*/``)
    and the timestamped sub-folder layout (``fold_*/run_<ts>/run_log.json``).
    When multiple timestamped sub-folders exist the most recent one is used.
    Works for any number of folds ≥ 1.
    """
    result: Dict[int, Dict[str, list]] = {}

    for nf_name in sorted(os.listdir(root)):
        nf_path = os.path.join(root, nf_name)
        if not os.path.isdir(nf_path):
            continue
        parts = nf_name.split("_")
        try:
            nf_int = int(parts[1])
        except (IndexError, ValueError):
            continue

        print(f"  {nf_name}")
        buckets: Dict[str, list] = {k: [] for k in _METRIC_KEYS}

        for fold_name in sorted(os.listdir(nf_path)):
            fold_path = os.path.join(nf_path, fold_name)
            if not os.path.isdir(fold_path):
                continue
            print(f"     {fold_name}")

            metrics_path = os.path.join(fold_path, "run_log.json")
            if not os.path.isfile(metrics_path):
                run_subdirs = sorted([
                    d for d in os.listdir(fold_path)
                    if os.path.isdir(os.path.join(fold_path, d))
                ])
                for sub in reversed(run_subdirs):
                    candidate = os.path.join(fold_path, sub, "run_log.json")
                    if os.path.isfile(candidate):
                        metrics_path = candidate
                        print(f"       → {sub}/run_log.json")
                        break
                else:
                    continue

            with open(metrics_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            entry = _resolve_best_log(data)
            for key in _METRIC_KEYS:
                val = entry.get(key, None)
                if val is not None:
                    buckets[key].append(float(val))

        result[nf_int] = buckets

    return result


# ---------------------------------------------------------------------------
# Path-label utilities  (identical to the original script)
# ---------------------------------------------------------------------------

def _path_segments(p: str) -> List[str]:
    """Return path segments, trimming everything before the first data_* segment."""
    segments = [
        part
        for part in Path(os.path.normpath(p)).parts
        if part not in ("", "/", "\\")
    ]

    for idx, segment in enumerate(segments):
        if segment.startswith("data_"):
            return segments[idx:]

    return segments


def _segments_to_label(segments: List[str]) -> str:
    parts: List[str] = []
    for seg in segments:
        human = _SEGMENT_LABELS.get(seg, seg)
        if human:
            parts.append(human)
    return " | ".join(parts)


def _smart_labels(paths: List[str]) -> Tuple[str, List[str]]:
    """
    Derive a shared title string and per-series legend labels from paths.

    Segments present in **all** paths become the common title; only the
    differing segments appear in the per-series legend labels.
    """
    seg_lists = [_path_segments(p) for p in paths]

    if len(paths) == 1:
        lbl = _segments_to_label(seg_lists[0])
        return "", [lbl or Path(paths[0]).name]

    common: List[str] = [
        seg for seg in seg_lists[0]
        if all(seg in sl for sl in seg_lists[1:])
    ]

    diff_lists: List[List[str]] = [
        [s for s in sl if s not in common]
        for sl in seg_lists
    ]

    common_title = _segments_to_label(common)

    labels: List[str] = []
    for i, diff in enumerate(diff_lists):
        lbl = _segments_to_label(diff)
        if not lbl:
            lbl = seg_lists[i][-1] if seg_lists[i] else Path(paths[i]).name
        labels.append(lbl)

    return common_title, labels


def _infer_data_sizes(paths: List[str]) -> Tuple[Optional[int], Optional[int]]:
    """Read n_train / n_test from the first run_log.json found under any path."""
    for root in paths:
        for nf_name in sorted(os.listdir(root)):
            nf_path = os.path.join(root, nf_name)
            if not os.path.isdir(nf_path):
                continue
            for fold_name in sorted(os.listdir(nf_path)):
                fold_path = os.path.join(nf_path, fold_name)
                if not os.path.isdir(fold_path):
                    continue
                candidate = os.path.join(fold_path, "run_log.json")
                if not os.path.isfile(candidate):
                    for sub in sorted(os.listdir(fold_path), reverse=True):
                        c2 = os.path.join(fold_path, sub, "run_log.json")
                        if os.path.isfile(c2):
                            candidate = c2
                            break
                if os.path.isfile(candidate):
                    try:
                        with open(candidate) as fh:
                            meta = json.load(fh).get("meta", {})
                        n_train = meta.get("n_train", meta.get("N_train", None))
                        n_test  = meta.get("n_test",  meta.get("N_test",  None))
                        return (
                            int(n_train) if n_train is not None else None,
                            int(n_test)  if n_test  is not None else None,
                        )
                    except Exception:
                        pass
    return None, None


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _metric_root(metric: str) -> str:
    """Strip the split suffix to get the base metric name, e.g. 'acc_train' → 'acc'."""
    return metric.replace("_train", "").replace("_val", "").replace("_test", "")


def _has_data(
    all_buckets: Dict[str, Dict[int, Dict[str, list]]],
    metric: str,
) -> bool:
    """True if at least one series has ≥ 1 finite value for *metric*."""
    for bdict in all_buckets.values():
        for nf_data in bdict.values():
            if any(np.isfinite(v) for v in nf_data.get(metric, [])):
                return True
    return False


def _compute_stats(
    all_buckets: Dict[str, Dict[int, Dict[str, list]]],
    series_names: List[str],
    nf_labels: List[int],
    metric: str,
    err: str,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[List[Optional[np.ndarray]]]]:
    """
    Compute per-series means, error widths, and raw fold arrays.

    Returns
    -------
    means_list  : list[ndarray(len(nf_labels))]  — NaN where no data
    errs_list   : list[ndarray(len(nf_labels))]  — NaN where n ≤ 1
    folds_list  : list[list[array|None]]          — raw fold values per nf
    """
    means_list, errs_list, folds_list = [], [], []
    for name in series_names:
        bdict = all_buckets[name]
        m_arr, e_arr, f_arr = [], [], []
        for nf in nf_labels:
            v = bdict.get(nf, {}).get(metric, None)
            if not v:
                m_arr.append(np.nan)
                e_arr.append(np.nan)
                f_arr.append(None)
                continue
            arr    = np.asarray(v, dtype=float)
            finite = arr[np.isfinite(arr)]
            n      = len(finite)
            m_arr.append(float(np.nanmean(arr)))
            f_arr.append(finite)
            if n <= 1:
                e_arr.append(np.nan)
            elif err == "sem":
                e_arr.append(float(np.std(finite, ddof=1) / np.sqrt(n)))
            else:
                e_arr.append(float(np.std(finite, ddof=1)))
        means_list.append(np.array(m_arr))
        errs_list.append(np.array(e_arr))
        folds_list.append(f_arr)
    return means_list, errs_list, folds_list


def _apply_pretty_style(ax: plt.Axes) -> None:
    """
    Apply a refined, publication-quality style to a single Axes object.

    Design choices
    --------------
    - Very light warm-grey plot background for depth without distraction.
    - Horizontal grid only, hairline weight, slightly warm tone.
    - Left and bottom spines only; both thinned and softened.
    - Tick marks replaced by subtle inward stubs so labels float cleanly.
    """
    BG         = "#F7F7F8"   # near-white with warmth
    SPINE_COL  = "#C0C0C8"   # muted blue-grey
    GRID_COL   = "#E4E4EA"   # very subtle
    TICK_COL   = "#B0B0BA"

    ax.set_facecolor(BG)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COL)
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_color(SPINE_COL)
    ax.spines["bottom"].set_linewidth(0.85)

    ax.yaxis.grid(True, color=GRID_COL, linewidth=0.75, linestyle="-", zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    ax.tick_params(
        axis="both", which="major",
        labelsize=11, length=4, width=0.85,
        color=TICK_COL, pad=5, direction="out",
    )
    ax.tick_params(axis="both", which="minor", length=0)


def _wrap_label(label: str, width: int = 28) -> str:
    """Wrap long legend labels so crowded comparisons stay readable."""
    return "\n".join(textwrap.wrap(label, width=width)) or label


def _panel_title_and_split(panel_title: str) -> Tuple[str, Optional[str]]:
    """Split titles like 'Accuracy — test' into metric and split strings."""
    if "—" not in panel_title:
        return panel_title.strip(), None
    metric_name, split_name = [part.strip() for part in panel_title.split("—", 1)]
    return metric_name, split_name.strip().lower()


def _draw_chance_line(ax: plt.Axes, metric: str) -> Optional[mlines.Line2D]:
    """
    Draw a horizontal dashed reference line at the chance level for
    metrics where a meaningful baseline exists (accuracy → 0.5,
    AUC-ROC → 0.5).  Returns the Line2D for legend inclusion, or None.
    """
    root = _metric_root(metric)
    level = _CHANCE_LEVEL.get(root, None)
    if level is None:
        return None
    line = ax.axhline(
        level,
        linestyle=(0, (5, 4)),   # custom dash — slightly shorter gaps
        linewidth=1.1,
        color="#B0B0BA",
        zorder=1,
        label=f"Chance ({level:.1f})",
    )
    return line


def _make_split_badge(split: str) -> str:
    """Return a plain split label without decorative icons."""
    return split.capitalize()


def _draw_info_card(
    fig: plt.Figure,
    info_x: float,
    y_top: float,
    card_w: float,
    accent_color: str,
    sections: List[Tuple[str, List[str]]],
) -> None:
    """
    Draw a minimal info block: a thin coloured rule, then plain text.

    No boxes, no strips — just clean typography in figure-fraction space
    so every font-size is a real point size.
    """
    LINE_H  = 0.038   # content line  (9 pt + comfortable leading)
    LBL_H   = 0.030   # section label (7 pt + leading)
    GAP_H   = 0.022   # vertical gap between sections
    PAD_T   = 0.016   # gap between rule and first label

    LBL_COL = "#9A9AB0"   # muted grey-blue for section labels
    TXT_COL = "#1C1C30"   # near-black for content

    tr = fig.transFigure

    # ── thin coloured rule at the top ────────────────────────────────────────
    fig.add_artist(mpatches.Rectangle(
        (info_x, y_top - 0.003), card_w, 0.003,
        facecolor=accent_color, edgecolor="none",
        transform=tr, clip_on=False, zorder=10,
    ))

    cursor = y_top - 0.003 - PAD_T   # walk downward from below the rule

    for s_idx, (label, lines) in enumerate(sections):
        if s_idx > 0:
            cursor -= GAP_H

        # section label — small, muted
        cursor -= LBL_H
        fig.text(
            info_x, cursor + LBL_H * 0.5,
            label,
            fontsize=7.5, color=LBL_COL,
            ha="left", va="center",
            transform=tr, zorder=11,
        )

        # content lines — larger, dark
        for line in lines:
            cursor -= LINE_H
            fig.text(
                info_x + 0.004, cursor + LINE_H * 0.5,
                line,
                fontsize=9.5, color=TXT_COL,
                ha="left", va="center",
                transform=tr, zorder=11,
            )


def _draw_header_chips(
    fig: plt.Figure,
    left_x: float,
    right_x: float,
    metric_title: str,
    split_key: Optional[str],
    common_title: Optional[str],
    size_subtitle: Optional[str],
    err_label: str,
) -> None:
    """
    Draw a clean title block with compact metadata chips.

    The previous design placed experiment/configuration details in a small
    bottom footnote.  This helper moves that information into the header so
    every saved figure reads naturally from top to bottom and avoids footer
    clutter.
    """
    title_color = "#17172A"
    muted_color = "#74748A"
    chip_bg = "#F0F0F4"
    chip_fg = "#343445"

    fig.text(
        left_x,
        0.965,
        metric_title,
        fontsize=19,
        fontweight="bold",
        color=title_color,
        ha="left",
        va="top",
        transform=fig.transFigure,
    )

    chips: List[Tuple[str, str, str]] = []

    if split_key:
        style = _SPLIT_STYLE.get(
            split_key.lower(),
            {"bg": "#EEEEEE", "fg": "#333333"},
        )
        chips.append((
            _make_split_badge(split_key),
            style["bg"],
            style["fg"],
        ))

    if size_subtitle:
        for item in size_subtitle.split("  |  "):
            clean_item = item.strip()
            if clean_item:
                chips.append((clean_item, chip_bg, chip_fg))

    chips.append((f"Error bars: {err_label}", chip_bg, chip_fg))

    cursor_x = left_x
    cursor_y = 0.890
    row_gap = 0.052
    chip_gap = 0.010

    for chip_text, bg, fg in chips:
        chip = fig.text(
            cursor_x,
            cursor_y,
            f"  {chip_text}  ",
            fontsize=9.2,
            fontweight="semibold",
            color=fg,
            ha="left",
            va="top",
            transform=fig.transFigure,
            bbox=dict(
                boxstyle="round,pad=0.28",
                facecolor=bg,
                edgecolor="none",
                alpha=0.97,
            ),
        )

        # Render once so Matplotlib can report the chip's true width.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = chip.get_window_extent(renderer=renderer)
        fig_w_px = fig.get_size_inches()[0] * fig.dpi
        chip_w = bbox.width / fig_w_px

        if cursor_x + chip_w > right_x and cursor_x > left_x:
            cursor_x = left_x
            cursor_y -= row_gap
            chip.set_position((cursor_x, cursor_y))
            fig.canvas.draw()
            bbox = chip.get_window_extent(renderer=renderer)
            chip_w = bbox.width / fig_w_px

        cursor_x += chip_w + chip_gap

    if common_title:
        wrapped_context = textwrap.fill(common_title, width=96)
        fig.text(
            left_x,
            0.810,
            wrapped_context,
            fontsize=8.3,
            color=muted_color,
            ha="left",
            va="top",
            linespacing=1.25,
            transform=fig.transFigure,
        )


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def plot_individual_panels(
    all_buckets: Dict[str, Dict[int, Dict[str, list]]],
    common_title: Optional[str] = None,
    size_subtitle: Optional[str] = None,
    err: str = "sem",
    connect_means: bool = True,
    nfs_to_show: Optional[List[int]] = None,
    show_folds: bool = False,
    save_dir: Optional[str] = None,
    no_show: bool = False,
    dpi: int = 150,
) -> None:
    """
    Produce one standalone figure per active metric panel.

    Parameters
    ----------
    all_buckets : dict
        ``{legend_label: {nf_int: {metric_key: [fold_val, ...]}}}``
    common_title : str, optional
        Shared experiment context derived from common path segments.
    size_subtitle : str, optional
        e.g. "4-fold CV  |  N_train=280  |  N_test=93"
    err : "sem" | "std"
        Error whisker type.
    connect_means : bool
        Draw a line through the mean points across nf values.
    nfs_to_show : list of int, optional
        Which nf values to include on the x-axis.
    show_folds : bool
        Overlay individual fold values as a jittered strip plot.
    save_dir : str, optional
        Directory where individual panel images are written.
        One PNG per panel, named by metric key.  Created if absent.
    no_show : bool
        Suppress ``plt.show()`` (for headless / batch use).
    dpi : int
        Resolution for saved images (default 150).
    """
    series_names = list(all_buckets.keys())
    n_s = len(series_names)

    # ── x-axis positions ─────────────────────────────────────────────────────
    if nfs_to_show is None:
        nfs_to_show = sorted(
            set().union(*[set(b.keys()) for b in all_buckets.values()])
        )
    nf_labels = list(nfs_to_show)
    x_base    = np.arange(len(nf_labels), dtype=float)

    # ── which panels have data ───────────────────────────────────────────────
    active_panels = {
        pos: info
        for pos, info in _ALL_PANELS.items()
        if _has_data(all_buckets, info[1])
    }

    if not active_panels:
        print("[WARN] No data found in any bucket — nothing to plot.")
        return

    # ── per-series style assignments ─────────────────────────────────────────
    # Colour and marker are the primary discriminators; all connecting lines stay solid.
    # Small x-offsets prevent whiskers from colliding when many series share
    # the same nf value.
    palette    = (_PALETTE    * (n_s // len(_PALETTE)    + 1))[:n_s]
    markers    = (_MARKERS    * (n_s // len(_MARKERS)    + 1))[:n_s]
    linestyles = (_LINESTYLES * (n_s // len(_LINESTYLES) + 1))[:n_s]
    # Keep all series visually close to their nf tick.
    # At n_s=10 the total spread is 0.20 = 20 % of tick spacing.
    offset_spread = min(0.20, 0.022 * n_s)
    x_offsets = (
        np.linspace(-offset_spread / 2, offset_spread / 2, n_s)
        if n_s > 1 else np.array([0.0])
    )

    # ── save directory ───────────────────────────────────────────────────────
    save_path_obj: Optional[Path] = None
    if save_dir is not None:
        save_path_obj = Path(save_dir).expanduser().resolve()
        save_path_obj.mkdir(parents=True, exist_ok=True)

    # ── err label for axis / title annotation ────────────────────────────────
    err_label = "SEM" if err == "sem" else "SD"

    # ── legend geometry — right-side external box ────────────────────────────
    # The legend lives outside the axes on the right.  The figure width
    # is extended to accommodate it; more series → wider legend column.
    legend_col_width = 2.6        # inches reserved for the single legend column
    legend_ncols     = 1          # keep legend entries stacked in one column
    legend_extra_w   = legend_col_width + 0.4
    data_area_w      = 7.2 if n_s <= 6 else 8.0
    fig_width        = data_area_w + legend_extra_w
    fig_height       = 5.6

    # Fraction of figure width that goes to the data area (for tight_layout)
    data_frac = data_area_w / fig_width

    # ── global rcParams ──────────────────────────────────────────────────────
    with mpl.rc_context({
        "font.family":               "DejaVu Sans",
        "axes.titlesize":            16,
        "axes.labelsize":            12,
        "legend.fontsize":           9.0,
        "legend.title_fontsize":     10.0,
        "legend.frameon":            True,
        "legend.framealpha":         0.97,
        "legend.edgecolor":          "#D4D4DC",
        "figure.facecolor":          "#FFFFFF",
        "savefig.facecolor":         "#FFFFFF",
        "savefig.bbox":              "tight",
        "savefig.pad_inches":        0.20,
    }):

        for (orig_r, orig_c), (panel_title, metric) in active_panels.items():

            means_list, errs_list, folds_list = _compute_stats(
                all_buckets, series_names, nf_labels, metric, err
            )

            # ── figure & axes ────────────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            left_ax  = 0.105
            right_ax = data_frac - 0.02
            fig.subplots_adjust(left=left_ax, right=right_ax, top=0.745, bottom=0.155)
            _apply_pretty_style(ax)

            # ── title block ──────────────────────────────────────────────────
            # Clean header: title + compact metadata chips.  Experiment context
            # now lives in the header rather than as a bottom footnote.
            metric_title, split_key = _panel_title_and_split(panel_title)
            _draw_header_chips(
                fig=fig,
                left_x=left_ax,
                right_x=right_ax,
                metric_title=metric_title,
                split_key=split_key,
                common_title=common_title,
                size_subtitle=size_subtitle,
                err_label=err_label,
            )

            # ── chance / reference line ──────────────────────────────────────
            chance_handle = _draw_chance_line(ax, metric)

            # ── series ──────────────────────────────────────────────────────
            legend_handles: List[mpl.artist.Artist] = []

            # Chance line first in legend
            if chance_handle is not None:
                root_m = _metric_root(metric)
                level  = _CHANCE_LEVEL[root_m]
                legend_handles.append(
                    mlines.Line2D(
                        [], [],
                        linestyle=(0, (5, 4)), linewidth=1.2,
                        color="#B0B0BA",
                        label=f"Chance  ({level:.1f})",
                    )
                )

            for i, name in enumerate(series_names):
                color     = palette[i]
                marker    = markers[i]
                linestyle = "-"
                x         = x_base + x_offsets[i]
                y         = means_list[i]
                ye        = errs_list[i]
                folds     = folds_list[i]

                finite_mask = np.isfinite(y)
                if not np.any(finite_mask):
                    continue

                # ── individual fold jitter (optional, drawn first / lowest z) ─
                if show_folds:
                    rng = np.random.default_rng(seed=i * 31 + 7)
                    for j, fd in enumerate(folds):
                        if fd is None or len(fd) == 0:
                            continue
                        jitter = rng.uniform(-0.045, 0.045, len(fd))
                        ax.scatter(
                            np.full(len(fd), x[j]) + jitter,
                            fd,
                            s=14, color=color, alpha=0.30,
                            zorder=2, edgecolors="none", marker="o",
                        )

                # ── connecting line ──────────────────────────────────────────
                if connect_means and np.sum(finite_mask) > 1:
                    ax.plot(
                        x[finite_mask], y[finite_mask],
                        linewidth=1.8, color=color,
                        linestyle=linestyle,
                        solid_capstyle="round",
                        alpha=0.75,
                        zorder=3,
                    )

                # ── whisker error bars ───────────────────────────────────────
                # Draw only where both mean and error are finite.
                has_err = finite_mask & np.isfinite(ye)
                if np.any(has_err):
                    ax.errorbar(
                        x[has_err], y[has_err],
                        yerr=ye[has_err],
                        fmt="none",
                        ecolor=color,
                        elinewidth=1.6,
                        capsize=5,
                        capthick=1.6,
                        alpha=0.85,
                        zorder=4,
                    )

                # Points without error info — plain markers, no whiskers
                no_err_only = finite_mask & ~np.isfinite(ye)
                if np.any(no_err_only):
                    ax.scatter(
                        x[no_err_only], y[no_err_only],
                        s=52, color=color, marker=marker, zorder=6,
                        edgecolors="white", linewidths=1.2,
                        alpha=0.90,
                    )

                # ── mean markers (on top of whiskers) ────────────────────────
                if np.any(finite_mask):
                    ax.scatter(
                        x[finite_mask], y[finite_mask],
                        s=60, color=color, marker=marker, zorder=7,
                        edgecolors="white", linewidths=1.3,
                    )

                # ── legend proxy ─────────────────────────────────────────────
                legend_handles.append(
                    mlines.Line2D(
                        [], [],
                        color=color,
                        linestyle=linestyle,
                        linewidth=2.0,
                        marker=marker,
                        markersize=7,
                        markerfacecolor=color,
                        markeredgecolor="white",
                        markeredgewidth=1.0,
                        label=_wrap_label(name),
                    )
                )

            # ── axes labels / ticks ──────────────────────────────────────────
            ax.set_xticks(x_base)
            ax.set_xticklabels(
                [str(n) for n in nf_labels],
                fontsize=11,
            )
            ax.set_xlabel(
                "Number of spatial filters  (nf)",
                labelpad=9, fontsize=12, color="#333344",
            )

            root = _metric_root(metric)
            ax.set_ylabel(
                _YLABELS.get(root, root.upper()),
                labelpad=9, fontsize=12, color="#333344",
            )

            if root in {"acc", "aucroc"}:
                # Accuracy and AUC-ROC are usually only informative above chance.
                # Keep the true signed/raw values; this only changes the displayed range.
                ax.set_ylim(0.5, 1.0)
                ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
                ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
            elif root == "brier":
                ax.set_ylim(-0.02, 1.06)
                ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
                ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))

            # Extra horizontal breathing room
            half_span = max(
                abs(x_offsets).max() + 0.22,
                0.38,
            )
            ax.set_xlim(x_base[0] - half_span, x_base[-1] + half_span)

            # ── external right-side legend ────────────────────────────────────
            if legend_handles:
                leg = ax.legend(
                    handles=legend_handles,
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    borderaxespad=0.0,
                    ncol=legend_ncols,
                    framealpha=0.97,
                    edgecolor="#D4D4DC",
                    handlelength=2.4,
                    handletextpad=0.65,
                    columnspacing=1.2,
                    borderpad=0.90,
                    labelspacing=0.65,
                )
                leg.get_frame().set_linewidth(0.8)

            # ── save ─────────────────────────────────────────────────────────
            if save_path_obj is not None:
                fname = f"{metric}.png"
                out   = save_path_obj / fname
                fig.savefig(out, dpi=dpi)
                print(f"  Saved → {out}")

            if not no_show:
                plt.show()
            else:
                plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_int_list(values: List[str]) -> List[int]:
    return [int(v) for v in values]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot cross-validation metrics from one or more GP spatial-filter "
            "experiment directories — individual-canvas edition.  "
            "Each metric panel is rendered as its own figure.  "
            "Each path must contain nf_*/fold_*/ sub-trees."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--roots",
        nargs="+",
        required=True,
        metavar="PATH",
        help=(
            "One or more result directories, each containing nf_<k>/fold_<i>/ "
            "sub-trees (with an optional run_<timestamp>/ level inside each fold). "
            "Common path segments become the figure subtitle; differing segments "
            "become the legend labels."
        ),
    )
    parser.add_argument(
        "--nfs",
        nargs="+",
        default=None,
        metavar="N",
        help=(
            "Spatial-filter counts to show on the x-axis (space-separated integers). "
            "Defaults to the union of all nf values found across the supplied paths."
        ),
    )
    parser.add_argument(
        "--err",
        default="sem",
        choices=["sem", "std"],
        help="Whisker type: standard error of the mean (sem) or std deviation (std).",
    )
    parser.add_argument(
        "--no-connect",
        action="store_true",
        help="Do not draw lines connecting mean points across nf values.",
    )
    parser.add_argument(
        "--show-folds",
        action="store_true",
        help=(
            "Overlay translucent dots for every individual fold value as a "
            "jittered strip plot.  Useful for diagnosing fold-to-fold variance."
        ),
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory where individual panel images are written. "
            "One PNG per panel, named by metric key (e.g. acc_test.png). "
            "Created automatically if it does not exist."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution (dots per inch) for saved images.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Suppress interactive windows (for headless / batch runs).",
    )

    return parser


def main() -> None:
    args  = build_argparser().parse_args()
    roots = [str(Path(p).expanduser().resolve()) for p in args.roots]

    for r in roots:
        if not os.path.isdir(r):
            raise FileNotFoundError(f"Path not found: {r}")

    # ── labels ──────────────────────────────────────────────────────────────
    common_title, labels = _smart_labels(roots)

    # ── load ─────────────────────────────────────────────────────────────────
    all_buckets: Dict[str, Dict[int, Dict[str, list]]] = {}
    for path, label in zip(roots, labels):
        print(f"\nLoading '{label}'  ({path})")
        all_buckets[label] = get_result_dict(path)

    # ── infer fold count and dataset sizes ───────────────────────────────────
    k_fold: Union[int, str] = "?"
    try:
        first_label = next(iter(all_buckets))
        first_nf    = next(iter(all_buckets[first_label]))
        lens = [
            len(v)
            for v in all_buckets[first_label][first_nf].values()
            if v
        ]
        if lens:
            k_fold = max(lens)
    except StopIteration:
        pass

    n_train, n_test = _infer_data_sizes(roots)

    size_parts: List[str] = []
    if k_fold != "?":
        size_parts.append(f"{k_fold}-fold CV")
    if n_train is not None:
        size_parts.append(f"N_train = {n_train}")
    if n_test is not None:
        size_parts.append(f"N_test = {n_test}")
    size_subtitle = "  |  ".join(size_parts) if size_parts else None

    # ── plot ─────────────────────────────────────────────────────────────────
    nfs = _parse_int_list(args.nfs) if args.nfs else None

    plot_individual_panels(
        all_buckets,
        common_title  = common_title or None,
        size_subtitle = size_subtitle,
        err           = args.err,
        connect_means = not args.no_connect,
        nfs_to_show   = nfs,
        show_folds    = args.show_folds,
        save_dir      = args.save_dir,
        no_show       = args.no_show,
        dpi           = args.dpi,
    )


if __name__ == "__main__":
    main()
