"""
Plot cross-validation metrics from one or more GP spatial-filter experiment runs.

Each path supplied via ``--roots`` must point to a directory that contains
``nf_<k>/fold_<i>/[run_<timestamp>/]run_log.json`` sub-trees.

Key features
------------
- **Multiple paths** → one coloured series per path in every panel.
- **Smart legend / title**: path segments that are identical across all
  supplied paths are factored out into the plot title; only the *differing*
  segments appear in the per-series legend labels.
- **Dynamic panels**: columns (train / val / test) and metric rows are shown
  only when at least one series has data for them — no empty panels.
- **Single-fold support**: error bars are omitted when only one fold is
  available; mean points are still plotted.
- **Save**: ``--save-path`` writes the figure; ``--no-show`` suppresses the
  interactive window (useful on headless servers).

Typical usage
-------------
Compare alignment strategies (all other settings identical)::

    python plot_metric_bci_competition_III_merged_spatial_filters.py \\
        --roots \\
            .../no_align/ard_False/kernel_rbf/spatialFilter_trainable \\
            .../euclidean_align/ard_False/kernel_rbf/spatialFilter_trainable \\
            .../riemann_align/ard_False/kernel_rbf/spatialFilter_trainable \\
        --nfs 1 2 4 8 16 20

Compare kernel types::

    python plot_metric_bci_competition_III_merged_spatial_filters.py \\
        --roots \\
            .../no_align/ard_False/kernel_linear/spatialFilter_trainable \\
            .../no_align/ard_False/kernel_rbf/spatialFilter_trainable
"""

from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


# ---------------------------------------------------------------------------
# Metric catalogue
# ---------------------------------------------------------------------------

_METRIC_KEYS: List[str] = [
    "nlml",
    "nlpd_train", "acc_train",  "brier_train",  "aucroc_train", "aucpr_train",
    "nlpd_val",   "acc_val",    "brier_val",    "aucroc_val",   "aucpr_val",
    "nlpd_test",  "acc_test",   "brier_test",   "aucroc_test",  "aucpr_test",
]

# Metrics whose natural range is [0, 1] — y-axis is pinned accordingly.
_BOUNDED_ROOTS = {"acc", "brier", "aucroc", "aucpr"}

# Full panel catalogue: (row, col) → (y-axis label, metric_key)
# Columns: 0 = train | 1 = val | 2 = test
# Rows:    0 = NLML/NLPD | 1 = Accuracy | 2 = Brier | 3 = AUCROC | 4 = AUCPR
_ALL_PANELS: Dict[Tuple[int, int], Tuple[str, str]] = {
    (0, 0): ("NLML",             "nlml"),
    (0, 1): ("NLPD (val)",       "nlpd_val"),
    (0, 2): ("NLPD (test)",      "nlpd_test"),
    (1, 0): ("Accuracy (train)", "acc_train"),
    (1, 1): ("Accuracy (val)",   "acc_val"),
    (1, 2): ("Accuracy (test)",  "acc_test"),
    (2, 0): ("Brier (train)",    "brier_train"),
    (2, 1): ("Brier (val)",      "brier_val"),
    (2, 2): ("Brier (test)",     "brier_test"),
    (3, 0): ("AUCROC (train)",   "aucroc_train"),
    (3, 1): ("AUCROC (val)",     "aucroc_val"),
    (3, 2): ("AUCROC (test)",    "aucroc_test"),
    (4, 0): ("AUCPR (train)",    "aucpr_train"),
    (4, 1): ("AUCPR (val)",      "aucpr_val"),
    (4, 2): ("AUCPR (test)",     "aucpr_test"),
}

# Human-readable translations for common path segment tokens.
# Empty string → token is silently dropped from labels (uninteresting default).
_SEGMENT_LABELS: Dict[str, str] = {
    "no_align":                "no align",
    "euclidean_align":         "Euclidean align",
    "riemann_align":           "Riemann align",
    "ard_True":                "ARD",
    "ard_False":               "no ARD",
    "kernel_rbf":              "RBF kernel",
    "kernel_linear":           "Linear kernel",
    "spatialFilter_trainable": "W trainable",
    "spatialFilter_fixed":     "W fixed (CSP)",
    "shuffle_True":            "shuffled",
    "shuffle_False":           "",   # silent — unshuffled is the expected default
}


# ---------------------------------------------------------------------------
# Result loading
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

            # Try flat layout first, then timestamped sub-folder.
            metrics_path = os.path.join(fold_path, "run_log.json")
            if not os.path.isfile(metrics_path):
                run_subdirs = sorted([
                    d for d in os.listdir(fold_path)
                    if os.path.isdir(os.path.join(fold_path, d))
                ])
                for sub in reversed(run_subdirs):   # most recent last alphabetically
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
# Path label / title utilities
# ---------------------------------------------------------------------------

def _path_segments(p: str) -> List[str]:
    """Split a normalised path into its folder-name tokens."""
    return [part for part in Path(os.path.normpath(p)).parts if part not in ("", "/", "\\")]


def _segments_to_label(segments: List[str]) -> str:
    """
    Translate a list of path-segment tokens into a human-readable string.

    Tokens present in ``_SEGMENT_LABELS`` are replaced; empty translations
    (silenced tokens) are dropped.  Remaining pieces are joined with ' | '.
    """
    parts: List[str] = []
    for seg in segments:
        human = _SEGMENT_LABELS.get(seg, seg)
        if human:
            parts.append(human)
    return " | ".join(parts)


def _smart_labels(paths: List[str]) -> Tuple[str, List[str]]:
    """
    Derive a shared title string and per-series legend labels from paths.

    Algorithm
    ---------
    1. Split every path into segment tokens.
    2. Segments present in **all** paths → shared title (common context).
    3. Segments that differ across paths → per-series legend label.
    4. Translate tokens through ``_SEGMENT_LABELS`` for readability.
    5. If a label would be empty after translation, fall back to the last
       differing segment or the directory basename.

    Returns
    -------
    common_title : str
    labels : list[str]  — one per input path, in the same order.
    """
    seg_lists = [_path_segments(p) for p in paths]

    if len(paths) == 1:
        lbl = _segments_to_label(seg_lists[0])
        return "", [lbl or Path(paths[0]).name]

    # Segments common to every path (preserving order of first path)
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
            # Fallback: last token of the original path
            lbl = seg_lists[i][-1] if seg_lists[i] else Path(paths[i]).name
        labels.append(lbl)

    return common_title, labels


# ---------------------------------------------------------------------------
# Data-size inference from run_log meta
# ---------------------------------------------------------------------------

def _infer_data_sizes(paths: List[str]) -> Tuple[Optional[int], Optional[int]]:
    """
    Read n_train / n_test from the first ``run_log.json`` found under any path.
    Returns ``(n_train, n_test)``; either may be ``None`` if not recorded.
    """
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
# Plotting
# ---------------------------------------------------------------------------

def _metric_root(metric: str) -> str:
    """Strip the split suffix to get the base metric name."""
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


def plot_nfs_runs_new(
    all_buckets: Dict[str, Dict[int, Dict[str, list]]],
    title: Optional[str] = None,
    err: str = "sem",
    offset_step: float = 0.08,
    connect_means: bool = True,
    nfs_to_show: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    no_show: bool = False,
) -> None:
    """
    Plot cross-validation metrics for one or more experimental conditions.

    Parameters
    ----------
    all_buckets : dict
        ``{legend_label: {nf_int: {metric_key: [fold_val, ...]}}}``
    title : str, optional
        Figure suptitle — typically the common path prefix.
    err : "sem" | "std"
        Error bar style.  Single-fold runs show points only (no error bar).
    offset_step : float
        Horizontal nudge between series so markers do not overlap.
    connect_means : bool
        Draw a line through mean points of each series.
    nfs_to_show : list of int, optional
        Which nf values to display on the x-axis.  Defaults to the union
        across all buckets.
    save_path : str, optional
        If given, the figure is saved here (format inferred from extension).
    no_show : bool
        If True, suppress ``plt.show()``.
    """
    series_names = list(all_buckets.keys())
    n_s = len(series_names)

    # x-axis labels
    if nfs_to_show is None:
        nfs_to_show = sorted(
            set().union(*[set(b.keys()) for b in all_buckets.values()])
        )
    nf_labels = list(nfs_to_show)
    x_base    = np.arange(len(nf_labels), dtype=float)

    # ------------------------------------------------------------------
    # Determine which panels actually have data → compact grid
    # ------------------------------------------------------------------
    active_panels = {
        pos: info
        for pos, info in _ALL_PANELS.items()
        if _has_data(all_buckets, info[1])
    }

    if not active_panels:
        print("[WARN] No data found in any bucket — nothing to plot.")
        return

    active_rows = sorted({r for r, _ in active_panels})
    active_cols = sorted({c for _, c in active_panels})
    row_map = {r: i for i, r in enumerate(active_rows)}
    col_map = {c: i for i, c in enumerate(active_cols)}
    n_rows  = len(active_rows)
    n_cols  = len(active_cols)

    # Per-panel size in inches — keeps panels readable regardless of grid size
    panel_w = 4.5
    panel_h = 3.2
    fig_w   = max(6, panel_w * n_cols)
    fig_h   = max(4, panel_h * n_rows)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )

    # Hide all axes; reveal only those with data
    for r in range(n_rows):
        for c in range(n_cols):
            axes[r, c].set_visible(False)

    # Colour cycle — one distinct colour per series
    color_cycle = (
        plt.rcParams["axes.prop_cycle"].by_key().get("color")
        or [f"C{i}" for i in range(10)]
    )
    x_offsets = [offset_step * (i - (n_s - 1) / 2.0) for i in range(n_s)]

    # ------------------------------------------------------------------
    # Statistics helpers
    # ------------------------------------------------------------------
    def _collect(bdict: Dict[int, Dict[str, list]], metric: str) -> List[Optional[list]]:
        return [bdict.get(nf, {}).get(metric, None) for nf in nf_labels]

    def _stats(metric: str):
        means_out, errs_out = [], []
        for name in series_names:
            series = _collect(all_buckets[name], metric)
            m_arr, e_arr = [], []
            for v in series:
                if not v:
                    m_arr.append(np.nan)
                    e_arr.append(np.nan)
                    continue
                arr    = np.asarray(v, dtype=float)
                finite = arr[np.isfinite(arr)]
                n      = len(finite)
                m_arr.append(float(np.nanmean(arr)))
                if n <= 1:
                    e_arr.append(np.nan)   # single fold — no error bar
                elif err == "sem":
                    e_arr.append(float(np.std(finite, ddof=1) / np.sqrt(n)))
                else:
                    e_arr.append(float(np.std(finite, ddof=1)))
            means_out.append(np.array(m_arr))
            errs_out.append(np.array(e_arr))
        return means_out, errs_out

    # ------------------------------------------------------------------
    # Draw each active panel
    # ------------------------------------------------------------------
    for (orig_r, orig_c), (ylabel, metric) in active_panels.items():
        ax = axes[row_map[orig_r], col_map[orig_c]]
        ax.set_visible(True)

        means_list, errs_list = _stats(metric)

        for i, name in enumerate(series_names):
            color = color_cycle[i % len(color_cycle)]
            x  = x_base + x_offsets[i]
            y  = means_list[i]
            ye = errs_list[i]

            finite_mean = np.isfinite(y)
            if not np.any(finite_mean):
                continue

            has_err = finite_mean & np.isfinite(ye)
            no_err  = finite_mean & ~np.isfinite(ye)

            first_plotted = False
            if np.any(has_err):
                ax.errorbar(
                    x[has_err], y[has_err], yerr=ye[has_err],
                    fmt="o", capsize=3, elinewidth=1.2,
                    markersize=5.5, color=color, linewidth=0,
                    label=name,
                )
                first_plotted = True
            if np.any(no_err):
                ax.plot(
                    x[no_err], y[no_err], "o",
                    markersize=5.5, color=color,
                    label=name if not first_plotted else "_nolegend_",
                )
            if connect_means and np.any(finite_mean):
                ax.plot(x[finite_mean], y[finite_mean],
                        linewidth=1.2, color=color)

        ax.set_xticks(x_base)
        ax.set_xticklabels([str(l) for l in nf_labels], fontsize=8)
        ax.set_xlabel("Spatial filters (nf)", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)

        if _metric_root(metric) in _BOUNDED_ROOTS:
            ax.set_ylim(0, 1)

        ax.grid(True, alpha=0.25)


    # ------------------------------------------------------------------
    # Layout: tight_layout first for panel spacing, then explicit margins
    # to guarantee visible room for the suptitle (top) and legend (bottom).
    # ------------------------------------------------------------------
    title_nlines = len(title.split("\n")) if title else 0
    top_margin   = 0.04 * title_nlines + 0.03   # ~0.04 per title line + padding
    leg_rows     = int(np.ceil(n_s / 4))         # legend wraps at 4 columns
    bot_margin   = 0.06 * leg_rows + 0.03        # ~0.06 per legend row + padding

    fig.tight_layout(rect=[0.0, bot_margin, 1.0, 1.0 - top_margin])

    # ------------------------------------------------------------------
    # Suptitle — placed inside the reserved top margin
    # ------------------------------------------------------------------
    if title:
        fig.suptitle(
            title,
            fontsize=11,
            y=1.0 - top_margin * 0.4,
            va="top",
        )

    # ------------------------------------------------------------------
    # Single shared legend anchored in the reserved bottom margin
    # ------------------------------------------------------------------
    legend_handles = [
        mlines.Line2D(
            [], [],
            color=color_cycle[i % len(color_cycle)],
            marker="o", linewidth=1.2, markersize=6,
            label=name,
        )
        for i, name in enumerate(series_names)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(n_s, 4),
        frameon=True,
        fontsize=9,
        bbox_to_anchor=(0.5, bot_margin * 0.35),
    )

    if save_path is not None:
        sp = Path(save_path).expanduser().resolve()
        sp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(sp, dpi=150)
        print(f"Figure saved -> {sp}")

    if not no_show:
        plt.show()



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_int_list(values: List[str]) -> List[int]:
    return [int(v) for v in values]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot cross-validation metrics from one or more GP spatial-filter "
            "experiment directories.  Each path must contain nf_*/fold_*/ sub-trees."
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
            "Common path segments become the plot title; differing segments become "
            "the legend labels."
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
        help="Error bar type: standard error of the mean (sem) or std deviation (std).",
    )
    parser.add_argument(
        "--no-connect",
        action="store_true",
        help="Do not draw lines connecting mean points across nf values.",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        metavar="FILE",
        help=(
            "Save the figure to this path (e.g. ./out/plot.png). "
            "Format is inferred from the extension (png, pdf, svg, …). "
            "Parent directories are created automatically."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Suppress the interactive window (useful for headless / batch runs).",
    )
    return parser


def main() -> None:
    args  = build_argparser().parse_args()
    roots = [str(Path(p).expanduser().resolve()) for p in args.roots]

    for r in roots:
        if not os.path.isdir(r):
            raise FileNotFoundError(f"Path not found: {r}")

    # ------------------------------------------------------------------
    # Derive title and legend labels from path structure
    # ------------------------------------------------------------------
    common_title, labels = _smart_labels(roots)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    all_buckets: Dict[str, Dict[int, Dict[str, list]]] = {}
    for path, label in zip(roots, labels):
        print(f"\nLoading '{label}'  ({path})")
        all_buckets[label] = get_result_dict(path)

    # ------------------------------------------------------------------
    # Infer fold count and data sizes for the title
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Compose suptitle
    # ------------------------------------------------------------------
    size_parts: List[str] = []
    if k_fold != "?":
        size_parts.append(f"{k_fold}-fold CV")
    if n_train is not None:
        size_parts.append(f"N_train={n_train}")
    if n_test is not None:
        size_parts.append(f"N_test={n_test}")

    title_lines: List[str] = []
    if common_title:
        title_lines.append(common_title)
    if size_parts:
        title_lines.append("  |  ".join(size_parts))
    title = "\n".join(title_lines) if title_lines else None

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    nfs = _parse_int_list(args.nfs) if args.nfs else None

    plot_nfs_runs_new(
        all_buckets,
        title         = title,
        err           = args.err,
        connect_means = not args.no_connect,
        nfs_to_show   = nfs,
        save_path     = args.save_path,
        no_show       = args.no_show,
    )


if __name__ == "__main__":
    main()
