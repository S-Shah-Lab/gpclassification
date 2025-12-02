"""
Plot the results coming from `run_class_bci_competition_III_merged_spatial_filters.py`
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from typing import Dict, List, Union, Optional, Hashable


def get_result_dict(root):
    
    result_dict = {}

    for nf_name in sorted(os.listdir(root)):
        nf_path = os.path.join(root, nf_name) 
        if not os.path.isdir(nf_path): continue
        print(f"  {nf_name}")

        nf_dummy = int(nf_name.split('_')[1])
        result_dict[nf_dummy] = {} # a dict for each num of spatial filters

        # Define buckets to store info
        nlml = [] # float
        acc_train = [] # float
        acc_test = [] # float
        brier_train = [] # float
        brier_test = [] # float

        for run_name in sorted(os.listdir(nf_path)):
            run_path = os.path.join(nf_path, run_name) # create path to here
            if not os.path.isdir(run_path): continue
            print(f"     {run_name}")

            metrics_path = os.path.join(run_path, "run_log.json")
            if os.path.isfile(metrics_path):
                # Load run_log.json and collect values
                with open(metrics_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                best_iter = data["meta"]["best_iter"]

                # Find the log dict with matching "step"
                match = None
                for d in data["logs"]:
                    if d["step"] == best_iter:
                        match = d
                        break

                # Store into buckets
                nlml.append(        match["nlml"]        )
                acc_train.append(   match["acc_train"]   )
                acc_test.append(    match["acc_test"]    )
                brier_train.append( match["brier_train"] )
                brier_test.append(  match["brier_test"]  )

        result_dict[nf_dummy]['nlml'       ] = nlml
        result_dict[nf_dummy]['acc_train'  ] = acc_train
        result_dict[nf_dummy]['acc_test'   ] = acc_test
        result_dict[nf_dummy]['brier_train'] = brier_train
        result_dict[nf_dummy]['brier_test' ] = brier_test
        
    return result_dict


def plot_nfs_runs(bucket: dict, title: str = None) -> None:
    """
    A dictionary with all info related to a given train / test split is given in input as `bucket`
    bucket.keys() are the number of spatial filters

    {nf0: {"nlml": [...],
           "acc_train": [...],
           "acc_test": [...],
           "brier_train": [...],
           "brier_test": [...]
        },
    }
    """
    labels = sorted(bucket.keys())

    nlmls = []
    acc_trains = []
    acc_tests = []
    brier_trains = []
    brier_tests = []

    for key in labels:

        nlmls.append(       bucket[key]['nlml']        )
        acc_trains.append(  bucket[key]['acc_train']  )
        acc_tests.append(   bucket[key]['acc_test']   )
        brier_trains.append(bucket[key]['brier_train'])
        brier_tests.append( bucket[key]['brier_test'] )

    medianprops_train    = dict(linestyle='-', linewidth=2.5, color=colors[0])
    medianprops_test     = dict(linestyle='-', linewidth=2.5, color=colors[1])
    meanpointprops_train = dict(marker='o', markerfacecolor=colors[0], markersize=7, markeredgecolor='none')
    meanpointprops_test  = dict(marker='^', markerfacecolor=colors[1], markersize=7, markeredgecolor='none')
    flierprops_train     = dict(marker='o', markerfacecolor='none', markersize=4, markeredgecolor=colors[0])
    flierprops_test      = dict(marker='^', markerfacecolor='none', markersize=4, markeredgecolor=colors[1])

    handles = {'train median': Line2D([], [], **medianprops_train), 
               'test median' : Line2D([], [], **medianprops_test), 
               'train mean'  : Line2D([], [], **meanpointprops_train), 
               'test mean'   : Line2D([], [], **meanpointprops_test), 
    }

    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    fig.suptitle(f'{title}')
    ax = ax.ravel()
    # NLML scatter per num of spatial filters
    ax[0].boxplot(nlmls, tick_labels=labels, flierprops=flierprops_train, medianprops=medianprops_train, meanprops=meanpointprops_train, showmeans=True)
    ax[0].set_xlabel("Number of spatial filters")
    ax[0].set_ylabel("NLML")    
    handle_labels = ['train median', 'train mean']
    ax[0].legend(handles=[handles[key] for key in handle_labels], labels=handle_labels, loc='best')
    # accuracy scatter per num of spatial filters
    ax[1].boxplot(acc_trains, tick_labels=labels, flierprops=flierprops_train, medianprops=medianprops_train, meanprops=meanpointprops_train, showmeans=True)
    ax[1].boxplot(acc_tests,  tick_labels=[''] * len(labels), flierprops=flierprops_test, medianprops=medianprops_test,  meanprops=meanpointprops_test, showmeans=True)
    ax[1].set_xlabel("Number of spatial filters")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_ylim(0, 1)
    handle_labels = ['train median', 'test median', 'train mean', 'test mean']
    ax[1].legend(handles=[handles[key] for key in handle_labels], labels=handle_labels, loc='best')
    # brier's scatterrper num of spatial filters
    ax[2].boxplot(brier_trains, tick_labels=labels, flierprops=flierprops_train, medianprops=medianprops_train, meanprops=meanpointprops_train, showmeans=True)
    ax[2].boxplot(brier_tests,  tick_labels=[''] * len(labels), flierprops=flierprops_test, medianprops=medianprops_test,  meanprops=meanpointprops_test, showmeans=True)
    ax[2].set_xlabel("Number of spatial filters")
    ax[2].set_ylabel("Brier score") 
    ax[2].set_ylim(0, 1)
    handle_labels = ['train median', 'test median', 'train mean', 'test mean']
    ax[2].legend(handles=[handles[key] for key in handle_labels], labels=handle_labels, loc='best')

    plt.tight_layout()
    plt.show()


def plot_nfs_runs_new(
    buckets: Union[Dict[str, dict], dict],
    title: Optional[str] = None,
    err: str = "std",
    offset_step: float = 0.08,
    connect_means: bool = True,
    show_train_test_panels: bool = True,
    nfs_to_show: Optional[List[Hashable]] = None,
) -> None:
    """
    Plot metrics for one or more experimental buckets using mean points with error bars.

    Layout (fixed):
        Row 0: [ NLML | Accuracy (train) | Brier (train) ]
        Row 1: [ (empty) | Accuracy (test) | Brier (test) ]

    If show_train_test_panels is False, only NLML (0,0), Accuracy (test) (1,1),
    and Brier (test) (1,2) are rendered. Unused panels are hidden.

    Other notes preserved from original:
    - Each metric is plotted as mean ± error (std or sem).
    - Multiple buckets are colored differently using Matplotlib's color cycle.
    - X positions are nudged per bucket so series don't overlap.
    """

    # -------------------------------------------------------------------------
    # Normalize input to a "name -> bucket" mapping
    # -------------------------------------------------------------------------
    def _looks_like_single_bucket(d: dict) -> bool:
        """Heuristic to detect the original single-bucket format."""
        if not isinstance(d, dict) or not d:
            return False
        sample_key = next(iter(d))
        return isinstance(d[sample_key], dict) and any(
            k in d[sample_key]
            for k in ("nlml", "acc_train", "acc_test", "brier_train", "brier_test")
        )

    if _looks_like_single_bucket(buckets):
        buckets = {"Bucket 1": buckets}  # type: ignore[assignment]

    bucket_names = list(buckets.keys())

    # Determine x-axis nf labels
    if nfs_to_show is None:
        # Use union across buckets if user didn't specify
        nfs_to_show = sorted(
            set().union(*[set(buckets[name].keys()) for name in bucket_names])
        )
    # Base x positions strictly follow the provided nfs_to_show order
    nf_labels = list(nfs_to_show)
    x_base = np.arange(len(nf_labels), dtype=float)

    def _collect_metric(bdict: dict, metric_key: str) -> List[Optional[List[float]]]:
        """
        For a single bucket dictionary, collect the list of values for each nf in nf_labels.
        Missing nf or metric_key yields None to indicate no data for that position.
        """
        vals_per_nf = []
        for nf in nf_labels:
            if nf in bdict and metric_key in bdict[nf]:
                vals_per_nf.append(bdict[nf][metric_key])
            else:
                vals_per_nf.append(None)
        return vals_per_nf

    def _compute_stats(metric_key: str):
        """
        Compute mean and error arrays for each bucket aligned to nf_labels.
        Missing values become NaN so they drop cleanly during plotting.
        """
        means, errs, masks = [], [], []
        for name in bucket_names:
            series = _collect_metric(buckets[name], metric_key)
            m_arr, e_arr, keep_mask = [], [], []
            for v in series:
                if v is None or len(v) == 0:
                    m_arr.append(np.nan)
                    e_arr.append(np.nan)
                    keep_mask.append(False)
                    continue
                v = np.asarray(v, dtype=float)
                m_arr.append(np.nanmean(v))
                if err == "sem":
                    # Standard error with ddof=1 and finite-count guard
                    finite = np.isfinite(v)
                    n_eff = np.sum(finite)
                    e_arr.append(
                        np.nan if n_eff <= 1
                        else np.nanstd(v[finite], ddof=1) / np.sqrt(n_eff)
                    )
                else:
                    e_arr.append(np.nanstd(v, ddof=1))
                keep_mask.append(True)
            means.append(np.array(m_arr, dtype=float))
            errs.append(np.array(e_arr, dtype=float))
            masks.append(np.array(keep_mask, dtype=bool))
        return means, errs, masks

    # -------------------------------------------------------------------------
    # Fixed 2x3 layout mapping
    # -------------------------------------------------------------------------
    # Always create a 2x3 grid; selectively plot into specific cells.
    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
    )

    if title:
        fig.suptitle(title)

    # Color cycle for buckets
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", None)
    if not color_cycle:
        color_cycle = [f"C{i}" for i in range(10)]

    # Compute per-bucket x offsets centered around zero
    n_buckets = len(bucket_names)
    offset_center = (n_buckets - 1) / 2.0
    x_offsets = [offset_step * (i - offset_center) for i in range(n_buckets)]

    # Define which panels to render based on the requested layout
    layout_all = {
        (0, 0): ("NLML", "nlml"),
        (0, 1): ("Accuracy (train)", "acc_train"),
        (0, 2): ("Brier (train)", "brier_train"),
        (1, 1): ("Accuracy (test)", "acc_test"),
        (1, 2): ("Brier (test)", "brier_test"),
    }

    # If train/test panels are disabled, keep only NLML and test metrics
    if not show_train_test_panels:
        layout = {(0, 0): layout_all[(0, 0)],
                  (1, 1): layout_all[(1, 1)],
                  (1, 2): layout_all[(1, 2)]}
    else:
        layout = layout_all

    # First hide everything; then render only the chosen positions
    for r in range(n_rows):
        for c in range(n_cols):
            axes[r, c].set_visible(False)

    # Render selected panels
    for (r, c), (pretty_name, metric_key) in layout.items():
        ax = axes[r, c]
        ax.set_visible(True)
        means, errs, masks = _compute_stats(metric_key)

        for i, name in enumerate(bucket_names):
            color = color_cycle[i % len(color_cycle)]
            x = x_base + x_offsets[i]
            y = means[i]
            yerr = errs[i]

            # Only plot finite points; keep x ticks regardless to preserve structure
            valid = np.isfinite(y) & np.isfinite(yerr)
            if not np.any(valid):
                continue

            ax.errorbar(
                x[valid],
                y[valid],
                yerr=yerr[valid],
                fmt="o",                 # circle marker for the mean
                capsize=3,
                elinewidth=1.2,
                linewidth=1.2 if connect_means else 0,
                markersize=5.5,
                label=name,
                color=color,
            )
            if connect_means:
                ax.plot(x[valid], y[valid], linewidth=1.2, color=color)

        # Always show the requested columns as ticks, even if empty for some buckets
        ax.set_xticks(x_base)
        ax.set_xticklabels([str(l) for l in nf_labels])
        ax.set_xlabel("Number of spatial filters")
        ax.set_ylabel(pretty_name)

        if "Accuracy" in pretty_name or "Brier" in pretty_name:
            ax.set_ylim(0, 1)

        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    plt.show()


# ---- Input root directory ----
#root = "/mnt/c/Users/scana/Desktop/gpc/results/data_set_IVa_symm_shrink002_aa"
#n_data, k_fold = 168, 6

#root = "/mnt/c/Users/scana/Desktop/gpc/results/data_set_IVa_symm_shrink002_av"
#n_data, k_fold = 84, 4

#root = "/mnt/c/Users/scana/Desktop/gpc/results/data_set_IVa_symm_shrink002_aa"
#n_data, k_fold = 224, 8

root = "/mnt/c/Users/scana/Desktop/gpc/results/data_set_IVa_symm_aa"
n_data, k_fold = 224, 8


n_subs = 1
kernel_type = 'RBF'
logged_flag = True
dicts_plot = {}

for align in os.listdir(root):
    align_path = os.path.join(root, align)
    
    for mode in os.listdir(align_path):
        mode_path = os.path.join(align_path, mode)
        
        result_dict = get_result_dict(mode_path)
        
        if mode   == 'spatialFilter'    : key = 'W'
        elif mode == 'spatialFilter_CSP': key = 'CSP'
        elif mode == 'spatialFilter_qr' : key = 'Q'
        
        if align   == 'no_align'       : pass
        elif align == 'riemann_align'  : key += ' (Riemann)'
        elif align == 'euclidean_align': key += ' (Euclidean)'
        
        dicts_plot.update( { key : result_dict } )
    
first_line = rf'{k_fold} folds x {n_subs} subjects --> N_data {n_data}'

if kernel_type == 'RBF': second_line = 'RBF covariance function'
else:                    second_line = 'linear covariance function'

if logged_flag: third_line = 'log-variance space'
else:           third_line = 'variance space'

plot_nfs_runs_new(
    dicts_plot,
    title=f'{first_line}\n{second_line} in {third_line}',
    err="std",
    nfs_to_show=[1, 2, 4, 8, 12, 16, 20],
)