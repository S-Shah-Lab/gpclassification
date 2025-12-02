"""
Plot the results coming from `run_class_bci_competition_III_focused_seed.py`
Plot the results coming from `run_class_bci_competition_III_random_seed.py`
"""

import os
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from collections import defaultdict


# ---- Input root directory ----
root = "/mnt/c/Users/scana/Desktop/gpc/results/gpflow/data_set_IVa_aa/seed_random"
# root = "/mnt/c/Users/scana/Desktop/gpc/results/gpflow/data_set_IVa_aa/seed_focused"

# ---- Buckets to store results ----
splits = []  # float(X.Y)
states = []  # int(A)
best_iters = []  # int
acc_tests = []  # float
acc_trains = []  # float
brier_trains = []  # float
brier_tests = []  # float
nlmls = []  # float
metric_paths = []  # str: path to metrics.json
N_trains = []  # float
N_tests = []  # float

# ---- Walk: split layer -> nf layer -> run layer -> run_log.json ----
for split_name in sorted(os.listdir(root)):
    split_path = os.path.join(root, split_name)
    if not os.path.isdir(split_path):
        continue
    else:
        print(f"{split_name}")

    # Extract X.Y as float from "split_X.Y"
    split_str = split_name.split("_", 1)[1]
    split_val = float(split_str)

    for nf_name in sorted(os.listdir(split_path)):
        nf_path = os.path.join(split_path, nf_name)
        if not os.path.isdir(nf_path):
            continue
        else:
            print(f"  {nf_name}")

        # Extract X as int from "nf_X"
        nf_str = nf_name.split("_", 1)[1]
        nf_val = int(nf_str)

        for run_name in sorted(os.listdir(nf_path)):
            run_path = os.path.join(nf_path, run_name)
            if not os.path.isdir(run_path):
                continue
            else:
                print(f"    {run_name}")

            metrics_path = os.path.join(run_path, "run_log.json")

            if os.path.isfile(metrics_path):
                # ---- Load run_log.json and collect values ----
                with open(metrics_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                best_iter = data["meta"]["best_iter"]
                state = data["meta"]["random_state"]
                N_train = data["meta"].get("N_train", np.nan)
                N_test = data["meta"].get("N_test", np.nan)

                # Find the log dict with matching "step"
                match = None
                for d in data["logs"]:
                    if d["step"] == best_iter:
                        match = d
                        break

                if match is None:
                    continue

                # Append to lists
                splits.append(split_val)
                states.append(state)
                best_iters.append(best_iter)
                acc_tests.append(match["acc_test"])
                acc_trains.append(match["acc_train"])
                brier_trains.append(match["brier_train"])
                brier_tests.append(match["brier_test"])
                nlmls.append(match["nlml"])
                metric_paths.append(metrics_path)
                N_trains.append(N_train)
                N_tests.append(N_test)

# Convert to numpy arrays for easy handling
splits_arr = np.array(splits, dtype=float)
acc_tests_arr = np.array(acc_tests, dtype=float)
acc_trains_arr = np.array(acc_trains, dtype=float)
brier_trains_arr = np.array(brier_trains, dtype=float)
brier_tests_arr = np.array(brier_tests, dtype=float)
nlml_arr = np.array(nlmls, dtype=float)
N_trains = np.array(N_trains, dtype=float)
N_tests = np.array(N_tests, dtype=float)

def _group_by_split_for_boxplots(
    splits_v: np.ndarray,
    nlml_v: np.ndarray,
    acc_tr_v: np.ndarray,
    acc_te_v: np.ndarray,
    brier_tr_v: np.ndarray,
    brier_te_v: np.ndarray,
):
    """Group metrics by split value for boxplot-style aggregation.

    Returns
    -------
    labels : list[float]
        Sorted unique split values.
    nlmls : list[list[float]]
        NLML samples per split.
    acc_trains, acc_tests, brier_trains, brier_tests : list[list[float]]
        Metric samples per split.
    counts : dict[float, int]
        Number of runs per split.
    """
    buckets = defaultdict(lambda: {
        "nlml": [],
        "acc_train": [],
        "acc_test": [],
        "brier_train": [],
        "brier_test": [],
    })

    for s, n, atr, ate, btr, bte in zip(
        splits_v, nlml_v, acc_tr_v, acc_te_v, brier_tr_v, brier_te_v
    ):
        b = buckets[s]
        b["nlml"].append(float(n))
        b["acc_train"].append(float(atr))
        b["acc_test"].append(float(ate))
        b["brier_train"].append(float(btr))
        b["brier_test"].append(float(bte))

    labels = sorted(buckets.keys())
    nlmls = [buckets[k]["nlml"] for k in labels]
    acc_trains = [buckets[k]["acc_train"] for k in labels]
    acc_tests = [buckets[k]["acc_test"] for k in labels]
    brier_trains = [buckets[k]["brier_train"] for k in labels]
    brier_tests = [buckets[k]["brier_test"] for k in labels]
    counts = {k: len(buckets[k]["nlml"]) for k in labels}

    return labels, nlmls, acc_trains, acc_tests, brier_trains, brier_tests, counts


def plot_random_seed_style_like_spatial():
    """Plot distributions by split using the boxplot style from
    plot_metric_spatial_filters.py (median lines, mean markers, overlaid
    train/test for accuracy and Brier). The inputs here are grouped across
    random seeds at each split value, regardless of the number of spatial
    filters. This adapts the style to the different data layout.
    """
    (
        labels,
        nlmls,
        acc_trains,
        acc_tests,
        brier_trains,
        brier_tests,
        counts,
    ) = _group_by_split_for_boxplots(
        splits_arr, nlml_arr, acc_trains_arr, acc_tests_arr, brier_trains_arr, brier_tests_arr
    )
    
    print(counts)

    # Styling props to match the spatial_filters script
    medianprops_train = dict(linestyle='-', linewidth=2.5, color='blue')
    medianprops_test = dict(linestyle='-', linewidth=2.5, color='orange')
    meanpointprops_train = dict(marker='o', markerfacecolor='blue', markersize=7, markeredgecolor='none')
    meanpointprops_test = dict(marker='^', markerfacecolor='orange', markersize=7, markeredgecolor='none')
    flierprops_train = dict(marker='o', markerfacecolor='none', markersize=4, markeredgecolor='blue')
    flierprops_test = dict(marker='^', markerfacecolor='none', markersize=4, markeredgecolor='orange')

    handles = {
        'train median': Line2D([], [], **medianprops_train),
        'test median': Line2D([], [], **medianprops_test),
        'train mean': Line2D([], [], **meanpointprops_train),
        'test mean': Line2D([], [], **meanpointprops_test),
    }

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Runs per split: {counts[0.1]}")
    ax = ax.ravel()

    # NLML boxplots per split
    ax[0].boxplot(
        nlmls,
        tick_labels=labels,
        flierprops=flierprops_train,
        medianprops=medianprops_train,
        meanprops=meanpointprops_train,
        showmeans=True,
    )
    ax[0].set_xlabel("Train fraction")
    ax[0].set_ylabel("NLML")
    handle_labels = ['train median', 'train mean']
    ax[0].legend(
        handles=[handles[key] for key in handle_labels],
        labels=handle_labels,
        loc='best',
    )

    # Accuracy: overlay train and test boxplots at same x positions
    ax[1].boxplot(
        acc_trains,
        tick_labels=labels,
        flierprops=flierprops_train,
        medianprops=medianprops_train,
        meanprops=meanpointprops_train,
        showmeans=True,
    )
    ax[1].boxplot(
        acc_tests,
        tick_labels=[''] * len(labels),
        flierprops=flierprops_test,
        medianprops=medianprops_test,
        meanprops=meanpointprops_test,
        showmeans=True,
    )
    ax[1].set_xlabel("Train fraction")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_ylim(0, 1)
    handle_labels = ['train median', 'test median', 'train mean', 'test mean']
    ax[1].legend(
        handles=[handles[key] for key in handle_labels],
        labels=handle_labels,
        loc='best',
    )

    # Brier: overlay train and test boxplots
    ax[2].boxplot(
        brier_trains,
        tick_labels=labels,
        flierprops=flierprops_train,
        medianprops=medianprops_train,
        meanprops=meanpointprops_train,
        showmeans=True,
    )
    ax[2].boxplot(
        brier_tests,
        tick_labels=[''] * len(labels),
        flierprops=flierprops_test,
        medianprops=medianprops_test,
        meanprops=meanpointprops_test,
        showmeans=True,
    )
    ax[2].set_xlabel("Train fraction")
    ax[2].set_ylabel("Brier score")
    ax[2].set_ylim(0, 1)
    handle_labels = ['train median', 'test median', 'train mean', 'test mean']
    ax[2].legend(
        handles=[handles[key] for key in handle_labels],
        labels=handle_labels,
        loc='best',
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_random_seed_style_like_spatial()
