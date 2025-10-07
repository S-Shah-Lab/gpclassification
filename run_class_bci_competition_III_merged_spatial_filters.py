from __future__ import annotations

import os, pickle
import gpflow
from typing import Dict, List, Tuple, Sequence

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from gp_classification import GPClassificationRunner
from Whitening import SpatialWhiteningDecomposition  # SVD + Whitening Rayleigh CSP


def _csp_filters_from_covs(X_cov: np.ndarray, y: np.ndarray, nf: int, *, max_rank=None) -> np.ndarray:
    """
    Implementation of the code provided by Jeremy in `Whitening.py` and `SVD.py`
    Compute CSP spatial filters W (channels x nf) from per-trial covariance matrices
    X_cov: (n_trials, n_channels, n_channels) sample covariances for trials in the TRAIN split
    y    : (n_trials,) with values in {0,1}
    nf   : even number of filters to return, taken symmetrically from top/bottom
    """
    if nf <= 0 or nf % 2 != 0:
        raise ValueError("nf must be a positive even integer")

    y = np.asarray(y).ravel().astype(int)
    if not {0, 1}.issuperset(set(np.unique(y))):
        raise ValueError("y must contain only {0,1}")

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    if idx0.size == 0 or idx1.size == 0:
        raise ValueError("Both classes must be present in the training split")

    # trace-normalize each trial covariance, then average per class
    def _tr_norm(c): 
        t = np.trace(c)
        return c / (t if t > 1e-12 else 1e-12)

    C0 = np.mean([_tr_norm(C) for C in X_cov[idx0]], axis=0)
    C1 = np.mean([_tr_norm(C) for C in X_cov[idx1]], axis=0)
    Csum = 0.5 * (C0 + C1 + (C0 + C1).T)  # symmetrize
    C1   = 0.5 * (C1 + C1.T)

    # Whiten w.r.t. Csum, then Rayleigh on C1 (stable even if rank-deficient)
    # Whitening + Rayleigh are provided by your code:
    #   - SpatialWhiteningDecomposition(...).Whiten(...)
    #   - .Rayleigh(H) gives R (orthonormal in whitened space) and W = P @ R
    d = SpatialWhiteningDecomposition(sensorCovariance=Csum, maxRank=max_rank)
    d.Rayleigh(C1)  # columns ordered by descending eigenvalue
    W_all = d.W  # (n_channels, n_sources)
    # pick symmetric extremes
    half = nf // 2
    sel = np.r_[0:half, W_all.shape[1] - half: W_all.shape[1]]
    return W_all[:, sel].copy()


def generate_kfold_indices(
    n_samples: int,
    n_splits: int = 8,
    shuffle: bool = True,
    random_state: int = 42
) -> List[Dict[str, np.ndarray]]:
    """
    Create KFold train/test index splits

    Returns:
        List[Dict[str, np.ndarray]]: A list with dicts having `train_idx` and `test_idx` arrays
    """
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    indices: List[Dict[str, np.ndarray]] = []
    for train_idx, test_idx in kf.split(np.arange(n_samples)):
        indices.append({"train_idx": train_idx, "test_idx": test_idx})
    return indices


def _stacked_barh_segmented(
    y_pos: float,
    left: float,
    total_width: float,
    counts: Dict[str, int],
    ax: plt.Axes,
    *,
    keys_order: Sequence[str] | None = None,
    color_map: Dict[str, str] | None = None,
    legend_once: set | None = None,
) -> None:
    """
    Draw a segmented horizontal bar (stacked) starting at `left` with total
    width `total_width`. Segment widths are proportional to `counts` values.
    Uses keys_order for stable segment order across folds. Adds each label
    to the legend only once via legend_once.
    """
    n = sum(counts.values())
    if n == 0 or total_width <= 0:
        return

    if legend_once is None:
        legend_once = set()

    keys = list(keys_order) if keys_order is not None else sorted(counts.keys())
    x0 = left
    for key in keys:
        c = counts.get(key, 0)
        if c <= 0:
            continue
        frac = c / n
        w = total_width * frac
        if w <= 0:
            continue

        # Only add a legend label the first time this key appears
        label = None
        if key not in legend_once:
            label = key
            legend_once.add(key)

        ax.barh(
            y_pos,
            w,
            left=x0,
            align="center",
            label=label,
            color=(color_map.get(key) if color_map and key in color_map else None),
            edgecolor="none",
        )
        x0 += w


def _add_split_arrows(ax: plt.Axes, split: float, n_folds: int) -> None:
    """
    Draw left/right horizontal arrows starting at `split`, placed
    below the last fold. Adds 'train' (left) and 'test' (right) labels.
    """
    arrow_y = -1.2
    delta = min(0.1, (1 - split) / 2)

    # Left arrow: from split -> slightly left
    x1, x2 = split, split - delta
    ax.annotate(
        "",
        xy=(x2, arrow_y),
        xytext=(x1, arrow_y),
        arrowprops=dict(arrowstyle="-|>", lw=1.5),
        annotation_clip=False,
    )
    ax.text((x1 + x2) / 2.0, arrow_y + 0.4, "train", ha="center", va="top", fontsize=10)

    # Right arrow: from split -> slightly right
    x1, x2 = split, split + delta
    ax.annotate(
        "",
        xy=(x2, arrow_y),
        xytext=(x1, arrow_y),
        arrowprops=dict(arrowstyle="-|>", lw=1.5),
        annotation_clip=False,
    )
    ax.text((x1 + x2) / 2.0, arrow_y + 0.4, "test", ha="center", va="top", fontsize=10)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(min(ymin, arrow_y - 0.3), ymax)


def plot_kfold_class_composition_from_dicts(
    folds: List[Dict[str, np.ndarray]],
    y: np.ndarray,
    *,
    k: int,
    class_names: Dict[int, str] | None = None,
    class_colors: Dict[int, str] | None = None,
    figsize: Tuple[float, float] = (10, 0.6),
) -> plt.Figure:
    """
    Plot with fixed class colors. X-axis is [0, 1], dashed split at 1 - 1/k.
    Train is [0, split), test is [split, 1]. Bars stacked by class composition.
    """
    split = 1.0 - 1.0 / float(k)
    n_folds = len(folds)

    fig, ax = plt.subplots(figsize=(figsize[0], max(figsize[1] * n_folds, 3)))

    unique_classes = np.unique(y)

    # Stable class names and order by integer class id
    id_to_name = {
        int(c): (class_names[int(c)] if class_names and int(c) in class_names else f"class {int(c)}")
        for c in unique_classes
    }
    keys_order = [id_to_name[int(c)] for c in sorted(unique_classes.astype(int))]

    # Build color map keyed by display name
    color_map = {}
    if class_colors:
        for cid, hexcol in class_colors.items():
            cid = int(cid)
            if cid in id_to_name:
                color_map[id_to_name[cid]] = hexcol

    legend_once: set = set()

    # Plot fold 0 at top, increasing downward
    for i, f in enumerate(folds):
        train_idx, test_idx = f["train_idx"], f["test_idx"]
        y_pos = n_folds - 1 - i

        train_counts = {id_to_name[int(c)]: int(np.sum(y[train_idx] == c)) for c in unique_classes}
        test_counts  = {id_to_name[int(c)]: int(np.sum(y[test_idx] == c))  for c in unique_classes}

        _stacked_barh_segmented(
            y_pos=y_pos,
            left=0.0,
            total_width=split,
            counts=train_counts,
            ax=ax,
            keys_order=keys_order,
            color_map=color_map,
            legend_once=legend_once,
        )

        _stacked_barh_segmented(
            y_pos=y_pos,
            left=split,
            total_width=1.0 - split,
            counts=test_counts,
            ax=ax,
            keys_order=keys_order,
            color_map=color_map,
            legend_once=legend_once,
        )

    ax.axvline(split, linestyle="--")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-1, n_folds)
    ax.set_yticks([n_folds - 1 - i for i in range(n_folds)])
    ax.set_yticklabels([f"Fold {i}" for i in range(n_folds)])
    ax.set_xlabel("Fraction of total dataset")
    ax.set_title(f"{k}-fold class composition")

    _add_split_arrows(ax, split=split, n_folds=n_folds)

    # Compact, deduped legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc="best", ncols=2, fontsize=9)

    plt.tight_layout()
    return fig


def _make_color_map_from_groups(
    unique_groups: Sequence[str],
    *,
    cmap_name: str = "tab20",
) -> Dict[str, str]:
    """
    Deterministic color map for group names. Sorted alphabetically.
    """
    unique_groups = [str(g) for g in unique_groups]
    unique_groups = sorted(unique_groups)

    cmap = plt.get_cmap(cmap_name)
    n_colors = getattr(cmap, "N", 20)
    color_map: Dict[str, str] = {}
    for i, g in enumerate(unique_groups):
        rgba = cmap(i % n_colors)
        hexcol = "#{:02X}{:02X}{:02X}".format(
            int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        )
        color_map[g] = hexcol
    return color_map


def plot_kfold_group_origin_from_dicts(
    folds: List[Dict[str, np.ndarray]],
    groups: Sequence[str],
    *,
    k: int,
    group_colors: Dict[str, str] | None = None,
    order: str = "global_freq",
    max_legend: int | None = None,
    figsize: Tuple[float, float] = (10, 0.6),
) -> plt.Figure:
    """
    For each fold, show a single horizontal bar over x in [0, 1].
    Vertical dashed line at split = 1 - 1/k. Train occupies [0, split),
    test occupies [split, 1]. Each region is stacked by GROUP of origin.
    """
    groups = np.asarray(groups, dtype=object)
    n_folds = len(folds)
    split = 1.0 - 1.0 / float(k)

    # Stable group order
    uniq, counts = np.unique(groups, return_counts=True)
    uniq = uniq.astype(str)

    if order == "alpha":
        keys_order = list(np.sort(uniq))
    else:
        order_idx = np.lexsort((uniq, -counts))
        keys_order = [uniq[i] for i in order_idx]

    # Colors: dynamic fallback + any user overrides
    if group_colors is None:
        color_map = _make_color_map_from_groups(uniq, cmap_name="tab20")
    else:
        dynamic_fallback = _make_color_map_from_groups(uniq, cmap_name="tab20")
        color_map = {**dynamic_fallback, **{str(k): v for k, v in group_colors.items()}}

    fig, ax = plt.subplots(figsize=(figsize[0], max(figsize[1] * n_folds, 3)))

    legend_once: set = set()

    for i, f in enumerate(folds):
        train_idx, test_idx = f["train_idx"], f["test_idx"]
        y_pos = n_folds - 1 - i

        train_counts = {str(g): int(np.sum(groups[train_idx] == g)) for g in uniq}
        test_counts  = {str(g): int(np.sum(groups[test_idx] == g))  for g in uniq}

        _stacked_barh_segmented(
            y_pos=y_pos,
            left=0.0,
            total_width=split,
            counts=train_counts,
            ax=ax,
            keys_order=keys_order,
            color_map=color_map,
            legend_once=legend_once,   # <-- keep the same set, don't rebuild it
        )

        _stacked_barh_segmented(
            y_pos=y_pos,
            left=split,
            total_width=1.0 - split,
            counts=test_counts,
            ax=ax,
            keys_order=keys_order,
            color_map=color_map,
            legend_once=legend_once,   # <-- same set
        )

    ax.axvline(split, linestyle="--")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-1, n_folds)
    ax.set_yticks([n_folds - 1 - i for i in range(n_folds)])
    ax.set_yticklabels([f"Fold {i}" for i in range(n_folds)])
    ax.set_xlabel("Fraction of total dataset")
    ax.set_title(f"{k}-fold source composition by group")

    _add_split_arrows(ax, split=split, n_folds=n_folds)

    # Deduped legend, with optional truncation
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        if max_legend is not None:
            keep_order = [k for k in keys_order if k in by_label][:max_legend]
            by_label = {k: by_label[k] for k in keep_order}
        ax.legend(by_label.values(), by_label.keys(), loc="best", ncols=2, fontsize=9)

    plt.tight_layout()
    return fig



# File to run on
path_to_file = "data_set_IVa_merged.pkl"
# data_set_IVa_aa: 168
# data_set_IVa_al: 224
# data_set_IVa_av:  84
# data_set_IVa_aw:  56
# data_set_IVa_ay:  28
# Total trials   : 560
# e.g. 8-fold is possible with 70 trials per fold

# Load input data: dict
with open(f"./data/{path_to_file}", "rb") as f:
    data = pickle.load(f)
    """
    data.keys()
    'X', 
    'Y', 
    'ch_names', 
    'ch_location', 
    'dataset_labels', 
    'trial_counts_by_file', 
    'window_seconds', 
    'band_hz', 
    'notes'
    """

kfolds = 8
n_samples = len(data["Y"])
folds_dicts = generate_kfold_indices(n_samples, n_splits=kfolds, shuffle=True, random_state=42)
k = len(folds_dicts)

# Number of spatial filters to consider
nfs = [2, 4, 8, 12, 16, 20, 30]

dataset_label = 'data_set_IVa_merged'

# Consider random states for random-influenced events
# - Initialization of W matrix if `spatialFilter_init` is set to `random`
# - train/test split if arrays are passed instead of dictionaries
spatialFilter_init = "random"
random_state = 42
W_trainable = True
spatial_filters = {}



# Visualize the folds
CLASS_COLORS = {
    0: "#4C78A8",  # class 0
    1: "#F58518",  # class 1
}
# Based on class composition
fig1 = plot_kfold_class_composition_from_dicts(
    folds=folds_dicts,
    y=data["Y"],
    k=k,
    class_names=None,  # or e.g. {0: "benign", 1: "malignant"}
    class_colors=CLASS_COLORS,
)
plt.show()
# Based on group of origin
fig2 = plot_kfold_group_origin_from_dicts(
    folds=folds_dicts,
    groups=data["groups"],
    k=len(folds_dicts),
    group_colors=None,          # dynamic palette
    order="global_freq",        # or "alpha"
    max_legend=20,
)
plt.show()




for nf in nfs:
    spatial_filters[nf] = [] # dictionary for spatial filters with given number of cols
    
    for fold_dict in folds_dicts:
        
        train_idx = fold_dict['train_idx']
        test_idx = fold_dict['test_idx']
        
        X_train = data['X'][train_idx]
        X_test = data['X'][test_idx]
        Y_train = data['Y'][train_idx]
        Y_test = data['Y'][test_idx]
        
        if W_trainable == False:
            # Now that train and test sets for a given fold have been defined, compute W matrix using CSP
            W_fold = _csp_filters_from_covs(X_train, Y_train, nf=nf, max_rank=None)
            spatial_filters[nf].append(W_fold)
            spatialFilter_init = W_fold
            results_dir = f"/mnt/c/Users/scana/Desktop/gpc/results/{dataset_label}/spatialFilter_CSP/nf_{nf}"
        
        else:
            results_dir = f"/mnt/c/Users/scana/Desktop/gpc/results/{dataset_label}/spatialFilter/nf_{nf}"
        
        # Generate dictionary for input
        my_dict = {
            "X": {
                "train": X_train,
                "test": X_test,
            },
            "Y": {
                "train": Y_train,
                "test": Y_test,
            },
        }

        runner = GPClassificationRunner(
            # Input variables
            X=my_dict["X"],
            Y=my_dict["Y"],
            dataset_label=dataset_label,
            ch_names=data["ch_names"],
            ch_xy=data["ch_location"],
            # Model / kernel
            spatialFilter_init=spatialFilter_init,
            nf=nf,
            eta_flag=False,
            ard_flag=False,
            logged_flag=True,
            kernel_type="RBF",
            model_class=gpflow.models.VGP,
            likelihood_class=gpflow.likelihoods.Bernoulli,
            # Training
            learning_rate=0.1,  # Adam default learning rate
            gamma=0.1,  # Natural gradient default learning rate
            maxiter=2500,
            pred_threshold=0.5,  # decision boundary in binary classification p(y=1) >= pred_threshold
            random_state=random_state,
            # ----- Policy flags for adaptation / early stopping
            use_validation_for_adaptation=False,  # if True and val exists, adapt LR/ES on val; else train-only
            enable_adaptation=True,  # enable LR reduce-on-plateau on chosen set
            enable_early_stopping=False,  # enable early stopping on chosen set
            # Run naming / Logging
            results_dir=results_dir,
            run_name=None,
        )
        runner.run()
