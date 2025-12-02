from __future__ import annotations

import pickle, os
from   typing      import Dict, List
from   collections import OrderedDict

import numpy                   as     np
import matplotlib.pyplot       as     plt
from   matplotlib.patches      import Patch
from   matplotlib.lines        import Line2D
from   sklearn.model_selection import KFold
from   gp_classification_gpy2  import GPClassificationRunner
from   Whitening               import (
                                    SpatialWhiteningDecomposition,
                                    ApplySpatialFilters,
                                    Covariance,
                                )  # SVD + Whitening + Rayleigh / SSA
from   align                   import align_split 


def _csp_filters_from_covs(X_cov: np.ndarray, y: np.ndarray, nf: int, *, max_rank=None) -> np.ndarray:
    """
    Implementation of the code provided by Jeremy in `Whitening.py` and `SVD.py`
    Compute CSP spatial filters W (channels, nf) from per-trial covariance matrices
    X_cov: (n_trials, n_channels, n_channels) sample covariances for trials in the TRAIN split
    y    : (n_trials,) with values in {0,1}
    nf   : number of filters to return, taken symmetrically from top/bottom if the number is even
    
    `max_rank` is a nice tool that can force a wanted number of filters, we leave it as None and pick the filters based on eigenvalues
    """
    if nf <= 0: 
        raise ValueError("nf must be a positive integer")

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
    # Whitening + Rayleigh are provided by Jeremy's code:
    #   - SpatialWhiteningDecomposition(...).Whiten(...)
    #   - .Rayleigh(H) gives R (orthonormal in whitened space) and W = P @ R
    d = SpatialWhiteningDecomposition(sensorCovariance=Csum, maxRank=max_rank)
    d.Rayleigh(C1)  # columns ordered by descending eigenvalue

    W_all = d.W  # (n_channels, n_sources)
    evals = np.asarray(d.eigenvalues)
    
    if nf > W_all.shape[1]:
        raise ValueError(
            f"Requested nf={nf}, but only {W_all.shape[1]} filters are available"
        )
    
    # Transform the obtained eigenvalues
    # This makes them look like an upsidedown V with vertex at 0
    scores = -np.abs(evals - 0.5) 
    
    # Indices of eigenvalues sorted from most informative (lowest score, first place, <0)
    # to least informative (highest score, last place, ~0)
    sorted_idx = np.argsort(scores)

    # Select the first nf most informative components
    sel = sorted_idx[:nf]

    # Return corresponding spatial filters
    W_sel = W_all[:, sel].copy()
    return W_sel

def generate_kfold_indices(
    n_samples: int,
    n_splits: int = 8,
    shuffle: bool = True,
    random_state: int = None,
) -> List[Dict[str, np.ndarray]]:
    """
    Create KFold train/test index splits

    Returns:
        List[Dict[str, np.ndarray]]: A list with dicts having `train_idx` and `test_idx` arrays
    """
    if random_state is None:
        kf = KFold(n_splits=n_splits, shuffle=shuffle)
    else:
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    indices: List[Dict[str, np.ndarray]] = []
    for train_idx, test_idx in kf.split(np.arange(n_samples)):
        indices.append({"train_idx": train_idx, "test_idx": test_idx})
    return indices

def generate_train_test_from_fold(X_cov, y, train_idx, test_idx, alignment=None):
    """
    X_cov: (N, s, s) covariances
    y    : (N,)
    Returns dicts X,Y with per-split covariances
    If `alignment` is provided, input covariance matrices are transformed
    """
    Xtr_raw = X_cov[train_idx]
    Xte_raw = X_cov[test_idx]
    
    if alignment is not None:
        Xtr_al, Xte_al, mean_Mat = align_split(Xtr_raw, Xte_raw, method=alignment)
    else:
        Xtr_al   = Xtr_raw
        Xte_al   = Xte_raw
        mean_Mat = None

    # hand aligned 3D arrays to the runner; it will flatten to (N, s*s) itself
    X_dict = {"train": Xtr_al, "test": Xte_al}
    Y_dict = {"train": y[train_idx].astype(int), "test": y[test_idx].astype(int)}
    return X_dict, Y_dict

def generate_train_test_from_fold_ssa(
    X_eeg: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    alignment: str | None = None,
    ssa_max_rank: int | None = None,
    n_stationary: int | None = None,
):
    """
    SSA on time-domain epochs, then reuse `generate_train_test_from_fold`
    for splitting + optional alignment
    """
    # 1) Fit SSA on TRAIN epochs only
    Xtr_epochs = X_eeg[train_idx]  # (N_tr, C, T)
    if Xtr_epochs.ndim != 3:
        raise ValueError(f"Expected X_eeg with shape (N, C, T); got {Xtr_epochs.shape}")

    ssa = SpatialWhiteningDecomposition(
        mixedSignals=Xtr_epochs,
        sensorAxis=1,          # channels axis in (N, C, T)
        maxRank=ssa_max_rank,
    )
    # epochAxis=0 (trials), sensorAxis=1 (channels)
    ssa.SSA(epochAxis=0, trainingSubset=None)
    W_ssa = ssa.W  # (C, U)

    # Optionally restrict to stationary components only
    eigs = np.asarray(ssa.eigenvalues).ravel()
    n_sources = W_ssa.shape[1]

    if n_sources == 0:
        raise RuntimeError("SSA returned zero sources; cannot select stationary components.")

    # Eigenvalues are in descending order (largest = most non-stationary)
    # Smallest eigenvalues → most stationary
    order_asc = np.argsort(eigs)  # ascending
    if n_stationary is None or n_stationary > n_sources:
        n_keep = n_sources
    else:
        n_keep = int(n_stationary)
        if n_keep <= 0:
            raise ValueError("n_stationary must be positive if specified.")

    keep_idx = order_asc[:n_keep] # keep the first n_keep components from the ascending order (these are the most stationary ones)
    W_ssa = W_ssa[:, keep_idx]    # (C, n_keep)
    # Projection in sensor space, C x C
    F_stat = W_ssa @ W_ssa.T      # (C, C)
    
    # 2) Apply SSA filters to ALL epochs for this fold
    # We want covariances for ALL N trials so that train_idx/test_idx
    X_all_stat = ApplySpatialFilters(
        signal=X_eeg,              # (N, C, T)
        spatialFilteringMatrix=F_stat,  # (C, C) -> still (N, C, T)
        sensorAxis=1,
    )

    # 3) Compute covariance matrices in channel space (not in SSA space)
    X_cov_stat = np.stack(
        [Covariance(trial.T, preservedAxis=-1) for trial in X_all_stat],
        axis=0,
    )  # (N, C, C)

    # If you want to work in SSA space (2) and (3) would be: 
    #X_all_proj = ApplySpatialFilters(
    #    signal=X_eeg,                 # (N, C, T)
    #    spatialFilteringMatrix=W_ssa, # -> (N, n_stationary, T)
    #    sensorAxis=1,
    #)
    #X_cov_ssa = np.stack(
    #    [Covariance(trial.T, preservedAxis=-1) for trial in X_all_proj],
    #    axis=0,
    #)  # (N, U', U')

    # 4) Reuse the standard splitter + aligner
    X_dict, Y_dict = generate_train_test_from_fold(
        X_cov_stat,
        y,
        train_idx,
        test_idx,
        alignment=alignment,
    )

    return X_dict, Y_dict, W_ssa

def plot_fold_class_mix(
    folds_dicts,
    y,
    class_labels=(0, 1),
    colors_train={0: "cornflowerblue", 1: "gold"  },
    colors_test={0 : "blue",           1: "orange"},
    figsize=None,
    bar_height=0.6,
    annotate=False,
):
    """
    Visualize class composition for each fold as a stacked horizontal bar

    Parameters
    ----------
    folds_dicts : list[dict]
        Each element is a dict with keys 'train_idx' and 'test_idx' (int arrays)
    y : array-like, shape (N,)
        Class labels for all samples. Must contain only the provided class_labels
    class_labels : tuple[int, int]
        The two class labels, in the order they should be shown (0 then 1)
    colors_train : dict[int, str]
        Mapping class -> color for the train segments
    colors_test : dict[int, str]
        Mapping class -> color for the test segments
    figsize : tuple[float, float] or None
        Size of the figure in inches. If None, chooses something sensible
    bar_height : float
        Height of each horizontal bar
    annotate : bool
        If True, writes percentages on the segments (tiny text, don't expect poetry)

    Notes
    -----
    - Each fold's bar spans [0, 1] by normalizing segment widths by total samples in that fold (train + test)
    - Segment order per bar:
        [train class_labels[0], train class_labels[1], test class_labels[0], test class_labels[1]]
    """

    # Basic input checks
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y must be a 1D array of class labels")

    n_folds = len(folds_dicts)
    if n_folds == 0:
        raise ValueError("folds_dicts is empty. Nothing to plot, nothing to judge")

    # Figure sizing heuristic so it doesn't look like a postage stamp
    if figsize is None:
        figsize = (10, max(2.5, 0.6 * n_folds))

    fig, ax = plt.subplots(figsize=figsize)
    plt.suptitle('Class mixture')

    y_positions = np.arange(n_folds)

    for i, fd in enumerate(folds_dicts):
        train_idx = np.asarray(fd["train_idx"], dtype=int)
        test_idx  = np.asarray(fd["test_idx"], dtype=int)

        # Counts for each class in train and test
        total = train_idx.size + test_idx.size
        if total == 0:
            # This would be impressive in a bad way
            continue

        counts = {
            ("train", class_labels[0]): np.sum(y[train_idx] == class_labels[0]),
            ("train", class_labels[1]): np.sum(y[train_idx] == class_labels[1]),
            ("test",  class_labels[0]): np.sum(y[test_idx]  == class_labels[0]),
            ("test",  class_labels[1]): np.sum(y[test_idx]  == class_labels[1]),
        }

        # Convert to proportions of the entire fold
        props = {k: v / total for k, v in counts.items()}

        # Stacking order: train 0, train 1, test 0, test 1
        segments = [
            ("train", class_labels[0], colors_train[class_labels[0]]),
            ("train", class_labels[1], colors_train[class_labels[1]]),
            ("test",  class_labels[0], colors_test[ class_labels[0]]),
            ("test",  class_labels[1], colors_test[ class_labels[1]]),
        ]

        left = 0.0
        for split, cls, color in segments:
            width = props[(split, cls)]
            if width > 0:
                ax.barh(
                    y_positions[i],
                    width,
                    left=left,
                    height=bar_height,
                    color=color,
                    edgecolor="black",
                    linewidth=0.5,
                )
                if annotate and width >= 0.04:
                    ax.text(
                        left + width / 2,
                        y_positions[i],
                        f"{width:.2f}",
                        va="center",
                        ha="center",
                        fontsize=8,
                        color="white",
                    )
                left += width

    # Axes cosmetics: 0..1 scale, nice gridlines, fold labels
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.8, n_folds - 1 + 0.8)
    ax.set_xlabel("Proportion (train + test = 1)")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"Fold {i}" for i in range(n_folds)])
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # Legend that actually helps
    legend_patches = [
        Patch(facecolor=colors_train[class_labels[0]], edgecolor="black", label="Train 0"),
        Patch(facecolor=colors_train[class_labels[1]], edgecolor="black", label="Train 1"),
        Patch(facecolor=colors_test[class_labels[0]],  edgecolor="black", label="Test 0" ),
        Patch(facecolor=colors_test[class_labels[1]],  edgecolor="black", label="Test 1" ),
    ]
    ax.legend(handles=legend_patches, loc="lower left", frameon=True, ncols=4)

    fig.tight_layout()
    return fig, ax

def plot_fold_file_source_mix(
    folds_dicts,
    groups,
    trial_counts_by_file,
    figsize=None,
    bar_height=0.6,
    annotate=False,
    alpha_train=1.0,
    alpha_test=0.45,
    cmap_name="Set2",   # lighter qualitative palette; avoids dark browns
    source_order="dict",  # "dict" to respect trial_counts_by_file order, or "alpha" to sort by name
    annotate_min_width=0.04,
    annotate_fontsize=12,  # bigger font
):
    """
    Visualize per-fold data origin as stacked horizontal bars.
    For each fold, segments are:
       [train source_1, train source_2, ..., test source_1, test source_2, ...]
    All widths are proportions of the total samples in the fold (train + test).

    Parameters
    ----------
    folds_dicts : list[dict]
        Each element has 'train_idx' and 'test_idx' arrays (indices into groups).
    groups : array-like of str, shape (N,)
        For each trial index i, groups[i] is the source file name it came from.
        Must be one of the keys in trial_counts_by_file.
    trial_counts_by_file : dict[str, int]
        Mapping from source name to total trials in the whole dataset. Used to
        determine the list and order of sources and for basic sanity checks.
    figsize : tuple[float, float] or None
        Figure size in inches. Defaults to something reasonable based on #folds.
    bar_height : float
        Height of each horizontal bar.
    annotate : bool
        If True, annotate segment proportions inside segments.
    alpha_train : float
        Opacity for train segments (1.0 = opaque).
    alpha_test : float
        Opacity for test segments (e.g., 0.45 = faded).
    cmap_name : str
        Name of a matplotlib colormap to pull distinct colors from. Defaults to a light set.
    source_order : {"dict", "alpha"}
        - "dict": preserve insertion order of trial_counts_by_file for sources.
        - "alpha": alphabetical order of sources.
    annotate_min_width : float
        Minimum segment width (in proportion units) to place a label.
    annotate_fontsize : int
        Font size for the labels (black text).

    Returns
    -------
    (fig, ax) : matplotlib Figure and Axes
    """

    groups = np.asarray(groups)
    if groups.ndim != 1:
        raise ValueError("groups must be a 1D array-like of source names per trial.")

    # Derive and order sources
    if source_order == "alpha":
        sources = sorted(trial_counts_by_file.keys())
    else:
        sources = list(OrderedDict(trial_counts_by_file).keys())

    # Basic sanity checks: groups must reference only known sources
    unknown = set(np.unique(groups)) - set(sources)
    if unknown:
        raise ValueError(f"groups contains unknown sources: {sorted(unknown)}")

    # Optional check that total trials match counts (won’t fail if user sliced data)
    total_in_groups = groups.size
    total_in_counts = int(np.sum(list(trial_counts_by_file.values())))
    if total_in_groups != total_in_counts:
        print(f"[warn] groups length = {total_in_groups}, "
              f"sum(trial_counts_by_file) = {total_in_counts}. They differ.")

    n_folds = len(folds_dicts)
    if n_folds == 0:
        raise ValueError("folds_dicts is empty. Nothing to plot.")

    # Colors: one distinct, light-ish color per source from a qualitative cmap
    cmap = plt.get_cmap(cmap_name)
    # Sample evenly; if sources == 1, avoid division by zero
    colors = {src: cmap(i / max(1, len(sources) - 1)) for i, src in enumerate(sources)}

    # Figure size heuristic
    if figsize is None:
        figsize = (12, max(2.5, 0.6 * n_folds))

    fig, ax = plt.subplots(figsize=figsize)
    plt.suptitle('File source mixture')
    
    y_positions = np.arange(n_folds)

    for i, fd in enumerate(folds_dicts):
        train_idx = np.asarray(fd["train_idx"], dtype=int)
        test_idx  = np.asarray(fd["test_idx"], dtype=int)
        total = train_idx.size + test_idx.size
        if total == 0:
            continue

        # Counts per source for train and test
        counts_train = {src: int(np.sum(groups[train_idx] == src)) for src in sources}
        counts_test  = {src: int(np.sum(groups[test_idx]  == src)) for src in sources}

        # Convert to proportions over the whole fold
        props_train = {src: counts_train[src] / total for src in sources}
        props_test  = {src: counts_test[src]  / total for src in sources}

        # Plot stacking: first all train segments, then all test segments
        left = 0.0

        # Train segments
        for src in sources:
            width = props_train[src]
            if width <= 0:
                continue
            ax.barh(
                y_positions[i],
                width,
                left=left,
                height=bar_height,
                color=colors[src],
                alpha=alpha_train,
                edgecolor="black",
                linewidth=0.5,
            )
            if annotate and width >= annotate_min_width:
                ax.text(
                    left + width / 2,
                    y_positions[i],
                    f"{width:.2f}",
                    va="center",
                    ha="center",
                    fontsize=annotate_fontsize,
                    color="black",       # bigger and black font for numbers
                )
            left += width

        # Test segments
        for src in sources:
            width = props_test[src]
            if width <= 0:
                continue
            ax.barh(
                y_positions[i],
                width,
                left=left,
                height=bar_height,
                color=colors[src],
                alpha=alpha_test,  # faded to indicate "test"
                edgecolor="black",
                linewidth=0.5,
            )
            if annotate and width >= annotate_min_width:
                ax.text(
                    left + width / 2,
                    y_positions[i],
                    f"(* {width:.2f})",  # prefix with a star for test blocks
                    va="center",
                    ha="center",
                    fontsize=annotate_fontsize,
                    color="black",       # bigger and black font
                )
            left += width

    # Axes cosmetics
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.8, n_folds - 1 + 0.8)
    ax.set_xlabel("Proportion (train + test = 1)")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"Fold {i}" for i in range(n_folds)])
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # Legend: per-source color + a key for train/test opacity
    source_patches = [Patch(facecolor=colors[src], edgecolor="black", label=src) for src in sources]
    key_train = Patch(facecolor="gray", edgecolor="black", alpha=alpha_train, label="Train")
    key_test  = Patch(facecolor="gray", edgecolor="black", alpha=alpha_test,  label="Test (* label)")

    leg1 = ax.legend(handles=source_patches, title="Sources", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    ax.add_artist(leg1)
    ax.legend(handles=[key_train, key_test], title="Segment type", bbox_to_anchor=(1.02, 0.35), loc="upper left")

    fig.tight_layout()
    return fig, ax

def plot_train_test_density_by_fold(
    folds_dicts,
    y,
    class_labels=(0, 1),
    color_map={0: "blue", 1: "orange"},
    other_color="lightgray",
    figsize=None,
    pointsize_other=8,
    pointsize_test=20,
    y_pad=0.6,
):
    """
    Plot which indices belong to the test split for each fold.

    For each fold, this draws a horizontal row of points over the index axis:
      - All indices [0, ..., N-1] appear as small gray points (other_color).
      - The indices that are in the test split are overplotted and colored
        by class (class 0 -> blue, class 1 -> orange by default).

    Parameters
    ----------
    folds_dicts : list[dict]
        Each element must contain 'train_idx' and 'test_idx' arrays (ints).
    y : array-like, shape (N,)
        Class labels for all samples (must include only values in class_labels).
    class_labels : tuple
        The two class labels, default (0, 1).
    color_map : dict
        Mapping {class_label: color} for test points.
    other_color : str
        Color for non-test indices.
    figsize : tuple or None
        Figure size in inches. If None, a size is chosen based on folds.
    pointsize_other : float
        Marker size for non-test indices.
    pointsize_test : float
        Marker size for test indices.
    y_pad : float
        Vertical padding beyond the top/bottom fold rows.

    Returns
    -------
    (fig, ax) : matplotlib Figure and Axes
    """
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y must be a 1D array of class labels.")

    n_folds = len(folds_dicts)
    if n_folds == 0:
        raise ValueError("folds_dicts is empty. Nothing to plot.")

    N = y.size
    all_idx = np.arange(N)

    if figsize is None:
        # Wide enough for indices, tall enough for folds
        # figsize = (max(10, min(18, N / 25)), max(2.5, 0.7 * n_folds))
        figsize = (10, max(2.5, 0.6 * n_folds))

    fig, ax = plt.subplots(figsize=figsize)

    # Plot a light gray baseline row for *all* indices at each fold
    fold_positions = np.arange(n_folds)
    for i in range(n_folds):
        y_row = np.full(N, fold_positions[i], dtype=float)
        ax.scatter(
            all_idx,
            y_row,
            s=pointsize_other,
            c=other_color,
            marker="s",
            edgecolor="none",
            linewidths=0,
            alpha=0.8,
        )

    # Overlay the test indices, colored by class
    for i, fd in enumerate(folds_dicts):
        test_idx = np.asarray(fd["test_idx"], dtype=int)

        # Split test indices by class
        for cls in class_labels:
            cls_mask = y[test_idx] == cls
            idx_cls = test_idx[cls_mask]
            if idx_cls.size == 0:
                continue
            y_row = np.full(idx_cls.size, fold_positions[i], dtype=float)
            ax.scatter(
                idx_cls,
                y_row,
                s=pointsize_test,
                c=color_map[cls],
                marker="s",
                edgecolor="black",
                linewidths=0.4,
                alpha=0.95,
                zorder=3,
            )

    # Cosmetics: labels, ticks, limits, grid
    ax.set_ylim(-y_pad, n_folds - 1 + y_pad)
    ax.set_yticks(fold_positions)
    ax.set_yticklabels([f"Fold {i}" for i in range(n_folds)])
    ax.set_xlabel("Index")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_xlim(-0.5, N - 0.5)

    # Legend
    legend_elems = [
        Line2D([0], [0], marker="s", color="none", label=f"Test: class {class_labels[0]}",
               markerfacecolor=color_map[class_labels[0]], markeredgecolor="black",
               markersize=np.sqrt(pointsize_test)),
        Line2D([0], [0], marker="s", color="none", label=f"Test: class {class_labels[1]}",
               markerfacecolor=color_map[class_labels[1]], markeredgecolor="black",
               markersize=np.sqrt(pointsize_test)),
    ]
    ax.legend(handles=legend_elems, loc="lower right", frameon=True)

    fig.tight_layout()
    return fig, ax

# File to run on
# data_set_IVa_aa: 168 trials # 6 folds
# data_set_IVa_al: 224 trials # 8 folds
# data_set_IVa_av:  84 trials # 4 folds
# data_set_IVa_aw:  56 trials
# data_set_IVa_ay:  28 trials
# Total trials   : 560 trials

path_to_file, kfolds, same_subject = "data_set_IVa_symm_aa.pkl",          4, True
#path_to_file, kfolds, same_subject = "data_set_IVa_symm_al.pkl",          8, True
#path_to_file, kfolds, same_subject = "data_set_IVa_symm_av.pkl",          4, True
#path_to_file, kfolds, same_subject = "data_set_IVa_symm_aa_al_av_aw.pkl", 4, False

config = {
    "path_to_file"   : path_to_file,            # Path
    "kfolds"         : kfolds,                  # Number of folds to use
    "shuffle_input"  : False,                   # Do you want to shuffle the trails in CV or use sequential order?
    "same_subject"   : same_subject,            # Does the data come from a single sub?
    
    "random_state"       : 10,                  # Set random seed
    "spatial_filters"    : {},                  # Collection of spatial filters
    "nfs"                : [1, 2, 4, 8, 16],    # Number of spatial filters to consider
    "spatialFilter_init" : 'random', # 'ones'   # Initialization of the spatial filter matrix ["ones", "random"]
    "dataset_label"      : path_to_file.split('.pkl')[0],
    
    "use_ssa"          : True,                  # Turn SSA pipeline on/off
    "ssa_max_rank"     : None,                  # int if you want to limit rank
    "ssa_n_stationary" : 10,                    # Number of stationary components to use (None = all)
    
    "csp_flag"    : True,                       # Run CSP for input matrices
    "eta_flag"    : False,                      # General scaling
    "ard_flag"    : False,                      # Spatial filter scaling
    "W_trainable" : True,                       # Let W change over training
    "logged_flag" : True,                       # Features in log-space
    "kernel_type" : "RBF",                      # Kernel structure ["Linear", "RBF"]
    "alignment"   : None,                       # Input transformation [None, "euclidean", "riemann"]
}

if config["eta_flag"] == False and config["ard_flag"] == False and config["W_trainable"] == False: config.update({"maxiter" : 10 })
if config["eta_flag"] == False and config["ard_flag"] == False and config["W_trainable"] == True : config.update({"maxiter" : 300})
if config["eta_flag"] == False and config["ard_flag"] == True  and config["W_trainable"] == False: config.update({"maxiter" : 300})
if config["eta_flag"] == False and config["ard_flag"] == True  and config["W_trainable"] == True : config.update({"maxiter" : 300})


# Import file and setup
# Load input data: dict
root_dir = "/mnt/c/Users/scana/Desktop/gpc/data"
root_dir = os.path.join(root_dir, f"{config["path_to_file"]}")
with open(root_dir, "rb") as f:
    data = pickle.load(f)
    
# Generate the indices for the folds
n_samples = len(data["Y"])
if config["same_subject"]:
    # `shuffle` determines if the input trials are consecutive or scrambled
    folds_dicts = generate_kfold_indices(n_samples, n_splits=config["kfolds"], shuffle=config["shuffle_input"]) 
else:
    # This is the case for multiple subjects
    # Make sure the folds identify a single subject as test
    #   folds_dicts is a list --> each element is a dictionary 
    #       each dictionary has two keys --> 'train_idx' and 'test_idx'
    #           the values associated to the keys are np.array containing a sequence of int values
    folds_dicts = []
    
    set_all = set(list(range(data['Y'].shape[0])))
    
    first_idx, last_idx = 0, 0
    
    for key in data['trial_counts_by_file'].keys():
        n         = data['trial_counts_by_file'][key]
        first_idx = last_idx
        last_idx  = n + last_idx
        
        set_test  = set(list(range(first_idx, last_idx, 1)))
        set_train = set_all.difference(set_test)
    
        folds_dicts.append({'train_idx': np.array(list(set_train)),
                            'test_idx' : np.array(list(set_test ))})
    
# Plot class mixtures
fig, ax = plot_fold_class_mix(folds_dicts, data["Y"], annotate=True)
plt.show()
# Plot source mixtures
fig, ax = plot_fold_file_source_mix(
    folds_dicts=folds_dicts,
    groups=data["groups"],
    trial_counts_by_file=data["trial_counts_by_file"],
    annotate=True  # or False if you don't want numbers on bars
)
plt.show()
# Plot test folds 
fig, ax = plot_train_test_density_by_fold(folds_dicts, y=data['Y'])
plt.show()


TOTAL_JOBS  = len(config['nfs']) * len(folds_dicts)
FAILED_JOBS = 0

for nf in config['nfs']:
    config["spatial_filters"][nf] = [] # dictionary for spatial filters with given number of cols
    
    for fold_dict in folds_dicts:
        # Determine input for this run
        train_idx      = fold_dict['train_idx']
        test_idx       = fold_dict['test_idx' ]
        
        if config["use_ssa"]:
            # New SSA pipeline
            X_dict, Y_dict, W_ssa = generate_train_test_from_fold_ssa(
                X_eeg=data["X_eeg"],
                y=data["Y"],
                train_idx=train_idx,
                test_idx=test_idx,
                alignment=config["alignment"],
                ssa_max_rank=config["ssa_max_rank"],
                n_stationary=config.get("ssa_n_stationary", None),
            )
            X_train = X_dict["train"]
            X_test  = X_dict["test"]
            Y_train = Y_dict["train"]
            Y_test  = Y_dict["test"]
        
        else:
            X_dict, Y_dict = generate_train_test_from_fold(data['X'], data['Y'], train_idx, test_idx, alignment=config["alignment"])
            X_train        = X_dict['train'] # data['X'][train_idx]
            X_test         = X_dict['test' ] # data['X'][test_idx ]
            Y_train        = Y_dict['train'] # data['Y'][train_idx]
            Y_test         = Y_dict['test' ] # data['Y'][test_idx ]
        
        # Determine output path
        results_dir  = f"/mnt/c/Users/scana/Desktop/gpc/gpy_results/"
        # Consider file name
        results_dir  = os.path.join(results_dir, f"{config["dataset_label"]}")
        # Consider SSA
        if   config["use_ssa"]: results_dir = os.path.join(results_dir, f"ssa_{config["use_ssa"]}")
        else:                   results_dir = os.path.join(results_dir, f"ssa_{config["use_ssa"]}")
        
        # Consider alignment
        if   config["alignment"] == 'riemann'  : results_dir = os.path.join(results_dir, f"{config["alignment"]}_align")
        elif config["alignment"] == 'euclidean': results_dir = os.path.join(results_dir, f"{config["alignment"]}_align")
        elif config["alignment"] is None       : results_dir = os.path.join(results_dir, f"no_align"         )
        # Consider ARD, kernel type, and whether W is fixed or trainable
        results_dir  = os.path.join(results_dir, f"ard_{config["ard_flag"]}")
        results_dir  = os.path.join(results_dir, f"kernel_{config["kernel_type"].lower()}")
        results_dir  = os.path.join(results_dir, f"spatialFilter")
        if config["W_trainable"]: results_dir += f"_trainable/"
        else:                     results_dir += f"_fixed/"
        # Consider number of spatial filters
        results_dir  = os.path.join(results_dir, f"nf_{nf}")
        
        # Use CSP for a seed for the W matrix
        if config["csp_flag"]:
            # Train and test sets for a given fold have been defined, compute W matrix using CSP
            W_fold = _csp_filters_from_covs(X_train, Y_train, nf=nf, max_rank=None)
            config["spatial_filters"][nf].append(W_fold)
            spatialFilter_init = W_fold
        else:
            spatialFilter_init = config["spatialFilter_init"]
        
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
        
        # Initialize the class and fit
        try:
            
            runner = GPClassificationRunner(
                X                 =my_dict["X"],
                Y                 =my_dict["Y"],
                dataset_label     =config["dataset_label"],
                ch_names          =data["ch_names"],
                ch_xy             =data["ch_location"],
                spatialFilter_init=spatialFilter_init,
                nf                =nf,
                eta_flag          =config["eta_flag"],
                ard_flag          =config["ard_flag"],
                W_trainable       =config["W_trainable"],
                logged_flag       =config["logged_flag"],
                kernel_type       =config["kernel_type"],
                maxiter           =config["maxiter"],
                random_state      =config["random_state"],
                results_dir       =results_dir,
                run_name          =None,
            )
            runner.fit()
        except:
            FAILED_JOBS +=1
            
            
print(f"Failed jobs: {FAILED_JOBS} / {TOTAL_JOBS}")
