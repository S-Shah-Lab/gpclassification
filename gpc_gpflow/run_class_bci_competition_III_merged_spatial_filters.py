from __future__ import annotations

import os, pickle
import gpflow
from   typing      import Dict, List, Tuple, Sequence
from   collections import OrderedDict

import numpy                   as np
import matplotlib.pyplot       as plt
from   matplotlib.patches      import Patch
from   matplotlib.lines        import Line2D
from   sklearn.model_selection import KFold
from   gp_classification       import GPClassificationRunner
from   Whitening               import SpatialWhiteningDecomposition  # SVD + Whitening Rayleigh CSP
from align                     import align_split 

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
    
    half = nf // 2
    if nf % 2 == 0:
        # pick symmetric extremes
        sel = np.r_[0:half, W_all.shape[1] - half: W_all.shape[1]]
        return W_all[:, sel].copy()
    else:
        # pick symmetric extremes
        # also pick the most informative out of the two next in line
        if -abs(d.eigenvalues[half] - 0.5) < -abs(d.eigenvalues[-half-1] - 0.5):
            sel = np.r_[0:half+1, W_all.shape[1] - half: W_all.shape[1]]
        else:
            sel = np.r_[0:half, W_all.shape[1] - half - 1: W_all.shape[1]]
        return W_all[:, sel].copy()

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

def make_fold_inputs(X_cov, y, train_idx, test_idx, align="riemann"):
    """
    X_cov: (N, s, s) original covariances (symmetrized; maybe trace-normalized upstream)
    y    : (N,)
    Returns dicts X,Y with per-split aligned covariances ready for GPClassificationRunner
    """
    Xtr_raw = X_cov[train_idx]
    Xte_raw = X_cov[test_idx]
    
    if align is not None:
        Xtr_al, Xte_al, M = align_split(Xtr_raw, Xte_raw, method=align)
    else:
        Xtr_al = Xtr_raw
        Xte_al = Xte_raw
        M      = None

    # hand aligned 3D arrays to the runner; it will flatten to (N, s*s) itself
    X_dict = {"train": Xtr_al, "test": Xte_al}
    Y_dict = {"train": y[train_idx].astype(int), "test": y[test_idx].astype(int)}
    return X_dict, Y_dict, M

def plot_fold_class_mix(
    folds_dicts,
    y,
    class_labels=(0, 1),
    colors_train={0: "cornflowerblue", 1: "gold"},
    colors_test={0: "blue", 1: "orange"},
    figsize=None,
    bar_height=0.6,
    annotate=False,
):
    """
    Visualize class composition for each fold as a stacked horizontal bar.

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
            ("test",  class_labels[0], colors_test[class_labels[0]]),
            ("test",  class_labels[1], colors_test[class_labels[1]]),
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

def plot_fold_source_mix(
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
    plt.suptitle('Source mixture')
    
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

def plot_test_indices_by_fold(
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
# data_set_IVa_aa: 168 # 6 folds
# data_set_IVa_al: 224 # 8 folds
# data_set_IVa_av:  84 # 4 folds
# data_set_IVa_aw:  56
# data_set_IVa_ay:  28
# Total trials   : 560

path_to_file, kfolds, same_subject = "data_set_IVa_symm_aa.pkl",          4, True
#path_to_file, kfolds, same_subject = "data_set_IVa_symm_al.pkl",          8, True
#path_to_file, kfolds, same_subject = "data_set_IVa_symm_av.pkl",          4, True
#path_to_file, kfolds, same_subject = "data_set_IVa_symm_aa_al_av_aw.pkl", 4, False

#path_to_file, kfolds, same_subject = "data_set_IVa_symm_shrink002_aa.pkl", 6, True
#path_to_file, kfolds, same_subject = "data_set_IVa_symm_shrink002_al.pkl", 8, True
#path_to_file, kfolds, same_subject = "data_set_IVa_symm_shrink002_av.pkl", 4, True

#path_to_file, kfolds, same_subject = "data_set_IVa_symm_shrink003_aa.pkl",          6, True
#path_to_file, kfolds, same_subject = "data_set_IVa_symm_shrink003_al.pkl",          8, True
#path_to_file, kfolds, same_subject = "data_set_IVa_symm_shrink003_av.pkl",          4, True
#path_to_file, kfolds, same_subject = "data_set_IVa_symm_shrink003_aa_al_av_aw.pkl", 4, False

#path_to_file, kfolds, same_subject = "data_set_IVa_symm_unittrace_aa.pkl", 6, True

#path_to_file, kfolds, same_subject = "data_set_IVa_symm_unittrace_shrink003_aa.pkl",          6, True
#path_to_file, kfolds, same_subject = "data_set_IVa_symm_unittrace_shrink003_al.pkl",          8, True
#path_to_file, kfolds, same_subject = "data_set_IVa_symm_unittrace_shrink003_av.pkl",          4, True
#path_to_file, kfolds, same_subject = "data_set_IVa_symm_unittrace_shrink003_aa_al_av_aw.pkl", 4, False

W_trainable = False
alignment = 'euclidean' # None # 'riemann'
same_subject = True

spatialFilter_init = "random" # handles initialization of W and train/test split if input is not dict type
random_state = 11
spatial_filters = {}
nfs = [1, 2, 4, 8, 12, 16, 20] # Number of spatial filters to consider

if W_trainable:
    maxiter = 400
else:
    maxiter = 200
dataset_label = path_to_file.split('.pkl')[0]

# Load input data: dict
with open(f"./data/{path_to_file}", "rb") as f:
    data = pickle.load(f)
    
# Generate the indices for the folds 
n_samples = len(data["Y"])
if same_subject:
    # Make sure the folds are consecutive
    folds_dicts = generate_kfold_indices(n_samples, n_splits=kfolds, shuffle=False) # sequential

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
        n = data['trial_counts_by_file'][key]
        first_idx = last_idx
        last_idx = n + last_idx
        
        set_test = set(list(range(first_idx, last_idx, 1)))
        set_train = set_all.difference(set_test)
    
        folds_dicts.append({'train_idx': np.array(list(set_train)),
                            'test_idx' : np.array(list(set_test ))})
    
kfolds = len(folds_dicts)


# Plot class mixtures
fig, ax = plot_fold_class_mix(folds_dicts, data["Y"], annotate=True)
plt.show()


# Plot source mixtures
fig, ax = plot_fold_source_mix(
    folds_dicts=folds_dicts,
    groups=data["groups"],
    trial_counts_by_file=data["trial_counts_by_file"],
    annotate=True  # or False if you hate numbers on bars
)
plt.show()


# Plot test folds 
fig, ax = plot_test_indices_by_fold(folds_dicts, y=data['Y'])
plt.show()



TOTAL_JOBS = len(nfs) * len(folds_dicts)
FAILED_JOBS = 0

for nf in nfs:
    spatial_filters[nf] = [] # dictionary for spatial filters with given number of cols
    
    for fold_dict in folds_dicts:
        
        train_idx = fold_dict['train_idx']
        test_idx = fold_dict['test_idx']
        
        X_dict, Y_dict, M = make_fold_inputs(data['X'], data['Y'], train_idx, test_idx, align=alignment)
        
        X_train = X_dict['train'] #data['X'][train_idx]
        X_test =  X_dict['test' ] #data['X'][test_idx]
        Y_train = Y_dict['train'] #data['Y'][train_idx]
        Y_test =  Y_dict['test' ] #data['Y'][test_idx]
        
        results_dir  = f"/mnt/c/Users/scana/Desktop/gpc/results/"
        results_dir += f"{dataset_label}/"
         
        if   alignment == 'riemann'  : results_dir += f"{alignment}_align/"
        elif alignment == 'euclidean': results_dir += f"{alignment}_align/"
        elif alignment is None       : results_dir += f"no_align/"
        
        results_dir += f"spatialFilter"
        if W_trainable: results_dir += f"/"
        else:           results_dir += f"_CSP/"
        results_dir += f"nf_{nf}"
        
        if W_trainable == False:
            # Now that train and test sets for a given fold have been defined, compute W matrix using CSP
            W_fold = _csp_filters_from_covs(X_train, Y_train, nf=nf, max_rank=None)
            spatial_filters[nf].append(W_fold)
            spatialFilter_init = W_fold         
        
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

        try:
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
                maxiter=maxiter,
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
        except:
            FAILED_JOBS +=1
            
            
print(f"Failed jobs: {FAILED_JOBS} / {TOTAL_JOBS}")
