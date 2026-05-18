"""
K-fold GP classification sweep for EEG motor-imagery covariance features.

Overview
--------
This is the main experiment script.  For each combination of ``(nf, fold)``
it:

1. Loads a pre-processed dataset pickle (produced by
   ``data_prep_motorimagery_bci_competition_III_merged.py``).
2. Builds train / test splits — either standard k-fold on trials or
   leave-one-subject-out based on ``trial_counts_by_file`` in the pickle.
3. Optionally aligns covariance matrices using Euclidean or Riemannian
   alignment (``align.align_split``).
4. Optionally initialises the spatial filter matrix ``W`` with CSP filters
   computed on the training split.
5. Runs ``GPClassificationRunner.fit()`` and saves per-fold metrics and
   artefacts under a structured results directory.

Results directory layout
------------------------
::

    <results-dir>/
      <dataset-label>/
        shuffle_<bool>/
          <align_tag>/
            ard_<bool>/
              kernel_<type>/
                spatialFilter_trainable/   (W is updated during training)
                spatialFilter_fixed/       (W is frozen at init)
                  nf_<k>/
                    fold_<i>/
                      run_log.json
                      *.png

Typical usage
-------------
::

    python run_class_bci_competition_III_merged_spatial_filters_gpy.py \\
        --data-path ./data/data_set_IVa_symm_aa.pkl \\
        --results-dir ./gpy_results \\
        --kfolds 4 --nfs 1 2 4 8 \\
        --kernel-type RBF --alignment riemann

Multi-stage optimizer example::

    python run_class_bci_competition_III_merged_spatial_filters_gpy.py \\
        --data-path ./data/data_set_IVa_symm_aa.pkl \\
        --results-dir ./gpy_results \\
        --nfs 4 8 --kfolds 4 \\
        --optimizer-stages "lbfgsb:50" "scg:250"

Optimizer stage format: ``"<name>:<max_iters>"`` or
``"<name>:<max_iters>:<lr>:<momentum>"`` for gradient-descent optimizers.
"""

from __future__ import annotations

import os
import pickle
import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.model_selection import KFold

from align import align_split
from gp_classification_gpy import GPClassificationRunner, OptimizerStage
from Whitening import (
    ApplySpatialFilters,
    Covariance,
    SpatialWhiteningDecomposition,
)


# ===========================================================================
# CSP initialisation
# ===========================================================================

def _csp_filters_from_covs(
    X_cov: np.ndarray,
    y: np.ndarray,
    nf: int,
    *,
    max_rank: Optional[int] = None,
) -> np.ndarray:
    """
    Compute ``nf`` CSP spatial filters from per-trial covariance matrices.

    The filters are selected by the magnitude of their generalised
    eigenvalues (those farthest from 0.5 are the most discriminative
    for two-class problems).

    Parameters
    ----------
    X_cov : np.ndarray, shape (n_trials, n_channels, n_channels)
        Per-trial covariance matrices from the training split.
    y : np.ndarray, shape (n_trials,)
        Binary labels in ``{0, 1}``.
    nf : int
        Number of spatial filters to return.
    max_rank : int, optional
        Imposed rank ceiling on the whitening step.  ``None`` uses the
        estimated full rank.

    Returns
    -------
    np.ndarray, shape (n_channels, nf)
        Selected spatial filter columns.

    Raises
    ------
    ValueError
        If ``nf`` exceeds the number of available filters, or if only one
        class is present in ``y``.
    """
    if nf <= 0:
        raise ValueError("nf must be a positive integer.")

    y = np.asarray(y).ravel().astype(int)
    if not {0, 1}.issuperset(set(np.unique(y))):
        raise ValueError("y must contain only labels in {0, 1}.")

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    if idx0.size == 0 or idx1.size == 0:
        raise ValueError("Both classes must be present in the training split.")

    def _tr_norm(C):
        t = np.trace(C)
        return C / (t if t > 1e-12 else 1e-12)

    # Trace-normalise and average within each class
    C0   = np.mean([_tr_norm(C) for C in X_cov[idx0]], axis=0)
    C1   = np.mean([_tr_norm(C) for C in X_cov[idx1]], axis=0)
    Csum = 0.5 * (C0 + C1)                 # pooled covariance (symmetric by construction)
    C1   = 0.5 * (C1 + C1.T)              # symmetrise class-1 covariance

    # Whiten w.r.t. the pooled covariance, then Rayleigh-quotient on C1
    d = SpatialWhiteningDecomposition(sensorCovariance=Csum, maxRank=max_rank)
    d.Rayleigh(C1)

    W_all = d.W                    # (n_channels, n_sources)
    evals = np.asarray(d.eigenvalues)

    if nf > W_all.shape[1]:
        raise ValueError(
            f"Requested nf={nf} but only {W_all.shape[1]} filters are available."
        )

    # Score each filter by how far its eigenvalue is from 0.5 (most
    # discriminative ↔ farthest from the centre → lowest score)
    scores     = -np.abs(evals - 0.5)
    sorted_idx = np.argsort(scores)
    W_sel      = W_all[:, sorted_idx[:nf]].copy()
    return W_sel


# ===========================================================================
# Fold construction
# ===========================================================================

def generate_kfold_indices(
    n_samples: int,
    n_splits: int = 8,
    shuffle: bool = True,
    random_state: Optional[int] = None,
) -> List[Dict[str, np.ndarray]]:
    """
    Create stratified k-fold train/test index dictionaries.

    Parameters
    ----------
    n_samples : int
        Total number of samples.
    n_splits : int
        Number of folds.
    shuffle : bool
        Whether to shuffle samples before splitting.
    random_state : int, optional
        Seed for reproducible shuffling.

    Returns
    -------
    list of dict
        Each element has keys ``"train_idx"`` and ``"test_idx"``, both
        ``np.ndarray`` of integer indices.
    """
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    return [
        {"train_idx": tr, "test_idx": te}
        for tr, te in kf.split(np.arange(n_samples))
    ]


# ===========================================================================
# Train / test split helpers
# ===========================================================================

def generate_train_test_from_fold(
    X_cov: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    alignment: Optional[str] = None,
    frac_inner_val: float = 0.15,
    random_state: Optional[int] = None,
):
    """
    Slice a fold from the full dataset, optionally align covariances, and
    carve a held-out inner validation set from the training fold.

    Alignment procedure (when ``alignment`` is not ``None``):

    1. The reference covariance ``M`` is estimated from the **full training
       fold** (all ``train_idx`` trials) using the chosen method
       (``"euclidean"`` or ``"riemann"``).  This maximises the statistical
       stability of ``M`` by using as many training samples as possible.
    2. The whitening transform ``W = M^{-1/2}`` is applied to every covariance
       in both the training fold and the test fold.
    3. The aligned training fold is then split into an inner training set and a
       held-out inner validation set.  Because the split happens *after*
       alignment, the validation set is automatically aligned with the same
       reference ``M`` — no separate alignment step is required and no
       information from the validation or test trials ever enters the
       computation of ``M``.

    When ``alignment`` is ``None`` (no alignment), steps 1–2 are skipped and
    the raw covariances are split directly.

    Parameters
    ----------
    X_cov : np.ndarray, shape (N, s, s)
        Full set of covariance matrices.
    y : np.ndarray, shape (N,)
        Labels for all trials.
    train_idx, test_idx : np.ndarray
        Integer index arrays for this fold.
    alignment : str or None
        Covariance alignment method: ``"none"`` / ``None`` skips alignment,
        ``"euclidean"`` uses the arithmetic mean, ``"riemann"`` uses the
        Riemannian (geometric) mean.  The reference is always estimated from
        the training fold only.
    frac_inner_val : float
        Fraction of the (aligned) training fold to reserve as held-out inner
        validation.  Default is ``0.15`` (15 %).  Set to ``0.0`` to disable
        the inner validation split (no ``"val"`` key will be returned).
    random_state : int or None
        Seed for the stratified train/val split.

    Returns
    -------
    X_dict : dict
        Keys: ``"train"``, ``"test"``, and (when ``frac_inner_val > 0``)
        ``"val"``.  Each value is an array of shape ``(n, s, s)``.
    Y_dict : dict
        Same key structure as ``X_dict``; values are integer label arrays.

    Notes
    -----
    The inner validation set is never used to estimate the alignment
    reference ``M``, to compute CSP filters, or to update GP parameters.
    It is used exclusively as the model-selection and early-stopping signal,
    providing a criterion that is independent of both the training objective
    and the held-out test fold.
    """
    from sklearn.model_selection import train_test_split as _tts

    Xtr = X_cov[train_idx]
    Xte = X_cov[test_idx]
    ytr = y[train_idx].astype(int)
    yte = y[test_idx].astype(int)

    # --- Step 1–2: alignment (reference from full training fold) ------------ #
    if alignment is not None:
        # align_split computes M from Xtr and applies W = M^{-1/2} to both.
        # Xtr is the full training fold here, so M is maximally stable.
        Xtr, Xte, _ = align_split(Xtr, Xte, method=alignment)
    # After this point Xtr is aligned. Xte is aligned with the same M.

    # --- Step 3: carve inner validation from the (already aligned) training -- #
    if frac_inner_val > 0.0:
        Xtr_inner, Xval, ytr_inner, yval = _tts(
            Xtr, ytr,
            test_size    = frac_inner_val,
            random_state = random_state,
            stratify     = ytr,
        )
        return (
            {"train": Xtr_inner, "val": Xval,  "test": Xte},
            {"train": ytr_inner, "val": yval,   "test": yte},
        )

    return (
        {"train": Xtr, "test": Xte},
        {"train": ytr, "test": yte},
    )

# ===========================================================================
# Diagnostic fold plots
# ===========================================================================

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
    Visualise class composition for each fold as a stacked horizontal bar.

    Parameters
    ----------
    folds_dicts : list of dict
        Each element has ``"train_idx"`` and ``"test_idx"`` arrays.
    y : array-like, shape (N,)
        Class labels.
    class_labels : tuple
        The two class labels to display (default ``(0, 1)``).
    colors_train, colors_test : dict
        ``{label: color}`` for train and test segments.
    figsize : tuple or None
    bar_height : float
    annotate : bool
        If ``True``, write proportions inside each segment.

    Returns
    -------
    (fig, ax)
    """
    y       = np.asarray(y)
    n_folds = len(folds_dicts)
    if n_folds == 0:
        raise ValueError("folds_dicts is empty.")
    if figsize is None:
        figsize = (10, max(2.5, 0.6 * n_folds))

    fig, ax     = plt.subplots(figsize=figsize)
    y_positions = np.arange(n_folds)
    plt.suptitle("Class mixture")

    for i, fd in enumerate(folds_dicts):
        train_idx = np.asarray(fd["train_idx"], dtype=int)
        test_idx  = np.asarray(fd["test_idx"],  dtype=int)
        total     = train_idx.size + test_idx.size
        if total == 0:
            continue

        counts = {
            ("train", class_labels[0]): np.sum(y[train_idx] == class_labels[0]),
            ("train", class_labels[1]): np.sum(y[train_idx] == class_labels[1]),
            ("test",  class_labels[0]): np.sum(y[test_idx]  == class_labels[0]),
            ("test",  class_labels[1]): np.sum(y[test_idx]  == class_labels[1]),
        }
        props = {k: v / total for k, v in counts.items()}

        segments = [
            ("train", class_labels[0], colors_train[class_labels[0]]),
            ("train", class_labels[1], colors_train[class_labels[1]]),
            ("test",  class_labels[0], colors_test[class_labels[0]]),
            ("test",  class_labels[1], colors_test[class_labels[1]]),
        ]
        left = 0.0
        for split, cls, color in segments:
            w = props[(split, cls)]
            if w > 0:
                ax.barh(y_positions[i], w, left=left, height=bar_height,
                        color=color, edgecolor="black", linewidth=0.5)
                if annotate and w >= 0.04:
                    ax.text(left + w / 2, y_positions[i], f"{w:.2f}",
                            va="center", ha="center", fontsize=8, color="white")
                left += w

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.8, n_folds - 1 + 0.8)
    ax.set_xlabel("Proportion (train + test = 1)")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"Fold {i}" for i in range(n_folds)])
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(handles=[
        Patch(facecolor=colors_train[class_labels[0]], edgecolor="black", label="Train 0"),
        Patch(facecolor=colors_train[class_labels[1]], edgecolor="black", label="Train 1"),
        Patch(facecolor=colors_test[class_labels[0]],  edgecolor="black", label="Test 0"),
        Patch(facecolor=colors_test[class_labels[1]],  edgecolor="black", label="Test 1"),
    ], loc="lower left", frameon=True, ncols=4)
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
    cmap_name="Set2",
    source_order="dict",
    annotate_min_width=0.04,
    annotate_fontsize=12,
):
    """
    Visualise per-fold data origin as stacked horizontal bars.

    Each segment represents the proportion of trials from a given source
    (subject / file) in the train or test split of that fold.

    Parameters
    ----------
    folds_dicts : list of dict
    groups : array-like, shape (N,)
        Source label for each trial (must be keys in ``trial_counts_by_file``).
    trial_counts_by_file : dict
        ``{source_name: n_trials}`` mapping.
    figsize, bar_height, annotate : see ``plot_fold_class_mix``
    alpha_train, alpha_test : float
        Opacity for train and test segments respectively.
    cmap_name : str
        Matplotlib colormap name for source colours.
    source_order : ``"dict"`` or ``"alpha"``
        Ordering of sources in the legend and stacking.
    annotate_min_width : float
        Minimum segment proportion needed to show a label.
    annotate_fontsize : int

    Returns
    -------
    (fig, ax)
    """
    groups  = np.asarray(groups)
    sources = (
        list(OrderedDict(trial_counts_by_file).keys())
        if source_order == "dict"
        else sorted(trial_counts_by_file.keys())
    )
    unknown = set(np.unique(groups)) - set(sources)
    if unknown:
        raise ValueError(f"groups contains unknown sources: {sorted(unknown)}")

    n_folds = len(folds_dicts)
    if n_folds == 0:
        raise ValueError("folds_dicts is empty.")

    cmap   = plt.get_cmap(cmap_name)
    colors = {src: cmap(i / max(1, len(sources) - 1)) for i, src in enumerate(sources)}

    if figsize is None:
        figsize = (12, max(2.5, 0.6 * n_folds))

    fig, ax     = plt.subplots(figsize=figsize)
    y_positions = np.arange(n_folds)
    plt.suptitle("File source mixture")

    for i, fd in enumerate(folds_dicts):
        train_idx = np.asarray(fd["train_idx"], dtype=int)
        test_idx  = np.asarray(fd["test_idx"],  dtype=int)
        total     = train_idx.size + test_idx.size
        if total == 0:
            continue

        props_train = {src: np.sum(groups[train_idx] == src) / total for src in sources}
        props_test  = {src: np.sum(groups[test_idx]  == src) / total for src in sources}

        left = 0.0
        for src in sources:
            w = props_train[src]
            if w > 0:
                ax.barh(y_positions[i], w, left=left, height=bar_height,
                        color=colors[src], alpha=alpha_train, edgecolor="black", linewidth=0.5)
                if annotate and w >= annotate_min_width:
                    ax.text(left + w / 2, y_positions[i], f"{w:.2f}",
                            va="center", ha="center", fontsize=annotate_fontsize, color="black")
                left += w

        for src in sources:
            w = props_test[src]
            if w > 0:
                ax.barh(y_positions[i], w, left=left, height=bar_height,
                        color=colors[src], alpha=alpha_test, edgecolor="black", linewidth=0.5)
                if annotate and w >= annotate_min_width:
                    ax.text(left + w / 2, y_positions[i], f"(*{w:.2f})",
                            va="center", ha="center", fontsize=annotate_fontsize, color="black")
                left += w

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.8, n_folds - 1 + 0.8)
    ax.set_xlabel("Proportion (train + test = 1)")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"Fold {i}" for i in range(n_folds)])
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    source_patches = [Patch(facecolor=colors[s], edgecolor="black", label=s) for s in sources]
    key_train = Patch(facecolor="gray", edgecolor="black", alpha=alpha_train, label="Train")
    key_test  = Patch(facecolor="gray", edgecolor="black", alpha=alpha_test,  label="Test (*)")
    leg1 = ax.legend(handles=source_patches, title="Sources",
                     bbox_to_anchor=(1.02, 1.0), loc="upper left")
    ax.add_artist(leg1)
    ax.legend(handles=[key_train, key_test], title="Segment type",
              bbox_to_anchor=(1.02, 0.35), loc="upper left")
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
    Plot the test-set index positions for each fold as a scatter diagram.

    All trial indices are shown as small grey squares; test indices are
    overplotted in class colour.

    Parameters
    ----------
    folds_dicts : list of dict
    y : array-like, shape (N,)
    class_labels, color_map : class labels and their display colours
    other_color : str  — colour for non-test background points
    figsize, pointsize_other, pointsize_test, y_pad : layout controls

    Returns
    -------
    (fig, ax)
    """
    y       = np.asarray(y)
    n_folds = len(folds_dicts)
    N       = y.size
    if figsize is None:
        figsize = (10, max(2.5, 0.6 * n_folds))

    fig, ax     = plt.subplots(figsize=figsize)
    fold_pos    = np.arange(n_folds)
    all_idx     = np.arange(N)

    for i in range(n_folds):
        ax.scatter(all_idx, np.full(N, fold_pos[i]), s=pointsize_other,
                   c=other_color, marker="s", edgecolor="none", alpha=0.8)

    for i, fd in enumerate(folds_dicts):
        test_idx = np.asarray(fd["test_idx"], dtype=int)
        for cls in class_labels:
            idx_cls = test_idx[y[test_idx] == cls]
            if idx_cls.size == 0:
                continue
            ax.scatter(idx_cls, np.full(idx_cls.size, fold_pos[i]),
                       s=pointsize_test, c=color_map[cls], marker="s",
                       edgecolor="black", linewidths=0.4, alpha=0.95, zorder=3)

    ax.set_ylim(-y_pad, n_folds - 1 + y_pad)
    ax.set_yticks(fold_pos)
    ax.set_yticklabels([f"Fold {i}" for i in range(n_folds)])
    ax.set_xlabel("Trial index")
    ax.set_xlim(-0.5, N - 0.5)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.legend(handles=[
        Line2D([0], [0], marker="s", color="none",
               markerfacecolor=color_map[class_labels[k]], markeredgecolor="black",
               markersize=np.sqrt(pointsize_test),
               label=f"Test: class {class_labels[k]}")
        for k in range(2)
    ], loc="lower right", frameon=True)
    fig.tight_layout()
    return fig, ax


# ===========================================================================
# Argument parsing
# ===========================================================================

def _parse_int_list(values: List[str]) -> List[int]:
    return [int(v) for v in values]


def _parse_optimizer_stages(stage_strings: List[str]) -> List[OptimizerStage]:
    """
    Parse optimizer stage strings into ``OptimizerStage`` objects.

    Format:
        ``"<n>:<max_iters>"``
        ``"<n>:<max_iters>:<log_every>"``
        ``"<n>:<max_iters>:<log_every>:<step_rate>"``
        ``"<n>:<max_iters>:<log_every>:<step_rate>:<momentum>"``

    ``log_every`` controls how many optimizer steps are batched into one
    ``model.optimize()`` call (i.e. per EP re-initialisation and per log
    entry).  Larger values are faster; smaller values give finer-grained
    early-stopping checks.  Defaults to 10 when omitted.

    Examples::

        "scg:300"                  -> SCG, 300 steps, log every 10
        "scg:300:10"               -> same, explicit
        "lbfgsb:50:5"              -> L-BFGS-B, 50 steps, log every 5
        "adadelta:400:20:0.01:0.9" -> Adadelta, 400 steps, log every 20
    """
    stages = []
    for s in stage_strings:
        parts = s.split(":")
        if len(parts) < 2:
            raise ValueError(
                f"Invalid optimizer stage '{s}'. "
                "Expected format: 'name:max_iters[:log_every[:step_rate[:momentum]]]'."
            )
        name      = parts[0]
        max_iters = int(parts[1])
        log_every = int(parts[2]) if len(parts) >= 3 else 10
        kwargs: dict = {}
        if len(parts) >= 4:
            kwargs["step_rate"] = float(parts[3])
        if len(parts) >= 5:
            kwargs["momentum"] = float(parts[4])
        stages.append(OptimizerStage(
            optimizer = name,
            max_iters = max_iters,
            log_every = log_every,
            kwargs    = kwargs,
        ))
    return stages


def build_argparser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the experiment runner."""
    parser = argparse.ArgumentParser(
        description="K-fold GP-classification sweep (GPy backend).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- I/O ----
    parser.add_argument("--data-path",   required=True, help="Path to the dataset .pkl file.")
    parser.add_argument("--results-dir", required=True, help="Base output folder.")
    parser.add_argument("--dataset-label", default=None,
                        help="Label for output folder names (defaults to pickle or filename stem).")

    # ---- Cross-validation ----
    parser.add_argument("--kfolds", type=int, default=4, help="Number of k-fold splits.")
    parser.add_argument("--shuffle-input", action="store_true",
                        help="Shuffle trials before KFold (single-subject mode only).")
    parser.add_argument("--multi-subject", action="store_true",
                        help="Leave-one-subject-out folding using 'trial_counts_by_file' in pickle.")
    parser.add_argument("--fold-seed", type=int, default=None,
                        help="Random seed for KFold shuffling.")

    # ---- Sweep ----
    parser.add_argument("--nfs", nargs="+", default=[1, 2],
                        help="Numbers of spatial filters to test (space-separated).")
    parser.add_argument("--spatialFilter-init", default="random",
                        help="W initialisation policy when --csp is not set: 'random' or 'ones'.")
    parser.add_argument("--random-state", type=int, default=10,
                        help="Training random seed (passed to GPClassificationRunner).")

    # ---- Feature extraction ----
    parser.add_argument("--csp", action="store_true",
                        help="Initialise W with CSP filters computed on each training fold.")
    parser.add_argument("--alignment", default="none",
                        choices=["none", "euclidean", "riemann"],
                        help="Covariance alignment method.")

    # ---- Model ----
    parser.add_argument("--kernel-type", default="RBF", choices=["Linear", "RBF"],
                        help="GP kernel type.")
    parser.add_argument("--eta", action="store_true",
                        help="Enable the global output-scale parameter eta.")
    parser.add_argument("--ard", action="store_true",
                        help="Enable per-filter ARD scaling.")
    parser.add_argument("--no-w-trainable", action="store_true",
                        help="Fix W at its initial value (no gradient updates).")
    parser.add_argument("--no-log", action="store_true",
                        help="Use raw variance features instead of log-variance.")

    # ---- Optimisation ----
    parser.add_argument(
        "--optimizer-stages",
        nargs="+",
        default=None,
        help=(
            "Multi-stage optimizer schedule. "
            "Format: 'name:max_iters[:log_every[:step_rate[:momentum]]]'. "
            "'log_every' is the number of optimizer steps per block (default 10); "
            "larger values are faster because EP re-initialises only once per block. "
            "Example: --optimizer-stages 'lbfgsb:50:5' 'scg:300:10'. "
            "If omitted, a single SCG stage of --maxiter steps is used."
        ),
    )
    parser.add_argument("--maxiter", type=int, default=None,
                        help="Total SCG steps for the default single-stage schedule. "
                             "Ignored when --optimizer-stages is provided.")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help=(
            "Number of parallel fold workers. Use 1 for serial execution. "
            "Values above 1 use process-based parallelism via joblib/loky."
        ),
    )
    parser.add_argument(
        "--blas-threads",
        type=int,
        default=None,
        help=(
            "Limit BLAS/OpenMP threads inside each worker. Useful with "
            "--n-jobs > 1 to avoid CPU oversubscription. Example: 1."
        ),
    )

    # ---- Inner validation split ----
    parser.add_argument(
        "--inner-val-frac",
        type    = float,
        default = 0.15,
        help    = (
            "Fraction of each training fold to hold out as an inner validation "
            "set used for model selection and early stopping (default: 0.15). "
            "Alignment is estimated on the full training fold before this split "
            "is made, so the validation set is automatically aligned with the "
            "same reference.  Set to 0 to disable the inner validation split "
            "and fall back to NLML-based early stopping."
        ),
    )

    # ---- Early stopping ----
    parser.add_argument("--es-patience", type=int, default=0,
                        help="Early-stopping patience (0 = disabled).")
    parser.add_argument("--es-min-delta", type=float, default=1e-4,
                        help="Minimum metric improvement to reset the patience counter.")

    # ---- Diagnostics ----
    parser.add_argument("--plots", action="store_true",
                        help="Show diagnostic fold plots before training.")

    return parser


def _infer_default_maxiter(
    *,
    eta_flag: bool,
    ard_flag: bool,
    w_trainable: bool,
) -> int:
    """
    Choose a reasonable default iteration count when none is specified.

    When only the GP hyperparameters need updating (no spatial filter
    learning) convergence is fast and 10 steps suffice.  Adding trainable
    parameters requires more iterations.
    """
    if not eta_flag and not ard_flag and not w_trainable:
        return 10
    return 300



# ===========================================================================
# Sweep execution helpers
# ===========================================================================

def _build_run_dir(
    *,
    results_base: Path,
    dataset_label: str,
    shuffle_input: bool,
    alignment: Optional[str],
    ard_flag: bool,
    kernel_type: str,
    w_trainable: bool,
    nf: int,
    fold_i: int,
) -> Path:
    """Return the output directory for one ``(nf, fold)`` run."""
    align_tag = f"{alignment}_align" if alignment else "no_align"
    w_tag = "spatialFilter_trainable" if w_trainable else "spatialFilter_fixed"
    return (
        results_base
        / dataset_label
        / f"shuffle_{bool(shuffle_input)}"
        / align_tag
        / f"ard_{bool(ard_flag)}"
        / f"kernel_{kernel_type.lower()}"
        / w_tag
        / f"nf_{nf}"
        / f"fold_{fold_i}"
    )


def _run_fold_sweep(
    *,
    fold_i: int,
    fold_dict: Dict[str, np.ndarray],
    nfs: List[int],
    data: dict,
    args: argparse.Namespace,
    dataset_label: str,
    ch_names: Optional[List[str]],
    ch_xy: Optional[dict],
    alignment: Optional[str],
    optimizer_stages: List[OptimizerStage],
    results_base: Path,
) -> int:
    """
    Run all ``nf`` settings for one fold and return the number of failures.

    Keeping all ``nf`` values for a fold in the same worker avoids recomputing
    the train/test split, optional alignment, and full CSP decomposition for
    every spatial-filter count.
    """
    # Make child-process plotting non-interactive.  This is safer on clusters
    # and when several workers save figures at the same time.
    if int(getattr(args, "n_jobs", 1)) != 1:
        os.environ.setdefault("MPLBACKEND", "Agg")

    train_idx = fold_dict["train_idx"]
    test_idx = fold_dict["test_idx"]

    X_dict, Y_dict = generate_train_test_from_fold(
        data["X"],
        data["Y"],
        train_idx,
        test_idx,
        alignment=alignment,
        frac_inner_val=float(args.inner_val_frac),
        random_state=int(args.random_state),
    )

    X_train = X_dict["train"]
    Y_train = Y_dict["train"]

    csp_filters_by_nf: Dict[int, np.ndarray] = {}
    if args.csp:
        max_nf = max(nfs)
        W_csp_full = _csp_filters_from_covs(X_train, Y_train, nf=max_nf)
        csp_filters_by_nf = {
            int(nf): W_csp_full[:, :int(nf)].copy()
            for nf in nfs
        }

    failed_jobs = 0
    for nf in nfs:
        nf = int(nf)
        run_dir = _build_run_dir(
            results_base=results_base,
            dataset_label=dataset_label,
            shuffle_input=bool(args.shuffle_input),
            alignment=alignment,
            ard_flag=bool(args.ard),
            kernel_type=args.kernel_type,
            w_trainable=not bool(args.no_w_trainable),
            nf=nf,
            fold_i=fold_i,
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        if args.csp:
            spatialFilter_init = csp_filters_by_nf[nf]
        else:
            spatialFilter_init = args.spatialFilter_init

        try:
            runner = GPClassificationRunner(
                X=X_dict,
                Y=Y_dict,
                dataset_label=dataset_label,
                ch_names=ch_names,
                ch_xy=ch_xy,
                spatialFilter_init=spatialFilter_init,
                nf=nf,
                eta_flag=bool(args.eta),
                ard_flag=bool(args.ard),
                W_trainable=not bool(args.no_w_trainable),
                logged_flag=not bool(args.no_log),
                kernel_type=args.kernel_type,
                optimizer_stages=optimizer_stages,
                es_patience=int(args.es_patience),
                es_min_delta=float(args.es_min_delta),
                random_state=int(args.random_state),
                results_dir=str(run_dir),
                run_name=None,
            )
            runner.fit()
        except Exception as exc:
            failed_jobs += 1
            print(f"[WARN] Failed — nf={nf}, fold={fold_i}: {exc}", flush=True)

    return failed_jobs


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = build_argparser()
    args   = parser.parse_args()

    nfs = _parse_int_list(args.nfs)

    data_path    = Path(args.data_path).expanduser().resolve()
    results_base = Path(args.results_dir).expanduser().resolve()
    results_base.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    with data_path.open("rb") as fh:
        data = pickle.load(fh)

    dataset_label = args.dataset_label or data.get("dataset_label", data_path.stem)
    ch_names      = data.get("ch_names", None)
    ch_xy         = data.get("ch_location", None)
    alignment     = None if args.alignment == "none" else args.alignment

    # ---- Build fold structure ----
    n_samples = int(len(data["Y"]))
    if not bool(args.multi_subject):
        folds_dicts = generate_kfold_indices(
            n_samples,
            n_splits   = int(args.kfolds),
            shuffle    = bool(args.shuffle_input),
            random_state=args.fold_seed,
        )
    else:
        if "trial_counts_by_file" not in data:
            raise KeyError(
                "Multi-subject folding requires 'trial_counts_by_file' in the dataset pickle. "
                "Run without --multi-subject for single-subject k-fold."
            )
        folds_dicts = []
        set_all = set(range(n_samples))
        first_idx = last_idx = 0
        for key, n in data["trial_counts_by_file"].items():
            first_idx = last_idx
            last_idx  = last_idx + int(n)
            set_test  = set(range(first_idx, last_idx))
            folds_dicts.append({
                "train_idx": np.array(sorted(set_all - set_test), dtype=int),
                "test_idx" : np.array(sorted(set_test), dtype=int),
            })

    if args.plots:
        fig, _ = plot_fold_class_mix(folds_dicts, data["Y"], annotate=True)
        plt.show()
        if "groups" in data and "trial_counts_by_file" in data:
            fig, _ = plot_fold_file_source_mix(
                folds_dicts=folds_dicts,
                groups=data["groups"],
                trial_counts_by_file=data["trial_counts_by_file"],
                annotate=True,
            )
            plt.show()
        fig, _ = plot_train_test_density_by_fold(folds_dicts, y=data["Y"])
        plt.show()

    # ---- Build optimizer schedule ----
    if args.optimizer_stages is not None:
        optimizer_stages = _parse_optimizer_stages(args.optimizer_stages)
    else:
        maxiter = args.maxiter
        if maxiter is None:
            maxiter = _infer_default_maxiter(
                eta_flag    = bool(args.eta),
                ard_flag    = bool(args.ard),
                w_trainable = not bool(args.no_w_trainable),
            )
        optimizer_stages = [OptimizerStage(optimizer="scg", max_iters=maxiter)]

    # ---- Run sweep ----
    total_jobs = len(nfs) * len(folds_dicts)
    n_jobs = max(1, int(args.n_jobs))

    if n_jobs == 1:
        failed_jobs = sum(
            _run_fold_sweep(
                fold_i=fold_i,
                fold_dict=fold_dict,
                nfs=nfs,
                data=data,
                args=args,
                dataset_label=dataset_label,
                ch_names=ch_names,
                ch_xy=ch_xy,
                alignment=alignment,
                optimizer_stages=optimizer_stages,
                results_base=results_base,
            )
            for fold_i, fold_dict in enumerate(folds_dicts)
        )
    else:
        os.environ.setdefault("MPLBACKEND", "Agg")
        if args.blas_threads is not None:
            blas_threads = str(max(1, int(args.blas_threads)))
            for env_name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                os.environ.setdefault(env_name, blas_threads)

        try:
            from joblib import Parallel, delayed
        except ImportError as exc:
            raise ImportError(
                "--n-jobs > 1 requires joblib. Install it with `pip install joblib` "
                "or rerun with --n-jobs 1."
            ) from exc

        print(
            f"Running {total_jobs} jobs as {len(folds_dicts)} fold tasks "
            f"with n_jobs={n_jobs}.",
            flush=True,
        )
        failures_by_fold = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_run_fold_sweep)(
                fold_i=fold_i,
                fold_dict=fold_dict,
                nfs=nfs,
                data=data,
                args=args,
                dataset_label=dataset_label,
                ch_names=ch_names,
                ch_xy=ch_xy,
                alignment=alignment,
                optimizer_stages=optimizer_stages,
                results_base=results_base,
            )
            for fold_i, fold_dict in enumerate(folds_dicts)
        )
        failed_jobs = int(sum(failures_by_fold))

    print(f"\nDone.  Failed jobs: {failed_jobs} / {total_jobs}")


if __name__ == "__main__":
    main()
