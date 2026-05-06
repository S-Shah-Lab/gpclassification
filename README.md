# GP-BCI: Gaussian Process Classification for EEG Motor Imagery

A Python toolkit for decoding motor-imagery EEG signals using **Gaussian Process (GP) classification** with jointly-optimised spatial filters.  The pipeline covers everything from raw covariance-feature extraction through cross-validated training and evaluation, with a rich set of diagnostic plots saved automatically per fold.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Key Features](#key-features)
  - [Cross-Validation and the Three-Way Data Split](#cross-validation-and-the-three-way-data-split)
  - [Shuffling & Random State](#shuffling--random-state)
  - [Covariance Alignment](#covariance-alignment)
  - [Spatial Filters (W) — CSP Init & Trainable/Fixed](#spatial-filters-w--csp-init--trainablefixed)
  - [Kernel Type](#kernel-type)
  - [ARD Scaling](#ard-scaling)
  - [Early Stopping and Model Selection](#early-stopping-and-model-selection)
  - [Multi-Stage Optimisation](#multi-stage-optimisation)
- [Results Directory Layout](#results-directory-layout)
- [Output Files](#output-files)
- [Module Reference](#module-reference)

---

## Overview

The package implements a full end-to-end BCI decoding pipeline:

```
Raw EEG trials
    └─► Per-trial covariance matrices              (data_prep_*.py)
         └─► Optional covariance alignment          (align.py)
              │   (M estimated from outer training pool only;
              │    M^{-1/2} applied to outer training pool + outer test fold)
              └─► Three-way split: inner-train / inner-val / outer-test
                   └─► Spatial filter matrix W
                        │   • CSP init path only: SpatialWhiteningDecomposition (Whitening.py)
                        │   • Default (random/ones): W lives inside CustomKernelGPy (kernels_gpy.py)
                        └─► Log-variance features  w_p^T Σ w_p
                             └─► GP classifier      (gp_classification_gpy.py)
                                  ├─► NLML minimisation on inner-train
                                  ├─► Val NLPD → early stopping & best-model selection
                                  └─► Per-fold metrics & plots saved to disk
```

The spatial filter matrix **W** can be either **fixed** (e.g. pre-computed CSP) or **jointly optimised** together with the GP hyperparameters via gradient-based methods, making it possible to learn task-discriminative projections end-to-end.

Cross-validation is the evaluation framework (default: **4-fold**, configurable with `--kfolds`; leave-one-subject-out also available via `--multi-subject`).  No permanently held-out test set is designated — each fold's withheld portion is that fold's test set.  Within each outer training fold the trials are further split into an **inner training set** (85 %, default) and a **held-out inner validation set** (15 %, default), used exclusively for model selection and early stopping.

The runner script saves one `run_log.json` and a set of diagnostic PNGs **per fold**.  Aggregating metrics across folds (averages, standard deviations, summary plots) is done separately by `plot_metric_bci_competition_III_merged_spatial_filters.py`.

---

## Repository Structure

```
.
├── run_class_bci_competition_III_merged_spatial_filters_gpy.py   # Main experiment script
├── gp_classification_gpy.py      # GP model, training loop, diagnostics
├── kernels_gpy.py                # Custom GPy kernel (RBF / Linear + spatial filter W)
├── Whitening.py                  # SpatialWhiteningDecomposition — used for CSP init only
├── align.py                      # Euclidean / Riemannian covariance alignment
├── SVD.py                        # SVD helper (whitening, sqrtm, pinv, …)
├── data_prep_motorimagery_bci_competition_III_merged.py  # Dataset preparation
└── plot_metric_bci_competition_III_merged_spatial_filters.py  # Cross-fold metric aggregation
```

---

## Installation

```bash
pip install numpy scipy scikit-learn matplotlib GPy
# Optional — required only for topomap plots (08_topomaps.png):
pip install mne
```

Python ≥ 3.8 is required.

---

## Quick Start

### Single-subject k-fold sweep

```bash
python run_class_bci_competition_III_merged_spatial_filters_gpy.py \
    --data-path  ./data/data_set_IVa_symm_aa.pkl \
    --results-dir ./results \
    --kfolds 4 \
    --nfs 1 2 4 8 \
    --kernel-type RBF \
    --alignment riemann \
    --csp \
    --ard \
    --inner-val-frac 0.15 \
    --es-patience 20
```

### Multi-subject (leave-one-subject-out)

```bash
python run_class_bci_competition_III_merged_spatial_filters_gpy.py \
    --data-path ./data/merged.pkl \
    --results-dir ./results \
    --multi-subject \
    --nfs 4 8 \
    --kernel-type RBF \
    --alignment riemann \
    --inner-val-frac 0.15
```

### Multi-stage optimiser

```bash
python run_class_bci_competition_III_merged_spatial_filters_gpy.py \
    --data-path ./data/data_set_IVa_symm_aa.pkl \
    --results-dir ./results \
    --nfs 4 8 --kfolds 4 \
    --optimizer-stages "lbfgsb:50" "scg:250" \
    --inner-val-frac 0.15 \
    --es-patience 30
```

### Disable the inner validation split (legacy behaviour)

```bash
# Falls back to NLML-based early stopping (no held-out validation)
python run_class_bci_competition_III_merged_spatial_filters_gpy.py \
    --data-path ./data/data_set_IVa_symm_aa.pkl \
    --results-dir ./results \
    --nfs 4 --kfolds 4 \
    --inner-val-frac 0
```

---

## Key Features

### Cross-Validation and the Three-Way Data Split

No permanently held-out test set is defined.  All trials enter the cross-validation loop.

> **Important:** The runner writes one `run_log.json` per fold — it does **not** average metrics across folds.  Cross-fold aggregation is handled by the separate `plot_metric_…` script.

#### Outer cross-validation (evaluation)

Two modes are supported:

| Mode | Flag | Description |
|---|---|---|
| **K-Fold** | *(default)* | Splits all trials into `--kfolds` folds (default: 4). One fold is the outer test set; the rest form the outer training pool. |
| **Leave-One-Subject-Out** | `--multi-subject` | Uses `trial_counts_by_file` in the dataset pickle; one subject per test fold. |

```bash
--kfolds 4              # number of outer folds (default: 4; any positive integer is valid)
--multi-subject         # switch to leave-one-subject-out folding
```

#### Inner split: held-out validation for model selection

Within each outer training pool, a stratified fraction is set aside as a **held-out inner validation set**:

```bash
--inner-val-frac 0.15   # fraction of outer training pool for inner validation (default: 0.15)
--inner-val-frac 0      # disable inner validation; fall back to NLML-based stopping
```

The inner validation set is used for exactly two purposes:

1. **Early stopping** — training halts when validation NLPD stops improving (see [Early Stopping](#early-stopping-and-model-selection)).
2. **Best-model selection** — the parameter snapshot with the lowest validation NLPD across the entire training trajectory is restored before test-set evaluation.

The inner validation set **never influences** gradient updates, covariance alignment, or CSP initialisation.

#### Why three-way splitting matters

When the spatial filter **W** is jointly optimised with the GP hyperparameters, the training NLML can decrease monotonically even as the model begins to overfit noise in the spatial filters.  Monitoring a genuinely held-out validation NLPD provides an independent signal that catches this overfitting before the test fold is touched.

For each `(nf, fold)` combination the script writes a self-contained result directory containing `run_log.json` and diagnostic PNGs.

---

### Shuffling & Random State

Two independent random seeds control reproducibility:

```bash
--shuffle-input         # shuffle trials before k-fold splitting (default: off)
--fold-seed 42          # RNG seed used by KFold when --shuffle-input is on
--random-state 10       # seed for W initialisation, inner-val split, and other stochastic steps
```

> **Note — `--shuffle-input` in multi-subject mode:** This flag applies only in **single-subject k-fold mode**.  In `--multi-subject` (LOSO) mode, fold construction is always sequential (one subject per fold) and `--shuffle-input` has no effect on which trials go into which fold.  The flag value still appears in the results path (`shuffle_True/False`) regardless of mode so that shuffled and unshuffled single-subject runs never overwrite each other, but the path token is meaningless for LOSO runs.

The `--random-state` seed is also used for the stratified inner validation split inside each fold, so the exact train/val/test partition is fully reproducible.

---

### Covariance Alignment

Between-session or between-subject differences in the mean covariance structure can be removed before feature extraction.  Three options are available:

```bash
--alignment none        # (default) no alignment — raw covariances used as-is
--alignment euclidean   # arithmetic mean whitening
--alignment riemann     # Riemannian (geometric) mean whitening  ← recommended
```

#### How alignment interacts with the three-way split

The alignment reference matrix **M** is estimated from the **full outer training pool** (all trials assigned to the training fold, before the inner-val split is made).  This maximises the statistical stability of **M** while keeping the test fold completely unseen.

Alignment then operates in exactly **two steps** (not three):

1. `M^{-1/2}` is applied to every covariance in the **outer training pool**.  The pool is then split into inner-train (85 %) and inner-val (15 %).  Both subsets are therefore automatically aligned with the same reference — the inner-val is **not** aligned separately; it simply inherits the transform applied to the pool before the split.
2. The same `M^{-1/2}` is applied to the **outer test fold** covariances in a separate call.

No information from the inner-val set or the test fold ever enters the computation of **M**.

```
outer training pool  (all train_idx trials)
    │
    ├─► estimate M from the pool (Euclidean or Riemannian mean of covariances)
    │
    ├─► apply M^{-1/2} to every covariance in the pool
    │        └─► stratified split → inner-train (85%) / inner-val (15%)
    │            (both subsets are already aligned; no separate step needed)
    │
    └─► apply M^{-1/2} to outer test fold covariances (separate call)
```

The alignment method is encoded in the results path (`no_align`, `euclidean_align`, `riemann_align`).

#### Euclidean vs. Riemannian mean

| Method | Description | When to use |
|---|---|---|
| `none` | No alignment | Single session, no between-session drift |
| `euclidean` | Arithmetic mean of training covariances | Fast; sufficient when sessions are similar |
| `riemann` | Riemannian (geometric) mean via iterative geodesic update | Recommended for multi-session or multi-subject data; affine-invariant |

---

### Spatial Filters (W) — CSP Init & Trainable/Fixed

The spatial filter matrix **W** of shape `(n_channels, nf)` projects multichannel covariances down to `nf` discriminative sources.

#### Initialisation

```bash
--csp                          # initialise W with CSP filters (recommended)
--spatialFilter-init random    # random Gaussian init (default when --csp is not set)
--spatialFilter-init ones      # constant init
--nfs 1 2 4 8                  # sweep over number of filters (default: 1 2)
```

**When `--csp` is set:** CSP filters are computed from the **inner training set** (after alignment has been applied, after the inner-val split) on each fold using `SpatialWhiteningDecomposition` from `Whitening.py`.  Filters are ranked by how far their generalised eigenvalue falls from 0.5 — those farthest from centre are the most class-discriminative and are selected first.

**When `--csp` is not set:** `Whitening.py` is not involved at all.  W is initialised directly inside `GPClassificationRunner` (random Gaussian or all-ones) and is stored and applied within `CustomKernelGPy` (`kernels_gpy.py`) for the entire training run.

#### Trainable vs. Fixed

```bash
# W is updated by gradient descent (default — jointly optimised with GP params)
--no-w-trainable   # freeze W at its initial value; only GP hyperparams are updated
```

The trainability setting is encoded in the results path as `spatialFilter_trainable` or `spatialFilter_fixed`.

---

### Kernel Type

Two kernel families are available for the GP covariance function:

```bash
--kernel-type RBF      # Squared-exponential (Gaussian) kernel  ← default
--kernel-type Linear   # Linear (dot-product) kernel
```

Both kernels are applied in the **projected feature space** defined by the spatial filter W.  The kernel type is encoded in the results path as `kernel_rbf` or `kernel_linear`.

---

### ARD Scaling

Automatic Relevance Determination assigns an independent log-scale to each spatial filter output, allowing the model to down-weight uninformative filters during optimisation:

```bash
--ard       # enable per-filter ARD log-scales
--eta       # enable a global output-scale parameter
```

The ARD setting is reflected in the results path (`ard_True` / `ard_False`).

---

### Early Stopping and Model Selection

Patience-based early stopping monitors the **model-selection metric** and halts training when no improvement exceeding `es_min_delta` is observed for `es_patience` consecutive optimisation steps.

#### Which metric is monitored

| Condition | Metric used |
|---|---|
| Inner validation set present (`--inner-val-frac > 0`) | **Validation NLPD** — independent of the training objective |
| No inner validation set (`--inner-val-frac 0`) | **Training NLML** — the optimisation objective itself |

Using validation NLPD is strongly preferred when W is trainable, because the NLML can decrease monotonically even during overfitting of the spatial filters.

```bash
--es-patience  20       # stop after 20 non-improving steps (default: 0 = disabled)
--es-min-delta 1e-4     # minimum improvement threshold to reset the patience counter (default: 1e-4)
```

The **best checkpoint** — the parameter vector with the lowest monitored metric across the entire training trajectory — is restored automatically before test-set evaluation.  This ensures that the reported metrics always correspond to the best-seen model, never to the final (potentially overfit) iterate.

#### Learning curve plot

The `01_learning_curves.png` diagnostic shows:

- **Black solid line** — training NLML (the quantity being minimised).
- **Grey dashed line** — validation NLPD (when an inner validation set is present).  Watching this line diverge upward from the NLML is the primary visual indicator of overfitting in the spatial filters.
- **Coloured lines (right axis)** — per-split accuracy (train, val, test).
- **Vertical dashed line** — the best-checkpoint iteration.
- **Vertical dotted red line** — the step at which early stopping fired (if it fired).

---

### Multi-Stage Optimisation

The optimiser schedule can be specified as a sequence of named stages, useful for warm-starting with a cheap method before refining with a more expensive one:

```bash
--optimizer-stages "lbfgsb:50" "scg:250"
# format: "<optimizer_name>:<max_iters>[:<log_every>[:<step_rate>[:<momentum>]]]"
```

If `--optimizer-stages` is omitted, a single SCG stage is used.  The default iteration budget depends on which parameters are trainable:

| Trainable parameters | Default `--maxiter` |
|---|---|
| GP hyperparams only (W fixed, no ARD, no eta) | 10 |
| Any of W / ARD / eta enabled | 300 |

---

## Results Directory Layout

Each `(nf, fold)` combination produces one leaf directory.  `spatialFilter_trainable` and `spatialFilter_fixed` are **mutually exclusive** branches — exactly one will appear per run, not both simultaneously:

```
<results-dir>/
  <dataset-label>/
    shuffle_<True|False>/
      <no_align|euclidean_align|riemann_align>/
        ard_<True|False>/
          kernel_<rbf|linear>/
            spatialFilter_trainable/    ─┐ mutually exclusive;
            spatialFilter_fixed/        ─┘ only one per run
              nf_<k>/
                fold_<i>/
                  run_log.json
                  01_learning_curves.png
                  02_threshold_sweep.png
                  03_calibration_curve.png
                  05_kernel_parameters.png    ← only when --eta or --ard active
                  06_kernel_W.png
                  08_topomaps.png             ← only when MNE installed + ch_xy present
                  09_confusion_matrix.png
                  10_features_and_boundary_(0, 1).png   ← only when nf == 2
                  16_singular_values.png
```

---

## Output Files

| File | Contents | Produced when |
|---|---|---|
| `run_log.json` | Per-iteration metrics (NLML, val NLPD, acc, Brier, AUC-ROC, AUC-PR), kernel parameter snapshots, and best-iteration predictions for all splits | Always |
| `01_learning_curves.png` | NLML (black) + val NLPD (grey dashed) + per-split accuracy over optimisation steps; best-iter and early-stopping markers | Always |
| `02_threshold_sweep.png` | ROC curve, PR curve, and six metrics (accuracy, precision, recall, F1, specificity, Youden's J) swept across probability thresholds | Always |
| `03_calibration_curve.png` | Reliability diagrams (predicted probability vs. empirical probability) for each available split | Always |
| `05_kernel_parameters.png` | η (global output scale) and per-filter ARD scale trajectories over optimisation steps | Only when `--eta` or `--ard` is active |
| `06_kernel_W.png` | Spatial filter weight trajectories over iterations; one subplot per filter column; median shown in bold | Always |
| `08_topomaps.png` | EEG scalp topomaps of the spatial filter columns at the best iteration | Only when **MNE is installed** (`pip install mne`) **and** `ch_xy` electrode coordinates are present in the dataset pickle |
| `09_confusion_matrix.png` | Confusion matrices for each available split at the best iteration | Always |
| `10_features_and_boundary_(0, 1).png` | 2D scatter of filter features (train + test) with interpolated GP decision boundary | Only when `nf == 2` |
| `16_singular_values.png` | Singular values of the `(N_train, nf)` feature matrix at the best iteration — useful for diagnosing effective dimensionality | Always |

---

## Module Reference

| Module | Responsibility |
|---|---|
| `run_class_bci_competition_III_merged_spatial_filters_gpy.py` | CLI entry point; fold generation, alignment, inner-val split, CSP init, sweep loop; writes one result directory per `(nf, fold)` |
| `gp_classification_gpy.py` | `GPClassificationRunner` — model build, training, val-NLPD early stopping, best-checkpoint restore, per-fold logging and plots |
| `kernels_gpy.py` | `CustomKernelGPy` — GPy-compatible kernel; holds and applies the spatial filter W throughout training; computes analytic gradients w.r.t. W, ARD, and η |
| `Whitening.py` | `SpatialWhiteningDecomposition` — used for CSP filter initialisation only (when `--csp` is active); also provides `ApplySpatialFilters` and `Covariance` utilities |
| `align.py` | `align_split` — Euclidean / Riemannian covariance alignment; reference M is always estimated from the outer training pool only |
| `SVD.py` | `SingularValueDecomposition` — SVD helper with cached derived quantities (whitener, sqrtm, isqrtm, pinv, …) |
| `data_prep_motorimagery_bci_competition_III_merged.py` | Loads raw `.mat` files, bandpass-filters EEG, extracts per-trial covariance matrices, writes merged dataset pickle |
| `plot_metric_bci_competition_III_merged_spatial_filters.py` | Reads per-fold `run_log.json` files; aggregates and plots metrics across folds |
