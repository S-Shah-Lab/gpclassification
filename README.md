# Gaussian Process Classification for Spatial Filter Learning

A Python toolkit for decoding EEG signals using **Gaussian Process (GP) classification** with jointly-optimised spatial filters.  The pipeline covers everything from raw covariance-feature extraction through cross-validated training and evaluation, with a rich set of diagnostic plots saved automatically per fold.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Key Features](#key-features)
  - [Cross-Validation](#cross-validation)
  - [Shuffling & Random State](#shuffling--random-state)
  - [Covariance Alignment](#covariance-alignment)
  - [Spatial Filters (W) — CSP Init & Trainable/Fixed](#spatial-filters-w--csp-init--trainablefixed)
  - [Kernel Type](#kernel-type)
  - [ARD Scaling](#ard-scaling)
  - [Early Stopping](#early-stopping)
  - [Multi-Stage Optimisation](#multi-stage-optimisation)
- [Results Directory Layout](#results-directory-layout)
- [Module Reference](#module-reference)
- [Citation](#citation)

---

## Overview

The package implements a full end-to-end BCI decoding pipeline:

```
Raw EEG trials
    └─► Per-trial covariance matrices           (data_prep_*.py)
         └─► Optional covariance alignment       (align.py)
              └─► Spatial filter projection W    (Whitening.py / CSP init)
                   └─► Log-variance features
                        └─► GP classifier        (gp_classification_gpy.py)
                             └─► Per-fold metrics & plots
```

The spatial filter matrix **W** can be either **fixed** (e.g. pre-computed CSP) or **jointly optimised** together with the GP hyperparameters via gradient-based methods, making it possible to learn task-discriminative projections end-to-end.

---

## Repository Structure

```
.
├── run_class_bci_competition_III_merged_spatial_filters_gpy.py   # Main experiment script
├── gp_classification_gpy.py      # GP model, training loop, diagnostics
├── kernels_gpy.py                # Custom GPy kernels (RBF, Linear + spatial filter)
├── Whitening.py                  # Spatial whitening & filter application
├── align.py                      # Euclidean / Riemannian covariance alignment
├── SVD.py                        # SVD helper (whitening, sqrtm, pinv, …)
└── data_prep_motorimagery_bci_competition_III_merged.py  # Dataset preparation
```

---

## Installation

```bash
pip install numpy scipy scikit-learn matplotlib GPy
# Optional — required only for topomap plots:
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
    --alignment riemann
```

### Multi-stage optimiser

```bash
python run_class_bci_competition_III_merged_spatial_filters_gpy.py \
    --data-path ./data/data_set_IVa_symm_aa.pkl \
    --results-dir ./results \
    --nfs 4 8 --kfolds 4 \
    --optimizer-stages "lbfgsb:50" "scg:250"
```

---

## Key Features

### Cross-Validation

Two cross-validation modes are supported, selected automatically:

| Mode | Flag | Description |
|---|---|---|
| **K-Fold** | *(default)* | Splits all trials into `--kfolds` folds. Suitable for single-session / single-subject data. |
| **Leave-One-Subject-Out** | `--multi-subject` | Uses `trial_counts_by_file` stored in the dataset pickle to define one fold per subject/session. |

```bash
--kfolds 8              # number of folds (default: 8)
--multi-subject         # switch to leave-one-subject-out
```

For each `(nf, fold)` combination the script writes a self-contained result directory containing `run_log.json` and diagnostic PNGs.

---

### Shuffling & Random State

Two independent random seeds control reproducibility:

```bash
--shuffle-input         # shuffle trials before k-fold splitting (default: off)
--fold-seed 42          # RNG seed used by KFold when shuffle is on
--random-state 10       # seed passed to GPClassificationRunner (weight init, …)
```

When `--shuffle-input` is **off**, folds are deterministic even without a seed. The `shuffle_<True|False>` token is included in the results path so that shuffled and unshuffled runs never overwrite each other.

---

### Covariance Alignment

Between-session or between-subject differences in the mean covariance can be removed before feature extraction with:

```bash
--alignment none        # (default) no alignment
--alignment euclidean   # arithmetic mean whitening
--alignment riemann     # Riemannian (geometric) mean whitening  ← recommended
```

The reference matrix **M** is estimated from the **training split only** (no data leakage). Both train and test covariances are then whitened:

```
C_aligned = M^{-1/2} · C · M^{-1/2}
```

The Riemannian mean is computed via iterative geodesic updates on the manifold of symmetric positive-definite matrices (Barachant et al., 2012). The alignment method is part of the results path (`no_align`, `euclidean_align`, `riemann_align`).

---

### Spatial Filters (W) — CSP Init & Trainable/Fixed

The spatial filter matrix **W** of shape `(n_channels, nf)` projects multichannel covariances down to `nf` discriminative sources.

#### Initialisation

```bash
--csp                          # initialise W with CSP filters (recommended)
--spatialFilter-init random    # random init (default when --csp is not set)
--spatialFilter-init ones      # constant init
--nfs 1 2 4 8                  # sweep over number of filters
```

When `--csp` is set, CSP filters are recomputed from scratch on each training fold to prevent leakage. Filters are ranked by how far their generalised eigenvalue falls from 0.5 (most class-discriminative first).

#### Trainable vs. Fixed

```bash
# W is updated by gradient descent (default — jointly optimised with GP params)
# W is frozen at its initial value
--no-w-trainable
```

The trainability setting is encoded in the results path as `spatialFilter_trainable` or `spatialFilter_fixed`.

---

### Kernel Type

Two kernel families are available for the GP covariance function:

```bash
--kernel-type RBF      # Squared-exponential (Gaussian) kernel  ← default
--kernel-type Linear   # Linear (dot-product) kernel
```

Both kernels are applied in the **projected feature space** defined by the spatial filter W, and the kernel type is encoded in the results path (`kernel_rbf`, `kernel_linear`).

---

### ARD Scaling

Automatic Relevance Determination assigns an independent length-scale to each spatial filter output, allowing the model to down-weight uninformative filters:

```bash
--ard       # enable per-filter ARD length-scales
--eta       # enable a global output-scale parameter
```

The ARD setting is reflected in the results path (`ard_True` / `ard_False`).

---

### Early Stopping

Patience-based early stopping monitors the tracked metric (NLML or validation NLPD) and halts training when no improvement exceeding `es_min_delta` is observed for `es_patience` consecutive iterations:

```bash
--es-patience  20       # stop after 20 non-improving iterations (0 = disabled)
--es-min-delta 1e-4     # minimum improvement to reset the patience counter
```

The best model checkpoint (lowest metric) is restored automatically before evaluation, so early stopping never degrades test performance relative to running until convergence.

---

### Multi-Stage Optimisation

The optimiser schedule can be specified as a sequence of named stages, useful for warm-starting with a cheap method before refining with a more expensive one:

```bash
--optimizer-stages "lbfgsb:50" "scg:250"
# format: "<optimizer_name>:<max_iters>[:<lr>[:<momentum>]]"
```

If `--optimizer-stages` is omitted, a single SCG stage is used. The iteration budget defaults automatically based on which parameters are trainable:

| Trainable parameters | Default `--maxiter` |
|---|---|
| GP hyperparams only (W fixed, no ARD, no eta) | 10 |
| Any of W / ARD / eta enabled | 300 |

---

## Results Directory Layout

```
<results-dir>/
  <dataset-label>/
    shuffle_<True|False>/
      <no_align|euclidean_align|riemann_align>/
        ard_<True|False>/
          kernel_<rbf|linear>/
            spatialFilter_trainable/
            spatialFilter_fixed/
              nf_<k>/
                fold_<i>/
                  run_log.json          ← per-iteration metrics
                  learning_curve.png
                  roc_curve.png
                  calibration.png
                  confusion_matrix.png
                  feature_scatter.png
                  topomap.png           ← requires MNE
                  kernel_params.png
                  singular_values.png
```

---

## Module Reference

| Module | Responsibility |
|---|---|
| `run_class_bci_competition_III_merged_spatial_filters_gpy.py` | CLI entry point; fold generation, CSP init, sweep loop |
| `gp_classification_gpy.py` | `GPClassificationRunner` — model build, training, early stopping, logging, plots |
| `kernels_gpy.py` | `CustomKernelGPy` — GPy-compatible kernel with learnable spatial filter W |
| `Whitening.py` | `SpatialWhiteningDecomposition`, `ApplySpatialFilters`, `Covariance` |
| `align.py` | `align_split` — Euclidean / Riemannian covariance alignment |
| `SVD.py` | `SingularValueDecomposition` — SVD helper with cached derived quantities |
| `data_prep_motorimagery_bci_competition_III_merged.py` | Dataset loading and covariance feature extraction |

