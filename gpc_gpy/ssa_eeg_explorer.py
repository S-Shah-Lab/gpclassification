#!/usr/bin/env python3
"""
SSA exploration on BCI-III style EEG data.

This script is meant to work with the same kind of pickled data dict used in
`run_class_bci_competition_III_merged_spatial_filters_gpy.py`, i.e. a file
containing at least:

    data["X_eeg"] : np.ndarray, shape (n_epochs, n_channels, n_times)

The script:
    1. Loads the pickle file.
    2. Runs SSA on the time-domain epochs using SpatialWhiteningDecomposition.
    3. Plots SSA eigenvalues (non-stationarity measure).
    4. Reconstructs stationary vs non-stationary subspaces and visualizes
       them as EEG time series for a chosen epoch and channel.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Uses your existing SSA implementation & helpers
from Whitening import (  # noqa: E402
    SpatialWhiteningDecomposition,
    ApplySpatialFilters,
)
# Whitening.SpatialWhiteningDecomposition and SSA are defined in Whitening.py
# and are already used from the main runner.  We just reuse them here.


def load_bci_pickle(path: Path) -> dict:
    """
    Load a BCI-III style pickle file.

    Parameters
    ----------
    path : Path
        Path to the .pkl file.

    Returns
    -------
    dict
        Dictionary containing at least the key "X_eeg".

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        If "X_eeg" is missing in the loaded dict.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")

    with path.open("rb") as f:
        data = pickle.load(f)

    if "X_eeg" not in data:
        raise KeyError(
            "Expected key 'X_eeg' in data dict. "
            f"Available keys: {list(data.keys())}"
        )

    return data


def run_ssa_on_eeg(
    X_eeg: np.ndarray,
    ssa_max_rank: int | None = None,
) -> SpatialWhiteningDecomposition:
    """
    Run SSA on time-domain EEG epochs.

    Parameters
    ----------
    X_eeg : np.ndarray
        EEG data with shape (n_epochs, n_channels, n_times).
    ssa_max_rank : int or None
        Optional maximum rank for the whitening step, passed to
        SpatialWhiteningDecomposition.

    Returns
    -------
    SpatialWhiteningDecomposition
        Fitted SSA object, with `.eigenvalues`, `.W`, `.A` and `.Z`
        (sourceSignals) populated.
    """
    if X_eeg.ndim != 3:
        raise ValueError(
            f"Expected X_eeg with shape (N, C, T); got {X_eeg.shape}"
        )

    # mixedSignals: (epochs, channels, time)
    # sensorAxis  : 1 (channels axis), consistent with the runner.
    ssa = SpatialWhiteningDecomposition(
        mixedSignals=X_eeg,
        sensorAxis=1,
        maxRank=ssa_max_rank,
    )

    # SSA works in whitened space, using epochAxis=0 (trials) and
    # sensorAxis=1 (channels/sources).
    ssa.SSA(epochAxis=0, trainingSubset=None)

    return ssa


def split_components_by_stationarity(
    eigenvalues: np.ndarray,
    n_stationary: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split SSA components into stationary and non-stationary sets.

    The SSA eigenvalues are ordered in descending non-stationarity:
    larger eigenvalues -> more non-stationary.

    We therefore define:
        - stationary components: those with the smallest eigenvalues
        - non-stationary components: those with the largest eigenvalues

    Parameters
    ----------
    eigenvalues : np.ndarray
        Vector of eigenvalues (1D).
    n_stationary : int or None
        How many components to treat as "stationary".
        If None, use floor(n_components / 2).

    Returns
    -------
    (stationary_idx, nonstationary_idx) : Tuple[np.ndarray, np.ndarray]
        Indices of stationary and non-stationary components.
    """
    eigs = np.asarray(eigenvalues).ravel()
    n_components = eigs.size

    if n_components == 0:
        raise ValueError("No SSA components found (eigenvalues array is empty).")

    if n_stationary is None:
        n_stationary = n_components // 2
        if n_stationary == 0:
            n_stationary = 1

    if n_stationary <= 0 or n_stationary > n_components:
        raise ValueError(
            f"n_stationary must be in [1, {n_components}], got {n_stationary}"
        )

    # Ascending -> smallest are most stationary
    order_asc = np.argsort(eigs)

    stationary_idx = order_asc[:n_stationary]
    nonstationary_idx = order_asc[-n_stationary:]

    return stationary_idx, nonstationary_idx


def reconstruct_subspace(
    ssa: SpatialWhiteningDecomposition,
    component_indices: Iterable[int],
) -> np.ndarray:
    """
    Reconstruct sensor-space signals from a subset of SSA components.

    Uses:
        X_rec = A[:, K]^T @ Z[:, K, :]
    implemented via ApplySpatialFilters, exactly as in PlotEpochs.

    Parameters
    ----------
    ssa : SpatialWhiteningDecomposition
        Fitted SSA object.
    component_indices : Iterable[int]
        Indices of SSA components to keep.

    Returns
    -------
    np.ndarray
        Reconstructed signals with shape (n_epochs, n_channels, n_times).
    """
    comp_idx = np.asarray(list(component_indices), dtype=int).ravel()
    if comp_idx.size == 0:
        raise ValueError("component_indices is empty.")

    Z = ssa.sourceSignals          # (N, U, T)
    A = ssa.spatialPatternMatrix   # (C, U)

    if Z is None or A is None:
        raise RuntimeError(
            "SSA has not been run or rotation not set; "
            "sourceSignals or spatialPatternMatrix is None."
        )

    # Select components
    Z_sel = Z[:, comp_idx, :]       # (N, K, T)
    A_sel = A[:, comp_idx]          # (C, K)

    # Project back to sensor space:
    # ApplySpatialFilters expects the 'sensorAxis' to be the component axis in Z.
    X_rec = ApplySpatialFilters(
        signal=Z_sel,               # (N, K, T)
        spatialFilteringMatrix=A_sel.T,  # (K, C)
        sensorAxis=1,               # axis 1 is "sensors" here (K)
    )  # -> (N, C, T)

    return X_rec


def plot_eigenvalues(
    eigenvalues: np.ndarray,
    normalize: bool = False,
) -> None:
    """
    Plot SSA eigenvalues (non-stationarity measure).

    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues from SSA.
    normalize : bool
        If True, divide by sum(eigenvalues) so they sum to 1.
    """
    eigs = np.asarray(eigenvalues).ravel()
    if eigs.size == 0:
        raise ValueError("Empty eigenvalues array.")

    if normalize:
        total = np.sum(eigs)
        if total > 0:
            eigs = eigs / total

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(eigs, marker="o")
    ax.set_xlabel("Component index")
    ax.set_ylabel("Non-stationarity (SSA eigenvalue)")
    ax.set_title("SSA eigenvalues (larger = more non-stationary)")
    ax.grid(True)
    fig.tight_layout()


def plot_epoch_channel_comparison(
    X_orig: np.ndarray,
    X_stat: np.ndarray,
    X_nonstat: np.ndarray,
    epoch_idx: int,
    channel_idx: int,
) -> None:
    """
    Plot original vs stationary vs non-stationary signals for one epoch & channel.

    Parameters
    ----------
    X_orig : np.ndarray
        Original sensor-space data, shape (N, C, T).
    X_stat : np.ndarray
        Reconstructed stationary subspace, shape (N, C, T).
    X_nonstat : np.ndarray
        Reconstructed non-stationary subspace, shape (N, C, T).
    epoch_idx : int
        Index of the epoch to visualize (0-based).
    channel_idx : int
        Index of the channel to visualize (0-based).
    """
    n_epochs, n_channels, n_times = X_orig.shape

    if not (0 <= epoch_idx < n_epochs):
        raise IndexError(
            f"epoch_idx={epoch_idx} is out of range [0, {n_epochs - 1}]"
        )
    if not (0 <= channel_idx < n_channels):
        raise IndexError(
            f"channel_idx={channel_idx} is out of range [0, {n_channels - 1}]"
        )

    t = np.arange(n_times)  # sample index as "time"

    sig_orig = X_orig[epoch_idx, channel_idx, :]
    sig_stat = X_stat[epoch_idx, channel_idx, :]
    sig_nonstat = X_nonstat[epoch_idx, channel_idx, :]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(t, sig_orig)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(
        f"Original signal (epoch {epoch_idx}, channel {channel_idx})"
    )
    axes[0].grid(True)

    axes[1].plot(t, sig_stat)
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("Stationary subspace reconstruction")
    axes[1].grid(True)

    axes[2].plot(t, sig_nonstat)
    axes[2].set_ylabel("Amplitude")
    axes[2].set_xlabel("Time (samples)")
    axes[2].set_title("Non-stationary subspace reconstruction")
    axes[2].grid(True)

    fig.tight_layout()



n_stationary = 2
epoch_idx    = 0
channel_idx  = 5

data_path = Path("/mnt/c/Users/scana/Desktop/gpc/data/data_set_IVa_symm_aa.pkl")
data = load_bci_pickle(data_path)

X_eeg = np.asarray(data["X_eeg"])
print(f"Loaded X_eeg with shape {X_eeg.shape}")

# 1) Run SSA on all epochs
ssa = run_ssa_on_eeg(X_eeg, ssa_max_rank=None)

# 2) Plot eigenvalues
plot_eigenvalues(ssa.eigenvalues, normalize=True)

# 3) Build stationary vs non-stationary reconstructions
stationary_idx, nonstationary_idx = split_components_by_stationarity(
    eigenvalues=ssa.eigenvalues,
    n_stationary=n_stationary,
)
print(f"Stationary components:     {stationary_idx.tolist()}")
print(f"Non-stationary components: {nonstationary_idx.tolist()}")

X_stat = reconstruct_subspace(ssa, stationary_idx)
X_nonstat = reconstruct_subspace(ssa, nonstationary_idx)

# 4) Plot original vs stationary vs non-stationary at selected epoch & channel
plot_epoch_channel_comparison(
    X_orig=ssa.mixedSignals,
    X_stat=X_stat,
    X_nonstat=X_nonstat,
    epoch_idx=epoch_idx,
    channel_idx=channel_idx,
)

# Show all figures
plt.show()