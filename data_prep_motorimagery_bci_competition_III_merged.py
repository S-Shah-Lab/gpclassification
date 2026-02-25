"""
BCI Competition IVa dataset preparation for GP covariance classification.

Overview
--------
This script loads one or more BCI Competition IVa subject ``.mat`` files,
applies a shared preprocessing pipeline, and saves a single merged pickle
that the experiment runner (``run_class_bci_competition_III_merged_spatial_filters_gpy.py``)
can consume directly.

Pipeline
--------
For each subject file the script:

1. **Determines a global post-cue window length** by scanning all files and
   taking the minimum usable duration across cue-onset intervals, accounting
   for a ``initial_time_to_remove`` skip (to avoid the motor-preparation
   transient) and a ``final_relax_time`` buffer at the end.
   The window is rounded to 0.1 s for consistency.

2. **Enforces a canonical channel order**: the channel list from the first
   file is adopted as the reference.  All subsequent files must match it
   exactly (content and order) or the script raises a ``ValueError``.

3. **Bandpass-filters** the continuous EEG (8–25 Hz, 4th-order zero-phase
   Butterworth, SOS form) and scales samples to microvolts (× 0.1).

4. **Extracts per-trial covariance matrices** over the global window starting
   ``initial_time_to_remove`` s after each cue onset.

5. **Optionally symmetrises, unit-trace-normalises, or shrinkage-regularises**
   each covariance matrix (controlled by flags at the bottom of the script).

6. **Concatenates** all valid trials across subjects and writes a single
   ``data/`` pickle containing covariance matrices, labels, raw EEG epochs,
   channel metadata, and trial counts per file.

Output pickle keys
------------------
``"X"``
    ``(N_total, C, C)`` float64 array of covariance matrices.
``"Y"``
    ``(N_total,)`` int64 label array; values in ``{0, 1}``.
``"X_eeg"``
    ``(N_total, C, T)`` float64 array of bandpass-filtered EEG epochs
    (channels × time, in canonical order).
``"ch_names"``
    ``(C,)`` array of lower-case channel names in canonical order.
``"ch_location"``
    ``{name: [x, y]}`` electrode coordinates (radially scaled by 1.2).
``"trial_counts_by_file"``
    ``{dataset_label: n_trials}`` mapping.
``"groups"``
    List of length ``N_total``; ``groups[i]`` is the subject label for
    trial ``i``.  Used by the runner for leave-one-subject-out folding.
``"window_seconds"``
    The global enforced window length in seconds.
``"band_hz"``
    ``[lowcut, highcut]`` in Hz.
``"notes"``
    Human-readable provenance string.

Usage
-----
Edit the configuration block inside ``main()`` and run::

    python data_prep_motorimagery_bci_competition_III_merged.py

Dependencies
------------
NumPy, SciPy (``signal``, ``io``), pickle, os
"""

import os
import pickle

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt


# ===========================================================================
# Signal processing
# ===========================================================================

def bandpass_filter_channelwise(
    X: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter channel-by-channel.

    Filtering one channel at a time caps peak memory usage, which matters
    when ``X`` has many samples.  The SOS representation is used for
    numerical stability at the chosen order.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_channels)
        Continuous EEG data in row-major sample order.
    lowcut : float
        Lower passband edge in Hz.
    highcut : float
        Upper passband edge in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int
        Filter order (default 4).

    Returns
    -------
    np.ndarray, shape (n_samples, n_channels), dtype float64
        Filtered signal; same shape as input.
    """
    sos = butter(order, [lowcut, highcut], btype="band", fs=fs, output="sos")
    Xf  = np.empty_like(X)
    for i in range(X.shape[1]):
        Xf[:, i] = sosfiltfilt(sos, X[:, i])
    return Xf


# ===========================================================================
# .mat file helpers
# ===========================================================================

def _load_mat_file(data_root: str, dataset_label: str) -> dict:
    """
    Load the IVa ``.mat`` file for a given subject label.

    Parameters
    ----------
    data_root : str
        Base directory containing ``{label}_mat/1000Hz/{label}.mat``.
    dataset_label : str
        Subject identifier string (e.g. ``"data_set_IVa_aa"``).

    Returns
    -------
    dict
        The ``scipy.io.loadmat`` output dictionary.
    """
    mat_path = os.path.join(
        data_root, f"{dataset_label}_mat", "1000Hz", f"{dataset_label}.mat"
    )
    return loadmat(mat_path)


def _extract_metadata(mat: dict):
    """
    Extract recording metadata from a loaded IVa ``.mat`` dictionary.

    Parameters
    ----------
    mat : dict
        Output of ``scipy.io.loadmat``.

    Returns
    -------
    file_name : str
    fs : float
        Sampling frequency in Hz.
    ch_names_all : list of str
        Lower-case channel labels in the original file order.
    ch_x_all : np.ndarray
        X electrode coordinates (arbitrary units).
    ch_y_all : np.ndarray
        Y electrode coordinates (arbitrary units).
    """
    file_name    = str(mat["nfo"][0][0][0][0])
    fs           = float(mat["nfo"][0][0][1][0][0])
    ch_names_all = [
        str(mat["nfo"][0][0][2][0][ch][0]).lower()
        for ch in range(mat["nfo"][0][0][2][0].shape[0])
    ]
    ch_x_all = mat["nfo"][0][0][3].flatten()
    ch_y_all = mat["nfo"][0][0][4].flatten()
    return file_name, fs, ch_names_all, ch_x_all, ch_y_all


def _per_dataset_min_window_seconds(
    cue_onsets: np.ndarray,
    fs: float,
    initial_time_to_remove: float,
    final_relax_time: float,
) -> float:
    """
    Compute the minimum usable post-cue window for a single recording.

    The available duration for each pair of consecutive cues is:
    ``delta_t − initial_time_to_remove − final_relax_time``.
    The minimum across all pairs is returned, rounded to 0.1 s.

    Parameters
    ----------
    cue_onsets : np.ndarray, shape (n_trials,)
        Sample indices of cue onsets.
    fs : float
        Sampling frequency in Hz.
    initial_time_to_remove : float
        Seconds to skip after each cue onset (avoids motor-preparation transient).
    final_relax_time : float
        Seconds to reserve at the end of each inter-cue interval.

    Returns
    -------
    float
        Minimum usable window in seconds (rounded to 0.1 s).
        Returns ``0.0`` if there are fewer than two cue onsets.
    """
    time_windows = []
    for t1, t2 in zip(cue_onsets[:-1], cue_onsets[1:]):
        delta_t  = (t2 - t1) / fs
        t_avail  = delta_t - initial_time_to_remove
        time_windows.append(t_avail - final_relax_time)

    if not time_windows:
        return 0.0

    return float(np.round(np.min(time_windows), 1))


# ===========================================================================
# Covariance postprocessing
# ===========================================================================

def _postprocess_cov(
    cov: np.ndarray,
    symm_flag: bool,
    unit_trace_flag: bool,
    shrink_flag: bool,
    lambda_shrink: float,
) -> np.ndarray:
    """
    Apply optional covariance regularisation steps in order.

    Steps (each applied only when its flag is ``True``):

    1. **Symmetrise**: ``C = 0.5 * (C + C.T)``  —  removes floating-point
       asymmetry introduced by ``numpy.cov``.
    2. **Unit-trace normalise**: ``C /= trace(C)``  —  removes global amplitude
       differences between trials (e.g. day-to-day power drift, impedance).
    3. **Shrinkage**: ``C = (1 − λ) C + λ target``  —  blends the empirical
       covariance with a scaled identity to improve conditioning.  When
       ``unit_trace_flag`` is active the target is ``I / d`` (unit-trace
       identity); otherwise it is ``trace(C) / d · I``.

    Parameters
    ----------
    cov : np.ndarray, shape (d, d)
        Raw sample covariance matrix.
    symm_flag : bool
    unit_trace_flag : bool
    shrink_flag : bool
    lambda_shrink : float
        Shrinkage coefficient in ``[0, 1]``.

    Returns
    -------
    np.ndarray, shape (d, d)
        Processed covariance matrix, or ``None`` if unit-trace normalisation
        would divide by a non-finite or non-positive trace.
    """
    if symm_flag:
        cov = 0.5 * (cov + cov.T)

    if unit_trace_flag:
        tr = np.trace(cov)
        if tr <= 0 or not np.isfinite(tr):
            return None   # signal to the caller to skip this trial
        cov = cov / tr

    if shrink_flag:
        d = cov.shape[0]
        if unit_trace_flag:
            # Covariance is unit-trace → blend with I/d
            target = np.eye(d) / d
        else:
            # Covariance has arbitrary scale → match trace with I · trace/d
            target = np.diag([np.trace(cov)] * d) / d
        cov = (1.0 - lambda_shrink) * cov + lambda_shrink * target

    return cov


# ===========================================================================
# Per-subject processing
# ===========================================================================

def process_subject(
    dataset_label: str,
    data_root: str,
    canonical_channels: list,
    global_window_seconds: float,
    band: tuple,
    initial_time_to_remove: float,
    symm_flag: bool,
    unit_trace_flag: bool,
    shrink_flag: bool,
    lambda_shrink: float,
) -> dict:
    """
    Load, filter, and covariance-extract one subject's recording.

    Parameters
    ----------
    dataset_label : str
        Subject identifier (e.g. ``"data_set_IVa_aa"``).
    data_root : str
        Base directory for the raw ``.mat`` files.
    canonical_channels : list of str
        Ordered list of channel names to extract (must match the file).
    global_window_seconds : float
        Common post-cue window length applied to all subjects.
    band : tuple of float
        ``(lowcut, highcut)`` in Hz for the bandpass filter.
    initial_time_to_remove : float
        Seconds to skip after each cue onset.
    symm_flag, unit_trace_flag, shrink_flag : bool
        Covariance postprocessing flags; see ``_postprocess_cov``.
    lambda_shrink : float
        Shrinkage coefficient.

    Returns
    -------
    dict with keys:
        ``"X_cov"``  — ``(n_trials, C, C)`` covariance matrices
        ``"Y_cov"``  — ``(n_trials,)`` int labels in ``{0, 1}``
        ``"X_seg"``  — ``(n_trials, C, T)`` bandpass-filtered epochs
        ``"n_trials"`` — number of kept labeled trials
    """
    mat = _load_mat_file(data_root, dataset_label)
    file_name, fs, ch_names_all, ch_x_all, ch_y_all = _extract_metadata(mat)

    # Re-order channels to the canonical order
    idx       = [ch_names_all.index(ch) for ch in canonical_channels]
    lowcut, highcut = band

    # Scale to µV and bandpass filter
    X = mat["cnt"][:, idx].astype(np.float64) * 0.1
    X_filtered = bandpass_filter_channelwise(X, lowcut, highcut, fs, order=4)

    # Labels: remap {1, 2} → {0, 1}; any label < 0 is unlabeled (skip)
    Y_raw     = mat["mrk"][0][0][1].flatten()
    Y         = np.where(Y_raw == 1, 0, np.where(Y_raw == 2, 1, Y_raw))
    cue_onset = mat["mrk"][0][0][0].flatten()

    n_samples_win = int(global_window_seconds * fs)
    offset        = int(initial_time_to_remove * fs)

    X_cov, Y_cov, X_seg = [], [], []

    for y_label, cue_t in zip(Y, cue_onset):
        t_start = int(cue_t) + offset
        t_end   = t_start + n_samples_win
        if t_end > X_filtered.shape[0]:
            continue  # window exceeds recording boundary

        segment = X_filtered[t_start:t_end, :]     # (T, C)
        cov     = np.cov(segment, rowvar=False, bias=False).astype(np.float64)

        cov = _postprocess_cov(
            cov, symm_flag, unit_trace_flag, shrink_flag, lambda_shrink
        )
        if cov is None:
            continue  # unit-trace normalisation failed (non-finite trace)

        if y_label >= 0:
            X_cov.append(cov)
            Y_cov.append(int(y_label))
            X_seg.append(segment.T.astype(np.float64))  # store as (C, T)

    C = len(canonical_channels)
    X_cov = np.stack(X_cov, axis=0) if X_cov else np.empty((0, C, C),        dtype=np.float64)
    Y_cov = np.array(Y_cov, dtype=np.int64) if Y_cov else np.empty((0,),       dtype=np.int64)
    X_seg = np.stack(X_seg, axis=0) if X_seg else np.empty((0, C, n_samples_win), dtype=np.float64)

    print(
        f"  {dataset_label}: kept {Y_cov.size} labeled trials  "
        f"({n_samples_win} samples = {global_window_seconds:.1f} s @ {fs:.0f} Hz)"
    )
    return {"X_cov": X_cov, "Y_cov": Y_cov, "X_seg": X_seg, "n_trials": int(Y_cov.size)}


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    """
    Run the full dataset preparation pipeline.

    Edit the configuration block below to change subjects, paths, or
    preprocessing flags.  The output pickle is written to ``result_root``.
    """

    # ------------------------------------------------------------------ #
    # Configuration                                                        #
    # ------------------------------------------------------------------ #
    dataset_labels = [
        "data_set_IVa_aa",
        "data_set_IVa_al",
        # "data_set_IVa_av",
        # "data_set_IVa_aw",
        # "data_set_IVa_ay",
    ]

    data_root    = "/mnt/c/Users/scana/Desktop/gpc/data"
    result_root  = "/mnt/c/Users/scana/Desktop/gpc/data"

    band                  = (8.0, 25.0)   # bandpass filter edges in Hz
    initial_time_to_remove = 1.0          # seconds to skip after each cue onset
    final_relax_time       = 2.25         # end-of-interval buffer in seconds

    # Covariance postprocessing flags
    symm_flag       = True    # symmetrise: C = 0.5 * (C + C.T)
    unit_trace_flag = False   # normalise: C /= trace(C)
    shrink_flag     = False   # Ledoit–Wolf-style shrinkage
    lambda_shrink   = 0.02    # shrinkage coefficient (ignored when shrink_flag=False)

    # ------------------------------------------------------------------ #
    # Output file naming                                                   #
    # ------------------------------------------------------------------ #
    ending = "_".join(lbl.split("_")[-1] for lbl in dataset_labels) + ".pkl"
    name_parts = ["data_set_IVa"]
    if symm_flag:        name_parts.append("symm")
    if unit_trace_flag:  name_parts.append("unittrace")
    if shrink_flag:      name_parts.append(f"shrink{str(lambda_shrink).replace('.', '')}")
    name_of_output = "_".join(name_parts) + "_" + ending

    # ------------------------------------------------------------------ #
    # Pass 1: determine global window length                               #
    # ------------------------------------------------------------------ #
    print("Scanning datasets to determine global window length …")
    global_window_seconds = np.inf
    canonical_channels    = None
    canonical_xy          = None
    per_file_window_s     = {}

    for dataset_label in dataset_labels:
        mat = _load_mat_file(data_root, dataset_label)
        _, fs, ch_names_all, ch_x_all, ch_y_all = _extract_metadata(mat)

        ch_x_all = ch_x_all * 1.2   # radial scaling for plotting convenience
        ch_y_all = ch_y_all * 1.2

        if canonical_channels is None:
            canonical_channels = list(ch_names_all)
            canonical_xy = {
                ch: [float(ch_x_all[i]), float(ch_y_all[i])]
                for i, ch in enumerate(canonical_channels)
            }
        else:
            if list(ch_names_all) != canonical_channels:
                raise ValueError(
                    f"Channel mismatch in {dataset_label}.  Expected "
                    f"{canonical_channels[:5]}…; got {ch_names_all[:5]}…  "
                    "Unify channel ordering before merging."
                )

        cue_onsets = mat["mrk"][0][0][0].flatten()
        per_win    = _per_dataset_min_window_seconds(
            cue_onsets, fs, initial_time_to_remove, final_relax_time
        )
        per_file_window_s[dataset_label] = per_win
        global_window_seconds = min(global_window_seconds, per_win)

    print(f"Global post-cue window: {global_window_seconds:.1f} s")
    print("Per-file windows (s):", per_file_window_s)

    # ------------------------------------------------------------------ #
    # Pass 2: extract covariance matrices for all subjects                 #
    # ------------------------------------------------------------------ #
    X_list, Y_list, X_seg_list = [], [], []
    trial_counts_by_file       = {}

    for dataset_label in dataset_labels:
        print(f"\n=== {dataset_label} ===")
        result = process_subject(
            dataset_label         = dataset_label,
            data_root             = data_root,
            canonical_channels    = canonical_channels,
            global_window_seconds = global_window_seconds,
            band                  = band,
            initial_time_to_remove= initial_time_to_remove,
            symm_flag             = symm_flag,
            unit_trace_flag       = unit_trace_flag,
            shrink_flag           = shrink_flag,
            lambda_shrink         = lambda_shrink,
        )
        X_list.append(result["X_cov"])
        Y_list.append(result["Y_cov"])
        X_seg_list.append(result["X_seg"])
        trial_counts_by_file[dataset_label] = result["n_trials"]

    # ------------------------------------------------------------------ #
    # Merge across subjects                                                #
    # ------------------------------------------------------------------ #
    C = len(canonical_channels)
    T = int(global_window_seconds * 1000)  # approximate; actual T set by process_subject

    X_merged     = np.concatenate(X_list,     axis=0) if X_list     else np.empty((0, C, C), dtype=np.float64)
    Y_merged     = np.concatenate(Y_list,     axis=0) if Y_list     else np.empty((0,),       dtype=np.int64)
    X_eeg_merged = np.concatenate(X_seg_list, axis=0) if X_seg_list else np.empty((0, C, T),  dtype=np.float64)

    # Build trial-to-subject mapping list
    groups = []
    for name, count in trial_counts_by_file.items():
        if count <= 0:
            raise ValueError(f"No trials retained for subject '{name}'.")
        groups.extend([str(name)] * count)

    lowcut, highcut = band
    final_dict = {
        "X"                   : X_merged,
        "Y"                   : Y_merged,
        "X_eeg"               : X_eeg_merged,
        "ch_names"            : np.array(canonical_channels),
        "ch_location"         : canonical_xy,
        "trial_counts_by_file": trial_counts_by_file,
        "groups"              : groups,
        "window_seconds"      : float(global_window_seconds),
        "band_hz"             : [float(lowcut), float(highcut)],
        "notes"               : (
            "Covariance matrices computed over a globally consistent "
            "post-cue window (same length for all subjects)."
        ),
    }

    print(f"\nMerged dataset: X={final_dict['X'].shape}, Y={final_dict['Y'].shape}")
    print("Trial counts:", final_dict["trial_counts_by_file"])

    # ------------------------------------------------------------------ #
    # Export                                                               #
    # ------------------------------------------------------------------ #
    os.makedirs(result_root, exist_ok=True)
    out_path = os.path.join(result_root, name_of_output)
    with open(out_path, "wb") as fh:
        pickle.dump(final_dict, fh)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
