"""
Merge BCI Competition IVa subject .mat files into a single dataset of covariance features using a globally consistent post-cue window

Description:
  - Scans all listed datasets to determine a global window length (seconds) by taking the minimum usable post-cue duration across files, given:
        initial_time_to_remove = 1.0 s
        final_relax_time       = 2.25 s
  - Enforces identical channel order across files; raises if mismatch
  - Loads EEG, scales samples to microvolts (x0.1), band-passes 8-25 Hz
    (Butterworth, order=4, SOS, zero-phase), then for each trial:
      • slices the global window after cue onset
      • computes sample covariance (C x C), float64
      • remaps labels {1,2} → {0,1} and drops unlabeled (<0)
  - Concatenates all trials into one dataset and records per-file trial counts

Inputs:
  - data_root: base path containing {dataset_label}_mat/1000Hz/{dataset_label}.mat
  - dataset_labels: list of subject IDs to include

Outputs:
  - ./data/data_set_IVa_merged.pkl containing:
      {
        "X": (N_total, C, C) float64 covariance matrices,
        "Y": (N_total,) int64 labels in {0,1},
        "ch_names": (C,) array of channel names (canonical order),
        "ch_location": {name: [x, y]} electrode coords (scaled x1.2),
        "dataset_labels": list[str],
        "trial_counts_by_file": [{"dataset_label": str, "n_trials": int}, ...],
        "window_seconds": float,    global enforced window length (s)
        "band_hz": [8.0, 25.0],
        "notes": "Covariance computed over a globally consistent post-cue window for all files"
      }

Notes:
  - Global window is rounded to 0.1 s and applied uniformly across datasets
  - Trials whose windows exceed recording bounds are skipped safely
  - Channel lists must match exactly (content and order) or the script raises
"""

import pickle, os
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.io import loadmat


def _print_message(message: str, which: str) -> None:
    # ANSI color codes for sprinkly terminal
    GREEN = "\033[92m"
    RESET = "\033[0m"

    if which == "start":
        print(f"\n=== FILE PREP START ===")
        print(f"{GREEN}{message}{RESET}")
    elif which == "end":
        print(f"=== FILE PREP END ===")
    else:
        return


def bandpass_filter_channelwise(X, lowcut, highcut, fs, order=4):
    """
    Memory-friendly zero-phase Butterworth bandpass on X of shape (samples, channels)
    Returns a new float64 array with the same shape as X
    """
    # Design in SOS form with explicit sampling freq (no pre-normalization needed)
    sos = butter(order, [lowcut, highcut], btype="band", fs=fs, output="sos")

    Xf = np.empty_like(X)  # same dtype/shape
    # Filter one channel at a time to cap memory usage
    for i in range(X.shape[1]):
        Xf[:, i] = sosfiltfilt(sos, X[:, i])
    return Xf


def _load_mat_file(dataset_label: str):
    """
    Load the IVa .mat dictionary for a given dataset label
    """
    mat_path = os.path.join(
        data_root, f"{dataset_label}_mat", "1000Hz", f"{dataset_label}.mat"
    )
    return loadmat(mat_path)


def _extract_metadata(mat):
    """
    Extract file metadata and channel info from the loaded .mat

    Returns:
        file_name : str
        fs : float
        ch_names_all : list[str] lowercase
        ch_x_all, ch_y_all : np.ndarray
    """
    file_name = str(mat["nfo"][0][0][0][0])
    fs = float(mat["nfo"][0][0][1][0][0])
    ch_names_all = [
        str(mat["nfo"][0][0][2][0][ch][0]).lower()
        for ch in range(mat["nfo"][0][0][2][0].shape[0])
    ]
    ch_x_all = mat["nfo"][0][0][3].flatten()
    ch_y_all = mat["nfo"][0][0][4].flatten()
    return file_name, fs, ch_names_all, ch_x_all, ch_y_all


def _per_dataset_min_window_seconds(cue_onsets, fs):
    """
    Compute the per-dataset minimum usable post-cue window in seconds given the skip-at-start and end-relax assumptions
    """
    time_windows = []
    deltas = []
    for t1, t2 in zip(cue_onsets[:-1], cue_onsets[1:]):
        delta_t = (t2 - t1) / fs
        deltas.append(delta_t)
        t_avail = delta_t - initial_time_to_remove
        t_min = t_avail - final_relax_time
        time_windows.append(t_min)

    time_windows = np.array(time_windows, dtype=np.float64)
    # If a dataset has only one cue, there is no t2; fallback to a conservative 0
    if time_windows.size == 0:
        return 0.0

    # Round to 0.1s to be consistent with original script behavior
    return float(np.round(np.min(time_windows), 1))



if __name__ == "__main__": 
# Statement in Python creates a gate
# Code runs only when the file is executed as a standalone script, not when imported into another module
# When Python file runs directly --> its __name__ attribute is set to "__main__"
# When the same file is imported by another script, __name__ takes the value of the module's filename

    symm_flag       = True
    unit_trace_flag = False
    shrink_flag     = False
    lambda_shrink = 0.02

    # Load .mat file
    dataset_labels = [
        "data_set_IVa_aa",
        "data_set_IVa_al",
        #"data_set_IVa_av",
        #"data_set_IVa_aw",
        #"data_set_IVa_ay",
    ]
    # Create label for output file and folder name, ending contains subject names and file type
    ending = []
    for dataset_label in dataset_labels:
        ending.append(dataset_label.split('_')[-1])
    ending = '_'.join(ending)
    ending += '.pkl'
    
    # File naming structure
    name_of_output = 'data_set_IVa_' 
    if symm_flag:
        name_of_output += 'symm_'
    if unit_trace_flag:
        name_of_output += 'unittrace_'
    if shrink_flag:
        name_of_output += f'shrink{str(lambda_shrink).replace('.', '')}_'
    name_of_output += ending
     

    data_root = "/mnt/c/Users/scana/Desktop/gpc/data"
    band = (8.0, 25.0)             # limits of Hz to consider for bandpass filter
    initial_time_to_remove = 1.0   # seconds to skip after each cue onset
    final_relax_time = 2.25        # seconds to reserve at the end (assume max relax)
    global_window_seconds = np.inf # time window that is updated with each file
    canonical_channels = None
    canonical_xy = None

    # Determination of mutual window length 
    print("Scanning datasets to determine global window length...")
    per_file_window_s = {}  # dataset_label -> seconds

    for dataset_label in dataset_labels:
        # Load file and store info
        mat = _load_mat_file(dataset_label)
        file_name, fs, ch_names_all, ch_x_all, ch_y_all = _extract_metadata(mat)

        # All files should have the same list of channels so this is repeated
        # Radial scale for plotting convenience, as in your original code
        ch_x_all = ch_x_all * 1.2
        ch_y_all = ch_y_all * 1.2

        # Establish canonical channel set and order on first file
        if canonical_channels is None:
            canonical_channels = list(ch_names_all)
            canonical_xy = {
                ch: [float(ch_x_all[i]), float(ch_y_all[i])]
                for i, ch in enumerate(canonical_channels)
            }
        else:
            # Sanity check: channel lists must match exactly (order and content)
            if list(ch_names_all) != canonical_channels:
                raise ValueError(
                    f"Channel mismatch in {dataset_label}. Expected exactly "
                    f"{canonical_channels[:5]}... got {ch_names_all[:5]}... "
                    "Unify channel ordering before merging."
                )

        # Extract cue onsets to determine window length for the file
        # Compare to existing window lengths and determine best window length
        cue_onsets = mat["mrk"][0][0][0].flatten()
        per_win = _per_dataset_min_window_seconds(cue_onsets, fs)
        per_file_window_s[dataset_label] = per_win
        global_window_seconds = min(global_window_seconds, per_win)

        print(f"Global post-cue window (seconds): {global_window_seconds:.1f}")
        print("Per-file windows (seconds):", per_file_window_s)
    # Window length to use has been established
    
    # Pre=process all files to generate the covariance matrices
    X_list     = []
    Y_list     = []
    X_seg_list = []
    trial_counts_by_file = {}  # {"dataset_label": n_trials}

    lowcut, highcut = band

    for dataset_label in dataset_labels:
        # Print message on terminal
        _print_message(message=f"{dataset_label}", which="start")

        # Load file and store info
        mat = _load_mat_file(dataset_label)
        file_name, fs, ch_names_all, ch_x_all, ch_y_all = _extract_metadata(mat)

        # Indices for canonical order
        idx = [ch_names_all.index(ch) for ch in canonical_channels]

        # Channel metadata in canonical order
        ch_names = np.array([ch_names_all[i] for i in idx])
        ch_x = (ch_x_all * 1.2)[idx]
        ch_y = (ch_y_all * 1.2)[idx]
        ch_location = {ch: [float(ch_x[i]), float(ch_y[i])] for i, ch in enumerate(ch_names)}

        # Signal selection and scaling to microvolts
        X = mat["cnt"][:, idx].astype(np.float64, copy=False)
        X *= 0.1 # scale to uV according to instructions provided by website

        # Bandpass filter
        X_filtered = bandpass_filter_channelwise(X, lowcut, highcut, fs, order=4)

        # Labels and cue onsets
        Y_raw = mat["mrk"][0][0][1].flatten()      # (nTrials,)
        Y = np.where(Y_raw == 1, 0, np.where(Y_raw == 2, 1, Y_raw)) # convert Y labels to 0 or 1
        cue_onsets = mat["mrk"][0][0][0].flatten() # (nTrials,)

        # Windowing with global seconds that was previously determined
        n_samples_win = int(global_window_seconds * fs)
        offset = int(initial_time_to_remove * fs)

        # Create covariance matrices using the allowed time window for each trial
        X_cov = []
        Y_cov = []
        X_seg = []
        
        for y_label, cue_t in zip(Y, cue_onsets):
            t_start = int(cue_t) + offset
            t_end = t_start + n_samples_win
            if t_end > X_filtered.shape[0]:
                # Skip trial if the window would exceed the recording length (for any reasons)
                continue
            segment = X_filtered[ t_start : t_end, : ] # shape (T, nChannels*)
            cov = np.cov(segment, rowvar=False, bias=False).astype(np.float64)
            
            if symm_flag:
                # Make the covariance numerically symmetric
                cov = 0.5 * (cov + cov.T)
        
            if unit_trace_flag:
                # Unit-trace normalization
                # Trace in covariance matrix represents total variance 
                #   Removes global power scaling between trials (e.g., electrode impedance, muscle tone, day-to-day amplitude drift)
                #   Keeps all trials on a common footing so features describe relative power distribution, not absolute magnitude
                #   Makes eigenvalues of class means land in [0,1], stabilizing whitening and CSP decomposition
                tr = np.trace(cov)
                if tr <= 0 or not np.isfinite(tr):
                    continue
                else:
                    cov /= tr 
            
            if shrink_flag:
                # Trace-preserving shrinkage
                # Shrinkage regularizes a noisy covariance estimate by blending it with a simple, well-behaved matrix
                #   Reduces numerical noise from short trial windows or ill-conditioned sensors
                #   Keeps total variance (the trace) fixed at 1
                #   Makes matrix inversion and whitening steps less suicidal 
                C = cov.shape[0]
                
                if unit_trace_flag:
                    # Cov matrix is unit trace normalized -> np.trace(cov) ~ 1
                    # Use I/p as target matrix
                    target = np.eye(C) / C
                    cov = (1 - lambda_shrink) * cov + lambda_shrink * target
                else:
                    # Cov matrix is not unit trace normalized -> np.trace(cov) != 1
                    # Use np.diag(np.trace(cov))/p as target matrix
                    target = np.diag([np.trace(cov)] * cov.shape[0]) / cov.shape[0]
                    cov = (1 - lambda_shrink) * cov + lambda_shrink * target
            
            
            if y_label >= 0:
                X_cov.append(cov)
                Y_cov.append(int(y_label))
                X_seg.append(segment.T.astype(np.float64)) # Store segment as (C, T) to be consistent with Whitening / SSA
            
        X_cov = np.stack(X_cov, axis=0) if X_cov else np.empty((0, len(canonical_channels), len(canonical_channels)), dtype=np.float64) # (trials, nChannels*, nChannels*)
        Y_cov = np.array(Y_cov, dtype=np.int64) if Y_cov else np.empty((0,), dtype=np.int64)
        X_seg = np.stack(X_seg, axis=0) if X_seg else np.empty((0, len(canonical_channels), n_samples_win),           dtype=np.float64) # (trials, nChannels*, T)

        X_list.append(X_cov)
        Y_list.append(Y_cov)
        
        try:
            X_seg_list.append(X_seg)
        except NameError:
            X_seg_list = [X_seg]
        
        trial_counts_by_file.update({dataset_label: int(Y_cov.size)})
        
        # Print to terminal the size of the available dataset
        print(f"{dataset_label}: kept {Y_cov.size} labeled trials with {n_samples_win} samples "
            f"({global_window_seconds:.1f} s @ {fs:.1f} Hz)")
        _print_message(message=f"{dataset_label}", which="end")

    # Concatenate across all files
    X_merged     = np.concatenate(X_list, axis=0)     if X_list     else np.empty((0, len(canonical_channels), len(canonical_channels)), dtype=np.float64)
    Y_merged     = np.concatenate(Y_list, axis=0)     if Y_list     else np.empty((0,), dtype=np.int64)
    X_eeg_merged = np.concatenate(X_seg_list, axis=0) if X_seg_list else np.empty((0, len(canonical_channels), n_samples_win), dtype=np.float64)
    
    # Generate group list based on trials' file origin
    groups = []
    for name, count in trial_counts_by_file.items():
        if count <= 0:
            raise ValueError(f"No trials in dataset '{name}'")
        groups.extend([str(name)] * int(count))

    # Final dictionary
    final_dict = {
        "X": X_merged,                                   # (N_total, C, C), float64
        "Y": Y_merged,                                   # (N_total,), int {0,1}
        "X_eeg": X_eeg_merged,                           # (N_total, C, T)
        "ch_names": np.array(canonical_channels),        # (C,)
        "ch_location": canonical_xy,                     # dict[str, [x, y]]
        "trial_counts_by_file": trial_counts_by_file,    # dicts with names and counts
        "groups": groups,                                # list of trial groups 
        "window_seconds": float(global_window_seconds),  # enforced window length (s)
        "band_hz": [float(lowcut), float(highcut)],      # [8.0, 25.0]
        "notes": "Covariance computed over a globally consistent post-cue window for all files",
    }

    print(f"\nMerged dataset shape: X={final_dict['X'].shape}, Y={final_dict['Y'].shape}")
    print("Trial counts by file:", final_dict["trial_counts_by_file"])

    # Export single merged pickle
    result_root = "/mnt/c/Users/scana/Desktop/gpc/data"
    os.makedirs(result_root, exist_ok=True)
    with open(f"{result_root}/{name_of_output}", "wb") as f:
        pickle.dump(final_dict, f)

    # How to import
    # with open("./data/data_set_IVa_merged.pkl", "rb") as f:
    #    my_dict_loaded = pickle.load(f)