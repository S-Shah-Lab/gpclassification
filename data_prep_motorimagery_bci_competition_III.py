"""
Load BCI Competition IVa .mat files, band-pass EEG to 8-25 Hz, extract a common post-cue window per trial, compute covariance features, and serialize dataset artifacts for downstream models

Description:
  - Reads subject files from: /mnt/c/Users/scana/Desktop/jeremy_work/{dataset_label}_mat/1000Hz/{dataset_label}.mat
  - Uses all available channels (lowercased), scales coordinates radially by 1.2, and converts samples to microvolts (x0.1)
  - Applies per-channel zero-phase Butterworth band-pass (order=4, 8-25 Hz) using SOS design and sosfiltfilt
  - Derives a mutual trial window by: drop first 1.0 s after cue onset; assume 2.25 s max relax at end; take the minimum remaining duration across trials
  - Computes per-trial sample covariance (C x C) over the common window
  - Maps labels from {1, 2} to {0, 1}; ignores unlabeled trials (Y < 0)
  - Writes a pickle with features and metadata for each dataset_label

Outputs:
  ./data/{dataset_label}.pkl containing:
    {
      "X": np.ndarray, shape (n_labeled_trials, C, C), dtype float32
      "Y": np.ndarray, shape (n_labeled_trials,), values {0, 1}
      "ch_names": np.ndarray[str], length C
      "ch_location": dict[str, [float x, float y]]  scaled by 1.2
      "dataset_label": str
    }

Usage:
  - Set `dataset_labels` as needed
  - Run the script; one {dataset_label}.pkl per subject will be written to ./data/

Notes:
  - Filtering is zero-phase but reflects at boundaries; windows start well after cue to reduce artifacts
  - Keep `fs` explicit in butter(...) when changing band edges
  - If you later split train/test, re-enable the commented train_test_split block
"""

import pickle
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split


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
    Returns a new float32 array with the same shape as X
    """
    # Design in SOS form with explicit sampling freq (no pre-normalization needed)
    sos = butter(order, [lowcut, highcut], btype="band", fs=fs, output="sos")

    Xf = np.empty_like(X)  # same dtype/shape
    # Filter one channel at a time to cap memory usage
    for i in range(X.shape[1]):
        Xf[:, i] = sosfiltfilt(sos, X[:, i])
    return Xf


# Load .mat file
# dataset_label = "data_set_IVa_aa"
# dataset_label = "data_set_IVa_al"
# dataset_label = "data_set_IVa_av"
# dataset_label = "data_set_IVa_aw"
# dataset_label = "data_set_IVa_ay"

dataset_labels = [
    "data_set_IVa_aa",
    "data_set_IVa_al",
    "data_set_IVa_av",
    "data_set_IVa_aw",
    "data_set_IVa_ay",
]

for dataset_label in dataset_labels:

    # Print message on terminal
    _print_message(message=f"{dataset_label}", which="start")

    data = loadmat(
        f"/mnt/c/Users/scana/Desktop/jeremy_work/{dataset_label}_mat/1000Hz/{dataset_label}.mat"
    )

    file_name = str(data["nfo"][0][0][0][0])
    fs = float(data["nfo"][0][0][1][0][0])  # sampling freq

    # Grab channel namesa and x,y coordinates
    ch_names_all = [
        str(data["nfo"][0][0][2][0][ch][0]).lower()
        for ch in range(data["nfo"][0][0][2][0].shape[0])
    ]  # lower case names
    ch_x_all = data["nfo"][0][0][3].flatten()
    ch_y_all = data["nfo"][0][0][4].flatten()

    # Apply radial scaling of 1.2
    ch_x_all = ch_x_all * 1.2
    ch_y_all = ch_y_all * 1.2

    """
    # Create a list of channels to use among the ones in the file, nChannels -> nChannels*
    ch_to_use = [
        "fp1",
        "fpz",
        "fp2",
        "af7",
        "af3",
        "af4",
        "af8",
        "f7",
        "f5",
        "f3",
        "f1",
        "fz",
        "f2",
        "f4",
        "f6",
        "f8",
        "ft7",
        "fc5",
        "fc3",
        "fc1",
        "fcz",
        "fc2",
        "fc4",
        "fc6",
        "ft8",
        "t7",
        "c5",
        "c3",
        "c1",
        "cz",
        "c2",
        "c4",
        "c6",
        "t8",
        "tp7",
        "cp5",
        "cp3",
        "cp1",
        "cpz",
        "cp2",
        "cp4",
        "cp6",
        "tp8",
        "p7",
        "p5",
        "p3",
        "p1",
        "pz",
        "p2",
        "p4",
        "p6",
        "p8",
        "po7",
        "po3",
        "poz",
        "po4",
        "po8",
        "o1",
        "oz",
        "o2",
    ]
    """

    # Use all channels in this case
    ch_to_use = ch_names_all

    # Grab indices of channels to use
    idx = [ch_names_all.index(ch) for ch in ch_to_use]

    # Channels to use and thier coordinate, build a dict
    ch_names = np.array([ch_names_all[i] for i in idx])
    ch_x = ch_x_all[idx]
    ch_y = ch_y_all[idx]
    ch_location = {
        ch: [float(ch_x[i]), float(ch_y[i])] for i, ch in enumerate(ch_names)
    }

    # Select channels to use as type float32
    X = data["cnt"][:, idx].astype(np.float32, copy=False)  # (nSamples, nChannels*)
    X *= 0.1  # scale to uV according to instructions

    # Bandpass filter 8-25 Hz selected channels
    lowcut, highcut = 8.0, 25.0
    X_filtered = bandpass_filter_channelwise(X, lowcut, highcut, fs, order=4)

    # Class labels and trial onset timing
    Y_ = data["mrk"][0][0][1].flatten()  # (nTrials,)
    Y = np.where(Y_ == 1, 0, np.where(Y_ == 2, 1, Y_))  # convert Y labels to 0 or 1

    cue_onsets = data["mrk"][0][0][0].flatten()  # (nTrials,)

    # Compute window length to be used for all trials
    # Instruction given:
    # - at the start: 3.5 s of visual cue are given
    # - at the end  : 1.75-2.25 s random relax time
    # Idea on how to proceed:
    # At each cue onset: remove first 1 s cause it could contain garbage
    # At end of trial  : remove last 2.25 s assuming all trials had the longest relax period
    # For each trial   : check how long is left, use the mutual min timing
    initial_time_to_remove = 1.0  # s to skip after cue
    final_relax_time = 2.25  # s to skip at the end

    time_windows = []
    deltas = []
    for t1, t2 in zip(cue_onsets[:-1], cue_onsets[1:]):
        delta_t = (t2 - t1) / fs  # time in seconds
        deltas.append(delta_t)
        t_avail = delta_t - initial_time_to_remove  # remove first second
        t_min = t_avail - final_relax_time  # assume max relax at end
        time_windows.append(t_min)
    time_windows = np.array(time_windows, dtype=np.float32)
    min_time_available = float(
        np.round(np.min(time_windows), 1)
    )  # mutual minimum time for each trial

    print(f"Delta time between cue onsets: [{np.min(deltas)}, {np.max(deltas)}]")
    print(f"Min time: {min_time_available} s")

    # Compute covariance matrix per trial
    X_cov = []
    n_samples_win = int(min_time_available * fs)
    offset = int(initial_time_to_remove * fs)

    for cue_t in cue_onsets:
        t_start = int(cue_t) + offset
        t_end = t_start + n_samples_win
        segment = X_filtered[t_start:t_end, :]  # shape (T, nChannels*)
        cov = np.cov(
            segment, rowvar=False, bias=False
        )  # shape (nChannels*, nChannels*)
        X_cov.append(cov.astype(np.float32))
    X_cov = np.stack(X_cov, axis=0)  # (trials, nChannels*, nChannels*)

    # Split into labelled and unlabelled data
    idx_labelled = Y >= 0
    X_labelled = X_cov[idx_labelled]
    Y_labelled = Y[idx_labelled]
    X_not_labelled = X_cov[~idx_labelled]
    Y_not_labelled = Y[~idx_labelled]

    # In this case it's better to just have all trials together as they are independent
    # X_train, X_test, Y_train, Y_test = train_test_split(
    #    X_labelled, Y_labelled, test_size=0.5, random_state=42
    # )

    # Generate dictionary to be exported
    my_dict = {
        "X": X_labelled,
        "Y": Y_labelled,
        "ch_names": ch_names,
        "ch_location": ch_location,
        "dataset_label": dataset_label,
    }

    # Print to terminal the size of the available dataset
    print(f"X (shape): {my_dict['X'].shape}")
    _print_message(message=f"{dataset_label}", which="end")

    # To export
    with open(f"./data/{dataset_label}.pkl", "wb") as f:
        pickle.dump(my_dict, f)

    # To import
    # with open("./data/{dataset_label}.pkl", "rb") as f:
    #    my_dict_loaded = pickle.load(f)
