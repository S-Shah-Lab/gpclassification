"""
Load annotated EEG from .fif, band-pass to 8-25 Hz, epoch by annotation labels, convert each epoch to a covariance matrix, and export train/test splits with metadata.

Description:
  - Reads an MNE Raw object from a .fif file (preload=True)
  - Applies band-pass filtering (8-25 Hz) and interpolates previously marked bad channels (resetting RAW.info['bads'])
  - Extracts events from annotations and epochs data for labels: left_{num}, right_{num}, where num ∈ {1..8} using tmin=0, tmax=1.5, baseline=None
  - Converts each time-domain epoch (channels x samples) to a covariance matrix (channels x channels) via np.cov
  - Concatenates left/right epochs, builds class labels {0: left, 1: right}, and splits using trial-number partitions: train_labels = [1, 3, 4, 7], test_labels = [2, 5, 6, 8]
  - Shuffles train and test sets independently and writes a pickle
  
Outputs:
  ./data/{dataset_label}.pkl containing:
    {
      "X": {"train": (N_tr, C, C), "test": (N_te, C, C)},
      "Y": {"train": (N_tr,),      "test": (N_te,)     },
      "ch_names": list[str],  montage channel order
      "train_labels": list[int],
      "test_labels": list[int],
      "dataset_label": str
    }

Assumptions:
  - Annotation keys exist for every requested f-string label (left_{num}/right_{num})
  - RAW has already been globally preprocessed outside this script as noted in comments
  - All epochs are EEG-only picks; no ICA components or MEG misc channels included

Notes:
  - `mne.Epochs(..., preload=True)` returns data as (n_epochs, n_channels, n_times)._
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
import pickle


def import_file_fif(path=None):
    """
    Imports EEG data from a .fif file using MNE-Python.

    Args:
        path (str, optional): The directory path where the .fif file is located.
                              If None, current working directory is assumed
        file_name (str): The name of the .fif file to be imported.

    Returns:
        tuple: A tuple containing the raw data object (`RAW`), the electrode montage (`montage`), and the sampling frequency (`fs`) extracted from the file.
    """
    # Read the .fif file into a RAW object with data preloaded
    RAW = mne.io.read_raw(path, preload=True)
    # Retrieve the montage used in the recording from the RAW object
    montage = RAW.get_montage()
    # Extract the sampling frequency from the RAW object's information
    fs = RAW.info["sfreq"]
    return RAW, montage, fs


# Import previously annotated mne.io.Raw object
# The RAW object has already been filtered between 1-40 Hz
# The RAW object has already been re-referenced to the avearge of the mastoids "tp9 tp10"
path_to_files = [
    # "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDHC001_ses-01_task-MotorImag_run-01.fif",
    # "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDHC002_ses-01_task-MotorImag_run-01.fif",
    # "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDHC002_ses-01_task-MotorImag_run-02.fif",
    # "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDHC003_ses-01_task-MotorImag_run-01.fif",
    # "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDHC004_ses-01_task-MotorImag_run-01.fif",
    # "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDHC005_ses-01_task-MotorImag_run-01.fif",
    # "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDHC006_ses-01_task-MotorImag_run-01.fif",
    # "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDHC007_ses-01_task-MotorImag_run-01.fif",
    # "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDHC008_ses-01_task-MotorImag_run-01.fif",
    # "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDHC009_ses-01_task-MotorImag_run-01.fif",
    # "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDHC010_ses-01_task-MotorImag_run-01.fif",
    # "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDHC011_ses-01_task-MotorImag_run-01.fif",
    # "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDHC012_ses-01_task-MotorImag_run-01.fif",
    # "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDHC013_ses-01_task-MotorImag_run-01.fif",
    # "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDHC014_ses-01_task-MotorImag_run-01.fif",
    # "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDHC015_ses-01_task-MotorImag_run-01.fif",
    # "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDHC016_ses-01_task-MotorImag_run-01.fif",
    # "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDHC017_ses-01_task-MotorImag_run-01.fif",
    # "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDHC018_ses-01_task-MotorImag_run-01.fif",
    # "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDHC019_ses-01_task-MotorImag_run-01.fif",
    # "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDHC020_ses-01_task-MotorImag_run-01.fif",
    # "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDHC021_ses-01_task-MotorImag_run-01.fif",
    # "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDHC022_ses-01_task-MotorImag_run-01.fif",
    # "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDHC023_ses-01_task-MotorImag_run-01.fif",
    # "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDHC024_ses-01_task-MotorImag_run-01.fif",
    # "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDHC025_ses-01_task-MotorImag_run-01.fif",
    # "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDHC026_ses-01_task-MotorImag_run-01.fif",
    "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDHC027_ses-01_task-MotorImag_run-01.fif",
]

fmin = 8
fmax = 25

train_labels = [1, 3, 4, 7]
test_labels = [2, 5, 6, 8]

all_labels = [1, 2, 3, 4, 5, 6, 7, 8]

for path_to_file in path_to_files:
    dataset_label = path_to_file.split("/")[-1].split(".")[0]

    RAW, montage, fs = import_file_fif(path=path_to_file)

    # Bandpass filter the data between
    RAW = RAW.filter(fmin, fmax)

    # Extract annotations from RAW
    events_from_annot, event_dict = mne.events_from_annotations(RAW)
    # fig = mne.viz.plot_events(
    #     events_from_annot, sfreq=fs, first_samp=RAW.first_samp, event_id=event_dict
    # )
    # fig.subplots_adjust(right=0.7)
    # plt.show()

    # RAW was saved without interpolation, RAW.info['bads'] contains the prevously identified bad channels, if any
    # Perform interpolation, with the option to reset the list of bad channels
    RAW.interpolate_bads(reset_bads=True)

    # Create epochs based on annotations
    epochs_left = []
    epochs_left_labels = []

    epochs_right = []
    epochs_right_labels = []

    for num in all_labels:
        for type_ in ["left", "right"]:
            # Generate epochs using annotations
            epochs_ = mne.Epochs(
                RAW,
                events_from_annot,
                event_id=event_dict[f"{type_}_{num}"],
                tmin=0,
                tmax=1.5,
                baseline=None,
                preload=True,
                verbose=False,
            ).get_data(
                picks="eeg"
            )  # shape (N_{num}, s, time_samples)

            if epochs_.shape[0] == 0:
                # No epochs for this trial number
                continue
            else:
                # Convert time series epochs to cov matrix epochs
                cov_epochs_ = []
                for i in range(epochs_.shape[0]):
                    cov_matrix = np.cov(epochs_[i, :, :])  # shape (s, s)
                    cov_epochs_.append(cov_matrix)
                cov_epochs_ = np.array(cov_epochs_)  # shape (N_{num}, s, s)

                # Store in appropriate container
                if type_ == "left":
                    epochs_left.append(cov_epochs_)  # shape (8, N_{num}, s, s)
                    epochs_left_labels.append(
                        [num] * len(cov_epochs_)
                    )  # Count and keep track where they come from
                elif type_ == "right":
                    epochs_right.append(cov_epochs_)
                    epochs_right_labels.append(
                        [num] * len(cov_epochs_)
                    )  # Count and keep track where they come from

    epochs_left = np.concatenate(epochs_left)  # shape (N_left, s, s)
    epochs_left_labels = np.concatenate(epochs_left_labels)  # shape (N_left,)
    N_left = epochs_left.shape[0]

    epochs_right = np.concatenate(epochs_right)  # shape (N_right, s, s)
    epochs_right_labels = np.concatenate(epochs_right_labels)  # shape (N_right,)
    N_right = epochs_right.shape[0]

    # Create array of cov matrices to be split into train and test
    X = np.concatenate((epochs_left, epochs_right))  # shape (N_left + N_right, s, s)
    X_labels = np.concatenate(
        (epochs_left_labels, epochs_right_labels)
    )  # shape (N_left + N_right,)
    # Create array of classification labels: 0 -> Left, 1 -> Right
    Y = np.concatenate((np.zeros(N_left), np.ones(N_right)))

    # Define slicing according to trial number using `train_labels` and `test_labels`
    slice_train = [True if label in train_labels else False for label in X_labels]
    slice_test = [True if label in test_labels else False for label in X_labels]

    # Split into train and test
    X_train = X[slice_train]
    Y_train = Y[slice_train]

    X_test = X[slice_test]
    Y_test = Y[slice_test]

    # Shuffle the sets together
    perm = np.random.permutation(len(Y_train))
    X_train = X_train[perm]
    Y_train = Y_train[perm]

    perm = np.random.permutation(len(Y_test))
    X_test = X_test[perm]
    Y_test = Y_test[perm]

    # Channel names sequence
    ch_names = montage.ch_names

    # Generate dictionary to be exported
    my_dict = {
        "X": {
            "train": X_train,
            "test": X_test,
        },
        "Y": {
            "train": Y_train,
            "test": Y_test,
        },
        "ch_names": ch_names,
        "train_labels": train_labels,
        "test_labels": test_labels,
        "dataset_label": dataset_label,
    }

    # To export
    with open(f"./data/{dataset_label}.pkl", "wb") as f:
        pickle.dump(my_dict, f)

    # To import
    # with open("my_dict.pkl", "rb") as f:
    #    my_dict_loaded = pickle.load(f)
