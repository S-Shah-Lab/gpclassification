import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import mne
from mne.channels import read_dig_fif


from train import train_model
from cv import cross_validate
from plot import (
    plot_param_evolution,
    plot_dependent_variable,
    plot_model_and_prediction,
    plot_cv_metrics,
    plot_feature_evolution,
    plot_confusion_evolution,
)

# Import montage for topomaps
montage = read_dig_fif(
    "/mnt/c/Users/scana/Dropbox/WCornell/develop/gpclassification/montage_egi_64.fif"
)
info = mne.create_info(ch_names=montage.ch_names, sfreq=1000.0, ch_types="eeg")
info.set_montage(montage)

left_id = []
for ch in ["fc1", "c1", "cp1", "fc3", "c3", "fc5", "c5", "cp5"]:
    left_id.append(montage.ch_names.index(ch))

right_id = []
for ch in ["fc2", "c2", "cp2", "fc4", "c4", "fc6", "c6", "cp6"]:
    right_id.append(montage.ch_names.index(ch))

############################################################################# TOY SAMPLE
# Load input data: X, Y
# X = np.load("./data/classification_X.npy")  # shape (N, s, s)
# Y = np.load("./data/classification_Y.npy")  # This is a binary class

############################################################################# REAL MOTOR IMAGERY SAMPLE
# Load input data: X, Y
X = np.load("./data/motorimagery_X.npy")  # shape (N, s, s)
Y = np.load("./data/motorimagery_Y.npy")  # This is a binary class
dataset_label = "sub-PDHC002_ses-01_run-01"
# X = X * 1e14

# Show the input covariance matrix
"""
fig, ax = plt.subplots(1,1)
im = ax.imshow(X[0, :, :])
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
fig.colorbar(
        im,
        cax=cbar_ax,
        orientation="vertical"
    )
"""

# Determine parameters from input arrays
N = X.shape[0]
s = X.shape[1]
nf = 2


# Flatten the covariance matrices in input
# This is done to make GPflow happy as it expect X to have shape [N, D]
X_flat = X.reshape(N, s * s)
Y = Y.reshape(len(Y), 1)


# Determine model kernel parameters
eta_flag = False
ard_flag = False
logged_flag = True
kernel_type = "RBF"
# Generate initial spatial filtering matrix
# Randomize the input weights
weights_flag = "random"

np.random.seed(2)
if weights_flag == "random":
    W_init = np.random.normal(0, 0.1, (s, nf))
elif weights_flag == "ones":
    W_init = np.ones((s, nf))
# OR
# Set a specific spatial filter matrix
# Define index of channels C3 and C4, this is based on montage used, in this case EGI 64
else:
    W_init = np.zeros((s, nf))  # Generate zero weights
    # Initialize first column to 1 for electrode C3, and second column to 1 for electrode C4
    for idx in left_id:
        W_init[idx, 0] = 1  # first col
    for idx in right_id:
        W_init[idx, 1] = 1  # second col


# Define split train / test
# This split breaks train sample from test sample
# Test sample is unique and always unseen until the very last moment
# Train sample is further broken down into train and validation according to the cross validation folds
frac_train = 0.5
frac_test = 1 - frac_train

X_train, X_test, Y_train, Y_test = train_test_split(
    X_flat,
    Y,
    test_size=frac_test,
    random_state=42,
    shuffle=True,
    stratify=None,
)


# Set up cross validation over multiple values of maxiter
# maxiter_options = [10, 25, 50, 100, 250, 500, 750, 1000]
# maxiter_options = [50, 100]
maxiter_options = [500]

best_iters = maxiter_options[-1]  # This is going to be updated anyway
model_selection = "accuracy"
if model_selection == "accuracy":
    best_val = 0
else:
    best_val = np.inf

# Initialize empty container for each maxiter tested
maxiter_stats = []

"""
for it in maxiter_options:
    print(f"Now working on maxiter: {it}")

    # Perform cross validation for each value of maxiter
    cv_stats = cross_validate(
        X_train,
        Y_train,
        W_init,
        ard_flag,
        eta_flag,
        logged_flag,
        kernel_type,
        maxiter=it,
        cv_splits=5,
    )

    # Store results (dict) for each maxiter
    maxiter_stats.append(
        {
            "maxiter": it,
            "train_elbo": [
                cv_stats[i]["train_elbo"] for i in range(len(cv_stats))
            ],  # evidence lower bound train
            "train_accuracy": [
                cv_stats[i]["train_accuracy"] for i in range(len(cv_stats))
            ],  # prediction accuracy train
            "train_logloss": [
                cv_stats[i]["train_logloss"] for i in range(len(cv_stats))
            ],  # cross entropy train
            "val_nlpd": [
                cv_stats[i]["val_nlpd"] for i in range(len(cv_stats))
            ],  # negative log-predictive density validation
            "val_accuracy": [
                cv_stats[i]["val_accuracy"] for i in range(len(cv_stats))
            ],  # prediction accuracy validation
            "val_logloss": [
                cv_stats[i]["val_logloss"] for i in range(len(cv_stats))
            ],  # cross entropy validation
        }
    )

    # Model selection
    # Check if the obtained negative log-predictive density validation is better than the one previously stored as best
    # Lower is better
    # if np.mean(maxiter_stats[-1]["val_nlpd"]) < best_val:
    #    best_val, best_iters = np.mean(maxiter_stats[-1]["val_nlpd"]), it

    # Check if the obtained cross entropy validation is better than the one previously stored as best
    # Lower is better
    # if np.mean(maxiter_stats[-1]["val_logloss"]) < best_val:
    #    best_val, best_iters = np.mean(maxiter_stats[-1]["val_logloss"]), it

    # Check if the obtained prediction accuracy validation is better than the one previously stored as best
    # Higher is better
    if np.mean(maxiter_stats[-1]["val_accuracy"]) > best_val:
        best_val, best_iters = np.mean(maxiter_stats[-1]["val_accuracy"]), it


plot_cv_metrics(maxiter_stats, metrics=["accuracy"], domains=["train", "val"])
plot_cv_metrics(maxiter_stats, metrics=["logloss"], domains=["train", "val"])
# plot_cv_metrics(maxiter_stats, metrics=["elbo"], domains=["train", "val"]) # There is no validation elbo


print(f"Optimal maxiter: {best_iters}")
"""

# Train final model with maxiter learned from cross validation
model, kernel, y_train_pred, w_arr, eta_arr, ard_arr = train_model(
    X_train,
    Y_train,
    W_init,
    ard_flag,
    eta_flag,
    logged_flag,
    kernel_type,
    maxiter=best_iters,
)


figs = plot_feature_evolution(
    X_train,
    Y_train,
    w_arr,
    best_iters,
    label=dataset_label + " Train",
    nPlots=10,
    logged_flag=logged_flag,
)
figs = plot_confusion_evolution(
    Y_train.flatten(), y_train_pred, best_iters, nPlots=10, labels=None, normalize=False
)

""""
figs = plot_feature_evolution(
    X_train, Y_train, w_arr, 10, label="Train", nPlots=10, logged_flag=logged_flag
)
figs = plot_confusion_evolution(
    Y_train.flatten(), y_train_pred, 10, nPlots=10, labels=None, normalize=False
)

"""

figs = plot_feature_evolution(
    X_test,
    Y_test,
    w_arr,
    best_iters,
    label=dataset_label + " Test",
    nPlots=10,
    logged_flag=logged_flag,
)
# figs = plot_feature_evolution(
#    X_test, Y_test, w_arr, best_iters, nPlots=1, logged_flag=logged_flag
# )

# Predict using the test set, show confusion matrix
probs, _ = model.predict_y(X_test)  # predictive P(y=1|x)
y_test_pred = (probs.numpy() >= 0.5).astype(int)
figs = plot_confusion_evolution(
    Y_test.flatten(),
    y_test_pred.reshape(1, y_test_pred.shape[0]),
    1,
    nPlots=1,
    labels=None,
    normalize=False,
)


# Plot final model predictions
if len(eta_arr) == 0:
    eta_arr = None
if len(ard_arr) == 0:
    ard_arr = None
plot_param_evolution(w_arr, eta_arr, ard_arr)
# plot_dependent_variable(model, X_train, Y_train, X_test, Y_test)
# plot_model_and_prediction(model, kernel, X_train, Y_train, X_test, Y_test)


fig, ax = plt.subplots(1, 2)
im, _ = mne.viz.plot_topomap(data=W_init[:, 0], pos=info, axes=ax[0], show=False)
im, _ = mne.viz.plot_topomap(data=W_init[:, 1], pos=info, axes=ax[1], show=False)
# fig.colorbar(im, ax=ax)
# ax.set_title("Channel Weights Topomap")
plt.show()


nPlots = 10
for i in [(j * best_iters) // (nPlots - 1) for j in range(nPlots - 1)] + [
    best_iters - 1
]:
    fig, ax = plt.subplots(1, 2)
    im, _ = mne.viz.plot_topomap(data=w_arr[i, :, 0], pos=info, axes=ax[0], show=False)
    im, _ = mne.viz.plot_topomap(data=w_arr[i, :, 1], pos=info, axes=ax[1], show=False)
    # fig.colorbar(im, ax=ax)
    # ax.set_title("Channel Weights Topomap")
    plt.show()
