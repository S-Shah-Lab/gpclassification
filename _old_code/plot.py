import matplotlib.pyplot as plt
import numpy as np
import gpflow
import tensorflow as tf

from sklearn.metrics import confusion_matrix


def plot_param_evolution(
    w_arr: np.array,
    eta_arr: np.array = None,
    ard_arr: np.array = None,
) -> None:
    """
    Plot evolution of kernel parameters over iterations
    """
    # Plot evolution of the W matrix
    iters, s, nf = w_arr.shape
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for j in range(nf):
        for i in range(s):
            ax[j].plot(range(iters), w_arr[:, i, j], label=f"W[{i},{j}]")
        ax[j].set_title(f"Evolution of W[{j}] entries")
        ax[j].set_xlabel("Iteration")
        ax[j].set_ylabel("Value")
        # ax[j].legend()
    plt.tight_layout()
    plt.show()

    if eta_arr:
        # Plot evolution of the global scaling
        plt.figure(figsize=(5, 4))
        plt.plot(eta_arr)
        plt.title(r"Global scaling $\eta$ over iterations")
        plt.xlabel("Iteration")
        plt.ylabel(r"$\eta$")
        plt.tight_layout()
        plt.show()

    if ard_arr:
        # Plot evolution of the individual filter scaling
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        for k in range(ard_arr.shape[1]):
            ax[k].plot(ard_arr[:, k], label=f"ARD[{k}]")
            ax[k].set_title("ARD parameters over iterations")
            ax[k].set_xlabel("Iteration")
            ax[k].set_ylabel("ARD value")
            ax[k].legend()
        plt.tight_layout()
        plt.show()


def plot_dependent_variable(
    model: gpflow.models.GPR,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
) -> None:
    """
    Plot True vs Predicted Y values for train and test samples in scatterplots side by side
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # Plot the training sample
    ax[0].set_title(f"Training sample: [N = {X_train.shape[0]}]")
    Y_train_pred, var = model.predict_f(X_train)
    ax[0].scatter(
        Y_train.flatten(), Y_train_pred.numpy().flatten(), color="black", marker="o"
    )
    ax[0].set_xlabel("True y")
    ax[0].set_ylabel("Predicted y")
    # Plot the test sample
    ax[1].set_title(f"Test sample: [N = {X_test.shape[0]}]")
    Y_test_pred, var = model.predict_f(X_test)
    ax[1].scatter(
        Y_test.flatten(), Y_test_pred.numpy().flatten(), color="blue", marker="o"
    )
    ax[1].set_xlabel("True y")
    ax[1].set_ylabel("Predicted y")
    #
    lims = np.linspace(
        np.min(
            [
                np.min(Y_train),
                np.min(Y_train_pred.numpy()),
                np.min(Y_test),
                np.min(Y_test_pred.numpy()),
            ]
        ),
        np.max(
            [
                np.max(Y_train),
                np.max(Y_train_pred.numpy()),
                np.max(Y_test),
                np.max(Y_test_pred.numpy()),
            ]
        ),
        100,
    )
    ax[0].plot(lims, lims, "r--", linewidth=1)
    ax[1].plot(lims, lims, "r--", linewidth=1)
    plt.tight_layout()
    plt.show()


def plot_model_and_prediction(
    model: gpflow.models.GPR,
    kernel: gpflow.kernels.Kernel,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
) -> None:
    """
    Plot True vs Predicted Y values for train and test samples in scatterplots side by side
    """

    s = kernel.W.numpy().shape[0]  # Number of sensors
    nf = kernel.W.numpy().shape[1]  # Number of filters

    # Plot Predicted Y values vs Wp @ Sigma @ Wp for train and test samples
    Y_train_pred, var_train = model.predict_f(X_train)
    Y_test_pred, var_test = model.predict_f(X_test)
    std_train = np.sqrt(var_train)
    std_test = np.sqrt(var_test)

    for i in range(nf):

        # Define the feature to use for plotting
        feat_train = (
            kernel.W.numpy()[:, i]
            @ X_train.reshape(len(X_train), s, s)
            @ kernel.W.numpy()[:, i]
        )
        feat_test = (
            kernel.W.numpy()[:, i]
            @ X_test.reshape(len(X_test), s, s)
            @ kernel.W.numpy()[:, i]
        )

        # Sorting index for training sample
        idx_train = np.argsort(feat_train)
        # Sorting index for test sample
        idx_test = np.argsort(feat_test)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # Plot training sample
        ax[0].scatter(
            feat_train,
            Y_train.flatten(),
            color="black",
            marker="x",
            label="Training point",
        )

        ax[0].plot(
            feat_train[idx_train],
            Y_train_pred.numpy().flatten()[idx_train],
            color="red",
            label="Mean (predicted)",
        )
        ax[0].fill_between(
            feat_train[idx_train],
            (Y_train_pred.numpy() - 2 * std_train).flatten()[idx_train],
            (Y_train_pred.numpy() - 2 * std_train).flatten()[idx_train],
            alpha=0.3,
            label=r"2$\sigma$ CI",
            color="orange",
        )
        ax[0].set_title(f"Training sample: [N = {X_train.shape[0]}]")
        ax[0].set_xlabel(f"Feature {i+1}")
        ax[0].set_ylabel(r"log($\alpha$ power)")
        ax[0].legend()
        # Plot test sample
        ax[1].scatter(
            feat_test,
            Y_test.flatten(),
            color="black",
            marker="x",
            label="Training point",
        )
        ax[1].plot(
            feat_test[idx_test],
            Y_test_pred.numpy().flatten()[idx_test],
            color="blue",
            label="Mean (predicted)",
        )
        ax[1].fill_between(
            feat_test[idx_test],
            (Y_test_pred.numpy() - 2 * std_test).flatten()[idx_test],
            (Y_test_pred.numpy() - 2 * std_test).flatten()[idx_test],
            alpha=0.3,
            label=r"2$\sigma$ CI",
            color="royalblue",
        )
        ax[1].set_title(f"Test sample: [N = {X_test.shape[0]}]")
        ax[1].set_xlabel(f"Feature {i+1}")
        ax[1].set_ylabel(r"log($\alpha$ power)")
        ax[1].legend()
        #
        plt.tight_layout()
        plt.show()


def plot_cv_metrics(stats, metrics, domains):
    """
    Plot cross validation metrics
    """
    plt.figure()
    for metric in metrics:
        for domain in domains:
            # Extract metrics
            maxiters = [d["maxiter"] for d in stats]
            means = [np.mean(d[f"{domain}_{metric}"]) for d in stats]
            stds = [np.std(d[f"{domain}_{metric}"]) for d in stats]
            # Plot with error bars
            plt.errorbar(
                maxiters,
                means,
                yerr=stds,
                fmt="o-",
                capsize=5,
                label=f"{domain} {metric}",
            )

    plt.xlabel("Parameter: maxiter")
    plt.ylabel("CV Metric")
    plt.legend()
    plt.tight_layout()
    plt.show()


def _compute_features(Sigma, W, logged_flag=True):
    """
    Plot features at a given state of the parameters W
    which corresponds to a certain iteration while the parameters are evolving

    This function doesn't currently take into account global scaling or filter specific scaling
    """
    Sw = tf.matmul(Sigma, W)  # [N, s, nf]
    # Applies w @ Σi @ w for each i with i being trial number
    # Sw has shape [N, s, nf]
    # W[None, :, :] has shape [1, s, nf]
    wSw = tf.reduce_sum(W[None, :, :] * Sw, axis=1)  # [N, nf]
    if logged_flag:
        # Log the resulting features
        wSw = tf.math.log(wSw + 1e-7)
    return wSw


def plot_feature_evolution(X, Y, w_arr, maxiter, label, nPlots=5, logged_flag=True):
    """
    Plot features evolution across iterations in a grid.
    """
    # Compute features at each iteration
    feat_over_iterations = []
    for w_i in w_arr:
        feat_over_iterations.append(
            _compute_features(X.reshape(X.shape[0], w_i.shape[0], w_i.shape[0]), w_i)
        )
    feat_over_iterations = np.stack(
        feat_over_iterations
    )  # shape [maxiter, N, nf], could or not be logged

    # Determine iterations to plot
    if nPlots > maxiter:
        idx = list(range(maxiter))
    elif nPlots == 1:
        idx = [0]
    else:
        idx = [(j * maxiter) // (nPlots - 1) for j in range(nPlots - 1)] + [maxiter - 1]

    # Color mapping according to a binary class system (0 and 1) which is grabbed from the Y variable
    if len(np.unique(Y.flatten())) == 2:
        true_label_colors = ["red" if y == 0 else "blue" for y in Y.flatten()]
    else:
        true_label_colors = "black"

    # Generte a grid layout
    nCols = int(np.ceil(np.sqrt(len(idx))))
    nRows = int(np.ceil(len(idx) / nCols))
    fig, axes = plt.subplots(nRows, nCols, figsize=(nCols * 4, nRows * 4))
    axes = axes.flatten()
    fig.suptitle(label, fontsize=13)

    # Plot each selected iteration according to iterations to plot
    for ax_idx, (ax, iter_idx) in enumerate(zip(axes, idx)):
        feats = feat_over_iterations[iter_idx]
        ax.scatter(feats[:, 0], feats[:, 1], color=true_label_colors)
        ax.set_title(f"Iteration {iter_idx}")

        # Only y-axis labels on first column
        col = ax_idx % nCols
        if col == 0:
            ax.set_ylabel("Feature 2")
        else:
            ax.set_yticklabels([])
            ax.set_ylabel("")

        # Only x-axis labels on bottom row
        row = ax_idx // nCols
        if row == nRows - 1:
            ax.set_xlabel("Feature 1")
        else:
            ax.set_xticklabels([])
            ax.set_xlabel("")

    # Turn off axis when redundant
    for ax in axes[len(idx) :]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    return fig


def plot_confusion_evolution(
    y_true,
    y_pred_arr,
    maxiter,
    nPlots=5,
    labels=None,
    normalize=False,
    cmap=plt.cm.viridis,
):
    """
    Plot confusion matrices for selected iterations in a grid.
    """
    # Determine iterations to plot
    if nPlots > maxiter:
        idx = list(range(maxiter))
    elif nPlots == 1:
        idx = [0]
    else:
        idx = [(j * maxiter) // (nPlots - 1) for j in range(nPlots - 1)] + [maxiter - 1]

    # Generate grid layout
    nCols = int(np.ceil(np.sqrt(len(idx))))
    nRows = int(np.ceil(len(idx) / nCols))
    fig, axes = plt.subplots(nRows, nCols, figsize=(nCols * 4, nRows * 4))

    # Ensure axes is always a flat list
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    # Pre-compute all selected confusion matrices to find global vmin/vmax
    cms = []
    for iter_idx in idx:
        cm = confusion_matrix(
            y_true,
            y_pred_arr[iter_idx],
            labels=labels,
            normalize="true" if normalize else None,
        )
        cms.append(cm)
    # Find global min/max across all matrices
    global_min = min(cm.min() for cm in cms)
    global_max = max(cm.max() for cm in cms)

    # Plot each confusion matrix
    for ax_idx, (ax, cm, iter_idx) in enumerate(zip(axes, cms, idx)):
        im = ax.imshow(
            cm, interpolation="nearest", cmap=cmap, vmin=global_min, vmax=global_max
        )
        fmt = ".2f" if normalize else "d"
        thresh = (global_max + global_min) / 2.0
        # Annotate cells
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
        ax.set_title(f"Iteration {iter_idx}")
        # Tick labels
        ax.set_xticks(np.arange(cm.shape[1]))
        ax.set_yticks(np.arange(cm.shape[0]))
        cls_labels = (
            labels
            if labels is not None
            else np.unique(np.concatenate((y_true, y_pred_arr[iter_idx])))
        )
        ax.set_xticklabels(cls_labels)
        ax.set_yticklabels(cls_labels)

        # only show y-labels on first col
        col = ax_idx % nCols
        if col != 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("True label")

        # only show x-labels on bottom row
        row = ax_idx // nCols
        if row == nRows - 1:
            ax.set_xlabel("Predicted label")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        else:
            ax.set_xticklabels([])

    # Turn off axis when redundant
    for ax in axes[len(cms) :]:
        ax.axis("off")

    # Add a single colorbar next to last plot
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(
        im,
        cax=cbar_ax,
        orientation="vertical",
        label="Proportion" if normalize else "Count",
    )

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # leave space for colorbar
    plt.show()

    return fig
