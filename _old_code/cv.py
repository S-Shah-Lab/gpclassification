import gpflow
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, accuracy_score
from train import train_model
from numpy.linalg import cholesky, solve


def calculate_nlml(
    model: gpflow.models.GPR, kernel: gpflow.kernels.Kernel, X: np.array, Y: np.array
) -> float:
    """
    Manually calculate the negative log-marginal-likelihood
    There are 3 terms that need to be evaluated:
        The "data fit" ->  1/2 Y.T @ Ky ^-1 @ Y
            requires to invert the covariance of Y or similar (similar is the trick here)
        The "complexity penalty" -> 1/2 log-determinant(Ky)
            requires the log-determinant of the covariance Y
        The "normalization" -> N/2 log(2pi)
            simple additive term
    """
    # Determine the number of samples form the input X
    N = X.shape[0]

    # Normalization term
    norm_term = 0.5 * N * np.log(2 * np.pi)

    # Evaluate the covariance of Y under the model in the following steps:
    # EXtrainact the noise variance from the model likelihood variance (sigma^2)
    sigma2 = float(model.likelihood.variance.numpy())
    # Compute the kernel using the given input X
    K = kernel.K(X=X).numpy()  # shape [N,N]
    # Add the kernel and the noise variance along the diagonal (using the identity matrix)
    Ky = K + sigma2 * np.eye(N)

    # Invert the covariance of Y to obtain Ky ^-1 (use Cholesky method for stability)
    # Cholesky decompose a matrix as follows: A = L * L.T
    L = cholesky(Ky)  # eXtrainact lower-triangular

    # We can use the following trick
    # log[det(Ky)] = log[det(L * L.T)] = log[det(L) * det(L.T)] = log[det(L)*det(L)] = 2 * log[det(L)]
    # But L is lower-triangular so det(L) = product elements along diagonal
    # log[det(Ky)] = 2 * log[ product along diagonal ] = 2 * sum of log(Lii)
    # Complexity term
    complex_term = float(0.5 * 2 * np.sum(np.log(np.diag(L))))

    # We need to eXtrainact Y.T @ Ky ^-1 @ Y
    # Using Cholesky decomp and matrix L we can evaluate Ky ^-1 @ Y easily
    # Solve system of linear equations: L v = Y to obtain v
    v = solve(L, Y)
    # Now, given v, solve system of linear equations: L.T alpha = v
    # L.T alpha = v
    # L L.T alpha = L v
    # K alpha = Y
    # K^-1 K alpha = K^-1 Y
    # alpha = K^-1 Y
    alpha = solve(L.T, v)
    # Data fit term
    data_fit_term = 0.5 * float(Y.T @ alpha)
    # Putting all together
    nlml = data_fit_term + complex_term + norm_term
    return nlml


def cross_validate(
    X: np.ndarray,
    Y: np.ndarray,
    W_init: np.ndarray,
    ard_flag: bool = False,
    eta_flag: bool = False,
    logged_flag: bool = True,
    kernel_type: str = "RBF",
    maxiter: int = 1000,
    cv_splits: int = 5,
    random_state: int = 42,
) -> list[dict]:
    """
    Perform K-fold cross-validation, returning train/validation losses per fold
        X: Input of shape [N, s*s], independent variable
        Y: Input of shape [N, 1], dependent variable
        W_init: Input matrix of shape [s, nf]
    """
    # Define wrapper for cross validation folding
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    # Initialize empty container for the fold results
    cv_stats = []

    for fold, (tr, va) in enumerate(kf.split(X), 1):
        # Define training sample
        Xtrain, Ytrain = X[tr], Y[tr]
        # Define validation sample
        Xvalid, Yvalid = X[va], Y[va]
        # Train model with current data sets
        model, kernel, *_ = train_model(
            Xtrain,
            Ytrain,
            W_init,
            ard_flag,
            eta_flag,
            logged_flag,
            kernel_type,
            maxiter,
        )

        # TRAIN metrics
        # Grab training set loss
        # While regression with Gaussian noise gives exact marginal likelihood (NLML), classification has a non-Gaussian likelihood
        # The normalization integral (or marginal likelihood) p(y|X) is intractable
        # This requires an approximation
        # One way to do it is to use a Variational GP (VGP) classification which maximizes the evidence lower bound (ELBO)
        # The VGP model's training loss is the evidence lower bound (ELBO)
        train_elbo = -float(model.training_loss().numpy())  # ELBO
        # Predictive probabilities on training set
        train_prob, _ = model.predict_y(Xtrain)
        train_prob = train_prob.numpy().flatten()
        # Generate predicted classes from probability using 0.5 as threshold
        train_preds = (train_prob >= 0.5).astype(int)
        # Accuracy of label prediction
        train_accuracy = accuracy_score(Ytrain.flatten(), train_preds)
        # Cross entropy for binary classification: for each entry L(y,p) = -(y log(p) + (1-y) log(1-p))
        # Logloss = - mean(L(y,p))
        train_logloss = log_loss(Ytrain.flatten(), train_prob, labels=[0, 1])

        # VALIDATION metrics
        # Predictive probabilities on training set
        val_prob, _ = model.predict_y(Xvalid)
        val_prob = val_prob.numpy().flatten()
        # Generate predicted classes from probability using 0.5 as threshold
        val_preds = (val_prob >= 0.5).astype(int)
        # Accuracy of label prediction
        val_accuracy = accuracy_score(Yvalid.flatten(), val_preds)
        val_logloss = log_loss(Yvalid.flatten(), val_prob, labels=[0, 1])

        # The NLPD is the analogue of the marginal likelihood conditioned on the training fit:
        logdens = model.predict_log_density((Xvalid, Yvalid))  # shape (N,)
        val_nlpd = -float(np.mean(logdens.numpy()))

        cv_stats.append(
            {
                "fold": fold,
                "train_elbo": train_elbo,
                "train_accuracy": float(train_accuracy),
                "train_logloss": float(train_logloss),
                "val_nlpd": val_nlpd,
                "val_accuracy": float(val_accuracy),
                "val_logloss": float(val_logloss),
            }
        )

    return cv_stats
