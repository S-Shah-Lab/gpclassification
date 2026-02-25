"""
Covariance alignment utilities for EEG trial data.

Overview
--------
When multiple sessions or subjects contribute trials to a single dataset,
the mean covariance matrix can shift between sources.  Aligning all
covariances to a common reference removes this between-source bias before
feeding them into a classifier.

This module provides:

- Helper functions for eigen-decomposition-based SPD matrix operations
  (square root, inverse square root, matrix log/exp) with eigenvalue clipping
  for numerical stability.
- An iterative Riemannian (geometric) mean estimator for SPD matrices using
  the affine-invariant metric.
- ``align_split``: the main entry point that computes a reference mean from
  the training set and applies the corresponding whitening transform to both
  train and test covariances.

All functions operate on symmetric positive definite (SPD) matrices.
Eigenvalues are clipped at ``1e-12`` throughout to guard against near-singular
inputs that arise from short trial windows or high-dimensional recording arrays.

Typical usage
-------------
::

    from align import align_split

    X_train_aligned, X_test_aligned, M = align_split(
        X_train_cov, X_test_cov, method="riemann"
    )

References
----------
Barachant et al. (2012). "Multiclass brain-computer interface classification
by Riemannian geometry." IEEE Transactions on Biomedical Engineering.
"""

from typing import Tuple

import numpy as np
from numpy.linalg import eigh


# ===========================================================================
# SPD matrix helpers
# ===========================================================================

def _spd_eig(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Eigen-decompose a symmetric positive definite (SPD) matrix.

    Uses ``numpy.linalg.eigh`` which exploits symmetry and returns eigenvalues
    in ascending order.  Eigenvalues are clipped at ``1e-12`` to prevent
    downstream numerical failures from near-zero or slightly negative values
    caused by floating-point rounding.

    Parameters
    ----------
    A : np.ndarray, shape (d, d)
        Square SPD matrix.

    Returns
    -------
    w : np.ndarray, shape (d,)
        Eigenvalues, clipped to ``[1e-12, ∞)``.
    V : np.ndarray, shape (d, d)
        Orthonormal eigenvectors as columns; ``A ≈ V @ diag(w) @ V.T``.
    """
    w, V = eigh(A)
    w    = np.clip(w, 1e-12, None)
    return w, V


def _sqrtm(A: np.ndarray) -> np.ndarray:
    """
    Compute the symmetric matrix square root of an SPD matrix.

    Uses the eigen-decomposition ``A = V diag(w) V^T`` and applies
    ``sqrt`` element-wise to the eigenvalues.

    Parameters
    ----------
    A : np.ndarray, shape (d, d)

    Returns
    -------
    np.ndarray, shape (d, d)
        Matrix ``S`` such that ``S @ S ≈ A``.
    """
    w, V = _spd_eig(A)
    return V @ np.diag(np.sqrt(w)) @ V.T


def _isqrtm(A: np.ndarray) -> np.ndarray:
    """
    Compute the inverse matrix square root of an SPD matrix.

    Applies ``1 / sqrt`` element-wise to the eigenvalues of ``A``.

    Parameters
    ----------
    A : np.ndarray, shape (d, d)

    Returns
    -------
    np.ndarray, shape (d, d)
        Matrix ``W`` such that ``W @ W ≈ A^{-1}``
        (and ``W @ A @ W ≈ I``).
    """
    w, V = _spd_eig(A)
    return V @ np.diag(1.0 / np.sqrt(w)) @ V.T


def _logm(A: np.ndarray) -> np.ndarray:
    """
    Compute the matrix logarithm of an SPD matrix.

    Applies ``log`` element-wise to the eigenvalues.  The result is a
    symmetric matrix (not necessarily positive definite).

    Parameters
    ----------
    A : np.ndarray, shape (d, d)

    Returns
    -------
    np.ndarray, shape (d, d)
        Matrix logarithm ``log(A)``.
    """
    w, V = _spd_eig(A)
    return V @ np.diag(np.log(w)) @ V.T


def _expm(S: np.ndarray) -> np.ndarray:
    """
    Compute the matrix exponential of a symmetric matrix.

    Applies ``exp`` element-wise to the eigenvalues.  When ``S = log(A)``
    for some SPD matrix ``A``, this recovers ``A``.

    Parameters
    ----------
    S : np.ndarray, shape (d, d)
        Symmetric (not necessarily positive definite) matrix.

    Returns
    -------
    np.ndarray, shape (d, d)
        Matrix exponential ``expm(S)``, which is SPD.
    """
    w, V = eigh(S)           # no clipping — eigenvalues can be negative here
    return V @ np.diag(np.exp(w)) @ V.T


# ===========================================================================
# Riemannian mean
# ===========================================================================

def geom_mean_spd(
    covs: np.ndarray,
    tol: float = 1e-9,
    max_iter: int = 64,
) -> np.ndarray:
    """
    Compute the Riemannian (geometric) mean of a set of SPD matrices.

    Implements the iterative gradient-descent algorithm on the manifold of
    SPD matrices equipped with the affine-invariant metric.  At each step the
    current estimate ``G`` is updated by moving along the geodesic defined by
    the mean of the tangent-space projections of all input matrices:

    .. math::

        G_{k+1} = G_k^{1/2} \\exp\\!\\left(
            \\frac{1}{n} \\sum_i \\log\\!\\left( G_k^{-1/2} C_i G_k^{-1/2} \\right)
        \\right) G_k^{1/2}

    Convergence is declared when the Frobenius norm of the mean log-update
    falls below ``tol``.

    Parameters
    ----------
    covs : np.ndarray, shape (n, d, d)
        Batch of SPD covariance matrices.
    tol : float
        Convergence tolerance on the Frobenius norm of the update.
    max_iter : int
        Maximum number of iterations before returning the current estimate.

    Returns
    -------
    np.ndarray, shape (d, d)
        Riemannian mean of the input matrices.

    Notes
    -----
    Initialised with the Euclidean (arithmetic) mean, which typically provides
    a warm start close to the Riemannian mean for mildly dispersed datasets.
    """
    G = np.mean(covs, axis=0)  # Euclidean mean as initialisation

    for _ in range(max_iter):
        G_inv_sqrt = _isqrtm(G)
        # Map all covariances to the tangent space at G and average
        logs  = np.array([_logm(G_inv_sqrt @ C @ G_inv_sqrt) for C in covs])
        delta = np.mean(logs, axis=0)

        if np.linalg.norm(delta, ord="fro") < tol:
            break

        # Geodesic update
        G_sqrt = _sqrtm(G)
        G      = G_sqrt @ _expm(delta) @ G_sqrt

    return G


# ===========================================================================
# Main alignment function
# ===========================================================================

def align_split(
    X_train_cov: np.ndarray,
    X_test_cov: np.ndarray,
    method: str = "riemann",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align train and test covariance matrices to a common reference.

    A reference matrix ``M`` is estimated from the **training set only**
    (to prevent data leakage from the test set).  The whitening transform
    ``W = M^{-1/2}`` is then applied to every covariance in both splits:

    .. math::

        C_\\text{aligned} = M^{-1/2} \\, C \\, M^{-1/2}

    After alignment the mean of the training covariances is approximately the
    identity matrix, which removes the global offset between recording sessions
    or subjects.

    Parameters
    ----------
    X_train_cov : np.ndarray, shape (n_train, d, d)
        Training covariance matrices; each must be SPD.
    X_test_cov : np.ndarray, shape (n_test, d, d)
        Test covariance matrices; each must be SPD.
    method : str
        Reference mean estimator:

        - ``"euclidean"`` — arithmetic mean of the training covariances.
          Fast but not invariant to affine transformations of the data.
        - ``"riemann"``   — Riemannian (geometric) mean via iterative
          geodesic update.  Affine-invariant; recommended default.

    Returns
    -------
    X_train_aligned : np.ndarray, shape (n_train, d, d)
        Aligned training covariances.
    X_test_aligned : np.ndarray, shape (n_test, d, d)
        Aligned test covariances.
    M : np.ndarray, shape (d, d)
        Reference covariance matrix used to build the whitening transform.
        Can be stored and applied to future data batches if needed.

    Raises
    ------
    ValueError
        If ``method`` is not ``"euclidean"`` or ``"riemann"``.
    """
    if method == "euclidean":
        M = np.mean(X_train_cov, axis=0)
    elif method == "riemann":
        M = geom_mean_spd(X_train_cov)
    else:
        raise ValueError(
            f"method must be 'euclidean' or 'riemann'; got '{method}'."
        )

    W = _isqrtm(M)  # whitening transform: W @ M @ W = I

    def _apply(X: np.ndarray) -> np.ndarray:
        # Vectorised: W[None] @ X[i] @ W[None] for all i simultaneously
        return np.array([W @ C @ W for C in X])

    return _apply(X_train_cov), _apply(X_test_cov), M
