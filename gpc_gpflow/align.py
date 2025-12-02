"""
Linear-algebra utilities for symmetric positive definite (SPD) matrices
and covariance alignment

This module provides:
- Eigen-decomposition helpers for SPD matrices with eigenvalue clipping
- Matrix functions: square root, inverse square root, matrix log, matrix exp
- Riemannian (and Euclidean) geometric mean of SPD covariance matrices
- Alignment of train/test covariance sets based on a reference mean
"""

from typing import Tuple
import numpy as np
from numpy.linalg import eigh

def _spd_eig(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the eigen-decomposition of a symmetric positive definite (SPD) matrix
    with eigenvalue clipping for numerical stability

    Parameters
    ----------
    A : np.ndarray
        Square SPD matrix of shape (d, d)

    Returns
    -------
    w : np.ndarray
        One-dimensional array of eigenvalues of shape (d,)
        Eigenvalues are clipped from below at 1e-12 to avoid numerical issues
    V : np.ndarray
        Orthonormal eigenvectors of shape (d, d), such that:
        A ≈ V @ diag(w) @ V.T
    """
    w, V = eigh(A)
    w = np.clip(w, 1e-12, None)
    return w, V


def _sqrtm(A: np.ndarray) -> np.ndarray:
    """
    Compute the matrix square root of an SPD matrix
    Uses eigen-decomposition and applies the square root to the eigenvalues

    Parameters
    ----------
    A : np.ndarray
        Square SPD matrix of shape (d, d)

    Returns
    -------
    np.ndarray
        Matrix square root of A with shape (d, d),
        such that sqrt(A) @ sqrt(A) ~ A
    """
    w, V = _spd_eig(A)
    return V @ np.diag(np.sqrt(w)) @ V.T


def _isqrtm(A: np.ndarray) -> np.ndarray:
    """
    Compute the inverse matrix square root of an SPD matrix

    Uses eigen-decomposition and applies the inverse square root
    to the eigenvalues

    Parameters
    ----------
    A : np.ndarray
        Square SPD matrix of shape (d, d)

    Returns
    -------
    np.ndarray
        Inverse matrix square root of A with shape (d, d),
        such that isqrt(A) @ isqrt(A) ~ A^{-1}
    """
    w, V = _spd_eig(A)
    return V @ np.diag(1.0 / np.sqrt(w)) @ V.T


def _logm(A: np.ndarray) -> np.ndarray:
    """
    Compute the matrix logarithm of an SPD matrix

    Uses eigen-decomposition and applies the natural logarithm
    to the eigenvalues

    Parameters
    ----------
    A : np.ndarray
        Square SPD matrix of shape (d, d)

    Returns
    -------
    np.ndarray
        Matrix logarithm of A with shape (d, d), i.e. log(A)
    """
    w, V = _spd_eig(A)
    return V @ np.diag(np.log(w)) @ V.T


def _expm(S: np.ndarray) -> np.ndarray:
    """
    Compute the matrix exponential of a symmetric matrix

    Uses eigen-decomposition and applies the exponential to the eigenvalues

    Parameters
    ----------
    S : np.ndarray
        Square symmetric matrix of shape (d, d)

    Returns
    -------
    np.ndarray
        Matrix exponential of S with shape (d, d), i.e. expm(S)
    """
    w, V = eigh(S)
    return V @ np.diag(np.exp(w)) @ V.T


def geom_mean_spd(
    covs: np.ndarray,
    tol: float = 1e-9,
    max_iter: int = 64,
) -> np.ndarray:
    """
    Compute the Riemannian (geometric) mean of SPD covariance matrices

    Implements the iterative algorithm on the manifold of SPD matrices
    using the affine-invariant Riemannian metric:

        G_{k+1} = sqrt(G_k) @ expm( mean_i( log( G_k^{-1/2} C_i G_k^{-1/2} ) ) ) @ sqrt(G_k)

    The iteration stops when the Frobenius norm of the update `delta`
    is below `tol` or when `max_iter` iterations have been reached

    Parameters
    ----------
    covs : np.ndarray
        Array of SPD covariance matrices with shape (n_samples, d, d)
    tol : float, optional
        Convergence tolerance on the Frobenius norm of the mean log-update
        Default is 1e-9
    max_iter : int, optional
        Maximum number of iterations. Default is 64

    Returns
    -------
    np.ndarray
        Geometric mean SPD matrix G of shape (d, d)
    """
    # Initialize with the Euclidean mean
    G = np.mean(covs, axis=0)

    for _ in range(max_iter):
        G_inv_sqrt = _isqrtm(G)
        # Project covariances into tangent space at G, take mean log
        logs = np.array(
            [_logm(G_inv_sqrt @ C @ G_inv_sqrt) for C in covs]
        )
        delta = np.mean(logs, axis=0)

        # Check convergence in Frobenius norm
        if np.linalg.norm(delta, ord="fro") < tol:
            break

        # Move along the geodesic defined by delta
        G_sqrt = _sqrtm(G)
        G = G_sqrt @ _expm(delta) @ G_sqrt

    return G


def align_split(
    X_train_cov: np.ndarray,
    X_test_cov: np.ndarray,
    method: str = "riemann",
):
    """
    Whiten / align train and test covariance matrices using a reference mean

    The function computes a reference covariance matrix `M` from the
    training set, either as:
    - Euclidean mean (`method="euclidean"`), or
    - Riemannian (geometric) mean (`method="riemann"`)

    It then constructs a whitening transform W = M^{-1/2} and applies:

        C_aligned = W @ C @ W

    to every covariance matrix in both train and test sets

    Parameters
    ----------
    X_train_cov : np.ndarray
        Training covariance matrices of shape (n_train, d, d), each assumed SPD
    X_test_cov : np.ndarray
        Test covariance matrices of shape (n_test, d, d), each assumed SPD
    method : str, optional
        Type of mean used for the reference matrix:
        - "euclidean" : arithmetic mean of covariances
        - "riemann"   : Riemannian (geometric) mean of covariances
        Default is "riemann"

    Returns
    -------
    X_train_aligned : np.ndarray
        Aligned training covariances of shape (n_train, d, d)
    X_test_aligned : np.ndarray
        Aligned test covariances of shape (n_test, d, d)
    M : np.ndarray
        Reference covariance matrix of shape (d, d) used to build W
    """
    if method == "euclidean":
        M = np.mean(X_train_cov, axis=0)
    elif method == "riemann":
        M = geom_mean_spd(X_train_cov)
    else:
        raise ValueError("method must be 'euclidean' or 'riemann'")

    # Whitening transform from the reference mean
    W = _isqrtm(M)

    def _apply(X: np.ndarray) -> np.ndarray:
        """
        Apply the alignment transform to a batch of covariance matrices.

        Parameters
        ----------
        X : np.ndarray
            Covariance matrices of shape (n_samples, d, d).

        Returns
        -------
        np.ndarray
            Aligned covariance matrices of shape (n_samples, d, d).
        """
        Y = W[None, :, :] @ X @ W[None, :, :]
        # Convert from broadcasting result to a clean (n_samples, d, d) array
        return np.array([C for C in Y])

    return _apply(X_train_cov), _apply(X_test_cov), M