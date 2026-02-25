"""
Custom GPy kernel for EEG covariance-based GP classification.

Overview
--------
This module defines ``CustomKernelGPy``, a GPy-compatible kernel that operates
directly on per-trial EEG covariance matrices.  The kernel:

  1. Receives flattened covariance matrices as input  (shape ``[N, s*s]``).
  2. Projects each covariance ``Σ_i`` through a set of ``nf`` spatial filters
     collected in the matrix ``W`` (shape ``[s, nf]``) to extract scalar
     per-filter variances  ``q_{i,p} = w_p^T Σ_i w_p``.
  3. Optionally applies a log transform:  ``u_{i,p} = log(q_{i,p})``.
  4. Optionally scales each feature by a learnable ARD weight:
     ``z_{i,p} = ard_p · u_{i,p}``.
  5. Evaluates either a **Linear** or **RBF** kernel on the resulting feature
     vectors ``z_i ∈ R^{nf}``, with an optional global scale ``eta``.

All parameters (``W``, ``ard_param``, ``eta_param``) are registered as GPy
``Param`` objects so they participate in gradient-based optimisation under the
EP inference framework.

Analytical gradients with respect to every trainable parameter are implemented
in ``update_gradients_full`` (full kernel matrix) and
``update_gradients_diag`` (diagonal only), as required by the GPy framework.

Dependencies
------------
- GPy  (``GPy.kern.src.kern.Kern``, ``GPy.core.parameterization.Param``)
- NumPy
"""

from typing import Optional, Tuple

import numpy as np
from GPy.core.parameterization import Param
from GPy.kern.src.kern import Kern


class CustomKernelGPy(Kern):
    """
    Spatial-filter covariance kernel for GP classification on EEG data.

    The kernel maps flattened covariance matrices ``X ∈ R^{N × s²}`` to a
    Gram matrix ``K ∈ R^{N × N}`` via learned spatial filters and an optional
    feature transformation (log, ARD scaling, global scaling).

    Parameters
    ----------
    W : np.ndarray, shape (s, nf)
        Initial spatial filter matrix.  Each column is one filter.
    W_trainable : bool
        If ``True`` (default), ``W`` is updated during optimisation.
        If ``False``, ``W`` is fixed at its initial value.
    ard_flag : bool
        If ``True``, a per-filter ARD scale vector ``ard ∈ R^{nf}`` is added
        as a trainable parameter (stored in log-space as ``ard_param``).
    eta_flag : bool
        If ``True``, a global variance scale ``eta > 0`` is added as a
        trainable parameter (stored in log-space as ``eta_param``).
    logged_flag : bool
        If ``True``, features are log-transformed before kernel evaluation:
        ``u_{i,p} = log(max(q_{i,p}, ε))``.
    kernel_type : str
        ``"Linear"`` or ``"RBF"``.
    active_dims : np.ndarray or None
        GPy active-dimensions mask (passed to the parent ``Kern``).
    name : str
        GPy parameter name string.
    """

    def __init__(
        self,
        W: np.ndarray,
        W_trainable: bool = True,
        ard_flag: bool = True,
        eta_flag: bool = True,
        logged_flag: bool = True,
        kernel_type: str = "RBF",
        active_dims: Optional[np.ndarray] = None,
        name: str = "custom_kernel_gpy",
    ) -> None:
        s, nf = W.shape
        # Initialise the parent Kern; input_dim is the flattened covariance size
        super().__init__(input_dim=s * s, active_dims=active_dims, name=name)

        # ------------------------------------------------------------------ #
        # Flags                                                                #
        # ------------------------------------------------------------------ #
        self._flag_ard    = bool(ard_flag)     # per-filter ARD scaling
        self._flag_eta    = bool(eta_flag)     # global output scaling
        self._flag_logged = bool(logged_flag)  # log-transform features
        self._kernel_type = kernel_type

        if kernel_type not in {"Linear", "RBF"}:
            raise ValueError("kernel_type must be 'Linear' or 'RBF'")

        # ------------------------------------------------------------------ #
        # Constants                                                            #
        # ------------------------------------------------------------------ #
        self.s    = int(s)    # number of EEG sensors
        self.nf   = int(nf)   # number of spatial filters (columns of W)
        self._eps = 1e-12     # numerical floor used inside log / division

        # ------------------------------------------------------------------ #
        # Trainable parameters                                                 #
        # ------------------------------------------------------------------ #

        # W — spatial filter matrix, shape [s, nf]
        self.W = Param("W", W.astype(np.float64))
        self.link_parameter(self.W)
        if W_trainable:
            self.W.unfix()
        else:
            self.W.fix()

        # ARD — per-filter log-scale vector, shape [nf]
        # Stored in log-space so that the effective scale ard = exp(ard_param) > 0
        if self._flag_ard:
            self.ard_param = Param("ard_param", np.zeros(self.nf))
            self.link_parameter(self.ard_param)
        else:
            self.ard_param = None

        # ETA — global log-scale scalar
        # Stored in log-space so that the effective scale eta = exp(eta_param) > 0
        if self._flag_eta:
            self.eta_param = Param("eta_param", np.array(0.0))
            self.link_parameter(self.eta_param)
        else:
            self.eta_param = None

        # RBF depends only on pairwise distances → it is stationary;
        # GPy uses this flag internally for certain optimisations
        self.is_stationary = (self._kernel_type == "RBF")

    # ------------------------------------------------------------------ #
    # Public parameter properties                                          #
    # ------------------------------------------------------------------ #

    @property
    def ard(self) -> np.ndarray:
        """
        Effective ARD scale vector, shape ``(nf,)``.

        Returns ``exp(ard_param)`` when ARD is enabled (guarantees positivity),
        or a vector of ones when ARD is disabled.
        """
        if self._flag_ard:
            return np.exp(self.ard_param)
        return np.ones(self.nf, dtype=np.float64)

    @property
    def eta(self) -> float:
        """
        Effective global output scale (scalar).

        Returns ``exp(eta_param)`` when eta is enabled (guarantees positivity),
        or ``1.0`` when eta is disabled.
        """
        if self._flag_eta:
            return float(np.exp(self.eta_param))
        return 1.0

    # ------------------------------------------------------------------ #
    # Private feature-computation helpers                                  #
    # ------------------------------------------------------------------ #

    def _reshape_input(self, X: np.ndarray) -> np.ndarray:
        """
        Reshape a flat input array into per-trial covariance matrices.

        Parameters
        ----------
        X : np.ndarray, shape (N, s*s)
            Batch of flattened covariance matrices.

        Returns
        -------
        np.ndarray, shape (N, s, s)
        """
        if X is None:
            return None
        return np.asarray(X, dtype=np.float64).reshape(-1, self.s, self.s)

    def _compute_features(self, Sigma: np.ndarray) -> np.ndarray:
        """
        Compute per-trial, per-filter feature values from covariance matrices.

        For each trial ``i`` and filter ``p`` the raw quadratic form is:
            ``q_{i,p} = w_p^T Σ_i w_p``

        Optionally applies:
          - Log transform:  ``u_{i,p} = log(max(q_{i,p}, ε))``
          - ARD scaling:    ``z_{i,p} = ard_p · u_{i,p}``

        Parameters
        ----------
        Sigma : np.ndarray, shape (N, s, s)
            Batch of SPD covariance matrices.

        Returns
        -------
        np.ndarray, shape (N, nf)
            Feature matrix ready for kernel evaluation.
        """
        # Σ @ W  →  [N, s, nf]
        Sw = np.einsum("Nsj,jf->Nsf", Sigma, self.W)
        # w^T Σ w  →  [N, nf]   (contract the sensor axis with W)
        wSw = np.einsum("Njf,jf->Nf", Sw, self.W)

        if self._flag_logged:
            wSw = np.log(np.maximum(wSw, self._eps))

        if self._flag_ard:
            # Broadcast [N, nf] * [nf,]
            wSw = wSw * self.ard

        return wSw  # [N, nf]

    def _compute_features_pair(
        self,
        X: np.ndarray,
        X2: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Reshape inputs and compute features for a (X, X2) pair.

        A single call handles both the symmetric case ``X2 is None`` (in which
        X2 reuses X's results) and the cross-covariance case ``X2 is not None``.

        Parameters
        ----------
        X : np.ndarray, shape (N, s*s)
        X2 : np.ndarray, shape (M, s*s), optional

        Returns
        -------
        Sigma1 : np.ndarray, shape (N, s, s)
        Sigma2 : np.ndarray, shape (M, s, s) or same as Sigma1 when X2 is None
        wSw1   : np.ndarray, shape (N, nf)   — features for X
        wSw2   : np.ndarray, shape (M, nf)   — features for X2 (or X)
        """
        Sigma1 = self._reshape_input(X)
        Sigma2 = self._reshape_input(X2) if X2 is not None else Sigma1

        wSw1 = self._compute_features(Sigma1)
        wSw2 = self._compute_features(Sigma2) if X2 is not None else wSw1

        return Sigma1, Sigma2, wSw1, wSw2

    def _apply_eta(self, K: np.ndarray) -> np.ndarray:
        """
        Multiply a kernel matrix (or vector) by the global scale ``eta``.

        Parameters
        ----------
        K : np.ndarray
            Kernel matrix of any shape.

        Returns
        -------
        np.ndarray
            ``eta * K``
        """
        return self.eta * K

    def _compute_kernel(self, wSw1: np.ndarray, wSw2: np.ndarray) -> np.ndarray:
        """
        Evaluate the chosen kernel on a pair of feature matrices.

        Parameters
        ----------
        wSw1 : np.ndarray, shape (N, nf)
        wSw2 : np.ndarray, shape (M, nf)

        Returns
        -------
        np.ndarray, shape (N, M)
            ``K[i, j] = eta · k(z_i, z_j)``

        Notes
        -----
        **Linear:**  ``k(z_i, z_j) = eta · (1 + z_i · z_j)``

        **RBF:**     ``k(z_i, z_j) = eta · exp(−‖z_i − z_j‖² / (2 · nf))``
                     The bandwidth is normalised by ``nf`` so that the scale of
                     the kernel does not drift as the number of filters grows.
        """
        if self._kernel_type == "Linear":
            Kmat = wSw1.dot(wSw2.T) + 1.0  # [N, M], bias term of 1
            return self._apply_eta(Kmat)

        elif self._kernel_type == "RBF":
            # Expand ‖z_i − z_j‖² = ‖z_i‖² − 2 z_i·z_j + ‖z_j‖²
            # using the broadcasting trick to avoid an explicit (N, M, nf) tensor
            sq1 = np.sum(wSw1 ** 2, axis=1, keepdims=True)  # [N, 1]
            sq2 = np.sum(wSw2 ** 2, axis=1, keepdims=True)  # [M, 1]
            dist_sq = sq1 - 2.0 * (wSw1 @ wSw2.T) + sq2.T  # [N, M]
            Kmat = np.exp(-0.5 * dist_sq / float(self.nf))
            return self._apply_eta(Kmat)

        else:
            raise ValueError(f"Unsupported kernel_type: '{self._kernel_type}'")

    # ------------------------------------------------------------------ #
    # GPy kernel interface                                                 #
    # ------------------------------------------------------------------ #

    def K(self, X: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute the full kernel (Gram) matrix.

        Parameters
        ----------
        X : np.ndarray, shape (N, s*s)
        X2 : np.ndarray, shape (M, s*s), optional
            If ``None``, the symmetric matrix ``K(X, X)`` is returned.

        Returns
        -------
        np.ndarray, shape (N, M) or (N, N)
        """
        *_, wSw1, wSw2 = self._compute_features_pair(X, X2)
        return self._compute_kernel(wSw1, wSw2)

    def Kdiag(self, X: np.ndarray) -> np.ndarray:
        """
        Compute only the diagonal of the kernel matrix ``K(X, X)``.

        This is more efficient than computing the full matrix and extracting
        the diagonal, because closed forms exist for both kernel types:

        - **Linear:** ``diag_i = eta · (1 + ‖z_i‖²)``
        - **RBF:**    ``diag_i = eta · 1``  (self-distance is zero)

        Parameters
        ----------
        X : np.ndarray, shape (N, s*s)

        Returns
        -------
        np.ndarray, shape (N,)
        """
        Sigma1 = self._reshape_input(X)
        wSw1   = self._compute_features(Sigma1)

        if self._kernel_type == "Linear":
            # Diagonal of wSw1 @ wSw1.T  is  sum of squared rows
            diag = np.sum(wSw1 * wSw1, axis=1) + 1.0
            return self._apply_eta(diag)

        elif self._kernel_type == "RBF":
            # k(z_i, z_i) = exp(0) = 1 for all i
            diag = np.ones(wSw1.shape[0], dtype=np.float64)
            return self._apply_eta(diag)

        else:
            raise ValueError(f"Unsupported kernel_type: '{self._kernel_type}'")

    # ------------------------------------------------------------------ #
    # Analytic gradients                                                   #
    # ------------------------------------------------------------------ #

    def update_gradients_full(
        self,
        dL_dK: np.ndarray,
        X: np.ndarray,
        X2: Optional[np.ndarray] = None,
    ) -> None:
        """
        Compute and accumulate parameter gradients from full-matrix sensitivities.

        Called by the GPy EP inference engine after computing ``dL/dK``.
        Sets ``.gradient`` on ``eta_param``, ``ard_param``, and ``W``.

        Parameters
        ----------
        dL_dK : np.ndarray, shape (N, M) or (N, N)
            Upstream gradient of the log-likelihood w.r.t. the kernel matrix.
        X : np.ndarray, shape (N, s*s)
        X2 : np.ndarray, shape (M, s*s), optional

        Notes
        -----
        The gradient chain is:

        ``L ← K ← η``                            (global scale)
        ``L ← K ← Z ← ard_param``               (ARD scale, via chain rule)
        ``L ← K ← Z ← u ← q ← W``              (spatial filters)

        where:
          ``q_{i,p} = w_p^T Σ_i w_p``           (quadratic form)
          ``u_{i,p} = log(q_{i,p})`` or ``q``   (optional log)
          ``z_{i,p} = ard_p · u_{i,p}``         (optional ARD)
        """
        # ---- 1.  Reshape inputs and compute intermediate quantities ----------
        Sigma1 = self._reshape_input(X)             # [N, s, s]
        Sw1    = np.einsum("Nsj,jf->Nsf", Sigma1, self.W)  # [N, s, nf]
        q1     = np.einsum("Njf,jf->Nf",  Sw1,    self.W)  # [N, nf]

        if X2 is None:
            Sigma2, Sw2, q2 = Sigma1, Sw1, q1
        else:
            Sigma2 = self._reshape_input(X2)
            Sw2    = np.einsum("Msj,jf->Msf", Sigma2, self.W)
            q2     = np.einsum("Mjf,jf->Mf",  Sw2,    self.W)

        # Log-transform bookkeeping
        if self._flag_logged:
            u1     = np.log(np.maximum(q1, self._eps))
            u2     = np.log(np.maximum(q2, self._eps))
            du_dq1 = 1.0 / np.maximum(q1, self._eps)  # d(log q)/dq = 1/q
            du_dq2 = 1.0 / np.maximum(q2, self._eps)
        else:
            u1 = q1;  u2 = q2
            du_dq1 = np.ones_like(q1)
            du_dq2 = np.ones_like(q2)

        # ARD bookkeeping — recompute Z from u to keep gradients consistent
        if self._flag_ard:
            ard = self.ard          # [nf],  exp(ard_param)
            Z1  = u1 * ard
            Z2  = u2 * ard
        else:
            ard = np.ones(self.nf, dtype=np.float64)
            Z1  = u1;  Z2 = u2

        # ---- 2.  Kernel matrix and eta gradient ----------------------------
        K = self._compute_kernel(Z1, Z2)    # [N, M]

        # dL/d(eta_param) = dL/dK · dK/deta · deta/d(eta_param)
        # dK/deta = K/eta,  deta/d(eta_param) = eta  →  net factor = K
        if self._flag_eta:
            self.eta_param.gradient = float(np.sum(dL_dK * K))

        # ---- 3.  dL/dZ for the two feature sets ---------------------------
        if self._kernel_type == "Linear":
            # K = eta · (1 + Z1 @ Z2^T)
            # dK/dZ1 = eta · Z2,  dK/dZ2 = eta · Z1
            eta = self.eta
            dL_dZ1 = dL_dK   @ (eta * Z2)   # [N, nf]
            dL_dZ2 = dL_dK.T @ (eta * Z1)   # [M, nf]

        else:  # RBF
            # K = eta · exp(−‖Z1 − Z2‖² / (2·nf))
            # dL/dZ1[i] = Σ_j dL_dK[i,j] · K[i,j] · −(Z1[i] − Z2[j]) / nf
            B      = dL_dK * K                                    # [N, M]
            sumB   = np.sum(B, axis=1, keepdims=True)             # [N, 1]
            dL_dZ1 = (-(1.0 / float(self.nf))) * (sumB * Z1 - B @ Z2)      # [N, nf]
            sumB2  = np.sum(B, axis=0, keepdims=True)             # [1, M]
            dL_dZ2 = (-(1.0 / float(self.nf))) * (sumB2.T * Z2 - B.T @ Z1) # [M, nf]

        # ---- 4.  ARD gradient (if enabled) ---------------------------------
        # z = ard · u  →  dz/d(ard_param) = dz/dard · dard/d(ard_param)
        #                                  = u · ard  (because ard = exp(ard_param))
        if self._flag_ard:
            self.ard_param.gradient = (
                np.sum(dL_dZ1 * (ard * u1), axis=0)
                + np.sum(dL_dZ2 * (ard * u2), axis=0)
            )

        # ---- 5.  W gradient ------------------------------------------------
        # q_{i,p} = w_p^T Σ_i w_p
        # dq/dw_p = 2 Σ_i w_p  →  accumulated via einsum over all (i, p) pairs
        dL_du1 = dL_dZ1 * ard
        dL_du2 = dL_dZ2 * ard
        dL_dq1 = dL_du1 * du_dq1   # [N, nf]
        dL_dq2 = dL_du2 * du_dq2   # [M, nf]

        G1 = 2.0 * np.einsum("Nsf,Nf->sf", Sw1, dL_dq1)  # [s, nf]
        G2 = 2.0 * np.einsum("Msf,Mf->sf", Sw2, dL_dq2)  # [s, nf]
        self.W.gradient = G1 + G2

    def update_gradients_diag(
        self,
        dL_dKdiag: np.ndarray,
        X: np.ndarray,
    ) -> None:
        """
        Compute and accumulate parameter gradients from diagonal sensitivities.

        Called by GPy when only the diagonal of the kernel matrix is needed
        (e.g. during certain EP steps or for noise estimation).

        Parameters
        ----------
        dL_dKdiag : np.ndarray, shape (N,)
            Upstream gradient of the log-likelihood w.r.t. the kernel diagonal.
        X : np.ndarray, shape (N, s*s)

        Notes
        -----
        For the **RBF** kernel the diagonal is constant (``eta · 1``), so
        ``W`` and ``ard_param`` receive zero gradient from this call.
        For the **Linear** kernel the diagonal depends on ``‖z_i‖²``, so
        ``W`` and ``ard_param`` do receive non-trivial gradients.
        """
        # ---- 1.  Intermediate quantities -----------------------------------
        Sigma1 = self._reshape_input(X)                          # [N, s, s]
        Sw1    = np.einsum("Nsj,jf->Nsf", Sigma1, self.W)       # [N, s, nf]
        q1     = np.einsum("Njf,jf->Nf",  Sw1,    self.W)       # [N, nf]

        if self._flag_logged:
            u1     = np.log(np.maximum(q1, self._eps))
            du_dq1 = 1.0 / np.maximum(q1, self._eps)
        else:
            u1 = q1
            du_dq1 = np.ones_like(q1)

        if self._flag_ard:
            ard = self.ard
            Z1  = u1 * ard
        else:
            ard = np.ones(self.nf, dtype=np.float64)
            Z1  = u1

        # ---- 2.  Kdiag and eta gradient ------------------------------------
        Kdiag = self.Kdiag(X)   # [N,]
        if self._flag_eta:
            self.eta_param.gradient = float(np.sum(dL_dKdiag * Kdiag))

        # ---- 3.  dL/dZ from diagonal structure ----------------------------
        if self._kernel_type == "Linear":
            # Kdiag_i = eta · (1 + ‖z_i‖²)
            # dKdiag_i/dZ_i = 2 · eta · z_i
            dL_dZ1 = dL_dKdiag[:, None] * (2.0 * self.eta * Z1)  # [N, nf]

        else:  # RBF
            # Kdiag_i = eta · 1  (no dependence on Z)
            dL_dZ1 = np.zeros_like(Z1)

        # ---- 4.  ARD gradient (diagonal) ----------------------------------
        if self._flag_ard:
            self.ard_param.gradient = np.sum(dL_dZ1 * (ard * u1), axis=0)

        # ---- 5.  W gradient (diagonal) ------------------------------------
        dL_du1 = dL_dZ1 * ard
        dL_dq1 = dL_du1 * du_dq1
        self.W.gradient = 2.0 * np.einsum("Nsf,Nf->sf", Sw1, dL_dq1)  # [s, nf]
