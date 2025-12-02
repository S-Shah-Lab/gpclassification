from typing import Optional, Tuple
import numpy as np

import GPy
from GPy.kern.src.kern import Kern
from GPy.core.parameterization import Param

class CustomKernelGPy(Kern):
    def __init__(
        self,
        W          : np.ndarray,
        W_trainable: bool = True,
        ard_flag   : bool = True,
        eta_flag   : bool = True,
        logged_flag: bool = True,
        kernel_type: str = "RBF",
        active_dims: Optional[np.ndarray] = None,
        name       : str = "custom_kernel_gpy",
    ):
        """
        TODO
        """
        s, nf = W.shape
        # As a child class, grab all the methods from the parent class
        super().__init__(input_dim=s*s, active_dims=active_dims, name=name)

        # Flags
        self._flag_ard    = bool(ard_flag)    # Individual feature scaling
        self._flag_eta    = bool(eta_flag)    # Global scaling
        self._flag_logged = bool(logged_flag) # Logged features
        self._kernel_type = kernel_type       # Type of kernel to use
        if kernel_type not in {"Linear", "RBF"}:
            raise ValueError("kernel_type must be 'Linear' or 'RBF'")

        # Constants
        self.s    = int(s)  # Number of sensors
        self.nf   = int(nf) # Number of spatial filters, cols of W
        self._eps = 1e-12   # Epsilon for log numerical problems

        # Parameters
        # (W) Spatial filter matrix with shape [s, nf]
        self.W = Param("W", W.astype(np.float64))
        self.link_parameter(self.W)
        if W_trainable:
            self.W.unfix()
        else:
            self.W.fix()

        if self._flag_ard:
            # (ARD) Automatic Relevance Detection determines scaling of each feature independently
            # This is a parameter for each filter of the spatial filter matrix, nf parameters in total
            self.ard_param = Param("ard_param", np.zeros(self.nf))
            self.link_parameter(self.ard_param)
        else:
            self.ard_param = None

        if self._flag_eta:
            # (ETA) Global variance parameter determines global scaling
            # This parameter is a unique scalar
            self.eta_param = Param("eta_param", np.array(0.0))
            self.link_parameter(self.eta_param)
        else:
            self.eta_param = None
        
        # Stationarity flag helps GPy internal logic
        # Because RBF only depends on distance between points, RBF kernel is stationary
        self.is_stationary = (self._kernel_type == "RBF")
        
    @property
    def ard(self) -> np.ndarray:
        # positive ARD vector, transformed with exponential
        if self._flag_ard:
            return np.exp(self.ard_param)
        else:
            return np.ones(self.nf, dtype=np.float64)

    @property
    def eta(self) -> float:
        # positive global scale, transformed with exponential
        if self._flag_eta:
            return float(np.exp(self.eta_param))
        else:
            return 1.0


    def _reshape_input(self, X: np.ndarray) -> np.ndarray:
        """
        Helper function that takes care of reshaping input into covarianc matrix form
        Reshape X [N, s*s] into Sigma [N, s, s]
        """
        if X is None:
            return None
        X = np.asarray(X, dtype=np.float64)
        return X.reshape(-1, self.s, self.s)


    def _apply_ard(self, wSw: np.ndarray) -> np.ndarray:
        """
        Multiply scaling for each filter col by broadcasting across the rows automatically
        If self._flag_ard is True, a parameter called self.ard is created and the framework will optimize it by default
        This option can be turned off to maintain the parameter constant using gpflow.set_trainable(self.ard, False)
        """
        if not self._flag_ard: 
            return wSw
        else:                  
            return wSw * self.ard # broadcast [N, nf] * [nf,]


    def _compute_features(self, Sigma: np.ndarray) -> np.ndarray:
        """
        Compute per-sample quadratic forms:              f_p^T Σ_i f_p
        The method takes into account optional log form: log( f_p^T Σ_i f_p )
        And ARD form:                                    ard_p * log( f_p^T Σ_i f_p )
        
        Σ  : [N, s, s]
        W  : [s, nf]
        Σw : [N, s, nf] --> This is standard matrix multiplication, each one of these columns is multiplied by a row of W.T
        wΣw: [N, nf]
        """
        # Sw = Σ @ W
        Sw = np.einsum("Nsj,jf->Nsf", Sigma, self.W) # shape [N, s, nf]
        # wSw comes from contraction of rows of W with cols of Sw
        wSw = np.einsum("Njf,jf->Nf", Sw, self.W)  # shape [N, nf]
        
        if self._flag_logged: 
            wSw = np.log(np.maximum(wSw, self._eps)) # epsilon for numerical stability
        
        if self._flag_ard:
            wSw = self._apply_ard(wSw)

        return wSw  # [N, nf]

 
    def _prep_elements(self, X: np.ndarray, X2: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        TODO
        """
        # Reshape flat covariance matrices in input from [N, s*s] to [N, s, s]
        Sigma1 = self._reshape_input(X)                                 # [N, s, s]
        Sigma2 = self._reshape_input(X2) if X2 is not None else Sigma1  # [M, s, s] if X2 is not None else [N, s, s]
        
        # Transform input covariance matrices into features -> linear or logged filtered variances
        wSw1 = self._compute_features(Sigma1)                                 # [N, nf]
        wSw2 = self._compute_features(Sigma2) if Sigma2 is not None else wSw1 # [M, nf] if X2 is not None else [N, nf]
        
        return Sigma1, Sigma2, wSw1, wSw2
    
        
    def _apply_eta(self, K: np.ndarray) -> np.ndarray:
        """
        Multiply by global scaling
        If self._flag_eta is True, a parameter called self.eta is created and the framework will optimize it by default
        This option can be turned off to maintain the parameter constant using gpflow.set_trainable(self.eta, False)
        """
        return self.eta * K
    

    def _compute_kernel(self, wSw: np.ndarray, wSw2: np.ndarray) -> np.ndarray:
        """
        TODO
        """
        if self._kernel_type == "Linear":
            # Evaluate linear kernel in the form z.T * z
            Kmat = wSw.dot(wSw2.T) # [N,M] if wSw != wSw2 else [N, N]
            # Add bias term to the result
            Kmat += 1.0
            return self._apply_eta(Kmat) # [N, M] if X2 is not None else [N, N]

        elif self._kernel_type == "RBF":
            # Evaluate RBF kernel in the Gaussian-like form
            # Perform the square distances
            wSw_sq  = np.sum(wSw**2,  axis=1, keepdims=True)   # [N, 1]
            wSw2_sq = np.sum(wSw2**2, axis=1, keepdims=True)   # [M, 1] if wSw != wSw2 else [N, 1]
            # Evaluate kernel using the broadcasting trick
            dist_sq = wSw_sq - 2 * (wSw @ wSw2.T) + wSw2_sq.T  # [N, M] if wSw != wSw2 else [N, N]
            Kmat = np.exp(-0.5 * dist_sq / float(self.nf))
            return self._apply_eta(Kmat) # [N, M] if X2 is not None else [N, N]
        
        else:
            raise ValueError(f"Unsupported kernel type: {self._kernel_type}")
        

    def K(self, X: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """
        TODO
        """
        # Evaluate features
        *_, wSw1, wSw2 = self._prep_elements(X, X2)
        
        # Generate the custom covariance function
        # This function treats both cases in which X == X2 or X != X2
        return self._compute_kernel(wSw1, wSw2) # [N, M] if X2 is not None else [N, N]
        

    def Kdiag(self, X: np.ndarray) -> np.ndarray:
        """
        TODO
        """
        # Reshape flat covariance matrices in input from [N, s*s] to [N, s, s]
        Sigma1 = self._reshape_input(X) # [N, s, s]
        
        # Transform input covariance matrices into features -> linear or logged filtered variances
        wSw1 = self._compute_features(Sigma1) # [N, nf]
        
        # Long way to evaluate the diagonal of the kernel with X2 == X
        # Generate the custom covariance function
        # This function treats both cases in which X == X2 or X != X2
        # Kmatdiag = self._compute_kernel(wSw1, wSw1) # [N, N]
        # Kmatdiag = self._apply_eta(Kmatdiag)        # [N, N]
        # return np.diag(Kmatdiag)                    # [N,]
    
        # Efficient way to evaluate the diagonal of the kernel with X2 == X
        # Linear case:
        if self._kernel_type == "Linear":
            '''
            Diagonal elements of wSw.dot(wSw.T):
                if we transpose wSw, 00 -> 00 but 01 -> 10
                the dot product means that the diagonal element at index i is given by (assume nf=2):
                    wSw_i0 * wSw.T_0i + wSw_i1 * wSw.T_1i = 
                    wSw_i0 * wSw_i0   + wSw_i1 * wSw_i1   =
                    ( wSw_i0 )**2     + ( wSw_i1 )**2
                this directly translates to squaring every elements of wSw and summing along the columns
            Finally rememeber to add the bias term + 1.0
            '''
            diag = np.sum( wSw1 * wSw1, axis=1 ) + 1.0
            return self._apply_eta( diag )
        
        # RBF case:
        elif self._kernel_type == "RBF":
            '''
            The distance for the diagonal elements goes to 0
            This directly translates to having the exponential going to 1
            '''
            diag = np.ones( wSw1.shape[0], dtype=np.float64 )
            return self._apply_eta( diag )
            

    def update_gradients_full(self, dL_dK: np.ndarray, X: np.ndarray,
                            X2: Optional[np.ndarray] = None) -> None:
        """
        Compute parameter gradients from full kernel matrix sensitivities dL_dK.
        This sets .gradient for self.eta_param, self.ard_param, and self.W.
        Shapes:
        X:  [N, s*s]
        X2: [M, s*s] or None => X2 = X
        dL_dK: [N, M] or [N, N]
        """
        # 1) Prep inputs and features
        Sigma1, Sigma2, Z1, Z2 = self._prep_elements(X, X2)            # Z are the current features after log/ARD
        N = Sigma1.shape[0]
        M = Sigma2.shape[0]

        # Also compute pre-log quadratic forms q = diag(Wᵀ Σ W), needed for dZ/dW
        Sw1 = np.einsum("Nsj,jf->Nsf", Sigma1, self.W)                 # [N,s,nf]
        q1  = np.einsum("Njf,jf->Nf", Sw1, self.W)                     # [N,nf]
        if X2 is None:
            Sigma2 = Sigma1
            Sw2, q2, Z2 = Sw1, q1, Z1
        else:
            Sw2 = np.einsum("Msj,jf->Msf", Sigma2, self.W)             # [M,s,nf]
            q2  = np.einsum("Mjf,jf->Mf", Sw2, self.W)                 # [M,nf]

        # For backprop through ARD/log we also need the "u" features (before ARD):
        if self._flag_logged:
            u1 = np.log(np.maximum(q1, self._eps))
            u2 = np.log(np.maximum(q2, self._eps))
            du_dq1 = 1.0 / np.maximum(q1, self._eps)
            du_dq2 = 1.0 / np.maximum(q2, self._eps)
        else:
            u1 = q1
            u2 = q2
            du_dq1 = np.ones_like(q1)
            du_dq2 = np.ones_like(q2)

        # Recompute Z from u and ARD (keeps gradients consistent if flags change)
        if self._flag_ard:
            ard = self.ard                                   # [nf]
            Z1 = u1 * ard
            Z2 = u2 * ard
        else:
            ard = np.ones(self.nf, dtype=np.float64)

        # 2) Compute K and the helpful weight term A = dL/dK ⊙ (∂K/∂Z) factors
        #    Also collect ∂L/∂η_param = sum(dL_dK ⊙ K)
        K = self._compute_kernel(Z1, Z2)                                  # [N,M]
        # η gradient (same for both kernels): dK/dη_param = K
        if self._flag_eta:
            self.eta_param.gradient = float(np.sum(dL_dK * K))

        # Derivatives wrt features (Z)
        if self._kernel_type == "Linear":
            # K = eta * (1 + Z1 @ Z2ᵀ)
            # ∂K/∂Z1 = eta * Z2ᵀ, ∂K/∂Z2 = eta * Z1ᵀ
            # So ∂L/∂Z1 = dL_dK @ (eta*Z2)
            eta = self.eta
            dL_dZ1 = dL_dK @ (eta * Z2)                                   # [N,nf]
            dL_dZ2 = dL_dK.T @ (eta * Z1)                                 # [M,nf]
        elif self._kernel_type == "RBF":
            # K = eta * exp(-0.5 * ||Z1 - Z2||^2 / nf)
            # ∂K/∂Z1 = K * (-(Z1 - Z2)/nf), ∂K/∂Z2 = K * (-(Z2 - Z1)/nf)
            # Compute difference tensor via gemm: for each pair, Z1_i - Z2_j
            # We use matrix products to avoid O(N*M*nf) explicit tensor if possible.
            # dL_dZ1[i,:] = sum_j dL_dK[i,j] * K[i,j] * (-(Z1[i]-Z2[j])/nf)
            # Implement as:
            B = dL_dK * K                                                 # [N,M]
            # sums needed:
            sumB = np.sum(B, axis=1, keepdims=True)                       # [N,1]
            dL_dZ1 = (-(1.0 / float(self.nf))) * (sumB * Z1 - B @ Z2)     # [N,nf]
            # symmetric for Z2:
            sumB2 = np.sum(B, axis=0, keepdims=True)                      # [1,M]
            dL_dZ2 = (-(1.0 / float(self.nf))) * (sumB2.T * Z2 - B.T @ Z1)# [M,nf]
        else:
            raise ValueError("Unsupported kernel type")

        # 3) ARD gradients (if enabled), via z = ard * u and ∂z/∂ard_param = ard * u
        if self._flag_ard:
            dL_dard_param = np.sum(dL_dZ1 * (ard * u1), axis=0)
            dL_dard_param += np.sum(dL_dZ2 * (ard * u2), axis=0)
            self.ard_param.gradient = dL_dard_param

        # 4) Backprop to u, then to q, then to W
        #    z = ard * u  => ∂L/∂u = ∂L/∂z * ard
        dL_du1 = dL_dZ1 * ard
        dL_du2 = dL_dZ2 * ard

        #    u = log(q) or u = q
        dL_dq1 = dL_du1 * du_dq1
        dL_dq2 = dL_du2 * du_dq2

        #    q_ip = f_pᵀ Σ_i f_p  ⇒ ∂q_ip/∂f_p = 2 Σ_i f_p
        #    Accumulate both sets (X and X2)
        G1 = 2.0 * np.einsum("Nsf,Nf->sf", Sw1, dL_dq1)                  # [s,nf]
        G2 = 2.0 * np.einsum("Msf,Mf->sf", Sw2, dL_dq2)                  # [s,nf]
        self.W.gradient = G1 + G2


    def update_gradients_diag(self, dL_dKdiag: np.ndarray, X: np.ndarray) -> None:
        """
        Compute parameter gradients from diagonal sensitivities dL_dKdiag.
        Sets gradients for eta/ard/W using diagonal-only info.
        Shapes:
        X: [N, s*s]
        dL_dKdiag: [N]
        """
        # 1) Prep
        Sigma1 = self._reshape_input(X)                                   # [N,s,s]
        Sw1 = np.einsum("Nsj,jf->Nsf", Sigma1, self.W)
        q1  = np.einsum("Njf,jf->Nf", Sw1, self.W)

        if self._flag_logged:
            u1 = np.log(np.maximum(q1, self._eps))
            du_dq1 = 1.0 / np.maximum(q1, self._eps)
        else:
            u1 = q1
            du_dq1 = np.ones_like(q1)

        if self._flag_ard:
            ard = self.ard
            Z1 = u1 * ard
        else:
            ard = np.ones(self.nf, dtype=np.float64)
            Z1 = u1

        # 2) Kdiag and η gradient
        Kdiag = self.Kdiag(X)                                              # [N,]
        if self._flag_eta:
            self.eta_param.gradient = float(np.sum(dL_dKdiag * Kdiag))

        # 3) ∂L/∂Z from diagonal structure
        if self._kernel_type == "Linear":
            # Kdiag_i = eta * (1 + ||Z_i||^2) => ∂Kdiag/∂Z_i = 2*eta*Z_i
            dL_dZ1 = (dL_dKdiag[:, None]) * (2.0 * self.eta * Z1)         # [N,nf]
        elif self._kernel_type == "RBF":
            # Kdiag_i = eta * 1 (distance is zero) ⇒ no Z dependence
            dL_dZ1 = np.zeros_like(Z1)
        else:
            raise ValueError("Unsupported kernel type")

        # 4) ARD diag gradient
        if self._flag_ard:
            # z = ard*u ⇒ ∂z/∂ard_param = ard*u
            self.ard_param.gradient = np.sum(dL_dZ1 * (ard * u1), axis=0)

        # 5) Backprop to W via q
        dL_du1 = dL_dZ1 * ard
        dL_dq1 = dL_du1 * du_dq1
        self.W.gradient = 2.0 * np.einsum("Nsf,Nf->sf", Sw1, dL_dq1)
