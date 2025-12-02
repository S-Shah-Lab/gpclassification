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
            
        
    def _update_gradient_eta(self, dL_dK: np.ndarray, K: np.ndarray) -> np.array:
        """
        TODO
        """
        return np.array( np.sum(dL_dK * K),  dtype=np.float64 )
        
        
    def _update_gradient_eta_diag(self, dL_dKdiag: np.array, Kdiag: np.array) -> np.array:
        """
        TODO
        """
        return np.array( np.sum(dL_dKdiag * Kdiag),  dtype=np.float64 )  
         
    
    def _update_gradient_ard(
        self,
        dL_dK: np.ndarray,
        K    : np.ndarray,
        wSw1 : np.ndarray,
        wSw2 : np.ndarray,
    ) -> np.ndarray:
        """
        TODO
        """
        ard = self.ard # [nf,]
        Z1 = wSw1      # [N, nf], post-ARD
        Z2 = wSw2      # [M, nf] or [N, nf], post-ARD
        V1 = Z1 / ard  # [N, nf], pre-ARD
        V2 = Z2 / ard  # [M, nf] or [N, nf], pre-ARD

        if self._kernel_type == "Linear":
            # Note that we refer to exp(eta) and exp(ard_p) as eta and ard_p
            # Linear kernel: K = eta * (Z1 Z2^T + 1)
            DL = dL_dK                       # [N, M]
            DL_Z2 = DL @ Z2                  # [N, nf]
            DL_V2 = DL @ V2                  # [N, nf]

            # For each feature p:
            # dL/d(ard_param_p) = eta * ard_p * ( sum_i V1_ip * (DL_Z2_ip) + sum_i Z1_ip * (DL_V2_ip) )
            term1 = np.sum(V1 * DL_Z2, axis=0)  # [nf,]
            term2 = np.sum(Z1 * DL_V2, axis=0)  # [nf,]
            
            grad_param = self.eta * ard * (term1 + term2)
            return grad_param

        elif self._kernel_type == "RBF":
            # RBF kernel: K_ij = eta * exp( -||z_i - z_j||^2 / (2 * nf) )
            N, nf = Z1.shape
            grad_param = np.zeros(nf, dtype=np.float64)

            for p in range(nf):
                Z1p = Z1[:, p][:, None]      # [N, 1]
                Z2p = Z2[:, p][None, :]      # [1, M]
                V1p = V1[:, p][:, None]      # [N, 1]
                V2p = V2[:, p][None, :]      # [1, M]

                dZ = Z1p - Z2p               # [N, M]
                dV = V1p - V2p               # [N, M]

                # Contribution at feature p
                contrib = dL_dK * K * (dZ * dV)  # [N, M]
                grad_ard = -(1.0 / float(self.nf)) * np.sum(contrib)

                grad_param[p] = ard[p] * grad_ard

            return grad_param

        else:
            raise ValueError(f"Unsupported kernel type: {self._kernel_type}")
  
  
    def _update_gradient_ard_diag(
        self,
        dL_dKdiag: np.ndarray,
        X        : np.ndarray,
    ) -> np.ndarray:
        """
        TODO
        """
        # Recompute features for X
        Sigma1 = self._reshape_input(X)          # [N, s, s]
        wSw1   = self._compute_features(Sigma1)  # [N, nf], post-ARD

        ard = self.ard                           # [nf,]
        Z1  = wSw1                               # [N, nf], post-ARD
        V1  = Z1 / ard                           # [N, nf], pre-ARD

        if self._kernel_type == "Linear":
            # Kdiag_i = eta * (||z_i||^2 + 1)
            # dKdiag_i/d(ard_p) = 2 * eta * z_ip * v_ip
            tmp = dL_dKdiag[:, None] * Z1 * V1   # [N, nf]
            grad_param = 2.0 * self.eta * ard * np.sum(tmp, axis=0)
            return grad_param

        elif self._kernel_type == "RBF":
            # Kdiag_i = eta, independent of ARD
            return np.zeros(self.nf, dtype=np.float64)

        else:
            raise ValueError(f"Unsupported kernel type: {self._kernel_type}")
     
       
    def _update_gradient_W(
        self,
        dL_dK: np.ndarray,
        K: np.ndarray,
        Sigma1: np.ndarray,
        Sigma2: np.ndarray,
        wSw1: np.ndarray,
        wSw2: np.ndarray,
    ) -> np.ndarray:
        """
        TODO
        """
        # If W is fixed, gradient can be zeroed, but computing it is harmless
        W = np.asarray(self.W, dtype=np.float64)    # [s, nf]
        ard = self.ard                              # [nf,]

        z1 = wSw1                                   # [N, nf]
        z2 = wSw2                                   # [M, nf]

        N, s, _ = Sigma1.shape
        M = Sigma2.shape[0]

        # Step 1: dL/dz1 and dL/dz2 from the kernel
        if self._kernel_type == "Linear":
            # dL/dz1 = eta * (dL_dK @ z2)
            # dL/dz2 = eta * (dL_dK.T @ z1)
            dL_dz1 = self.eta * (dL_dK   @ z2)      # [N, nf]
            dL_dz2 = self.eta * (dL_dK.T @ z1)      # [M, nf]

        elif self._kernel_type == "RBF":
            # RBF case: use K and distances
            dL_dz1 = np.zeros_like(z1, dtype=np.float64)
            dL_dz2 = np.zeros_like(z2, dtype=np.float64)

            for p in range(self.nf):
                Z1p = z1[:, p][:, None]             # [N, 1]
                Z2p = z2[:, p][None, :]             # [1, M]
                diff = Z1p - Z2p                    # [N, M]

                # For z1: - (d_ij / nf), for z2: + (d_ij / nf)
                common = dL_dK * K * (diff / float(self.nf))  # [N, M]

                # Sum over partner index
                dL_dz1[:, p] += -np.sum(common, axis=1)       # [N]
                dL_dz2[:, p] +=  np.sum(common, axis=0)       # [M]

        else:
            raise ValueError(f"Unsupported kernel type: {self._kernel_type}")

        # Step 2: backprop through ARD scaling: z = ard * v
        if self._flag_ard:
            dL_dv1 = dL_dz1 * ard         # [N, nf]
            dL_dv2 = dL_dz2 * ard         # [M, nf]
        else:
            dL_dv1 = dL_dz1
            dL_dv2 = dL_dz2

        # Step 3: backprop through log / identity to q (raw quadratic forms)
        # Recompute raw q1, q2 and the symmetric term (Sigma + Sigma^T) W
        Sw1 = np.einsum("Nsj,jf->Nsf", Sigma1, W)   # Σ_i W, [N, s, nf]
        Sw2 = np.einsum("Msj,jf->Msf", Sigma2, W)   # Σ_j W, [M, s, nf]

        Stw1 = np.einsum("Njs,jf->Nsf", Sigma1, W)  # Σ_i^T W, [N, s, nf]
        Stw2 = np.einsum("Mjs,jf->Msf", Sigma2, W)  # Σ_j^T W, [M, s, nf]

        S_W1 = Sw1 + Stw1                           # (Σ_i + Σ_i^T) W, [N, s, nf]
        S_W2 = Sw2 + Stw2                           # (Σ_j + Σ_j^T) W, [M, s, nf]

        # Quadratic forms q1 = w^T Σ_i w, q2 = w^T Σ_j w
        q1 = np.einsum("Njf,jf->Nf", Sw1, W)        # [N, nf]
        q2 = np.einsum("Mjf,jf->Mf", Sw2, W)        # [M, nf]

        if self._flag_logged:
            # dv/dq = 1/q for q > eps, else 0
            dv_dq1 = np.zeros_like(q1, dtype=np.float64)
            dv_dq2 = np.zeros_like(q2, dtype=np.float64)

            mask1 = q1 > self._eps
            mask2 = q2 > self._eps

            dv_dq1[mask1] = 1.0 / q1[mask1]
            dv_dq2[mask2] = 1.0 / q2[mask2]

            dL_dq1 = dL_dv1 * dv_dq1            # [N, nf]
            dL_dq2 = dL_dv2 * dv_dq2            # [M, nf]
        else:
            dL_dq1 = dL_dv1
            dL_dq2 = dL_dv2

        # Step 4: accumulate gradient w.r.t. W
        # grad_W1[r, f] = sum_i dL_dq1[i, f] * S_W1[i, r, f]
        grad_W1 = np.einsum("Nf,Nsf->sf", dL_dq1, S_W1)  # [s, nf]

        # grad_W2[r, f] = sum_j dL_dq2[j, f] * S_W2[j, r, f]
        grad_W2 = np.einsum("Mf,Msf->sf", dL_dq2, S_W2)  # [s, nf]

        grad_W = grad_W1 + grad_W2

        return grad_W
       
    
    def _update_gradient_W_diag(
        self,
        dL_dKdiag: np.ndarray,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        TODO
        """
        W = np.asarray(self.W, dtype=np.float64)       # [s, nf]
        ard = self.ard                                 # [nf,]

        # Reshape input and recompute features z (after log + ARD)
        Sigma1 = self._reshape_input(X)                # [N, s, s]
        z1 = self._compute_features(Sigma1)            # [N, nf]

        if self._kernel_type == "RBF":
            # Kdiag_i = eta, independent of z and W
            return np.zeros_like(W, dtype=np.float64)

        if self._kernel_type != "Linear":
            raise ValueError(f"Unsupported kernel type: {self._kernel_type}")

        # Step 1: dL/dz from the diagonal of the linear kernel
        # Kdiag_i = eta * (||z_i||^2 + 1)
        # dKdiag_i/dz_{i,f} = 2 * eta * z_{i,f}
        dL_dKdiag = dL_dKdiag.reshape(-1)              # [N,]
        factor = (2.0 * self.eta) * dL_dKdiag          # [N,]
        dL_dz1 = factor[:, None] * z1                  # [N, nf]

        # Step 2: backprop through ARD scaling: z = ard * v
        if self._flag_ard:
            dL_dv1 = dL_dz1 * ard                      # [N, nf]
        else:
            dL_dv1 = dL_dz1

        # Step 3: backprop through log / identity to q
        Sw1 = np.einsum("Nsj,jf->Nsf", Sigma1, W)      # Σ_i W, [N, s, nf]
        Stw1 = np.einsum("Njs,jf->Nsf", Sigma1, W)     # Σ_i^T W, [N, s, nf]
        S_W1 = Sw1 + Stw1                              # (Σ_i + Σ_i^T) W

        q1 = np.einsum("Njf,jf->Nf", Sw1, W)           # [N, nf]

        if self._flag_logged:
            dv_dq1 = np.zeros_like(q1, dtype=np.float64)
            mask1 = q1 > self._eps
            dv_dq1[mask1] = 1.0 / q1[mask1]

            dL_dq1 = dL_dv1 * dv_dq1                  # [N, nf]
        else:
            dL_dq1 = dL_dv1

        # Step 4: accumulate gradient w.r.t. W
        # grad_W[r, f] = sum_i dL_dq1[i, f] * S_W1[i, r, f]
        grad_W = np.einsum("Nf,Nsf->sf", dL_dq1, S_W1)

        return grad_W
       
            
    def update_gradients_full(self, dL_dK: np.ndarray, X: np.ndarray, X2: Optional[np.ndarray] = None) -> None:
        """
        TODO
        """
        # Evaluate features
        Sigma1, Sigma2, wSw1, wSw2 = self._prep_elements(X, X2)
        # Sigma1: [N, s, s], wSw1: [N, nf]
        # Sigma2: [M, s, s], wSw2: [M, nf]
        
        # Generate the custom covariance function
        K = self._compute_kernel(wSw1, wSw2) # [N, M] if X2 is not None else [N, N]
        
        # Gradient operations  
        if self._flag_eta:
            # dL/d(eta_param) = sum_{i,j} dL_dK[i,j] * dK[i,j]/d(eta_param)
            #                 = sum_{i,j} dL_dK[i,j] * K[i,j]
            self.eta_param.gradient = self._update_gradient_eta(
                dL_dK=dL_dK, 
                K=K,
            )
            
        if self._flag_ard:
            # dL/d(ard_param_p) = dL/d(ard_p) * d(ard_p)/d(ard_param_p) = dL/d(ard_p) * ard_p
            # dL/d(ard_p) = dL/dK * dK/d(ard_p)
            # For this we need to consider different kernel types (linear or RBF)
            self.ard_param.gradient = self._update_gradient_ard(
                dL_dK=dL_dK,
                K=K,
                wSw1=wSw1,
                wSw2=wSw2,
            )
            
        self.W.gradient = self._update_gradient_W(
            dL_dK=dL_dK,
            K=K,
            Sigma1=Sigma1,
            Sigma2=Sigma2,
            wSw1=wSw1,
            wSw2=wSw2,
        )
        
        
    def update_gradients_diag(self, dL_dKdiag: np.ndarray, X: np.ndarray) -> None:
        """
        TODO
        """
        # Diagonal of the covariance function
        Kdiag = self.Kdiag(X)  # shape [N,]

        if self._flag_eta:
            # dL/d(eta_param) = sum_i dL_d(Kdiag[i]) * Kdiag[i]
            self.eta_param.gradient = self._update_gradient_eta_diag(
                dL_dKdiag=dL_dKdiag, 
                Kdiag=Kdiag,
            )
        
        if self._flag_ard:
            self.ard_param.gradient = self._update_gradient_ard_diag(
                dL_dKdiag=dL_dKdiag, 
                X=X,
            )
            
        self.W.gradient = self._update_gradient_W_diag(
            dL_dKdiag=dL_dKdiag,
            X=X,
        )
        