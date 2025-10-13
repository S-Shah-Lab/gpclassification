import gpflow
import tensorflow as tf
import numpy as np
from gpflow.utilities import positive, to_default_float


class CustomKernel(gpflow.kernels.Kernel):
    def __init__(
        self,
        W: np.ndarray,
        W_trainable: bool = True,
        ard_flag: bool = True,
        eta_flag: bool = True,
        logged_flag: bool = True,
        kernel_type: str = "RBF",
    ):
        """
        Custom GPflow kernel operating on covariance matrices with spatial filtering
            W: Spatial filter matrix of shape [s, nf]
            W_trainable: if False, keep W fixed during training
            ard_flag: Use ARD scaling per feature
            eta_flag: Use a global scaling
            logged_flag: log-transform features
            kernel_type: "Linear" or "RBF"
        """
        # As a child class, grab all the methods from the parent class
        super().__init__()
        # Define dimensions from input spatial filter matrix
        self.s = W.shape[0]  # Number of sensors
        self.nf = W.shape[1]  # Number of spatial filters, cols of W

        # Define flags as for input
        self._flag_ard = ard_flag  # Individual feature scaling
        self._flag_eta = eta_flag  # Global scaling
        self._flag_logged = logged_flag  # Logged features
        self._kernel_type = kernel_type  # Type of kernel to use

        # GPflow parameter
        # Spatial filter matrix
        # W has shape [s, nf], and its p column is called w
        self.W = gpflow.Parameter(W, trainable=W_trainable)

        # Automatic Relevance Detection (ARD) determines scaling of each feature independently
        # Initialize parameter at 1.0, allows constrained representation forcing it to be positive
        # This is a parameter for each filter of the spatial filter matrix, nf parameters in total
        self.ard = (
            gpflow.Parameter(np.ones(self.nf), transform=positive())
            if self._flag_ard
            else None
        )

        # Global variance parameter determines global scaling
        # Initialize parameter at 10, allows constrained representation forcing it to be positive
        # This parameter is unique
        self.eta = (
            gpflow.Parameter(1.0, transform=positive()) if self._flag_eta else None
        )

    @property
    def input_dim(self) -> int:
        # Set input dimensions manually otherwise GPflow is not happy
        # This is not necessary now I think, but it doesn't hurt
        return self.s * self.s

    def _reshape_input(self, X: tf.Tensor) -> tf.Tensor:
        """
        Reshape flat inputs back into covariance matrices
            X: Tensor of shape [N, s*s], previously flattened

        return: Tensor of shape [N, s, s]
        """
        # Reshape input tensor to original form
        # Due to dimensionality problematics, the input has been reshaped to match [N, s*s]
        # Here we reshape it to its original form [N, s, s]
        return tf.reshape(X, [-1, self.s, self.s])

    def _compute_features(self, Sigma: tf.Tensor) -> tf.Tensor:
        """
        Transform the input covariance matrices to the linear or logged filtered variances: computes features for each trial
            Sigma: Covariance matrices of shape [N, s, s]

        return: Linear or logged filtered variances of shape [N, nf]
        """
        # This is where the input feature Sigma (covariance matrix) is transformed by the spatial filters for our model
        # The assumption is: tf.shape(Sigma) = [N, s, s]
        #   N: number of trials
        #   s: number of sensors

        # Ensure dtype consistency with GPflow default float and params
        Sigma = to_default_float(Sigma)
        if Sigma.dtype != self.W.dtype:
            Sigma = tf.cast(Sigma, self.W.dtype)
        
        # These changes might help in case of Cholesky decomposition problematics
        # if matrix is not symmetric --> Symmetrize (this is very slow)
        # Sigma = 0.5*(Sigma + tf.transpose(Sigma, perm=[0,2,1]))
        # if matrix is getting values at 0 --> Tiny ridge on diagonal
        # Sigma = Sigma + tf.eye(self.s, dtype=Sigma.dtype)[None, :, :] * 1e-12

        # Applies Σi @ w for each i with i being trial number
        # This step performs the operation for each spatial filter column of index p
        # Sigma has shape [N, s, s]
        # self.W has shape [s, nf]
        Sw = tf.matmul(Sigma, self.W)  # [N, s, nf]
        # Applies w @ Σi @ w for each i with i being trial number
        # Sw has shape [N, s, nf]
        # self.W[None, :, :] has shape [1, s, nf]
        wSw = tf.reduce_sum(self.W[None, :, :] * Sw, axis=1)  # [N, nf]
        # Each column of wSw contains the quadratic form for each covariance matrix (N) with each filter column (nf)

        """
        # This is the equivalent to do w @ Sigma @ w with more steps
        # It is an alternative method that leads to the same final result for wSw
        
        N = tf.shape(Sigma)[0] # Number of trials
        # Flatten Sigma
        Sigma_flat = tf.reshape(Sigma, [N, self.s * self.s])  # [N, s*s]
        # Form outer product for each filter w * w.T
        W_outer = tf.einsum("ik,jk->ijk", self.W, self.W)  # [s, s, nf]
        # Flatten outer product
        W_flat = tf.reshape(W_outer, [self.s * self.s, self.nf])  # [s*s, nf]
        # Evaluate w.T * Sigma * w for each filter column
        wSw = tf.matmul(Sigma_flat, W_flat)  # [N, nf]
        """
        if self._flag_logged:
            # Log the resulting features with guard for 0 values
            eps = tf.cast(1e-12, wSw.dtype)
            wSw = tf.math.log(tf.maximum(wSw, eps))
            # In case of numerical stability
            # Note here: the covariance matrices in input should not be given in V^2
            # That entails a very low order of magnitude ~1e-12/1e-14 which is way lower than 1e-7 which is added for numerical stability
            # If instead the covariance matrices are in uV^2 the 1e-7 factor is not a problem
            # wSw = tf.math.log(wSw + 1e-7)  # Just in case, for numerical stability
        return wSw

    def _compute_kernel(self, wSw: tf.Tensor, wSw2: tf.Tensor) -> tf.Tensor:
        """
        Compute custom kernel between feature sets wSw and wSw2
            wSw: Features of shape [N, nf]
            wSw2: Features of shape [M, nf]

        return: Kernel matrix shape [N, M]
        """
        # Define the custom kernel, the type is expressed by a flag at initializzation
        if self._kernel_type == "Linear":
            # Perform z.T * z
            Kmat = tf.matmul(wSw, tf.transpose(wSw2))
            # Add bias term to the result
            Kmat += 1
            return Kmat

        elif self._kernel_type == "RBF":
            # Perform the square distances
            wSw_sq = tf.reduce_sum(tf.square(wSw), axis=1, keepdims=True)  # [N, 1]
            wSw2_sq = tf.reduce_sum(
                tf.square(wSw2), axis=1, keepdims=True
            )  # [M, 1] if wSw != wSw2 else [N,1]
            dist_sq = (
                wSw_sq - 2 * tf.matmul(wSw, tf.transpose(wSw2)) + tf.transpose(wSw2_sq)
            )  # [N,M] else [N,N]
            # Divide by 2 * nf and put into exponential
            Kmat = tf.exp(-0.5 * dist_sq / tf.cast(self.nf, tf.float64))
            return Kmat

        else:
            raise ValueError(f"Unsupported kernel type: {self._kernel_type}")

    def K(self, X: tf.Tensor, X2: tf.Tensor = None) -> tf.Tensor:
        """
        Compute the (noiseless) covariance matrix
            X: Inputs of shape [N, s*s], this is a series of flat covariance matrices
            X2: Optional inputs of shape [M, s*s] else it becomes X of shape [N, s*s], same concept as X

        return: Covariance matrix of shape [N, N] or [N, M]
        """
        # This is where you implement the kernel function itself
        # This takes two arguments, X and X2. By convention, we make the second argument optional (it defaults to None)
        # Inside K, all the computation must be done with TensorFlow

        # Reshape flat covariance matrices in input from [N, s*s] to [N, s, s]
        Sigma = self._reshape_input(X)
        Sigma2 = self._reshape_input(X2) if X2 is not None else None

        # Transform the input features
        # The input covariance matrices are transformed to the linear or logged filtered variances
        wSw = self._compute_features(Sigma)
        wS2w = self._compute_features(Sigma2) if Sigma2 is not None else wSw

        if self._flag_ard:
            # Multiply scaling for each filter col by broadcasting across the rows automatically
            # If self._flag_ard is True, a parameter called self.ard is created and the framework will optimize it by default
            # This option can be turned off to maintain the parameter constant using gpflow.set_trainable(self.ard, False)
            wSw = wSw * tf.exp(self.ard)
            wS2w = wS2w * tf.exp(self.ard)
        else:
            pass

        # Generate the custom covariance function
        # This function treats both cases in which X == X2 or X != X2
        Kmat = self._compute_kernel(wSw, wS2w)

        if self._flag_eta:
            # Multiply by global scaling
            # If self._flag_eta is True, a parameter called self.eta is created and the framework will optimize it by default
            # This option can be turned off to maintain the parameter constant using gpflow.set_trainable(self.eta, False)
            return tf.exp(self.eta) * Kmat
        else:
            return Kmat

    def K_diag(self, X: tf.Tensor) -> tf.Tensor:
        """
        Compute diagonal of covariance matrix for inputs X
            X: Inputs shape [N, s*s]

        return: Tensor length N
        """
        # Reshape flat inputs from [N, s*s] to [N, s, s]
        Sigma = self._reshape_input(X)
        # Transform the input features
        wSw = self._compute_features(Sigma)

        if self._flag_ard:
            # Multiply scaling for each filter col by broadcasting across the rows automatically
            wSw = wSw * tf.exp(self.ard)

        # Generate the custom covariance function
        Kfull = self._compute_kernel(wSw, wSw)

        if self._flag_eta:
            # Multiply by global scaling
            Kfull = tf.exp(self.eta) * Kfull

        # Return diagonal
        return tf.linalg.diag_part(Kfull)
