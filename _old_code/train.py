import gpflow
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from kernels import CustomKernel


def train_model(
    X: np.ndarray,
    Y: np.ndarray,
    W_init: np.ndarray,
    ard_flag: bool = False,
    eta_flag: bool = False,
    logged_flag: bool = True,
    kernel_type: str = "RBF",
    maxiter: int = 1000,
) -> tuple:
    """
    Train a GPR with CustomKernel, recording parameter histories
        X: Input of shape [N, s*s], independent variable
        Y: Input of shape [N, 1], dependent variable
        W_init: Input matrix of shape [s, nf]

    Returns:
        model: trained GPR model
        kernel: last iteration version of the kernel
        w_arr: np.array of W matrices per iteration
        eta_arr: np.array of eta scalars per iteration
        ard_arr: np.array of ARD vectors per iteration
    """

    # Define the kernel by using the custom kernel
    kernel = CustomKernel(W_init, ard_flag, eta_flag, logged_flag, kernel_type)
    # Initialize the model with the data and custom kernel
    # Define the likelihood to use in the classifier
    # p(y=1 ∣ f) = σ(f)
    # p(y=0 ∣ f) = 1 − σ(f)
    likelihood = gpflow.likelihoods.Bernoulli()
    # Initialize the model with the data and custom kernel
    model = gpflow.models.VGP(
        data=(X, Y),
        kernel=kernel,
        likelihood=likelihood,
        num_latent_gps=1,  # For binary classification we need one latent GP
    )

    # Initialize lists to keep track of parameters evolution over the iterations
    w_arr, eta_arr, ard_arr = [], [], []
    y_train_pred_arr = []

    def step_callback(step, variables, values):
        """
        Define a callable that gets called once after each optimisation step
            step: is the optimisation step counter
            variables: is the list of trainable variables
            values: is the list of tensors of matching shape that contains their value at this optimisation step
        """
        probs, _ = model.predict_y(X)  # predictive P(y=1|x)
        y_train_pred_arr.append((probs.numpy() >= 0.5).astype(int).flatten())
        w_arr.append(kernel.W.numpy().copy())
        if kernel.eta is not None:
            eta_arr.append(kernel.eta.numpy())
        if kernel.ard is not None:
            ard_arr.append(kernel.ard.numpy().copy())

    # Run minimization step
    # By default the GPClassification minimizes the negative variational evidence lower bound (ELBO) on the likelihood
    # E_{q(F)} [ log p(Y|F) ] - KL[ q(F) || p(F)]
    # Replace SciPy optimization (which gave problems in classification) with Adam optimizer
    optimizer = tf.optimizers.Adam(learning_rate=0.01)

    # Remove tf.function decorator to ensure eager updates and proper gradient tracking
    def train_step():
        with tf.GradientTape() as tape:
            loss = model.training_loss()
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    # Perform fixed number of training steps, recording loss and parameters
    for step in range(maxiter):
        loss = train_step()
        step_callback(step, None, None)
        # print(f"Step {step}, Loss: {loss.numpy():.4f}")

    y_train_pred_arr = np.stack(y_train_pred_arr)  # shape [iters, N]
    w_arr = np.stack(w_arr)  # shape [iters, s, nf]
    if kernel.eta is not None:
        eta_arr = np.array([eta_arr])  # shape [iters, ]
    if kernel.ard is not None:
        ard_arr = np.stack(ard_arr)  # shape [iters, nf]

    return model, kernel, y_train_pred_arr, w_arr, eta_arr, ard_arr
