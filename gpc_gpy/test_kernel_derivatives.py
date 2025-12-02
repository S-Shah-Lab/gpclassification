import GPy
import numpy as np
from kernels_gpy2 import CustomKernelGPy  # adjust path if needed


def compute_abs_rel_err(grad_an, grad_fd):
    diff = np.abs(grad_an - grad_fd)
    abs_err = float(np.max(diff))

    denom = np.maximum(np.abs(grad_an), np.abs(grad_fd))
    rel = np.zeros_like(denom)
    mask = denom > 0
    rel[mask] = diff[mask] / denom[mask]
    rel_err = float(np.max(rel))

    return abs_err, rel_err

def finite_difference_param(
    kern,
    X,
    X2,
    param,
    dL_dK=None,
    dL_dKdiag=None,
    eps=1e-6,
    use_diag=False,
):
    """
    Compute finite-difference gradient for a single GPy Param

    Parameters
    ----------
    kern      : CustomKernelGPy
        Kernel to test
    X         : ndarray [N, s*s]
        First input
    X2        : ndarray [M, s*s] or None
        Second input. If None, kernel uses X2=X
    param     : GPy.core.parameterization.Param
        Parameter to perturb (e.g. kern.W, kern.ard_param, kern.eta_param)
    dL_dK     : ndarray [N, M], optional
        Sensitivity for full kernel. Required if use_diag=False
    dL_dKdiag : ndarray [N], optional
        Sensitivity for diagonal. Required if use_diag=True
    eps       : float
        Finite-difference step size
    use_diag  : bool
        If True, use Kdiag; otherwise, use full K

    Returns
    -------
    grad_fd   : ndarray, same shape as param.values
        Finite-difference approximation to dL/dparam
    """
    param_values = param.values
    grad_fd = np.zeros_like(param_values)

    it = np.nditer(param_values, flags=["multi_index"], op_flags=["readwrite"])
    for _ in it:
        idx  = it.multi_index
        orig = param_values[idx]

        # Treat -> theta + eps
        param_values[idx] = orig + eps
        if use_diag:
            Kdiag_p = kern.Kdiag(X)
            Lp = float(np.sum(dL_dKdiag * Kdiag_p)) 
        else:
            K_p = kern.K(X, X2)
            Lp  = float(np.sum(dL_dK * K_p)) # L( theta + eps )

        # Treat -> theta - eps
        param_values[idx] = orig - eps
        if use_diag:
            Kdiag_m = kern.Kdiag(X)
            Lm = float(np.sum(dL_dKdiag * Kdiag_m)) 
        else:
            K_m = kern.K(X, X2)
            Lm = float(np.sum(dL_dK * K_m)) # L( theta - eps )

        # central difference
        grad_fd[idx] = (Lp - Lm) / (2.0 * eps)

        # restore original value
        param_values[idx] = orig

    return grad_fd

def check_gradients_full(
    kern,
    X,
    X2=None,
    eps=1e-6,
    atol=1e-5,
    rtol=1e-4,
):
    """
    Check gradients from update_gradients_full for W, ard_param, eta_param

    Parameters
    ----------
    kern : CustomKernelGPy
    X    : ndarray [N, s*s]
    X2   : ndarray [M, s*s] or None
    eps  : float
        Finite-difference step
    atol : float
        Absolute tolerance
    rtol : float
        Relative tolerance

    Returns
    -------
    report : dict
        Holds per-parameter max abs and rel error
    """
    N = X.shape[0]
    if X2 is None:
        M = N
    else:
        M = X2.shape[0]

    dL_dK = np.ones((N, M))

    # Reset gradients to zero explicitly
    if hasattr(kern, "W"):
        kern.W.gradient[:] = 0.0
    if hasattr(kern, "ard_param") and kern.ard_param is not None:
        kern.ard_param.gradient[:] = 0.0
    if hasattr(kern, "eta_param") and kern.eta_param is not None:
        kern.eta_param.gradient[:] = 0.0

    # Call analytic gradients
    kern.update_gradients_full(dL_dK, X, X2)

    report = {}

    # Helper to compare analytic vs numeric
    def _compare(name, param):
        if param is None:
            return None

        grad_fd = finite_difference_param(
            kern=kern,
            X=X,
            X2=X2,
            param=param,
            dL_dK=dL_dK,
            use_diag=False,
            eps=eps,
        )
        grad_an = param.gradient 

        # Compare
        abs_err, rel_err = compute_abs_rel_err(grad_an, grad_fd)

        report[name] = {
            "abs_error": float(abs_err),
            "rel_error": float(rel_err),
            "ok": bool(abs_err < atol or rel_err < rtol),
        }

    _compare("W", kern.W)
    if hasattr(kern, "ard_param"):
        _compare("ard_param", kern.ard_param)
    if hasattr(kern, "eta_param"):
        _compare("eta_param", kern.eta_param)

    return report

def check_gradients_diag(
    kern,
    X,
    eps=1e-6,
    atol=1e-5,
    rtol=1e-4,
):
    """
    Check gradients from update_gradients_diag for W, ard_param, eta_param.

    Parameters
    ----------
    kern : CustomKernelGPy
    X : ndarray [N, s*s]
    eps : float
        Finite-difference step.
    atol : float
        Absolute tolerance.
    rtol : float
        Relative tolerance.

    Returns
    -------
    report : dict
        Holds per-parameter max abs and rel error.
    """
    N = X.shape[0]
    dL_dKdiag = np.ones(N)

    # Reset gradients
    if hasattr(kern, "W"):
        kern.W.gradient[:] = 0.0
    if hasattr(kern, "ard_param") and kern.ard_param is not None:
        kern.ard_param.gradient[:] = 0.0
    if hasattr(kern, "eta_param") and kern.eta_param is not None:
        kern.eta_param.gradient[:] = 0.0

    # Analytic
    kern.update_gradients_diag(dL_dKdiag, X)

    report = {}

    def _compare(name, param):
        if param is None:
            return None

        grad_fd = finite_difference_param(
            kern=kern,
            X=X,
            X2=None,
            param=param,
            dL_dKdiag=dL_dKdiag,
            use_diag=True,
            eps=eps,
        )
        grad_an = param.gradient

        abs_err, rel_err = compute_abs_rel_err(grad_an, grad_fd)

        report[name] = {
            "abs_error": float(abs_err),
            "rel_error": float(rel_err),
            "ok": bool(abs_err < atol or rel_err < rtol),
        }

    _compare("W", kern.W)
    if hasattr(kern, "ard_param"):
        _compare("ard_param", kern.ard_param)
    if hasattr(kern, "eta_param"):
        _compare("eta_param", kern.eta_param)

    return report

def make_random_covariances(N, s, rng):
    """
    Build N random SPD-ish covariance matrices, vectorized as [N, s*s].
    """
    mats = []
    for _ in range(N):
        A = rng.normal(size=(s, s))
        Sigma = A @ A.T + 1e-3 * np.eye(s)
        mats.append(Sigma.reshape(-1))
    return np.stack(mats, axis=0)

def run_all_tests():
    rng = np.random.default_rng(np.random.randint(1, 100))

    s  = 3 # sensors
    nf = 2 # filters
    N  = 5 # dim 1   
    M  = 4 # dim 2

    W0 = rng.normal(size=(s, nf)) # Initialize spatial filter matrix

    # Generate random covariances for the test
    X  = make_random_covariances(N, s, rng)
    X2 = make_random_covariances(M, s, rng)

    # List of configurations to run 
    configs = [
        {"kernel_type": "Linear", "ard_flag": True,  "eta_flag": True,  "logged_flag": True },
        {"kernel_type": "Linear", "ard_flag": False, "eta_flag": True,  "logged_flag": True },
        {"kernel_type": "RBF",    "ard_flag": True,  "eta_flag": True,  "logged_flag": True },
        {"kernel_type": "RBF",    "ard_flag": True,  "eta_flag": False, "logged_flag": False},
    ]

    results = []
    for cfg in configs:
        kern = CustomKernelGPy(
            W=W0.copy(),
            W_trainable=True,
            ard_flag=cfg["ard_flag"],
            eta_flag=cfg["eta_flag"],
            logged_flag=cfg["logged_flag"],
            kernel_type=cfg["kernel_type"],
        )

        rep_full = check_gradients_full(kern, X, X2) # Run test for gradients 
        rep_diag = check_gradients_diag(kern, X)     # Run test for gradients diagonal

        results.append((cfg, rep_full, rep_diag))

    return results

if __name__ == "__main__":
    
    results = run_all_tests()
    
    for cfg, rep_full, rep_diag in results:
        print("Config:", cfg)
        print("  Full gradients:")
        for name, r in rep_full.items():
            print(f"    {name}: abs={r['abs_error']:.3e}, rel={r['rel_error']:.3e}, ok={r['ok']}")
        print("  Diag gradients:")
        for name, r in rep_diag.items():
            print(f"    {name}: abs={r['abs_error']:.3e}, rel={r['rel_error']:.3e}, ok={r['ok']}")
        print()
