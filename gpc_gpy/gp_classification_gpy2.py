"""
GP classification with a custom kernel in GPy using EP approximation on covariance matrices
"""

from __future__ import annotations

# type hint is the concept of addying type information to the code
# type annotation is the syntax used (:) or (->) to implement the type hint concept
# type annotation is a way to clarify the expected type of variables and parameters
# Python interpreter stores type annotations as strings (more concise type hint syntax)
# Allows the use of type hints for types that are defined later in the module preventin NameError

# ---------------------- Imports ----------------------
import json
import math
from copy import deepcopy
import datetime as dt
from dataclasses import dataclass, asdict
from itertools import combinations

# dataclass: decorator that examines a class to get the fields (class variable with a type annotations)
# asdict: method that converts the dataclass object into a dictionary (grabs fields and stores them into a dict)

from pathlib import Path

# Path: represents paths to files as objects which have methods (contrary to os.path which represents them as strings)

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)  # runtime support for type hints


# ---------------------- Third party libraries (mandatory) ----------------------
import GPy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from scipy.interpolate import griddata
from scipy.linalg import svd
from scipy.sparse.linalg import svds
from kernels_gpy2 import CustomKernelGPy  # custom covariance function

# ---------------------- Third party libraries (optionals for extra glitters) ----------------------
try:
    import mne  # handles EEG specific objects like montage
    from mne.channels import make_dig_montage

    HAS_MNE = True
except Exception:
    HAS_MNE = False


# -------------------------- Type aliases for annotations ---------------
# Define a type hint which could be an array or a dict
ArrayOrDict = Union[np.ndarray, Dict[str, np.ndarray]]


# -------------------------- Light dataclasses --------------------------
@dataclass
class IterLog:
    """
    Per-iteration logging
    """

    step:         int
    # Train set
    nlml:         float  # negative log marginal likelihood
    nlpd_train:   Optional[float]  # negative log probability density
    acc_train:    Optional[float]  # accuracy
    brier_train:  Optional[float]  # brier's score
    aucroc_train: Optional[float]  # area under the curve ROC
    aucpr_train:  Optional[float]  # area under the curve precision-recall
    # Validation set
    nlpd_val:     Optional[float]
    acc_val:      Optional[float]
    brier_val:    Optional[float]
    aucroc_val:   Optional[float]
    aucpr_val:    Optional[float]
    # Test set
    nlpd_test:    Optional[float]
    acc_test:     Optional[float]
    brier_test:   Optional[float]
    aucroc_test:  Optional[float]
    aucpr_test:   Optional[float]
    # Kernel
    W:            List[List[float]]  # spatial filter weights
    eta:          Optional[float]  # global scaling
    ard:          Optional[List[float]]  # per-filter scaling

@dataclass
class RunLog:
    """
    Container for the entire run logs, converted to JSON format
    """

    meta:          Dict[str, Any]     # config information
    logs:          List[IterLog]      # per-iteration metrics, learning rates, kernel parameters
    # Train set
    #p_train_seq:  List[List[float]]  # predicted probabilties (one list per iteration)
    #p_train_best: List[float]        # predicted probabilities (last iteration only)
    #y_train_seq:  List[List[int]]    # label sequences (one list per iteration)
    #y_train_best: List[int]          # label sequences (one list per iteration)
    # Validation set
    #p_val_seq:    List[List[float]]
    #p_val_best:   List[float]
    #y_val_seq:    List[List[int]]
    #y_val_best:   List[int]
    # Test set
    #p_test_seq:   List[List[float]]
    #p_test_best:  List[float]
    #y_test_seq:   List[List[int]]
    #y_test_best:  List[int]
    p_train_best:  List[float]        # Train set
    y_train_best:  List[int]
    p_val_best:    List[float]        # Validation set
    y_val_best:    List[int]
    p_test_best:   List[float]        # Test set
    y_test_best:   List[int]

# -------------------------- Utility helpers --------------------------
def _ensure_dir(p: Path) -> None:
    """
    Create folder and parents if they do not exist
    If parents is True, any missing parents of this path are created as needed
    If exist_ok is False, FileExistsError is raised if the target directory already exists
    """
    p.mkdir(parents=True, exist_ok=True)


def _now_stamp(mode: str = "") -> str:
    """
    Return a timestamp string YYYYMMDD_HHMMSS
    This is used as label for run folder names so they do not overwrite
    """
    if mode == "nice":
        return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


# =============================================================================
# ============================== Main Class ===================================
# =============================================================================
class GPClassificationRunner:
    """
    Handles the entire classification process trying to be as generic as possible
    From data loading, training, logging, and visual outputs
    """

    def __init__(
        self,
        # Input variables
        X: ArrayOrDict,
        Y: ArrayOrDict,
        dataset_label: str,
        ch_names: List[str],                   # Names of EEG channels
        ch_xy: Dict[str, Tuple[float, float]], # Coordinates of EEG channels
        # Model / kernel
        spatialFilter_init: str = "random",    # 'random' | 'ones' | 'focused'
        nf: int = 2,                           # Number of spatial filter cols
        eta_flag: bool = False,
        ard_flag: bool = False,
        W_trainable: bool = True,
        logged_flag: bool = True,
        kernel_type: str = "RBF",
        # Training
        maxiter: int = 1000,          # Number of max iterations to perform, by default perform all of them
        pred_threshold: float = 0.5, 
        random_state: int = 42,
        enable_early_stopping: bool = False,
        # ----- New data split controls (only for array inputs)
        frac_val: float = 0.2,
        frac_test: float = 0.2,
        # Run naming / Logging
        results_dir: str = "./results",
        run_name: Optional[str] = None,
    ) -> None:
        # Store I/O
        self.X = X
        self.Y = Y
        self.dataset_label = dataset_label
        self.ch_names = [c.lower() for c in ch_names]  # enforce lower-case for channel lookup
        self.ch_xy = {k.lower(): v for k, v in ch_xy.items()}

        if HAS_MNE:
            # Build montage for visualization of spatial filter
            self.montage_info = self._build_montage_from_xy(self.ch_names, self.ch_xy)

        # Store model / kernel choices
        self.spatialFilter_init = spatialFilter_init
        self.nf = nf
        self.eta_flag = eta_flag
        self.ard_flag = ard_flag
        self.W_trainable = W_trainable
        self.logged_flag = logged_flag
        self.kernel_type = kernel_type

        # Store training config
        self.maxiter = maxiter
        self.pred_threshold = pred_threshold
        self.random_state = random_state
        self.enable_early_stopping = enable_early_stopping
        # Split mode in case X and Y are arrays
        self.frac_val = 0 if frac_val is None else float(frac_val)
        self.frac_test = 0 if frac_test is None else float(frac_test)

        # Store Run naming / Logging
        self.results_root = Path(results_dir)
        self.run_name = run_name or f"run_{_now_stamp()}"
        self.run_dir = self.results_root / self.run_name
        _ensure_dir(self.run_dir)  # Create folder

        # Placeholders updated by `_load_and_prepare_data`
        self.has_train = False
        self.has_val   = False
        self.has_test  = False

        # Optional training flags
        self.use_validation_for_adaptation: bool = False

        self.s: int       = 0  # number of EEG sensors
        self.N_train: int = 0
        self.N_val: int   = 0
        self.N_test: int  = 0

        self.X_train: Optional[np.ndarray]     = None  # (N_train, D)
        self.X_val: Optional[np.ndarray]       = None  # (N_val,   D)
        self.X_test: Optional[np.ndarray]      = None  # (N_test,  D)
        self.Y_train: Optional[np.ndarray]     = None  # (N_train, 1)
        self.Y_val: Optional[np.ndarray]       = None  # (N_val,   1)
        self.Y_test: Optional[np.ndarray]      = None  # (N_test,  1)

        self.W_init: Optional[np.ndarray]      = None  # (s, nf)
        self.model: Optional[GPy.models.Model] = None
        self.kernel: Optional[CustomKernelGPy] = None

        self.run_log: Optional[RunLog] = None

        # Best-checkpoint tracking
        self._best_score: float               = float("inf")
        self._best_iter: Optional[int]        = None
        self._best_metric_name: Optional[str] = None
        self._best_params: Optional[Dict]     = None  # parameter dict snapshot

        self._p_train_best = None
        self._p_val_best   = None
        self._p_test_best  = None
        self._y_train_best = None
        self._y_val_best   = None
        self._y_test_best  = None

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~ High level method ~~~~~~~~~~~~~~~
    def fit(self) -> None:
        """
        Execute the full pipeline
        """
        self._print_message(which="start") ###################################### START
        self._create_config_file()
        # Generates -> self.cfg

        ######################################################################### PREPARATION PHASE
        self._load_and_prepare_data() 
        # Generates -> self.X_train,   self.X_val,   self.X_test
        #              self.Y_train,   self.Y_val,   self.Y_test
        #              self.has_train, self.has_val, self.has_test
        #              self.N_train,   self.N_val,   self.N_test
        self._initialize_W_matrix()    
        # Generates -> self.W_init
        self._build_model()
        # Generates -> self.kernel, self.likelihood, self.model

        ######################################################################### TRAIN / LOG PHASE
        self._train()
        self._build_and_write_runlog()  # Build and write RunLog

        ######################################################################### PLOTTING PHASE
        self._make_visual_summary()  # Visual outputs
        self._print_message(which="end") ######################################## END

    def _make_visual_summary(self) -> None:
        """
        Generate visual outputs as summary of the training process (PNGs + GIFs)
        """
        self.colors = {"train": "dodgerblue", "val": "forestgreen", "test": "orangered"}

        self._plot_learning_curves()
        self._plot_threshold_sweep()
        self._plot_calibration_curves()
        self._plot_kernel_scaling()
        self._plot_kernel_W()

        self._plot_topomap()

        self._plot_confusion_matrix()

        pairs = combinations(range(self.nf), 2)
        if self.nf == 2:
            for pair in pairs:
                self.feature_pair = pair
                self._plot_features_and_boundary()  # as of now this won't be pretty for nf > 2
        else:
            pass

        self._plot_sv()

        # TODO: plot decision boundary and features

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~ Low level methods ~~~~~~~~~~~~~~~
    def _print_message(self, which: str) -> None:
        """
        Method used to print messages on terminal, mostly for quick use while things run
        """
        # ANSI color codes for sprint-level theatrics (kept from original style)
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RESET = "\033[0m"
        if which == "start":
            print(f"[RUN START] {_now_stamp(mode='nice')}")
            print(f"{GREEN}{self.run_name}_nf{self.nf}{RESET}\n")
        elif which == "end":
            print(YELLOW + f"[RUN END] {_now_stamp(mode='nice')}" + RESET)
        else:
            return

    def _create_config_file(self) -> None:
        """
        Generate `self.cfg` for bookkeeping and recalling
        Done at the *start* of the run
        Shapes are appended after data load
        """
        self.cfg: Dict[str, Any] = {
            # Naming
            "run_name": self.run_name,
            "dataset_label": self.dataset_label,
            "results_dir": str(self.results_root.resolve()),
            "timestamp_start": _now_stamp(),
            # IO
            "data_input_mode": "dict" if isinstance(self.X, dict) else "array",
            "#channels": len(self.ch_names),
            # Model
            "spatialFilter_init": (
                {"type": "array", "shape": list(self.spatialFilter_init.shape)}
                if isinstance(self.spatialFilter_init, np.ndarray)
                else self.spatialFilter_init
            ),
            "nf": self.nf,
            "eta_flag": self.eta_flag,
            "ard_flag": self.ard_flag,
            "logged_flag": self.logged_flag,
            "kernel_type": self.kernel_type,
            # Training
            "maxiter": self.maxiter,
            "random_state": self.random_state,
            "frac_val": self.frac_val if hasattr(self, "frac_val") else None,
            "frac_test": self.frac_test if hasattr(self, "frac_test") else None,
            "enable_early_stopping": self.enable_early_stopping,
        }

    def _build_montage_from_xy(
        self,
        ch_names: List[str],
        ch_xy: Dict[str, Tuple[float, float]],
        default_z: float = 0.0,
    ) -> mne._fiff.meas_info.Info:
        """
        Build an MNE DigMontage from 2D XY coords (assume z=0)
        Generate MNE Info object associated to the montage

        ch_names: list of channel labels (lower-case)
        ch_xy   : dict name -> (x, y)
        """
        ch_pos = {}
        missing = []
        for name in ch_names:
            if name in ch_xy:
                x, y = ch_xy[name]
                ch_pos[name] = (x, y, default_z)
            else:
                missing.append(name)
        if missing:
            print(
                f"[montage] Warning: {len(missing)} channels missing XY coords: "
                f"{missing[:8]}{'...' if len(missing)>8 else ''}"
            )
        montage = make_dig_montage(ch_pos=ch_pos, coord_frame="head")
        info_full = mne.create_info(ch_names=ch_names, sfreq=1000.0, ch_types="eeg")
        info_full.set_montage(montage)

        return info_full

    # ----------------- Data preparation ----------------- #
    def _load_and_prepare_data(self) -> None:
        """
        Handle both input modes for X and Y (arrays OR dicts)
        Supports optional validation set and flexible splitting for arrays
        Always flattens X from (N, s, s) to (N, D) to meet GPy input requirement

        Limitations: if a dicts are provided, all not specified keys won't be automatically defined with splits

        Sets:
        self.X_train: (N_train, D)
        self.Y_train: (N_train, 1)
        self.X_val  : (N_val,   D) or None
        self.Y_val  : (N_val,   1) or None
        self.X_test : (N_test,  D) or None
        self.Y_test : (N_test,  1) or None

        self.has_train, self.has_val, self.has_test
        self.N_train,   self.N_val,   self.N_test
        """

        def _to_array(a):
            """
            Convert list-like to np.ndarray without copying unnecessarily
            """
            return a if isinstance(a, np.ndarray) else np.asarray(a)

        def _to_col(Ya: np.ndarray) -> np.ndarray:
            # Reshape into (N, 1)
            return np.asarray(Ya).reshape(-1, 1)

        def _flatten_3d_to_2d(X):
            """
            Flatten X of shape (N, s, s) to shape (N, D) where D is s * s
            """
            X = _to_array(X)
            N = X.shape[0]
            s = X.shape[1]
            return X.reshape(N, s * s)

        def _set_attrs(Xtr, Ytr, Xva, Yva, Xte, Yte, verbose=True):
            """
            Set attributes and counters for all sets
            """
            self.X_train, self.Y_train = Xtr, Ytr
            self.X_val,   self.Y_val   = Xva, Yva
            self.X_test,  self.Y_test  = Xte, Yte

            self.has_train = Xtr is not None
            self.has_val   = Xva is not None
            self.has_test  = Xte is not None

            self.N_train = int(len(Xtr)) if self.has_train else 0
            self.N_val   = int(len(Xva)) if self.has_val   else 0
            self.N_test  = int(len(Xte)) if self.has_test  else 0

            # Update the config file with dimensions
            self.cfg.update(
                {
                    "N_train": self.N_train,
                    "N_val"  : self.N_val,
                    "N_test" : self.N_test,
                }
            )

            if verbose:
                print(
                    f"  [Input] Train: {self.N_train}, Val: {self.N_val}, Test: {self.N_test}"
                )

        # Validate fraction settings if needed
        frac_val  = float(getattr(self, "frac_val",  0.0) or 0.0)
        frac_test = float(getattr(self, "frac_test", 0.0) or 0.0)
        if not (0.0 <= frac_val <= 1.0 and 0.0 <= frac_test <= 1.0):
            raise ValueError("frac_val and frac_test must be in (0, 1)")

        # Case 1: X,Y are dict
        # Expect shapes like:
        #   self.X = {'train': Xtr, 'val': Xva, 'test': Xte}
        #   self.Y = {'train': Ytr, 'val': Yva, 'test': Yte}
        # If keys missing, no new sets are created to fill in
        if isinstance(self.X, dict) and isinstance(self.Y, dict):
            # Extract values from dictionary, if not found they are set to None
            Xtr = self.X.get("train")
            Xva = self.X.get("val")
            Xte = self.X.get("test")

            Ytr = self.Y.get("train")
            Yva = self.Y.get("val")
            Yte = self.Y.get("test")

            self.s = Xtr.shape[-1]

            # Need to determine is any of them is missing (and None)
            # Use frac_val and frac_test to generate the missing ones
            # If any of the splits are present, assign them
            if Xtr is not None:
                Xtr = _flatten_3d_to_2d(Xtr)  # shape (N_train, s*s)
                Ytr = _to_col(Ytr)            # shape (N_train, 1)
            else:
                raise ValueError("Input require at least `train`")
            if Xva is not None:
                Xva = _flatten_3d_to_2d(Xva)  # shape (N_val, s*s)
                Yva = _to_col(Yva)            # shape (N_val, 1)
            if Xte is not None:
                Xte = _flatten_3d_to_2d(Xte)  # shape (N_test, s*s)
                Yte = _to_col(Yte)            # shape (N_test, 1)

            _set_attrs(Xtr, Ytr, Xva, Yva, Xte, Yte)
            return

        # Case 2: array-like inputs
        # Split data into train / validation / test sets using `random_seed`
        # This is done by checking the passed fractions `frac_val` and `frac_test`
        else:
            self.s = self.X.shape[-1]
            X_all = _flatten_3d_to_2d(self.X)  # shape (N, s*s)
            Y_all = _to_col(self.Y)            # shape (N, 1)

            if frac_val == 0.0 and frac_test == 0.0:
                # All data is used for train
                _set_attrs(X_all, Y_all, None, None, None, None)
                return

            if frac_test > 0.0:
                # Split out test first if `frac_test` exists, then val from remaining data
                X_tmp, X_te, Y_tmp, Y_te = train_test_split(
                    X_all,
                    Y_all,
                    test_size=frac_test,
                    random_state=self.random_state,
                    shuffle=True,
                )
            else:
                # No test set, all data becomes temporary
                X_tmp, Y_tmp, X_te, Y_te = X_all, Y_all, None, None

            if frac_val > 0.0:
                # Split the temporary set into train and val
                X_tr, X_va, Y_tr, Y_va = train_test_split(
                    X_tmp,
                    Y_tmp,
                    test_size=frac_val,
                    random_state=self.random_state,
                    shuffle=True,
                )
            else:
                # No validation set, temporary set is all train
                X_tr, Y_tr, X_va, Y_va = X_tmp, Y_tmp, None, None

            # Final assignment
            _set_attrs(X_tr, Y_tr, X_va, Y_va, X_te, Y_te)
            return

    def _initialize_W_matrix(self) -> None:
        """
        Initilize the spatial filter matrix W according to the provided configuration as self.W_init : (s, nf)
        Accepts either:
            - a string policy ("random" | "ones" | "focused")  -> trainable W
            - a NumPy array of shape (s, nf)                   -> NON-trainable W (fixed)
        """
        rng = np.random.default_rng(self.random_state)  # Define random state

        # Different `self.spatialFilter_init` allow for different behaviors
        if isinstance(self.spatialFilter_init, np.ndarray):
            # Matrix behavior
            # Depending on `W_trainable` this can act as a seed or a fixed spatial filter
            W_arr = np.asarray(self.spatialFilter_init, dtype=np.float64)
            if W_arr.ndim != 2:
                raise ValueError("spatialFilter_init array must be 2D with shape (s, nf)")
            if W_arr.shape[0] != self.s:
                raise ValueError(f"spatialFilter_init has {W_arr.shape[0]} rows, expected s={self.s}")
            if W_arr.shape[1] != self.nf:
                raise ValueError(f"spatialFilter_init has {W_arr.shape[1]} cols, expected nf={self.nf}")
            self.W_init = W_arr.copy()

        else:
            # String behavior
            # Depending on `W_trainable` this can act as a seed or a fixed spatial filter
            if self.spatialFilter_init == "random":
                # Randomize initial coefficients using Gaussian -> N(0, 1)
                self.W_init = rng.normal(loc=0.0, scale=1.0, size=(self.s, self.nf))

            elif self.spatialFilter_init == "ones":
                # Set all initial coefficients to 1
                self.W_init = np.ones((self.s, self.nf), dtype=np.float64)

            else:
                raise ValueError(f"Unknown spatialFilter_init: {self.spatialFilter_init}")

        # Update config file
        self.cfg.update({
            "W_init_shape": self.W_init.shape,
            "W_trainable": bool(getattr(self, "W_trainable", True)),
            "W_source": "array" if isinstance(self.spatialFilter_init, np.ndarray) else str(self.spatialFilter_init),
        })

    # ----------------- Model / Kernel builders ----------------- #
    def _build_model_kernel(self) -> None:
        """
        Build a GPy kernel to pass to the model
        """
        # Initialize the kernel using the provided CustomKernelGPy
        self.kernel = CustomKernelGPy(
            self.W_init,
            W_trainable=self.W_trainable,
            ard_flag=self.ard_flag,
            eta_flag=self.eta_flag,
            logged_flag=self.logged_flag,
            kernel_type=self.kernel_type,
        )
        return

    def _build_model(self) -> None:
        """
        Build a GPy model with chosen kernel
        """
        self._build_model_kernel()
        assert self.kernel is not None
        assert self.X_train is not None and self.Y_train is not None

        X_train = np.asarray(self.X_train, dtype=np.float64)
        Y_train = np.asarray(self.Y_train, dtype=int)

        if Y_train.ndim == 1:
            Y_train = Y_train.reshape(-1, 1)

        ep = GPy.inference.latent_function_inference.EP()
        self.model = GPy.models.GPClassification(
            X=X_train,
            Y=Y_train,
            kernel=self.kernel,
            inference_method=ep,
        )

    def _print_state_on_terminal(self) -> None:
        """
        Print training information on terminal as an update
        """
        def _fmt(v):
            """
            Format a number to 3 decimals or return '/' when missing/non-finite
            """
            return f"{float(v):.3f}" if (v is not None and np.isfinite(v)) else "/"

        # Grab last logged metrics
        last = self.logs[-1]

        # Define rows of text to print
        accuracy = f" | accuracy ({_fmt(last.acc_train)}, {_fmt(last.acc_val)}, {_fmt(last.acc_test)})"
        brier    = f" | brier    ({_fmt(last.brier_train)}, {_fmt(last.brier_val)}, {_fmt(last.brier_test)})"
        nlpd     = f" | nlpd     ({_fmt(last.nlpd_train)}, {_fmt(last.nlpd_val)}, {_fmt(last.nlpd_test)})"

        print(
            f"[{int(last.step/self.maxiter * 100)}%] Iter {last.step:4d}/{self.maxiter} | nlml {last.nlml:.3f}"
            f"{accuracy}"
            f"{brier}"
            f"{nlpd}"
        )

    # ----------------- Predictions / Metrics ----------------- #
    def _predict_prob(self, model: GPy.models.Model, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for inputs X
        The probability values are bound to [0,1]
        """
        if X is None:
            return None
        mu, _ = model.predict(X)
        return mu.ravel()

    def _compute_metrics(
        self, y_true: Optional[np.ndarray], p: Optional[np.ndarray]
    ) -> Dict[str, Optional[float]]:
        """
        Compute classification metrics at current iteration
        If new metrics need to be taken into account, they can be computed and added here
        """
        # Define default container for all metrics
        metrics: Dict[str, Optional[float]] = {
            "acc"   : None,
            "brier" : None,
            "aucroc": None,
            "aucpr" : None,
            "nlpd"  : None,
        }
        # Safety check
        if y_true is None or p is None: return metrics

        y_true = y_true.ravel().astype(int)
        p      = p.ravel().astype(np.float64)

        # Predicted labels at current iterations (consistent with choice of `pred_threshold`)
        y_hat = (p >= self.pred_threshold).astype(int)

        # Accuracy, assumes classes are well balanced, otherwise highly biased metric
        metrics["acc"] = float(accuracy_score(y_true, y_hat))

        # Brier score
        metrics["brier"] = float(brier_score_loss(y_true, p))

        # ROC AUC (fails if only one class present)
        try:
            metrics["aucroc"] = float(roc_auc_score(y_true, p))
        except Exception:
            metrics["aucroc"] = None

        # PR AUC
        try:
            prec, rec, _ = precision_recall_curve(y_true, p)
            # use numpy trapezoid based integration to extract area
            metrics["aucpr"] = float(np.trapz(rec, prec))
        except Exception:
            metrics["aucpr"] = None

        # Negative log predictive density (Bernoulli log loss / cross entropy)
        # Both the overall sum or the average value are ok, the average value is more translatable
        eps = 1e-12
        p_clip = np.clip(p, eps, 1.0 - eps) # (safeguard) clip probabilities to eps < p < 1-eps
        ll = y_true * np.log(p_clip) + (1 - y_true) * np.log(1.0 - p_clip)
        metrics["nlpd"] = float(-np.mean(ll))

        return metrics

    def _snapshot_kernel(self) -> Dict[str, Optional[Any]]:
        """
        Grab kernel parameters
        """
        W = None
        eta = None
        ard = None

        if self.kernel is not None:
            # W
            try:
                W_param = getattr(self.kernel, "W", None)
                W  = None if W_param is None else np.asarray(W_param).tolist()
            except Exception:
                W = None

            # eta (scalar)
            try:
                eta_attr = getattr(self.kernel, "eta", None)
                eta = None if eta_attr is None else float(eta_attr)
            except Exception:
                eta = None

            # ard (vector)
            try:
                ard_attr = getattr(self.kernel, "ard", None)
                ard = None if ard_attr is None else np.asarray(ard_attr).ravel().tolist()
            except Exception:
                ard = None

        return {"W": W, "eta": eta, "ard": ard}

    def _snapshot_iteration(self, p_train: np.array, p_val: np.array, p_test: np.array) -> None:
        """
        Save per-iteration predictions, labels, and metrics for logging
        """
        # Snapshot metrics (dict)
        m_train = self._compute_metrics(self.Y_train, p_train)
        m_val   = self._compute_metrics(self.Y_val,   p_val  )
        m_test  = self._compute_metrics(self.Y_test,  p_test )

        kernel_snapshot = self._snapshot_kernel()       # dict for W, eta, ARD
        nlml_ = float(-self.model.log_likelihood())     # current training NLML

        # store IterLog
        self.logs.append(
            IterLog(
                step=int(self.step),
                nlml=nlml_,
                nlpd_train=m_train["nlpd"],
                acc_train=m_train["acc"],
                brier_train=m_train["brier"],
                aucroc_train=m_train["aucroc"],
                aucpr_train=m_train["aucpr"],
                nlpd_val=m_val["nlpd"],
                acc_val=m_val["acc"],
                brier_val=m_val["brier"],
                aucroc_val=m_val["aucroc"],
                aucpr_val=m_val["aucpr"],
                nlpd_test=m_test["nlpd"],
                acc_test=m_test["acc"],
                brier_test=m_test["brier"],
                aucroc_test=m_test["aucroc"],
                aucpr_test=m_test["aucpr"],
                W=kernel_snapshot["W"],
                eta=kernel_snapshot["eta"],
                ard=kernel_snapshot["ard"],
            )
        )

    # ----------------- Training loop ----------------- #
    def _selection_metric(self, last: IterLog) -> Tuple[str, Optional[float]]:
        """
        Decide which metric defines the `best` model
        If validation set is used, use nlpd_val; else use nlml
        """
        if self.use_validation_for_adaptation and self.has_val:
            name, value = "nlpd_val", last.nlpd_val
        else:
            name, value = "nlml",    last.nlml
        return name, value

    def _check_for_best_iteration(
        self, 
        p_train_at_iter: np.array = None, 
        p_val_at_iter  : np.array = None, 
        p_test_at_iter : np.array = None,
    ) -> None:
        """
        If the current iteration improves the chosen metric, snapshot parameters
        """
        last = self.logs[-1] # grab last stored metrics
        name, value = self._selection_metric(last)
        if value is None or not np.isfinite(value):
            return
        # This is a minimization process, we check for lower values
        if value < self._best_score:
            self._best_score       = float(value)
            self._best_iter        = self.step
            self._best_metric_name = name

            # Snapshot GPy parameters (trainable + non-trainable)
            self._best_params = self.model.param_array.copy()

        # Also snapshot best probabilities and labels at this iteration
        if p_train_at_iter is not None:
            self._p_train_best = np.asarray(p_train_at_iter, dtype=float).ravel()
            self._y_train_best = (self._p_train_best >= self.pred_threshold).astype(int)
        if p_val_at_iter is not None:
            self._p_val_best = np.asarray(p_val_at_iter, dtype=float).ravel()
            self._y_val_best = (self._p_val_best >= self.pred_threshold).astype(int)
        if p_test_at_iter is not None:
            self._p_test_best = np.asarray(p_test_at_iter, dtype=float).ravel()
            self._y_test_best = (self._p_test_best >= self.pred_threshold).astype(int)

    def _check_for_early_stopping(self) -> None:
        """
        During the training process check for the conditions to enforce early stopping
        """
        pass
    
    def _train(self) -> None:
        """
        Perform training process by defining the loss function and performing steps with Adam and NG optimizers
        """
        # Define loss function
        self.loss_fn = lambda m: float(-m.log_likelihood())

        # Printing and logging
        self.logs: List[IterLog] = []                   # Initialize containers for per-iter logging
        print_terminal_fr = max(1, self.maxiter // 10)  # Frequency of terminal msg

        # Training process (iterative)
        for self.step in range(1, self.maxiter + 1):
            # Perform a training iteration
            self.model.optimize(optimizer="scg", messages=False, max_iters=1)

            # Predict probabilities at current iteration
            p_train_at_iter = self._predict_prob(self.model, self.X_train)
            p_val_at_iter   = self._predict_prob(self.model, self.X_val  ) if self.has_val  else None
            p_test_at_iter  = self._predict_prob(self.model, self.X_test ) if self.has_test else None

            # Snapshot kernel and metrics at current iteration
            # Build and store the IterLog
            self._snapshot_iteration(p_train_at_iter, p_val_at_iter, p_test_at_iter)

            # Track improvement for `best` iteration
            self._check_for_best_iteration(p_train_at_iter, p_val_at_iter, p_test_at_iter)

            # Print info to terminal
            if self.step % print_terminal_fr == 0 or self.step == 1:
                self._print_state_on_terminal()

            # Early stopping
            if self.enable_early_stopping:
                self._check_for_early_stopping()

        # Restore best iteration / checkpoint
        if self._best_params is not None:
            self.model.param_array[:] = self._best_params
            self.model._trigger_params_changed()
            print(
                f"  [Restore] Best {self._best_metric_name}: {self._best_score:.3f} at iter {self._best_iter}"
            )
            # Record best info into meta
            self.cfg.update(
                {
                    "best_iter": self._best_iter,
                    "best_metric": self._best_metric_name,
                    "best_metric_value": self._best_score,
                }
            )
        else:
            print(f"  [Restore] Failed")

    # ----------------- RunLog ----------------- #
    def _build_and_write_runlog(self) -> None:
        """
        Convert numpy arrays to lists
        Build RunLog and write JSON file
        """

        def _tolist_seq(seq: List[np.ndarray]) -> List[List[float]]:
            return [arr.astype(float).ravel().tolist() for arr in seq]

        def _tolist_seq_int(seq: List[np.ndarray]) -> List[List[int]]:
            return [arr.astype(int).ravel().tolist() for arr in seq]

        # Evaluate final predicted probabilities and labels
        # `final` doesn't mean from the last iteration but from the best iteration
        # `best` comes from the metric under examination, most of the time it's NLML
        # but it coudl be NLPD from the validation test
        # Build best-only snapshots directly from stored best arrays
        p_train_best = [] if self._p_train_best is None else self._p_train_best.astype(float).ravel().tolist()
        p_val_best   = [] if self._p_val_best   is None else self._p_val_best.astype(float).ravel().tolist()
        p_test_best  = [] if self._p_test_best  is None else self._p_test_best.astype(float).ravel().tolist()

        y_train_best = [] if self._y_train_best is None else np.asarray(self._y_train_best, dtype=int).ravel().tolist()
        y_val_best   = [] if self._y_val_best   is None else np.asarray(self._y_val_best,   dtype=int).ravel().tolist()
        y_test_best  = [] if self._y_test_best  is None else np.asarray(self._y_test_best,  dtype=int).ravel().tolist()

        self.run_log = RunLog(
            meta=deepcopy(self.cfg),
            logs=self.logs,
            p_train_best = p_train_best,
            y_train_best = y_train_best,
            p_val_best   = p_val_best,
            y_val_best   = y_val_best,
            p_test_best  = p_test_best,
            y_test_best  = y_test_best,
        )

        # JSON serialization
        _ensure_dir(self.run_dir)
        with open(self.run_dir / "run_log.json", "w") as f:
            json.dump(asdict(self.run_log), f, indent=2)

    # ----------------- Visual Summary ----------------- #
    def _get_split(self, name: str) -> Tuple:
        """
        Create tuple of (y, p) for a given split (train, val, test)
        Both are set to None if val or test sets are missing
        """
        if name == "train":
            y = self.Y_train.ravel().astype(int)
            p = np.array(self.run_log.p_train_best)
        elif name == "val":
            if not getattr(self, "has_val", False):
                return None, None
            y = self.Y_val.ravel().astype(int)
            p = np.array(self.run_log.p_val_best)
        else:  # last case is "test"
            if not getattr(self, "has_test", False):
                return None, None
            y = self.Y_test.ravel().astype(int)
            p = np.array(self.run_log.p_test_best)
        return y, p

    def _get_best_snapshot(self, split: str) -> Tuple:
        """
        Return (y_true, p_best, y_best) for a split in {'train', 'val', 'test'}

        This function ONLY reads the best snapshot saved in `self.run_log`
        It never touches per-iteration sequences because those were removed
        """
        if split == "train":
            y_true = np.asarray(self.Y_train).ravel()
            p_best = np.asarray(self.run_log.p_train_best or [], dtype=float).ravel()
            y_best = np.asarray(self.run_log.y_train_best or [], dtype=int).ravel()
        elif split == "val":
            y_true = np.asarray(self.Y_val).ravel()
            p_best = np.asarray(self.run_log.p_val_best or [], dtype=float).ravel()
            y_best = np.asarray(self.run_log.y_val_best or [], dtype=int).ravel()
        elif split == "test":
            y_true = np.asarray(self.Y_test).ravel()
            p_best = np.asarray(self.run_log.p_test_best or [], dtype=float).ravel()
            y_best = np.asarray(self.run_log.y_test_best or [], dtype=int).ravel()
        else:
            raise ValueError(f"Unknown split: {split}")

        # If best labels weren't stored, derive them from probabilities
        if y_best.size == 0 and p_best.size > 0:
            thr = float(self.pred_threshold)
            y_best = (p_best >= thr).astype(int)

        return y_true, p_best, y_best

    def _get_best_predictions(self, split: str):
        """
        Returns (y_true, y_pred_best) for a split
        """
        y_true, _, y_best = self._get_best_snapshot(split)
        return y_true, y_best

    def _get_best_probabilities(self, split: str):
        """
        Returns (y_true, p_best) for a split
        """
        y_true, p_best, _ = self._get_best_snapshot(split)
        return y_true, p_best

    def _plot_learning_curves(self) -> None:
        """
        Plot curves used in training process (e.g. neg-elbo as nlml, nlpd_val, ...)

        Note: As the training process evolves, NLML should decrease over iterations while other metrics such as accuracy should push toward 1
              Jumps in the metric are allowed due to approximation methods and additional considerations, this being said, it shouldn't be too crazy
        """
        # TODO: currently only plotting NLML as metric but it also need to plot nlpd_val
        # which curve can be grabbed from self._best_metric_name

        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        # -------------------------------------------------------
        steps = [l.step for l in self.run_log.logs]
        nlml  = [l.nlml for l in self.run_log.logs]

        ax1.plot(steps, nlml, linewidth=2, color="black", label="NLML")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Neg-ELBO")
        ax1.set_xlim(0, max(steps) if steps else self.maxiter)
        # -------------------------------------------------------
        ax2 = (
            ax1.twinx()
        )  # generate parallel y-axis on the right side for the accuracy range
        acc_train = [l.acc_train for l in self.run_log.logs]
        ax2.plot(
            steps,
            acc_train,  # train
            linestyle="-",
            linewidth=1.2,
            color=self.colors["train"],
            label="train",
        )

        if self.has_val:
            acc_val = [l.acc_val for l in self.run_log.logs]
            ax2.plot(
                steps,
                acc_val,  # val
                linestyle="-",
                linewidth=1.2,
                color=self.colors["val"],
                label="val",
            )

        if self.has_test:
            acc_test = [l.acc_test for l in self.run_log.logs]
            ax2.plot(
                steps,
                acc_test,  # test
                linestyle="-",
                linewidth=1.2,
                color=self.colors["test"],
                label="test",
            )

        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Accuracy")
        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [l.get_label() for l in lines], loc="best")
        fig.tight_layout()
        fig.savefig(self.run_dir / "01_learning_curves.png", dpi=150)
        plt.close(fig)        

    def _plot_threshold_sweep(self, verbose: bool = False) -> None:
        """
        Plot threshold sweeps for Accuracy, Precision, Recall, F1, Specificity, Youden's J for train
        and for val/test if available Uses probabilities from the best iteration

        Note: In a binary classification problem with balanced labels the prediction probability threshold should be found around 0.5
        """

        # Define some helper methods
        def _fmt(x) -> str:
            """
            Helper for printing format
            """
            return f"{x:.3f}" if (x is not None and np.isfinite(x)) else "/"

        def _get_split(name: str) -> Tuple:
            """
            Create tuple of (y, p) for a given split (train, val, test)
            Both are set to None if val or test sets are missing
            """
            if name == "train":
                # y = self.Y_train.ravel().astype(int)
                # p = np.array(self.run_log.p_train_best)
                y, p = self._get_best_probabilities(split=name)
            elif name == "val":
                if not getattr(self, "has_val", False):
                    return None, None
                # y = self.Y_val.ravel().astype(int)
                # p = np.array(self.run_log.p_val_best)
                y, p = self._get_best_probabilities(split=name)
            else:  # last case is "test"
                if not getattr(self, "has_test", False):
                    return None, None
                # y = self.Y_test.ravel().astype(int)
                # p = np.array(self.run_log.p_test_best)
                y, p = self._get_best_probabilities(split=name)
            return y, p

        def _aucs(y: np.ndarray, p: np.ndarray) -> Tuple:
            """
            Create tuple of (ROC-AUC, PR-AUC) for a given pair of labels and predicted probabilities (y,p)
            Both are set to None if they can't be computed
            """
            if y is None or p is None or len(p) == 0:
                return None, None
            roc = None
            ap = None
            try:
                roc = float(roc_auc_score(y, p))
            except Exception:
                pass
            try:
                ap = float(average_precision_score(y, p))
            except Exception:
                pass
            return roc, ap

        def _metrics_at_all_thresholds(
            y: np.ndarray, p: np.ndarray, thr_seq: np.array
        ) -> Dict:
            """
            Compute a predefined set of metrics given the true labels and the predicted probabilities (y,p)
            Scan the decision boundary (looking at different probability values) to identify best threshold
            """
            if y is None or p is None or len(p) == 0:
                # Set everything to NaN since the metrics can't be computed
                nan = np.full_like(thr_seq, np.nan, dtype=float)
                return {
                    "acc": nan,
                    "prec": nan,
                    "rec": nan,
                    "f1": nan,
                    "spec": nan,
                    "youden": nan,
                }

            # Define output dict
            out = {"acc": [], "prec": [], "rec": [], "f1": [], "spec": [], "youden": []}
            y = y.astype(int)

            # Scan different threshold values
            for th in thr_seq:
                yhat = (p >= th).astype(int)
                # true positive  (y = 1, p > threshold)
                TP = np.sum((yhat == 1) & (y == 1))
                # false positive (y = 0, p > threshold)
                FP = np.sum((yhat == 1) & (y == 0))
                # true negative  (y = 0, p < threshold)
                TN = np.sum((yhat == 0) & (y == 0))
                # false negative (y = 1, p < threshold)
                FN = np.sum((yhat == 0) & (y == 1))
                N = len(y)

                # Accuracy: ratio of correct predictions among the total cases
                acc = (TP + TN) / N if N else np.nan
                # Precision: number of true positive divided by number of samples predicted as positive (punished by high FP)
                precision = TP / (TP + FP) if (TP + FP) > 0 else np.nan
                # Recall: number of true positive divided by number of samples that should have been identified as positive (punished by high FN)
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # TPR
                # F1: harmonic mean of precision and recall
                f1 = (
                    (2 * precision * recall / (precision + recall))
                    if (not np.isnan(precision) and (precision + recall) > 0)
                    else np.nan
                )
                # Sensitivity (recall): how good your ability to detect positive cases is (given all predicted y=1)
                # Specificity: how good your ability to detect negative cases is (given all true y=0)
                specificity = TN / (TN + FP) if (TN + FP) > 0 else np.nan
                # False positive rate: rate of predicted as positive but y=0 out of all the y=0
                fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
                # Youden's J: sensitivity + specificity - 1
                #           = sensitivity - (1 - specificity)
                #           = recall      - fpr
                youden = recall - fpr  # TPR - FPR

                out["acc"].append(acc)
                out["prec"].append(precision)
                out["rec"].append(recall)
                out["f1"].append(f1)
                out["spec"].append(specificity)
                out["youden"].append(youden)

            for k in out:
                out[k] = np.asarray(out[k], dtype=float)
            return out

        def _best_idx(arr: np.ndarray) -> int:
            """
            Index of maximum, treating NaNs as -inf; earliest index when multiple possibilities
            """
            arr = np.asarray(arr, dtype=float)
            scores = np.where(np.isnan(arr), -np.inf, arr)
            idxs = np.where(scores == np.max(scores))[0]
            return int(idxs[0]) if idxs.size else 0

        def _roc_points(y, p):
            if y is None or p is None or len(p) == 0:
                return None, None
            try:
                fpr, tpr, _ = roc_curve(y, p)
                return fpr, tpr
            except Exception:
                return None, None

        def _pr_points(y, p):
            if y is None or p is None or len(p) == 0:
                return None, None
            try:
                prec, rec, _ = precision_recall_curve(y, p)
                return prec, rec
            except Exception:
                return None, None

        # Generate threshold grid, with 51 points delta threshold is 0.02
        thr_seq = np.linspace(0.0, 1.0, 51)

        y_tr, p_tr = _get_split("train")
        y_va, p_va = _get_split("val")
        y_te, p_te = _get_split("test")

        auc_tr, ap_tr = _aucs(y_tr, p_tr)
        auc_va, ap_va = _aucs(y_va, p_va)
        auc_te, ap_te = _aucs(y_te, p_te)

        met_tr = _metrics_at_all_thresholds(y_tr, p_tr, thr_seq)
        met_va = _metrics_at_all_thresholds(y_va, p_va, thr_seq)
        met_te = _metrics_at_all_thresholds(y_te, p_te, thr_seq)

        fpr_tr, tpr_tr = _roc_points(y_tr, p_tr)
        fpr_va, tpr_va = _roc_points(y_va, p_va)
        fpr_te, tpr_te = _roc_points(y_te, p_te)

        prec_tr, rec_tr = _pr_points(y_tr, p_tr)
        prec_va, rec_va = _pr_points(y_va, p_va)
        prec_te, rec_te = _pr_points(y_te, p_te)

        # Plot
        fig, axes = plt.subplots(4, 2, figsize=(10.5, 12.0), sharex=True)
        axes = axes.ravel()

        # ROC
        ax_roc = axes[0]
        ax_roc.plot([0, 1], [0, 1], ls="--", color="0.7", lw=1)  # diagonal
        if fpr_tr is not None:
            ax_roc.plot(
                fpr_tr,
                tpr_tr,
                color=self.colors["train"],
                lw=2,
                label=f"train AUC = {_fmt(auc_tr)}",
            )
        if getattr(self, "has_val", False) and fpr_va is not None:
            ax_roc.plot(
                fpr_va,
                tpr_va,
                color=self.colors["val"],
                lw=2,
                label=f"val AUC = {_fmt(auc_va)}",
            )
        if getattr(self, "has_test", False) and fpr_te is not None:
            ax_roc.plot(
                fpr_te,
                tpr_te,
                color=self.colors["test"],
                lw=2,
                label=f"test AUC = {_fmt(auc_te)}",
            )
        # ax_roc.set_title("ROC curve")
        ax_roc.set_xlim(0, 1)
        ax_roc.set_ylim(0, 1)
        ax_roc.set_xlabel("FPR")
        ax_roc.set_ylabel("TPR")
        ax_roc.grid(alpha=0.25)
        ax_roc.legend(loc="lower right", fontsize=8, frameon=True)

        # PR
        ax_pr = axes[1]
        if prec_tr is not None:
            ax_pr.plot(
                rec_tr,
                prec_tr,
                color=self.colors["train"],
                lw=2,
                label=f"train AP = {_fmt(ap_tr)}",
            )
        if getattr(self, "has_val", False) and prec_va is not None:
            ax_pr.plot(
                rec_va,
                prec_va,
                color=self.colors["val"],
                lw=2,
                label=f"val AP = {_fmt(ap_va)}",
            )
        if getattr(self, "has_test", False) and prec_te is not None:
            ax_pr.plot(
                rec_te,
                prec_te,
                color=self.colors["test"],
                lw=2,
                label=f"test AP = {_fmt(ap_te)}",
            )
        # ax_pr.set_title("Precision-Recall curve")
        ax_pr.set_xlim(0, 1)
        ax_pr.set_ylim(0, 1)
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.grid(alpha=0.25)
        ax_pr.legend(loc="lower left", fontsize=8, frameon=True)

        # Additional metric sweep
        items = [
            ("Accuracy", "acc", (0.0, 1.0)),
            ("Precision", "prec", (0.0, 1.0)),
            ("Recall (TPR)", "rec", (0.0, 1.0)),
            ("F1 score", "f1", (0.0, 1.0)),
            ("Specificity", "spec", (0.0, 1.0)),
            ("Youden's J", "youden", (-1.0, 1.0)),
        ]
        items_dict = {lbl: ylim for (lbl, _, ylim) in items}

        def _legend_label(split_name: str, vals: np.ndarray):
            """
            Return legend string 'split: t*=x val=y' with fallbacks
            """
            if vals is None or not np.isfinite(vals).any():
                return f"{split_name}: t*=/ val=/", None, None
            j = _best_idx(vals)
            t_star = thr_seq[j]
            v_star = vals[j]
            return f"{split_name}: t*={_fmt(t_star)}  val={_fmt(v_star)}", j, v_star

        def _plot_curve(ax, label, key):
            # train
            lab_tr, j_tr, v_tr = _legend_label("train", met_tr[key])
            ax.plot(
                thr_seq, met_tr[key], color=self.colors["train"], lw=2, label=lab_tr
            )
            if j_tr is not None and np.isfinite(v_tr):
                ax.plot(thr_seq[j_tr], v_tr, "^", color=self.colors["train"], ms=6)

            if self.has_val:
                lab_va, j_va, v_va = _legend_label("val", met_va[key])
                ax.plot(
                    thr_seq, met_va[key], color=self.colors["val"], lw=2, label=lab_va
                )
                if j_va is not None and np.isfinite(v_va):
                    ax.plot(thr_seq[j_va], v_va, "^", color=self.colors["val"], ms=6)

            if self.has_test:
                lab_te, j_te, v_te = _legend_label("test", met_te[key])
                ax.plot(
                    thr_seq, met_te[key], color=self.colors["test"], lw=2, label=lab_te
                )
                if j_te is not None and np.isfinite(v_te):
                    ax.plot(thr_seq[j_te], v_te, "^", color=self.colors["test"], ms=6)

            ax.set_ylabel(label)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(*items_dict[label])
            ax.grid(alpha=0.25)
            ax.legend(loc="lower right", fontsize=8, frameon=True)

        # start plotting the six curves from axes[2] onward to keep the new top row intact
        for ax, (lbl, key, ylim) in zip(axes[2:], items):
            _plot_curve(ax, lbl, key)

        # Only show x-label for the bottom panels
        axes[-2].set_xlabel("Probability threshold")
        axes[-1].set_xlabel("Probability threshold")

        title = (
            "Threshold Sweep - ROC-AUC (train / val / test): "
            f"{_fmt(auc_tr)} / {_fmt(auc_va)} / {_fmt(auc_te)}   |   "
            "PR-AUC: "
            f"{_fmt(ap_tr)} / {_fmt(ap_va)} / {_fmt(ap_te)}"
        )
        fig.suptitle(title, y=0.985, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(self.run_dir / "02_threshold_sweep.png", dpi=150)
        plt.close(fig)

        if verbose:
            # Print some info on the terminal
            def _best_report(name, met):
                if met is None:
                    return
                for metric_key, pretty in [
                    ("acc", "Accuracy"),
                    ("prec", "Precision"),
                    ("rec", "Recall"),
                    ("f1", "F1"),
                    ("spec", "Specificity"),
                    ("youden", "Youden"),
                ]:
                    arr = met[metric_key]
                    if arr is None or not np.isfinite(arr).any():
                        print(f"  [{name}] {pretty:<11s}: t*=/  val=/")
                        continue
                    j = _best_idx(arr)
                    print(
                        f"  [{name}] {pretty:<11s}: t*={_fmt(thr_seq[j])}  val={_fmt(arr[j])}"
                    )

            print("[threshold_sweep] Best thresholds:")
            _best_report("train", met_tr)
            if self.has_val:
                _best_report("val", met_va)
            if self.has_test:
                _best_report("test", met_te)

    def _generate_calibration_curves(self, n_bins: int = 10) -> List:
        """
        Generate calibration curves with initial equal-width bins in [0, 1]
        Any bin with fewer than 3 points is adaptively merged into the adjacent bin that has
        fewer points (ties merge left)

        Note: A well calibrated classification model predicts p times the data in a bin of probabilty p
              e.g. all data in the first p bin should have P(y=1) -> 0, similarly all data data in the last p bin should have P(y=1) -> 1
                   additionally, all data in the p bin at 0.2 should have P(y=1) -> 0.2
        """

        splits = {
            "train": True,
            "val"  : getattr(self, "has_val",  False),
            "test" : getattr(self, "has_test", False),
        }
        curves = []

        for key, value in splits.items():
            if value == False:
                curves.append(None)
            else:
                # Extract labels and probabilities
                y_true, p = self._get_split(key)
                brier     = brier_score_loss(y_true, p) # Brier's score

                # If too few test points, don't compute the plot
                if p.size < 3:
                    print(f"Not enough points for calibration curve: {p.size}")

                # Start with `n_bins` equal-width bins in [0, 1]
                initial_bins = int(n_bins)
                edges = list(np.linspace(0.0, 1.0, initial_bins + 1))

                # Assign each point to a bin index in [0, initial_bins-1]
                # Since we provide edges in increasing order and `right==False` -> edges[i-1] <= x < edges[i]
                idx = (
                    np.digitize(p, edges, right=False) - 1
                )  # Needs to shift by 1 to start index at 0

                # Distribute p among the bins to see bin population
                bin_indices = [
                    np.where(idx == b)[0].tolist() for b in range(int(n_bins))
                ]

                # Process the bins such that each one of them has enough samples, otherwise merge
                # Merge into the neighbor (left/right) with fewer points; ties -> left
                MIN_PER_BIN = 3
                while True:
                    counts = [len(ix) for ix in bin_indices]
                    if len(bin_indices) == 1 or all(c >= MIN_PER_BIN for c in counts):
                        break

                    # pick first underfilled bin
                    b     = next(i for i, c in enumerate(counts) if c < MIN_PER_BIN)
                    left  = b - 1 if b - 1 >= 0 else None
                    right = b + 1 if b + 1 < len(bin_indices) else None

                    if left is None and right is None:
                        # single bin case
                        break
                    elif left is None:
                        # merge current into right
                        bin_indices[right].extend(bin_indices[b])
                        del bin_indices[b]
                    elif right is None:
                        # merge current into left
                        bin_indices[left].extend(bin_indices[b])
                        del bin_indices[b]
                    else:
                        # choose neighbor with fewer points; tie -> left
                        if counts[left] <= counts[right]:
                            bin_indices[left].extend(bin_indices[b])
                            del bin_indices[b]
                        else:
                            bin_indices[b].extend(bin_indices[right])
                            del bin_indices[right]

                # Compute empirical fraction (y values) and mean predicted probabilities (x values)
                frac_pos, mean_pred = [], []
                for ix in bin_indices:
                    if not ix:  # safety
                        continue
                    frac_pos.append(float((y_true[ix] == 1).mean()))
                    mean_pred.append(float(p[ix].mean()))

                curves.append([key, mean_pred, frac_pos, brier])

        return curves

    def _plot_calibration_curves(self) -> None:
        """
        Plot calibration curves generated by the method `_generate_calibration_curves`
        Train, val and test are plot accordingly when available
        """
        curves = self._generate_calibration_curves()

        fig, ax = plt.subplots(figsize=(5.6, 5.6))
        ax.plot(
            [0, 1],
            [0, 1],
            linestyle=":",
            linewidth=2,
            color="black",
            label="Perfectly calibrated",
        )
        markers = {"train": "o", "val": "s", "test": "^"}
        for curve in curves:
            if curve is not None:
                name, mean_pred, frac_pos, brier = curve
                ax.plot(
                    mean_pred,
                    frac_pos,
                    marker=markers.get(name, "o"),
                    linewidth=1.5,
                    color=self.colors[name],
                    label=f"{name} (Brier={brier:.3f})",
                )
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Empirical probability")
        ax.set_title("Calibration curves")
        # ax.grid(alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(self.run_dir / "03_calibration_curve.png", dpi=150)
        plt.close(fig)

    def _plot_kernel_scaling(self) -> None:
        """
        Plot kernel scaling hyperparameters: self.eta (float per iteration)
                                             self.ard (float per spatial filter per iteration)
        At least one between self.eta_flag and self.ard_flag need to be True
        """
        if self.eta_flag or self.ard_flag:
            # At least one of them is a trainable parameter
            fig, ax = plt.subplots(figsize=(8, 4.5))
            # -------------------------------------------------------
            steps = [l.step for l in self.run_log.logs]

            if self.eta_flag:
                etas = [
                    l.eta if l.eta is not None else np.nan for l in self.run_log.logs
                ]  # List[ float ]
                ax.plot(steps, etas, linewidth=2, color="black", label=r"$\eta")

            if self.ard_flag:
                for k in range(self.nf):
                    vals_k = [
                        (l.ard[k] if (l.ard is not None and len(l.ard) > k) else np.nan)
                        for l in self.run_log.logs
                    ]  # List [ float ]
                    ax.plot(steps, vals_k, linewidth=2, label=f"ARD[{k}]")

            ax.set_xlabel("Iteration")
            ax.set_ylabel("Hyperparameter")
            ax.set_xlim(0, self.maxiter)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(ncol=2, fontsize=8)
            fig.tight_layout()
            fig.savefig(self.run_dir / "05_kernel_parameters.png", dpi=150)
            plt.close(fig)

    def _plot_kernel_W_old(self) -> None:
        """
        Plot kernel spatial filter weights and their evolution over the iterations, one plot per spatial filter
        self.nf provides the number of spatial filters that have been created
        self.run_log.logs[0].W is a list with sub-lists:
            the list has len() = self.s, one entry per channel
            the sub-lists have len() = self.nf, one entry per spatial filter
        """

        fig, ax = plt.subplots(1, self.nf, figsize=(int(5 * self.nf), 4))
        # -------------------------------------------------------
        steps = [l.step for l in self.run_log.logs]  # shape (self.maxiter, )
        Ws = np.array(
            [l.W for l in self.run_log.logs]
        )  # shape (self.maxiter, self.s, self.nf)

        for k in range(self.nf):
            ax[k].plot(
                steps, Ws[:, :, k], linewidth=1.2
            )  # self.s plots in the same panel
            ax[k].set_xlabel("Iteration")
            ax[k].set_ylabel(f"W[:,{k}]")
            ax[k].set_xlim(0, self.maxiter)
        fig.tight_layout()
        fig.savefig(self.run_dir / "06_kernel_W.png", dpi=150)
        plt.close(fig)

    def _plot_kernel_W(self) -> None:
        """
        Plot kernel spatial filter weights and their evolution over iterations.
        Works with both a single spatial filter (nf == 1) and multiple filters (nf >= 2).

        Expected shapes per iteration log:
        - l.W is (s, nf) or a list-of-lists equivalent.
        - steps is a list/array of length T == number of logs.
        """
        # --- Basic validation ----------------------------------------------------
        if getattr(self, "run_log", None) is None or not getattr(self.run_log, "logs", []):
            return  # Nothing to plot. Actions have consequences.

        logs = self.run_log.logs
        steps = np.asarray([getattr(l, "step", i + 1) for i, l in enumerate(logs)], dtype=float)
        T = steps.size

        # --- Normalize W across iterations into (T, s, nf) ----------------------
        Ws_list = []
        for l in logs:
            W = np.asarray(l.W, dtype=float)  # tolerate list-of-lists
            # Accept (s, nf) or (s,) if nf == 1
            if W.ndim == 1:
                # Treat as (s,) for single filter. Expand to (s, 1).
                if getattr(self, "nf", 1) == 1 and W.size == getattr(self, "s", W.size):
                    W = W.reshape(-1, 1)
                else:
                    # If this is malformed, skip this iteration.
                    W = np.full((getattr(self, "s", 1), getattr(self, "nf", 1)), np.nan)
            elif W.ndim != 2:
                W = np.full((getattr(self, "s", 1), getattr(self, "nf", 1)), np.nan)

            s_expected = getattr(self, "s", W.shape[0])
            nf_expected = getattr(self, "nf", W.shape[1])

            # Fix channel dimension if off by trivial reshape
            if W.shape[0] != s_expected and W.size == s_expected * nf_expected:
                W = W.reshape(s_expected, nf_expected)

            # Slice or pad filter dimension to nf_expected
            if W.shape[1] >= nf_expected:
                W = W[:, :nf_expected]
            else:
                pad = np.full((W.shape[0], nf_expected - W.shape[1]), np.nan)
                W = np.concatenate([W, pad], axis=1)

            # Slice or pad channel dimension to s_expected
            if W.shape[0] >= s_expected:
                W = W[:s_expected, :]
            else:
                pad = np.full((s_expected - W.shape[0], W.shape[1]), np.nan)
                W = np.concatenate([W, pad], axis=0)

            Ws_list.append(W)

        # Stack to (T, s, nf)
        Ws = np.stack(Ws_list, axis=0).astype(float)
        s, nf = Ws.shape[1], Ws.shape[2]
        if nf == 0 or T == 0:
            return

        # --- Create axes that behave for nf == 1 and nf > 1 ----------------------
        fig_width = max(5, int(5 * nf))  # don’t make a postage stamp
        fig, axes = plt.subplots(1, nf, figsize=(fig_width, 4))
        if nf == 1:
            axes = [axes]  # make it indexable like an array

        # --- Plot: each panel k shows all s channel traces for that filter -------
        for k in range(nf):
            ax = axes[k]
            # Plot s lines, one per channel, across iterations
            # Ws[:, ch, k] is the ch-th channel’s weight trajectory for filter k
            ax.plot(steps, Ws[:, :, k], linewidth=1.2, alpha=0.9)
            ax.set_xlabel("Iteration")
            ax.set_ylabel(f"W[:, {k}]")
            # Be a tiny bit robust if maxiter is missing or silly
            xmax = float(getattr(self, "maxiter", steps.max() if T else 1))
            ax.set_xlim(steps.min() if T else 0.0, xmax)
            ax.grid(True, linewidth=0.4, alpha=0.3)

            # Optional: show a faint median to anchor the spaghetti
            with np.errstate(invalid="ignore"):
                med = np.nanmedian(Ws[:, :, k], axis=1)
            ax.plot(steps, med, linewidth=2.0)  # default color, thicker line
            ax.set_title(f"Filter {k}")

        fig.tight_layout()
        fig.savefig(self.run_dir / "06_kernel_W.png", dpi=150)
        plt.close(fig)

    def _retrieve_spatial_filter(self, f: int, iter: int) -> np.array:
        """
        Retrieve a given spatial filter column of index `f` at the iteration `iter`
        """
        # Safety checks
        if self.run_log is None or not self.run_log.logs:
            return np.zeros(self.s, dtype=float)

        Ws = np.array([l.W for l in self.run_log.logs])  # (maxiter, s, nf)
        iter_idx = int(iter) - 1
        if iter_idx < 0 or iter_idx >= Ws.shape[0]:
            return np.zeros(self.s, dtype=float)
        if f < 0 or f >= Ws.shape[2]:
            return np.zeros(self.s, dtype=float)

        return Ws[iter_idx, :, f]

    def _plot_topomap(
        self,
        iter: Optional[int] = None,
        weights: Optional[List] = None,
        fs: Optional[List[int]] = [],
    ) -> None:
        """
        Plot one or multiple spatial filters as topomaps
        If `iter` is provided it overwrites the `best` iteration

        Recursive policy:
        - If `weights` is provided: plot the spatial filter/s
        - Else if `fs` provided: collect the corresponding weights from the `best` iteration, then recall method with `weights`
        - Else: collect all spatial filters from the `best` iteration, then recall method with `weights`

        Parameters
        ----------
        iter: Optional[int]
            Iteration number to consider instead of `best` iteration
        weights : Optional[List[np.ndarray]]
            A list where each entry is a 1D array of length `self.s` containing channel weights
            If provided, these are plotted as-is
        fs : Optional[List[int]]
            Indices of spatial filters to plot from the best iteration (only used when `weights` is None)
            Preserved for axis titles when plotting a subset
        """

        # Need MNE and a valid montage to plot topomaps
        if HAS_MNE:
            if weights is None:
                # `weights` are not provided so retrive them from `best` iteration
                iter_to_use = iter if iter is not None else self._best_iter

                if len(fs) > 0:
                    # Plot only a selection of spatial filters at `best` iteration
                    w_list = [
                        self._retrieve_spatial_filter(f=f, iter=iter_to_use) for f in fs
                    ]
                    # Re-enter with weights provided
                    return self._plot_topomap(iter=iter_to_use, weights=w_list, fs=fs)
                else:
                    # Plot all spatial filters at `best` iteration
                    all_fs = list(range(int(self.nf)))
                    w_list = [
                        self._retrieve_spatial_filter(f=f, iter=iter_to_use)
                        for f in all_fs
                    ]
                    return self._plot_topomap(
                        iter=iter_to_use, weights=w_list, fs=all_fs
                    )

            else:
                # `weights` are provided, plot these spatial filter/s
                cols = []
                for w in weights:
                    w = np.asarray(w, dtype=float).ravel()
                    cols.append(w)

                # Assemble (s, k) matrix for plotting k topomaps
                W_t = (
                    np.column_stack(cols)
                    if len(cols) > 0
                    else np.zeros((int(self.s), 0))
                )
                k = W_t.shape[1]
                if k == 0:
                    return

                # Build figure with one axis per filter
                fig, axes = plt.subplots(1, k, figsize=(4 * k, 4))
                if k == 1:
                    axes = [axes]

                # Iteration label for title
                if iter is not None and iter > self.maxiter:
                    t = "?"
                else:
                    t = iter if iter is not None else self._best_iter

                # Plot each provided weight vector as a topomap
                for i in range(k):
                    mne.viz.plot_topomap(
                        W_t[:, i],
                        self.montage_info,
                        axes=axes[i],
                        show=False,
                        sphere=1.2,
                    )
                    # Title shows the original filter index if provided, otherwise the local column index
                    label_idx = fs[i] if (fs is not None and i < len(fs)) else i
                    axes[i].set_title(f"W[:,{label_idx}]  Iter {t}")

                fig.tight_layout()
                fig.savefig(self.run_dir / "08_topomaps.png", dpi=150)
                plt.close(fig)

        else:
            return

    def _plot_confusion_matrix(self) -> None:
        """
        Plot confusion matrix for all the available sets
        P(y=1) > self.pred_threshold
        """
        
        y, y_pred = self._get_best_predictions(split='train')
        cm     = [confusion_matrix(y, y_pred)]
        label  = ["train"]
        ks     = 1

        if self.has_val:
            y, y_pred = self._get_best_predictions(split='val')
            cm.append(confusion_matrix(y, y_pred))
            label.append("val")
            ks += 1

        if self.has_test:
            y, y_pred = self._get_best_predictions(split='test')
            cm.append(confusion_matrix(y, y_pred))
            label.append("test")
            ks += 1

        # Plot
        fig, axes = plt.subplots(1, ks, figsize=(4 * ks, 4))
        if ks == 1:
            axes = [axes]

        # Plot each provided weight vector as a topomap
        for k in range(ks):
            vmax = cm[k].max()
            axes[k].imshow(cm[k], cmap="Greens", vmin=0, vmax=vmax)
            if k == 0:
                axes[k].set_title(f"{label[k]} (Iter {self._best_iter})", fontsize=9)
            else:
                axes[k].set_title(f"{label[k]}", fontsize=9)
            axes[k].set_xlabel(
                f"Predicted P(y=1) > {self.pred_threshold}", fontsize=9
            )
            axes[k].set_ylabel("True", fontsize=9)
            for (i, j), v in np.ndenumerate(cm[k]):
                axes[k].text(j, i, int(v), ha="center", va="center", fontsize=9)

        fig.tight_layout()
        fig.savefig(self.run_dir / "09_confusion_matrix.png", dpi=150)
        plt.close(fig)

    def _compute_feature(self, f: int, iter: int) -> Dict[str, np.ndarray]:
        """
        Compute features using a given spatial filter column of index `f` at the iteration `iter`
        To do so the method calls _retrieve_spatial_filter()

        This method strongly depends on the kernel used
        The feature matches the kernel's construction: z_f = w_f^T Σ_i w_f, optionally log-transformed
        and ARD-scaled to mirror the kernel space
        """
        iter_idx = iter - 1

        # Retrieve the spatial filter at `iter`
        W = self._retrieve_spatial_filter(f=f, iter=iter).astype(float).ravel()

        def _compute_on_split(X_flat: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if X_flat is None:
                return None

            # Reshape flattened (N, s*s) back to (N, s, s)
            X_flat = np.asarray(X_flat, dtype=float)
            N = X_flat.shape[0]
            Sigma = X_flat.reshape(N, self.s, self.s)

            # Compute z = w^T Σ w using vectorized tensordot
            Sw = np.tensordot(Sigma, W, axes=([2], [0]))  # (N, s)
            wSw = np.sum(Sw * W[None, :], axis=1)  # (N,)

            # Optional log-transform to match kernel behavior
            if getattr(self, "logged_flag", False):
                wSw = np.log(wSw)

            # Optional ARD scaling for spatial filter `f` on this feature if available at `iter`
            try:
                if getattr(self, "ard_flag", False) and self.run_log is not None:
                    ard_vec = self.run_log.logs[iter_idx].ard
                    if ard_vec is not None and len(ard_vec) > f:
                        wSw = wSw * np.exp(ard_vec[f])
            except Exception:
                # If ARD is not present skip scaling
                pass

            return wSw.astype(float)

        out: Dict[str, np.ndarray] = {}

        out["train"] = _compute_on_split(self.X_train)
        if self.has_val:
            out["val"] = _compute_on_split(self.X_val)
        if self.has_test:
            out["test"] = _compute_on_split(self.X_test)
        return out

    def _compute_decision_boundary(self, iter: int) -> Dict[str, Any]:
        """
        Build a 2D decision surface over a two-feature space using interpolation from predicted probabilties
        on the train set P(y=1) at iteration `iter`

        Feature-pair selection:
            - If self.feature_pair exists, use it (clamped to [0, nf-1]).
            - Otherwise default to (0, 1)

        Returns

        Dict[str, Any]
            A dictionary containing:
                'XX', 'YY', 'ZZ' : meshgrid and interpolated surface
                'f1', 'f2'       : chosen feature indices
                'fX_*', 'fY_*'   : raw 2D features per split (if available)
        """
        iter_idx = int(iter) - 1

        # Choose feature pair
        f1, f2 = 0, 1

        # Compute features for the given pair
        fX_dict = self._compute_feature(f=f1, iter=iter)
        fY_dict = self._compute_feature(f=f2, iter=iter)

        fX = np.asarray(fX_dict["train"], dtype=float).ravel()
        fY = np.asarray(fY_dict["train"], dtype=float).ravel()

        if fX is None or fY is None or fX.size == 0:
            return {}

        # Grid extent from train set with small padding
        fX_min, fX_max = float(np.min(fX)), float(np.max(fX))
        fY_min, fY_max = float(np.min(fY)), float(np.max(fY))
        pad_x = 0.05 * (fX_max - fX_min + 1e-12)
        pad_y = 0.05 * (fY_max - fY_min + 1e-12)

        x_lin = np.linspace(fX_min - pad_x, fX_max + pad_x, 300)
        y_lin = np.linspace(fY_min - pad_y, fY_max + pad_y, 300)
        XX, YY = np.meshgrid(x_lin, y_lin)

        # Train predicted probabilities at `iter`
        y, p = self._get_best_probabilities(split='train')

        # Concatenate along the second axis
        pts = np.c_[fX, fY]
        # Interpolate P(y=1) onto the grid
        try:
            ZZ = griddata(points=pts, values=p, xi=(XX, YY), method="cubic")
        except Exception:
            ZZ = griddata(points=pts, values=p, xi=(XX, YY), method="linear")

        # Fill any holes with nearest neighbor interpolation
        nan_mask = np.isnan(ZZ)
        if np.any(nan_mask):
            ZZ[nan_mask] = griddata(
                points=pts, values=p, xi=(XX[nan_mask], YY[nan_mask]), method="nearest"
            )

        return {
            "XX": XX,
            "YY": YY,
            "ZZ": ZZ,
            "f1": f1,
            "f2": f2,
            "fX_train": fX,
            "fY_train": fY,
            "fX_val": fX_dict.get("val"),
            "fY_val": fY_dict.get("val"),
            "fX_test": fX_dict.get("test"),
            "fY_test": fY_dict.get("test"),
        }

    def _add_decision_boundary(
        self,
        iter: int,
        levels: Optional[List[float]] = None,
    ) -> None:
        """
        Add contour lines of the decision surface on the current axis at a given iteration `iter`
        Plot the level at self.pred_threshold as the main one in black, plot any additional levels in grey
        """
        if levels is None:
            levels = [self.pred_threshold, 0.1, 0.9]

        # Cache boundary per-iteration to avoid recomputation if the axis is redrawn
        boundary = getattr(self, "_last_boundary", None)
        if boundary is None or boundary.get("iter") != int(iter):
            boundary = self._compute_decision_boundary(iter=iter)
            boundary["iter"] = int(iter)
            self._last_boundary = boundary

        if not boundary:
            return

        XX, YY, ZZ = boundary["XX"], boundary["YY"], boundary["ZZ"]
        ax = plt.gca()

        # Emphasize the decision threshold
        if self.pred_threshold in levels:
            thr = self.pred_threshold
            cs_thr = ax.contour(
                XX, YY, ZZ, levels=[thr], linewidths=2.0, colors="black"
            )
            ax.clabel(cs_thr, fmt={thr: f"p={thr:.2f}"}, inline=True, fontsize=8)
            other = [lv for lv in levels if lv != thr]
        else:
            other = list(levels)

        # Draw auxiliary levels
        if other:
            ax.contour(
                XX, YY, ZZ, levels=other, linewidths=1.0, colors="grey", linestyles="--"
            )

    def _plot_features_and_boundary(self) -> None:
        """
        Scatter the selected feature pair for train / val / test and overlay the decision boundary
        Feature pair is read from `self.feature_pair` if present; otherwise defaults to (0, 1)
        Only when 2 filter columns are required the boundary is shown
        """
        # Define iteration using self._best_iter
        iter = self._best_iter
        boundary = self._compute_decision_boundary(iter=iter)
        if not boundary:
            return

        f1, f2 = boundary["f1"], boundary["f2"]
        fX_train, fY_train = boundary["fX_train"], boundary["fY_train"]
        # fX_val, fY_val = boundary.get("fX_val"), boundary.get("fY_val")
        fX_test, fY_test = boundary.get("fX_test"), boundary.get("fY_test")

        fig, ax = plt.subplots(figsize=(6, 5))

        # Train points
        if fX_train is not None and fY_train is not None:
            ax.scatter(
                fX_train,
                fY_train,
                c=["orange" if y == 0 else "navy" for y in self.Y_train.ravel()],
                s=44,
                marker="o",
                linewidth=0.4,
                alpha=0.2,
                # label="train",
            )
            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="k",
                    markersize=8,
                    alpha=0.2,
                    label="Train",
                )
            ]

        # Validation points
        """
        if self.has_val and fX_val is not None and fY_val is not None:
            ax.scatter(
                fX_val,
                fY_val,
                c=["orange" if y == 0 else "navy" for y in self.Y_val.ravel()],
                s=28,
                marker="s",
                linewidth=0.4,
                alpha=0.7,
                #label="val",
            )
            handles.append(plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="k",
                markersize=8,
                alpha=0.7,
                label="Val",
            ))
        """

        # Test points
        if self.has_test and fX_test is not None and fY_test is not None:
            ax.scatter(
                fX_test,
                fY_test,
                c=["orange" if y == 0 else "navy" for y in self.Y_test.ravel()],
                s=22,
                marker="^",
                linewidth=0.4,
                alpha=1,
                # label="test",
            )
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="^",
                    color="w",
                    markerfacecolor="k",
                    markersize=6,
                    alpha=1,
                    label="Test",
                )
            )

        # Overlay boundary
        self._add_decision_boundary(
            iter=iter, levels=[self.pred_threshold, 0.1, 0.9]
        )

        # Labels and aesthetics
        x_label = rf"$w_{f1}^T \Sigma w_{f1}$"
        y_label = rf"$w_{f2}^T \Sigma w_{f2}$"
        x_label = (
            f"log({x_label})" if getattr(self, "logged_flag", False) else x_label
        )
        y_label = (
            f"log({y_label})" if getattr(self, "logged_flag", False) else y_label
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"Iter {iter}", fontsize=9)
        ax.legend(handles=handles, loc="best", fontsize=8, frameon=True)
        fig.tight_layout()
        fig.savefig(
            self.run_dir / f"10_features_and_boundary_{self.feature_pair}.png",
            dpi=150,
        )
        plt.close(fig)

    def _compute_features_matrix(self, iter: int) -> np.ndarray:
        """
        Generate a matrix of features per spatial filter at a given `iter` iteration
        The resulting matrix is gonna have shape (self.N_train, self.nf)
        """
        mat = np.zeros((self.N_train, self.nf))

        for f in range(int(self.nf)):
            # dict of features a `iter` for a given filter index `f`
            feats = self._compute_feature(f=f, iter=iter)
            mat[:, f] = feats["train"]

        return mat

    def _compute_svd(self, mat: np.ndarray, k: Optional[int] = None) -> np.array:
        """
        Perform SVD on a given matrix `mat`, with `k` being the number of largest eigenvalues to consider
        `k` = None uses full rank
        """
        try:
            if k is None:
                # Full SVD; use 'gesvd' under the hood
                # with `mat` of shape (M, N) `full_matrices` being True means U and Vh are of shape (M, M), (N, N)
                # being False, the shapes are (M, K) and (K, N), where K = min(M, N)
                s = svd(mat, full_matrices=True, compute_uv=False)
            else:
                # Top-k eigenvalues via Lanczos
                # This is best for large/sparse matrices, requires flip ascending order
                s = np.sort(svds(mat, k=k, return_singular_vectors=False))
            return s
        except:
            return None

    def _plot_sv(self, iter: Optional[int] = None) -> None:
        """
        Plot singular values at a given `iter` iteration
        The idea is to find out the rank of meaningful dimension in the feature space at a given iteration
        """
        if iter is None:
            # Use `best` iteration
            iter = self._best_iter
            self._plot_sv(iter=iter)
        else:
            mat = self._compute_features_matrix(iter=iter)
            s = self._compute_svd(mat)  # full dimension
            s = np.sort(s)[::-1]  # sort descending for nicer visual
            xs = np.arange(1, len(s) + 1)

            # Plot
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(
                xs,
                s,
                marker="o",
                color="black",
                markerfacecolor="gold",
                markeredgecolor="black",
                linewidth=1.5,
            )
            ax.set_xlim(np.min(xs) - 0.5, np.max(xs) + 0.5)
            ax.set_ylabel("Singular value")
            ax.set_xlabel("Index")
            ttl = f"Iter {iter}"
            ax.set_title(ttl, fontsize=9)
            # ax.grid(True, linestyle="--", alpha=0.4)
            fig.tight_layout()
            fig.savefig(self.run_dir / "16_singular_values.png", dpi=150)
            plt.close(fig)
