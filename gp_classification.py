"""
DATA SHAPES (core)
------------------
Let:
  N  = number of trials
  s  = number of EEG sensors/electrodes
  nf = number of spatial filters (columns of W)
  D  = s*s (flattened input dimension)

Inputs:
  X, Y can be:

  A) Arrays:
     - X: (N, s, s) covariance matrices
     - Y: (N, ) or (N, 1) binary {0,1}
     -> internal train/test split with `frac_train`

  B) Dicts with explicit splits:
     - X: dict with keys among {"train", "val", "test"}; each entry (N_*, s, s)
     - Y: dict with matching keys; each entry (N_*,) or (N_*,1)

Outputs (core tensors inside runner):
  X_train: (N_train, D), Y_train: (N_train, 1)
  X_val  : (N_val,   D) | None,  Y_val  : (N_val, 1) | None
  X_test : (N_test,  D) | None,  Y_test : (N_test, 1) | None
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
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)  # runtime support for type hints


# ---------------------- Third party libraries (mandatory) ----------------------
import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    auc,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from scipy.interpolate import griddata
from scipy.linalg import svd
from scipy.sparse.linalg import svds
from kernels import CustomKernel  # custom covariance function

# ---------------------- Third party libraries (optionals for extra glitters) ----------------------
try:
    import imageio.v2 as imageio  # handles GIF creation; accepts in-memory numpy frames

    HAS_IMAGEIO = True
except Exception:
    HAS_IMAGEIO = False

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

    step: int
    # Train set
    nlml: float  # negative log marginal likelihood
    nlpd_train: Optional[float]  # negative log probability density
    acc_train: Optional[float]  # accuracy
    brier_train: Optional[float]  # brier's score
    aucroc_train: Optional[float]  # area under the curve ROC
    aucpr_train: Optional[float]  # area under the curve precision-recall
    # Validation set
    nlpd_val: Optional[float]
    acc_val: Optional[float]
    brier_val: Optional[float]
    aucroc_val: Optional[float]
    aucpr_val: Optional[float]
    # Test set
    nlpd_test: Optional[float]
    acc_test: Optional[float]
    brier_test: Optional[float]
    aucroc_test: Optional[float]
    aucpr_test: Optional[float]
    # Kernel
    W: List[List[float]]  # spatial filter weights
    eta: Optional[float]  # global scaling
    ard: Optional[List[float]]  # per-filter scaling
    kernel_eigs: Optional[List[float]]  # kernel eigenvalues
    # Training process
    lr: List[float]  # Adam learning rate
    ema: List[float]  # Exponential moving average of training metric
    gamma: List[float]  # Natural gradient gamma


@dataclass
class RunLog:
    """
    Container for the entire run logs, converted to JSON format
    """

    meta: Dict[str, Any]  # config
    logs: List[IterLog]  # per-iteration logged info
    # Train set
    p_train_seq: List[List[float]]  # predicted probabilties (one list per iteration)
    p_train_best: List[float]  # predicted probabilities (last iteration only)
    y_train_seq: List[List[int]]  # label sequences (one list per iteration)
    y_train_best: List[int]  # label sequences (one list per iteration)
    # Validation set
    p_val_seq: List[List[float]]
    p_val_best: List[float]
    y_val_seq: List[List[int]]
    y_val_best: List[int]
    # Test set
    p_test_seq: List[List[float]]
    p_test_best: List[float]
    y_test_seq: List[List[int]]
    y_test_best: List[int]


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
        ch_names: List[str],  # names of EEG channels
        ch_xy: Dict[str, Tuple[float, float]],  # coordinates of EEG channels
        # Model / kernel
        spatialFilter_init: str = "random",  # 'random' | 'ones' | 'manual'
        nf: int = 2,  # number of spatial filter cols
        eta_flag: bool = False,
        ard_flag: bool = False,
        logged_flag: bool = True,
        kernel_type: str = "RBF",
        model_class: type = gpflow.models.VGP,
        model_kwargs: Optional[Dict] = None,
        likelihood_class: type = gpflow.likelihoods.Bernoulli,
        likelihood_kwargs: Optional[Dict] = None,
        training_loss_fn: Optional[Callable[[gpflow.Module], tf.Tensor]] = None,
        predict_y_fn: Optional[
            Callable[[gpflow.Module, np.ndarray], Tuple[np.ndarray, np.ndarray]]
        ] = None,
        # Training
        learning_rate: float = 0.01,  # Adam default learning rate
        gamma: float = 0.1,  # Natural gradient default learning rate
        maxiter: int = 1000,
        pred_threshold: float = 0.5,  # decision boundary in binary classification p(y=1) >= pred_threshold
        random_state: int = 42,
        # ----- New data split controls (only for array inputs)
        frac_val: float = 0.5,
        frac_test: float = 0.5,
        # ----- Policy flags for adaptation / early stopping
        use_validation_for_adaptation: bool = False,  # if True and val exists, adapt LR/ES on val; else train-only
        enable_adaptation: bool = False,  # enable LR reduce-on-plateau on chosen set
        enable_early_stopping: bool = False,  # enable early stopping on chosen set
        # ----- K-fold CV on training set
        kfolds: int = 0,  # 0 disables; >1 runs CV on training set and plots NLML bands
        # GIF controls
        gif_flag: bool = True,  # generate GIFs
        gif_stride: int = 20,  # sample every k iterations
        gif_max_frames: int = 50,  # auto-raise stride to cap frames
        synced_gif: bool = True,  # generate synced dashboard GIF
        topomap_filters_for_gif: int = 2,  # animate first k cols of W
        # Run naming / Logging
        results_dir: str = "./results",
        run_name: Optional[str] = None,
    ) -> None:
        # Store I/O
        self.X = X
        self.Y = Y
        self.dataset_label = dataset_label
        self.ch_names = [
            c.lower() for c in ch_names
        ]  # enforce lower-case for channel lookup
        self.ch_xy = {k.lower(): v for k, v in ch_xy.items()}

        if HAS_MNE:
            # Build montage for visualization of spatial filter
            self.montage_info = self._build_montage_from_xy(self.ch_names, self.ch_xy)

        # Store model / kernel choices
        self.spatialFilter_init = spatialFilter_init
        self.nf = nf
        self.eta_flag = eta_flag
        self.ard_flag = ard_flag
        self.logged_flag = logged_flag
        self.kernel_type = kernel_type
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.likelihood_class = likelihood_class
        self.likelihood_kwargs = likelihood_kwargs or {}
        self.external_training_loss_fn = training_loss_fn
        self.external_predict_y_fn = predict_y_fn

        # Store training config
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.maxiter = maxiter
        self.pred_threshold = pred_threshold
        self.random_state = random_state
        # Split mode in case X and Y are arrays
        self.frac_val = 0 if frac_val is None else float(frac_val)
        self.frac_test = 0 if frac_test is None else float(frac_test)
        self.use_validation_for_adaptation = bool(use_validation_for_adaptation)
        self.enable_adaptation = bool(enable_adaptation)
        self.enable_early_stopping = bool(enable_early_stopping)
        self.kfolds = int(kfolds)

        # Store GIF config
        self.gif_flag = gif_flag
        self.gif_stride = max(1, int(gif_stride))
        self.gif_max_frames = gif_max_frames
        self.synced_gif = synced_gif
        self.topomap_filters_for_gif = max(1, int(topomap_filters_for_gif))

        # Store Run naming / Logging
        self.results_root = Path(results_dir)
        self.run_name = run_name or f"run_{_now_stamp()}"
        self.run_dir = self.results_root / self.run_name
        _ensure_dir(self.run_dir)  # Create folder

        # Placeholders updated by `_load_and_prepare_data`,
        self.has_train = False
        self.has_val = False
        self.has_test = False

        self.s: int = 0  # number of EEG sensors
        self.N_train: int = 0
        self.N_val: int = 0
        self.N_test: int = 0

        self.X_train: Optional[np.ndarray] = None  # (N_train, D)
        self.X_val: Optional[np.ndarray] = None  # (N_val, D)
        self.X_test: Optional[np.ndarray] = None  # (N_test, D)
        self.Y_train: Optional[np.ndarray] = None  # (N_train, 1)
        self.Y_val: Optional[np.ndarray] = None  # (N_val, 1)
        self.Y_test: Optional[np.ndarray] = None  # (N_test, 1)

        self.W_init: Optional[np.ndarray] = None  # (s, nf)
        self.model: Optional[gpflow.models.Model] = None
        self.kernel: Optional[CustomKernel] = None

        # Outputs and logs
        self.p_train_seq: List[np.ndarray] = []  # list of probabilty arrays (N_train,)
        self.p_val_seq: List[np.ndarray] = []  # (N_val,)
        self.p_test_seq: List[np.ndarray] = []  # (N_test,)
        self.y_train_seq: List[np.ndarray] = []  # list of label arrays (N_train,)
        self.y_val_seq: List[np.ndarray] = []  # (N_val,)
        self.y_test_seq: List[np.ndarray] = []  # (N_test,)

        self.run_log: Optional[RunLog] = None

        # Best-checkpoint tracking
        self._best_score: float = float("inf")
        self._best_iter: Optional[int] = None
        self._best_metric_name: Optional[str] = None
        self._best_params: Optional[Dict] = None  # gpflow parameter dict snapshot

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~ High level method ~~~~~~~~~~~~~~~
    def run(self) -> None:
        """
        Execute the full pipeline
        """
        self._print_message(which="start")
        self._create_config_file()  # Build config file for reproducibility
        self._load_and_prepare_data()  # X_train, X_val, X_test, Y_train, Y_val, Y_test, N_train, N_val, N_test, s

        self._initialize_W_matrix()  # W_init

        # Define model by specifying kernel, likelihood, and method
        self._build_model()  # self.kernel, self.likelihood, self.model
        # self._warm_start_variational()

        # Define optimizers
        self._build_optimizers()  # self.opt, self.natgrad

        self._train()
        self._write_config_file()  # Write config file to `config.json`
        self._build_and_write_runlog()  # Build and write RunLog

        self._make_visual_summary()  # Visual outputs

        self._print_message(which="end")

    def _make_visual_summary(self) -> None:
        """
        Generate visual outputs as summary of the training process (PNGs + GIFs)
        """
        self.colors = {"train": "dodgerblue", "val": "forestgreen", "test": "orangered"}

        self._plot_learning_curves()
        self._plot_threshold_sweep()
        self._plot_calibration_curves()
        self._plot_learning_rates()
        self._plot_kernel_scaling()
        self._plot_kernel_W()
        self._plot_kernel_eigs()  # diagnostic for Grahm matrix
        self._plot_topomap()

        self._plot_confusion_matrix()

        pairs = combinations(range(self.nf), 2)
        for pair in pairs:
            self.feature_pair = pair
            self._plot_features_and_boundary()  # as of now this won't be pretty for nf > 2

        # Diagnostic on the variational parameters
        self._plot_vgp_latent_marginals()
        self._plot_posterior_q_standardized()
        self._plot_posterior_q_correlation_block()
        self._plot_posterior_covariance_eigs()
        self._plot_uncertainty_vs_error()

        self._plot_sv()
        self._plot_sv_evolution()

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
            print(f"{GREEN}{self.run_name}{RESET}\n")
        elif which == "end":
            print(YELLOW + f"[RUN END] {_now_stamp(mode='nice')}" + RESET)
        else:
            return

    def _create_config_file(self) -> None:
        """
        Generate config.json for bookkeeping and recalling
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
            "spatialFilter_init": self.spatialFilter_init,
            "nf": self.nf,
            "eta_flag": self.eta_flag,
            "ard_flag": self.ard_flag,
            "logged_flag": self.logged_flag,
            "kernel_type": self.kernel_type,
            "model_class": self.model_class.__name__,
            "likelihood_class": self.likelihood_class.__name__,
            "has_custom_training_loss_fn": self.external_training_loss_fn is not None,
            "has_custom_predict_y_fn": self.external_predict_y_fn is not None,
            # Training
            "learning_rate_default": self.learning_rate,
            "gamma_default": self.gamma,
            "maxiter": self.maxiter,
            "pred_threshold": self.pred_threshold,
            "random_state": self.random_state,
            "frac_val": self.frac_val if hasattr(self, "frac_val") else None,
            "frac_test": self.frac_test if hasattr(self, "frac_test") else None,
            "use_validation_for_adaptation": self.use_validation_for_adaptation,
            "enable_adaptation": self.enable_adaptation,
            "enable_early_stopping": self.enable_early_stopping,
            "kfolds": self.kfolds,
            # GIF controls
            "gif_flag": self.gif_flag,
            "gif_stride": self.gif_stride,
            "gif_max_frames": self.gif_max_frames,
            "synced_gif": self.synced_gif,
            "topomap_filters_for_gif": self.topomap_filters_for_gif,
        }

    def _write_config_file(self) -> None:
        """
        Write the final version of the config file
        """
        _ensure_dir(self.run_dir)
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(self.cfg, f, indent=2)

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
        Always flattens X from (N, s, s) to (N, D) to meet GPflow input requirement

        Limitations: if a dicts are provided, all not specified keys won't be automatically defined with splits

        Sets:
        self.X_train: (N_train, D)
        self.Y_train: (N_train, 1)
        self.X_val  : (N_val, D) or None
        self.Y_val  : (N_val, 1) or None
        self.X_test : (N_test, D) or None
        self.Y_test : (N_test, 1) or None

        self.has_train, self.has_val, self.has_test
        self.N_train, self.N_val, self.N_test
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
            self.X_val, self.Y_val = Xva, Yva
            self.X_test, self.Y_test = Xte, Yte

            self.has_train = Xtr is not None
            self.has_val = Xva is not None
            self.has_test = Xte is not None

            self.N_train = int(len(Xtr)) if self.has_train else 0
            self.N_val = int(len(Xva)) if self.has_val else 0
            self.N_test = int(len(Xte)) if self.has_test else 0

            # Update the config file with dimensions
            self.cfg.update(
                {
                    "N_train": self.N_train,
                    "N_val": self.N_val,
                    "N_test": self.N_test,
                }
            )

            if verbose:
                print(
                    f"  [Input] Train: {self.N_train}, Val: {self.N_val}, Test: {self.N_test}"
                )

        # Validate fraction settings if needed
        frac_val = float(getattr(self, "frac_val", 0.0) or 0.0)
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
                Ytr = _to_col(Ytr)  # shape (N_train, 1)
            else:
                raise ValueError("Input require at least `train`")
            if Xva is not None:
                Xva = _flatten_3d_to_2d(Xva)  # shape (N_val, s*s)
                Yva = _to_col(Yva)  # shape (N_val, 1)
            if Xte is not None:
                Xte = _flatten_3d_to_2d(Xte)  # shape (N_test, s*s)
                Yte = _to_col(Yte)  # shape (N_test, 1)

            _set_attrs(Xtr, Ytr, Xva, Yva, Xte, Yte)
            return

        # Case 2: array-like inputs
        # Split by frac_val/frac_test if > 0; otherwise use everything as train.
        else:
            self.s = self.X.shape[-1]
            X_all = _flatten_3d_to_2d(self.X)  # shape (N, s*s)
            Y_all = _to_col(self.Y)  # shape (N, 1)

            # All data is used for train
            if frac_val == 0.0 and frac_test == 0.0:
                _set_attrs(X_all, Y_all, None, None, None, None)
                return

            # Split out test first if frac_test exists, then val from remaining data
            if frac_test > 0.0:
                X_tmp, X_te, Y_tmp, Y_te = train_test_split(
                    X_all,
                    Y_all,
                    test_size=frac_test,
                    random_state=self.random_state,
                    shuffle=True,
                )
            else:
                X_tmp, Y_tmp, X_te, Y_te = X_all, Y_all, None, None

            if frac_val > 0.0:
                X_tr, X_va, Y_tr, Y_va = train_test_split(
                    X_tmp,
                    Y_tmp,
                    test_size=frac_val,
                    random_state=self.random_state,
                    shuffle=True,
                )
            else:
                X_tr, Y_tr, X_va, Y_va = X_tmp, Y_tmp, None, None

            # Final assignment
            _set_attrs(X_tr, Y_tr, X_va, Y_va, X_te, Y_te)
            return

    def _initialize_W_matrix(self) -> None:
        """
        Initilize the spatial filter matrix W according to the provided configuration as self.W_init : (s, nf)
        """
        rng = np.random.default_rng(self.random_state)  # Define random state

        # Initialize W according to flag
        if self.spatialFilter_init == "random":
            # Randomize initial coefficients using Gaussian -> N(0, 0.1)
            self.W_init = rng.normal(loc=0.0, scale=0.1, size=(self.s, self.nf))

        elif self.spatialFilter_init == "ones":
            # Set all initial coefficients to 1
            self.W_init = np.ones((self.s, self.nf), dtype=np.float64)

        elif self.spatialFilter_init == "manual":
            # Custom configuration for initial coefficients
            # All set to 0 except channels commonly involved in motor command following
            self.W_init = np.zeros((self.s, self.nf))

            if self.nf > 2:
                print(
                    f"Warning: More than 2 spatial filters initialized, `manual` currenlty needs fixing!"
                )

            # Heuristic motor indices (best-effort), set the weights corresponding to those channels to 1
            def _idx_if_present(name: str) -> Optional[int]:
                return self.ch_names.index(name) if name in self.ch_names else None

            # Find indices of selected channels based on the provided list of channels
            # good for right hand motor imagery
            left_id = [
                i
                for i in map(
                    _idx_if_present,
                    ["fc1", "c1", "cp1", "fc3", "c3", "fc5", "c5", "cp5"],
                )
                if i is not None
            ]
            # good for right foot motor imagery
            right_id = [
                i
                for i in map(
                    _idx_if_present,
                    ["f1", "fz", "fc1", "fcz", "c1", "cz"],
                )
                if i is not None
            ]
            """
            # good for left hand motor imagery
            right_id = [
                i
                for i in map(
                    _idx_if_present,
                    ["fc2", "c2", "cp2", "fc4", "c4", "fc6", "c6", "cp6"],
                )
                if i is not None
            ]
            """
            for idx in left_id:
                if idx is not None and idx < self.s:
                    self.W_init[idx, 0] = (
                        1  # Set first col for left hemisphere channels --> Move right
                    )
            if self.nf > 1:
                for idx in right_id:
                    if idx is not None and idx < self.s:
                        self.W_init[idx, 1] = (
                            1  # Set second col for right hemisphere channels --> Move left
                        )
        else:
            raise ValueError(f"Unknown spatialFilter_init: {self.spatialFilter_init}")

        # Update config file
        self.cfg.update({"W_init_shape": self.W_init.shape})

    # ----------------- Model / Kernel builders ----------------- #
    def _build_kernel(self) -> gpflow.kernels.Kernel:
        """
        Build a GPflow kernel to pass to the model
        """
        # Initialize the kernel using the provided CustomKernel
        self.kernel = CustomKernel(
            self.W_init,
            ard_flag=self.ard_flag,
            eta_flag=self.eta_flag,
            logged_flag=self.logged_flag,
            kernel_type=self.kernel_type,
        )
        return

    def _build_likelihood(self) -> gpflow.likelihoods.Likelihood:
        """
        Build a GPflow likelihood to pass to the model
        """
        # Define the likelihood to use in the classifier, by default we are using Bernoulli for a binary classification
        # p(y=1 ∣ f) = σ(f)
        # p(y=0 ∣ f) = 1 − σ(f)
        # Likelihood for classification purpose needs to be specified, can't use Gaussian
        # Most likely no need for kwargs
        self.likelihood = self.likelihood_class(**self.likelihood_kwargs)
        return

    def _build_model(self) -> gpflow.models.Model:
        """
        Build a GPflow model with chosen kernel, likelihood, and method
        """
        self._build_kernel()  # self.kernel
        self._build_likelihood()  # self.likelihood

        # Define the model to use for the classification, by default we are using Variational Gaussian Process (VGP)
        # Try to instantiate with data (VGP, GPR, etc. usually accept it)
        Xtr = tf.convert_to_tensor(self.X_train, dtype=tf.float64)
        Ytr = tf.convert_to_tensor(self.Y_train, dtype=tf.float64)
        try:
            self.model = self.model_class(
                data=(Xtr, Ytr),
                kernel=self.kernel,
                likelihood=self.likelihood,
                num_latent_gps=1,  # For binary classification we need only one latent GP
                **self.model_kwargs,
            )
        except TypeError:
            # Fallback for models that don't accept `data` (e.g., SVGP)
            self.model = self.model_class(
                kernel=self.kernel,
                likelihood=self.likelihood,
                num_latent_gps=1,  # For binary classification we need only one latent GP
                **self.model_kwargs,
            )
            # Note: for SVGP you likely need a custom `training_loss_fn`
            # that closes over minibatches and a separate dataset.
        return

    def _warm_start_variational(
        self, mu_scale: float = 2.0, jitter: float = 1e-6
    ) -> None:
        """
        Heuristic warm start for variational parameters
        - q_mu: align with labels (±1) and scaled by mu_scale
        - q_sqrt: identity Cholesky (per output), with tiny jitter
        Works for both full and diagonal parameterizations
        """
        if not hasattr(self, "model") or self.model is None:
            return
        if not (hasattr(self.model, "q_mu") and hasattr(self.model, "q_sqrt")):
            return

        # Build ±1 targets from {0,1} labels
        y = np.asarray(self.Y_train).reshape(-1)
        ypm = 2.0 * y - 1.0  # 0 -> -1, 1 -> +1

        # q_mu shape [N, P]
        q_mu = self.model.q_mu
        N, P = int(q_mu.shape[0]), int(q_mu.shape[1])
        mu = (ypm[:, None] * mu_scale).astype(np.float64)
        if P > 1:
            mu = np.tile(mu, (1, P))
        self.model.q_mu.assign(mu)

        # q_sqrt can be [P, N, N] (full) or [P, N] (diag)
        q_sqrt = self.model.q_sqrt
        if len(q_sqrt.shape) == 3:
            # full: per-output lower-triangular Cholesky factors
            P_, N_, _ = map(int, q_sqrt.shape)
            eye = np.eye(N_, dtype=np.float64) * (1.0 + jitter)
            L = np.stack([np.tril(eye.copy()) for _ in range(P_)], axis=0)
            q_sqrt.assign(L)
        elif len(q_sqrt.shape) == 2:
            # diag: per-output sqrt-variances
            P_, N_ = map(int, q_sqrt.shape)
            q_sqrt.assign(np.ones((P_, N_), dtype=np.float64) * (1.0 + jitter))
        else:
            # shrug, leave it alone if a custom form shows up
            pass

    def _build_optimizers(self) -> None:
        """
        Build GPflow optimizers and a dictionary for learning rate adaptation
        """
        # Initialization of self contained structure for learning rate adaptation
        self._lr_state = {
            "step": 0,
            "lr": self.learning_rate,  # current LR value
            "base_lr": self.learning_rate,  # starting LR value
            "min_lr": 1e-5,  # mininum allowed
            "max_lr": self.learning_rate,  # maximum allowed
            "decay_factor": 0.5,  # decay factor on plateau
            "patience": max(
                int(0.05 * self.maxiter), 20
            ),  # number of steps allowed for repeated plateau behavior
            "cooldown": min(
                int(0.05 * self.maxiter), 35
            ),  # general counter to avoid instant reactions
            "tolerance": 1e-3,  # value used to define change in metric
            "best": 1e4,  # best metric value seen
            "ema": None,  # exponential moving average (EMA) of metric
            "ema_beta": 0.8,  # decay factor for EMA
            "warmup_steps": max(int(0.02 * self.maxiter), 10),  # ~2% warmup steps
        }

        # Optimizer 1 for kernel hyperparameters
        # - Adam: Keras variant; learning_rate must be a Python float to avoid type errors
        # Replace SciPy optimization (which gave problems in classification) with Adam optimizer
        # Adaptive Moment Estimation (Adam) is an adaptive `learning_rate` optimization algorithm
        # Combines the benefits of two other popular optimizers: Momentum and RMSprop
        # Adaptive: calculates individual adaptive `learning_rate` for each parameter of the model
        # Momentum: accelerate convergence by accumulating past gradients and moving in the direction of consistent descent
        #           (from cconcept of physics momentum, rolling down a hill)
        # RMSprop: scales the `learning_rate` for each parameter based on the root mean square of past squared gradients
        self.opt = tf.keras.optimizers.Adam(learning_rate=float(self.learning_rate))

        # Optimizer 2 for variational parameters (VGP) if needed
        # - NaturalGradient: gamma is predefined according to custom schedule
        # Detect whether the model actually exposes variational params to set up natural gradient
        use_natgrad = hasattr(self.model, "q_mu") and hasattr(self.model, "q_sqrt")
        self.natgrad = (
            gpflow.optimizers.NaturalGradient(gamma=self.gamma) if use_natgrad else None
        )
        return

    def _adapt_gamma(self) -> float:
        """
        Adaptive natural-gradient step size
        This is a customized schedule:
        1) Warmup (5% of self.maxiter): linearly ramping from 0 -> self.gamma
        2) Boosting (next 15%): boost from self.gamma to 5 * self.gamma
        3) Cosine decay (remaining 80%): decaying from 5 * self.gamma to 0.5 * self.gamma
        """
        # Define values of gammas to use in the schedule
        gamma_main = self.gamma
        gamma_boost = 5 * self.gamma
        gamma_floor = 0.5 * self.gamma
        safety_gamma = 1e-6

        # Phase boundaries
        warmup_steps = max(int(math.ceil(0.05 * self.maxiter)), 1)  # first 5% of steps
        warmup_end = min(warmup_steps, self.maxiter)  # at 5% overall

        boost_steps = max(int(math.ceil(0.15 * self.maxiter)), 1)  # next 15% of steps
        boost_end = min(warmup_steps + boost_steps, self.maxiter)  # at 20% overall

        if self.step <= warmup_end:
            # Warmup (5% of maxiter)
            gamma_step = gamma_main * (self.step / warmup_end)
            return max(float(gamma_step), safety_gamma)

        elif self.step <= boost_end:
            # Boosting (next 15%)
            gamma_step = gamma_boost
            return max(float(gamma_step), safety_gamma)

        else:
            # Cosine decay (remaining 80%), this decay is independent from any metric
            remaining_steps = self.maxiter - boost_end

            if remaining_steps <= 0:
                gamma_step = gamma_floor
                return max(float(gamma_step), safety_gamma)
            else:
                k = self.step - boost_end  # number of steps after boosting phase
                # Cosine function maps 0 and pi to 1 and -1
                cos_term = 0.5 * (1.0 + math.cos(math.pi * (k / remaining_steps)))
                gamma_step = gamma_floor + (gamma_boost - gamma_floor) * cos_term
                return max(float(gamma_step), safety_gamma)

    def _step_natural_gradient(self) -> None:
        """
        Perform a step of natural gradient
        Gamma parameter is adapted before the step
        """
        if self.natgrad is not None:
            # Adapt gamma value before taking the step
            self.natgrad.gamma = self._adapt_gamma()  # always happens
            # Natrual Gradient step
            self.natgrad.minimize(
                lambda: self.loss_fn(self.model),
                var_list=[(self.model.q_mu, self.model.q_sqrt)],
            )
            return
        else:
            return

    def _current_adaptation_metric_value(self) -> Optional[float]:
        """
        Return the scalar metric for LR adaptation at the current model state:
        - validation NLPD if available and enabled,
        - otherwise training NLML
        """
        try:
            if getattr(self, "use_validation_for_adaptation", False) and getattr(
                self, "has_val", False
            ):
                p_val = self._predict_prob(self.model, self.X_val)
                m_val = self._compute_metrics(self.Y_val, p_val)
                nlpd = m_val.get("nlpd", None)
                return float(nlpd) if nlpd is not None else None
            else:
                return float(self.loss_fn(self.model).numpy())
        except Exception:
            return None

    def _adapt_learning_rate(self) -> float:
        """
        Adapt Adam learning rate using ONLY the EMA of the chosen metric
        Metric:
            - NLML on train, unless validation is present AND
            `use_validation_for_adaptation` is True, then use val NLPD
        Delta:
            - delta := EMA_t - EMA_{t-1} of the chosen metric.
        Policy:
            - |delta| <= tolerance      -> plateau; decay on patience, cooldown.
            - delta >= +big_delta       -> big WORSE jump; stronger decay, cooldown.
            - delta <= -big_delta       -> big BETTER drop; gentle growth, cooldown.
            - otherwise                 -> small move; reset plateau on improvement.

        The method _build_optimizers() builds the dictionary `self._lr_state` which is apdated as the training progresses
        """
        s = self._lr_state

        # Generate some default new keys for the learning rate dictionary `self._lr_state`
        s.setdefault("plateau_count", 0)  # number of steps in EMA plateau
        s.setdefault("big_change_mult", 6)  # scaling factor to define big change
        s.setdefault("growth_factor", 1.2)  # LR bump on big improvement
        s.setdefault("min_steps_between_growth", 40)
        s.setdefault("last_growth_step", -(10**9))
        s.setdefault("cooldown_max", min(35, int(0.05 * self.maxiter)))
        s.setdefault("ema_prev", None)

        # Tick step and cooldown
        s["step"] = self.step

        if s["step"] == 1:
            print(f"  [lr] warm up phase")  # starting point

        if s.get("cooldown", 0) > 0:
            # Decrease cooldown counter each step
            s["cooldown"] = int(s["cooldown"]) - 1

        # Warmup: linear ramp to `base_lr`
        if s["step"] <= int(s["warmup_steps"]):
            prev_lr = float(s["lr"])
            warm_lr = float(s["base_lr"]) * (
                s["step"] / float(s["warmup_steps"])
            )  # ramping
            new_lr = max(float(warm_lr), float(s["min_lr"]))
            # Check the new LR is not close in value to the previous one (default tolerance 1e-8)
            if not np.isclose(new_lr, prev_lr):
                # Assign new LR
                s["lr"] = float(new_lr)
                self.opt.learning_rate = float(new_lr)

            # Prime EMA during warmup so delta is defined later
            mv = self._current_adaptation_metric_value()
            if mv is not None and np.isfinite(mv):
                if s.get("ema") is None:
                    # Assing metric to EMA
                    s["ema"] = float(mv)
                else:
                    # Calculate EMA using beta decay factor
                    # Higher beta values preserve more memories
                    beta = float(s["ema_beta"])
                    s["ema"] = float(beta * s["ema"] + (1.0 - beta) * float(mv))
                s["ema_prev"] = float(s["ema"])
            return float(s["lr"])

        # Compute current metric and update EMA
        mv = self._current_adaptation_metric_value()
        if mv is None or not np.isfinite(mv):
            return float(s["lr"])

        if s.get("ema") is None:
            # Assing metric to EMA
            s["ema"] = float(mv)
            s["ema_prev"] = float(s["ema"])
            return float(s["lr"])

        prev_ema = float(s["ema"])
        beta = float(s["ema_beta"])
        curr_ema = float(beta * prev_ema + (1.0 - beta) * float(mv))
        s["ema"] = curr_ema

        # Define delta to judge EMA evolution
        delta = float(curr_ema - prev_ema)  # EMA-only delta
        tol = float(s["tolerance"])
        big_delta = float(s["big_change_mult"]) * tol
        lr = float(s["lr"])
        new_lr = lr
        changed = False

        # Decision logic
        if abs(delta) <= tol:
            # Plateau logic, EMA didn't change significantly
            s["plateau_count"] += 1
            if s["plateau_count"] >= int(s["patience"]) and s.get("cooldown", 0) == 0:
                # Too many steps on plateau, no more patience, time to adapt LR with decay
                new_lr = max(lr * float(s["decay_factor"]), float(s["min_lr"]))
                s["plateau_count"] = 0  # reset count to 0
                s["cooldown"] = int(s["cooldown_max"])  # reset cooldown to max
                changed = True

        elif delta >= big_delta:
            # Positive jump, with EMA increasing
            if s.get("cooldown", 0) == 0:
                # Enough steps with EMA increasing, time to adapt LR with decay
                new_lr = max(lr * float(s["decay_factor"]), float(s["min_lr"]))
                s["plateau_count"] = 0  # reset count to 0
                s["cooldown"] = int(s["cooldown_max"])  # reset cooldown to max
                changed = True
            else:
                s["plateau_count"] = 0  # reset count to 0

        elif delta <= -big_delta:
            # Negative jump, with EMA decreasing
            # Judge if enough step have passed between last LR bump, if so
            can_grow = s.get("cooldown", 0) == 0 and (
                s["step"] - s["last_growth_step"]
            ) >= int(s["min_steps_between_growth"])
            if can_grow:
                # Bump LR for faster convergence
                new_lr = min(lr * float(s["growth_factor"]), float(s["max_lr"]))
                s["plateau_count"] = 0  # reset count to 0
                s["cooldown"] = int(s["cooldown_max"])  # reset cooldown to max
                s["last_growth_step"] = s["step"]  # overwrite last step of LR growth
                changed = True

        else:
            # Minor change: reset plateau on improvement only and keep current LR
            if delta < 0:
                s["plateau_count"] = 0  # reset count to 0

        if changed and not np.isclose(new_lr, lr):
            # Overwrite LR with adapted LR
            s["lr"] = float(new_lr)
            self.opt.learning_rate = float(new_lr)
            print(f"  [lr] changed to {s['lr']:.6g} at iter {self.step}")

        # Advance EMA prev
        s["ema_prev"] = float(curr_ema)
        return float(s["lr"])

    def _step_adam(self) -> None:
        """
        Perform a step of Adam
        Learning rate is fixed but Adam adapts the step on the hyperparameters individually with RMSprop
        """
        with tf.GradientTape() as tape:
            nlml = self.loss_fn(self.model)
        grads = tape.gradient(nlml, self.model.trainable_variables)
        # Filter out None gradients (constrained/unused vars)
        gv = [
            (g, v)
            for g, v in zip(grads, self.model.trainable_variables)
            if g is not None
        ]
        if gv:
            gs, vs = zip(*gv)
            # Gradient clipping could be considered and inserted here
            # clip_norm = 5
            # gs, _ = tf.clip_by_global_norm(gs, clip_norm)
            self.opt.apply_gradients(zip(gs, vs))
            return
        else:
            return

    def _print_state_on_terminal(self) -> None:
        """
        Print training information on terminal as an update to the user
        """

        def _fmt(v):
            """
            Format a number to 3 decimals or return '/' when missing/non-finite
            """
            return f"{float(v):.3f}" if (v is not None and np.isfinite(v)) else "/"

        last = self.logs[-1]

        train_tail = (
            f" | acc_train {last.acc_train:.3f} | brier_train {last.brier_train:.3f}"
        )
        val_tail = (
            f" | acc_val {last.acc_val:.3f} | brier_val {last.brier_val:.3f}"
            if getattr(self, "has_val", False)
            else ""
        )
        test_tail = (
            f" | acc_test {last.acc_test:.3f} | brier_test {last.brier_test:.3f}"
            if getattr(self, "has_test", False)
            else ""
        )

        accuracy = f" | accuracy ({_fmt(last.acc_train)}, {_fmt(last.acc_val)}, {_fmt(last.acc_test)})"
        brier = f" | brier ({_fmt(last.brier_train)}, {_fmt(last.brier_val)}, {_fmt(last.brier_test)})"
        nlpd = f" | nlpd ({_fmt(last.nlpd_train)}, {_fmt(last.nlpd_val)}, {_fmt(last.nlpd_test)})"

        print(
            f"[{int(last.step/self.maxiter * 100)}%] Iter {last.step:4d}/{self.maxiter} | nlml {last.nlml:.3f}"
            # f"{train_tail}"
            # f"{val_tail}"
            # f"{test_tail}"
            f"{accuracy}"
            f"{brier}"
            f"{nlpd}"
        )

    # ----------------- Predictions / Metrics ----------------- #
    def _predict_prob(self, model: gpflow.models.Model, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for inputs X
        The probability values are bound to [0,1]
        If `predict_y_fn` is provided, it is used; else model.predict_y is used

        The addition of the `predict_y_fn` extends the freedom to modify the predicted probabilties:

        1) Any calibration on the fly (example with temperature scaling):

            def predict_with_temp(m, X_np, T=1.5):
                fmean, fvar = m.predict_f(tf.convert_to_tensor(X_np, tf.float64))
                z = fmean.numpy() / T
                p = 1.0 / (1.0 + np.exp(-z))
                return p, fvar.numpy()

        2) Ensamble prediction coming from average of multiple models:

            def predict_ensemble(models, X_np):
                ps = [mdl.predict_y(tf.convert_to_tensor(X_np, tf.float64))[0].numpy() for mdl in models]
                p = np.mean(ps, axis=0)
                return p, None

        3) I'm sure there are other reasons...

        """
        if self.external_predict_y_fn is not None:
            # Use provided function to predict probabilties
            probs, _ = self.external_predict_y_fn(model, X)  # mean, var
            return np.asarray(probs).ravel()

        # Use standard method to predict probabilties
        # This computes the predictive class probability by integrating over the uncertainty in latent function f
        # p(Y=1 ∣ X∗) = E(f∼N(mu, var))[σ(f)]
        # p(Y=0 ∣ X∗) = 1 - p(Y=1 ∣ X∗)
        # This is p(Y=1 ∣ X∗), probability of X being label 1
        probs, _ = model.predict_y(
            tf.convert_to_tensor(X, dtype=tf.float64)
        )  # mean, var
        return probs.numpy().ravel()

    def _compute_metrics(
        self, y_true: Optional[np.ndarray], p: Optional[np.ndarray]
    ) -> Dict[str, Optional[float]]:
        """
        Compute classification metrics at current iteration
        If new metrics need to be taken into account, they can be computed and added here
        """
        # Define default container for all metrics
        metrics: Dict[str, Optional[float]] = {
            "acc": None,
            "brier": None,
            "aucroc": None,
            "aucpr": None,
            "nlpd": None,
        }
        # Safety check
        if y_true is None or p is None:
            return metrics

        y_true = y_true.ravel().astype(int)
        p = p.ravel().astype(np.float64)

        # Predicted labels at current iterations (consistent with choice of `pred_threshold`)
        y_hat = (p >= self.pred_threshold).astype(int)

        # Accuracy, assumes classes are well balanced, otherwise highly biased
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
        p_clip = np.clip(p, eps, 1.0 - eps)
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
        eigs = None

        if self.kernel is not None:
            # W
            try:
                W_param = getattr(self.kernel, "W")
                W = (
                    W_param.numpy().tolist()
                    if hasattr(W_param, "numpy")
                    else np.array(W_param).tolist()
                )
            except Exception:
                W = None

            # eta (scalar)
            try:
                eta_param = getattr(self.kernel, "eta")
                if eta_param is not None:
                    eta = float(
                        eta_param.numpy() if hasattr(eta_param, "numpy") else eta_param
                    )
            except Exception:
                eta = None

            # ard (vector)
            try:
                ard_param = getattr(self.kernel, "ard")
                if ard_param is not None:
                    ard_np = (
                        ard_param.numpy()
                        if hasattr(ard_param, "numpy")
                        else np.array(ard_param)
                    )
                    ard = ard_np.ravel().tolist()
            except Exception:
                ard = None

            # Optional eigenvalues of Gram on a small subset
            try:
                Xg = self.X_train if self.X_train is not None else None
                if Xg is not None:
                    # Could use reduce dim to keep it light
                    # m = min(64, Xg.shape[0])
                    # Otherwise full rank
                    m = Xg.shape[0]
                    K = self.kernel.K(
                        tf.convert_to_tensor(Xg[:m], dtype=tf.float64)
                    ).numpy()
                    eigs = np.maximum(np.linalg.eigvalsh(K), 0.0).tolist()
            except Exception:
                eigs = None

        return {"W": W, "eta": eta, "ard": ard, "eigs": eigs}

    def _snapshot_iteration(self, p_train, p_val, p_test) -> None:
        """
        Save per-iteration predictions, labels, and metrics for logging
        """
        # Predicted labels at current iterations (consistent with choice of `pred_threshold`)
        yhat_train = (
            (p_train >= self.pred_threshold).astype(int)
            if p_train is not None
            else None
        )
        yhat_val = (
            (p_val >= self.pred_threshold).astype(int) if p_val is not None else None
        )
        yhat_test = (
            (p_test >= self.pred_threshold).astype(int) if p_test is not None else None
        )

        # Record sequences (store empty lists if no probabilities are available)
        self.p_train_seq.append(
            p_train if p_train is not None else np.array([], dtype=np.float64)
        )
        self.p_val_seq.append(
            p_val if p_val is not None else np.array([], dtype=np.float64)
        )
        self.p_test_seq.append(
            p_test if p_test is not None else np.array([], dtype=np.float64)
        )

        self.y_train_seq.append(
            yhat_train if yhat_train is not None else np.array([], dtype=int)
        )
        self.y_val_seq.append(
            yhat_val if yhat_val is not None else np.array([], dtype=int)
        )
        self.y_test_seq.append(
            yhat_test if yhat_test is not None else np.array([], dtype=int)
        )

        # Snapshot metrics (dict)
        m_train = self._compute_metrics(self.Y_train, p_train)
        m_val = self._compute_metrics(self.Y_val, p_val)
        m_test = self._compute_metrics(self.Y_test, p_test)

        # Snapshot kernel params (dict)
        kernel_snapshot = self._snapshot_kernel()

        # Current training NLML
        nlml_ = float(self.loss_fn(self.model).numpy())

        # Append IterLog
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
                kernel_eigs=kernel_snapshot["eigs"],
                lr=self._lr_state["lr"],
                ema=self._lr_state["ema"],
                gamma=self.natgrad.gamma,
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
            name, value = "nlml", last.nlml
        return name, value

    def _check_for_best_iteration(self) -> None:
        """
        If the current iteration improves the chosen metric, snapshot parameters
        """
        last = self.logs[-1]
        name, value = self._selection_metric(last)
        if value is None or not np.isfinite(value):
            return
        # This is a minimization process
        if value < self._best_score:
            self._best_score = float(value)
            self._best_iter = self.step
            self._best_metric_name = name
            # Snapshot GPflow parameters (trainable + non-trainable)
            # Keep tensors detached so later mutations don't alias
            params_dict = gpflow.utilities.parameter_dict(self.model)
            self._best_params = {k: tf.identity(v) for k, v in params_dict.items()}

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
        self.loss_fn = self.external_training_loss_fn or (lambda m: m.training_loss())

        # Printing and logging
        self.logs: List[IterLog] = []  # Initialize containers for per-iter logging
        print_terminal_fr = max(1, self.maxiter // 10)  # Frequency of terminal msg

        # Training process (iterative)
        for self.step in range(1, self.maxiter + 1):
            # Perform a training iteration
            # (a) NaturalGradient step on variational parameters
            # Gamma is updated before the step is taken, if needed
            self._step_natural_gradient()
            # (b) Adam step on hyperparameters
            # Learning rate is adapted before the step is taken, if needed
            if self.enable_adaptation:
                self._adapt_learning_rate()
            self._step_adam()

            # Predict probabilities at current iteration
            p_train_at_iter = self._predict_prob(self.model, self.X_train)
            p_val_at_iter = (
                self._predict_prob(self.model, self.X_val) if self.has_val else None
            )
            p_test_at_iter = (
                self._predict_prob(self.model, self.X_test) if self.has_test else None
            )

            # Snapshot kernel and metrics at current iteration
            # Build and store the IterLog
            self._snapshot_iteration(p_train_at_iter, p_val_at_iter, p_test_at_iter)

            # Track improvement for `best` iteration
            self._check_for_best_iteration()

            # Print info to terminal
            if self.step % print_terminal_fr == 0 or self.step == 1:
                self._print_state_on_terminal()

            # Early stopping
            if self.enable_early_stopping:
                self._check_for_early_stopping()

        # Restore best iteration / checkpoint
        if self._best_params is not None:
            gpflow.utilities.multiple_assign(self.model, self._best_params)
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
        Build RunLog  and write JSON file
        Convert numpy arrays to lists
        """

        def _tolist_seq(seq: List[np.ndarray]) -> List[List[float]]:
            return [arr.astype(float).ravel().tolist() for arr in seq]

        def _tolist_seq_int(seq: List[np.ndarray]) -> List[List[int]]:
            return [arr.astype(int).ravel().tolist() for arr in seq]

        # Evaluate final predicted probabilities and labels
        # `final` doesn't mean from the last iteration but from the best iteration
        # `best` comes from the metric under examination, most of the time it's NLML
        # but it coudl be NLPD from the validation test
        # Choose best iteration index for `final` snapshot
        # Index of the best iteration
        best_idx = self._best_iter - 1

        p_train_best = (
            self.p_train_seq[best_idx].ravel().tolist() if self.p_train_seq else []
        )
        p_val_best = self.p_val_seq[best_idx].ravel().tolist() if self.p_val_seq else []
        p_test_best = (
            self.p_test_seq[best_idx].ravel().tolist() if self.p_test_seq else []
        )

        y_train_best = (
            self.y_train_seq[best_idx].ravel().tolist() if self.y_train_seq else []
        )
        y_val_best = self.y_val_seq[best_idx].ravel().tolist() if self.y_val_seq else []
        y_test_best = (
            self.y_test_seq[best_idx].ravel().tolist() if self.y_test_seq else []
        )

        self.run_log = RunLog(
            meta=deepcopy(self.cfg),
            logs=self.logs,
            p_train_seq=_tolist_seq(self.p_train_seq),
            p_train_best=p_train_best,
            y_train_seq=_tolist_seq_int(self.y_train_seq),
            y_train_best=y_train_best,
            p_val_seq=_tolist_seq(self.p_val_seq),
            p_val_best=p_val_best,
            y_val_seq=_tolist_seq_int(self.y_val_seq),
            y_val_best=y_val_best,
            p_test_seq=_tolist_seq(self.p_test_seq),
            p_test_best=p_test_best,
            y_test_seq=_tolist_seq_int(self.y_test_seq),
            y_test_best=y_test_best,
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
        nlml = [l.nlml for l in self.run_log.logs]
        ema = [l.ema for l in self.run_log.logs]

        ax1.plot(steps, nlml, linewidth=2, color="black", label="NLML")
        if (np.array(ema) == None).all() == False:
            ax1.plot(steps, ema, linewidth=1.5, color="grey", alpha=0.5, label="EMA")
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
            "val": getattr(self, "has_val", False),
            "test": getattr(self, "has_test", False),
        }
        curves = []

        for key, value in splits.items():
            if value == False:
                curves.append(None)
            else:
                # Extract labels and probabilities
                y_true, p = self._get_split(key)
                brier = brier_score_loss(y_true, p)  # Brier's score

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
                    b = next(i for i, c in enumerate(counts) if c < MIN_PER_BIN)
                    left = b - 1 if b - 1 >= 0 else None
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

    def _plot_learning_rates(self) -> None:
        """
        Plot learning rate (Adam), natural gradient gamma
        """
        iters = range(1, len(self.run_log.logs) + 1)
        lrs = [l.lr for l in self.run_log.logs]
        gamms = [l.gamma for l in self.run_log.logs]

        fig, ax1 = plt.subplots(figsize=(6, 4))
        # Plot learning rate (left y-axis)
        ax1.plot(iters, lrs, label="Adam LR", color="tab:blue")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Learning Rate", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        # Create a second axis for gamma
        ax2 = ax1.twinx()
        ax2.plot(iters, gamms, label="NatGrad gamma", color="tab:orange")
        ax2.set_ylabel("Gamma", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        # Combine legends from all axes
        lines, labels = [], []
        # for ax in [ax1, ax2, ax3]:
        for ax in [ax1, ax2]:
            l, lab = ax.get_legend_handles_labels()
            lines.extend(l)
            labels.extend(lab)
        ax1.legend(lines, labels, loc="upper right")

        fig.tight_layout()
        fig.savefig(self.run_dir / "04_learning_rates.png", dpi=150)
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

    def _plot_kernel_W(self) -> None:
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

    def _plot_kernel_eigs(self) -> None:
        """
        Plot kernel eigenvalues at the `best` iteration

        Uses the eigenvalues stored in self.run_log.logs[self._best_iter - 1].kernel_eigs
        Produces a semilog plot of sorted eigenvalues and an overlaid cumulative
        energy curve
        Annotates condition number and the effective rank at 99% energy
        """
        # Safety checks
        if self.run_log is None or not self.run_log.logs or self._best_iter is None:
            return

        best_idx = int(self._best_iter - 1)
        if best_idx < 0 or best_idx >= len(self.run_log.logs):
            return

        eigs = self.run_log.logs[best_idx].kernel_eigs
        if eigs is None or len(eigs) == 0:
            # Nothing to plot
            return

        # Prepare data
        v = np.asarray(eigs, dtype=float)
        v = np.where(np.isfinite(v), v, 0.0)
        v = np.maximum(v, 0.0)  # guard against tiny negatives from numerics
        v_sorted = np.sort(v)[::-1]
        idx = np.arange(1, len(v_sorted) + 1)

        # Cumulative energy
        total = float(v_sorted.sum())
        cum = np.cumsum(v_sorted) / total if total > 0 else None

        # Diagnostics
        vmax = float(np.max(v_sorted)) if v_sorted.size else 0.0
        vpos = v_sorted[v_sorted > 0]
        vmin_pos = float(np.min(vpos)) if vpos.size else 0.0
        cond = float(vmax / vmin_pos) if vmin_pos > 0 else float("inf")
        eff_rank = int(np.searchsorted(cum, 0.99) + 1) if cum is not None else 0

        # Plot
        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        ax1.semilogy(
            idx,
            v_sorted,
            marker="o",
            color="black",
            markerfacecolor="lime",
            markeredgecolor="black",
            linewidth=1.5,
            label="kernel eigenval",
        )
        ax1.set_xlim(np.min(idx), np.max(idx))

        ax1.set_xlabel("Eigenvalue index (sorted)")
        ax1.set_ylabel("Eigenvalue")

        # Cumulative on twin axis
        if cum is not None:
            ax2 = ax1.twinx()
            ax2.plot(
                idx,
                cum,
                linestyle="--",
                linewidth=1.2,
                color="black",
                label="cumulative energy",
            )
            ax2.set_ylim(0, 1.02)
            ax2.set_ylabel("Cumulative fraction")

            # Mark effective rank at 99%
            ax2.axhline(0.99, linestyle=":", linewidth=1.0, color="grey")
            ax2.axvline(eff_rank, linestyle=":", linewidth=1.0, color="grey")

        # Text box with diagnostics
        txt = [
            f"n = {len(v_sorted)}",
            f"spectral condition num = {cond:.2e}" if np.isfinite(cond) else "cond=∞",
            f"eff. rank@99% = {eff_rank}" if cum is not None else "eff. rank@99%=/",
            f"trace(K) = {total:.3g}",
        ]
        ax1.text(
            0.02,
            0.02,
            "\n".join(txt),
            transform=ax1.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, linewidth=0.5),
        )

        # Legend handling
        lines, labels = ax1.get_legend_handles_labels()
        if cum is not None:
            l2, lab2 = ax2.get_legend_handles_labels()
            lines += l2
            labels += lab2
        if lines:
            ax1.legend(
                lines,
                labels,
                title=f"Iter {self._best_iter}",
                loc="best",
                fontsize=8,
                title_fontsize=8,
            )

        fig.tight_layout()
        fig.savefig(self.run_dir / "07_kernel_eigs.png", dpi=150)
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

    def _plot_confusion_matrix(self, iter: Optional[int] = None) -> None:
        """
        Plot confusion matrix for all the available sets
        P(y=1) > self.pred_threshold
        If `iter` is provided use that specific iteration, otherwise use the `best` iteration
        """

        if iter is None:
            iter = self._best_iter
            self._plot_confusion_matrix(iter=iter)

        else:
            iter_idx = iter - 1

            y = self.Y_train.ravel().astype(int)
            y_pred = np.array(self.run_log.y_train_seq[iter_idx]).ravel()
            cm = [confusion_matrix(y, y_pred)]
            label = ["train"]
            ks = 1

            if self.has_val:
                y = self.Y_val.ravel().astype(int)
                y_pred = np.array(self.run_log.y_val_seq[iter_idx]).ravel()
                cm.append(confusion_matrix(y, y_pred))
                label.append("val")
                ks += 1

            if self.has_test:
                y = self.Y_test.ravel().astype(int)
                y_pred = np.array(self.run_log.y_test_seq[iter_idx]).ravel()
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
                    axes[k].set_title(f"{label[k]} (Iter {iter})", fontsize=9)
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
        Build a 2D decision surface in the selected feature-pair space using interpolation from predicted probabilties
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
        f1, f2 = getattr(self, "feature_pair", (0, 1))
        f1 = int(np.clip(int(f1), 0, int(self.nf) - 1))
        f2 = int(np.clip(int(f2), 0, int(self.nf) - 1))
        if f1 == f2 and int(self.nf) > 1:
            f2 = (f1 + 1) % int(self.nf)

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
        p = np.asarray(self.run_log.p_train_seq[iter_idx], dtype=float).ravel()

        # Interpolate P(y=1) onto the grid
        pts = np.c_[fX, fY]
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

    def _plot_features_and_boundary(self, iter: Optional[int] = None) -> None:
        """
        Scatter the selected feature pair for train / val / test and overlay the decision boundary
        Feature pair is read from `self.feature_pair` if present; otherwise defaults to (0, 1)

        TODO: when more than 2 features are generated, the boundary decision projection to a pair's plane is bad

        """
        # Define iteration, if not provided use self._best_iter
        if iter is None:
            iter = self._best_iter
            self._plot_features_and_boundary(iter)
        else:
            boundary = self._compute_decision_boundary(iter=iter)
            if not boundary:
                return

            f1, f2 = boundary["f1"], boundary["f2"]
            fX_train, fY_train = boundary["fX_train"], boundary["fY_train"]
            fX_val, fY_val = boundary.get("fX_val"), boundary.get("fY_val")
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

    def _get_posterior_q(self) -> Optional[Dict[str, Any]]:
        """
        Extract posterior mean m and variance diag v from (q_mu, q_sqrt)
        Supports both full and diagonal variational covariances

        Note: f is the latent function
              q(f) is introduced by the variational method
              q(f) ~ N(m, S) with S = L L.T using Cholesky decomposition

              This is where they are stored:
                self.model.q_mu (for m_i)
                self.model.q_sqrt (for sqrt(v_i))

              with S = q_sqrt q_sqrt.T
              and v_i = S_ii or v_i = q_sqrt ^2 if diagonal
        """
        if not (hasattr(self.model, "q_mu") and hasattr(self.model, "q_sqrt")):
            return None

        # Mean
        m = self.model.q_mu.numpy().ravel().astype(np.float64)

        # Covariance representation
        qs = self.model.q_sqrt.numpy()
        eps = 1e-12

        if qs.ndim == 3:  # [P, N, N] full Cholesky, P=1 for binary
            L = qs[0].astype(np.float64)
            # diag variances are row-wise sums of L^2
            v = np.sum(L * L, axis=1)
            # logdet(S) = 2 * sum(log(diag(L)))
            diagL = np.abs(np.diag(L)) + eps
            logdetS = float(2.0 * np.sum(np.log(diagL)))
            full = True
        elif qs.ndim == 2:  # [P, N] diagonal representation, P=1
            d = qs[0].astype(np.float64)
            v = d * d
            logdetS = float(np.sum(np.log(v + eps)))
            L, full = None, False
        else:  # very defensive fallback
            d = np.ravel(qs).astype(np.float64)
            v = d * d
            logdetS = float(np.sum(np.log(v + eps)))
            L, full = None, False

        return {"m": m, "v": v, "L": L, "logdetS": logdetS, "full": full}

    def _plot_vgp_latent_marginals(self) -> None:
        """
        Latent mean vs uncertainty for training points
        Note: confident predictions should have large |m_i| values and small sqrt(v_i)
              large means and variances could lead to a confused posterior

        This is where they are stored:
            self.model.q_mu (for m_i)
            self.model.q_sqrt (for sqrt(v_i))
            Make sure self.model.q_mu.trainable == True and self.model.q_sqrt.trainable == True
            Make sure self.model.q_sqrt.shape has form [P, N, N], this means

        Errorbar plot: m_i ± 2*sqrt(v_i), downsampled for readability

        Case of unwanted situation:
            IF almost all points: mean ~0 with wide symmetric bars
                The posterior over f at most training inputs is not pulled away from the prior
                The model is not extracting separability from the data except for a handful of cases
        """
        q = self._get_posterior_q()
        if q is None:
            return
        m, v = q["m"], q["v"]
        N = m.size
        # show the most uncertain points (largest variance), capped for readability
        k = min(N, 200)
        sel = np.argsort(-v)[:k]

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.errorbar(np.arange(k), m[sel], yerr=2.0 * np.sqrt(v[sel]), fmt=".", lw=1)
        ax.axhline(0.0, linestyle="--", linewidth=1, color="0.5")
        ax.set_xlabel("Index (top-variance subset)")
        ax.set_ylabel("Latent mean ± 2σ")
        ax.set_title("VGP latent marginals")
        fig.tight_layout()
        fig.savefig(self.run_dir / "11_vgp_latent_marginals.png", dpi=150)
        plt.close(fig)

    def _plot_posterior_q_standardized(self) -> None:
        """
        Histogram of standardized latents z_i = m_i / sqrt(v_i)
        If the variational posterior is well-behaved, this should look roughly N(0, 1) under a prior-ish regime
        Post-training might deviate based on the data fit

        Note: Should be spread away from 0 on confidently separable data
              A heap near 0 means the model can't separate classes with the current kernel/hyperparams

        Case of unwanted situation:
            A lot of entries at z ~ 0, with few outliers
            The training data is not informative
            A broader |z| spread, with a healthy chunk beyond |z| > 2 would be noticed
            Can be identified as "nearly prior" behavior
        """
        q = self._get_posterior_q()
        if q is None:
            return
        m, v = q["m"], q["v"]
        z = m / (np.sqrt(v) + 1e-12)

        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        ax.hist(z, bins=40, density=True, alpha=0.8)
        ax.set_xlabel("z = m / σ")
        ax.set_ylabel("Density")
        ax.set_title("Posterior standardized latents")
        fig.tight_layout()
        fig.savefig(self.run_dir / "12_posterior_q_standardized.png", dpi=150)
        plt.close(fig)

    def _plot_posterior_q_correlation_block(self, max_block: int = 64) -> None:
        """
        Correlation heatmap for a subset of the variational covariance
        For full q_sqrt, builds S_I = L[I,:] L[I,:]^T for evenly spaced indices I
        For diagonal q_sqrt, the correlation matrix is identity

        Case of unwanted situation:
            A diagonal q_sqrt translates to independence across training points in q(f)
            If intended full covariances, this is a bug
            If intended diag, it is working as designed but it limits what the posterior can express

            Correlation heatmap ~ identity
            It is possible that your kernel prior is already near diagonal, K ~ σ^2 I
        """
        q = self._get_posterior_q()
        if q is None:
            return

        N = q["m"].size
        k = min(max_block, N)
        idx = np.linspace(0, N - 1, k, dtype=int)

        if q["full"] and q["L"] is not None:
            Lsub = q["L"][idx, :]  # (k, N)
            Ssub = Lsub @ Lsub.T  # (k, k)
            sd = np.sqrt(np.diag(Ssub) + 1e-12)
            C = Ssub / (sd[:, None] * sd[None, :])
        else:
            C = np.eye(k, dtype=float)

        fig, ax = plt.subplots(figsize=(5.6, 5.6))
        im = ax.imshow(C, vmin=-1.0, vmax=1.0, cmap="RdBu_r", origin="lower")
        ax.set_title("Posterior correlation (subset)")
        ax.set_xlabel("Index")
        ax.set_ylabel("Index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(self.run_dir / "13_posterior_q_correlation_block.png", dpi=150)
        plt.close(fig)

    def _plot_posterior_covariance_eigs(self) -> None:
        """
        Spectrum of the posterior covariance S = q_sqrt q_sqrt.T
        For diagonal forms, this reduces to sorted variances
        For full forms, uses a subset for stability

        The rank at 99% cumulative sum show the number of dimensions needed to explain the variance in the dataset

        Note: Eigenvalues of S tell you effective dimensionality of the posterior variability
              Compute condition number, effective rank at 95-99%, and cumulative energy
              If rank collapses to 1, the kernel is pretending the world is a straight line

        Case of unwanted situation:
            Sorted eigs almost flat around ~1 after a brief ramp
            For diagonal q, these are just the variances, so v_i ~ constant at ~1 for almost all i
            Translated to the posterior variance barely shrank from the prior
        """
        q = self._get_posterior_q()
        if q is None:
            return

        eps = 1e-16

        # Build eigenvalues
        if q["full"] and q["L"] is not None:
            # use a modest subset (max 256 values) so this doesn't turn into a PhD in linear algebra at runtime
            m = min(256, q["L"].shape[0])
            Ls = q["L"][:m, :]
            Ssub = Ls @ Ls.T
            eigs = np.maximum(np.linalg.eigvalsh(Ssub), eps)
        else:
            # Diagonal q, eigenvalues are the variances
            eigs = np.maximum(q["v"], eps)

        v_sorted = np.sort(eigs)[::-1]
        idxs = np.arange(1, len(v_sorted) + 1)

        # Cumulative energy
        total = float(v_sorted.sum())
        cum = np.cumsum(v_sorted) / (total if total > 0 else 1.0)

        # Diagnostics
        vmax = float(v_sorted[0]) if v_sorted.size else 0.0
        vpos = v_sorted[v_sorted > 0]
        vmin_pos = float(vpos[-1]) if vpos.size else 0.0
        cond = float(vmax / vmin_pos) if vmin_pos > 0 else float("inf")
        eff_rank = int(np.searchsorted(cum, 0.99) + 1) if total > 0 else 0

        # Plot: match the style of _plot_kernel_eigs()
        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        ax1.semilogy(
            idxs,
            v_sorted,
            marker="o",
            color="black",
            markerfacecolor="royalblue",
            markeredgecolor="black",
            linewidth=1.5,
            label="posterior cov eigenval",
        )
        ax1.set_xlabel("Eigenvalue index (sorted)")
        ax1.set_ylabel("Eigenvalue")
        # ax1.set_title("Posterior covariance eigenvalues (q)")
        ax1.set_xlim(np.min(idxs), np.max(idxs))

        # Cumulative energy on twin axis
        ax2 = ax1.twinx()
        ax2.plot(
            idxs,
            cum,
            linestyle="--",
            linewidth=1.2,
            color="black",
            label="cumulative energy",
        )
        ax2.set_ylim(0, 1.02)
        ax2.set_ylabel("Cumulative fraction")

        # Mark effective rank at 99%
        ax2.axhline(0.99, linestyle=":", linewidth=1.0, color="grey")
        ax2.axvline(eff_rank, linestyle=":", linewidth=1.0, color="grey")

        # Text box with diagnostics
        txt = [
            f"n = {len(v_sorted)}",
            f"spectral condition num = {cond:.2e}" if np.isfinite(cond) else "cond=∞",
            f"eff. rank@99% = {eff_rank}",
            f"sum = {total:.3g}",
        ]
        ax1.text(
            0.02,
            0.02,
            "\n".join(txt),
            transform=ax1.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, linewidth=0.5),
        )

        # Legend handling
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if lines1 or lines2:
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)

        fig.tight_layout()
        fig.savefig(self.run_dir / "14_posterior_covariance_eigs.png", dpi=150)
        plt.close(fig)

    def _plot_uncertainty_vs_error(self) -> None:
        """
        Relate uncertainty to mistakes on the train set at the `best` iteration
        x-axis: predictive probability p(y=1|x)
        y-axis: posterior σ = sqrt(v)
        Color wrong vs correct using your train labels and decision threshold `self.pred_threshold`

        Note: wrong labels should coincide with high v_i (high y values) or small |z_i|

        Case of unwanted situation:
            Posterior σ is ~1 for nearly everything, regardless of p(y=1|x)
            IF wrong points don't have larger σ
                It means your current "uncertainty" (from q) isn't predictive of error
                Either q_sqrt didn't move, or the probabilities are computed in a way that ignores v (e.g., using only the mean f)
                If the probabilities are true variational predictives, expect high variance to shove probabilities toward 0.5
        """
        q = self._get_posterior_q()
        if q is None:
            return
        if self.Y_train is None or not self.run_log or not self.run_log.p_train_best:
            return

        y = self.Y_train.ravel().astype(int)
        p = np.array(self.run_log.p_train_best, dtype=np.float64)
        yhat = (p >= self.pred_threshold).astype(int)
        err = yhat != y
        sigma = np.sqrt(q["v"] + 1e-12)

        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        ax.scatter(p[~err], sigma[~err], s=12, alpha=0.6, label="correct")
        ax.scatter(p[err], sigma[err], s=12, alpha=0.9, label="wrong")
        ax.set_xlabel("Predicted probability (train)")
        ax.set_ylabel("Posterior σ")
        ax.set_title("Uncertainty vs. classification errors (train)")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(self.run_dir / "15_uncertainty_vs_error.png", dpi=150)
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

    def _plot_sv_evolution(self) -> None:
        """
        Plot singular values over the iterations in a raster plot-like visualization
        A box over the plot highlights the `best` iteration
        """

        mat_iter = np.zeros((self.nf, self.maxiter))

        for iter in range(int(self.maxiter)):
            mat = self._compute_features_matrix(iter=iter)
            try:
                s = self._compute_svd(mat)  # full dimension
                s = np.sort(s)[::-1]  # sort descending for nicer visual
            except:
                s = np.array([np.nan] * self.nf)
            mat_iter[:, iter] = s

        fig, ax = plt.subplots(figsize=(7, 4.5))
        im = ax.imshow(
            mat_iter,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="plasma",
        )
        cbar = plt.colorbar(im, ax=ax, vmin=0)
        ax.set_xlabel("Iter")
        ax.set_ylabel("Singular values")
        Lylim = self.nf - 0.5
        ax.set_xlim(-0.5, self.maxiter - 0.5)
        ax.set_ylim(-0.5, Lylim)
        # Box around best column
        from matplotlib.patches import Rectangle

        rect = Rectangle(
            (self._best_iter - 0.5, -0.5),
            1.0,
            self.nf,
            fill=False,
            linewidth=2.0,
            edgecolor="red",
        )
        ax.add_patch(rect)
        # Arrow + label
        ax.annotate(
            f"best @ iter {self._best_iter}",
            xy=(self._best_iter, self.nf + 0.1),
            xytext=(self._best_iter, self.nf + 0.8),
            ha="center",
            arrowprops=dict(arrowstyle="->", lw=1.5, color="red"),
            color="red",
            fontsize=9,
        )

        ax.grid(False)
        fig.tight_layout()
        fig.savefig(self.run_dir / "17_singular_values_raster.png", dpi=150)
        plt.close(fig)
