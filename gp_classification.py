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
import os
from copy import deepcopy
import datetime as dt
from dataclasses import dataclass, asdict

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
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
import gpflow

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
        self.W_over_iters: List[np.ndarray] = []  # list of (s, nf)

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

        # if self.kfolds and self.kfolds > 1:
        #    print(f"[CV] Running {self.kfolds}-fold on the train set")
        #    self._kfold_cv_nlml()

        self._initialize_W_matrix()  # W_init
        self._train()
        # self._make_visual_summary()  # Visual outputs
        self._write_config_file()  # Write config file to `config.json`
        self._build_and_write_runlog()  # Build and write RunLog
        self._print_message(which="end")

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
            left_id = [
                i
                for i in map(
                    _idx_if_present,
                    ["fc1", "c1", "cp1", "fc3", "c3", "fc5", "c5", "cp5"],
                )
                if i is not None
            ]
            right_id = [
                i
                for i in map(
                    _idx_if_present,
                    ["fc2", "c2", "cp2", "fc4", "c4", "fc6", "c6", "cp6"],
                )
                if i is not None
            ]
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

    def _build_optimizers(self) -> None:
        """
        Build GPflow optimizers
        """
        # Initialization of self contained structure for learning rate adaptation
        self._lr_state = {
            "step": 0,
            "lr": self.learning_rate,
            "base_lr": self.learning_rate,
            "min_lr": 1e-5,
            "decay_factor": 0.5,  # decay factor on plateau
            "patience": max(int(0.15 * self.maxiter), 50),  # ~15% of self.maxiter
            "cooldown": 0,
            "tolerance": 1e-4,
            "best": 1e4,
            "ema": None,  # evaluation of exponential moving average (EMA)
            "ema_beta": 0.9,  # decay factor for EMA
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
        2) Boosting (next 15%): boost from self.gamma to 3 * self.gamma
        3) Cosine decay (remaining 80%): decaying from 3 * self.gamma to 0.5 * self.gamma
        """
        # Define values of gammas to use in the schedule
        gamma_main = self.gamma
        gamma_boost = 3 * self.gamma
        gamma_floor = 0.5 * self.gamma
        safety_gamma = 1e-12

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
            # Adapt gamma value before taking the step if adaptation flag is True
            if self.enable_adaptation:
                self.natgrad.gamma = self._adapt_gamma()
            # Natrual Gradient step
            self.natgrad.minimize(
                lambda: self.loss_fn(self.model),
                var_list=[(self.model.q_mu, self.model.q_sqrt)],
            )
            return
        else:
            return

    def _adapt_learning_rate(self) -> float:
        """
        Adapt Adam base learning rate with the following schedule
        1) Warmup (2% of self.maxiter): linearly ramping from 0 to self.learning_rate
        2) Reduce-on-plateau: when the monitored objective stops improving, implement cosine decay

        Monitored metric:
        - If `use_validation_for_adaptation` and a validation set exist, use `nlpd_val`
        - Otherwise fall back to `nlml` (training objective)
        """
        # Grab current state of structure for learning rate adaptation
        st = self._lr_state
        st["step"] = self.step

        # Helper function to set optimizer LR with clipping
        def _assign_lr(new_lr: float) -> float:
            new_lr = float(np.clip(new_lr, st["min_lr"], st["base_lr"]))
            try:
                # Works when the optimizer stores a tf.Variable
                self.opt.learning_rate.assign(new_lr)
            except Exception:
                # Fallback when it's a plain Python float hyperparameter
                self.opt.learning_rate = new_lr
            st["lr"] = new_lr
            return new_lr

        # Warmup (2% of maxiter): linearly ramping from 0 to self.learning_rate
        if st["step"] <= st["warmup_steps"]:
            # Assign and update learning rate
            return _assign_lr(st["base_lr"] * st["step"] / max(1, st["warmup_steps"]))

        # Consider to either use a fixed value learning rate after warmup or decay schedule
        else:
            # Grab last entry in the log report, this should exist since warmup has already happened
            last = self.logs[-1]

            # Define metric to use for learning rate adaptation
            # Check for flags, specifically validation first (if requested and available), else training
            use_val = bool(
                getattr(self, "use_validation_for_adaptation", False)
            ) and bool(getattr(self, "has_val", False))
            metric = None
            if use_val:
                metric = last.nlpd_val  # grab NLPD on validation set
            else:
                metric = last.nlml  # grab NLML on train set

            # If metric is unusable, keep current learning rate
            if metric is None or not np.isfinite(metric):
                return _assign_lr(st["lr"])  # assign and update learning rate
            else:
                if st["ema"] is None:
                    st["ema"] = metric  # assign current metric to EMA
                else:
                    # Update EMA using current metric and previous EMA
                    st["ema"] = (
                        st["ema_beta"] * st["ema"] + (1.0 - st["ema_beta"]) * metric
                    )

                # Evaluate signed relative change
                signed_rel_change = (st["ema"] - st["best"]) / (st["best"] + 1e-12)
                # Check if EMA is better than best value, compare it to tolerance
                if signed_rel_change < 0 and abs(signed_rel_change) > st["tolerance"]:
                    st["best"] = st["ema"]  # update best value with current EMA
                    st["cooldown"] = st["patience"]  # reset cooldown
                    return _assign_lr(
                        st["lr"]
                    )  # assign current learning rate (no update)
                else:
                    # No update on `best` value occurs
                    if st["cooldown"] > 0:
                        st["cooldown"] -= 1  # count a step down from cooldown
                        return _assign_lr(
                            st["lr"]
                        )  # assign current learning rate (no update)
                    else:
                        # Ran out of `patience`, too long in plateau condition
                        # Reset `patience` and apply decay to learning rate
                        st["cooldown"] = st["patience"]
                        new_lr = st["lr"] * st["decay_factor"]
                        print(f"  [lr] changed to {new_lr:.4f} at iter {self.step}")
                        return _assign_lr(
                            new_lr
                        )  # assign new learning rate (decay update)

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

    def _snapshot_kernel_params(self) -> Dict[str, Optional[Any]]:
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

            # Optional eigenvalues of Gram on a small subset (cheap-ish)
            try:
                Xg = self.X_train if self.X_train is not None else None
                if Xg is not None:
                    # Use a small subset to keep it light
                    m = min(64, Xg.shape[0])
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
        kernel_snapshot = self._snapshot_kernel_params()

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
        """ """
        # Define model by specifying kernel, likelihood, and method
        self._build_model()  # self.kernel, self.likelihood, self.model

        # Define optimizers
        self._build_optimizers()  # self.opt, self.natgrad

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
