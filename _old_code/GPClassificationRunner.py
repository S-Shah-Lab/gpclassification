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
     - X: {"train": (N_train, s, s), "test": (N_test, s, s)}
     - Y: {"train": (N_train,) or (N_train,1), "test": (N_test,) or (N_test,1)}
  X_train (GPflow):  (N_train, D) = (N_train, s*s)
  X_test  (GPflow):  (N_test, D)
  Y_train (GPflow):  (N_train, 1)
  Y_test  (GPflow):  (N_test, 1)

Kernel params inside CustomKernel (from kernels.py):
  W        : (s, nf) trainable spatial filters
  eta(opt) : () global kernel function scalar (if eta_flag)
  ard(opt) : (nf,) kernel per-filter scaling (if ard_flag)

Predictions:
  p_train (per iter): (N_train,) probabilities in [0,1]
  p_test  (per iter): (N_test,)

Major third-party packages used
-------------------------------
- numpy: numerical operations
- matplotlib: plotting
- scikit-learn: model selection utilities (train/test split, metrics)
- gpflow: Gaussian Process library (TensorFlow-based)
- tensorflow: auto-diff + optimizers
- mne: EEG sensor layouts + topomap visualization
- imageio: GIF writing **from NumPy arrays** (no temp files needed)

Author: Giacomo Scanavini
"""

# --------------------------------------------- Python specific libraries ----------------------------------------------
from __future__ import annotations

# type hint is the concept of addying type information to the code
# type annotation is the syntax used (:) or (->) to implement the type hint concept
# type annotation is a way to clarify the expected type of variables and parameters
# Python interpreter stores type annotations as strings (more concise type hint syntax)
# Allows the use of type hints for types that are defined later in the module preventin NameError

import json
import math
import datetime as dt
from dataclasses import dataclass, asdict

# dataclass: decorator that examines a class to get the fields (class variable with a type annotations)
# asdict: method that converts the dataclass object into a dictionary (grabs fields and stores them into a dict)

from pathlib import Path

# Path: represents paths to files as objects which have methods (contrary to os.path which represents them as strings)

from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)  # runtime support for type hints

# ----------------------------------------- Third party libraries (mandatory) ------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import gpflow  # GP models (TensorFlow backend)
import tensorflow as tf
from scipy.interpolate import griddata
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
    average_precision_score,
)
from kernels import CustomKernel  # custom covariance function


# ------------------------------- Third party libraries (optionals for extra glitters) ---------------------------------
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

################################################################################################################ UTILITY


def _now_stamp(mode: str = "") -> str:
    """
    Return a timestamp string YYYYMMDD_HHMMSS
    This is used as label for run folder names so they do not overwrite
    """
    if mode == "nice":
        return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: Path) -> None:
    """
    Create folder and parents if they do not exist
    If parents is True, any missing parents of this path are created as needed
    If exist_ok is False, FileExistsError is raised if the target directory already exists
    """
    p.mkdir(parents=True, exist_ok=True)


############################################################################################## DATA LOGS FOR BOOKKEEPING


@dataclass
class IterLog:
    """
    Per-iteration training record

    Attributes:
      step       : int; iteration index
      nlml       : float; e.g. negative ELBO (VGP.training_loss), approximation of NLML in GP classification
      acc_train  : float; train accuracy at threshold `pred_threshold`
      acc_test   : float; test accuracy at threshold `pred_threshold`
      brier      : float; Brier score loss on test
      grad_norm  : float; L2 norm of gradients that step
      param_norm : float; L2 norm of model parameters that step
      kernel_eigs: Optional[List[float]]; eigenvalues of kernel function (must be non negative since the kernel must be Gram)
      eta        : Optional[float]; scalar kernel scale (if enabled by _flag_eta)
      ard        : Optional[List[float]]; per-filter scale (if enabled by _flag_ard)
    """

    step: int
    nlml: float
    acc_train: float
    acc_test: float
    brier: float
    grad_norm: float
    param_norm: float
    kernel_eigs: Optional[List[float]] = None
    eta: Optional[float] = None
    ard: Optional[List[float]] = None


@dataclass
class RunLog:
    """
    Container for the entire run logs
    This is converted to JSON format

    Attributes:
      meta:          Dict[str,str]; config snapshot (strings for portability)
      logs:          List[IterLog]; one object entry per training iteration
      p_train_seq:   List[List[int]]; per-iter train prob predictions (prob: 0 < x < 1)
      p_test_seq:    List[List[int]]; per-iter test prob predictions (prob: 0 < x < 1)
      p_train_final: List[float]; final train probabilities (N_train,)
      p_test_final:  List[float]; final test probabilities (N_test,)
      y_train_seq:   List[List[int]]; per-iter train hard predictions (labels: 0/1)
      y_test_seq:    List[List[int]]; per-iter test hard predictions (labels: 0/1)
      y_train_final: List[float]; final train labels (N_train,)
      y_test_final:  List[float]; final test labels (N_test,)
    """

    meta: Dict[str, str]
    logs: List[IterLog]
    p_train_seq: List[List[int]]
    p_test_seq: List[List[int]]
    p_train_final: List[float]
    p_test_final: List[float]
    y_train_seq: List[List[int]]
    y_test_seq: List[List[int]]
    y_train_final: List[float]
    y_test_final: List[float]

    def to_json(self) -> str:
        """
        Return a pretty-printed JSON string of the entire run log
        """
        d = asdict(self)
        return json.dumps(d, indent=2)


###################################################################################################### GP CLASSIFICATION

ArrayOrDict = Union[
    np.ndarray, Dict[str, np.ndarray]
]  # define a type hint which could be an array or a dict


class GPClassificationRunner:
    """
    Handles the entire classification process trying to be as generic as possible
    From data loading, training, logging, and visual outputs

    Key Config (constructor):

    Data shapes (expected):

    """

    def __init__(
        self,
        # Input variables
        X: ArrayOrDict,
        Y: ArrayOrDict,
        dataset_label: str,
        ch_names: List[str],
        ch_xy: Dict[str, Tuple[float, float]],
        # Model / kernel
        weights_init: str = "random",  # 'random' | 'ones' | 'manual'
        nf: int = 2,
        eta_flag: bool = False,
        ard_flag: bool = False,
        logged_flag: bool = True,
        kernel_type: str = "RBF",
        # Training
        frac_train: float = 0.5,  # used only if X,Y are arrays
        model_class: type = gpflow.models.VGP,
        model_kwargs: Optional[Dict] = None,
        likelihood_class: type = gpflow.likelihoods.Bernoulli,
        likelihood_kwargs: Optional[Dict] = None,
        training_loss_fn: Optional[Callable[[gpflow.Module], tf.Tensor]] = None,
        predict_y_fn: Optional[
            Callable[[gpflow.Module, np.ndarray], Tuple[np.ndarray, np.ndarray]]
        ] = None,
        learning_rate: float = 0.01,
        maxiter: int = 500,
        pred_threshold: float = 0.5,  # decision boundary in binary classification p(y=1) >= pred_threshold
        random_state: int = 42,
        # GIF controls
        gif_flag: bool = True,  # generate or not gifs
        gif_stride: int = 1,  # sample every k iterations
        gif_max_frames: Optional[int] = None,  # auto-raise stride to cap frames
        synced_gif: bool = True,  # build synced dashboard GIF
        topomap_filters_for_gif: int = 2,  # animate first k filters of W
        # Run naming / Logging
        results_dir: str = "./results",
        run_name: Optional[str] = None,
    ) -> None:

        # --- Inputs variables ---
        self.X: ArrayOrDict = X
        self.Y: ArrayOrDict = Y
        self.dataset_label = dataset_label

        # --- Model / Kernel ---
        self.weights_init = weights_init
        self.nf = nf
        self.eta_flag = eta_flag
        self.ard_flag = ard_flag
        self.logged_flag = logged_flag
        self.kernel_type = kernel_type

        # --- Training ---
        self.frac_train = frac_train
        self.model_class = model_class
        self.likelihood_class = likelihood_class
        self.model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
        self.likelihood_kwargs = (
            {} if likelihood_kwargs is None else dict(likelihood_kwargs)
        )
        self.external_training_loss_fn = training_loss_fn
        self.external_predict_y_fn = predict_y_fn
        self.learning_rate = learning_rate
        self.maxiter = maxiter
        self.pred_threshold = pred_threshold
        self.random_state = random_state

        # --- GIF controls ---
        self.gif_flag = gif_flag
        self.gif_stride = max(1, int(gif_stride))
        self.gif_max_frames = gif_max_frames
        self.synced_gif = synced_gif
        self.topomap_filters_for_gif = max(1, int(topomap_filters_for_gif))

        # --- Run naming / Logging ---
        self.results_root = Path(results_dir)
        self.run_name = run_name or f"run_{_now_stamp()}"
        self.run_dir = self.results_root / self.run_name
        _ensure_dir(self.run_dir)  # Create folder

        if HAS_MNE:
            # Build montage for visualization of spatial filter
            self.ch_names = [str(n).lower() for n in ch_names]
            self.ch_xy = {
                str(k).lower(): (float(x), float(y)) for k, (x, y) in ch_xy.items()
            }
            self.montage_info = self._build_montage_from_xy(self.ch_names, self.ch_xy)

        # --- Placeholders set after data load / training ---
        self.X_train: Optional[np.ndarray] = None  # (N_train, D)
        self.X_test: Optional[np.ndarray] = None  # (N_test, D)
        self.Y_train: Optional[np.ndarray] = None  # (N_train, 1)
        self.Y_test: Optional[np.ndarray] = None  # (N_test, 1)
        self.W_init: Optional[np.ndarray] = None  # (s, nf)
        self.model: Optional[gpflow.models.Model] = None
        self.kernel: Optional[CustomKernel] = None

        # Per-iteration snapshots
        self.W_over_iters: List[np.ndarray] = []  # list of (s, nf)
        self.p_train_seq: List[np.ndarray] = []  # list of probabilty arrays (N_train,)
        self.p_test_seq: List[np.ndarray] = []  # list of probabilty arrays (N_test,)
        self.y_train_seq: List[np.ndarray] = []  # list of label arrays (N_train,)
        self.y_test_seq: List[np.ndarray] = []  # list of label arrays (N_test,)

        self.run_log: Optional[RunLog] = None

        # Persist config files now (so aborted runs still have configs)
        self._write_config_files()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~ High level method ~~~~~~~~~~~~~~~

    def run(self) -> None:
        """
        Execute the full pipeline
        """
        self._print_message(which="start")

        self._load_and_prepare_data()  # X_train, X_test, Y_train, Y_test, N_train, N_test, s
        self._initialize_W_matrix()  # W_init
        # self._train_and_log()  # kernel, model, W_over_iters, p_train_seq, p_test_seq, y_train_seq, y_test_seq
        self._train_and_log_early_stop()  # adds natural gradient for variational parameters and early stop
        self._save_metrics_json()  # Logging
        self._make_visual_summary()  # Visual outputs

        self._print_message(which="end")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~ Low level methods ~~~~~~~~~~~~~~~

    # --------------- Bookkeeping / Logging --------------- #
    def _print_message(self, which: str) -> None:
        """
        Method used to print messages on terminal, mostly for quick use while things run
        """
        # ANSI color codes for sprinkly terminal
        GREEN = "\033[92m"
        CYAN = "\033[96m"
        RESET = "\033[0m"

        if which == "start":
            print(f"\n=== RUN START: {_now_stamp(mode='nice')} ===")
            print(f"{GREEN}{self.run_name}{RESET}\n")
        elif which == "end":
            print(f"\n=== RUN END: {_now_stamp(mode='nice')} ===")
        else:
            return

    def _save_metrics_json(self) -> None:
        """
        Write per-iteration metrics and predictions to metrics.json
        """
        # Manually check self.run_log has been generated
        # AssertionError is raised if a method relying on run_log is called and this is None
        assert self.run_log is not None
        with open(self.run_dir / "metrics.json", "w") as f:
            f.write(self.run_log.to_json())

    def _write_config_files(self) -> None:
        """
        Write config.json (machine-friendly) and config.txt (human-friendly) for bookkeeping and recalling
        Done at the *start* of the run
        Shapes are appended after data load
        """
        cfg = {
            # Naming
            "run_name": self.run_name,
            "dataset_label": self.dataset_label,
            "results_dir": str(self.results_root.resolve()),
            "timestamp_start": _now_stamp(),
            # Input variables
            "data_input_mode": "dict" if isinstance(self.X, dict) else "array",
            "channels_count": len(self.ch_names),
            # Model / kernel
            "weights_init": self.weights_init,
            "nf": self.nf,
            "eta_flag": self.eta_flag,
            "ard_flag": self.ard_flag,
            "logged_flag": self.logged_flag,
            "kernel_type": self.kernel_type,
            # Training
            "frac_train": self.frac_train,
            "model_class": self.model_class.__name__,
            "model_kwargs": self.model_kwargs,
            "likelihood_class": self.likelihood_class.__name__,
            "likelihood_kwargs": self.likelihood_kwargs,
            "has_custom_training_loss_fn": self.external_training_loss_fn is not None,
            "has_custom_predict_y_fn": self.external_predict_y_fn is not None,
            "learning_rate": self.learning_rate,
            "maxiter": self.maxiter,
            "pred_threshold": self.pred_threshold,
            "random_state": self.random_state,
            # GIF controls
            "gif_flag": self.gif_flag,
            "gif_stride": self.gif_stride,
            "gif_max_frames": self.gif_max_frames,
            "synced_gif": self.synced_gif,
            "topomap_filters_for_gif": self.topomap_filters_for_gif,
        }
        _ensure_dir(self.run_dir)
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)

        lines = [
            "RUN CONFIGURATION",
            "=================",
            f"Run name       : {self.run_name}",
            f"Dataset label  : {self.dataset_label}",
            f"Results root   : {self.results_root.resolve()}",
            f"Timestamp start: {cfg['timestamp_start']}",
            "",
            "Input variables",
            "---------------",
            f"Data input mode       : {'dict' if isinstance(self.X, dict) else 'array'}",
            f"Number of EEG channels: {len(self.ch_names)}",
            "",
            "Model / Kernel",
            "---------------",
            f"W matrix initiation : {self.weights_init}",
            f"# filters (nf)      : {self.nf}",
            f"Eta (filter scaling): {self.eta_flag}",
            f"ARD (global)        : {self.ard_flag}",
            f"Features logged     : {self.logged_flag}",
            f"Kernel type         : {self.kernel_type}",
            "",
            "Training",
            "--------",
            f"Train fraction   : {self.frac_train} (only used for array input)",
            f"Model used       : {self.model_class.__name__}",
            f"Likelihood used  : {self.likelihood_class.__name__}",
            f"Learning rate    : {self.learning_rate}",
            f"Max iterations   : {self.maxiter}",
            f"Class 1 threshold: {self.pred_threshold}",
            f"Random state     : {self.random_state}",
            "",
            "GIF / Summary",
            "-------------",
            f"Frame sampling rate : {self.gif_stride}",
            f"Max frames          : {self.gif_max_frames}",
            f"Synced mode         : {self.synced_gif}",
            f"# filters to animate: {self.topomap_filters_for_gif}",
            "",
            "(shapes are added after data load)",
        ]
        with open(self.run_dir / "config.txt", "w") as f:
            f.write("\n".join(lines))

    def _append_to_config_txt(self, extra_lines: List[str]) -> None:
        """
        Append human-readable notes to config.txt
        Multiple methods call this method at runtime
        """
        path = self.run_dir / "config.txt"
        with open(path, "a") as f:
            f.write("\n" + "\n".join(extra_lines) + "\n")

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

    # --------------- Data loading and preparation --------------- #
    def _load_and_prepare_data(self) -> None:
        """
        Handle both input modes for X and Y (arrays OR dicts)
        Train and test sets are arranged accordingly
        X in input to GPflow must also be flatten to (N, s*s) to respect the requirement shape (N, D)

        Sets:
          X_train: (N_train, D)
          X_test : (N_test, D)
          Y_train: (N_train, 1)
          Y_test : (N_test, 1)
        """

        def _to_col(Ya: np.ndarray) -> np.ndarray:
            # Reshape into (N, 1)
            return np.asarray(Ya).reshape(-1, 1)

        # Case: X,Y are dict
        if isinstance(self.X, dict) and isinstance(self.Y, dict):
            # Explicit split into train and test if they are provided as such in a dict
            Xtr = np.asarray(self.X["train"], dtype=np.float64)  # shape (N_train, s, s)
            Xte = np.asarray(self.X["test"], dtype=np.float64)  # shape (N_test, s, s)

            Ytr = _to_col(self.Y["train"]).astype(np.float64)  # shape (N_train,1)
            Yte = _to_col(self.Y["test"]).astype(np.float64)  # shape (N_test, 1)

            # Quick check on shape requirements
            assert (
                Xtr.ndim == 3 and Xtr.shape[1] == Xtr.shape[2]
            ), "X['train'] must be (N_train, s, s)"
            assert (
                Xte.ndim == 3 and Xte.shape[1] == Xte.shape[2]
            ), "X['test'] must be (N_test, s, s)"
            assert (
                Ytr.shape[0] == Xtr.shape[0]
            ), "len(Y['train']) must match len(X['train'])"
            assert (
                Yte.shape[0] == Xte.shape[0]
            ), "len(Y['test']) must match len(X['test'])"

            self.N_train, self.s, _ = Xtr.shape
            self.N_test = Xte.shape[0]
            D = self.s * self.s

            self.X_train = Xtr.reshape(
                self.N_train, D
            )  # flatten (N_train, s, s) to (N_train, s * s)
            self.X_test = Xte.reshape(
                self.N_test, D
            )  # flatten (N_test, s, s) to (N_test, s * s)
            self.Y_train = Ytr
            self.Y_test = Yte

        # Case: X,Y are arrays (train and test will be generate from this)
        else:
            X = np.asarray(self.X, dtype=np.float64)
            Y = _to_col(np.asarray(self.Y))
            assert X.ndim == 3 and X.shape[1] == X.shape[2], "X must be (N, s, s)"
            N, self.s, _ = X.shape
            D = self.s * self.s

            X_flat = X.reshape(N, D)
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
                X_flat,
                Y.astype(np.float64),
                test_size=1 - self.frac_train,
                random_state=self.random_state,
                shuffle=True,
                stratify=None,
            )

            self.N_train = self.X_train.shape[0]
            self.N_test = self.X_test.shape[0]

        # Update config.txt with derived sizes
        self._append_to_config_txt(
            [
                f"Shapes: X_train={self.X_train.shape}, Y_train={self.Y_train.shape}, "
                f"X_test={self.X_test.shape}, Y_test={self.Y_test.shape}",
            ]
        )

    def _initialize_W_matrix(self) -> None:
        """
        Initilize the spatial filter matrix W according to the provided configuration

        Sets:
          self.W_init : (s, nf)
        """
        rng = np.random.default_rng(self.random_state)  # Define random state

        # Initialize W according to flag
        if self.weights_init == "random":
            # Randomize initial coefficients using Gaussian -> N(0, 0.1)
            self.W_init = rng.normal(0.0, 0.1, size=(self.s, self.nf))

        elif self.weights_init == "ones":
            # Set all initial coefficients to 1
            self.W_init = np.ones((self.s, self.nf))

        elif self.weights_init == "manual":
            # Custom configuration for initial coefficients
            # All set to 0 except channels commonly involved in motor command following
            self.W_init = np.zeros((self.s, self.nf))

            if self.nf > 2:
                print(
                    f"Warning: More than 2 spatial filters initialized, currenlty needs fixing!"
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
            raise ValueError(f"Unknown weights_init: {self.weights_init}")

        # Update config.txt with derived sizes
        self._append_to_config_txt(
            [
                f"W_init shape: {self.W_init.shape}",
            ]
        )

    # --------------- Model Building --------------- #
    def _build_model(self) -> gpflow.models.Model:
        """
        Build a GPflow model with a CustomKernel and chosen likelihood
        """
        # Initialize the kernel using the provided CustomKernel
        kernel = CustomKernel(
            self.W_init,
            ard_flag=self.ard_flag,
            eta_flag=self.eta_flag,
            logged_flag=self.logged_flag,
            kernel_type=self.kernel_type,
        )
        self.kernel = kernel

        # Define the likelihood to use in the classifier, by default we are using Bernoulli for a binary classification
        # p(y=1 ∣ f) = σ(f)
        # p(y=0 ∣ f) = 1 − σ(f)
        # Likelihood for classification purpose needs to be specified, can't use Gaussian
        # Most likely no need for kwargs
        likelihood = self.likelihood_class(**self.likelihood_kwargs)

        # Define the model to use for the classification, by default we are using Variational Gaussian Process (VGP)
        # Try to instantiate with data (VGP, GPR, etc. usually accept it)
        Xtr = tf.convert_to_tensor(self.X_train, dtype=tf.float64)
        Ytr = tf.convert_to_tensor(self.Y_train, dtype=tf.float64)
        try:
            model = self.model_class(
                data=(Xtr, Ytr),
                kernel=kernel,
                likelihood=likelihood,
                num_latent_gps=1,  # For binary classification we need only one latent GP
                **self.model_kwargs,
            )
        except TypeError:
            # Fallback for models that don't accept `data` (e.g., SVGP)
            model = self.model_class(
                kernel=kernel,
                likelihood=likelihood,
                num_latent_gps=1,  # For binary classification we need only one latent GP
                **self.model_kwargs,
            )
            # Note: for SVGP you likely need a custom `training_loss_fn`
            # that closes over minibatches and a separate dataset.

        return model

    def _predict_prob(self, model: gpflow.models.Model, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for inputs X: returns an array of probabilities bound to [0,1]
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

    def _compute_kernel_eigs(
        self, model: gpflow.models.Model, max_n: int = 400
    ) -> Optional[np.ndarray]:
        """
        Eigenvalues of K(X, X) as a diagnostic (optional)
        """
        try:
            X = self.X_train[: min(max_n, len(self.X_train))]
            K = model.kernel.K(tf.convert_to_tensor(X, dtype=tf.float64)).numpy()
            eigs = np.linalg.eigvalsh(K)
            return eigs
        except Exception:
            return None

    def _train_and_log(self) -> None:
        """
        Manual training loop with per-iteration logging and W(t) snapshots

        If you supply `training_loss_fn`, that function will be called at each step;
        otherwise `model.training_loss()` is used (by default ELBO)

        By default the gpflow.models.VGP minimizes the negative variational evidence lower bound (ELBO)
        Defined as:
        E_{q(F)} [ log p(Y|F) ] - KL[ q(F) || p(F)]

        The addition of the `training_loss_fn` extends the freedom to generate a callable and pass it to the training process
        This accommodates (may not be limited to) the following cases:

        1) SVGP + minibatching setups

            def make_svgp_loss(dataset, batch_size):
                it = iter(tf.data.Dataset.from_tensor_slices(dataset).shuffle(10_000).batch(batch_size).repeat())
                def loss_fn(m):
                    Xb, Yb = next(it)
                    return m.training_loss((tf.cast(Xb, tf.float64), tf.cast(Yb, tf.float64)))
                return loss_fn

        2) Additional regularization / constraints

            def loss_with_weight_decay(m, l2=1e-4):
                base = m.training_loss()
                reg  = tf.add_n([tf.nn.l2_loss(v) for v in m.trainable_variables])
                return base + l2 * reg

        3) KL annealing

            def make_annealed_loss(T):
                step = tf.Variable(0, trainable=False, dtype=tf.float64)
                def loss(m):
                    # for VGP: ELBO = E[log p(y|f)] - KL; training_loss = -ELBO
                    base = m.training_loss()
                    beta = tf.minimum(1.0, step / T)  # ramp up KL weight
                    step.assign_add(1.0)
                    # If you need explicit KL, compute -ELBO pieces yourself; else approximate by scaling base
                    return (1.0 - beta) * base + beta * base
                return loss

        4) Custom objective such as penalties on calibration / ECE, margin losses, class-weighted likelihoods, ...
        """

        # Initialize kernel + likelihood + model
        self.model = self._build_model()

        # Replace SciPy optimization (which gave problems in classification) with Adam optimizer
        # Adaptive Moment Estimation (Adam) is an adaptive `learning_rate` optimization algorithm
        # Combines the benefits of two other popular optimizers: Momentum and RMSprop
        # Adaptive: calculates individual adaptive `learning_rate` for each parameter of the model
        # Momentum: accelerate convergence by accumulating past gradients and moving in the direction of consistent descent
        #           (from cconcept of physics momentum, rolling down a hill)
        # RMSprop: scales the `learning_rate` for each parameter based on the root mean square of past squared gradients
        lr_var = tf.Variable(self.learning_rate, dtype=tf.float64)
        opt = tf.optimizers.Adam(learning_rate=self.learning_rate)

        # Choose loss function (see method docstring above)
        if self.external_training_loss_fn is not None:
            loss_fn = self.external_training_loss_fn
        else:
            # Default: use the model's training loss on full batch
            # Define `loss_fn` for generality
            def loss_fn(m):  # m: gpflow.model
                return m.training_loss()

        # Initialize containers for per-iter logging
        logs: List[IterLog] = []

        # Training steps
        for step in range(1, self.maxiter + 1):
            # Perform a training step
            with tf.GradientTape() as tape:
                nlml = loss_fn(self.model)  # scalar Tensor
            grads = tape.gradient(nlml, self.model.trainable_variables)
            opt.apply_gradients(zip(grads, self.model.trainable_variables))

            # W matrix -> per-iter snapshot
            self.W_over_iters.append(self.kernel.W.numpy().copy())

            # Global scaling -> per-iter snapshot
            eta_val = (
                float(self.kernel.eta.numpy()) if self.kernel.eta is not None else None
            )

            # (ARD) Spatial filter scaling -> per-iter snapshot
            ard_val = (
                self.kernel.ard.numpy().tolist()
                if self.kernel.ard is not None
                else None
            )

            # Prediction probabilities y=1 -> per-iter snapshot
            p_train_at_iter = self._predict_prob(
                self.model, self.X_train
            )  # p(y=1 | X_train, model at that iteration)
            p_test_at_iter = self._predict_prob(
                self.model, self.X_test
            )  # p(y=1 | X_test, model at that iteration)
            self.p_train_seq.append(
                p_train_at_iter.tolist()
            )  # storing, must be list of lists
            self.p_test_seq.append(
                p_test_at_iter.tolist()
            )  # storing, must be list of lists

            # Prediction labels y=1 -> per-iter snapshot
            y_train_at_iter = (p_train_at_iter >= self.pred_threshold).astype(
                int
            )  # True -> 1, False -> 0
            y_test_at_iter = (p_test_at_iter >= self.pred_threshold).astype(
                int
            )  # True -> 1, False -> 0
            self.y_train_seq.append(
                y_train_at_iter.tolist()
            )  # storing, must be list of lists
            self.y_test_seq.append(
                y_test_at_iter.tolist()
            )  # storing, must be list of lists

            # Metrics
            acc_train = accuracy_score(
                self.Y_train.ravel(),
                y_train_at_iter,
                normalize=True,
                sample_weight=None,
            )  # fraction bound [0,1], the higher the better
            acc_test = accuracy_score(
                self.Y_test.ravel(), y_test_at_iter, normalize=True, sample_weight=None
            )  # fraction bound [0,1], the higher the better
            brier = brier_score_loss(
                self.Y_test.ravel(), p_test_at_iter
            )  # mean squared difference between the predicted probability and the actual outcome, fraction bound [0,1], the lower the better

            # Gradient norm: Euclidean distance of partial derivatives of loss w.r.t. each trainable parameter
            # if changes from large to tiny -> normal training process, potentially reaching convergence
            # if tiny from the start        -> might be over-regularized OR have poor kernel init
            # if sudden spikes              -> possibly unstable steps, bad learning rate, or numerical issues
            grad_norm = float(
                math.sqrt(
                    sum(
                        float(np.sum(np.square(g.numpy())))
                        for g in grads
                        if g is not None
                    )
                )
            )

            # Parameter norm: Euclidean distance of parameter values
            # if unstably grows                      -> risk of exploding parameters (overfitting, numerical instability, kernel variance or lengthscales are blowing up)
            # if collapses to near-zero unexpectedly -> strong regularization, rigit, potential underfitting
            param_norm = float(
                math.sqrt(
                    sum(
                        float(np.sum(np.square(v.numpy())))
                        for v in self.model.trainable_variables
                    )
                )
            )

            # Eigenvalues of the kernel
            # Mostly diagnostic on Gram property (non-negative)
            # Numerican stability for Cholesky decomposition
            eigs = self._compute_kernel_eigs(self.model) if (step % 10 == 0) else None

            # Generate per-iter logging, tracks evolution at each iteration
            logs.append(
                IterLog(
                    step=step,
                    nlml=float(nlml.numpy()),
                    acc_train=acc_train,
                    acc_test=acc_test,
                    brier=brier,
                    grad_norm=grad_norm,
                    param_norm=param_norm,
                    kernel_eigs=(eigs.tolist() if eigs is not None else None),
                    eta=eta_val,
                    ard=ard_val,
                )
            )

            # Visual update for Terminal
            if step % (self.maxiter // 5) == 0:
                print(
                    f"Iter {step:4d} / {self.maxiter} | NLML {logs[-1].nlml:.3f} | acc_train {acc_train:.3f} | acc_test {acc_test:.3f} | brier_test {brier:.3f} "
                )

        # Final probs for static plots
        self.p_train_final = self._predict_prob(self.model, self.X_train)
        self.p_test_final = self._predict_prob(self.model, self.X_test)
        # Final labels
        self.y_train_final = (self.p_train_final >= self.pred_threshold).astype(
            int
        )  # True -> 1, False -> 0
        self.y_test_final = (self.p_test_final >= self.pred_threshold).astype(
            int
        )  # True -> 1, False -> 0

        meta = {
            "run_dir": str(self.run_dir.resolve()),
            "dataset_label": self.dataset_label,
            "#channels": str(len(self.ch_names)),
            "timestamp_end": _now_stamp(),
            "weights_init": self.weights_init,
            "nf": str(self.nf),
            "eta_flag": str(self.eta_flag),
            "ard_flag": str(self.ard_flag),
            "logged_flag": str(self.logged_flag),
            "kernel_type": self.kernel_type,
            "model_class": self.model_class.__name__,
            "likelihood_class": self.likelihood_class.__name__,
            "maxiter": str(self.maxiter),
            "learning_rate": str(self.learning_rate),
            "pred_threshold": str(self.pred_threshold),
            "random_state": str(self.random_state),
        }

        self.run_log = RunLog(
            meta=meta,
            logs=logs,
            p_train_seq=self.p_train_seq,
            p_test_seq=self.p_test_seq,
            p_train_final=self.p_train_final.tolist(),
            p_test_final=self.p_test_final.tolist(),
            y_train_seq=self.y_train_seq,
            y_test_seq=self.y_test_seq,
            y_train_final=self.y_train_final.tolist(),
            y_test_final=self.y_test_final.tolist(),
        )

    def _train_and_log_early_stop(self) -> None:
        """
        Train the GP classification model with mixed optimization:
        - NaturalGradient for variational parameters (q_mu, q_sqrt) with a two-phase
        gamma schedule (warm-up, then lower/stable).
        - Adam for all remaining trainable variables (kernel + likelihood params),
        including gradient clipping and reduce-on-plateau scheduling.
        - Early stopping based on validation (here: test) metric and ELBO plateau.
        - Best checkpoint restore to ensure we keep the best validation performance.

        The function preserves the existing logging protocol:
        - Per-iteration snapshots of probabilities, labels, and kernel weights.
        - Metrics (accuracy, Brier score, norms) and optional eigen-diagnostics.
        - RunLog structure and metadata remain unchanged downstream.

        Why this design:
        - Natural gradients are geometry-aware for Gaussian variational families
        and typically stabilize / accelerate ELBO ascent for classification.
        - Adam remains a strong choice for hyperparameters that do not live on
        distribution manifolds (kernel/likelihood).
        - A two-phase gamma schedule lets the variational posterior adapt quickly
        early on (higher gamma), then settle stably (lower gamma) as hyperparams
        fine-tune, reducing oscillation.
        """

        # -------------------------------------------------------------------------
        # 1) Model construction (unchanged API).
        #    Keep the rest of the pipeline agnostic to which model is built here.
        # -------------------------------------------------------------------------
        self.model = self._build_model()

        # -------------------------------------------------------------------------
        # 2) Optimizers
        #    - Adam: Keras variant; learning_rate must be a Python float to avoid
        #      type errors. We will adjust it in-place via assign during training.
        #    - NaturalGradient: created once, but we will change `gamma` on-the-fly
        #      each iteration to implement a two-phase schedule.
        # -------------------------------------------------------------------------
        opt = tf.keras.optimizers.Adam(learning_rate=float(self.learning_rate))

        # Detect whether the model actually exposes variational params.
        use_natgrad = hasattr(self.model, "q_mu") and hasattr(self.model, "q_sqrt")
        natgrad = gpflow.optimizers.NaturalGradient(gamma=0.1) if use_natgrad else None

        self.lr_seq = []
        self.gamma_seq = []
        self.ng_norm_seq = []

        # -------------------------------------------------------------------------
        # 3) Loss function
        #    - Use external loss if provided; otherwise default to training_loss().
        #    - Note: training_loss() in GPflow returns the negative ELBO (NLML-ish).
        # -------------------------------------------------------------------------
        if self.external_training_loss_fn is not None:
            loss_fn = self.external_training_loss_fn
        else:

            def loss_fn(m):
                return m.training_loss()

        # -------------------------------------------------------------------------
        # 4) Training configuration and state
        #    - Early stopping on validation metric (test accuracy here) with patience.
        #    - ELBO EMA plateau detection to avoid stopping at transient improvements.
        #    - Gradient clipping to tame occasional large updates from Adam.
        #    - LR reduce-on-plateau when validation metric stalls for a while.
        # -------------------------------------------------------------------------
        patience = 200  # stop if no val-acc improvement for this many steps
        min_delta = 1e-4  # minimal val-acc improvement to reset patience
        ema_beta = 0.98  # ELBO EMA smoothing coefficient
        plateau_tol = 1e-4  # relative ELBO EMA change threshold for "flat"
        plateau_required = 50  # number of consecutive "flat" steps to consider plateau
        clip_norm = 5.0  # global-norm gradient clipping
        lr_decay_factor = 0.5  # multiplicative LR decay on plateau
        lr_min = 1e-5  # floor on learning rate
        lr_plateau_patience = 150  # how long to wait (no val-acc gains) before LR decay

        # ---------------- Two-phase NaturalGradient gamma schedule -----------------
        # Rationale:
        # - Early iterations benefit from a larger gamma to quickly shape q(f).
        # - Later, a smaller gamma reduces oscillation as hyperparameters tune.
        gamma_ramp = 0.5  # higher gamma in warm-up for faster q updates
        gamma_main = 0.1  # stable gamma afterward

        ng_early_steps = int(self.maxiter * 0.01)  # First 1% of iteration steps
        ng_ramp_steps = int(self.maxiter * 0.1)  # First 10% of iteration steps

        # Best-checkpoint bookkeeping
        best_val = -np.inf  # best validation (test) accuracy observed
        self.best_step = 0  # iteration at which best_val occurred
        steps_since_best = 0  # counter since last best improvement
        steps_plateau = 0  # counter for consecutive ELBO-EMA "flat" steps
        ema_nlml_prev = None  # previous EMA value of negative ELBO
        best_vars = None  # snapshot of best model weights

        # Logs container (unchanged type)
        logs: List[IterLog] = []

        # Utility: snapshot/restore model weights for best-checkpointing ----------
        def _snapshot_weights():
            return [v.numpy().copy() for v in self.model.trainable_variables]

        def _restore_weights(snap):
            for v, arr in zip(self.model.trainable_variables, snap):
                v.assign(arr)

        # Progress print cadence (unchanged behavior)
        print_every_n_iter = max(1, self.maxiter // 10)

        # -------------------------------------------------------------------------
        # 5) Training loop
        #    Each iteration performs up to two sub-steps:
        #    (a) NaturalGradient update for (q_mu, q_sqrt) if present.
        #    (b) Adam update for all trainable variables with gradient clipping.
        #    Then we compute metrics, update logs, manage early stopping, LR schedule
        # -------------------------------------------------------------------------
        for step in range(1, self.maxiter + 1):
            # ------------------------- (a) NaturalGradient ------------------------
            # Only update variational parameters with NatGrad; set gamma per schedule
            if use_natgrad:
                # Select gamma according to the multi-phase schedule
                if step <= ng_early_steps:
                    natgrad.gamma = gamma_main * step / ng_early_steps  # Step function

                elif step > ng_early_steps and step <= ng_ramp_steps:
                    natgrad.gamma = gamma_ramp  # Higer value for faster convergence

                else:
                    natgrad.gamma = gamma_main  # Finish value for gamma

                # Perform the NG step on the variational parameters only.
                # This uses the Fisher geometry of the Gaussian family for stable,
                # scale-aware updates to q(f).
                natgrad.minimize(
                    lambda: loss_fn(self.model),
                    var_list=[(self.model.q_mu, self.model.q_sqrt)],
                )

            # ------------------------------ (b) Adam ------------------------------
            # Adam update for all trainable variables. We compute gradients of the
            # negative ELBO w.r.t. all variables and apply them with gradient clipping.
            with tf.GradientTape() as tape:
                nlml = loss_fn(self.model)  # negative ELBO (lower is better)
            grads = tape.gradient(nlml, self.model.trainable_variables)

            # Filter out None gradients (can happen for constrained/unused vars).
            gv = [
                (g, v)
                for g, v in zip(grads, self.model.trainable_variables)
                if g is not None
            ]
            if gv:
                # Global-norm clipping guards against occasional large steps that
                # could destabilize the variational optimization.
                gs, vs = zip(*gv)
                gs, _ = tf.clip_by_global_norm(gs, clip_norm)
                opt.apply_gradients(zip(gs, vs))

            # ------------------------ 6) Per-iter snapshots -----------------------
            # Track optimizer state per iteration for diagnostics
            if use_natgrad:
                self.gamma_seq.append(float(natgrad.gamma))
                # Compute NG update norm (Euclidean norm of Δq_mu and Δq_sqrt)
                ng_norm = 0.0
                ng_norm += np.sum(np.square(self.model.q_mu.numpy()))
                ng_norm += np.sum(np.square(self.model.q_sqrt.numpy()))
                self.ng_norm_seq.append(float(np.sqrt(ng_norm)))
            else:
                self.gamma_seq.append(None)
                self.ng_norm_seq.append(None)

            self.lr_seq.append(float(opt.learning_rate.numpy()))

            # Preserve your existing time-series logging of kernel weights, per-iter
            # probabilities/labels; downstream plotting relies on these arrays.
            self.W_over_iters.append(self.kernel.W.numpy().copy())

            eta_val = (
                float(self.kernel.eta.numpy()) if self.kernel.eta is not None else None
            )
            ard_val = (
                self.kernel.ard.numpy().tolist()
                if self.kernel.ard is not None
                else None
            )

            # Predicted probabilities at this iteration (train/test).
            p_train_at_iter = self._predict_prob(self.model, self.X_train)
            p_test_at_iter = self._predict_prob(self.model, self.X_test)
            self.p_train_seq.append(p_train_at_iter.tolist())
            self.p_test_seq.append(p_test_at_iter.tolist())

            # Thresholded labels (consistent with your pred_threshold).
            y_train_at_iter = (p_train_at_iter >= self.pred_threshold).astype(int)
            y_test_at_iter = (p_test_at_iter >= self.pred_threshold).astype(int)
            self.y_train_seq.append(y_train_at_iter.tolist())
            self.y_test_seq.append(y_test_at_iter.tolist())

            # --------------------------- 7) Metrics -------------------------------
            # Train/test accuracy, Brier on test; norms for diagnostics; optional
            # eigen-diagnostics at a reduced cadence to control runtime.
            acc_train = accuracy_score(
                self.Y_train.ravel(), y_train_at_iter, normalize=True
            )
            acc_test = accuracy_score(
                self.Y_test.ravel(), y_test_at_iter, normalize=True
            )
            brier = brier_score_loss(self.Y_test.ravel(), p_test_at_iter)

            # grad/param norms for diagnostics (safe if some grads None)
            try:
                grad_norm = float(
                    math.sqrt(sum(float(np.sum(np.square(g.numpy()))) for g, _ in gv))
                )
            except Exception:
                grad_norm = float("nan")

            param_norm = float(
                math.sqrt(
                    sum(
                        float(np.sum(np.square(v.numpy())))
                        for v in self.model.trainable_variables
                    )
                )
            )

            eigs = self._compute_kernel_eigs(self.model) if (step % 10 == 0) else None

            logs.append(
                IterLog(
                    step=step,
                    nlml=float(nlml.numpy()),
                    acc_train=acc_train,
                    acc_test=acc_test,  # used as "validation" proxy here
                    brier=brier,
                    grad_norm=grad_norm,
                    param_norm=param_norm,
                    kernel_eigs=(eigs.tolist() if eigs is not None else None),
                    eta=eta_val,
                    ard=ard_val,
                )
            )

            if step % print_every_n_iter == 0 or step == 1:
                print(
                    f"({int(step * 100 / self.maxiter)} %) Iter {step:4d} / {self.maxiter} | "
                    f"NLML {logs[-1].nlml:.3f} | "
                    f"acc_train {acc_train:.3f} | acc_test {acc_test:.3f} | "
                    f"brier_test {brier:.3f}"
                )

            # --------------------- 8) Early-stopping logic ------------------------
            # (1) Track best validation accuracy and checkpoint weights
            if acc_test > best_val + min_delta:
                best_val = acc_test
                self.best_step = step
                steps_since_best = 0
                best_vars = _snapshot_weights()
            else:
                steps_since_best += 1

            # (2) ELBO EMA plateau detection: prevents stopping on transient noise
            nlml_now = float(nlml.numpy())
            if ema_nlml_prev is None:
                ema_nlml_prev = nlml_now
            ema_nlml = ema_beta * ema_nlml_prev + (1.0 - ema_beta) * nlml_now
            rel_change = abs(ema_nlml - ema_nlml_prev) / (abs(ema_nlml_prev) + 1e-12)
            steps_plateau = steps_plateau + 1 if rel_change < plateau_tol else 0
            ema_nlml_prev = ema_nlml

            # (3) Reduce Adam LR if validation is flat for a while.
            if steps_since_best > 0 and steps_since_best % lr_plateau_patience == 0:
                current_lr = float(opt.learning_rate.numpy())
                new_lr = max(lr_min, current_lr * lr_decay_factor)
                if new_lr < current_lr:
                    # Keep dtype consistent with optimizer policy.
                    opt.learning_rate.assign(
                        tf.constant(new_lr, dtype=opt.learning_rate.dtype)
                    )
                    print(f"    [lr] reduced to {new_lr:.6f} at iter {step}")

            # (4) Stop if BOTH (a) no val improvement for 'patience' and
            #     (b) ELBO EMA has been flat for 'plateau_required' steps
            if steps_since_best >= patience and steps_plateau >= plateau_required:
                print(
                    f"  * [Early stop] iter {step}: "
                    f"No improvement for {patience} steps + ELBO plateau"
                )
                self.last_step = step
                break

        # -------------------------------------------------------------------------
        # 9) Restore the best checkpoint (ensures we keep the best validation model).
        # -------------------------------------------------------------------------
        if best_vars is not None:
            _restore_weights(best_vars)
            print(
                f"  * [Restore] best val acc {best_val:.3f} at iter {self.best_step}; restored weights"
            )

        # -------------------------------------------------------------------------
        # 10) Final predictions for static plots (unchanged behavior).
        # -------------------------------------------------------------------------
        self.p_train_final = self._predict_prob(self.model, self.X_train)
        self.p_test_final = self._predict_prob(self.model, self.X_test)
        self.y_train_final = (self.p_train_final >= self.pred_threshold).astype(int)
        self.y_test_final = (self.p_test_final >= self.pred_threshold).astype(int)

        # -------------------------------------------------------------------------
        # 11) RunLog assembly (structure unchanged for downstream consumers).
        # -------------------------------------------------------------------------
        meta = {
            "run_dir": str(self.run_dir.resolve()),
            "dataset_label": self.dataset_label,
            "#channels": str(len(self.ch_names)),
            "timestamp_end": _now_stamp(),
            "weights_init": self.weights_init,
            "nf": str(self.nf),
            "eta_flag": str(self.eta_flag),
            "ard_flag": str(self.ard_flag),
            "logged_flag": str(self.logged_flag),
            "kernel_type": self.kernel_type,
            "model_class": self.model_class.__name__,
            "likelihood_class": self.likelihood_class.__name__,
            "maxiter": str(self.maxiter),
            "learning_rate": str(self.learning_rate),
            "gamma_main": gamma_main,
            "gamma_ramp": gamma_ramp,
            "pred_threshold": str(self.pred_threshold),
            "random_state": str(self.random_state),
            "last_step": self.last_step if hasattr(self, "last_step") else self.maxiter,
            "best_step": self.best_step,
        }

        self.run_log = RunLog(
            meta=meta,
            logs=logs,
            p_train_seq=[list(x) for x in self.p_train_seq],
            p_test_seq=[list(x) for x in self.p_test_seq],
            p_train_final=self.p_train_final.tolist(),
            p_test_final=self.p_test_final.tolist(),
            y_train_seq=[list(x) for x in self.y_train_seq],
            y_test_seq=[list(x) for x in self.y_test_seq],
            y_train_final=self.y_train_final.tolist(),
            y_test_final=self.y_test_final.tolist(),
        )

    # --------------- Visual Outputs --------------- #
    def _make_visual_summary(self) -> None:
        """
        Generate visual outputs as summary of the training process (PNGs + GIFs)
        """
        # Manually check self.run_log has been generated
        # AssertionError is raised if a method relying on run_log is called and this is None
        assert self.run_log is not None
        # Proceed with generating the plots
        self._plot_learning_curves()  # static, all iterations
        self._plot_roc_pr()  # static, last iteration
        self._threshold_sweep()  # static, changing values of `pred_threshold` for different metrics
        self._plot_kernel_parameters()  # static, parameters evolution
        self._plot_calibration(
            n_bins=10
        )  # static, calibration curve on final iteration predicted on test
        self._plot_optimizer_diagnostics()  # learning rate, gamma, grad-norm

        # Identify which iterations to use for the GIFs
        self.iters_gif = self._iter_indices_for_gifs(
            len(self.run_log.logs)
        )  # generate indices for GIF frames

        self._animate_confusions(which="train")  # static, last iteration & GIF
        self._animate_confusions(which="test")  # static, last iteration & GIF

        self._compute_features_over_iterations()  # generate feature evolutions for train and test
        self._compute_boundary()  # generate bounday decision mesh for GIF frames

        self._animate_features()  # static, last iteration & GIF
        self._animate_weights_topomaps()  # GIF
        self._animate_synced_dashboard()

        # self._animate_threshold_sweep()
        # self._plot_uncertainty_vs_error()
        # self._plot_risk_coverage()
        # self._plot_kernel_spectrum_snapshots()
        # self._plot_kernel_component_trajectories()

    def _iter_indices_for_gifs(self, total_num_logs: int) -> np.ndarray:
        """
        Compute iteration indices for GIF frames taking into account stride/frame-cap
        Returns: (M,) array of iteration indices in [1..total_num_logs]
        """
        # Determine indices of iterations to consider for GIF creation
        idx = np.arange(1, total_num_logs + 1, dtype=int)[:: self.gif_stride]
        # If `gif_max_frames` is provided and is currently not satisfied -> Initial `gif_stride` was underestimated
        # Reevaluate the indices with a new `gif_stride`
        if self.gif_max_frames is not None and len(idx) > self.gif_max_frames:
            stride = math.ceil(total_num_logs / self.gif_max_frames)
            idx = np.arange(1, total_num_logs + 1, stride, dtype=int)
        # Make sure the last iteration is included
        if total_num_logs not in idx:
            idx = np.append(idx, total_num_logs)
        return idx

    def _fig_to_frame(self, fig) -> np.ndarray:
        """
        Convert a Matplotlib figure to a uint8 (H,W,3) RGB frame **in memory**
        Avoids temp files (fixes Windows/Dropbox PermissionError)
        """
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
        buf = buf[:, :, [1, 2, 3, 0]]  # ARGB->RGBA
        frame = buf[:, :, :3].copy()  # drop alpha
        return frame

    # --- 1) Learning curves --- #
    def _plot_learning_curves(self) -> None:
        """
        Plot NLML (or approximation) alongside accuracy scores for train and test sets
        """
        # Grab information stored into the `run_log`
        steps = [l.step for l in self.run_log.logs]
        nlml = [l.nlml for l in self.run_log.logs]
        acc_train = [l.acc_train for l in self.run_log.logs]
        acc_test = [l.acc_test for l in self.run_log.logs]

        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        # Plot NLML (neg-ELBO) which is minimized in training
        ax1.plot(steps, nlml, color="black", label="NLML", linewidth=2)
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Neg-ELBO (NLML approximation)")
        ax1.set_xlim(0, self.maxiter)
        # Generate new axis to plot accuracy scores
        ax2 = ax1.twinx()
        ax2.plot(
            steps, acc_train, color="blue", label="Accuracy (Train)", linestyle="--"
        )
        ax2.plot(
            steps, acc_test, color="orange", label="Accuracy (Test)", linestyle=":"
        )
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Accuracy score")
        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [l.get_label() for l in lines], loc="best")
        fig.tight_layout()
        fig.savefig(self.run_dir / "01_learning_curves.png", dpi=150)
        plt.close(fig)

    # --- 2) ROC / Precision-Recall curves --- #
    def _plot_roc_pr(self) -> None:
        """
        Generate ROC and Precision-Recall curves with probabilities from last iteration
        """
        y = self.Y_test.ravel()
        p = np.array(self.run_log.p_test_final)
        # fpr: Out of all the actual negatives, how many did we incorrectly classify as positive?
        # tpr: Out of all the actual positives, how many did we correctly identify as positive?
        fpr, tpr, _ = roc_curve(y, p)
        # prec: Out of all the predicted positives, how many are actually positive?
        # rec: Out of all the actual positives, how many did we catch? This is the same as tpr
        prec, rec, _ = precision_recall_curve(y, p)
        roc_auc = auc(fpr, tpr)
        ap = average_precision_score(y, p)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.2))
        ax1.plot([0, 1], [0, 1], linestyle=":", linewidth=1)
        ax1.plot(fpr, tpr, linewidth=2, label=f"AUC={roc_auc:.3f}")
        ax1.set_title("ROC curve")
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.legend(loc="lower right")
        ax2.plot(rec, prec, linewidth=2, label=f"AP={ap:.3f}")
        ax2.set_title("PR curve")
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(self.run_dir / "02_roc_pr_curves.png", dpi=150)
        plt.close(fig)

    # --- 3) Threshold sweep curves --- #
    def _threshold_sweep(self, verbose=False) -> None:
        """
        Generate multiple curves over test probabilities, plotting:
        - Accuracy, Precision, Recall, F1, Specificity, Youden's J (TPR - FPR)
        - Constant ROC-AUC and PR-AUC
        This plot should inform of best location for self.pred_threshold based on predicted probabilities using last iteration model
        """
        y = self.Y_test.ravel().astype(int)
        # Use final test probabilities from this run (last iteration)
        p = np.array(self.run_log.p_test_final)

        # Threshold grid, 51 points means delta_threshold = 0.02
        thr_seq = np.linspace(0.0, 1.0, 51)

        # Fixed, threshold-independent metrics
        fpr_curve, tpr_curve, _ = roc_curve(y, p)
        roc_auc = auc(fpr_curve, tpr_curve)
        prec_curve, rec_curve, _ = precision_recall_curve(y, p)
        pr_auc = average_precision_score(y, p)  # AP (PR-AUC)

        # Helper to compute per-threshold metrics
        def _metrics_at(th):
            yhat = (p >= th).astype(int)
            TP = np.sum((yhat == 1) & (y == 1))
            FP = np.sum((yhat == 1) & (y == 0))
            TN = np.sum((yhat == 0) & (y == 0))
            FN = np.sum((yhat == 0) & (y == 1))
            N = len(y)

            acc = (TP + TN) / N if N else np.nan
            precision = TP / (TP + FP) if (TP + FP) > 0 else np.nan
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # TPR
            f1 = (
                (2 * precision * recall / (precision + recall))
                if (not np.isnan(precision) and (precision + recall) > 0)
                else np.nan
            )
            specificity = TN / (TN + FP) if (TN + FP) > 0 else np.nan
            fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
            youden = recall - fpr  # TPR - FPR
            return acc, precision, recall, f1, specificity, youden

        # Compute all curves
        accs, precs, recs, f1s, specs, youdens = [], [], [], [], [], []
        for th in thr_seq:
            a, pr, rc, f1, sp, jd = _metrics_at(th)
            accs.append(a)
            precs.append(pr)
            recs.append(rc)
            f1s.append(f1)
            specs.append(sp)
            youdens.append(jd)

        # Helper to find best threshold index (maximize), handling NaNs
        def _best_idx(vals):
            arr = np.asarray(vals, dtype=float)
            scores = np.where(np.isnan(arr), -np.inf, arr)
            best_val = np.max(scores)
            # choose smallest threshold achieving the max
            idxs = np.where(scores == best_val)[0]
            return int(idxs[0]) if idxs.size else 0

        best = {
            "Accuracy": _best_idx(accs),
            "Precision": _best_idx(precs),
            "Recall (TPR)": _best_idx(recs),
            "F1 score": _best_idx(f1s),
            "Specificity": _best_idx(specs),
            "Youden's J (TPR - FPR)": _best_idx(youdens),
        }

        # Plot
        fig, axes = plt.subplots(3, 2, figsize=(10.5, 9.0), sharex=True)
        axes = axes.ravel()
        curves = [
            ("Accuracy", accs, (0.0, 1.0)),
            ("Precision", precs, (0.0, 1.0)),
            ("Recall (TPR)", recs, (0.0, 1.0)),
            ("F1 score", f1s, (0.0, 1.0)),
            ("Specificity", specs, (0.0, 1.0)),
            ("Youden's J (TPR - FPR)", youdens, (-1.0, 1.0)),
        ]

        for ax, (label, vals, ylim) in zip(axes, curves):
            yvals = np.asarray(vals, dtype=float)
            ax.plot(thr_seq, yvals, linewidth=2)
            j = best[label]
            th_star = thr_seq[j]
            val_star = yvals[j]
            # vertical marker + orange dot
            ax.axvline(th_star, color="#ff7f0e", lw=2, alpha=0.9)
            if not np.isnan(val_star):
                ax.plot(th_star, val_star, "o", color="#ff7f0e", ms=7)
                ax.text(
                    th_star,
                    1.02,
                    f"t*={th_star:.2f}\n{val_star:.3f}",
                    color="#ff7f0e",
                    ha="center",
                    va="bottom",
                    transform=ax.get_xaxis_transform(),
                    fontsize=9,
                )
            ax.set_ylabel(label)
            ax.set_ylim(*ylim)
            ax.grid(alpha=0.25)

        axes[-2].set_xlabel("Threshold")
        axes[-1].set_xlabel("Threshold")

        # Title with constant AUCs
        fig.suptitle(
            f"Threshold Sweep — Test set   |   ROC-AUC={roc_auc:.3f}   PR-AUC={pr_auc:.3f}",
            y=0.985,
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(self.run_dir / "03_threshold_sweep.png", dpi=150)
        plt.close(fig)

        # Optional: print best thresholds to console for quick copy/paste
        if verbose:
            print("[threshold_sweep] Best thresholds:")
            for k, j in best.items():
                print(f"  {k:<22s}: t*={thr_seq[j]:.3f}")

    # --- 4) Global and sptial filter scaling evolution --- #
    def _plot_kernel_parameters(self) -> None:
        """
        Plot the evolution over the iterations of the kernel parameters
        This only happens for the flags that have been set as True
        """
        steps = [l.step for l in self.run_log.logs]

        if self.eta_flag or self.ard_flag:
            fig, ax = plt.subplots(figsize=(6, 4))

            if self.eta_flag:
                eta_vals = [
                    l.eta if l.eta is not None else np.nan for l in self.run_log.logs
                ]
                if not np.all(np.isnan(eta_vals)):
                    ax.plot(steps, eta_vals, color="black", label=r"$\eta$")

            if self.ard_flag:
                for k in range(self.nf):
                    vals_k = [
                        (l.ard[k] if (l.ard is not None and len(l.ard) > k) else np.nan)
                        for l in self.run_log.logs
                    ]
                    ax.plot(steps, vals_k, label=f"ard[{k}]")

            ax.set_xlabel("Iteration")
            ax.set_ylabel("Parameter value")
            ax.set_xlim(0, self.maxiter)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(ncol=2, fontsize=8)
            fig.tight_layout()
            fig.savefig(self.run_dir / "4_kernel_parameters.png", dpi=150)
            plt.close(fig)

        fig, ax = plt.subplots(1, self.nf, figsize=(int(5 * self.nf), 4))
        for k in range(self.nf):
            ax[k].plot(steps, np.array(self.W_over_iters)[:, :, k])
            ax[k].set_xlabel("Iteration")
            ax[k].set_ylabel(f"W[:,{k}]")
            ax[k].set_xlim(0, self.maxiter)
        fig.tight_layout()
        fig.savefig(self.run_dir / "4_kernel_W.png", dpi=150)
        plt.close(fig)

    # --- 5) Calibration curve --- #
    def _plot_calibration(self, n_bins: int = 10) -> None:
        """
        Generate a calibration curve with initial equal-width bins in [0, 1]
        Any bin with fewer than 3 points is merged into the adjacent bin that has
        fewer points (ties merge left)
        """
        # Ground truth and predicted probabilities
        y_true = self.Y_test.ravel()  # used for y values ->
        p = np.asarray(
            self.run_log.p_test_final, dtype=float
        )  # used for x values -> mean in a bin

        # If too few test points, don't compute the plot
        if p.size < 3:
            print(f"Not enough test points for calibration curve: {self.N_test}")
            return

        # Start with `n_bins` equal-width bins in [0, 1]
        initial_bins = int(n_bins)
        edges = list(np.linspace(0.0, 1.0, initial_bins + 1))

        # Assign each point to a bin index in [0, initial_bins-1]
        # Since we provide edges in increasing order and `right==False` -> edges[i-1] <= x < edges[i]
        idx = (
            np.digitize(p, edges, right=False) - 1
        )  # Needs to shift by 1 to start index at 0

        # Distribute p among the bins to see bin population
        bin_indices = [np.where(idx == b)[0].tolist() for b in range(int(n_bins))]

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

        # Plot calibration curve
        fig, ax = plt.subplots(figsize=(4.8, 4.8))
        ax.plot(
            [0, 1], [0, 1], linestyle=":", linewidth=1, label="Perfectly calibrated"
        )
        ax.plot(mean_pred, frac_pos, marker="o", label="Model (test)")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Empirical probability")

        # Add scores
        brier = brier_score_loss(y_true, p)
        ax.set_title(f"Calibration curve\nBrier={brier:.3f}")
        ax.legend(loc="best")

        fig.tight_layout()
        fig.savefig(self.run_dir / "05_calibration_curve.png", dpi=150)
        plt.close(fig)

    # --- 6) Training low level parameters --- #
    def _plot_optimizer_diagnostics(self) -> None:
        """
        Plot learning rate (Adam), natural gradient gamma, and NG norm vs iterations
        """
        iters = range(1, len(self.run_log.logs) + 1)

        fig, ax1 = plt.subplots(figsize=(6, 4))

        # Plot learning rate (left y-axis)
        ax1.plot(iters, self.lr_seq, label="Adam LR", color="tab:blue")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Learning Rate", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        # Create a second axis for gamma
        ax2 = ax1.twinx()
        ax2.plot(iters, self.gamma_seq, label="NatGrad gamma", color="tab:orange")
        ax2.set_ylabel("Gamma", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        # Add a third axis for NG norm
        """
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 60))  # offset for clarity
        ax3.plot(iters, self.ng_norm_seq, label="NatGrad norm", color="tab:green")
        ax3.set_ylabel("NG Norm", color="tab:green")
        ax3.tick_params(axis="y", labelcolor="tab:green")
        """
        # Combine legends from all axes
        lines, labels = [], []
        # for ax in [ax1, ax2, ax3]:
        for ax in [ax1, ax2]:
            l, lab = ax.get_legend_handles_labels()
            lines.extend(l)
            labels.extend(lab)
        ax1.legend(lines, labels, loc="upper right")

        fig.tight_layout()
        fig.savefig(self.run_dir / "06_training_parameters.png", dpi=150)
        plt.close(fig)

    # --- 11) Confusion matrix evolution --- #
    def _animate_confusions(self, which: str) -> None:
        """
        Generate confusion matrices for train and test using the last iteration
        Generate a GIF with confusion matrix evolution for either train or test set
        """
        # Determine if we are using train or test set
        if which == "train":
            y_true = self.Y_train.ravel()
        else:
            y_true = self.Y_test.ravel()

        # Generate the static plot
        # grab predicted labels at a given iteration
        if which == "train":
            y_pred = self.run_log.y_train_seq[-1]
        else:
            y_pred = self.run_log.y_test_seq[-1]
        cm = confusion_matrix(y_true, np.array(y_pred))
        vmax = cm.max()

        fig, ax = plt.subplots(figsize=(4.8, 4.8))
        cm = confusion_matrix(y_true, np.array(y_pred))
        ax.imshow(cm, cmap="Greens", vmin=0, vmax=vmax)
        ax.set_title(
            f"{which.capitalize()} Confusion (iter {self.last_step if hasattr(self, "last_step") else self.maxiter})"
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, int(v), ha="center", va="center")
        fig.tight_layout()
        fig.savefig(self.run_dir / f"11_confusion_matrix_{which}.png", dpi=150)
        plt.close(fig)

        # Generate the GIF
        # Verify library requirement
        if not HAS_IMAGEIO or not self.gif_flag:
            return

        frames = []
        vmax = None
        # First consider all selected iterations to understand vmax for the color
        for t in self.iters_gif:
            # grab predicted labels at a given iteration
            if which == "train":
                y_pred = self.run_log.y_train_seq[t - 1]
            else:
                y_pred = self.run_log.y_test_seq[t - 1]
            cm = confusion_matrix(y_true, np.array(y_pred))
            vmax = (
                cm.max() if vmax is None else max(vmax, cm.max())
            )  # capture max values for future colorbar

        # Now repeat to generate the frames and the GIF with proper colors
        for t in self.iters_gif:
            # grab predicted labels at a given iteration
            if which == "train":
                y_pred = self.run_log.y_train_seq[t - 1]
            else:
                y_pred = self.run_log.y_test_seq[t - 1]

            fig, ax = plt.subplots(figsize=(4.8, 4.8))
            cm = confusion_matrix(y_true, np.array(y_pred))
            ax.imshow(cm, cmap="Greens", vmin=0, vmax=vmax)
            ax.set_title(f"{which.capitalize()} Confusion (iter {t})")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, int(v), ha="center", va="center")
            fig.tight_layout()
            frames.append(self._fig_to_frame(fig))
            plt.close(fig)
        imageio.mimsave(
            self.run_dir / f"11_confusion_evolution_{which}.gif", frames, duration=0.6
        )

    # --- 12) Feature evolution / Boundary --- #
    def _compute_features(self, Sigma: np.ndarray, W: np.ndarray) -> tf.Tensor:
        """
        Compute features at a given state of the parameters W
        This function doesn't take into account global scaling
        """
        Sw = tf.matmul(Sigma, W)  # [N, s, nf]
        # Applies w @ Σi @ w for each i with i being trial number
        # Sw has shape [N, s, nf]
        # W[None, :, :] has shape [1, s, nf]
        wSw = tf.reduce_sum(W[None, :, :] * Sw, axis=1)  # [N, nf]
        if self.logged_flag:
            # Log the resulting features
            wSw = tf.math.log(wSw)
        if self.ard_flag:
            # Scale each spatial filter independently
            ard = tf.exp(self.kernel.ard)  # positive scale per feature
            wSw = wSw * ard  # broadcast over rows
        return wSw.numpy()

    def _compute_features_over_iterations(self) -> None:
        """
        Compute the feature evolution over the iterations
        This is used later in multiple plots
        """
        Sigma_train = self.X_train.reshape(
            self.N_train, self.s, self.s
        )  # train cov matrices

        Sigma_test = self.X_test.reshape(
            self.N_test, self.s, self.s
        )  # test cov matrices
        # Compute features at each iteration
        # Features may be logged based on `self.logged_flag`
        # Features may be scaled per spatial filter based on `self.ard_flag`
        self.feats_over_iterations_train = []
        for W_t in self.W_over_iters:
            self.feats_over_iterations_train.append(
                self._compute_features(Sigma_train, W_t)
            )
        self.feats_over_iterations_train = np.stack(
            self.feats_over_iterations_train
        )  # shape [maxiter, N_train, nf]

        self.feats_over_iterations_test = []
        for W_t in self.W_over_iters:
            self.feats_over_iterations_test.append(
                self._compute_features(Sigma_test, W_t)
            )
        self.feats_over_iterations_test = np.stack(
            self.feats_over_iterations_test
        )  # shape [maxiter, N_test, nf]

    def _compute_boundary(self) -> None:
        """
        Generate boundary decisions using interpolation from predicted probabilties on the train set
        This is a temporary solution while the kernel/model doesn't accept feature space coordinates
        """
        # Determine grid for boundary decision
        f1_min, f1_max = np.percentile(
            self.feats_over_iterations_train[:, :, 0], [1, 99]
        )
        f2_min, f2_max = np.percentile(
            self.feats_over_iterations_train[:, :, 1], [1, 99]
        )
        pad1 = 0.05 * (f1_max - f1_min + 1e-12)
        pad2 = 0.05 * (f2_max - f2_min + 1e-12)
        x_lin = np.linspace(f1_min - pad1, f1_max + pad1, 300)
        y_lin = np.linspace(f2_min - pad2, f2_max + pad2, 300)
        self.XX, self.YY = np.meshgrid(x_lin, y_lin)
        # Interpolate predicted probabilties to assign them to the grid
        self.ZZ_over_iterations = []
        for t in self.iters_gif:
            # Grab features at given iteration t
            feats = self.feats_over_iterations_train[t - 1]  # last iteration
            f1 = feats[:, 0]
            f2 = feats[:, 1]
            # GP probabilities for these exact N samples at given iteration
            p = np.asarray(self.p_train_seq[t - 1])
            # Interpolate p onto grid for a smooth boundary
            ZZ = griddata(
                points=np.c_[f1, f2],
                values=p,
                xi=(self.XX, self.YY),
                method="cubic",
            )
            # Fill holes with nearest
            mask = np.isnan(ZZ)
            if np.any(mask):
                ZZ_near = griddata(
                    points=np.c_[f1, f2],
                    values=p,
                    xi=(self.XX[mask], self.YY[mask]),
                    method="nearest",
                )
                ZZ[mask] = ZZ_near
            self.ZZ_over_iterations.append(ZZ)

    def _plot_boundary(self, levels: List[float], iter: int, ax: plt.Axes) -> None:
        """
        Plot decision boundary contours at a given iteration `iter`
        Plot the first level as the main in black, plot additional levels in grey
        """
        # Main decision boundary is black, all others are grey
        colors = ["grey"] * len(levels)
        idx_main = levels.index(self.pred_threshold)
        colors[idx_main] = "black"

        cs_main = ax.contour(
            self.XX,
            self.YY,
            self.ZZ_over_iterations[iter],
            levels=levels,
            colors=colors,
            linewidths=1.0,
        )
        ax.clabel(
            cs_main,
            cs_main.levels,
            inline=True,
            fmt=lambda v: f"{v:.1f}",
            fontsize=9,
        )

    def _plot_features_at_current_iter(self, iter: int, ax: plt.Axes) -> None:
        """
        Plot features at a given iteration `iter`
        """
        feats_train = self.feats_over_iterations_train[iter]  # last iteration
        f1_train = feats_train[:, 0]
        f2_train = feats_train[:, 1]

        feats_test = self.feats_over_iterations_test[iter]  # last iteration
        f1_test = feats_test[:, 0]
        f2_test = feats_test[:, 1]

        ax.scatter(
            f1_train,
            f2_train,
            c=["orange" if y == 0 else "navy" for y in self.Y_train.flatten()],
            s=36,
            marker="o",
            linewidth=0.4,
            alpha=0.3,
        )
        ax.scatter(
            f1_test,
            f2_test,
            c=["orange" if y == 0 else "navy" for y in self.Y_test.flatten()],
            s=18,
            marker="o",
            linewidth=0.4,
            alpha=1,
        )

    def _animate_features(self) -> None:
        """
        Generate features scatterplot using last iteration, show decision boundary
        Generate a GIF with features and decision boundary evolution for either train or test set
        """
        # Generate static plot with model last iteration
        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        # Boundary contours using last iteration
        self._plot_boundary(levels=[0.1, self.pred_threshold, 0.9], iter=-1, ax=ax)
        # Features plot
        self._plot_features_at_current_iter(iter=-1, ax=ax)
        ax.set_title(
            f"Iteration {self.last_step if hasattr(self, "last_step") else self.maxiter}"
        )

        train_handle = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="k",
            markersize=8,
            alpha=0.3,
            label="Train",
        )
        test_handle = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="k",
            markersize=7,
            label="Test",
        )
        ax.legend(
            handles=[train_handle, test_handle], loc="upper right"
        )  # Marker-only legend

        labels = [
            r"Feature 1:  $f_{0}^{T}\Sigma f_{0}$",
            r"Feature 2:  $f_{1}^{T}\Sigma f_{1}$",
        ]
        if self.logged_flag:
            labels = [
                r"Feature 1:  $log(f_{0}^{T}\Sigma f_{0})$",
                r"Feature 2:  $log(f_{1}^{T}\Sigma f_{1})$",
            ]
            if self.ard_flag:
                labels = [
                    r"Feature 1: $e^{l_{0}}log(f_{0}^{T}\Sigma f_{0})$",
                    r"Feature 2: $e^{l_{1}}log(f_{1}^{T}\Sigma f_{1})$",
                ]
        else:
            if self.ard_flag:
                labels = [
                    r"Feature 1: $e^{l_{0}}(f_{0}^{T}\Sigma f_{0})$",
                    r"Feature 2: $e^{l_{1}}(f_{1}^{T}\Sigma f_{1})$",
                ]

        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        fig.tight_layout()
        fig.savefig(self.run_dir / f"12_features_space.png", dpi=150)
        plt.close(fig)

        # Generate the GIF
        # Verify library requirement
        if not HAS_IMAGEIO or not self.gif_flag:
            return

        frames = []
        for i, t in enumerate(self.iters_gif):
            fig, ax = plt.subplots(figsize=(5.5, 4.8))
            # Boundary contour plots
            self._plot_boundary(levels=[0.1, self.pred_threshold, 0.9], iter=i, ax=ax)
            # Features plot
            self._plot_features_at_current_iter(iter=t - 1, ax=ax)
            ax.set_title(f"Iteration {t}")
            ax.legend(
                handles=[train_handle, test_handle], loc="upper right"
            )  # comes from static plot
            ax.set_xlabel(labels[0])  # comes from static plot
            ax.set_ylabel(labels[1])  # comes from static plot
            fig.tight_layout()
            frames.append(self._fig_to_frame(fig))
            plt.close(fig)

        imageio.mimsave(
            self.run_dir / f"12_features_evolution.gif", frames, duration=0.6
        )

    # --- 13) W topomap --- #
    def _animate_weights_topomaps(self) -> None:
        """
        Generate topomap of spatial filters
        """
        # Generate the GIF
        # Verify library requirement
        if not HAS_IMAGEIO or not self.gif_flag:
            return
        if not HAS_IMAGEIO or not HAS_MNE:
            return

        frames = []
        for t in self.iters_gif:
            W_t = self.W_over_iters[t - 1]  # (s, nf)
            fig, axes = plt.subplots(1, self.nf, figsize=(4 * self.nf, 4))
            for i in range(self.nf):
                mne.viz.plot_topomap(
                    W_t[:, i],
                    self.montage_info,
                    axes=axes[i],
                    show=False,
                    sphere=1.2,
                )
                axes[i].set_title(f"W[:,{i}]  Iteration {t}")
            fig.tight_layout()
            frames.append(self._fig_to_frame(fig))
            plt.close(fig)
        imageio.mimsave(self.run_dir / "13_weights_topomap.gif", frames, duration=0.5)

    # --- 14) Synced dashboard --- #
    def _animate_synced_dashboard(self) -> None:
        """
        Generate a dashboard that sync learning and evolution of some of the important parameters
        """
        # Generate the GIF
        # Verify library requirement
        if not HAS_IMAGEIO or not self.gif_flag:
            return

        y_true_train = self.Y_train.ravel()
        y_true_test = self.Y_test.ravel()

        frames = []
        for idx, t in enumerate(self.iters_gif):
            W_t = self.W_over_iters[t - 1]  # spatial filter weights at iteration t

            fig = plt.figure(figsize=(14, 8))
            gs = fig.add_gridspec(2, 3)
            fig.suptitle(f"{self.dataset_label} - Iteration {t}")
            # Upper level
            ax_curve = fig.add_subplot(gs[0, 0])
            ax_cm_train = fig.add_subplot(gs[0, 1])
            ax_cm_test = fig.add_subplot(gs[0, 2])
            # Lower level
            ax_feat = fig.add_subplot(gs[1, 0])
            ax_w0 = fig.add_subplot(gs[1, 1])
            ax_w1 = fig.add_subplot(gs[1, 2])

            # Learning curves (top left) -------------------------------------------------------------------------------
            steps = [l.step for l in self.run_log.logs[:t]]
            nlml = [l.nlml for l in self.run_log.logs[:t]]
            acc_train = [l.acc_train for l in self.run_log.logs[:t]]
            acc_test = [l.acc_test for l in self.run_log.logs[:t]]
            ax_curve.plot(steps, nlml, color="black", label="NLML", linewidth=2)
            ax_curve.set_xlabel("Iteration")
            ax_curve.set_ylabel("Neg-ELBO (NLML approximation)")
            ax_curve.set_xlim(0, self.maxiter)
            # Generate new axis to plot accuracy scores
            ax2 = ax_curve.twinx()
            ax2.plot(
                steps, acc_train, color="blue", label="Accuracy (Train)", linestyle="--"
            )
            ax2.plot(
                steps, acc_test, color="orange", label="Accuracy (Test)", linestyle=":"
            )
            ax2.set_ylim(0, 1)
            ax2.set_ylabel("Accuracy score")
            lines = ax_curve.get_lines() + ax2.get_lines()
            ax_curve.legend(lines, [l.get_label() for l in lines], loc="best")

            # Confusion matrix (train, top right) ----------------------------------------------------------------------
            y_pred = np.array(self.run_log.y_train_seq[t - 1])
            cm = confusion_matrix(y_true_train, y_pred)
            ax_cm_train.imshow(cm, cmap="Greens", vmin=0)
            for (i, j), value in np.ndenumerate(cm):
                ax_cm_train.text(j, i, int(value), ha="center", va="center")
            ax_cm_train.set_title("Train")
            ax_cm_train.set_xlabel("Predicted")
            ax_cm_train.set_ylabel("True")
            # Confusion matrix (test, top right) -----------------------------------------------------------------------
            y_pred = np.array(self.run_log.y_test_seq[t - 1])
            cm = confusion_matrix(y_true_test, y_pred)
            ax_cm_test.imshow(cm, cmap="Greens", vmin=0)
            for (i, j), value in np.ndenumerate(cm):
                ax_cm_test.text(j, i, int(value), ha="center", va="center")
            ax_cm_test.set_title("Test")
            ax_cm_test.set_xlabel("Predicted")
            ax_cm_test.set_ylabel("True")

            # Feature evolution (train, bottom left) -------------------------------------------------------------------
            # Boundary contour plots
            self._plot_boundary(
                levels=[0.1, self.pred_threshold, 0.9], iter=idx, ax=ax_feat
            )
            # Features plot
            self._plot_features_at_current_iter(iter=t - 1, ax=ax_feat)
            ax_feat.set_title(f"Iteration {t}")

            train_handle = plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="k",
                markersize=8,
                alpha=0.3,
                label="Train",
            )
            test_handle = plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="k",
                markersize=7,
                label="Test",
            )
            ax_feat.legend(
                handles=[train_handle, test_handle], loc="upper right"
            )  # Marker-only legend

            labels = [
                r"Feature 1:  $f_{0}^{T}\Sigma f_{0}$",
                r"Feature 2:  $f_{1}^{T}\Sigma f_{1}$",
            ]
            if self.logged_flag:
                labels = [
                    r"Feature 1:  $log(f_{0}^{T}\Sigma f_{0})$",
                    r"Feature 2:  $log(f_{1}^{T}\Sigma f_{1})$",
                ]
                if self.ard_flag:
                    labels = [
                        r"Feature 1: $e^{l_{0}}log(f_{0}^{T}\Sigma f_{0})$",
                        r"Feature 2: $e^{l_{1}}log(f_{1}^{T}\Sigma f_{1})$",
                    ]
            else:
                if self.ard_flag:
                    labels = [
                        r"Feature 1: $e^{l_{0}}(f_{0}^{T}\Sigma f_{0})$",
                        r"Feature 2: $e^{l_{1}}(f_{1}^{T}\Sigma f_{1})$",
                    ]

            ax_feat.set_xlabel(labels[0])
            ax_feat.set_ylabel(labels[1])

            # W topomaps (bottom right) --------------------------------------------------------------------------------
            mne.viz.plot_topomap(
                W_t[:, 0], self.montage_info, axes=ax_w0, show=False, sphere=1.2
            )
            ax_w0.set_title("W[:,0]")
            mne.viz.plot_topomap(
                W_t[:, 1], self.montage_info, axes=ax_w1, show=False, sphere=1.2
            )
            ax_w1.set_title("W[:,1]")

            # Wrap the dashboard at iteration t
            fig.tight_layout()
            frames.append(self._fig_to_frame(fig))
            plt.close(fig)

        imageio.mimsave(self.run_dir / "00_synced_dashboard.gif", frames, duration=0.5)
