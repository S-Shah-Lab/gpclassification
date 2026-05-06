"""
GP classification runner for EEG covariance-based motor-imagery decoding.

Overview
--------
This module provides GPClassificationRunner, the central class that
orchestrates the full training pipeline:

  1. Data preparation -- accepts either pre-split dicts or flat arrays;
     flattens covariance matrices from (N, s, s) to (N, s²).
  2. Spatial-filter initialisation -- supports random, constant, or a
     user-supplied (s, nf) NumPy array (e.g. from CSP).
  3. Model construction -- builds a GPy.models.GPClassification instance
     with the custom CustomKernelGPy covariance function.
  4. Multi-stage optimisation -- each stage is described by an
     OptimizerStage dataclass that specifies the GPy optimizer name,
     number of steps, and optional optimizer-specific keyword arguments
     (learning rate, momentum, …).  A sensible single-stage default is
     provided; advanced users supply a list of stages.
  5. Best-checkpoint tracking -- the model state (param_array) and the
     predicted probabilities at the iteration with the lowest chosen metric
     (NLML or validation NLPD) are recorded and restored after training.
  6. Early stopping -- patience-based: training halts when the tracked metric
     has not improved by at least es_min_delta for es_patience
     consecutive iterations.
  7. Logging -- per-iteration metrics, kernel snapshots, and final predictions
     are serialised to run_log.json in the output directory.
  8. Visual summaries -- PNG plots for learning curves, threshold sweeps,
     calibration, kernel parameters, topomaps, confusion matrix, feature
     scatter + decision boundary, and singular values.

Dependencies
------------
- GPy, NumPy, scikit-learn, SciPy, Matplotlib
- kernels_gpy.CustomKernelGPy
- MNE (optional; required only for topomap plots)
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
import datetime as dt
from dataclasses import dataclass, asdict, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import GPy
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.linalg import svd
from scipy.sparse.linalg import svds
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from kernels_gpy import CustomKernelGPy

# ---------------------------------------------------------------------------
# Optional MNE import (topomap visualisation only)
# ---------------------------------------------------------------------------
try:
    import mne
    from mne.channels import make_dig_montage
    HAS_MNE = True
except Exception:
    HAS_MNE = False

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ArrayOrDict = Union[np.ndarray, Dict[str, np.ndarray]]


# ===========================================================================
# Dataclasses
# ===========================================================================

@dataclass
class OptimizerStage:
    """
    Configuration for a single optimisation stage.

    A training run can consist of one or more stages executed sequentially.
    Each stage calls GPy.Model.optimize once with max_iters steps
    and the specified optimizer name.

    Parameters
    ----------
    optimizer : str
        GPy optimizer identifier.  Supported values:

        - "scg"     -- Scaled Conjugate Gradient (default GPy optimizer;
                          good general-purpose choice, no learning-rate knob).
        - "lbfgsb"  -- L-BFGS-B via scipy.optimize; often converges
                          faster in the early phase but can overshoot.
        - "adadelta"— Adaptive learning-rate gradient descent;
                          accepts learning_rate and momentum kwargs.
        - "rprop"   -- Resilient back-propagation; robust to gradient
                          scale differences.
    max_iters : int
        Number of optimisation steps to run in this stage.
    kwargs : dict
        Additional keyword arguments forwarded verbatim to
        GPy.Model.optimize (which in turn passes them to the paramz
        optimizer class constructor).

        The available kwargs depend on which optimizer is selected:

        - "scg" / "lbfgsb" -- accept convergence tolerances:
          xtol, ftol, gtol.  **No learning rate.**
          SCG determines its own step length via curvature estimation
          (internal parameter sigma0 = 1e-7); it cannot be overridden
          by the user.

        - "adadelta" -- accepts step_rate (learning rate),
          decay, momentum.
          Backed by climin.adadelta.Adadelta.

        - "rprop" -- accepts step_rate (initial step size).
          Backed by the climin RProp implementation.

        - "adam" -- accepts step_rate, decay,
          decay_mom1, decay_mom2, momentum, offset.
          Backed by climin.adam.Adam.

        **Important:** the keyword for the learning rate in climin-backed
        optimizers is step_rate, not learning_rate.

    Examples
    --------
    Single stage (equivalent to the old hard-coded behaviour)::

        stage = OptimizerStage(optimizer="scg", max_iters=300)

    Two-stage warm-up + fine-tune::

        stages = [
            OptimizerStage(optimizer="adadelta", max_iters=200,
                           kwargs={"step_rate": 0.01, "momentum": 0.9}),
            OptimizerStage(optimizer="scg",      max_iters=150,
                           kwargs={"xtol": 1e-6, "ftol": 1e-6}),
        ]

    SCG with tighter convergence tolerances::

        stage = OptimizerStage(
            optimizer = "scg",
            max_iters = 400,
            kwargs    = {"xtol": 1e-8, "ftol": 1e-8},
        )
    """
    optimizer  : str  = "scg"
    max_iters  : int  = 300
    log_every  : int  = 10
    kwargs     : dict = field(default_factory=dict)


def build_default_optimizer(maxiter: int) -> List[OptimizerStage]:
    """
    Return a single-stage SCG schedule as the default optimizer list.

    This matches the original behaviour of the old _train method while
    remaining compatible with the new multi-stage interface.

    Parameters
    ----------
    maxiter : int
        Total number of SCG steps.

    Returns
    -------
    List[OptimizerStage]
        A one-element list containing an SCG stage.
    """
    return [OptimizerStage(optimizer="scg", max_iters=maxiter)]


@dataclass
class IterLog:
    """Per-iteration snapshot of metrics and kernel parameters."""

    step         : int
    # Training set
    nlml         : float
    nlpd_train   : Optional[float]
    acc_train    : Optional[float]
    brier_train  : Optional[float]
    aucroc_train : Optional[float]
    aucpr_train  : Optional[float]
    # Validation set
    nlpd_val     : Optional[float]
    acc_val      : Optional[float]
    brier_val    : Optional[float]
    aucroc_val   : Optional[float]
    aucpr_val    : Optional[float]
    # Test set
    nlpd_test    : Optional[float]
    acc_test     : Optional[float]
    brier_test   : Optional[float]
    aucroc_test  : Optional[float]
    aucpr_test   : Optional[float]
    # Kernel parameters
    W            : List[List[float]]   # spatial filter matrix snapshot
    eta          : Optional[float]     # global output scale
    ard          : Optional[List[float]]  # per-filter ARD scales


@dataclass
class RunLog:
    """Container for the complete run record; serialised to JSON."""

    meta         : Dict[str, Any]    # configuration and final metadata
    logs         : List[IterLog]     # one entry per optimisation step
    # Best-iteration predictions
    p_train_best : List[float]
    y_train_best : List[int]
    p_val_best   : List[float]
    y_val_best   : List[int]
    p_test_best  : List[float]
    y_test_best  : List[int]


# ===========================================================================
# Utility helpers
# ===========================================================================

def _ensure_dir(p: Path) -> None:
    """Create directory and all parents if they do not already exist."""
    p.mkdir(parents=True, exist_ok=True)


def _now_stamp(mode: str = "") -> str:
    """
    Return a timestamp string.

    Parameters
    ----------
    mode : str
        "nice" → "YYYY-MM-DD HH:MM:SS"; anything else → "YYYYMMDD_HHMMSS".
    """
    fmt = "%Y-%m-%d %H:%M:%S" if mode == "nice" else "%Y%m%d_%H%M%S"
    return dt.datetime.now().strftime(fmt)


# ===========================================================================
# Main class
# ===========================================================================

class GPClassificationRunner:
    """
    End-to-end GP classification pipeline for EEG covariance features.

    Parameters
    ----------
    X : np.ndarray or dict
        Covariance matrices.  Either:

        - A dict with keys "train" (required), "val" (optional),
          "test" (optional), each mapping to an array of shape
          (N, s, s).
        - A flat array of shape (N, s, s) that will be split using
          frac_val and frac_test.
    Y : np.ndarray or dict
        Class labels ({0, 1}).  Same structure as X.
    dataset_label : str
        Human-readable name used in output paths and config files.
    ch_names : list of str or None
        EEG channel names (used for topomap plots when MNE is available).
        Pass ``None`` when channel metadata is unavailable; topomap plots
        will be skipped.
    ch_xy : dict or None
        Mapping {channel_name: (x, y)} of 2D electrode coordinates
        (used for topomap plots when MNE is available).  Pass ``None``
        when coordinates are unavailable.
    spatialFilter_init : str or np.ndarray
        How to initialise the spatial filter matrix W ∈ R^{s × nf}:

        - "random"   -- i.i.d. Gaussian N(0, 1) samples.
        - "ones"     -- all-ones matrix.
        - np.ndarray -- shape (s, nf) matrix provided directly
                           (e.g. CSP filters).  Acts as a seed when
                           W_trainable=True or as a fixed filter
                           when W_trainable=False.
    nf : int
        Number of spatial filters (columns of W).
    eta_flag : bool
        Enable/disable the global output-scale parameter eta.
    ard_flag : bool
        Enable/disable per-filter ARD scaling.
    W_trainable : bool
        If False, W is fixed at its initial value and not updated.
    logged_flag : bool
        If True, features are log-transformed: z = log(w^T Σ w).
    kernel_type : str
        "RBF" or "Linear".
    optimizer_stages : list of OptimizerStage, optional
        Multi-stage optimisation schedule.  If None, a single SCG stage
        of maxiter steps is used (backward-compatible default).
    maxiter : int
        Total iteration budget.  Used to build the default single-stage
        schedule when optimizer_stages is None.  Ignored when
        optimizer_stages is provided explicitly.
    es_patience : int
        Early-stopping patience: number of consecutive **optimisation steps**
        without improvement before training is halted.  Set to 0 to
        disable.  Internally converted to blocks via
        ceil(es_patience / log_every) so the effective tolerance in steps
        is always consistent regardless of log_every.
    es_min_delta : float
        Minimum absolute improvement in the tracked metric to reset the
        patience counter.
    use_val_for_selection : bool
        When True (default) and a validation split is present (i.e. the
        input dict contains a "val" key, or the inner validation split
        was carved from the training fold by generate_train_test_from_fold),
        both model selection (best-checkpoint) and early stopping are driven
        by the **validation NLPD** rather than the training NLML.  This
        ensures that neither criterion has ever seen the held-out test fold
        and that the selection metric is genuinely independent of the
        training objective.  Set to False to force NLML-based selection
        even when a validation set is available.
    pred_threshold : float
        Probability threshold for binary classification (default 0.5).
    random_state : int
        Seed for NumPy random number generator (W initialisation, data splits).
    frac_val : float
        Fraction of data to hold out as validation when X is an array.
        Ignored when X is a dict (the dict is used as-is).
    frac_test : float
        Fraction of data to hold out as test when X is an array.
        Ignored when X is a dict.
    results_dir : str
        Root directory under which all output files are saved.
    run_name : str, optional
        Sub-folder name for this specific run.  Defaults to a timestamp.
    """

    def __init__(
        self,
        X: ArrayOrDict,
        Y: ArrayOrDict,
        dataset_label: str,
        ch_names: Optional[List[str]],
        ch_xy: Optional[Dict[str, Tuple[float, float]]],
        # Spatial filter
        spatialFilter_init: Union[str, np.ndarray] = "random",
        nf: int = 2,
        # Kernel flags
        eta_flag: bool = False,
        ard_flag: bool = False,
        W_trainable: bool = True,
        logged_flag: bool = True,
        kernel_type: str = "RBF",
        # Optimisation
        optimizer_stages: Optional[List[OptimizerStage]] = None,
        maxiter: int = 300,
        # Early stopping
        es_patience: int = 0,
        es_min_delta: float = 1e-4,
        # Model / validation selection
        use_val_for_selection: bool = True,
        # Inference
        pred_threshold: float = 0.5,
        random_state: int = 42,
        # Data splits (used only when X/Y are arrays)
        frac_val: float = 0.0,
        frac_test: float = 0.0,
        # Output
        results_dir: str = "./results",
        run_name: Optional[str] = None,
    ) -> None:

        # ------------------------------------------------------------------ #
        # Store inputs                                                         #
        # ------------------------------------------------------------------ #
        self.X = X
        self.Y = Y
        self.dataset_label = dataset_label
        self.ch_names = [c.lower() for c in ch_names] if ch_names is not None else []
        self.ch_xy    = {k.lower(): v for k, v in ch_xy.items()} if ch_xy is not None else {}

        if HAS_MNE and self.ch_names and self.ch_xy:
            self.montage_info = self._build_montage_from_xy(self.ch_names, self.ch_xy)

        # ------------------------------------------------------------------ #
        # Kernel / model flags                                                 #
        # ------------------------------------------------------------------ #
        self.spatialFilter_init = spatialFilter_init
        self.nf          = nf
        self.eta_flag    = eta_flag
        self.ard_flag    = ard_flag
        self.W_trainable = W_trainable
        self.logged_flag = logged_flag
        self.kernel_type = kernel_type

        # ------------------------------------------------------------------ #
        # Optimisation schedule                                                #
        # ------------------------------------------------------------------ #
        # If no stages are provided, fall back to a single SCG stage using the
        # legacy `maxiter` parameter so existing call-sites are unaffected.
        if optimizer_stages is None:
            self.optimizer_stages = build_default_optimizer(maxiter)
        else:
            self.optimizer_stages = list(optimizer_stages)

        # Total iteration count across all stages (used for progress display)
        self.maxiter = sum(s.max_iters for s in self.optimizer_stages)

        # ------------------------------------------------------------------ #
        # Early stopping                                                       #
        # ------------------------------------------------------------------ #
        self.es_patience  = int(es_patience)
        self.es_min_delta = float(es_min_delta)
        # Whether to use validation NLPD (rather than training NLML) for
        # model selection and early stopping when a validation set exists.
        # Automatically activated in _load_and_prepare_data when has_val=True.
        self._use_val_for_selection: bool = bool(use_val_for_selection)
        # Runtime counters -- reset at the start of _train
        self._es_counter         : int   = 0
        self._es_best            : float = float("inf")
        self._es_stopped         : bool  = False
        self._es_patience_blocks : int   = 0   # derived per-stage from es_patience / log_every

        # ------------------------------------------------------------------ #
        # Inference / split config                                             #
        # ------------------------------------------------------------------ #
        self.pred_threshold = pred_threshold
        self.random_state   = random_state
        self.frac_val       = 0.0 if frac_val  is None else float(frac_val)
        self.frac_test      = 0.0 if frac_test is None else float(frac_test)

        # ------------------------------------------------------------------ #
        # Output paths                                                         #
        # ------------------------------------------------------------------ #
        self.results_root = Path(results_dir)
        self.run_name     = run_name or f"run_{_now_stamp()}"
        self.run_dir      = self.results_root / self.run_name
        _ensure_dir(self.run_dir)

        # ------------------------------------------------------------------ #
        # Placeholders -- populated by setup methods                           #
        # ------------------------------------------------------------------ #
        self.has_train = False
        self.has_val   = False
        self.has_test  = False

        self.s       : int = 0
        self.N_train : int = 0
        self.N_val   : int = 0
        self.N_test  : int = 0

        self.X_train : Optional[np.ndarray] = None
        self.X_val   : Optional[np.ndarray] = None
        self.X_test  : Optional[np.ndarray] = None
        self.Y_train : Optional[np.ndarray] = None
        self.Y_val   : Optional[np.ndarray] = None
        self.Y_test  : Optional[np.ndarray] = None

        self.W_init  : Optional[np.ndarray]      = None
        self.model   : Optional[GPy.models.Model] = None
        self.kernel  : Optional[CustomKernelGPy]  = None

        self.run_log : Optional[RunLog] = None

        # Best-checkpoint state
        self._best_score       : float          = float("inf")
        self._best_iter        : Optional[int]  = None
        self._best_metric_name : Optional[str]  = None
        self._best_params      : Optional[np.ndarray] = None

        self._p_train_best : Optional[np.ndarray] = None
        self._p_val_best   : Optional[np.ndarray] = None
        self._p_test_best  : Optional[np.ndarray] = None
        self._y_train_best : Optional[np.ndarray] = None
        self._y_val_best   : Optional[np.ndarray] = None
        self._y_test_best  : Optional[np.ndarray] = None

        # Placeholder — overwritten by _load_and_prepare_data once has_val
        # is known.  Do not set to True here: has_val is always False at
        # __init__ time so doing so would be meaningless.
        self.use_validation_for_adaptation: bool = False

    # =======================================================================
    # Public entry point
    # =======================================================================

    def fit(self) -> None:
        """
        Execute the full pipeline end-to-end.

        Stages
        ------
        1. Create config file (bookkeeping).
        2. Load and prepare data.
        3. Initialise W.
        4. Build GPy model.
        5. Train (multi-stage optimisation + early stopping).
        6. Write run log (JSON).
        7. Generate visual summaries (PNGs).
        """
        self._print_message("start")
        self._create_config_file()

        self._load_and_prepare_data()
        self._initialize_W_matrix()
        self._build_model()

        self._train()
        self._build_and_write_runlog()
        self._make_visual_summary()

        self._print_message("end")

    # =======================================================================
    # Setup methods
    # =======================================================================

    def _print_message(self, which: str) -> None:
        """Print a coloured start/end banner to the terminal."""
        GREEN  = "\033[92m"
        YELLOW = "\033[93m"
        RESET  = "\033[0m"
        if which == "start":
            print(f"[RUN START] {_now_stamp(mode='nice')}")
            print(f"{GREEN}{self.run_name}_nf{self.nf}{RESET}")
        elif which == "end":
            print(YELLOW + f"[RUN END] {_now_stamp(mode='nice')}" + RESET + "\n\n")

    def _create_config_file(self) -> None:
        """
        Populate self.cfg with the run configuration for bookkeeping.

        Called at the start of fit so the config is written before any
        training begins.  Data-shape entries are added later by
        _load_and_prepare_data.
        """
        # Serialise the optimizer schedule to plain dicts
        stages_repr = [
            {"optimizer": s.optimizer, "max_iters": s.max_iters, "kwargs": s.kwargs}
            for s in self.optimizer_stages
        ]

        self.cfg: Dict[str, Any] = {
            "run_name"      : self.run_name,
            "dataset_label" : self.dataset_label,
            "results_dir"   : str(self.results_root.resolve()),
            "timestamp_start": _now_stamp(),
            # Data
            "data_input_mode": "dict" if isinstance(self.X, dict) else "array",
            "#channels"      : len(self.ch_names),
            # Model
            "spatialFilter_init": (
                {"type": "array", "shape": list(self.spatialFilter_init.shape)}
                if isinstance(self.spatialFilter_init, np.ndarray)
                else self.spatialFilter_init
            ),
            "nf"           : self.nf,
            "eta_flag"     : self.eta_flag,
            "ard_flag"     : self.ard_flag,
            "logged_flag"  : self.logged_flag,
            "kernel_type"  : self.kernel_type,
            # Optimisation
            "optimizer_stages"     : stages_repr,
            "maxiter_total"        : self.maxiter,
            "es_patience_steps"    : self.es_patience,
            "es_min_delta"         : self.es_min_delta,
            "use_val_for_selection": self._use_val_for_selection,
            # Inference
            "pred_threshold" : self.pred_threshold,
            "random_state"   : self.random_state,
            "frac_val"       : self.frac_val,
            "frac_test"      : self.frac_test,
        }

    def _build_montage_from_xy(
        self,
        ch_names: List[str],
        ch_xy: Dict[str, Tuple[float, float]],
        default_z: float = 0.0,
    ) -> "mne._fiff.meas_info.Info":
        """
        Build an MNE Info object from 2D electrode XY coordinates.

        Parameters
        ----------
        ch_names : list of str
            Lower-case channel names.
        ch_xy : dict
            {name: (x, y)} coordinate mapping.
        default_z : float
            Z-coordinate assigned to all electrodes (default 0.0).

        Returns
        -------
        mne.Info
        """
        ch_pos, missing = {}, []
        for name in ch_names:
            if name in ch_xy:
                x, y = ch_xy[name]
                ch_pos[name] = (x, y, default_z)
            else:
                missing.append(name)

        if missing:
            print(
                f"[montage] {len(missing)} channels lack XY coords: "
                f"{missing[:8]}{'...' if len(missing) > 8 else ''}"
            )

        montage    = make_dig_montage(ch_pos=ch_pos, coord_frame="head")
        info_full  = mne.create_info(ch_names=ch_names, sfreq=1000.0, ch_types="eeg")
        info_full.set_montage(montage)
        return info_full

    # -----------------------------------------------------------------------
    # Data preparation
    # -----------------------------------------------------------------------

    def _load_and_prepare_data(self) -> None:
        """
        Ingest self.X / self.Y and populate the train/val/test arrays.

        Handles two input modes:

        **Dict mode** -- self.X is a dict with at least a "train"
        key.  "val" and "test" are optional.  Each value must be an
        array of shape (N, s, s); it is flattened to (N, s²).

        **Array mode** -- self.X is a single array (N, s, s) that is
        split into train/val/test according to self.frac_val and
        self.frac_test.

        After this method, the following attributes are set:

        - self.X_train, self.Y_train -- always present.
        - self.X_val,   self.Y_val   -- None when no val split.
        - self.X_test,  self.Y_test  -- None when no test split.
        - self.s        -- number of EEG sensors.
        - self.N_train, self.N_val, self.N_test
        - self.has_train, self.has_val, self.has_test
        """

        def _to_col(Ya: np.ndarray) -> np.ndarray:
            return np.asarray(Ya).reshape(-1, 1)

        def _flatten(X: np.ndarray) -> np.ndarray:
            """Flatten (N, s, s) → (N, s*s); accepts lists as well."""
            X = X if isinstance(X, np.ndarray) else np.asarray(X)
            N = X.shape[0]
            return X.reshape(N, -1)

        def _set(Xtr, Ytr, Xva, Yva, Xte, Yte) -> None:
            self.X_train, self.Y_train = Xtr, Ytr
            self.X_val,   self.Y_val   = Xva, Yva
            self.X_test,  self.Y_test  = Xte, Yte

            self.has_train = Xtr is not None
            self.has_val   = Xva is not None
            self.has_test  = Xte is not None

            self.N_train = int(len(Xtr)) if self.has_train else 0
            self.N_val   = int(len(Xva)) if self.has_val   else 0
            self.N_test  = int(len(Xte)) if self.has_test  else 0

            self.cfg.update({"N_train": self.N_train,
                             "N_val"  : self.N_val,
                             "N_test" : self.N_test})
            print(f"  [Data] train={self.N_train}  val={self.N_val}  test={self.N_test}")

        # Validate split fractions
        fv = float(self.frac_val  or 0.0)
        ft = float(self.frac_test or 0.0)
        if not (0.0 <= fv <= 1.0 and 0.0 <= ft <= 1.0):
            raise ValueError("frac_val and frac_test must be in [0, 1]")

        # ---- Dict input ----
        if isinstance(self.X, dict) and isinstance(self.Y, dict):
            Xtr = self.X.get("train")
            if Xtr is None:
                raise ValueError("Input dict must contain at least the 'train' key.")
            self.s = Xtr.shape[-1]

            Xtr = _flatten(Xtr);  Ytr = _to_col(self.Y["train"])
            Xva = _flatten(self.X["val"])  if "val"  in self.X else None
            Yva = _to_col( self.Y["val"])  if "val"  in self.Y else None
            Xte = _flatten(self.X["test"]) if "test" in self.X else None
            Yte = _to_col( self.Y["test"]) if "test" in self.Y else None

            _set(Xtr, Ytr, Xva, Yva, Xte, Yte)
            # Activate val-based model selection now that has_val is known.
            self.use_validation_for_adaptation = (
                self._use_val_for_selection and self.has_val
            )
            if self.use_validation_for_adaptation:
                print("  [Selection] Early stopping and best-model selection "
                      "will use validation NLPD.")
            else:
                print("  [Selection] No validation set — using training NLML "
                      "for early stopping and best-model selection.")
            return

        # ---- Array input ----
        self.s   = self.X.shape[-1]
        X_all    = _flatten(self.X)
        Y_all    = _to_col(self.Y)

        if fv == 0.0 and ft == 0.0:
            _set(X_all, Y_all, None, None, None, None)
            # use_validation_for_adaptation stays False (no val set); print status.
            print("  [Selection] No validation set — using training NLML "
                  "for early stopping and best-model selection.")
            return

        # Split off test first, then validation from the remainder
        if ft > 0.0:
            X_tmp, Xte, Y_tmp, Yte = train_test_split(
                X_all, Y_all, test_size=ft,
                random_state=self.random_state, shuffle=True,
            )
        else:
            X_tmp, Y_tmp, Xte, Yte = X_all, Y_all, None, None

        if fv > 0.0:
            Xtr, Xva, Ytr, Yva = train_test_split(
                X_tmp, Y_tmp, test_size=fv,
                random_state=self.random_state, shuffle=True,
            )
        else:
            Xtr, Ytr, Xva, Yva = X_tmp, Y_tmp, None, None

        _set(Xtr, Ytr, Xva, Yva, Xte, Yte)
        self.use_validation_for_adaptation = (
            self._use_val_for_selection and self.has_val
        )
        if self.use_validation_for_adaptation:
            print("  [Selection] Early stopping and best-model selection "
                  "will use validation NLPD.")
        else:
            print("  [Selection] No validation set — using training NLML "
                  "for early stopping and best-model selection.")

    def _initialize_W_matrix(self) -> None:
        """
        Initialise the spatial filter matrix self.W_init of shape (s, nf).

        Accepts either a string policy or a pre-computed NumPy array:

        - "random"     -- i.i.d. Gaussian samples from N(0, 1).
        - "ones"       -- all-ones matrix.
        - np.ndarray   -- shape (s, nf) array copied directly.
        """
        rng = np.random.default_rng(self.random_state)

        if isinstance(self.spatialFilter_init, np.ndarray):
            W = np.asarray(self.spatialFilter_init, dtype=np.float64)
            if W.ndim != 2:
                raise ValueError("spatialFilter_init array must be 2D (s, nf).")
            if W.shape[0] != self.s:
                raise ValueError(
                    f"spatialFilter_init has {W.shape[0]} rows; expected s={self.s}."
                )
            if W.shape[1] != self.nf:
                raise ValueError(
                    f"spatialFilter_init has {W.shape[1]} cols; expected nf={self.nf}."
                )
            self.W_init = W.copy()

        elif self.spatialFilter_init == "random":
            self.W_init = rng.normal(0.0, 1.0, size=(self.s, self.nf))

        elif self.spatialFilter_init == "ones":
            self.W_init = np.ones((self.s, self.nf), dtype=np.float64)

        else:
            raise ValueError(
                f"Unknown spatialFilter_init: '{self.spatialFilter_init}'. "
                "Expected 'random', 'ones', or a NumPy array."
            )

        self.cfg.update({
            "W_init_shape": list(self.W_init.shape),
            "W_trainable" : bool(self.W_trainable),
            "W_source"    : (
                "array" if isinstance(self.spatialFilter_init, np.ndarray)
                else str(self.spatialFilter_init)
            ),
        })

    # -----------------------------------------------------------------------
    # Model construction
    # -----------------------------------------------------------------------

    def _build_model(self) -> None:
        """
        Construct the GPy GPClassification model.

        Builds the custom kernel first, then wraps it with EP inference.
        """
        self.kernel = CustomKernelGPy(
            self.W_init,
            W_trainable = self.W_trainable,
            ard_flag    = self.ard_flag,
            eta_flag    = self.eta_flag,
            logged_flag = self.logged_flag,
            kernel_type = self.kernel_type,
        )

        X_train = np.asarray(self.X_train, dtype=np.float64)
        Y_train = np.asarray(self.Y_train, dtype=int).reshape(-1, 1)

        ep = GPy.inference.latent_function_inference.EP()
        self.model = GPy.models.GPClassification(
            X=X_train, Y=Y_train, kernel=self.kernel, inference_method=ep,
        )

    # =======================================================================
    # Training
    # =======================================================================

    def _train(self) -> None:
        """
        Run the multi-stage optimisation loop with optional early stopping.

        For each OptimizerStage, the optimizer is called in blocks of
        stage.log_every steps rather than one step at a time.  This is
        critical for performance: calling model.optimize(max_iters=1)
        N times forces EP to re-initialise its site parameters on every
        call (on_optimization_start), whereas calling
        model.optimize(max_iters=log_every) N/log_every times pays that
        cost only once per block.  For typical EP on ~100 samples this
        reduces wall-clock time by 5–20×.

        Metrics (NLML, accuracy, Brier score) are logged once per block,
        at the end of each block.  The step counter increments by
        log_every per log entry, not by 1.

        Early stopping is specified in **steps** (self.es_patience).  At
        the start of each stage it is converted to blocks via
        ceil(es_patience / log_every) and stored in
        self._es_patience_blocks.  This keeps the patience threshold
        consistent in step-space regardless of log_every.

        After all stages (or early stopping), the best checkpoint is restored.
        """
        self.logs: List[IterLog] = []

        # Reset early-stopping state
        self._es_counter         = 0
        self._es_best            = float("inf")
        self._es_stopped         = False
        self._es_patience_blocks = 0   # set per-stage below

        self.step = 0  # global step counter (increments by log_every per block)
        total_blocks = sum(
            max(1, s.max_iters // s.log_every) for s in self.optimizer_stages
        )
        print_freq  = max(1, total_blocks // 10)
        block_count = 0  # counts blocks across all stages (for print_freq)

        for stage_idx, stage in enumerate(self.optimizer_stages):
            if self._es_stopped:
                break

            log_every   = max(1, stage.log_every)
            n_blocks    = max(1, stage.max_iters // log_every)
            remainder   = stage.max_iters - n_blocks * log_every  # leftover steps

            # Convert step-based patience to blocks for this stage's log_every.
            # ceil ensures we never stop *earlier* than the user-specified step count.
            self._es_patience_blocks = (
                max(1, math.ceil(self.es_patience / log_every))
                if self.es_patience > 0 else 0
            )

            print(
                f"  [Stage {stage_idx + 1}/{len(self.optimizer_stages)}] "
                f"optimizer={stage.optimizer}  max_iters={stage.max_iters}  "
                f"log_every={log_every}"
                + (
                    f"  es_patience={self.es_patience} steps "
                    f"({self._es_patience_blocks} blocks)"
                    if self.es_patience > 0 else ""
                )
                + (f"  kwargs={stage.kwargs}" if stage.kwargs else "")
            )

            for block_i in range(n_blocks):
                # Run log_every steps in one optimize call -- EP re-inits only once
                self.model.optimize(
                    optimizer = stage.optimizer,
                    messages  = False,
                    max_iters = log_every,
                    **stage.kwargs,
                )
                self.step   += log_every
                block_count += 1

                # Predict and log once per block
                p_train = self._predict_prob(self.X_train)
                p_val   = self._predict_prob(self.X_val)  if self.has_val  else None
                p_test  = self._predict_prob(self.X_test) if self.has_test else None

                self._snapshot_iteration(p_train, p_val, p_test)
                self._check_for_best_iteration(p_train, p_val, p_test)

                if block_count % print_freq == 0 or block_count == 1:
                    self._print_state_on_terminal()

                if self.es_patience > 0:
                    self._check_for_early_stopping()
                    if self._es_stopped:
                        print(
                            f"  [EarlyStopping] Triggered at step {self.step} "
                            f"(patience={self.es_patience} steps = "
                            f"{self._es_patience_blocks} blocks at log_every={log_every}, "
                            f"min_delta={self.es_min_delta})"
                        )
                        break

            # Run any leftover steps that didn't fill a complete block
            if remainder > 0 and not self._es_stopped:
                self.model.optimize(
                    optimizer = stage.optimizer,
                    messages  = False,
                    max_iters = remainder,
                    **stage.kwargs,
                )
                self.step   += remainder
                block_count += 1

                p_train = self._predict_prob(self.X_train)
                p_val   = self._predict_prob(self.X_val)  if self.has_val  else None
                p_test  = self._predict_prob(self.X_test) if self.has_test else None

                self._snapshot_iteration(p_train, p_val, p_test)
                self._check_for_best_iteration(p_train, p_val, p_test)

                if self.es_patience > 0:
                    self._check_for_early_stopping()

        # Restore best checkpoint
        if self._best_params is not None:
            self.model.param_array[:]  = self._best_params
            self.model._trigger_params_changed()
            print(
                f"  [Restore] Best {self._best_metric_name}: "
                f"{self._best_score:.4f} at step {self._best_iter}"
            )
            self.cfg.update({
                "best_iter"        : self._best_iter,
                "best_metric"      : self._best_metric_name,
                "best_metric_value": self._best_score,
            })
        else:
            print("  [Restore] No best checkpoint found; keeping final state.")

    # -----------------------------------------------------------------------
    # Prediction and metrics
    # -----------------------------------------------------------------------

    def _predict_prob(self, X: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Predict class-1 probabilities for a batch of inputs.

        Parameters
        ----------
        X : np.ndarray, shape (N, s*s), or None
            Flattened covariance matrices.

        Returns
        -------
        np.ndarray, shape (N,), or None when X is None.
        """
        if X is None:
            return None
        mu, _ = self.model.predict(X)
        return mu.ravel()

    def _compute_metrics(
        self,
        y_true: Optional[np.ndarray],
        p: Optional[np.ndarray],
    ) -> Dict[str, Optional[float]]:
        """
        Compute a standard set of classification metrics.

        Parameters
        ----------
        y_true : np.ndarray or None
        p : np.ndarray or None
            Predicted class-1 probabilities in [0, 1].

        Returns
        -------
        dict with keys: "acc", "brier", "aucroc", "aucpr", "nlpd".
        All values are None if either argument is None.
        """
        metrics: Dict[str, Optional[float]] = {
            "acc"   : None,
            "brier" : None,
            "aucroc": None,
            "aucpr" : None,
            "nlpd"  : None,
        }
        if y_true is None or p is None:
            return metrics

        y_true  = y_true.ravel().astype(int)
        p       = p.ravel().astype(np.float64)
        y_hat   = (p >= self.pred_threshold).astype(int)

        metrics["acc"]   = float(accuracy_score(y_true, y_hat))
        metrics["brier"] = float(brier_score_loss(y_true, p))

        try:
            metrics["aucroc"] = float(roc_auc_score(y_true, p))
        except Exception:
            pass

        try:
            prec, rec, _ = precision_recall_curve(y_true, p)
            metrics["aucpr"] = float(np.trapz(prec, rec))
        except Exception:
            pass

        eps     = 1e-12
        p_clip  = np.clip(p, eps, 1.0 - eps)
        ll      = y_true * np.log(p_clip) + (1 - y_true) * np.log(1.0 - p_clip)
        metrics["nlpd"] = float(-np.mean(ll))

        return metrics

    def _snapshot_kernel(self) -> Dict[str, Optional[Any]]:
        """
        Capture the current kernel parameter values.

        Returns
        -------
        dict with keys "W", "eta", "ard"; values are Python
        lists / floats / None for easy JSON serialisation.
        """
        W = eta = ard = None
        if self.kernel is not None:
            try:
                W_attr = getattr(self.kernel, "W", None)
                W = None if W_attr is None else np.asarray(W_attr).tolist()
            except Exception:
                pass
            try:
                eta_attr = getattr(self.kernel, "eta", None)
                eta = None if eta_attr is None else float(eta_attr)
            except Exception:
                pass
            try:
                ard_attr = getattr(self.kernel, "ard", None)
                ard = None if ard_attr is None else np.asarray(ard_attr).ravel().tolist()
            except Exception:
                pass
        return {"W": W, "eta": eta, "ard": ard}

    def _snapshot_iteration(
        self,
        p_train: Optional[np.ndarray],
        p_val:   Optional[np.ndarray],
        p_test:  Optional[np.ndarray],
    ) -> None:
        """
        Record metrics and kernel state for the current step into self.logs.

        Parameters
        ----------
        p_train, p_val, p_test : np.ndarray or None
            Predicted probabilities on each split at this step.
        """
        m_train = self._compute_metrics(self.Y_train, p_train)
        m_val   = self._compute_metrics(self.Y_val,   p_val)
        m_test  = self._compute_metrics(self.Y_test,  p_test)

        ksnap  = self._snapshot_kernel()
        nlml   = float(-self.model.log_likelihood())

        self.logs.append(IterLog(
            step         = int(self.step),
            nlml         = nlml,
            nlpd_train   = m_train["nlpd"],
            acc_train    = m_train["acc"],
            brier_train  = m_train["brier"],
            aucroc_train = m_train["aucroc"],
            aucpr_train  = m_train["aucpr"],
            nlpd_val     = m_val["nlpd"],
            acc_val      = m_val["acc"],
            brier_val    = m_val["brier"],
            aucroc_val   = m_val["aucroc"],
            aucpr_val    = m_val["aucpr"],
            nlpd_test    = m_test["nlpd"],
            acc_test     = m_test["acc"],
            brier_test   = m_test["brier"],
            aucroc_test  = m_test["aucroc"],
            aucpr_test   = m_test["aucpr"],
            W            = ksnap["W"],
            eta          = ksnap["eta"],
            ard          = ksnap["ard"],
        ))

    def _selection_metric(self, last: IterLog) -> Tuple[str, Optional[float]]:
        """
        Choose which scalar metric drives model selection and early stopping.

        Uses nlpd_val when a validation set is available and
        use_validation_for_adaptation is True; falls back to nlml
        otherwise.

        Returns
        -------
        (name, value) : (str, float or None)
        """
        if self.use_validation_for_adaptation and self.has_val:
            return "nlpd_val", last.nlpd_val
        return "nlml", last.nlml

    def _check_for_best_iteration(
        self,
        p_train: Optional[np.ndarray],
        p_val:   Optional[np.ndarray],
        p_test:  Optional[np.ndarray],
    ) -> None:
        """
        Update the best-checkpoint if the current step improves the tracked metric.

        Stores a copy of model.param_array and the current predicted
        probabilities when a new best is found.

        Parameters
        ----------
        p_train, p_val, p_test : np.ndarray or None
            Predicted probabilities at the current step.
        """
        last = self.logs[-1]
        name, value = self._selection_metric(last)

        if value is None or not np.isfinite(value):
            return

        if value < self._best_score:
            self._best_score       = float(value)
            self._best_iter        = self.step
            self._best_metric_name = name
            self._best_params      = self.model.param_array.copy()

            # Store probabilities and derived labels at the new best step
            if p_train is not None:
                self._p_train_best = np.asarray(p_train, dtype=float).ravel()
                self._y_train_best = (self._p_train_best >= self.pred_threshold).astype(int)
            if p_val is not None:
                self._p_val_best = np.asarray(p_val, dtype=float).ravel()
                self._y_val_best = (self._p_val_best >= self.pred_threshold).astype(int)
            if p_test is not None:
                self._p_test_best = np.asarray(p_test, dtype=float).ravel()
                self._y_test_best = (self._p_test_best >= self.pred_threshold).astype(int)

    def _check_for_early_stopping(self) -> None:
        """
        Check whether early stopping should fire after the current block.

        self.es_patience is specified in **steps** by the user.  It is
        converted to blocks at the start of each stage and stored in
        self._es_patience_blocks.  The counter self._es_counter
        increments by one per block (not per step), so comparing it against
        _es_patience_blocks keeps the effective tolerance consistent in
        step-space.

        Sets self._es_stopped = True when the patience limit is reached.
        The metric used is the same as the model-selection metric
        (nlpd_val or nlml).

        A step is considered an improvement when the metric decreases by more
        than self.es_min_delta relative to the running best.
        """
        last = self.logs[-1]
        _, value = self._selection_metric(last)

        if value is None or not np.isfinite(value):
            return

        if value < self._es_best - self.es_min_delta:
            # Genuine improvement: reset counter and update running best
            self._es_best    = float(value)
            self._es_counter = 0
        else:
            self._es_counter += 1
            if self._es_counter >= self._es_patience_blocks:
                self._es_stopped = True

    def _print_state_on_terminal(self) -> None:
        """Print a compact one-line progress summary for the current step.

        When an inner validation set is present the line includes both the
        training NLML and the validation NLPD so it is easy to spot
        divergence (rising val NLPD while NLML keeps falling) during a run.
        """
        def _f(v):
            return f"{float(v):.3f}" if (v is not None and np.isfinite(v)) else "/"

        last = self.logs[-1]
        pct  = int(self.step / self.maxiter * 100)

        # Build the loss part: always show NLML; add val NLPD when available
        loss_str = f"nlml {last.nlml:.3f}"
        if self.has_val and last.nlpd_val is not None and np.isfinite(last.nlpd_val):
            sel = " *" if self.use_validation_for_adaptation else ""
            loss_str += f" | nlpd_val {last.nlpd_val:.3f}{sel}"

        print(
            f"[{pct:3d}%] step {last.step:4d}/{self.maxiter} | "
            f"{loss_str} | "
            f"acc (tr={_f(last.acc_train)}, va={_f(last.acc_val)}, te={_f(last.acc_test)}) | "
            f"brier (tr={_f(last.brier_train)}, va={_f(last.brier_val)}, te={_f(last.brier_test)})"
        )

    # =======================================================================
    # RunLog serialisation
    # =======================================================================

    def _build_and_write_runlog(self) -> None:
        """
        Assemble the RunLog dataclass and serialise it to run_log.json.

        Only the best-iteration predictions (not per-iteration sequences) are
        stored to keep file sizes manageable.
        """
        def _safe_list(arr, dtype):
            return [] if arr is None else np.asarray(arr, dtype=dtype).ravel().tolist()

        self.run_log = RunLog(
            meta         = deepcopy(self.cfg),
            logs         = self.logs,
            p_train_best = _safe_list(self._p_train_best, float),
            y_train_best = _safe_list(self._y_train_best, int),
            p_val_best   = _safe_list(self._p_val_best,   float),
            y_val_best   = _safe_list(self._y_val_best,   int),
            p_test_best  = _safe_list(self._p_test_best,  float),
            y_test_best  = _safe_list(self._y_test_best,  int),
        )

        _ensure_dir(self.run_dir)
        with open(self.run_dir / "run_log.json", "w") as fh:
            json.dump(asdict(self.run_log), fh, indent=2)

    # =======================================================================
    # Snapshot accessors
    # =======================================================================

    def _get_best_snapshot(self, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (y_true, p_best, y_best) for a given data split.

        Parameters
        ----------
        split : str
            One of "train", "val", "test".

        Returns
        -------
        y_true : np.ndarray, shape (N,)
        p_best : np.ndarray, shape (N,)   -- probabilities at best iteration
        y_best : np.ndarray, shape (N,)   -- predicted labels at best iteration
        """
        if split == "train":
            y_true = np.asarray(self.Y_train).ravel()
            p_best = np.asarray(self.run_log.p_train_best or [], dtype=float).ravel()
            y_best = np.asarray(self.run_log.y_train_best or [], dtype=int).ravel()
        elif split == "val":
            y_true = np.asarray(self.Y_val).ravel()
            p_best = np.asarray(self.run_log.p_val_best or [],   dtype=float).ravel()
            y_best = np.asarray(self.run_log.y_val_best or [],   dtype=int).ravel()
        elif split == "test":
            y_true = np.asarray(self.Y_test).ravel()
            p_best = np.asarray(self.run_log.p_test_best or [],  dtype=float).ravel()
            y_best = np.asarray(self.run_log.y_test_best or [],  dtype=int).ravel()
        else:
            raise ValueError(f"Unknown split: '{split}'. Expected 'train', 'val', or 'test'.")

        # Derive labels from probabilities when the stored vector is empty
        if y_best.size == 0 and p_best.size > 0:
            y_best = (p_best >= self.pred_threshold).astype(int)

        return y_true, p_best, y_best

    def _get_best_probabilities(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (y_true, p_best) for a split."""
        y_true, p_best, _ = self._get_best_snapshot(split)
        return y_true, p_best

    def _get_best_predictions(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (y_true, y_pred_best) for a split."""
        y_true, _, y_best = self._get_best_snapshot(split)
        return y_true, y_best

    # =======================================================================
    # Visual summary
    # =======================================================================

    def _make_visual_summary(self) -> None:
        """
        Generate all PNG diagnostic plots and save them to self.run_dir.

        Plots produced (saved to self.run_dir):

        - 01_learning_curves.png    -- NLML (black) + val NLPD (grey dashed,
                                       when inner val exists) on the left axis;
                                       per-split accuracy on the right axis.
        - plot_02_threshold_sweep.png   -- ROC, PR, and metric-vs-threshold
        - plot_03_calibration_curve.png -- reliability diagrams
        - plot_05_kernel_parameters.png -- eta and ARD trajectories
        - plot_06_kernel_W.png          -- spatial filter weight trajectories
        - plot_08_topomaps.png          -- topomap of best-iteration filters (MNE only)
        - plot_09_confusion_matrix.png  -- confusion matrices for each split
        - plot_10_features_and_boundary_NxM.png -- 2D feature scatter + boundary (nf==2 only)
        - plot_16_singular_values.png   -- singular values of the feature matrix
        """
        self.colors = {"train": "dodgerblue", "val": "forestgreen", "test": "orangered"}

        self._plot_learning_curves()
        self._plot_threshold_sweep()
        self._plot_calibration_curves()
        self._plot_kernel_scaling()
        self._plot_kernel_W()
        self._plot_topomap()
        self._plot_confusion_matrix()

        if self.nf == 2:
            for pair in combinations(range(self.nf), 2):
                self.feature_pair = pair
                self._plot_features_and_boundary()

        self._plot_sv()

    # -----------------------------------------------------------------------
    # Learning curves
    # -----------------------------------------------------------------------

    def _plot_learning_curves(self) -> None:
        """
        Plot training dynamics over optimisation steps.

        Left axis (black / grey):
          - NLML (solid black) — the training objective being minimised.
          - Val NLPD (dashed grey) — the held-out validation negative log
            predictive density used for model selection and early stopping
            when an inner validation set is present.  Plotting it on the same
            axis as NLML makes it easy to see when the two diverge, which is
            the primary signal that the model is starting to overfit.

        Right axis (colours):
          - Per-split accuracy trajectories (train, val, test).

        A vertical dashed line marks the best iteration selected by the
        model-selection metric (val NLPD when available, NLML otherwise).
        When early stopping fired, a second vertical line marks the step at
        which training halted.
        """
        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        steps    = [l.step for l in self.run_log.logs]
        nlml     = [l.nlml for l in self.run_log.logs]
        x_max    = max(steps) if steps else self.maxiter

        # --- Left axis: NLML + optional val NLPD ----------------------------
        ax1.plot(steps, nlml, linewidth=2, color="black", label="NLML (train)")
        ax1.set_xlabel("Optimisation step")
        ax1.set_ylabel("Neg log-marginal-likelihood / NLPD")
        ax1.set_xlim(0, x_max)

        # Val NLPD — plotted only when an inner validation set exists
        if self.has_val:
            nlpd_val = [l.nlpd_val for l in self.run_log.logs]
            if any(v is not None and np.isfinite(v) for v in nlpd_val):
                ax1.plot(
                    steps, nlpd_val,
                    linewidth  = 1.8,
                    color      = "dimgrey",
                    linestyle  = "--",
                    label      = "NLPD (val)",
                    alpha      = 0.85,
                )

        # --- Best-iteration marker ------------------------------------------
        if self._best_iter is not None:
            metric_label = getattr(self, "_best_metric_name", "metric")
            ax1.axvline(
                self._best_iter,
                color     = "black",
                linestyle = "--",
                linewidth = 1,
                alpha     = 0.5,
                label     = f"best {metric_label}={self._best_iter}",
            )

        # --- Early-stopping marker ------------------------------------------
        if self._es_stopped and steps and max(steps) < self.maxiter:
            ax1.axvline(
                max(steps),
                color     = "red",
                linestyle = ":",
                linewidth = 1.2,
                alpha     = 0.7,
                label     = f"ES@{max(steps)}",
            )

        # --- Right axis: accuracy per split ---------------------------------
        ax2 = ax1.twinx()
        for split, attr in [("train", "acc_train"), ("val", "acc_val"), ("test", "acc_test")]:
            has = getattr(self, f"has_{split}", split == "train")
            if has:
                vals = [getattr(l, attr) for l in self.run_log.logs]
                ax2.plot(steps, vals, linewidth=1.2,
                         color=self.colors[split], label=f"acc {split}")

        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Accuracy")

        lines  = ax1.get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="best", fontsize=8)

        fig.tight_layout()
        fig.savefig(self.run_dir / "01_learning_curves.png", dpi=150)
        plt.close(fig)

    # -----------------------------------------------------------------------
    # Threshold sweep
    # -----------------------------------------------------------------------

    def _plot_threshold_sweep(self, verbose: bool = False) -> None:
        """
        Sweep the decision threshold from 0 to 1 and plot multiple metrics.

        Panels produced (8 sub-plots in a 4×2 grid):

        - ROC curve and PR curve (top row).
        - Accuracy, Precision, Recall, F1, Specificity, Youden's J vs threshold.

        For each metric, a triangle marker indicates the threshold that
        maximises that metric.
        """

        def _fmt(x) -> str:
            return f"{x:.3f}" if (x is not None and np.isfinite(x)) else "/"

        def _aucs(y, p):
            if y is None or p is None or len(p) == 0:
                return None, None
            roc_val = ap = None
            try:
                roc_val = float(roc_auc_score(y, p))
            except Exception:
                pass
            try:
                ap = float(average_precision_score(y, p))
            except Exception:
                pass
            return roc_val, ap

        def _metrics_at_thresholds(y, p, thr_seq):
            """Compute acc/precision/recall/f1/specificity/youden at each threshold."""
            nan = np.full_like(thr_seq, np.nan, dtype=float)
            if y is None or p is None or len(p) == 0:
                return {k: nan for k in ("acc", "prec", "rec", "f1", "spec", "youden")}

            out = {k: [] for k in ("acc", "prec", "rec", "f1", "spec", "youden")}
            y   = y.astype(int)
            for th in thr_seq:
                yhat = (p >= th).astype(int)
                TP = int(np.sum((yhat == 1) & (y == 1)))
                FP = int(np.sum((yhat == 1) & (y == 0)))
                TN = int(np.sum((yhat == 0) & (y == 0)))
                FN = int(np.sum((yhat == 0) & (y == 1)))
                N  = len(y)

                acc  = (TP + TN) / N if N else np.nan
                prec = TP / (TP + FP) if (TP + FP) > 0 else np.nan
                rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                f1   = (2 * prec * rec / (prec + rec)
                        if (not np.isnan(prec) and (prec + rec) > 0) else np.nan)
                spec = TN / (TN + FP) if (TN + FP) > 0 else np.nan
                fpr  = FP / (FP + TN) if (FP + TN) > 0 else 0.0

                out["acc"].append(acc);   out["prec"].append(prec)
                out["rec"].append(rec);   out["f1"].append(f1)
                out["spec"].append(spec); out["youden"].append(rec - fpr)

            return {k: np.asarray(v, dtype=float) for k, v in out.items()}

        def _best_idx(arr):
            arr   = np.asarray(arr, dtype=float)
            score = np.where(np.isnan(arr), -np.inf, arr)
            idxs  = np.where(score == np.max(score))[0]
            return int(idxs[0]) if idxs.size else 0

        def _roc(y, p):
            if y is None or p is None or len(p) == 0:
                return None, None
            try:
                fpr, tpr, _ = roc_curve(y, p); return fpr, tpr
            except Exception:
                return None, None

        def _pr(y, p):
            if y is None or p is None or len(p) == 0:
                return None, None
            try:
                prec, rec, _ = precision_recall_curve(y, p); return prec, rec
            except Exception:
                return None, None

        thr_seq = np.linspace(0.0, 1.0, 51)

        y_tr, p_tr = self._get_best_probabilities("train")
        y_va, p_va = self._get_best_probabilities("val")  if self.has_val  else (None, None)
        y_te, p_te = self._get_best_probabilities("test") if self.has_test else (None, None)

        auc_tr, ap_tr = _aucs(y_tr, p_tr)
        auc_va, ap_va = _aucs(y_va, p_va)
        auc_te, ap_te = _aucs(y_te, p_te)

        met_tr = _metrics_at_thresholds(y_tr, p_tr, thr_seq)
        met_va = _metrics_at_thresholds(y_va, p_va, thr_seq)
        met_te = _metrics_at_thresholds(y_te, p_te, thr_seq)

        fpr_tr, tpr_tr = _roc(y_tr, p_tr)
        fpr_va, tpr_va = _roc(y_va, p_va)
        fpr_te, tpr_te = _roc(y_te, p_te)
        prec_tr, rec_tr = _pr(y_tr, p_tr)
        prec_va, rec_va = _pr(y_va, p_va)
        prec_te, rec_te = _pr(y_te, p_te)

        fig, axes = plt.subplots(4, 2, figsize=(10.5, 12.0), sharex=False)
        axes = axes.ravel()

        # ROC
        ax = axes[0]
        ax.plot([0, 1], [0, 1], ls="--", color="0.7", lw=1)
        if fpr_tr is not None:
            ax.plot(fpr_tr, tpr_tr, color=self.colors["train"], lw=2,
                    label=f"train AUC={_fmt(auc_tr)}")
        if self.has_val and fpr_va is not None:
            ax.plot(fpr_va, tpr_va, color=self.colors["val"], lw=2,
                    label=f"val AUC={_fmt(auc_va)}")
        if self.has_test and fpr_te is not None:
            ax.plot(fpr_te, tpr_te, color=self.colors["test"], lw=2,
                    label=f"test AUC={_fmt(auc_te)}")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="FPR", ylabel="TPR")
        ax.grid(alpha=0.25);  ax.legend(loc="lower right", fontsize=8)

        # PR
        ax = axes[1]
        if prec_tr is not None:
            ax.plot(rec_tr, prec_tr, color=self.colors["train"], lw=2,
                    label=f"train AP={_fmt(ap_tr)}")
        if self.has_val and prec_va is not None:
            ax.plot(rec_va, prec_va, color=self.colors["val"], lw=2,
                    label=f"val AP={_fmt(ap_va)}")
        if self.has_test and prec_te is not None:
            ax.plot(rec_te, prec_te, color=self.colors["test"], lw=2,
                    label=f"test AP={_fmt(ap_te)}")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision")
        ax.grid(alpha=0.25);  ax.legend(loc="lower left", fontsize=8)

        # Per-metric threshold curves
        items = [
            ("Accuracy",      "acc",    (0.0, 1.0)),
            ("Precision",     "prec",   (0.0, 1.0)),
            ("Recall (TPR)",  "rec",    (0.0, 1.0)),
            ("F1 score",      "f1",     (0.0, 1.0)),
            ("Specificity",   "spec",   (0.0, 1.0)),
            ("Youden's J",    "youden", (-1.0, 1.0)),
        ]
        for ax, (lbl, key, ylim) in zip(axes[2:], items):
            for met, split in [(met_tr, "train"), (met_va, "val"), (met_te, "test")]:
                has = getattr(self, f"has_{split}", split == "train")
                if not has:
                    continue
                vals = met[key]
                j    = _best_idx(vals)
                ax.plot(thr_seq, vals, color=self.colors[split], lw=2,
                        label=f"{split}: t*={_fmt(thr_seq[j])} val={_fmt(vals[j])}")
                if np.isfinite(vals[j]):
                    ax.plot(thr_seq[j], vals[j], "^", color=self.colors[split], ms=6)
            ax.set(xlim=(0, 1), ylim=ylim, ylabel=lbl)
            ax.grid(alpha=0.25);  ax.legend(loc="lower right", fontsize=8)

        axes[-2].set_xlabel("Probability threshold")
        axes[-1].set_xlabel("Probability threshold")

        title = (
            f"ROC-AUC (tr/va/te): {_fmt(auc_tr)}/{_fmt(auc_va)}/{_fmt(auc_te)}  |  "
            f"PR-AUC: {_fmt(ap_tr)}/{_fmt(ap_va)}/{_fmt(ap_te)}"
        )
        fig.suptitle(title, y=0.985, fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(self.run_dir / "02_threshold_sweep.png", dpi=150)
        plt.close(fig)

    # -----------------------------------------------------------------------
    # Calibration curves
    # -----------------------------------------------------------------------

    def _generate_calibration_curves(self, n_bins: int = 10) -> List:
        """
        Build reliability diagram data for all available splits.

        Uses equal-width bins; any bin with fewer than 3 points is
        adaptively merged into the neighbouring bin with fewer samples.

        Parameters
        ----------
        n_bins : int
            Initial number of probability bins in [0, 1].

        Returns
        -------
        List of [split_name, mean_pred, frac_pos, brier] entries,
        or None for splits that are absent.
        """
        splits = {
            "train": True,
            "val"  : self.has_val,
            "test" : self.has_test,
        }
        curves = []

        for key, present in splits.items():
            if not present:
                curves.append(None)
                continue

            y_true, p = self._get_best_probabilities(key)
            brier      = brier_score_loss(y_true, p)

            if p.size < 3:
                print(f"[calibration] Too few points for split '{key}': {p.size}")
                curves.append(None)
                continue

            edges      = list(np.linspace(0.0, 1.0, n_bins + 1))
            idx        = np.digitize(p, edges, right=False) - 1
            bin_indices = [np.where(idx == b)[0].tolist() for b in range(n_bins)]

            MIN_PER_BIN = 3
            while True:
                counts = [len(ix) for ix in bin_indices]
                if len(bin_indices) == 1 or all(c >= MIN_PER_BIN for c in counts):
                    break
                b     = next(i for i, c in enumerate(counts) if c < MIN_PER_BIN)
                left  = b - 1 if b - 1 >= 0 else None
                right = b + 1 if b + 1 < len(bin_indices) else None
                if left is None and right is None:
                    break
                elif left is None:
                    bin_indices[right].extend(bin_indices[b]); del bin_indices[b]
                elif right is None:
                    bin_indices[left].extend(bin_indices[b]);  del bin_indices[b]
                else:
                    if counts[left] <= counts[right]:
                        bin_indices[left].extend(bin_indices[b]); del bin_indices[b]
                    else:
                        bin_indices[b].extend(bin_indices[right]); del bin_indices[right]

            frac_pos, mean_pred = [], []
            for ix in bin_indices:
                if not ix:
                    continue
                frac_pos.append(float((y_true[ix] == 1).mean()))
                mean_pred.append(float(p[ix].mean()))

            curves.append([key, mean_pred, frac_pos, brier])

        return curves

    def _plot_calibration_curves(self) -> None:
        """Plot reliability diagrams for train / val / test."""
        curves  = self._generate_calibration_curves()
        markers = {"train": "o", "val": "s", "test": "^"}

        fig, ax = plt.subplots(figsize=(5.6, 5.6))
        ax.plot([0, 1], [0, 1], ":", lw=2, color="black", label="Perfectly calibrated")

        for curve in curves:
            if curve is None:
                continue
            name, mean_pred, frac_pos, brier = curve
            ax.plot(mean_pred, frac_pos,
                    marker=markers.get(name, "o"),
                    linewidth=1.5,
                    color=self.colors[name],
                    label=f"{name} (Brier={brier:.3f})")

        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Empirical probability")
        ax.set_title("Calibration curves")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(self.run_dir / "03_calibration_curve.png", dpi=150)
        plt.close(fig)

    # -----------------------------------------------------------------------
    # Kernel parameter plots
    # -----------------------------------------------------------------------

    def _plot_kernel_scaling(self) -> None:
        """
        Plot eta and ARD scale trajectories over iterations.

        Only produced when at least one of eta_flag or ard_flag is
        True.  The x-axis is clipped to the actual last logged step so
        early-stopped runs are not shown with a large blank region on the right.
        A dashed line marks the best iteration and a dotted red line marks the
        early-stopping trigger point when applicable.
        """
        if not (self.eta_flag or self.ard_flag):
            return

        steps = [l.step for l in self.run_log.logs]
        x_max = max(steps) if steps else self.maxiter

        fig, ax = plt.subplots(figsize=(8, 4.5))

        if self.eta_flag:
            etas = [l.eta if l.eta is not None else np.nan for l in self.run_log.logs]
            ax.plot(steps, etas, lw=2, color="black", label=r"$\eta$")

        if self.ard_flag:
            for k in range(self.nf):
                vals = [
                    (l.ard[k] if (l.ard is not None and len(l.ard) > k) else np.nan)
                    for l in self.run_log.logs
                ]
                ax.plot(steps, vals, lw=2, label=f"ARD[{k}]")

        # Mark best iteration
        if self._best_iter is not None:
            ax.axvline(self._best_iter, color="black", linestyle="--",
                       linewidth=1, alpha=0.5, label=f"best={self._best_iter}")

        # Mark early-stopping trigger point
        if self._es_stopped and steps and max(steps) < self.maxiter:
            ax.axvline(max(steps), color="red", linestyle=":",
                       linewidth=1.2, alpha=0.7, label=f"ES@{max(steps)}")

        ax.set_xlabel("Optimisation step")
        ax.set_ylabel("Hyperparameter value")
        ax.set_xlim(0, x_max)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(self.run_dir / "05_kernel_parameters.png", dpi=150)
        plt.close(fig)

    def _plot_kernel_W(self) -> None:
        """
        Plot the evolution of spatial filter weights over iterations.

        One subplot per filter column; each subplot overlays all s channel
        traces.  A thick line shows the per-step median across channels.
        Handles nf == 1 and nf > 1 uniformly.

        The x-axis right edge is the actual last logged step, not the full
        iteration budget, so early-stopped runs do not show a blank region.
        A dashed line marks the best iteration and a dotted red line marks the
        early-stopping trigger point when applicable.
        """
        if not (self.run_log and self.run_log.logs):
            return

        logs  = self.run_log.logs
        steps = np.asarray([l.step for l in logs], dtype=float)
        T     = steps.size
        x_min = float(steps.min()) if T else 0.0
        x_max = float(steps.max()) if T else float(self.maxiter)

        # Normalise stored W values to shape (T, s, nf)
        Ws_list = []
        for l in logs:
            W = np.asarray(l.W, dtype=float)
            if W.ndim == 1:
                W = W.reshape(-1, 1) if self.nf == 1 else np.full((self.s, self.nf), np.nan)
            elif W.ndim != 2:
                W = np.full((self.s, self.nf), np.nan)

            # Repair shape mismatches via reshape / pad / slice
            if W.shape != (self.s, self.nf):
                if W.size == self.s * self.nf:
                    W = W.reshape(self.s, self.nf)
                else:
                    W = np.full((self.s, self.nf), np.nan)

            Ws_list.append(W)

        Ws = np.stack(Ws_list, axis=0)  # (T, s, nf)

        fig, axes = plt.subplots(1, self.nf, figsize=(max(5, 5 * self.nf), 4))
        axes = [axes] if self.nf == 1 else list(axes)

        for k, ax in enumerate(axes):
            ax.plot(steps, Ws[:, :, k], lw=1.2, alpha=0.9)
            with np.errstate(invalid="ignore"):
                ax.plot(steps, np.nanmedian(Ws[:, :, k], axis=1), lw=2.0)

            # Mark best iteration
            if self._best_iter is not None:
                ax.axvline(self._best_iter, color="black", linestyle="--",
                           linewidth=1, alpha=0.5,
                           label=f"best={self._best_iter}" if k == 0 else None)

            # Mark early-stopping trigger point
            if self._es_stopped and T and x_max < self.maxiter:
                ax.axvline(x_max, color="red", linestyle=":",
                           linewidth=1.2, alpha=0.7,
                           label=f"ES@{int(x_max)}" if k == 0 else None)

            ax.set_xlabel("Optimisation step")
            ax.set_ylabel(f"W[:, {k}]")
            ax.set_title(f"Filter {k}")
            ax.set_xlim(x_min, x_max)
            ax.grid(lw=0.4, alpha=0.3)
            if k == 0 and (self._best_iter is not None or self._es_stopped):
                ax.legend(fontsize=7)

        fig.tight_layout()
        fig.savefig(self.run_dir / "06_kernel_W.png", dpi=150)
        plt.close(fig)

    # -----------------------------------------------------------------------
    # Topomap
    # -----------------------------------------------------------------------

    def _step_to_log_idx(self, step: int) -> int:
        """
        Map an absolute step number to the index of the closest log entry.

        Because logs are written once per block (every log_every steps),
        the step values stored in self.run_log.logs are multiples of
        log_every (e.g. 10, 20, 30, …) rather than consecutive integers.
        A direct step - 1 offset therefore gives the wrong index.

        This helper finds the log entry whose step field is closest to
        the requested value, falling back to the last entry on out-of-range
        inputs.

        Parameters
        ----------
        step : int
            Absolute step number (as stored in IterLog.step).

        Returns
        -------
        int
            Index into self.run_log.logs.
        """
        if self.run_log is None or not self.run_log.logs:
            return 0
        steps = [l.step for l in self.run_log.logs]
        # Find the index of the entry whose step is closest to the requested value
        idx = int(np.argmin(np.abs(np.array(steps) - step)))
        return idx

    def _retrieve_spatial_filter(self, f: int, iter: int) -> np.ndarray:
        """
        Retrieve a single spatial filter column from the logged W matrices.

        Parameters
        ----------
        f : int
            Filter index (column of W).
        iter : int
            Step number (as stored in IterLog.step), not a 1-based count
            of log entries.  Use self._best_iter to get the best snapshot.

        Returns
        -------
        np.ndarray, shape (s,)
        """
        if self.run_log is None or not self.run_log.logs:
            return np.zeros(self.s, dtype=float)

        Ws  = np.array([l.W for l in self.run_log.logs])   # (T, s, nf)
        idx = self._step_to_log_idx(iter)
        if not (0 <= f < Ws.shape[2]):
            return np.zeros(self.s, dtype=float)
        return Ws[idx, :, f]

    def _plot_topomap(
        self,
        iter: Optional[int] = None,
        weights: Optional[List[np.ndarray]] = None,
        fs: Optional[List[int]] = None,
    ) -> None:
        """
        Plot spatial filter columns as EEG topomaps (requires MNE).

        When called with no arguments, all nf filters at the best
        iteration are plotted.  The method calls itself recursively once
        weights are resolved so the plotting code is shared.

        Parameters
        ----------
        iter : int, optional
            Override for the best iteration index.
        weights : list of np.ndarray, optional
            Pre-computed weight vectors.  If provided, plotted directly.
        fs : list of int, optional
            Filter indices to plot from a specific iteration (only when
            weights is None).
        """
        if not HAS_MNE:
            return

        iter_to_use = iter if iter is not None else self._best_iter

        if weights is None:
            all_fs = fs if (fs is not None and len(fs) > 0) else list(range(self.nf))
            w_list = [self._retrieve_spatial_filter(f=f, iter=iter_to_use) for f in all_fs]
            return self._plot_topomap(iter=iter_to_use, weights=w_list, fs=all_fs)

        cols = [np.asarray(w, dtype=float).ravel() for w in weights]
        W_t  = np.column_stack(cols) if cols else np.zeros((self.s, 0))
        k    = W_t.shape[1]
        if k == 0:
            return

        t    = iter_to_use if (iter_to_use is not None and iter_to_use <= self.maxiter) else "?"
        fig, axes = plt.subplots(1, k, figsize=(4 * k, 4))
        axes = [axes] if k == 1 else list(axes)

        for i, ax in enumerate(axes):
            mne.viz.plot_topomap(W_t[:, i], self.montage_info,
                                 axes=ax, show=False, sphere=1.2)
            label_idx = (fs[i] if (fs is not None and i < len(fs)) else i)
            ax.set_title(f"W[:, {label_idx}]  Iter {t}")

        fig.tight_layout()
        fig.savefig(self.run_dir / "08_topomaps.png", dpi=150)
        plt.close(fig)

    # -----------------------------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------------------------

    def _plot_confusion_matrix(self) -> None:
        """
        Plot confusion matrices for all available splits side by side.

        The threshold self.pred_threshold is used to derive predicted
        labels from the stored best-iteration probabilities.
        """
        splits = [("train", True), ("val", self.has_val), ("test", self.has_test)]
        cms, labels = [], []

        for split, has in splits:
            if not has:
                continue
            y, y_pred = self._get_best_predictions(split)
            cms.append(confusion_matrix(y, y_pred))
            labels.append(split)

        k    = len(cms)
        fig, axes = plt.subplots(1, k, figsize=(4 * k, 4))
        axes = [axes] if k == 1 else list(axes)

        for ax, cm, lbl in zip(axes, cms, labels):
            ax.imshow(cm, cmap="Greens", vmin=0, vmax=cm.max())
            title = f"{lbl} (Iter {self._best_iter})" if lbl == "train" else lbl
            ax.set_title(title, fontsize=9)
            ax.set_xlabel(f"Predicted P(y=1) > {self.pred_threshold}", fontsize=9)
            ax.set_ylabel("True", fontsize=9)
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, int(v), ha="center", va="center", fontsize=9)

        fig.tight_layout()
        fig.savefig(self.run_dir / "09_confusion_matrix.png", dpi=150)
        plt.close(fig)

    # -----------------------------------------------------------------------
    # Feature scatter + decision boundary (nf == 2 only)
    # -----------------------------------------------------------------------

    def _compute_feature(self, f: int, iter: int) -> Dict[str, np.ndarray]:
        """
        Compute the scalar feature z_f = w_f^T Σ w_f for all splits.

        Mirrors the feature computation used inside the kernel, including the
        optional log transform and ARD scaling, so the scatter plots are in
        the same space as the kernel.

        Parameters
        ----------
        f : int
            Spatial filter index.
        iter : int
            Iteration at which to retrieve w_f.

        Returns
        -------
        dict with keys "train", optionally "val" and "test".
        """
        iter_idx = self._step_to_log_idx(iter)
        w        = self._retrieve_spatial_filter(f=f, iter=iter).astype(float).ravel()

        def _on_split(X_flat):
            if X_flat is None:
                return None
            X_flat = np.asarray(X_flat, dtype=float)
            Sigma  = X_flat.reshape(X_flat.shape[0], self.s, self.s)
            Sw     = np.tensordot(Sigma, w, axes=([2], [0]))  # (N, s)
            wSw    = np.sum(Sw * w[None, :], axis=1)          # (N,)
            if self.logged_flag:
                wSw = np.log(np.maximum(wSw, 1e-12))
            try:
                if self.ard_flag and self.run_log is not None:
                    ard_vec = self.run_log.logs[iter_idx].ard
                    if ard_vec is not None and len(ard_vec) > f:
                        wSw = wSw * np.exp(ard_vec[f])
            except Exception:
                pass
            return wSw.astype(float)

        out = {"train": _on_split(self.X_train)}
        if self.has_val:
            out["val"]  = _on_split(self.X_val)
        if self.has_test:
            out["test"] = _on_split(self.X_test)
        return out

    def _compute_decision_boundary(self, iter: int) -> Dict[str, Any]:
        """
        Interpolate a 2D decision surface over the first two feature dimensions.

        Parameters
        ----------
        iter : int
            Iteration at which to compute features.

        Returns
        -------
        dict with keys "XX", "YY", "ZZ" (meshgrid + surface),
        "f1", "f2" (feature indices), and per-split raw features.
        Returns an empty dict when the feature computation fails.
        """
        f1, f2   = 0, 1
        fX_dict  = self._compute_feature(f=f1, iter=iter)
        fY_dict  = self._compute_feature(f=f2, iter=iter)

        fX = np.asarray(fX_dict["train"], dtype=float).ravel()
        fY = np.asarray(fY_dict["train"], dtype=float).ravel()
        if fX.size == 0:
            return {}

        pad_x = 0.05 * (fX.max() - fX.min() + 1e-12)
        pad_y = 0.05 * (fY.max() - fY.min() + 1e-12)
        XX, YY = np.meshgrid(
            np.linspace(fX.min() - pad_x, fX.max() + pad_x, 300),
            np.linspace(fY.min() - pad_y, fY.max() + pad_y, 300),
        )

        _, p = self._get_best_probabilities("train")
        pts  = np.c_[fX, fY]
        try:
            ZZ = griddata(pts, p, (XX, YY), method="cubic")
        except Exception:
            ZZ = griddata(pts, p, (XX, YY), method="linear")

        nan_mask = np.isnan(ZZ)
        if np.any(nan_mask):
            ZZ[nan_mask] = griddata(pts, p, (XX[nan_mask], YY[nan_mask]), method="nearest")

        return {
            "XX": XX, "YY": YY, "ZZ": ZZ,
            "f1": f1, "f2": f2,
            "fX_train": fX,         "fY_train": fY,
            "fX_test" : fX_dict.get("test"), "fY_test": fY_dict.get("test"),
        }

    def _add_decision_boundary(
        self,
        iter: int,
        levels: Optional[List[float]] = None,
    ) -> None:
        """
        Overlay contour lines of the decision surface on the current axes.

        The contour at self.pred_threshold is drawn in black (thick);
        additional levels (e.g. 0.1, 0.9) are drawn in grey dashed.

        Parameters
        ----------
        iter : int
        levels : list of float, optional
            Probability levels to draw.  Defaults to [pred_threshold, 0.1, 0.9].
        """
        if levels is None:
            levels = [self.pred_threshold, 0.1, 0.9]

        boundary = self._compute_decision_boundary(iter=iter)
        if not boundary:
            return

        XX, YY, ZZ = boundary["XX"], boundary["YY"], boundary["ZZ"]
        ax  = plt.gca()
        thr = self.pred_threshold

        if thr in levels:
            cs = ax.contour(XX, YY, ZZ, levels=[thr], linewidths=2.0, colors="black")
            ax.clabel(cs, fmt={thr: f"p={thr:.2f}"}, inline=True, fontsize=8)
            other = [lv for lv in levels if lv != thr]
        else:
            other = list(levels)

        if other:
            ax.contour(XX, YY, ZZ, levels=other, linewidths=1.0,
                       colors="grey", linestyles="--")

    def _plot_features_and_boundary(self) -> None:
        """
        Scatter the selected 2D feature pair for each split and overlay the
        decision boundary.

        Only executed when self.nf == 2.  Uses probabilities from the
        best iteration for the boundary interpolation.
        """
        iter     = self._best_iter
        boundary = self._compute_decision_boundary(iter=iter)
        if not boundary:
            return

        f1, f2   = boundary["f1"], boundary["f2"]
        fX_train = boundary["fX_train"]
        fY_train = boundary["fY_train"]
        fX_test  = boundary.get("fX_test")
        fY_test  = boundary.get("fY_test")

        fig, ax = plt.subplots(figsize=(6, 5))

        # Train scatter
        ax.scatter(
            fX_train, fY_train,
            c=["orange" if y == 0 else "navy" for y in self.Y_train.ravel()],
            s=44, marker="o", linewidth=0.4, alpha=0.2,
        )
        handles = [plt.Line2D([0], [0], marker="o", color="w",
                               markerfacecolor="k", markersize=8, alpha=0.2, label="Train")]

        # Test scatter
        if self.has_test and fX_test is not None:
            ax.scatter(
                fX_test, fY_test,
                c=["orange" if y == 0 else "navy" for y in self.Y_test.ravel()],
                s=22, marker="^", linewidth=0.4, alpha=1.0,
            )
            handles.append(plt.Line2D([0], [0], marker="^", color="w",
                                       markerfacecolor="k", markersize=6, alpha=1.0, label="Test"))

        self._add_decision_boundary(iter=iter, levels=[self.pred_threshold, 0.1, 0.9])

        x_lbl = rf"$w_{f1}^T \Sigma w_{f1}$"
        y_lbl = rf"$w_{f2}^T \Sigma w_{f2}$"
        if self.logged_flag:
            x_lbl = f"log({x_lbl})";  y_lbl = f"log({y_lbl})"

        ax.set_xlabel(x_lbl);  ax.set_ylabel(y_lbl)
        ax.set_title(f"Iter {iter}", fontsize=9)
        ax.legend(handles=handles, loc="best", fontsize=8, frameon=True)
        fig.tight_layout()
        fig.savefig(self.run_dir / f"10_features_and_boundary_{self.feature_pair}.png", dpi=150)
        plt.close(fig)

    # -----------------------------------------------------------------------
    # Singular values
    # -----------------------------------------------------------------------

    def _compute_features_matrix(self, iter: int) -> np.ndarray:
        """
        Stack per-filter features into a matrix of shape (N_train, nf).

        Parameters
        ----------
        iter : int

        Returns
        -------
        np.ndarray, shape (N_train, nf)
        """
        mat = np.zeros((self.N_train, self.nf))
        for f in range(self.nf):
            mat[:, f] = self._compute_feature(f=f, iter=iter)["train"]
        return mat

    def _plot_sv(self, iter: Optional[int] = None) -> None:
        """
        Plot the singular values of the training feature matrix.

        Useful for diagnosing the effective dimensionality of the feature
        space at the best iteration.

        Parameters
        ----------
        iter : int, optional
            Iteration to use.  Defaults to self._best_iter.
        """
        iter = iter if iter is not None else self._best_iter
        mat  = self._compute_features_matrix(iter=iter)

        try:
            s = svd(mat, full_matrices=True, compute_uv=False)
        except Exception:
            return

        s  = np.sort(s)[::-1]
        xs = np.arange(1, len(s) + 1)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(xs, s, marker="o", color="black",
                markerfacecolor="gold", markeredgecolor="black", lw=1.5)
        ax.set_xlim(xs.min() - 0.5, xs.max() + 0.5)
        ax.set_xlabel("Index")
        ax.set_ylabel("Singular value")
        ax.set_title(f"Feature matrix singular values -- Iter {iter}", fontsize=9)
        fig.tight_layout()
        fig.savefig(self.run_dir / "16_singular_values.png", dpi=150)
        plt.close(fig)
