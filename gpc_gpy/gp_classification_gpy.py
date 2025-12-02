"""
GP classification with a custom kernel in GPy using EP approximation on covariance matrices
"""

from __future__ import annotations

# ---------------------- Imports ----------------------
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)  # runtime support for type hints


# ---------------------- Third party libraries (mandatory) ----------------------
import matplotlib.pyplot as plt
import numpy as np
import GPy
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from kernels_gpy import CustomKernelGPy  # custom covariance function

# ---------------------- Third party libraries (optionals for extra glitters) ----------------------
try:
    import mne  # handles EEG specific objects like montage

    HAS_MNE = True
except Exception:
    HAS_MNE = False


# ---------------------- Typing aliases ----------------------
# Define a type hint which could be an array or a dict
ArrayOrDict = Union[np.ndarray, Dict[str, np.ndarray]]


# ---------------------- Small helpers ----------------------
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
    import datetime as _dt
    if mode == "nice":
        return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------- Logging structs ----------------------
@dataclass
class IterLog:
    """Metrics for one training iteration."""

    step: int
    nlml: float
    nlpd_train: Optional[float]   # negative log probability density
    nlpd_val: Optional[float]
    nlpd_test: Optional[float]
    acc_train: Optional[float]    # accuracy
    acc_val: Optional[float]
    acc_test: Optional[float]
    aucroc_train: Optional[float] # area under the curve ROC
    aucroc_val: Optional[float]
    aucroc_test: Optional[float]
    aucpr_train: Optional[float]  # area under the curve precision-recall
    aucpr_val: Optional[float]
    aucpr_test: Optional[float]
    brier_train: Optional[float]  # brier's score
    brier_val: Optional[float]
    brier_test: Optional[float]


@dataclass
class RunLog:
    """
    Container for the entire run logs, converted to JSON format
    """

    meta: Dict[str, Any]
    logs: List[IterLog]


class GPClassificationRunner:
    """
    Handles end-to-end binary classification using GPy and a custom kernel.
    Responsible for data handling, model building, training, and evaluation.
    """

    def __init__(
        self,
        # Input variables
        X: ArrayOrDict,
        Y: ArrayOrDict,
        dataset_label: str,
        ch_names: List[str],  # Names of EEG channels
        ch_xy: Dict[str, Tuple[float, float]],  # Coordinates of EEG channels
        # Model / kernel
        spatialFilter_init: str = "random",  # 'random' | 'ones' | 'focused'
        nf: int = 2,  # Number of spatial filter cols
        eta_flag: bool = False,
        ard_flag: bool = False,
        logged_flag: bool = True,
        kernel_type: str = "RBF",
        # Training
        maxiter: int = 50,  # EP steps
        frac_val: float = 0.0,
        frac_test: float = 0.0,
        random_state: int = 0,
        # Storage
        results_dir: str = "./results",
        run_name: Optional[str] = None,
    ) -> None:
        self.dataset_label = dataset_label
        self.ch_names = ch_names
        self.ch_xy = ch_xy
        self.spatialFilter_init = spatialFilter_init
        self.nf = int(nf)
        self.eta_flag = bool(eta_flag)
        self.ard_flag = bool(ard_flag)
        self.logged_flag = bool(logged_flag)
        self.kernel_type = kernel_type

        self.maxiter = int(maxiter)
        self.frac_val = float(frac_val)
        self.frac_test = float(frac_test)
        self.random_state = int(random_state)

        # Store Run naming / Logging
        self.results_root = Path(results_dir)
        self.run_name = run_name or f"run_{_now_stamp()}"
        self.run_dir = self.results_root / self.run_name
        _ensure_dir(self.run_dir)  # Create folder

        # Placeholders updated by `_load_and_prepare_data`
        self.has_train = False
        self.has_val = False
        self.has_test = False

        # Optional training flags
        self.use_validation_for_adaptation: bool = False

        self.X_train: Optional[np.ndarray] = None  # (N_train, D)
        self.X_val: Optional[np.ndarray] = None  # (N_val, D)
        self.X_test: Optional[np.ndarray] = None  # (N_test, D)
        self.Y_train: Optional[np.ndarray] = None  # (N_train, 1)
        self.Y_val: Optional[np.ndarray] = None  # (N_val, 1)
        self.Y_test: Optional[np.ndarray] = None  # (N_test, 1)

        # Data unpack and reshape
        self.X = X
        self.Y = Y
        self.s: Optional[int] = None  # set after reading X

        # Kernel/model state
        self.W_init: Optional[np.ndarray] = None
        self.W_trainable: bool = True
        self.kernel: Optional[CustomKernelGPy] = None
        self.model: Optional[Any] = None

        # Logs
        self.logs: List[IterLog] = []
        self.best_step: int = 0

        # Prepare data and initialize W
        self._load_and_prepare_data()
        self._initialize_W_matrix()

    # ---------------------- Data handling ----------------------
    def _flatten_if_needed(self, X: np.ndarray) -> np.ndarray:
        """
        Ensure X is 2D: (N, D). If X is (N, s, s), flatten to (N, s*s) to meet model input requirement.
        """
        if X.ndim == 3:
            N, s1, s2 = X.shape
            if s1 != s2:
                raise ValueError(f"Covariance matrices must be square: got {s1}x{s2}")
            return X.reshape(N, s1 * s2)
        elif X.ndim == 2:
            return X
        else:
            raise ValueError("X must be 2D or 3D array")

    def _ensure_label_shape(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y.shape[1] != 1:
            raise ValueError("Labels must be of shape (N,) or (N,1)")
        return y.astype(int)

    def _split_arrays(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Split numpy arrays into train/val/test according to fractions.
        """
        X = self._flatten_if_needed(X)
        Y = self._ensure_label_shape(Y)

        N = X.shape[0]
        if not (0 <= self.frac_val < 1 and 0 <= self.frac_test < 1):
            raise ValueError("frac_val and frac_test must be in [0,1)")
        if self.frac_val + self.frac_test >= 1:
            raise ValueError("frac_val + frac_test must be < 1")

        N_test = int(round(self.frac_test * N))
        N_val = int(round(self.frac_val * (N - N_test)))

        idx = np.arange(N)
        rng = np.random.default_rng(self.random_state)
        rng.shuffle(idx)

        test_idx = idx[:N_test]
        remain = idx[N_test:]
        val_idx = remain[:N_val]
        train_idx = remain[N_val:]

        Xtr, Ytr = X[train_idx], Y[train_idx]
        Xva, Yva = X[val_idx], Y[val_idx]
        Xte, Yte = X[test_idx], Y[test_idx]

        return Xtr, Ytr, Xva, Yva, Xte, Yte

    def _load_and_prepare_data(self) -> None:
        """
        Accept dicts with keys {train,val,test} or arrays. Always flatten covariances.
        """
        X = self.X
        Y = self.Y

        if isinstance(X, dict) and isinstance(Y, dict):
            Xtr = X.get("train")
            Xva = X.get("val")
            Xte = X.get("test")
            Ytr = Y.get("train")
            Yva = Y.get("val")
            Yte = Y.get("test")

            self.s = Xtr.shape[-1]

            if Xtr is not None:
                Xtr = self._flatten_if_needed(Xtr)
                Ytr = self._ensure_label_shape(Ytr)
                self.X_train, self.Y_train = Xtr, Ytr
                self.has_train = True

            if Xva is not None:
                Xva = self._flatten_if_needed(Xva)
                Yva = self._ensure_label_shape(Yva)
                self.X_val, self.Y_val = Xva, Yva
                self.has_val = True

            if Xte is not None:
                Xte = self._flatten_if_needed(Xte)
                Yte = self._ensure_label_shape(Yte)
                self.X_test, self.Y_test = Xte, Yte
                self.has_test = True

            if not self.has_train:
                raise ValueError("At least training set must be provided in dict form.")

        elif isinstance(X, np.ndarray) and isinstance(Y, np.ndarray):
            self.s = X.shape[-1] if X.ndim == 3 else int(round(np.sqrt(X.shape[-1])))
            Xtr, Ytr, Xva, Yva, Xte, Yte = self._split_arrays(X, Y)
            self.X_train, self.Y_train = Xtr, Ytr
            self.X_val, self.Y_val = Xva, Yva
            self.X_test, self.Y_test = Xte, Yte
            self.has_train = True
            self.has_val = Xva.size > 0
            self.has_test = Xte.size > 0
        else:
            raise TypeError("X and Y must be both dicts or both numpy arrays")

    # ---------------------- W initialization ----------------------
    def _initialize_W_matrix(self) -> None:
        """
        Initialize the spatial filter matrix W according to configuration.
        If a matrix is provided, W is fixed. Otherwise initialize per policy and trainable.
        """
        rng = np.random.default_rng(self.random_state)

        if isinstance(self.spatialFilter_init, np.ndarray):
            W_arr = np.asarray(self.spatialFilter_init, dtype=np.float64)
            if W_arr.ndim != 2:
                raise ValueError("spatialFilter_init array must be 2D with shape (s, nf)")
            if W_arr.shape[0] != self.s:
                raise ValueError(f"spatialFilter_init has {W_arr.shape[0]} rows, expected s={self.s}")
            if W_arr.shape[1] != self.nf:
                raise ValueError(f"spatialFilter_init has {W_arr.shape[1]} cols, expected nf={self.nf}")
            self.W_init = W_arr.copy()
            self.W_trainable = False
        else:
            self.W_trainable = True

            if self.spatialFilter_init == "random":
                self.W_init = rng.normal(loc=0.0, scale=0.1, size=(self.s, self.nf))
            elif self.spatialFilter_init == "ones":
                self.W_init = np.ones((self.s, self.nf), dtype=np.float64)
            elif self.spatialFilter_init == "focused":
                self.W_init = np.zeros((self.s, self.nf))

                if self.nf > 2:
                    print(
                        "Warning: More than 2 spatial filters initialized, `focused` currently needs fixing!"
                    )

                def _idx_if_present(name: str) -> Optional[int]:
                    return self.ch_names.index(name) if name in self.ch_names else None

                for cname in ["C3", "Cz", "C4", "CP3", "CPz", "CP4"]:
                    idx = _idx_if_present(cname)
                    if idx is not None and self.nf > 0:
                        self.W_init[idx, 0] = 1.0
                if self.nf > 1:
                    for cname in ["FC3", "FCz", "FC4"]:
                        idx = _idx_if_present(cname)
                        if idx is not None:
                            self.W_init[idx, 1] = 1.0
            else:
                raise ValueError(
                    "Unknown spatialFilter_init. Use 'random', 'ones', 'focused', or an ndarray."
                )

    # ---------------------- Model building ----------------------
    def _build_model_kernel(self) -> None:
        """Build the CustomKernelGPy object using the initialized W."""
        if self.X_train is None or self.Y_train is None:
            raise RuntimeError("Data must be prepared before building the model")

        s = self.s
        if len(self.ch_names) != s:
            raise ValueError(f"len(ch_names)={len(self.ch_names)} does not match s={s} from data")
        if self.W_init.shape[0] != s:
            raise ValueError(f"W_init has wrong shape {self.W_init.shape}, expected (s, nf)")

        self.kernel = CustomKernelGPy(
            W=self.W_init,
            W_trainable=self.W_trainable,
            ard_flag=self.ard_flag,
            eta_flag=self.eta_flag,
            logged_flag=self.logged_flag,
            kernel_type=self.kernel_type,
        )

    def _build_model(self) -> None:
        """Build a GPy GPClassification model with EP inference and CustomKernelGPy."""
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

    # ---------------------- Prediction helpers ----------------------
    def _predict_prob(self, model: Any, X: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Predict p(y=1|x) from model; returns None if X is None."""
        if X is None:
            return None
        mu, _ = model.predict(X)
        return mu.ravel()

    def _metrics_on_split(self, y_true: Optional[np.ndarray], p: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
        """Compute metrics on one split given true labels and predicted probabilities."""
        metrics: Dict[str, Optional[float]] = {
            "acc": None,
            "aucroc": None,
            "aucpr": None,
            "brier": None,
            "nlpd": None,
        }

        if y_true is None or p is None:
            return metrics

        y_true = y_true.astype(int).ravel()
        p = np.clip(p, 1e-8, 1 - 1e-8)

        try:
            y_hat = (p >= 0.5).astype(int)
            metrics["acc"] = float(accuracy_score(y_true, y_hat))
        except Exception:
            metrics["acc"] = None

        try:
            metrics["brier"] = float(brier_score_loss(y_true, p))
        except Exception:
            metrics["brier"] = None

        try:
            metrics["aucroc"] = float(roc_auc_score(y_true, p))
        except Exception:
            metrics["aucroc"] = None

        # PR area via Average Precision (standard)
        try:
            metrics["aucpr"] = float(average_precision_score(y_true, p))
        except Exception:
            metrics["aucpr"] = None

        # Simple Bernoulli negative log-likelihood
        try:
            metrics["nlpd"] = float(-(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)).mean())
        except Exception:
            metrics["nlpd"] = None

        return metrics

    def _snapshot_kernel_params(self) -> Dict[str, Any]:
        assert self.kernel is not None
        W_list = self.kernel.W.tolist()
        eta_val = float(self.kernel.eta)
        ard_vec = self.kernel.ard.tolist() if self.ard_flag else None
        return {"W": W_list, "eta": eta_val, "ard": ard_vec}

    def _snapshot_iteration(
        self,
        p_train_at_iter: Optional[np.ndarray],
        p_val_at_iter: Optional[np.ndarray],
        p_test_at_iter: Optional[np.ndarray],
    ) -> None:
        assert self.model is not None

        # EP objective as negative log marginal likelihood
        nlml_ = float(-self.model.log_likelihood())

        train_metrics = self._metrics_on_split(self.Y_train, p_train_at_iter)
        val_metrics = self._metrics_on_split(self.Y_val, p_val_at_iter)
        test_metrics = self._metrics_on_split(self.Y_test, p_test_at_iter)

        log = IterLog(
            step=self.step,
            nlml=nlml_,
            nlpd_train=train_metrics["nlpd"],
            nlpd_val=val_metrics["nlpd"],
            nlpd_test=test_metrics["nlpd"],
            acc_train=train_metrics["acc"],
            acc_val=val_metrics["acc"],
            acc_test=test_metrics["acc"],
            aucroc_train=train_metrics["aucroc"],
            aucroc_val=val_metrics["aucroc"],
            aucroc_test=test_metrics["aucroc"],
            aucpr_train=train_metrics["aucpr"],
            aucpr_val=val_metrics["aucpr"],
            aucpr_test=test_metrics["aucpr"],
            brier_train=train_metrics["brier"],
            brier_val=val_metrics["brier"],
            brier_test=test_metrics["brier"],
        )
        self.logs.append(log)

    def _selection_metric(self, last: IterLog) -> Tuple[str, Optional[float]]:
        """
        Decide which metric defines the `best` model.
        If validation is available and `use_validation_for_adaptation` is True, select on nlpd_val; otherwise nlml.
        """
        if self.use_validation_for_adaptation and self.has_val:
            name, value = "nlpd_val", last.nlpd_val
        else:
            name, value = "nlml", last.nlml
        return name, value

    def _check_for_best_iteration(
        self,
        p_train_at_iter: Optional[np.ndarray] = None,
        p_val_at_iter: Optional[np.ndarray] = None,
        p_test_at_iter: Optional[np.ndarray] = None,
    ) -> None:
        self._snapshot_iteration(p_train_at_iter, p_val_at_iter, p_test_at_iter)
        last = self.logs[-1]
        name, value = self._selection_metric(last)

        if value is None:
            return

        if len(self.logs) == 1 or value < getattr(self, f"best_{name}", np.inf):
            setattr(self, f"best_{name}", value)
            self.best_step = self.step
            self.best_kernel_state = self._snapshot_kernel_params()

    # ---------------------- Training ----------------------
    def train(self) -> None:
        if self.model is None:
            raise RuntimeError("Model must be built before training")

        # Loss function: negative log marginal likelihood by default
        self.loss_fn = lambda m: float(-m.log_likelihood())

        self.logs = []
        print_fr = max(1, self.maxiter // 10)

        for self.step in range(1, self.maxiter + 1):
            self.model.optimize(optimizer="scg", messages=False, max_iters=1)

            p_train = self._predict_prob(self.model, self.X_train)
            p_val = self._predict_prob(self.model, self.X_val) if self.has_val else None
            p_test = self._predict_prob(self.model, self.X_test) if self.has_test else None

            self._check_for_best_iteration(p_train, p_val, p_test)

            if self.step % print_fr == 0 or self.step == 1 or self.step == self.maxiter:
                last = self.logs[-1]
                print(
                    f"step {self.step:03d} | nlml={last.nlml:.3f} | "
                    f"acc_tr={last.acc_train if last.acc_train is not None else np.nan:.3f} | "
                    f"acc_va={last.acc_val if last.acc_val is not None else np.nan:.3f} | "
                    f"acc_te={last.acc_test if last.acc_test is not None else np.nan:.3f}"
                )

        # Restore best parameters if tracked
        if hasattr(self, "best_kernel_state"):
            state = self.best_kernel_state
            assert self.kernel is not None
            self.kernel.W[:] = np.asarray(state["W"])  # type: ignore[index]

    # ---------------------- Public API ----------------------
    def fit(self) -> "GPClassificationRunner":
        self._build_model()
        self.train()
        return self

    def predict_proba(self, which: str = "test") -> Optional[np.ndarray]:
        if self.model is None:
            raise RuntimeError("Call fit() before predict_proba().")
        if which == "train":
            return self._predict_prob(self.model, self.X_train)
        if which == "val":
            return self._predict_prob(self.model, self.X_val)
        if which == "test":
            return self._predict_prob(self.model, self.X_test)
        raise ValueError("which must be one of {'train','val','test'}")

    def to_json(self) -> Dict[str, Any]:
        meta = {
            "dataset": self.dataset_label,
            "channels": self.ch_names,
            "s": self.s,
            "nf": self.nf,
            "flags": {
                "eta": self.eta_flag,
                "ard": self.ard_flag,
                "logged": self.logged_flag,
                "kernel": self.kernel_type,
                "W_trainable": self.W_trainable,
            },
            "splits": {
                "has_train": self.has_train,
                "has_val": self.has_val,
                "has_test": self.has_test,
            },
            "steps": self.maxiter,
        }
        return {
            "meta": meta,
            "logs": [log.__dict__ for log in self.logs],
        }


if __name__ == "__main__":
    print("This module defines GPClassificationRunner. Import and use in your pipeline.")
