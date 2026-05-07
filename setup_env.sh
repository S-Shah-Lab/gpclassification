#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — create a conda environment for GP-BCI
#
# Usage:
#   bash setup_env.sh            # creates env named "gp_bci"
#   bash setup_env.sh myenv      # creates env with a custom name
#
# After the script completes:
#   conda activate gp_bci
#   python run_class_bci_competition_III_merged_spatial_filters_gpy.py --help
# =============================================================================

set -euo pipefail

ENV_NAME="${1:-gp_bci}"

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
GREEN="\033[0;32m"; YELLOW="\033[0;33m"; RED="\033[0;31m"; RESET="\033[0m"
info()    { echo -e "${GREEN}[INFO]${RESET}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
section() { echo -e "\n${GREEN}========== $* ==========${RESET}"; }

# ---------------------------------------------------------------------------
# 0. Sanity checks
# ---------------------------------------------------------------------------
section "Pre-flight checks"

if ! command -v conda &>/dev/null; then
    echo -e "${RED}[ERROR]${RESET} conda not found. Install Miniconda or Anaconda first."
    echo "        https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

CONDA_VERSION=$(conda --version 2>&1 | awk '{print $2}')
info "conda $CONDA_VERSION detected"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    warn "Environment '$ENV_NAME' already exists."
    read -rp "  Remove and recreate it? [y/N] " REPLY
    if [[ "$REPLY" =~ ^[Yy]$ ]]; then
        info "Removing existing environment '$ENV_NAME'..."
        conda env remove -n "$ENV_NAME" -y
    else
        warn "Aborting. Activate the existing env with:  conda activate $ENV_NAME"
        exit 0
    fi
fi

# ---------------------------------------------------------------------------
# 1. Version constraints — rationale
# ---------------------------------------------------------------------------
#
# GPy 1.13.2 (pip-only) imposes two hard ceilings that drive all other pins:
#
#   numpy  < 2.0.0    →  numpy==1.26.4   (last 1.x release, long-term stable)
#   scipy  <= 1.12.0  →  scipy==1.12.0   (GPy ceiling; mne>=1.8 requires >=1.9 ✓)
#
# Everything else is pinned to the newest version compatible with those two.
#
#   Python 3.11        — GPy requires >=3.9; 3.11 is stable LTS
#   matplotlib >=3.8   — mne requires >=3.6; no upper bound needed
#   scikit-learn       — no hard bounds from any package; install latest
#   mne 1.8.0          — requires scipy>=1.9 ✓ and numpy>=1.23 ✓
#   BCI2000Tools       — optional; used by Whitening.py for file I/O helpers
#                        (the pipeline runs without it; gracefully skipped)
#
# ---------------------------------------------------------------------------

section "Creating conda environment: $ENV_NAME (Python 3.11)"

conda create -n "$ENV_NAME" -y \
    -c conda-forge \
    python=3.11 \
    "numpy=1.26.4" \
    "scipy=1.12.0" \
    "matplotlib>=3.8,<4" \
    scikit-learn \
    pip

# ---------------------------------------------------------------------------
# 2. Activate and install pip-only packages
# ---------------------------------------------------------------------------
section "Installing pip-only packages"

# Resolve the environment's pip without requiring the user to activate first.
CONDA_PREFIX=$(conda info --base)
ENV_PIP="$CONDA_PREFIX/envs/$ENV_NAME/bin/pip"

if [[ ! -x "$ENV_PIP" ]]; then
    # Fallback for non-standard conda layouts
    ENV_PIP=$(conda run -n "$ENV_NAME" which pip)
fi

info "Using pip: $ENV_PIP"

# --- Core: GPy and its direct dependencies ---
# paramz is installed first to avoid GPy pulling an older version.
"$ENV_PIP" install \
    "paramz==0.9.6" \
    "GPy==1.13.2"

# --- Optional: MNE (topomap plots) ---
# mne 1.8.0 is the last release that fits within scipy<=1.12.0.
# The 'hdf5' extra adds h5py for reading/writing .fif files.
"$ENV_PIP" install "mne==1.8.0"

# --- Optional: BCI2000Tools (Whitening.py file I/O helpers) ---
# Silently skipped at runtime if absent; install it here for completeness.
"$ENV_PIP" install "BCI2000Tools==1.1.0"

# ---------------------------------------------------------------------------
# 3. Smoke test
# ---------------------------------------------------------------------------
section "Smoke test"

conda run -n "$ENV_NAME" python - <<'PYEOF'
import sys
print(f"Python {sys.version}")

ok = True

def check(label, fn):
    global ok
    try:
        result = fn()
        print(f"  [OK]  {label}{(' — ' + result) if result else ''}")
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        ok = False

import numpy as np
check("numpy",       lambda: np.__version__)

import scipy
check("scipy",       lambda: scipy.__version__)

import matplotlib
check("matplotlib",  lambda: matplotlib.__version__)

import sklearn
check("scikit-learn",lambda: sklearn.__version__)

import GPy
check("GPy",         lambda: GPy.__version__)

try:
    import mne
    check("mne",     lambda: mne.__version__)
except ImportError as e:
    print(f"  [WARN] mne not importable: {e}")

try:
    from BCI2000Tools.Container import Bunch
    check("BCI2000Tools", lambda: None)
except ImportError:
    print("  [INFO] BCI2000Tools not importable (optional — pipeline runs without it)")

# GPy numpy-2.0 guard
arr = np.ones((3, 3))
import GPy.kern
k = GPy.kern.RBF(3)
print(f"  [OK]  GPy kernel instantiation")

# scipy ceiling guard
from scipy.linalg import svd
svd(arr)
print(f"  [OK]  scipy.linalg.svd")

if not ok:
    print("\n[WARN] One or more checks failed — see above.")
    sys.exit(1)
else:
    print("\nAll checks passed.")
PYEOF

# ---------------------------------------------------------------------------
# 4. Done
# ---------------------------------------------------------------------------
section "Setup complete"

echo ""
info "Activate the environment with:"
echo ""
echo "    conda activate $ENV_NAME"
echo ""
info "Then run the pipeline, e.g.:"
echo ""
echo "    python run_class_bci_competition_III_merged_spatial_filters_gpy.py \\"
echo "        --data-path ./data/your_dataset.pkl \\"
echo "        --results-dir ./results \\"
echo "        --nfs 1 2 4 --kfolds 4 \\"
echo "        --kernel-type RBF --alignment riemann \\"
echo "        --csp --ard --inner-val-frac 0.15 --es-patience 20"
echo ""
warn "Note: topomap plots (08_topomaps.png) also require electrode coordinates"
warn "      (ch_xy) to be present in the dataset pickle, in addition to MNE."
