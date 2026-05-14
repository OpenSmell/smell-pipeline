#!/usr/bin/env bash
set -e

echo "=== OpenSmell Pipeline Setup ==="

# ── 1. Locate neighbor repos ─────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

NEIGHBOR_DIR="$(dirname "$SCRIPT_DIR")/session-invariance"
CHEMOPRINT_APPS="$(dirname "$SCRIPT_DIR")/Chemoprint Apps"

# ── 2. Copy CNN model from session-invariance ────────────────────────────────
if [ -f "$NEIGHBOR_DIR/model_cnn.pth" ]; then
    echo "[1/4] Copying CNN model from session-invariance …"
    cp "$NEIGHBOR_DIR/model_cnn.pth" models/cnn_classifier.pth
else
    echo "[1/4] ⚠️  model_cnn.pth not found at $NEIGHBOR_DIR"
    echo "       Place it manually at models/cnn_classifier.pth"
fi

# ── 3. Copy Leffingwell model from Chemoprint Apps ───────────────────────────
LEFF_MODEL="$CHEMOPRINT_APPS/leffingwell-smell-predictor/models/leffingwell_model.joblib"
LEFF_COLS="$CHEMOPRINT_APPS/leffingwell-smell-predictor/models/odor_columns.json"
LEFF_META="$CHEMOPRINT_APPS/leffingwell-smell-predictor/models/model_meta.json"
LEFF_FREQ="$CHEMOPRINT_APPS/leffingwell-smell-predictor/models/odor_frequencies.json"

mkdir -p models/leffingwell

if [ -f "$LEFF_MODEL" ]; then
    echo "[2/4] Copying Leffingwell predictor model …"
    cp "$LEFF_MODEL" models/leffingwell/
else
    echo "[2/4] ⚠️  Leffingwell model not found at $LEFF_MODEL"
    echo "       The demo will skip odor prediction without it."
fi

for f in "$LEFF_COLS" "$LEFF_META" "$LEFF_FREQ"; do
    [ -f "$f" ] && cp "$f" models/leffingwell/
done

# ── 4. Check Python environment ──────────────────────────────────────────────
echo "[3/4] Checking Python environment …"

PYTHON="python3"
# Prefer the odor conda environment
if [ -f "$HOME/miniconda3/envs/odor/bin/python" ]; then
    PYTHON="$HOME/miniconda3/envs/odor/bin/python"
    echo "  Using: $PYTHON"
elif [ -f "$HOME/miniconda3/bin/python" ]; then
    PYTHON="$HOME/miniconda3/bin/python"
    echo "  Using: $PYTHON"
fi

# Check that key packages are available
$PYTHON -c "import numpy; import pandas; import torch; import sklearn" 2>/dev/null \
    && echo "  ✅  Core packages (numpy, pandas, torch, sklearn)" \
    || echo "  ⚠️  Missing core packages — run: conda activate odor && pip install numpy pandas torch scikit-learn"

$PYTHON -c "from rdkit import Chem; print('  ✅  RDKit:', Chem.RDKFingerprint)" 2>/dev/null \
    || echo "  ⚠️  RDKit not installed — run: conda activate odor && pip install rdkit-pypi"

$PYTHON -c "import streamlit" 2>/dev/null \
    && echo "  ✅  Streamlit" \
    || echo "  ⚠️  Streamlit not installed — run: conda activate odor && pip install streamlit"

$PYTHON -c "import joblib" 2>/dev/null \
    && echo "  ✅  joblib" \
    || echo "  ⚠️  joblib not installed — run: conda activate odor && pip install joblib"

# ── 5. Verify the key files ──────────────────────────────────────────────────
echo "[4/4] Verification …"
MISSING=0
for f in models/cnn_classifier.pth data/foodb_chemoprints.csv models/foodb_json.zip; do
    if [ -f "$f" ]; then
        echo "  ✅ $f"
    else
        echo "  ❌ $f"
        MISSING=$((MISSING + 1))
    fi
done

if [ "$MISSING" -eq 0 ]; then
    echo "  ─────────────────────────────"
    echo "  Setup complete."
else
    echo "  ─────────────────────────────"
    echo "  ⚠️  $MISSING files missing — check the warnings above."
fi
