#!/usr/bin/env python3
"""
Minimal chemoprint utilities vendored for the demo.
SMILES → 29-dim chemoprint, Leffingwell odor prediction, FAISS similarity.
"""
import os, json, warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent

# ── RDKit chemoprint (from canonical chemoprint package) ──────────────────
_CHEMOPRINT_CACHE = {}

def chemoprint_from_smiles(smiles):
    if smiles in _CHEMOPRINT_CACHE:
        return _CHEMOPRINT_CACHE[smiles]
    try:
        from chemoprint import chemoprint_from_smiles as _canonical_cp
    except ImportError:
        return None
    result = _canonical_cp(smiles)
    _CHEMOPRINT_CACHE[smiles] = result
    return result


# ── Leffingwell odor predictor ───────────────────────────────────────────────
_LEFF_MODEL = None
_LEFF_ODORS = None

def load_leffingwell():
    global _LEFF_MODEL, _LEFF_ODORS
    if _LEFF_MODEL is not None:
        return _LEFF_MODEL, _LEFF_ODORS
    import joblib
    leff_dir = ROOT / "models" / "leffingwell"
    model_path = leff_dir / "leffingwell_model.joblib"
    cols_path = leff_dir / "odor_columns.json"
    if not model_path.exists() or not cols_path.exists():
        return None, None
    _LEFF_MODEL = joblib.load(str(model_path))
    with open(str(cols_path)) as f:
        _LEFF_ODORS = json.load(f)
    return _LEFF_MODEL, _LEFF_ODORS

def predict_odors(chemoprint_vector, threshold=0.3):
    model, odors = load_leffingwell()
    if model is None:
        return None
    X = np.array([chemoprint_vector], dtype=np.float32)
    probs_list = model.predict_proba(X)
    probs = np.array([p[0][1] if p.shape[1] > 1 else 0.0 for p in probs_list])
    results = [{"odor": odors[i], "confidence": float(probs[i])} for i in range(len(odors))
               if probs[i] >= threshold]
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results


# ── Similarity (cosine similarity over the 44 FooDB chemoprints) ─────────────
_SIM_VECS = None
_SIM_NAMES = None

def _load_sim_data():
    global _SIM_VECS, _SIM_NAMES
    if _SIM_VECS is not None:
        return _SIM_VECS, _SIM_NAMES
    csv_path = ROOT / "data" / "foodb_chemoprints.csv"
    if not csv_path.exists():
        return None, None
    import pandas as pd
    df = pd.read_csv(str(csv_path), index_col=0)
    vecs = df.values.astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    _SIM_VECS = vecs / np.maximum(norms, 1e-8)
    _SIM_NAMES = df.index.tolist()
    return _SIM_VECS, _SIM_NAMES

def find_similar(chemoprint_vector, top_k=5):
    vecs, names = _load_sim_data()
    if vecs is None:
        return None
    query = np.array(chemoprint_vector, dtype=np.float32)
    qnorm = np.linalg.norm(query)
    query = query / max(qnorm, 1e-8)
    scores = vecs @ query
    idx = np.argsort(scores)[::-1][:top_k]
    return [{"substance": names[i], "similarity": float(scores[i])} for i in idx]
