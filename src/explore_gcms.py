#!/usr/bin/env python3
"""
SmellNet GC‑MS → Chemoprint feasibility check
==============================================
1. Finds the processed GC‑MS CSVs from the SmellNet dataset.
2. For each substance, reads VOC name / concentration / SMILES.
3. Validates SMILES with RDKit, computes the 29‑dim chemoprint.
4. Computes mixture chemoprint as a concentration‑weighted average.
5. Saves substance → chemoprint mapping (mixture_chemoprints.csv).
6. Reports coverage stats.
"""

import os, json, io, glob, warnings
import numpy as np
import pandas as pd
from pathlib import Path

# ---- RDKit (may need conda/pip install rdkit-pypi) ----
try:
    from chemoprint import chemoprint_from_smiles
except ImportError:
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski, GraphDescriptors
    except ImportError:
        print("❌ RDKit not installed.  pip install rdkit-pypi")
        exit(1)
    print("⚠️  canonical chemoprint package not installed; falling back to RDKit directly. pip install chemoprint")

# ---- HuggingFace datasets ----
try:
    from datasets import load_dataset
except ImportError:
    print("❌ datasets not installed.  pip install datasets")
    exit(1)

# ======================================================================
# Chemoprint function (from canonical chemoprint package, imported above)
# ======================================================================


# ======================================================================
# 1. Locate the GC‑MS processed data
# ======================================================================
print("🔍 Searching for SmellNet GC‑MS processed data …")

# First, try the dedicated config (if it exists)
found_gcms_dir = None
try:
    ds = load_dataset("DeweiFeng/smell-net", "gcms_processed", split="train")
    # If successful, find cached CSV files
    hub = os.path.expanduser("~/.cache/huggingface/hub/datasets--DeweiFeng--smell-net")
    if os.path.isdir(hub):
        snap = os.path.join(hub, "snapshots")
        revs = sorted(os.listdir(snap), reverse=True)
        for r in revs:
            cand = os.path.join(snap, r, "gcms_processed")
            if os.path.isdir(cand):
                found_gcms_dir = cand
                break
except Exception:
    pass

# Fallback: look inside the base_data cache for a gcms_processed folder
if not found_gcms_dir:
    hub = os.path.expanduser("~/.cache/huggingface/hub/datasets--DeweiFeng--smell-net")
    if os.path.isdir(hub):
        snap = os.path.join(hub, "snapshots")
        revs = sorted(os.listdir(snap), reverse=True)
        for r in revs:
            for sub in ["gcms_processed", "gcms_data"]:
                cand = os.path.join(snap, r, sub)
                if os.path.isdir(cand):
                    found_gcms_dir = cand
                    break
            if found_gcms_dir:
                break

if not found_gcms_dir:
    # Last resort: download the dataset again and manually search the cache
    print("⚠️  GC‑MS directory not found in cache; forcing dataset download …")
    ds = load_dataset("DeweiFeng/smell-net", "base_data", split="train")
    # now re-scan cache
    hub = os.path.expanduser("~/.cache/huggingface/hub/datasets--DeweiFeng--smell-net")
    if os.path.isdir(hub):
        snap = os.path.join(hub, "snapshots")
        revs = sorted(os.listdir(snap), reverse=True)
        for r in revs:
            for sub in ["gcms_processed", "gcms_data"]:
                cand = os.path.join(snap, r, sub)
                if os.path.isdir(cand):
                    found_gcms_dir = cand
                    break
            if found_gcms_dir:
                break

if not found_gcms_dir:
    print("❌ Could not locate GC‑MS processed CSV files.")
    print("   Check the dataset on HuggingFace: https://huggingface.co/datasets/DeweiFeng/smell-net")
    exit(1)

print(f"✅ Found GC‑MS data at: {found_gcms_dir}")

# Collect all CSV files in that directory (including subdirs)
csv_files = glob.glob(os.path.join(found_gcms_dir, "**/*.csv"), recursive=True)
if not csv_files:
    csv_files = glob.glob(os.path.join(found_gcms_dir, "*.csv"))
print(f"📄 {len(csv_files)} CSV files found")

# ======================================================================
# 2. Parse each CSV
# ======================================================================
substance_chemoprints = {}  # substance -> mixture chemoprint (29,)
total_vocs = 0
missing_smiles = 0
valid_smiles = 0
substances_with_data = 0

for csv_path in csv_files:
    fname = os.path.basename(csv_path)
    # assume substance name from filename (first part before underscore or extension)
    substance = os.path.splitext(fname)[0].split("_")[0]  # coarse, but works for most
    # If substance contains "mixture" or "base", refine later.

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"⚠️  Skipping {fname}: read error {e}")
        continue

    # Try to identify columns for VOC name, concentration, SMILES
    # Common patterns: 'VOC', 'Name', 'Compound', 'SMILES', 'Concentration'
    # We'll look for columns that contain these strings (case insensitive)
    cols = [c.lower() for c in df.columns]
    voc_col = None
    conc_col = None
    smiles_col = None

    for c in df.columns:
        cl = c.lower()
        if 'voc' in cl or 'name' in cl or 'compound' in cl:
            voc_col = c
        if 'conc' in cl or 'amount' in cl or 'percent' in cl:
            conc_col = c
        if 'smiles' in cl or 'smile' in cl:
            smiles_col = c

    if voc_col is None or conc_col is None:
        # Maybe the columns are unnamed; try first three columns
        if len(df.columns) >= 3:
            voc_col = df.columns[0]
            smiles_col = df.columns[1] if 'smiles' in df.columns[1].lower() else None
            conc_col = df.columns[2]
        else:
            print(f"⚠️  {fname}: cannot identify VOC/Concentration/SMILES columns. Columns: {list(df.columns)}")
            continue

    # If there's a SMILES column, use it; otherwise maybe it's in the name and we need to map.
    # We'll try to use the SMILES column directly.
    if smiles_col is None:
        # Maybe the VOC column actually contains SMILES? Not ideal. We'll skip.
        print(f"⚠️  {fname}: no SMILES column found, cannot compute chemoprints")
        continue

    # Now iterate rows
    chem_list = []
    weights = []
    for _, row in df.iterrows():
        total_vocs += 1
        smiles = str(row[smiles_col]).strip()
        if not smiles or smiles == 'nan':
            missing_smiles += 1
            continue
        cp = chemoprint_from_smiles(smiles)
        if cp is None:
            missing_smiles += 1
            continue
        valid_smiles += 1
        conc = float(row[conc_col]) if conc_col else 0.0
        chem_list.append(cp)
        weights.append(max(conc, 0.0))  # avoid negative

    if len(chem_list) == 0:
        print(f"⚠️  {substance}: no valid chemoprints from {fname}")
        continue

    # Compute weighted average
    w = np.array(weights, dtype=np.float32)
    w_sum = w.sum()
    if w_sum <= 0:
        # equal weight
        mixture_cp = np.mean(chem_list, axis=0)
    else:
        mixture_cp = np.average(chem_list, axis=0, weights=w)

    substance_chemoprints[substance] = mixture_cp
    substances_with_data += 1

# ======================================================================
# 3. Report
# ======================================================================
print("\n" + "=" * 60)
print("📊 SmellNet GC‑MS → Chemoprint Coverage Report")
print("=" * 60)
print(f"GC‑MS directory: {found_gcms_dir}")
print(f"CSV files found: {len(csv_files)}")
print(f"Substances with computed chemoprint: {substances_with_data}")
print(f"Total VOCs encountered: {total_vocs}")
print(f"VOCs with valid SMILES & chemoprint: {valid_smiles} ({100*valid_smiles/total_vocs:.1f}%)"
      if total_vocs > 0 else "N/A")
print(f"Missing / invalid SMILES: {missing_smiles}")

if substance_chemoprints:
    print(f"\n✅ Feasibility: Chemoprints can be computed for {len(substance_chemoprints)} substances.")
    # Save mapping
    out_df = pd.DataFrame.from_dict(substance_chemoprints, orient='index')
    out_df.columns = [f"cp_{i}" for i in range(29)]
    out_path = "mixture_chemoprints.csv"
    out_df.to_csv(out_path)
    print(f"📁 Mixture chemoprints saved to {out_path}")
else:
    print("\n❌ No substances with valid chemoprints found. Decoder training not feasible with this data.")

print("\nDone.")