#!/usr/bin/env python3
"""
Build chemoprints for SmellNet substances using local FooDB JSON zip.
Streams Content.json from inside the zip (no disk extraction of the 3.5 GB file).
Uses the canonical `chemoprint` package (v1.0) for chemoprint generation.
"""
import json, zipfile
import numpy as np
from pathlib import Path
from chemoprint import chemoprint_from_smiles

ROOT = Path(__file__).resolve().parent.parent
FOODB_ZIP = ROOT / "models" / "foodb_json.zip"
FOODB_DIR = "foodb_2020_04_07_json"
OUTPUT_DIR = ROOT / "data"

FOOD_ID_MAP = {
    "allspice": 288, "almond": 148, "angelica": 1, "apple": 105,
    "asparagus": 21, "avocado": 130, "banana": 208, "brazil_nut": 24,
    "broccoli": 34, "brussel_sprouts": 32, "cabbage": 881, "cashew": 11,
    "cauliflower": 31, "chervil": 332, "chives": 9, "cinnamon": 586,
    "cloves": 179, "coriander": 61, "cumin": 67, "dill": 13,
    "garlic": 8, "ginger": 206, "hazelnut": 62, "kiwi": 4,
    "lemon": 54, "mandarin_orange": 56, "mango": 106, "mint": 113,
    "mugwort": 20, "mustard": 673, "nutmeg": 118, "oregano": 102,
    "peach": 149, "pear": 152, "pili_nut": 440, "pineapple": 12,
    "potato": 175, "radish": 153, "saffron": 63, "star_anise": 90,
    "strawberry": 83, "sweet_potato": 92, "tomato": 171, "turnip": 36,
}

# ── Chemoprint function (from canonical chemoprint package) ──────────────────

# ── 1. Load food names ───────────────────────────────────────────────────────
print("📖 Loading food names …")
food_map = {}
zf = zipfile.ZipFile(FOODB_ZIP)
with zf.open(f"{FOODB_DIR}/Food.json") as f:
    for line in f:
        entry = json.loads(line)
        food_map[entry["id"]] = entry["name"]

name_matches = {}
for s, fid in FOOD_ID_MAP.items():
    if fid in food_map:
        name_matches[s] = (fid, food_map[fid])
        print(f"  ✅ {s:20s} -> {food_map[fid]}")

target_fids = set(FOOD_ID_MAP.values()) & set(food_map.keys())
print(f"\n🎯 {len(target_fids)} foods matched")

# ── 2. Stream Content.json from zip → food_id → compound_ids ────────────────
print("\n🔗 Streaming Content.json from zip …")
food_to_cids = {fid: set() for fid in target_fids}
count = 0
zf = zipfile.ZipFile(FOODB_ZIP)
with zf.open(f"{FOODB_DIR}/Content.json") as f:
    for line in f:
        entry = json.loads(line)
        if entry.get("source_type") != "Compound":
            continue
        fid = entry.get("food_id")
        if fid in target_fids:
            food_to_cids[fid].add(entry["source_id"])
        count += 1
        if count % 500000 == 0:
            print(f"    ... scanned {count}")

print(f"  Scanned {count} content entries")
all_cids = set()
for fid in target_fids:
    n = len(food_to_cids[fid])
    all_cids.update(food_to_cids[fid])
    print(f"  {food_map[fid]:30s}: {n} compounds")
print(f"  → {len(all_cids)} unique compound IDs")

# ── 3. Load SMILES for needed compounds ─────────────────────────────────────
print("\n🧪 Loading compound SMILES …")
compound_smiles = {}
zf = zipfile.ZipFile(FOODB_ZIP)
with zf.open(f"{FOODB_DIR}/Compound.json") as f:
    for line in f:
        entry = json.loads(line)
        cid = entry["id"]
        if cid in all_cids:
            smi = entry.get("moldb_smiles")
            if smi and smi != "null":
                compound_smiles[cid] = smi
print(f"  {len(compound_smiles)} SMILES found")

# ── 4. Compute chemoprints (cached per unique SMILES) ───────────────────────
print("\n⚗️  Computing chemoprints …")
chemoprints = {}
for s, (fid, fname) in name_matches.items():
    cids = food_to_cids.get(fid, set())
    vecs = []
    for cid in cids:
        smi = compound_smiles.get(cid)
        if smi:
            cp = chemoprint_from_smiles(smi)
            if cp is not None:
                vecs.append(cp)
    if vecs:
        chemoprints[s] = np.mean(vecs, axis=0)
        print(f"  ✅ {s:20s}: {len(vecs)}/{len(cids)} compounds")
    else:
        print(f"  ⚠️  {s:20s}: 0/{len(cids)} compounds")

# ── 5. Save ──────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Chemoprints: {len(chemoprints)}/{len(FOOD_ID_MAP)}")
missing = [s for s in FOOD_ID_MAP if s not in chemoprints]
if missing:
    print(f"Missing: {missing}")

if chemoprints:
    keys = sorted(chemoprints.keys())
    arr = np.array([chemoprints[k] for k in keys])
    import pandas as pd
    df = pd.DataFrame(arr, index=keys, columns=[f"cp_{i}" for i in range(arr.shape[1])])
    df.index.name = "substance"
    csv_path = OUTPUT_DIR / "foodb_chemoprints.csv"
    df.to_csv(csv_path)
    np.save(OUTPUT_DIR / "foodb_chemoprints.npy", arr)
    with open(OUTPUT_DIR / "foodb_substances.txt", "w") as f:
        f.writelines(f"{k}\n" for k in keys)
    print(f"✅ Saved to {csv_path}")
else:
    print("❌ No chemoprints")
