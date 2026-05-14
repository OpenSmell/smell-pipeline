#!/usr/bin/env python3
"""
Build chemoprints for SmellNet substances using local FooDB JSON zip.
Streams Content.json from inside the zip (no disk extraction of the 3.5 GB file).
"""
import json, os, sys, zipfile
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski, GraphDescriptors

RDLogger.logger().setLevel(RDLogger.ERROR)

FOODB_ZIP = "foodb_json.zip"
FOODB_DIR = "foodb_2020_04_07_json"
COMPOUND_CP_CACHE = "compound_chemoprints.npy"
OUTPUT_CSV = "foodb_chemoprints.csv"

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

# ── Chemoprint function ──────────────────────────────────────────────────────
_CP_CACHE = {}
def chemoprint(smiles):
    if smiles in _CP_CACHE:
        return _CP_CACHE[smiles]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        _CP_CACHE[smiles] = None
        return None
    props = [
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
        Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumAliphaticRings(mol),
        Descriptors.FractionCSP3(mol), Descriptors.TPSA(mol),
        Descriptors.NumValenceElectrons(mol), Descriptors.HeavyAtomCount(mol),
        GraphDescriptors.Chi0(mol), GraphDescriptors.Chi1(mol),
        GraphDescriptors.Kappa1(mol),
    ]
    fg_smarts = {
        "alcohol": "[OX2H]", "aldehyde": "[CX3H1](=O)[#6]",
        "ketone": "[#6][CX3](=O)[#6]", "carboxylic_acid": "[CX3](=O)[OX2H1]",
        "amine": "[NX3;H2,H1;!$(NC=O)]", "ester": "[#6][CX3](=O)[OX2H0][#6]",
        "ether": "[OD2]([#6])[#6]", "nitrile": "[NX1]#[CX2]",
        "amide": "[NX3][CX3](=[OX1])", "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
        "thiol": "[SX2H]", "sulfide": "[SX2]([#6])[#6]",
        "aromatic": "a", "alkene": "[CX3]=[CX3]",
    }
    for smarts in fg_smarts.values():
        patt = Chem.MolFromSmarts(smarts)
        props.append(1.0 if patt and mol.HasSubstructMatch(patt) else 0.0)
    result = np.array(props, dtype=np.float32)
    _CP_CACHE[smiles] = result
    return result

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
            cp = chemoprint(smi)
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
    np.save("foodb_chemoprints.npy", arr)
    with open("foodb_substances.txt", "w") as f:
        f.writelines(f"{k}\n" for k in keys)
    import pandas as pd
    df = pd.DataFrame(arr, index=keys, columns=[f"cp_{i}" for i in range(arr.shape[1])])
    df.index.name = "substance"
    df.to_csv(OUTPUT_CSV)
    print(f"✅ Saved to {OUTPUT_CSV}")
else:
    print("❌ No chemoprints")
