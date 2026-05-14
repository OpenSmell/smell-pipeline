#!/usr/bin/env python3
"""
Step 1: Extract 128-dim latent vectors from the CNN for all 44 FooDB-covered
SmellNet substances. Averages all segments across all recordings of each substance.
Saves: data/substance_latents.csv (44 × 128)
"""
import os, glob, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "cnn_classifier.pth"
CSV_OUT = ROOT / "data" / "substance_latents.csv"
FOODB_CSV = ROOT / "data" / "foodb_chemoprints.csv"

# ── CNN architecture (must match session-invariance exactly) ─────────────────
class ConvSmellNet(nn.Module):
    def __init__(self, num_sensors=6, num_classes=50, seq_len=100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(num_sensors, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.drop = nn.Dropout(0.3)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x, return_latent=False):
        x = x.transpose(1, 2)
        latent = self.features(x).squeeze(-1)
        latent = self.drop(latent)
        out = self.classifier(latent)
        if return_latent:
            return out, latent
        return out

# ── Load model ───────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = ConvSmellNet(num_sensors=6, num_classes=50, seq_len=100).to(device)
state = torch.load(str(MODEL_PATH), map_location=device, weights_only=True)
model.load_state_dict(state)
model.eval()
print(f"Loaded: {MODEL_PATH}")

# ── Load FooDB-covered substances ────────────────────────────────────────────
foodb = pd.read_csv(str(FOODB_CSV), index_col=0)
target_substances = set(foodb.index)
print(f"Target substances (from FooDB): {len(target_substances)}")

# ── Find SmellNet CSV files in HuggingFace cache ─────────────────────────────
hub = os.path.expanduser("~/.cache/huggingface/hub/datasets--DeweiFeng--smell-net")
snap = os.path.join(hub, "snapshots")
rev = sorted(os.listdir(snap), reverse=True)[0]
data_root = os.path.join(snap, rev, "base_data")
csv_files = glob.glob(os.path.join(data_root, "**/*.csv"), recursive=True)
print(f"SmellNet CSV files found: {len(csv_files)}")

# ── Process: for each CSV, extract segments → CNN → latent ──────────────────
SENSOR_NAMES = ['NO2', 'C2H5OH', 'VOC', 'CO', 'Alcohol', 'LPG']
SEGMENT_LEN = 100
STRIDE = SEGMENT_LEN // 2

latents_by_substance = defaultdict(list)
substances_found = set()

for fpath in csv_files:
    fname = os.path.basename(fpath)
    stem = os.path.splitext(fname)[0]
    parts = stem.split("_")
    if len(parts) < 2 or not parts[-1].isdigit():
        continue
    substance = "_".join(parts[:-1])
    if substance not in target_substances:
        continue

    try:
        df = pd.read_csv(fpath)
    except Exception:
        continue

    sensor_cols = []
    for expected in SENSOR_NAMES:
        found = [c for c in df.columns if c.lower() == expected.lower()]
        if found:
            sensor_cols.append(found[0])
        else:
            break
    if len(sensor_cols) != 6:
        continue

    raw = df[sensor_cols].values.astype(np.float32)
    N = raw.shape[0]
    if N < SEGMENT_LEN:
        segments = [raw]
    else:
        segments = [raw[i:i + SEGMENT_LEN] for i in range(0, N - SEGMENT_LEN + 1, STRIDE)]

    X = np.stack(segments)  # (n_segments, 100, 6)
    X_t = torch.tensor(X, dtype=torch.float32, device=device)

    with torch.no_grad():
        _, latents = model(X_t, return_latent=True)
    latents_by_substance[substance].append(latents.cpu().numpy())
    substances_found.add(substance)

print(f"Substances with latent vectors: {len(substances_found)}")
missing = target_substances - substances_found
if missing:
    print(f"  Missing (no CSVs found): {sorted(missing)}")

# ── Average across all segments and all recordings ───────────────────────────
substance_means = {}
for s, arr_list in latents_by_substance.items():
    all_latents = np.concatenate(arr_list, axis=0)  # (total_segments, 128)
    substance_means[s] = all_latents.mean(axis=0)
    print(f"  {s:20s}: {all_latents.shape[0]:4d} segments → 128-dim mean")

# ── Save ─────────────────────────────────────────────────────────────────────
keys = sorted(substance_means.keys())
arr = np.array([substance_means[k] for k in keys])
df_out = pd.DataFrame(arr, index=keys, columns=[f"latent_{i}" for i in range(128)])
df_out.index.name = "substance"
df_out.to_csv(str(CSV_OUT))
print(f"\n✅ Saved: {CSV_OUT} ({len(keys)} substances × 128 dimensions)")
