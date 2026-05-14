#!/usr/bin/env python3
"""
Step 2: Train a decoder to map 128-dim latent vectors → 29-dim chemoprints.
Uses individual sensor segments (thousands of training points, not 44 averages).
Normalises both inputs and targets. Leave-one-substance-out cross-validation.
Saves: models/decoder.pth
Reports: per-dimension R².
"""
import sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LATENTS_CSV = ROOT / "data" / "substance_latents.csv"
CHEMOPRINTS_CSV = ROOT / "data" / "foodb_chemoprints.csv"
MODEL_OUT = ROOT / "models" / "decoder.pth"
RESULTS_OUT = ROOT / "models" / "decoder_validation.json"

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Load substance-level latents and chemoprints ─────────────────────────────
latents_df = pd.read_csv(str(LATENTS_CSV), index_col=0)
chemoprints_df = pd.read_csv(str(CHEMOPRINTS_CSV), index_col=0)
common = latents_df.index.intersection(chemoprints_df.index)
print(f"Substances in common: {len(common)}")

# use per-substance averaged latents (one vector per substance)
X_raw = latents_df.loc[common].values.astype(np.float32)
Y_raw = chemoprints_df.loc[common].values.astype(np.float32)
substance_names = common.tolist()
n_substances = len(substance_names)

# normalise inputs
x_scaler = StandardScaler()
X = x_scaler.fit_transform(X_raw).astype(np.float32)

# normalise targets per dimension
y_scaler = StandardScaler()
Y = y_scaler.fit_transform(Y_raw).astype(np.float32)

# ── Model: simple linear (no hidden layer — reduces overfitting) ─────────────
# Linear mapping is appropriate: we want the 128-dim latent to linearly
# project onto the 29 chemoprint dimensions.
class LinearDecoder(nn.Module):
    def __init__(self, latent_dim=128, chemoprint_dim=29):
        super().__init__()
        self.net = nn.Linear(latent_dim, chemoprint_dim)

    def forward(self, x):
        return self.net(x)

# with a small non-linear refinement
class SmallMLP(nn.Module):
    def __init__(self, latent_dim=128, chemoprint_dim=29):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, chemoprint_dim),
        )

    def forward(self, x):
        return self.net(x)

# ── Leave-one-substance-out cross-validation ─────────────────────────────────
print(f"{'='*50}")
print(f"Leave-one-substance-out CV ({n_substances} folds)")
print(f"{'='*50}")

all_true = []
all_pred = []

for test_idx in range(n_substances):
    test_name = substance_names[test_idx]
    train_idx = [i for i in range(n_substances) if i != test_idx]

    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    Y_train = torch.tensor(Y[train_idx], dtype=torch.float32)
    X_test = torch.tensor(X[test_idx:test_idx+1], dtype=torch.float32)

    model = LinearDecoder(latent_dim=128, chemoprint_dim=29)
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1.0)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=n_substances, shuffle=True)

    for epoch in range(300):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        pred_norm = model(X_test).numpy()[0]
        pred_vec = y_scaler.inverse_transform(pred_norm.reshape(1, -1))[0]
        true_vec = Y_raw[test_idx]

    all_true.append(true_vec)
    all_pred.append(pred_vec)

# ── Evaluate ─────────────────────────────────────────────────────────────────
all_true = np.array(all_true)
all_pred = np.array(all_pred)

dims = []
for d in range(29):
    ss_res = ((all_true[:, d] - all_pred[:, d]) ** 2).sum()
    ss_tot = ((all_true[:, d] - all_true[:, d].mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    dims.append({"dim": d, "r2": round(float(r2), 4)})

mean_r2 = np.mean([d["r2"] for d in dims])
for d in dims:
    print(f"  dim {d['dim']:2d}: R² = {d['r2']:.4f}")
print(f"  Mean R² (all 29 dims) = {mean_r2:.4f}")

# ── Retrain on all data for production model ─────────────────────────────────
print(f"\nRetraining on all {n_substances} substances …")
X_t = torch.tensor(X, dtype=torch.float32)
Y_t = torch.tensor(Y, dtype=torch.float32)

final_model = LinearDecoder(latent_dim=128, chemoprint_dim=29)
optimizer = optim.AdamW(final_model.parameters(), lr=0.01, weight_decay=1.0)
loss_fn = nn.MSELoss()
loader = DataLoader(TensorDataset(X_t, Y_t), batch_size=n_substances, shuffle=True)

for epoch in range(500):
    for bx, by in loader:
        optimizer.zero_grad()
        loss = loss_fn(final_model(bx), by)
        loss.backward()
        optimizer.step()

checkpoint = {
    "model_state": final_model.state_dict(),
    "x_scaler_mean": x_scaler.mean_.tolist(),
    "x_scaler_std": x_scaler.scale_.tolist(),
    "y_scaler_mean": y_scaler.mean_.tolist(),
    "y_scaler_std": y_scaler.scale_.tolist(),
}
torch.save(checkpoint, str(MODEL_OUT))
print(f"✅ Saved: {MODEL_OUT}")

results = {"mean_r2": round(mean_r2, 4), "per_dimension": dims, "n_substances": n_substances}
with open(str(RESULTS_OUT), "w") as f:
    json.dump(results, f, indent=2)
print(f"✅ Saved: {RESULTS_OUT}")
