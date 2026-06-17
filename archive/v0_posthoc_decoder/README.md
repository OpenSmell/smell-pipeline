# v0 Post-hoc Decoder — Failed Approach (Archived)

> **Status:** Archived. R² near zero.
> **Replaced by:** MAE + contrastive encoder (see `opensmell/encoder`)

## What was attempted

Train a **post-hoc linear decoder** to map from a classification CNN's 128-dim latent space to 29-dim chemoprint vectors:

1. Extract 128-dim latent vectors from the ConvSmellNet (trained for 50-class substance discrimination on SmellNet data).
2. Average all segments per substance → 44 vectors × 128 dims.
3. Train a `LinearDecoder(128→29)` with L2 regularisation to predict FooDB chemoprints.
4. Leave-one-substance-out cross-validation.

## Why it failed

| Problem | Detail |
|---------|--------|
| **CNN trained for classification, not chemistry** | The latent space organises substances by discriminative features (what makes substances **different**), not chemical properties (what makes them **similar** by composition). |
| **44 data points is too few** | A post-hoc decoder with 128×29 = 3,712 parameters cannot generalise from 44 examples. |
| **No chemical signal in the latent space** | Mean R² = -0.08 across 29 dims — worse than predicting the mean. The CNN's latent space simply does not encode chemoprint information linearly. |
| **Per-substance averaging discards information** | Averaging 100+ segments per substance loses within-substance variation that could help the decoder. |

## What was learned

1. **The latent space must be trained with chemical reconstruction from the start**, not bolted on later. A classification-only CNN ignores chemistry.
2. **44 data points is insufficient for supervised decoder training.** Even a linear model overfits. A self-supervised approach (MAE) can use all unlabelled segments.
3. **Post-hoc approaches don't work.** The chemoprint decoder must be part of the training objective, not an afterthought.

## What replaces it

The encoder (`opensmell/encoder`) trains a 256-dim latent space with:

- **Masked autoencoder (MAE)** pretraining — learns to reconstruct sensor waveforms from partial observations. This captures chemical information without labels.
- **Contrastive fine-tuning** — pulls same-substance latents together, pushes different-substance latents apart.
- **Domain-adversarial loss** — makes latents invariant to session and device.

The chemoprint head is then trained on the **frozen** encoder, using pooled segment-level latents (thousands, not 44).

## Archived files

- `train_decoder.py` — LinearDecoder + SmallMLP training with LOOCV
- `extract_latents.py` — Per-substance average latent extraction from ConvSmellNet

## Preserved files (still valid)

- `data/foodb_chemoprints.csv` — 44 substances × 29 chemoprint features (ground truth)
- `data/foodb_substances.txt` — Substance names
- `src/build_chemoprints.py` — FooDB extraction pipeline (streams 3.5 GB Content.json from zip)
- `src/explore_gcms.py` — Historical GC-MS CSV inspection
- `src/chemoprint_utils.py` — Vendored chemoprint + Leffingwell + similarity utilities
- `models/foodb_json.zip` — FooDB database dump
- `models/leffingwell/` — Odor predictor model (optional, 400 MB)
