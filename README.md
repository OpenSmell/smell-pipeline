# OpenSmell Pipeline

Data pipeline utilities for the OpenSmell project.

## What's here

| Path | Contents |
|------|----------|
| `data/foodb_chemoprints.csv` | 44 substances × 29 chemoprint features (ground truth from FooDB) |
| `data/foodb_substances.txt` | Substance names for the 44 FooDB-covered substances |
| `src/build_chemoprints.py` | Extracts chemoprints from FooDB JSON dump (streams 3.5 GB from zip) |
| `src/chemoprint_utils.py` | Vendored chemoprint + Leffingwell + similarity utilities |
| `src/explore_gcms.py` | Historical GC-MS CSV inspection (dead end documented) |
| `models/foodb_json.zip` | FooDB database dump (87 MB) |
| `models/leffingwell/` | Odor predictor model (optional, 400 MB) |
| `archive/v0_posthoc_decoder/` | Failed post-hoc decoder approach (R² near zero) |

## The core product: `data/foodb_chemoprints.csv`

This is the primary output — 44 food substances with 29-dimensional chemoprint vectors computed from FooDB GC-MS volatile compound data. It's the ground truth for training the encoder's chemoprint head.

## Pipeline (historical)

The original pipeline attempted to train a post-hoc decoder from a classification CNN's latent space to chemoprints. R² was near zero — the CNN's latent space (trained for classification) does not encode chemical properties linearly. This approach is archived at `archive/v0_posthoc_decoder/`.

The correct approach (in development at `opensmell/encoder`) trains the latent space with MAE + contrastive + domain-adversarial loss from the start, with the chemoprint head trained on the frozen encoder.

## Requirements

- Python 3.10+
- `conda activate odor` (has rdkit, torch, numpy, pandas, scikit-learn, joblib)

## 6 missing substances

Chamomile, chestnuts, peanuts, pecans, pistachios, walnuts — not in FooDB. 44/50 = 88% coverage.

## License

Open source.
