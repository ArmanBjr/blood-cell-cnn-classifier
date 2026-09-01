# Blood Cell Classification with CNNs

Project 2 for **Fundamentals of Computational Intelligence** at Ferdowsi University of Mashhad (FUM). Four-class white blood cell image classification using custom CNNs, architecture search, transfer learning, interpretability (Grad-CAM), and an honest evaluation pipeline on the held-out test set.

> Course report: [`report/report.pdf`](report/report.pdf)

**Authors:** AmirHosein Abolfazli · **Arman Bijari** — [ArmanBjr](https://github.com/ArmanBjr)  
**Professor:** Dr. Fazl Ersi

---

## Task

| Label | Cell Type |
|---|---|
| 0 | EOSINOPHIL |
| 1 | LYMPHOCYTE |
| 2 | MONOCYTE |
| 3 | NEUTROPHIL |

---

## Dataset

[Blood Cell Images — Kaggle (paultimothymooney)](https://www.kaggle.com/datasets/paultimothymooney/blood-cells)

| Split | Images |
|---|---|
| TRAIN | 9,958 |
| TEST | 2,488 |
| TEST_SIMPLE | 72 |

Place the dataset at `Data/` in the repo root (not committed — too large):

```
Data/
├── TRAIN/
│   ├── EOSINOPHIL/
│   ├── LYMPHOCYTE/
│   ├── MONOCYTE/
│   └── NEUTROPHIL/
├── TEST/
└── TEST_SIMPLE/
```

---

## Quick Start

```bash
git clone https://github.com/ArmanBjr/CNN-IML.git
cd CNN-IML

python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt
jupyter lab notebooks/
```

Tested with Python 3.10 + TensorFlow 2.15.

---

## Notebooks (run in order)

| Phase | Notebook | Description |
|---|---|---|
| 1 | `01_eda.ipynb` | Exploratory data analysis, class balance, augmentation preview |
| 2 | `02_custom_cnn.ipynb` | Baseline custom CNN trained from scratch |
| 3 | `03_improved_cnn_v3.ipynb` | Architecture experiments (BN, dropout, depth, batch size) |
| 4 | `04_interpretability.ipynb` | Conv filters, feature maps, Grad-CAM |
| 5 | `05_transfer_learning.ipynb` | Transfer learning (EfficientNetV2B0, MobileNetV2), TTA, ensemble |
| 5b | `05b_leakage_demo.ipynb` | Controlled demo of train/test leakage (why naive splits inflate accuracy) |

A frozen course submission copy lives in `submission/notebooks/`.

---

## Results (held-out TEST split)

| Model | Test Accuracy | Macro F1 | Macro AUC |
|---|---:|---:|---:|
| Phase 2 — Custom CNN | 85.61% | — | — |
| Phase 3 — Improved CNN | 87.94% | — | — |
| EfficientNetV2B0 | 86.85% | 0.872 | 0.947 |
| EfficientNetV2B0 + TTA | 87.66% | 0.880 | 0.960 |
| MobileNetV2 | **88.66%** | **0.889** | 0.930 |
| MobileNetV2 + TTA | 88.58% | 0.889 | 0.969 |
| Ensemble (Eff + Mob) | 88.14% | 0.884 | 0.939 |
| **Ensemble + TTA (final)** | **88.34%** | **0.886** | **0.971** |

Full comparison table: [`outputs/phase5imp_results.csv`](outputs/phase5imp_results.csv)

---

## Project Structure

```
CNN-IML/
├── README.md
├── LICENSE
├── requirements.txt
├── configs/config.yaml
├── notebooks/              # main workflow (phases 1–5)
├── submission/notebooks/   # course submission snapshot
├── src/
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── gradcam.py
│   ├── visualize.py
│   └── models/
│       ├── custom_cnn.py
│       └── transfer.py
├── outputs/
│   ├── figures/            # EDA, training curves, confusion matrices, Grad-CAM
│   ├── logs/               # per-experiment CSV logs
│   └── phase5imp_results.csv
└── report/
    └── report.pdf
```

---

## Key Design Choices

- **Leakage-safe evaluation** — models are trained on `TRAIN/` only; `TEST/` is never used for tuning or augmentation fitting.
- **Phase 3 ablations** — systematic comparison of augmentation, batch norm, dropout, depth, and batch size on the custom CNN.
- **Phase 5b** — documents why pooling TRAIN+TEST before splitting produces misleading 95–99% scores seen in many public notebooks on this dataset.

---

## Authors & License

**AmirHosein Abolfazli** · **Arman Bijari** — [ArmanBjr](https://github.com/ArmanBjr)

Released under the [MIT License](LICENSE).
