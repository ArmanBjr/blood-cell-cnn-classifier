# Blood Cell Classification with CNNs

**Course:** Computational Intelligence (مبانی هوش محاسباتی)  
**University:** Ferdowsi University of Mashhad  
**Team:** AmirHosein Abolfazli · Arman Bijari  

---

## Task

4-class white blood cell image classification:

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

Place the dataset at `../Data/` relative to this repo (not committed — too large).

Expected structure:
```
Data/
├── TRAIN/
│   ├── EOSINOPHIL/
│   ├── LYMPHOCYTE/
│   ├── MONOCYTE/
│   └── NEUTROPHIL/
├── TEST/
│   └── ...
└── TEST_SIMPLE/
    └── ...
```

---

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.10 + TensorFlow 2.15.

---

## Notebooks (run in order)

| Notebook | Description |
|---|---|
| `01_eda.ipynb` | Exploratory data analysis & visualization |
| `02_custom_cnn.ipynb` | Custom CNN trained from scratch |
| `03_transfer_learning.ipynb` | VGG16 · ResNet50V2 · MobileNetV2 · EfficientNetB3 |
| `04_ensemble_eval.ipynb` | Ensemble + full comparison of all models |
| `05_explainability.ipynb` | Grad-CAM · LIME · t-SNE |
| `06_test_simple.ipynb` | Inference on TEST_SIMPLE with confidence scores |

---

## Results (to be filled after training)

| Model | Test Accuracy | F1 (macro) |
|---|---|---|
| Custom CNN | — | — |
| MobileNetV2 | — | — |
| VGG16 | — | — |
| ResNet50V2 | — | — |
| EfficientNetB3 | — | — |
| Ensemble (top 3) | — | — |

---

## Project Structure

```
CNN-IML/
├── PLAN.md
├── README.md
├── requirements.txt
├── configs/config.yaml
├── notebooks/
├── src/
│   ├── data_loader.py
│   ├── augmentation.py
│   ├── evaluate.py
│   ├── visualize.py
│   ├── gradcam.py
│   └── models/
├── outputs/
│   ├── figures/
│   └── logs/
└── report/
```
