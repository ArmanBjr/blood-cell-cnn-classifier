# Project Plan — Blood Cell Classification with CNNs

**Course:** مبانی هوش محاسباتی (Computational Intelligence)  
**Instructor:** Dr. Fazl Arthi  
**Team:** AmirHosein Abolfazli (4022262035) · Arman Bijari (4021262131)  
**Semester:** Spring 1404–1405

---

## Objective

Classify white blood cell (WBC) images into **4 categories**:
`EOSINOPHIL` · `LYMPHOCYTE` · `MONOCYTE` · `NEUTROPHIL`

using Convolutional Neural Networks — both from scratch and via transfer learning.

---

## Dataset

| Split | Images | Per Class |
|---|---|---|
| TRAIN | 9,958 | ~2,490 (balanced) |
| TEST | 2,488 | ~622 (balanced) |
| TEST_SIMPLE | 72 | small demo holdout |

- Format: JPEG, ~360×363px RGB
- Structure: one folder per class (compatible with `flow_from_directory`)
- Source: [Blood Cell Images — Kaggle (paultimothymooney)](https://www.kaggle.com/datasets/paultimothymooney/blood-cells)

---

## Phase 1 — EDA & Visualization (`notebooks/01_eda.ipynb`)

- [ ] Sample grid: 8 images per class (32 total)
- [ ] Class distribution bar chart → confirm balance
- [ ] Pixel intensity histograms per class
- [ ] Mean image per class
- [ ] Side-by-side: raw vs. augmented samples
- [ ] Image dimension verification

**Deliverable:** figures saved to `outputs/figures/eda_*`

---

## Phase 2 — Custom CNN from Scratch (`notebooks/02_custom_cnn.ipynb`)

### Architecture
```
Input (224×224×3)
→ [Conv2D(32,3×3) → BN → ReLU → MaxPool(2×2)] × 1
→ [Conv2D(64,3×3) → BN → ReLU → MaxPool(2×2)] × 1
→ [Conv2D(128,3×3) → BN → ReLU → MaxPool(2×2)] × 1
→ [Conv2D(256,3×3) → BN → ReLU] × 1
→ GlobalAveragePooling2D
→ Dense(256, ReLU) → Dropout(0.5)
→ Dense(4, Softmax)
```

### Training Setup
- Optimizer: Adam (lr=1e-3)
- Loss: categorical_crossentropy
- Epochs: up to 50 (EarlyStopping patience=7)
- Callbacks: EarlyStopping · ReduceLROnPlateau(factor=0.3, patience=4) · ModelCheckpoint
- Augmentation: rotation ±15° · horizontal+vertical flip · zoom ±10% · width/height shift ±10%
- Input size: 224×224

### Target: ≥ 90% test accuracy

---

## Phase 3 — Transfer Learning (`notebooks/03_transfer_learning.ipynb`)

Fine-tune 4 pretrained ImageNet models with identical custom head:
```
Base Model (frozen) → GlobalAveragePooling2D → Dense(256, ReLU) → Dropout(0.5) → Dense(4, Softmax)
```

| Model | Input Size | Expected Accuracy |
|---|---|---|
| MobileNetV2 | 224×224 | ~97–98% |
| VGG16 | 224×224 | ~97–99% |
| ResNet50V2 | 224×224 | ~98–99% |
| EfficientNetB3 | 300×300 | ~98–99% |

### Two-Stage Fine-Tuning
1. **Stage 1** (10 epochs): freeze all base layers, train head only at lr=1e-3
2. **Stage 2** (20 epochs): unfreeze top 30 layers, fine-tune end-to-end at lr=1e-5

---

## Phase 4 — Ensemble (`notebooks/04_ensemble_eval.ipynb`)

- Soft-voting ensemble of top 3 transfer models
- Load saved weights, average softmax outputs
- **Target: ≥ 99% test accuracy**
- Full comparative table: all models + ensemble

---

## Phase 5 — Explainability & Visualizations (`notebooks/05_explainability.ipynb`)

### Grad-CAM
- Generate heatmaps for 2 correct + 1 misclassified sample per class
- Overlay on original images
- Saves to `outputs/figures/gradcam_*`

### LIME
- Superpixel-based explanation for 1 sample per class
- Shows which image regions drive the prediction

### t-SNE Feature Space
- Extract penultimate-layer features from best model
- 2D t-SNE plot colored by class → shows learned representation quality

---

## Phase 6 — Final Evaluation (all in `notebooks/04_ensemble_eval.ipynb`)

Per model:
- [ ] Test accuracy & loss
- [ ] Training/validation loss and accuracy curves
- [ ] Confusion matrix heatmap
- [ ] Classification report (precision · recall · F1 per class)
- [ ] ROC curves + AUC (one-vs-rest)

Comparative:
- [ ] Bar chart: test accuracy of all models
- [ ] Table: all metrics side-by-side

---

## Phase 7 — TEST_SIMPLE Inference (`notebooks/06_test_simple.ipynb`)

- Run best model on all 72 TEST_SIMPLE images
- Display image grid with predicted label + confidence score
- Highlight misclassified cases

---

## Phase 8 — Report (`report/`)

LaTeX report in Persian (dark theme, Vazir font) covering:
1. Introduction & motivation
2. Dataset description & EDA findings
3. Custom CNN architecture & design choices
4. Transfer learning approach & fine-tuning strategy
5. Results & comparative analysis (tables + figures)
6. Grad-CAM & explainability discussion
7. Conclusion

---

## File Structure

```
CNN-IML/
├── PLAN.md
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── config.yaml          # all hyperparameters & paths
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_custom_cnn.ipynb
│   ├── 03_transfer_learning.ipynb
│   ├── 04_ensemble_eval.ipynb
│   ├── 05_explainability.ipynb
│   └── 06_test_simple.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # ImageDataGenerator pipelines
│   ├── augmentation.py      # augmentation config
│   ├── evaluate.py          # metrics, confusion matrix, ROC
│   ├── visualize.py         # plotting utilities
│   ├── gradcam.py           # Grad-CAM implementation
│   └── models/
│       ├── __init__.py
│       ├── custom_cnn.py    # build_custom_cnn()
│       └── transfer.py      # build_transfer_model()
├── outputs/
│   ├── figures/             # all saved plots
│   └── logs/                # CSV training logs
├── saved_models/            # .keras saved weights (gitignored)
└── report/
    ├── *.tex                # LaTeX source
    ├── assets/              # figures for report
    └── fonts/               # Vazir font files
```

---

## Scoring Checklist

| Item | Points |
|---|---|
| Custom CNN + training + evaluation | Core |
| Transfer learning (≥2 models) | Core |
| Confusion matrix + classification report | Core |
| Multiple model comparison | Core |
| Data augmentation | +marks |
| Grad-CAM visualizations | +marks |
| Ensemble model | +marks |
| LIME / explainability | +marks |
| t-SNE feature space | +marks |
| ROC/AUC curves | +marks |
| TEST_SIMPLE predictions with confidence | +marks |
| Clean, reproducible code + report | +marks |
