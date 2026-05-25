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
- License: CC0 (Public Domain)
- Origin: NIH (National Institutes of Health) microscopic blood smear images
- Source: [Blood Cell Images — Kaggle (paultimothymooney)](https://www.kaggle.com/datasets/paultimothymooney/blood-cells)
- Note: the Kaggle dataset is **already augmented** — our additional augmentation is on top of that

---

## Literature Review & Benchmarks

All results below are on the same 4-class Kaggle dataset (EOSINOPHIL / LYMPHOCYTE / MONOCYTE / NEUTROPHIL) unless noted.

### Custom CNN Results from Literature

| Architecture | Accuracy | Notes |
|---|---|---|
| Custom separable CNN (6 blocks) | **99.12%** | 6 conv blocks with depthwise-separable convolutions, BatchNorm, Dropout; input 120×120; SHAP+LIME explainability added — [PMC11332798](https://pmc.ncbi.nlm.nih.gov/articles/PMC11332798/) |
| Custom CNN (3 conv blocks) | 92.67% | Filters: 80→64→64, MaxPool, Dropout 0.25/0.5, Dense 128, Adadelta optimizer, 30 epochs — [Medium/TDS](https://medium.com/data-science/building-a-blood-cell-classification-model-using-keras-and-tfjs-5f7601ace931) |
| Simple 2–3 layer CNN | ~87% | Typical university project baseline — [GitHub: kunmishra2599](https://github.com/kunmishra2599/Bloodcell-Classification-CNN) |

### Transfer Learning Results from Literature

| Model | Accuracy | Notes |
|---|---|---|
| ReRNet (ResNet50 + SNN/ELM/dRVFL ensemble) | **99.97%** | ResNet50 backbone, replace final layers with 3 randomized NNs + majority voting — [PMC10061646](https://pmc.ncbi.nlm.nih.gov/articles/PMC10061646/) |
| Ensemble of 27 CNNs (ImageNet pretrained) | 99.51% | Combines predictions from 27 pretrained models — [arXiv:2110.09508](https://arxiv.org/abs/2110.09508) |
| Custom proposed model | 99.57% | Research paper, specific arch not detailed |
| VGG16 + MobileNet (per-class routing) | Lymphocyte/Monocyte: 100%, Eosinophil: 99.35%, Neutrophil: 99.81% | 2024 hybrid study |
| ResNet50 (transfer learning) | 98.31% | Standard fine-tuning — [IEEE:10561449](https://ieeexplore.ieee.org/document/10561449/) |
| VGG16 (properly fine-tuned) | 97.39%–99% | Multiple studies — proper fine-tuning is critical (see pitfall below) |
| MobileNetV2 | 92.01%–98.36% | Wide range depending on fine-tuning strategy |
| DenseNet201 | 95.21% | [PMC11332798](https://pmc.ncbi.nlm.nih.gov/articles/PMC11332798/) |
| InceptionV3 | 93.1%–97.3% | Multiple studies |
| EfficientNetB3 | ~98% (F1: 94.30%) | [arXiv:2508.06535](https://arxiv.org/abs/2508.06535) |
| VGG19 | ~91.8% | Multiple studies |
| VGG16 (poorly fine-tuned) | **53.8%** | ⚠️ Same architecture, wrong training strategy — proof that fine-tuning procedure matters enormously — [GitHub: kd91](https://github.com/kd91/Classification-blood-cell-images) |

### Key Papers

| Paper | Link | Takeaway |
|---|---|---|
| Deep CNNs for Peripheral Blood Cell Classification (2021) | [arXiv:2110.09508](https://arxiv.org/abs/2110.09508) | Ensemble of 27 CNNs → 99.51%; single best model ~97% |
| ReRNet: A Deep Learning Network for Classifying Blood Cells (2023) | [PMC10061646](https://pmc.ncbi.nlm.nih.gov/articles/PMC10061646/) | ResNet50 + randomized neural network ensemble → 99.97% |
| Explainable AI-based blood cell classification with optimized CNN (2024) | [PMC11332798](https://pmc.ncbi.nlm.nih.gov/articles/PMC11332798/) | 6-block separable CNN + SHAP/LIME → 99.12% |
| Classification of All Blood Cell Images using ML and DL (2023) | [arXiv:2308.06300](https://arxiv.org/abs/2308.06300) | Broad comparison; transfer learning range 94–99% |
| A Deep Learning Approach for Blood Cells Classification (IEEE 2024) | [IEEE:10561449](https://ieeexplore.ieee.org/document/10561449/) | VGG16, MobileNetV2, ResNet50V2, Xception head-to-head |
| White Blood Cell Classification: CNN and ViT (MDPI 2023) | [MDPI](https://www.mdpi.com/1999-4893/16/11/525) | CNN vs Vision Transformer comparison on WBC |
| Hybrid AlexNet-GoogleNet-SVM (Springer 2021) | [Springer](https://link.springer.com/article/10.1007/s42452-021-04485-9) | CNN feature extractor + SVM classifier hybrid |
| Deep transfer learning for WBC classification (Springer 2024) | [Springer](https://link.springer.com/article/10.1007/s11042-024-19133-8) | Transfer learning review on histopathological images |

### Relevant Code References

| Repo / Notebook | Approach | Accuracy |
|---|---|---|
| [paultimothymooney — Kaggle](https://www.kaggle.com/code/paultimothymooney/identify-blood-cell-subtypes-from-images) | Original dataset baseline notebook | Baseline |
| [kunmishra2599/Bloodcell-Classification-CNN](https://github.com/kunmishra2599/Bloodcell-Classification-CNN) | Custom CNN, TF/Keras | ~87% |
| [kd91/Classification-blood-cell-images](https://github.com/kd91/Classification-blood-cell-images) | Custom CNN (87%) + VGG16 (53.8% — poorly tuned) | 87% / 53.8% |
| [masoudnick/White-Blood-Cells-Classification](https://github.com/masoudnick/White-Blood-Cells-Classification) | InceptionV3 transfer learning | N/A |
| [feriniqation — Kaggle kernel](https://www.kaggle.com/code/feriniqation/cnn-blood-cells-classification) | Custom CNN Kaggle kernel | N/A |

---

## Known Pitfalls & Lessons from Literature

1. **VGG16 fine-tuning can collapse to 53%** if you unfreeze the whole base at once with a high learning rate. Always use two-stage training: head first, then low-lr fine-tune of top layers only.
2. **Vertical flip is valid** for microscopy — cells have no canonical orientation, so vertical flip is a legitimate augmentation (unlike natural images where it would be unnatural).
3. **Brightness/color jitter** simulates different staining intensities across microscope slides — useful augmentation for this domain specifically.
4. **MONOCYTE and LYMPHOCYTE are the most commonly confused pair** in the literature. Pay attention to their per-class F1 scores and check the confusion matrix carefully.
5. **Input size matters**: pretrained models were trained on 224×224 (VGG, ResNet, MobileNet) or 300×300 (EfficientNetB3). Do not resize all to 128×128 for speed — it degrades transfer learning significantly.
6. **ImageNet normalization** (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) should be used instead of simple /255 rescaling when using pretrained models, because the pretrained weights expect it. Keras applications handle this internally via their `preprocess_input` function — use it.
7. **GAP > Flatten** before the dense head for transfer learning. GlobalAveragePooling2D reduces overfitting and makes the model input-size-agnostic.
8. **The dataset is already augmented** by the original Kaggle author. Our augmentation adds further variation on top — do not over-augment.

---

## Preprocessing Pipeline

### For Custom CNN
```
load JPEG → resize to 224×224 → rescale to [0, 1] → augment (train only)
```

### For Pretrained Transfer Models
```
load JPEG → resize to model's native size → apply model's preprocess_input()
           (handles ImageNet mean/std normalization internally)
```

### Data Augmentation (train set only — never apply to val/test)
| Transform | Value | Reason |
|---|---|---|
| Rotation | ±15° | Cells appear at all orientations |
| Horizontal flip | True | Orientation-invariant |
| Vertical flip | True | Valid for microscopy images |
| Width/height shift | ±10% | Cell may not be centered |
| Zoom | ±10% | Variable magnification |
| Brightness | ±20% | Staining intensity variation across slides |
| Fill mode | nearest | Avoids black borders from shifts |

---

## Phase 1 — EDA & Visualization (`notebooks/01_eda.ipynb`)

- [ ] Sample grid: 8 images per class (32 total)
- [ ] Class distribution bar chart → confirm balance
- [ ] Pixel intensity histograms per class (check staining variation)
- [ ] Mean image per class (reveals typical cell morphology)
- [ ] Side-by-side: raw vs. augmented samples (show all transforms)
- [ ] Image dimension verification (confirm all ~360×363)

**Deliverable:** figures saved to `outputs/figures/eda_*`

---

## Phase 2 — Custom CNN from Scratch (`notebooks/02_custom_cnn.ipynb`)

### Architecture
```
Input (224×224×3)
→ [Conv2D(32, 3×3, padding=same) → BN → ReLU → MaxPool(2×2)] ×1
→ [Conv2D(64, 3×3, padding=same) → BN → ReLU → MaxPool(2×2)] ×1
→ [Conv2D(128, 3×3, padding=same) → BN → ReLU → MaxPool(2×2)] ×1
→ [Conv2D(256, 3×3, padding=same) → BN → ReLU]               ×1
→ GlobalAveragePooling2D
→ Dense(256, ReLU) → Dropout(0.5)
→ Dense(4, Softmax)
```

Design rationale:
- BatchNorm after every Conv prevents internal covariate shift and acts as regularizer
- Progressive filter doubling (32→64→128→256) is standard practice
- GAP instead of Flatten reduces parameters and overfitting
- Dropout(0.5) at the dense layer is the strongest regularization point

### Training Setup
| Param | Value |
|---|---|
| Optimizer | Adam (lr=1e-3) |
| Loss | categorical_crossentropy |
| Max epochs | 50 |
| EarlyStopping patience | 7 |
| ReduceLROnPlateau factor | 0.3, patience=4 |
| Batch size | 32 |
| Val split | 15% of TRAIN |
| Input size | 224×224 |

### Callbacks
```python
EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, min_lr=1e-7)
ModelCheckpoint('saved_models/custom_cnn_best.keras', save_best_only=True)
CSVLogger('outputs/logs/custom_cnn.csv')
```

### Target: ≥ 90% test accuracy  
Literature baseline for similar custom CNNs: 87–92.67%

---

## Phase 3 — Transfer Learning (`notebooks/03_transfer_learning.ipynb`)

### Custom Head (identical for all models)
```
Base Model (frozen) → GlobalAveragePooling2D → Dense(256, ReLU) → Dropout(0.5) → Dense(4, Softmax)
```

### Models

| Model | Input Size | Unfreeze (Stage 2) | Literature Accuracy | Notes |
|---|---|---|---|---|
| MobileNetV2 | 224×224 | top 30 layers | 92–98% | Fastest; good for lightweight deployment |
| VGG16 | 224×224 | last conv block (4 layers) | 97–99% | Only unfreeze last block — more layers risks instability |
| ResNet50V2 | 224×224 | top 30 layers | 98–99% | Best balance of speed and accuracy |
| EfficientNetB3 | 300×300 | top 30 layers | ~98% | Highest capacity; use its own `preprocess_input` |

### Two-Stage Fine-Tuning Protocol
```
Stage 1 — Feature Extraction (10 epochs)
  base.trainable = False
  optimizer = Adam(lr=1e-3)
  → trains only the custom head
  → fast convergence, no risk of destroying pretrained weights

Stage 2 — Fine-Tuning (up to 20 epochs)
  unfreeze top N layers of base (see table above)
  optimizer = Adam(lr=1e-5)   ← 100× lower than Stage 1
  → gently adapts high-level features to blood cell domain
  → EarlyStopping still active
```

**Critical**: use `model.predict` for evaluation only after `base.trainable = False` is set properly and `training=False` is passed to the base — BN layers behave differently at train vs inference time.

---

## Phase 4 — Ensemble (`notebooks/04_ensemble_eval.ipynb`)

- Load saved `.keras` weights for top 3 transfer models
- **Soft-voting**: average the softmax probability vectors, then argmax
- No retraining needed — pure inference-time combination
- Expected to push accuracy above any single model (~99%+)

```python
proba = (model_a.predict(X) + model_b.predict(X) + model_c.predict(X)) / 3
y_pred = np.argmax(proba, axis=1)
```

---

## Phase 5 — Explainability & Visualizations (`notebooks/05_explainability.ipynb`)

### Grad-CAM
- Target layer: last Conv2D layer of the model (highest spatial resolution before GAP)
- Generate for: 2 correctly classified + 1 misclassified sample **per class** (12 images total)
- Overlay heatmap on original cell image
- Show side-by-side: Original | Heatmap | Overlay
- Relevance: confirms the model looks at the cell nucleus/cytoplasm, not image artifacts

### LIME (Local Interpretable Model-agnostic Explanations)
- Superpixel segmentation of 1 sample per class
- Highlights which image regions positively/negatively drive the prediction
- Library: `lime.lime_image.LimeImageExplainer`

### t-SNE Feature Space
- Extract penultimate-layer (pre-softmax) feature vectors from best model
- Run t-SNE (perplexity=30, n_iter=1000) on TEST set features
- 2D scatter plot colored by true class
- Well-separated clusters = model has learned meaningful representations

---

## Phase 6 — Final Evaluation (`notebooks/04_ensemble_eval.ipynb`)

### Per-model metrics
- [ ] Test accuracy & loss
- [ ] Training/validation accuracy and loss curves
- [ ] Confusion matrix heatmap (seaborn, annotated counts)
- [ ] Classification report: precision · recall · F1 · support per class
- [ ] ROC curves + AUC (one-vs-rest, 4 curves per model)

### Cross-model comparison
- [ ] Bar chart: test accuracy all models + ensemble
- [ ] Summary table: accuracy · macro-F1 · training time

### Watch for: MONOCYTE vs LYMPHOCYTE confusion
These are the hardest pair in the literature. If off-diagonal counts are high there, note it in the report.

---

## Phase 7 — TEST_SIMPLE Inference (`notebooks/06_test_simple.ipynb`)

- Run best model (or ensemble) on all 72 TEST_SIMPLE images
- Display 6×12 image grid: each image annotated with predicted label + confidence %
- Color-code: green border = correct, red border = wrong
- Report overall accuracy on TEST_SIMPLE

---

## Phase 8 — Report (`report/`)

LaTeX report in Persian (dark theme, Vazir font) covering:
1. Introduction & medical motivation
2. Dataset description & EDA findings (with figures)
3. Preprocessing & augmentation design choices
4. Custom CNN architecture & rationale
5. Transfer learning: models, two-stage protocol, why each model was chosen
6. Results & comparative analysis (tables + figures from outputs/)
7. Grad-CAM & explainability — what the model learned
8. Conclusion & future work (e.g., Vision Transformers, larger datasets)

---

## File Structure

```
CNN-IML/
├── PLAN.md
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── config.yaml              # all hyperparameters & paths
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_custom_cnn.ipynb
│   ├── 03_transfer_learning.ipynb
│   ├── 04_ensemble_eval.ipynb
│   ├── 05_explainability.ipynb
│   └── 06_test_simple.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # ImageDataGenerator pipelines
│   ├── evaluate.py              # metrics, confusion matrix, ROC
│   ├── visualize.py             # plotting utilities
│   ├── gradcam.py               # Grad-CAM implementation
│   └── models/
│       ├── __init__.py
│       ├── custom_cnn.py        # build_custom_cnn()
│       └── transfer.py          # build_transfer_model(), unfreeze_top_layers()
├── outputs/
│   ├── figures/                 # all saved plots (gittracked)
│   └── logs/                    # CSV training logs (gittracked)
├── saved_models/                # .keras saved weights (gitignored — too large)
└── report/
    ├── *.tex                    # LaTeX source
    ├── assets/                  # figures for report (copied from outputs/figures/)
    └── fonts/                   # Vazir font files
```

---

## Scoring Checklist

| Item | Status |
|---|---|
| Custom CNN + training + evaluation | Core |
| Transfer learning (≥2 models) | Core |
| Confusion matrix + classification report | Core |
| Multiple model comparison table/chart | Core |
| Data augmentation (documented + shown visually) | +marks |
| Grad-CAM visualizations with overlay | +marks |
| Ensemble (soft-voting, top 3 models) | +marks |
| LIME superpixel explainability | +marks |
| t-SNE feature space visualization | +marks |
| ROC/AUC curves (one-vs-rest) | +marks |
| TEST_SIMPLE predictions with confidence grid | +marks |
| Two-stage fine-tuning (not just frozen base) | +marks |
| Clean, reproducible notebooks + LaTeX report | +marks |
