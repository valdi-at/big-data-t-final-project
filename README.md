# Exoplanet Classification: Combined Kepler + TESS Dataset

| Name                        | NRP        |
| --------------------------- | ---------- |
| Mohamad Valdi Ananda Tauhid | 5025221238 |

## TL;DR

Trains a Random Forest classifier on labeled transit signals from both the Kepler and TESS missions to predict whether a transit signal is a real planet or a false positive. Achieves 0.95 ROC-AUC on a held-out test set. Also breaks down performance per telescope to check whether the model generalizes across missions.

---

## Data

Both datasets are downloaded from the NASA Exoplanet Archive TAP service.

| Source                        | Rows  | Labeled rows used |
| ----------------------------- | ----- | ----------------- |
| Kepler KOI Cumulative Catalog | 9,564 | 7,327             |
| TESS TOI List                 | 7,931 | 2,398             |

- Kepler: [cumulative table](https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv)
- TESS: [toi table](https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+toi&format=csv)

Unlabeled candidates (`CANDIDATE` in Kepler, `PC` and `APC` in TESS) are dropped since they have no ground-truth label. After combining and cleaning, the working dataset has 9,725 rows.

---

## What We(I) Did

### Preprocessing

Kepler and TESS use different column names and disposition schemes, so they are aligned before combining. TESS column names are remapped to Kepler's convention. Dispositions are encoded as binary labels: confirmed planets (Kepler `CONFIRMED`, TESS `CP`/`KP`) become 1, and false positives (`FALSE POSITIVE`, `FP`, `EB`) become 0.

`koi_impact` is excluded from the feature set. TESS does not provide it, and filling half the rows with a constant zero would mislead the model.

### Feature Engineering

Two features are computed from existing columns:

- `radius_ratio`: planet radius divided by stellar radius, converted to the same units. When this disagrees with the depth-implied ratio, it often signals a diluted eclipsing binary masquerading as a planet.
- `transit_duty_cycle`: transit duration as a fraction of the orbital period. Real planets tend to have tight, consistent values; outliers usually point to contamination or a grazing geometry.

This brings the total feature count to 9 (7 base + 2 engineered).

### Model

A scikit-learn `Pipeline` with two steps:

1. `StandardScaler` - centers and scales each feature to unit variance, computed on training data only.
2. `RandomForestClassifier` - 200 trees, `class_weight='balanced'` to compensate for the ~58/42 false-positive-to-confirmed imbalance.

The dataset is split 80/20 with stratification, giving 7,780 training rows and 1,945 test rows.

### Evaluation

Test-set results on the combined dataset:

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 0.887 |
| Precision | 0.827 |
| Recall    | 0.921 |
| F1        | 0.871 |
| ROC-AUC   | 0.955 |

Per-mission breakdown on the same test set:

| Mission | N     | Accuracy | F1    | ROC-AUC |
| ------- | ----- | -------- | ----- | ------- |
| Kepler  | 1,449 | 0.908    | 0.885 | 0.967   |
| TESS    | 496   | 0.827    | 0.843 | 0.908   |

The model performs better on Kepler rows, which is expected given that ~75% of training data comes from Kepler. The TESS drop in precision suggests TESS false positives are harder to separate in this feature space, likely because TESS targets brighter, noisier stars.

---

## How to Run

```bash
# install dependencies
pip install -r requirements.txt

# open the notebook
jupyter notebook notebook.ipynb
```

Run cells top to bottom. The notebook expects `kepler_cumulative.csv` and `tess_toi.csv` in the working directory.
