# Exoplanet Classification: Combined Kepler + TESS Dataset

| Name                        | NRP        |
| --------------------------- | ---------- |
| Mohamad Valdi Ananda Tauhid | 5025221238 |

## TL;DR

Trains four classifiers (Random Forest, XGBoost, LightGBM, CatBoost) on labeled transit signals from both the Kepler and TESS missions to predict whether a transit signal is a real planet or a false positive. All four models reach ~0.95 ROC-AUC on a held-out test set. The notebook compares them across six metrics, cross-validation stability, training speed, and per-mission performance, then picks a winner.

---

## Topic & Importance

Transit photometry is basically a planet passing in front of its star blocks a sliver of starlight, and that dip in brightness is detectable from Earth. The problem is that eclipsing binaries, instrument noise, and background contaminants produce nearly identical dips, and at the scale of Kepler and TESS the false positives pile up fast. About 58% of Kepler's labeled signals are false positives. Confirming a real planet requires follow-up spectroscopy that takes months per candidate, so a classifier that can triage thousands of signals from the archive alone is worth building. This project pulls from both missions because Kepler and TESS observe different stellar populations, which gives the model something harder to overfit to than a single telescope's quirks.

## Data

Both datasets are downloaded from the NASA Exoplanet Archive TAP service.

| Source                        | Rows  | Labeled rows used |
| ----------------------------- | ----- | ----------------- |
| Kepler KOI Cumulative Catalog | 9,564 | 7,327             |
| TESS TOI List                 | 7,931 | 2,398             |

- Kepler: [cumulative table](https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv)
- TESS: [toi table](https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+toi&format=csv)

Unlabeled candidates (`CANDIDATE` in Kepler, `PC` and `APC` in TESS) are dropped since they have no ground-truth label. After combining and cleaning, the working dataset has 9,725 rows.

### Kepler — `koi_disposition` (target label)

| Value            | Meaning                                                       | Used          |
| ---------------- | ------------------------------------------------------------- | ------------- |
| `CONFIRMED`      | Signal independently verified as a real planet                | Yes → label 1 |
| `FALSE POSITIVE` | Ruled out; usually an eclipsing binary or instrument artifact | Yes → label 0 |
| `CANDIDATE`      | Not yet vetted; no ground-truth label                         | Dropped       |

### TESS — `tfopwg_disp` (target label)

Assigned by the TESS Follow-up Observing Program Working Group (TFOPWG).

| Value | Meaning                                                                 | Used                     |
| ----- | ----------------------------------------------------------------------- | ------------------------ |
| `CP`  | Confirmed Planet — independently verified                               | Yes → label 1            |
| `KP`  | Known Planet — previously known planet re-detected by TESS              | Yes → label 1            |
| `FP`  | False Positive — ruled out                                              | Yes → label 0            |
| `EB`  | Eclipsing Binary — a stellar pair whose mutual eclipses mimic a transit | Yes → label 0            |
| `PC`  | Planet Candidate — not yet vetted                                       | Dropped                  |
| `APC` | Ambiguous Planet Candidate                                              | Dropped                  |
| `FA`  | False Alarm — instrumental or systematic artifact                       | Not in labeled rows used |

### Features

TESS column names are remapped to Kepler's convention before combining. The table below shows both names and the physical quantity each represents.

| Kepler column  | TESS column   | Description                                      | Unit        |
| -------------- | ------------- | ------------------------------------------------ | ----------- |
| `koi_period`   | `pl_orbper`   | Orbital period                                   | days        |
| `koi_prad`     | `pl_rade`     | Planet radius                                    | Earth radii |
| `koi_depth`    | `pl_trandep`  | Transit depth (fraction of stellar flux blocked) | ppm         |
| `koi_duration` | `pl_trandurh` | Transit duration                                 | hours       |
| `koi_steff`    | `st_teff`     | Stellar effective temperature                    | K           |
| `koi_srad`     | `st_rad`      | Stellar radius                                   | Solar radii |
| `koi_insol`    | `pl_insol`    | Insolation flux received by the planet           | Earth flux  |

`koi_impact` (the transit impact parameter) is excluded. TESS does not provide it, so including it would mean filling half the rows with a placeholder value.

Two additional features are computed from the columns above:

| Feature              | Formula                            | Why                                                                                                       |
| -------------------- | ---------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `radius_ratio`       | `koi_prad / (koi_srad × 109.076)`  | When this disagrees with the depth-implied ratio, it often flags a diluted eclipsing binary               |
| `transit_duty_cycle` | `koi_duration / (koi_period × 24)` | Real planets tend to have tight, consistent values; outliers point to contamination or a grazing geometry |

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

### Models

All four models are wrapped in a scikit-learn `Pipeline` with `StandardScaler` as the first step, so features measured in large units (stellar temperature in Kelvin) don't swamp features measured in small ones (radius ratio). Each handles the ~58/42 class imbalance differently:

| Model         | Imbalance handling                                    |
| ------------- | ----------------------------------------------------- |
| Random Forest | `class_weight='balanced'`                             |
| XGBoost       | `scale_pos_weight` (negative-to-positive count ratio) |
| LightGBM      | `is_unbalance=True`                                   |
| CatBoost      | `auto_class_weights='Balanced'`                       |

The dataset is split 80/20 with stratification, giving 7,780 training rows and 1,945 test rows.

### Evaluation

Test-set results for all four models:

| Model         | Accuracy | Precision | Recall | F1     | ROC-AUC |
| ------------- | -------- | --------- | ------ | ------ | ------- |
| Random Forest | 0.8869   | 0.8267    | 0.9208 | 0.8712 | 0.9548  |
| XGBoost       | 0.8859   | 0.8178    | 0.9332 | 0.8717 | 0.9571  |
| LightGBM      | 0.8812   | 0.8167    | 0.9208 | 0.8656 | 0.9552  |
| CatBoost      | 0.8853   | 0.8197    | 0.9282 | 0.8706 | 0.9568  |

XGBoost edges out the others on both F1 and ROC-AUC. All four models are close enough that the differences are not practically meaningful at this scale. LightGBM trains fastest at 0.34s; XGBoost is the slowest at ~30s.

Per-mission breakdown (Random Forest, as the baseline):

| Mission | N     | Accuracy | F1    | ROC-AUC |
| ------- | ----- | -------- | ----- | ------- |
| Kepler  | 1,449 | 0.908    | 0.885 | 0.967   |
| TESS    | 496   | 0.827    | 0.843 | 0.908   |

All models perform better on Kepler rows, which is expected given that ~75% of training data comes from Kepler. The TESS drop in precision suggests TESS false positives are harder to separate in this feature space, likely because TESS targets brighter, noisier stars.

---

## How to Run

```bash
# install dependencies
pip install -r requirements.txt

# open the notebook
jupyter notebook notebook.ipynb
```

Run cells top to bottom. The notebook expects `kepler_cumulative.csv` and `tess_toi.csv` in the working directory.
