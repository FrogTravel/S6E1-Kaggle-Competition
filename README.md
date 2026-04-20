# Kaggle Playground Series S6E1 — Predicting Student Test Scores

Solution for the [January 2026 Kaggle Playground Series](https://kaggle.com/competitions/playground-series-s6e1).

**Final placement: 1246 / 4319** (top ~29%)
**Public LB RMSE: ~8.70** (best single run: `8.70111`)
**Local CV RMSE (20% holdout): 8.7365**

## Task

Predict `exam_score` (continuous) for ~630k synthetic student records given 10 features: `age`, `gender`, `course`, `study_hours`, `class_attendance`, `internet_access`, `sleep_hours`, `sleep_quality`, `study_method`, `facility_rating`, `exam_difficulty`. Metric is RMSE.

## Approach summary

A blend of two gradient boosted tree models (XGBoost + CatBoost) on a shared preprocessing pipeline. Blend weight is found by hill-climbing on a validation split. Final predictions are clipped to the observed target range before submission.

```
              ┌──────────────────────────┐
              │  shared preprocessor     │
              │  (target enc + OHE + num)│
              └───────┬──────────┬───────┘
                      │          │
               ┌──────▼────┐  ┌──▼──────────┐
               │  XGBoost  │  │  CatBoost   │
               │ 1000 est. │  │ 5000 iter.  │
               └──────┬────┘  └──┬──────────┘
                      │          │
                      └────┬─────┘
                           │
                  hill-climb on RMSE
                           │
                       np.clip(min, max)
                           │
                       submission
```

## Feature engineering & preprocessing

EDA on `groupby(col).exam_score.mean()` showed the strongest signal in `sleep_quality`, `facility_rating`, and `study_method` (spreads of 5–12 points), while `age`, `gender`, `course`, `internet_access`, and `exam_difficulty` were nearly flat (spreads < 1.5 points).

Given that shape, the final encoding scheme was:

| Feature | Encoding |
|---|---|
| `sleep_quality` | target encoding (mean exam_score per category) |
| `internet_access` | target encoding |
| `gender` | target encoding |
| `study_method` | target encoding |
| `facility_rating` | target encoding |
| `exam_difficulty` | target encoding |
| `course` | one-hot encoding |
| `age` | passthrough (numeric) |
| numeric columns | passthrough |

Target encodings were precomputed global means from the full training set — simple and somewhat leaky, but on 630k rows with low-cardinality categoricals the bias is minimal and it worked better in practice than OHE across the board (target encoding everything except `course` knocked local CV from ~8.757 down to ~8.736).

Additional feature attempts (kept in the notebook but ultimately not helpful):
- `feature_formula = sleep_quality / sleep_hours` — neutral
- `interaction_1 = study_hours * class_attendance` — neutral
- `sleep_hours` binning — slightly worse
- Concatenating the [original exam score prediction dataset](https://www.kaggle.com/datasets/exam-score-prediction-dataset) — slightly worse (~8.757)

## Models

**XGBoost**
```python
XGBRegressor(
    random_state=42,
    n_estimators=1000,
    learning_rate=0.05,
    n_jobs=4,
)
```
`n_estimators` was tuned by hand: 500 underfit, 1000 was the sweet spot, 1250+ overfit (val RMSE climbed from 8.7607 at 1000 to 8.8094 at 5000). Solo LB: **8.72425**.

**CatBoost**
```python
CatBoostRegressor(
    depth=8,
    learning_rate=0.03,
    iterations=5000,
    loss_function="RMSE",
    random_seed=42,
)
```
Solo local RMSE: **8.7443**.

## Blending

Two-model hill climbing on the weight `w` in `w * xgb + (1 - w) * cat`, starting at `w=0.5` with step `0.05` halving down to `1e-3`. This gave a consistent ~0.015 RMSE drop over the better single model.

```python
def hill_climb_weight(y_true, pred1, pred2, init_w=0.5, step=0.05, min_step=1e-3, max_iter=200):
    w, best = init_w, rmse(y_true, init_w*pred1 + (1-init_w)*pred2)
    for _ in range(max_iter):
        improved = False
        for delta in (+step, -step):
            wn = w + delta
            if 0 <= wn <= 1:
                s = rmse(y_true, wn*pred1 + (1-wn)*pred2)
                if s < best:
                    best, w, improved = s, wn, True
        if not improved:
            step /= 2
            if step < min_step: break
    return w, best
```

Best blended local RMSE: **8.7365**.

## Post-processing

Predictions are clipped to `[y_valid.min(), y_valid.max()]`. This gave a small but consistent improvement on both LB and CV — the GBDTs occasionally predict outside the observed range, and the target is naturally bounded (it's a test score).

## What worked

- **Target encoding over OHE** for most categoricals — best single change (~0.02 RMSE).
- **XGBoost + CatBoost blend** — the two models' errors were uncorrelated enough that even a simple linear blend helped.
- **Early stopping via `n_estimators` search** instead of grid search — cheap and effective.
- **Clipping predictions** to the observed target range.

## What didn't work

- Linear models (OLS / Ridge / Lasso / ElasticNet) — best was ~9.77 RMSE, far behind any tree model.
- LightGBM alone — 8.82, worse than both XGB and CatBoost.
- Stacking (LGBM + Ridge → RF meta-learner) — no gain over the simple 2-model blend.
- Hand-crafted interaction features (`study_hours * class_attendance`, `sleep_quality / sleep_hours`).
- Binning `sleep_hours`.
- Appending the original (non-synthetic) exam score dataset to training.

## Things I would try with more time

- Proper k-fold target encoding (instead of global means) to reduce leakage.
- K-fold CV for blend weight and for honest RMSE estimates — I used a single 80/20 split.
- Bayesian hyperparameter search for CatBoost depth and learning rate.
- A third model with different inductive bias (e.g. a small MLP) to add to the blend.
- Pseudo-labeling from the confident test-set predictions.

## Files

- `catboost-opt-target-encoding.ipynb` — full notebook with the pipeline, experiment log, and submission generation.

## Reproducing

1. Place `train.csv`, `test.csv`, `sample_submission.csv` under `../input/playground-series-s6e1/`.
2. Run the notebook top to bottom. Output is written to `sample_submission.csv` in the working directory.
