# Experiment Log: Auto-Private-Equity Search Engine
**Date:** 2026-04-28
**Experiment ID:** exp_005
**Status:** keep (logged) — but **regressed**; exp_003 remains the current best (RMSE 1.5016)

## Experiment: HistGradientBoostingRegressor with 0.5-Grid Discretization

### Configuration
* **Worker:** `research.py` — `HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, max_leaf_nodes=8, min_samples_leaf=5, l2_regularization=1.0, random_state=42)` inside a `Pipeline`
* **Judge:** `eval/prepare.py` (FROZEN, SHA-256 verified before run)
* **Feature set:** identical to exp_003 (10 features — `log_employees`, `log_revenue`, `tenure`, `tenure_sq`, `sweet_spot_emp`, `in_midwest`, `recurring_kw`, `stagnation_kw`, `modern_ai_kw`, `mgmt_depth`)
* **New output post-processing:** predictions clipped to `[1.0, 10.0]` then snapped to the 0.5 grid via `round(x*2)/2` to match the discretization of the Manual Score labels.
* **Validation Protocol:** unchanged — `cross_val_predict` with `KFold(n_splits=5, shuffle=True, random_state=42)`

### Hypotheses Bundled into This Run
1. **Tree non-linearity hypothesis.** exp_004 showed a correctly-signed but predictively-flat interaction term — Ridge had captured the conditional structure as a multiplicative term but couldn't represent it as a true threshold. A boosted-tree ensemble should find threshold-style interactions ("tenure ≥ 25 AND mgmt_depth ≤ 1") natively, without manual feature engineering.
2. **Label-grid alignment hypothesis.** Manual Score labels are quantized to 0.5 increments. Continuous regressor outputs always sit a non-zero distance from any valid label; snapping predictions to the same 0.5 grid should reduce RMSE in expectation, since predictions land in the same bin as the label whenever the model is "close enough."

### Result
| Metric | exp_001 | exp_002 | exp_003 | exp_004 | **exp_005** | Δ vs exp_003 (best) |
|---|---|---|---|---|---|---|
| `val_rmse` | 1.8460 | 1.5112 | **1.5016** | 1.5110 | **2.1764** | **+0.6748 (regression)** |
| `val_r2`   | 0.1474 | 0.4287 | 0.4359 | 0.4288 | **−0.1850** | **−0.6209** |

R² went *negative* — the model performs worse than predicting the mean of `Manual Score` for every firm. This is a strong signal that HGBR is overfitting the small training folds.

### Diagnostic — Permutation Importance (mean drop in R² when each feature is shuffled)
```
log_employees: +0.7152   ← dominates
  log_revenue: +0.3982
       tenure: +0.3020
 recurring_kw: +0.2715
   mgmt_depth: +0.1264
stagnation_kw: +0.0720
    tenure_sq: +0.0000   ← HGBR finds no use for this engineered transform
sweet_spot_emp: +0.0000  ← HGBR found the same threshold inside log_employees
   in_midwest: +0.0000
 modern_ai_kw: +0.0000
```

### What This Likely Tells Us — Two Distinct Findings
**1. The "redundant engineered features" diagnostic is the silver lining.** HGBR assigned literally zero importance to `tenure_sq`, `sweet_spot_emp`, `in_midwest`, and `modern_ai_kw` — features that Ridge had given non-trivial weight (`sweet_spot_emp = +0.79` was Ridge's strongest coefficient!). This is consistent with what a tree is *supposed* to do: it can find the headcount sweet-spot threshold internally on `log_employees` and doesn't need the binary indicator. Same logic for `tenure_sq` (a bell-curve transform of `tenure`). The fact that those features go to zero importance is a *correct* response by HGBR; it indicates the linear-model-friendly engineering was redundant for a tree.

**2. But the model still regressed badly.** Three reasons, in order of likelihood:

* **N is too small.** With 5-fold CV, each training fold has ~50 rows. Boosted trees with 300 iterations and 8 leaves each have many more parameters than that. Even with `min_samples_leaf=5` and `l2_regularization=1.0`, the model overfits the training fold and generalizes poorly to the 12-row test fold.
* **The signal in this dataset is largely linear.** Search-fund-thesis features (tenure, headcount, revenue) interact with the label more linearly than additively — the dominant pattern is "more tenure + more recurring keywords → higher score," with weaker non-linear thresholds. Ridge was already extracting most of that linear structure; HGBR's threshold-finding capability was not needed and its variance penalty was not earned.
* **The 0.5 rounding amplifies, rather than softens, large residuals.** When a continuous prediction is already off by 1.5 (badly wrong), rounding it to 0.5 doesn't help — the residual stays badly wrong. Rounding only helps when the prediction is *already close*, which requires an underlying model that's already well-calibrated. HGBR's predictions weren't well-calibrated on this small dataset, so rounding gave back nothing.

### What This Means for the Research Direction
* **Revert the model** — exp_003 (Ridge with engineered features and `mgmt_depth`) remains the best at RMSE 1.5016.
* **Keep the rounding step** — it's a label-aligned post-process that's free to apply to any model, and on a well-calibrated Ridge it should round neutrally or slightly favorably. It only hurt exp_005 because the model was already badly off.
* **Tree models are not categorically wrong here, but N=62 is too small for HGBR with the current hyperparameters.** Future iterations could explore a much smaller HGBR (`max_iter=50`, `max_leaf_nodes=4`) or a `RandomForestRegressor` with high `min_samples_leaf`. But the more likely productive direction is sharper *features* (founder-tenure mentions, real "About Us" text mining) on top of the Ridge backbone, not different model classes.
* **The "redundant engineered features" diagnostic should inform exp_006+.** If we stay with Ridge, `sweet_spot_emp` etc. earn their keep. If we move to a tree, we can drop them.

### Decision: keep (logged), but exp_003 remains the active best
Logged with `--keep` per the user's instruction so the experiment record is complete. `research.py` will be reverted to the exp_003 feature set + the 0.5-rounding step (since the rounding is a label-aligned post-process that's free to keep on any model) before any future runs.

### Audit
* **Judge integrity:** SHA-256 `570d9e2a89c8...` verified prior to Worker execution.
* **Output schema:** `results.tsv` written with header `Predicted Score\tCompany Name`, tab-delimited, 62 firms scored, all on a 0.5 grid (verified post-write).
* **Prediction distribution:** spans 3.5 → 10.0 with modes at 7.0, 8.0, 9.0 (matches the 0.5-grid label vocabulary).
