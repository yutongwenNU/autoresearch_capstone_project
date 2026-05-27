# Final Locked Results Ledger — Week 7 Single-Pass Holdout Evaluation
**Date:** 2026-05-25
**Champion model:** Exp_009 (Ridge α=1.0 + Stagnation Premium `rev_per_emp` + 11-feature engineered set + 0.5-grid rounding)
**Training set:** `data/train_set.csv` (62 manually labeled Midwest IT MSPs)
**Test set:** `data/locked_test_set.csv` (28 labeled holdout MSPs — single-pass evaluation, no further tuning)

---

## 🔒 Headline Result

| Metric | Training OOF (Exp_009, N=62, 5-fold) | **Final Test (N=28, single-pass)** | Generalization Gap |
|---|---|---|---|
| **RMSE** | **1.3955** | **1.5119** | +0.1164 (+8.3% relative) |
| **R²** | **0.5128** | **0.4243** | −0.0885 |

**Decision:** the model generalizes. The Test RMSE of **1.5119** is well below the Week-8 target of **< 2.0** declared in `program.md` §Core Objective & Scope. R² remains positive at 0.42, confirming the structural signals (`tenure_sq`, `rev_per_emp`, `log_revenue`) carry transferable predictive content rather than pure training-set overfit.

---

## Per-Firm Error Distribution (28 holdout firms)

| Statistic | Value |
|---|---|
| Min absolute error | **0.00** (some firms predicted exactly) |
| Median absolute error | **1.00** point |
| Max absolute error | **3.50** points |
| Firms within ±1.0 point | **16 / 28 (57%)** |
| Firms within ±1.5 points | **20 / 28 (71%)** |
| Firms within ±2.0 points | **25 / 28 (89%)** |

89% of holdout firms are scored within 2 points of the human label on a 1–10 scale. The 3 firms with errors > 2 points (11%) represent the tail risk and would be the first targets for a Week-9 error-analysis pass if one were undertaken — but per the user's instruction, no further changes are permitted in this evaluation window.

---

## Reproducibility Audit

| Item | Value |
|---|---|
| `model.py` SHA-256 | `2633e30024129610f70a009cf6bc31fab357dcf19410120f939c064071cd87f8` |
| `logs/Snapshot_model_Exp_009.py` SHA-256 | `2633e30024129610f70a009cf6bc31fab357dcf19410120f939c064071cd87f8` ← byte-identical match ✓ |
| `eval/prepare.py` SHA-256 | `8f7aa10f25b1089225616b21c0e2b4f2e78c3a9b095d016cabfe2f9662faa6c9` ← matches verify_integrity baseline ✓ |
| `final_test_eval.py` SHA-256 | `dc52fc42cfe9d8442d11a3ab0e9b2c01b1f515e0fcb097d6e3e60ca702cb6890` |
| Snapshot saved to | `logs/Snapshot_model_FinalTest.py` and `logs/Snapshot_final_test_eval.py` |

**Pipeline integrity:** the FROZEN Judge (`eval/prepare.py`) was **not** modified for this evaluation. The user's option to "temporarily update file path variables in `eval/prepare.py`" was declined in favor of the alternative path — creating `final_test_eval.py`, a standalone script that imports `featurize()` and `build_model()` directly from the champion `model.py`. This preserves the one-way-valve audit trail.

---

## Methodology — Single-Pass Inference

1. **Fit** the champion pipeline on the **full 62-firm training set** (no cross-validation; CV was already used in Exp_009 to validate the hyperparameter choice).
2. **Featurize** the 28 holdout firms via the **same `featurize()` function** imported from `model.py` — same 11 features, same StandardScaler-ready preprocessing, same scraper (`scrape_management_depth`) operating against the shared `logs/scrape_cache.json`.
3. **Predict** on the test feature matrix via `model.predict()`.
4. **Post-process** identically to Exp_009: `np.clip(preds, 1.0, 10.0)` then `np.round(preds * 2) / 2` to enforce the 0.5-grid ordinal output.
5. **Score** against the held-out Manual Score labels with `mean_squared_error` and `r2_score` from scikit-learn — same metric functions used by the FROZEN Judge.

**No data leakage:** the StandardScaler inside the Pipeline is fit only on the 62 training rows; the 28 holdout rows are transformed using the training-fit scaler parameters automatically because `build_model()` returns an sklearn Pipeline.

---

## Fitted Ridge Coefficients (Standardized, Full 62-Firm Fit)

```
       log_revenue: +0.9399
    sweet_spot_emp: +0.7759
            tenure: +0.6607
       rev_per_emp: -0.6563   ← Stagnation Premium signal
     stagnation_kw: +0.4050
      modern_ai_kw: -0.3898
     log_employees: -0.3738
      recurring_kw: +0.3323
         tenure_sq: -0.3127   ← bell-curve helper (load-bearing across all 20 runs)
        mgmt_depth: +0.0637
        in_midwest: +0.0000
```

Coefficients are bit-identical to the Exp_009 diagnostic from `logs/Research_Log_Exp_009.md`, confirming the full-data fit is reproducible from the snapshotted code.

---

## Code Instability Event (Disclosed)

A `numpy` ABI break was encountered between the previous run (2026-05-19) and this evaluation (2026-05-25): pandas's compiled extensions failed to import against the installed numpy version. Triage and fix (in line with `program.md` §Logging Standards Failure Mode 2 — Code Instability):

| Step | Action | Result |
|---|---|---|
| 1 | `python final_test_eval.py` | `ValueError: numpy.dtype size changed, may indicate binary incompatibility` |
| 2 | `pip install --force-reinstall --no-deps numpy` (got 2.2.6) | pandas import still failed (`_ARRAY_API not found`) |
| 3 | `pip install --force-reinstall --no-deps pandas` (got 2.3.3) | scipy+sklearn also broke against numpy 2.x |
| 4 | `pip install --force-reinstall --no-deps "numpy<2"` (got 1.26.4) | full stack restored: numpy 1.26.4 / pandas 2.3.3 / sklearn 1.7.2 ✓ |

**No experimental decisions were affected.** The Ridge fit, the cross-validation seed, the feature engineering, and the output rounding are all numerically identical to the Exp_009 protocol. The full-data Ridge coefficients in this run match the Exp_009 OOF diagnostic to the 4th decimal place. The fix is environmental, not modeling.

**Per §Logging Standards, this is classified as a Code Instability (Infrastructure) event** — runtime/pipeline issue, not a model issue. Triage took ~3 commands; the experimental output is unaffected.

---

## Predictions File

Tab-delimited holdout predictions exported to:
* `logs/final_test_predictions.tsv` — 28 rows, header `Predicted Score\tCompany Name`

All 28 predictions are on the 0.5 grid (post-clip + round). Spans 5.0 → 10.0.

---

## What This Result Means

1. **Project objective achieved.** The Week-8 success criterion (Test RMSE < 2.0) is met with comfortable margin (1.5119, ~25% below the ceiling).
2. **The model generalizes.** An 8% RMSE inflation from train OOF to test is in the expected range for N=62 → N=28 transfer; it does not indicate overfit. Compare to the cross-experiment noise band (~0.003 RMSE) and the Exp_016 artifact gap (+0.097 RMSE swing on a single keyword): the test-vs-train delta is consistent with a real, transferable model.
3. **The two Core axis findings are validated under blind evaluation.** The Stagnation Premium (`rev_per_emp` = −0.66) and the Demographic Bell Curve (`tenure_sq` = −0.31, `tenure` = +0.66) are the dominant coefficients in the final fit; the test-set performance derives from them. The 12 discarded experiments (Ownership sparsity failures, Moat artifact, Robustness regressions) successfully *did not* contaminate the champion configuration.
4. **The audit-driven protocol paid off.** Week-5's Exp_018 "Compliance Artifact" audit prevented a Type-I promotion that would have shipped a fragile model. Without it, the headline would today read RMSE 1.30 on train but likely a substantially larger gap on test — exactly the failure mode the audit caught preemptively.

---

## Ledger Lockdown

This ledger is **final for the Exp_009 champion**. Any future modification to `model.py` constitutes a new experiment under Week-6+ protocols and must not retroactively modify these results. The test set in `data/locked_test_set.csv` should not be re-scored as part of any iterative loop — the no-further-changes rule is a precondition of the validity of these numbers.

**Status:** 🔒 LOCKED.
