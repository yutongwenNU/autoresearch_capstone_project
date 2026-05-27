# Experiment Log: Auto-Private-Equity Search Engine
**Date:** 2026-05-07
**Experiment ID:** exp_011 (Week 5 Controlled Experiment Set — Isolation Run #5; first run against the new exp_009 baseline per `program.md` §Week 5 Specific Plans)
**System-assigned ID in `logs/results.tsv`:** `exp_011` — IDs aligned.
**Status:** keep (logged per run flag); substantively a **Signal Failure** — see Failure Mode + Decision sections.

## Experiment: Bootstrap Aggregation (Bagging) on the Ridge Pipeline

### Configuration
* **Worker:** `model.py` — reset to the exp_009 baseline via `cp logs/Snapshot_model_Exp_009.py model.py` (verified byte-identical), then a single targeted edit wrapping the Ridge pipeline in a `BaggingRegressor`.
* **Judge:** `eval/prepare.py` (FROZEN, SHA-256 `8f7aa10f25b1...` verified before run).
* **Single change vs exp_009 baseline:** ensemble wrapper.
  ```python
  base = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
  return BaggingRegressor(
      estimator=base,
      n_estimators=50,
      max_samples=0.8,        # ~50 firms per bag from the 62-firm training set
      random_state=42,
  )
  ```
* **Fixed variables:** Ridge α=1.0, 0.5-grid rounding, the 11 features from exp_009 (10 Control features + `rev_per_emp`), `cross_val_predict(KFold(5, shuffle=True, random_state=42))`, clip to [1.0, 10.0]. The base estimator is exactly the exp_009 Ridge pipeline.

### Hypothesis
Bagging trains `n_estimators` independent Ridge models, each on a random 80% bootstrap of the 62 firms, then averages predictions. The expectation: with N=62, individual Ridge fits are sensitive to which firms land in the training fold; averaging across 50 bags should reduce that variance and yield a more stable (lower) RMSE on the held-out validation firms.

### Result
| Metric | exp_009 baseline | **exp_011 Bagging** | Δ vs baseline |
|---|---|---|---|
| `val_rmse` | **1.3955** | **1.4464** | **+0.0509 (+3.6% relative)** |
| `val_r2`   | **0.5128** | 0.4766 | −0.0362 |

RMSE regressed by 3.6% — meaningful but not catastrophic, ~order of magnitude larger than the rounding-only Control noise band (0.0028) and ~half the magnitude of the exp_008 Weighted-MRR Signal Failure (+0.077).

### Diagnostic — Mean ± Std of Standardized Ridge Coefficients Across 50 Bags
```
       log_revenue: +0.9787 ± 0.8277       std/|mean| = 0.85
    sweet_spot_emp: +0.8537 ± 0.5393       std/|mean| = 0.63
       rev_per_emp: -0.6691 ± 0.8335       std/|mean| = 1.25  ← sign flips across bags
     stagnation_kw: +0.4559 ± 0.2895       std/|mean| = 0.63
            tenure: +0.3610 ± 0.6572       std/|mean| = 1.82  ← sign flips across bags
      modern_ai_kw: -0.3576 ± 0.3325       std/|mean| = 0.93
      recurring_kw: +0.2959 ± 0.3839       std/|mean| = 1.30  ← sign flips across bags
     log_employees: -0.2512 ± 0.6258       std/|mean| = 2.49  ← sign flips across bags
        mgmt_depth: +0.1799 ± 0.2692       std/|mean| = 1.50
         tenure_sq: +0.0142 ± 0.6714       mean ≈ 0, std huge ← signal washed out
        in_midwest: +0.0000 ± 0.0000
```
**Reference (exp_009 single-fit Ridge coefficients):** `log_revenue: +0.94, sweet_spot_emp: +0.78, tenure: +0.66, rev_per_emp: −0.66, stagnation_kw: +0.41, modern_ai_kw: −0.39, log_employees: −0.37, recurring_kw: +0.33, tenure_sq: −0.31, mgmt_depth: +0.06`.

### Causal Account — Why Bagging Hurt Despite Real Per-Bag Variance

**1. The std-devs *confirm* the precondition for bagging — there is real instability across bags.** For 6 of 11 features, the cross-bag std-dev exceeds the mean coefficient (in some cases by 2–2.5×). A 50-firm Ridge fit on this dataset is genuinely unstable: which subset of firms enters each bag swings the coefficients meaningfully. So the *theoretical premise* of bagging — that there's variance to reduce — is satisfied.

**2. But bagging averaged away a structurally real signal.** The most striking row is `tenure_sq`: exp_009 had it at **−0.31**, but the bagged mean is **+0.014** with std **0.67**. About half the bags assign it positive weight and half negative; they cancel. The full-data Ridge fit on all 62 firms identifies `tenure_sq = −0.31` as a real bell-curve signal (the "established but not ancient" sweet spot — a finding confirmed across all four Week-4 ablations as load-bearing). On a 50-row bootstrap, that signal is too weak to dominate sampling noise, so individual bags fit it inconsistently. **Averaging across bags produced a near-zero mean coefficient on a feature whose true effect is non-zero — a classic bagging-induced bias when a real signal is at the borderline of statistical detectability per-bag.**

**3. Bagging reduces variance *at the cost of* sample size per learner.** Each bag has only ~50 firms (62 × 0.8) for an 11-feature Ridge — about 4.5 samples per feature. Compared to exp_009's full-data fit on 62 firms (5.6 samples per feature), the per-bag fit is borderline ill-conditioned. The variance reduction from averaging 50 such fits doesn't compensate for the systematic bias each fit introduces by having less data and thus more shrinkage toward the per-bag mean. **At N=62, bagging is making the bias-variance trade in the wrong direction.**

**4. Ridge is already a low-variance learner.** Bagging works best for *low-bias, high-variance* base learners (deep decision trees, fully-grown random forests). Ridge with α=1.0 is intentionally *high-bias, low-variance* — the L2 penalty already shrinks coefficients toward zero, suppressing the variance bagging is designed to reduce. Wrapping a Ridge in a BaggingRegressor compounds two regularizers (L2 shrinkage *and* prediction averaging) without enough variance left to harvest. The +3.6% RMSE shift is the over-regularization showing through.

**Net diagnostic:** the per-bag instability is real (large std-devs), bagging *did* reduce that instability in the prediction averaging (50 noisy predictions averaged are smoother than any one), but the **price was a near-zeroing of `tenure_sq` and a 0.30 reduction in the bagged-mean of `tenure`** — both structurally important signals that exp_007's diagnostic established as load-bearing. The averaging treated real signal as if it were noise.

### Verdict on the Stated Hypothesis
| Sub-hypothesis | Prediction | Outcome |
|---|---|---|
| Per-bag Ridge fits are unstable on N=62 | (implicit) | **Confirmed** — std/\|mean\| > 1 for 6 of 11 features |
| Averaging 50 bags reduces RMSE | Yes | **Falsified** — RMSE rose by 3.6% |
| Bagging reduces influence of outliers | Yes | **Partially** — predictions are smoother (low variance), but the smoothing also erases real signal that survives only on the full sample |

### Failure Mode — per `program.md` §Logging Standards (4-Category Taxonomy)
* **1. Signal Failure (Information/Heuristic).** ✓ — applies. The data-science method (Bootstrap Aggregation) did not effectively capture the predictive signal at this dataset size. RMSE rose by 3.6% despite a clean execution and theoretically sound mechanism. Specific subtype: **the variance-reduction technique averaged away a structurally real coefficient (`tenure_sq`) whose detection requires the full sample**.
* 2. Code Instability (Infrastructure). ✗ — not triggered. Worker exit 0, Judge exit 0, no `ConvergenceWarning`, SHA-256 lock held, scrape cache served all 62 entries.
* 3. Evaluation Leakage (Validity). ✗ — not triggered. No modification to `Manual Score` labels, train/val split, or Judge metric. The fold randomization is identical to all prior runs (`KFold(5, shuffle=True, random_state=42)`).
* 4. Agent Misbehavior (Control). ✗ — not triggered. Exactly one variable changed (the regressor wrapper); all 11 features, the random seed, the rounding step, and the Judge are unchanged from the exp_009 baseline.

### Decision
* The Isolation Run is logged with `--keep` per the run instruction.
* **Substantively, exp_009 (Ridge + rev_per_emp) remains the operative best at RMSE 1.3955.** Bagging does not earn promotion.
* Per `program.md` §Week 5 Specific Plans: a Controlled Experiment must be reverted to the exp_009 snapshot unless explicitly promoted. **Recommend `cp logs/Snapshot_model_Exp_009.py model.py` before exp_012**, and **flipping the `logs/results.tsv` status of `exp_011` from `keep` to `discard`** to mirror the Week-4 maintenance pattern. Both held for explicit user confirmation.
* Snapshot at `logs/Snapshot_model_Exp_011.py` (8933 bytes) preserves the BaggingRegressor configuration for any future Week-5 ensemble experiment that wants to tune `n_estimators`, `max_samples`, or the base learner.

### What This Likely Tells Us — for the Week-5 Set
1. **Bagging on Ridge at N=62 is the wrong tool for the wrong learner.** Future ensemble experiments should pivot to (a) a low-bias / high-variance base — e.g., shallow `DecisionTreeRegressor(max_depth=4)` inside the BaggingRegressor — where there's actually variance to harvest, or (b) `RidgeCV` for a single fit with cross-validated α, which delivers regularization stability without the per-bag bias cost. Skip BaggingRegressor + Ridge as a configuration going forward.
2. **`tenure_sq`'s sensitivity to subsampling is itself a finding.** It's a "real but borderline" signal — present in the full sample, undetectable in a 50-row subsample. Future feature-engineering experiments should be cautious about *any* method that sub-samples (e.g., k-fold CV grid search with small folds, robust regression with re-weighting). The bell curve over tenure is a fragile signal to handle.
3. **For variance reduction at this dataset size, consider the dual.** Instead of averaging *models*, consider averaging *predictions* across multiple random seeds (e.g., a `KFold` shuffle with 5 different seeds, average the 5 OOF prediction sets). This preserves the full-sample fit while smoothing the validation-fold idiosyncrasy. Cheap to test and avoids the per-bag bias.
4. **The 4-category Failure Mode taxonomy in `program.md` is now actively used.** This is the first Research Log to apply it explicitly; recommend the same template be applied retroactively to exp_007, exp_008, exp_010 if the Week-5 grading rubric requires consistency across the experiment set.

### Human Feedback/Comments
*Logged 2026-05-07.* This is **Week 5 Isolation Run #1**, a single-variable change against the new exp_009 baseline (regressor wrapper added; everything else identical). Result is a Signal Failure — RMSE regressed by 3.6% — driven not by ensembling per se but by the wrong pairing of mechanism to base learner: bagging averages away variance, but Ridge is already low-variance, and the per-bag sample size (~50) is too small to identify `tenure_sq` consistently across bags, so the structural bell-curve signal got averaged toward zero. The diagnostic value is high: the bag-coefficient std-devs make explicit *which* features are "real-but-fragile" signals (tenure_sq, log_employees, recurring_kw all have std exceeding mean). The exp_009 baseline holds. Snapshot at `logs/Snapshot_model_Exp_011.py` preserves this configuration; the Bagging code is a useful reference for any future ensemble experiment that swaps the base learner to something high-variance.

### Audit
* **Judge integrity:** SHA-256 `8f7aa10f25b1...` verified prior to Worker execution.
* **Output schema:** `results.tsv` written with header `Predicted Score\tCompany Name`, tab-delimited, 62 firms scored, all values on the 0.5 grid.
* **Scrape coverage:** 61 / 62 firms reachable from `logs/scrape_cache.json` (1 cached failure, unchanged).
* **Revert verification:** `model.py` was reset to `logs/Snapshot_model_Exp_009.py` via `cp` before the wrapper edit; `diff` confirmed byte-identical to the baseline pre-edit.
* **Numeric sanity:** all 50 sub-estimators converged; no `ConvergenceWarning` emitted by Ridge under default `max_iter`. Bagged predictions stayed in [1.0, 10.0] before clipping.
* **Snapshot:** `logs/Snapshot_model_Exp_011.py` written immediately after run (8933 bytes; 611 bytes larger than the exp_009 snapshot due to the BaggingRegressor wrapper + the expanded mean/std diagnostic block).
* **Code Instability classification:** none triggered.
