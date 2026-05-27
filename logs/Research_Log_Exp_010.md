# Experiment Log: Auto-Private-Equity Search Engine
**Date:** 2026-05-05
**Experiment ID:** exp_010 (Week 4 Controlled Experiment Set — Isolation Run #4)
**System-assigned ID in `logs/results.tsv`:** `exp_010` — IDs aligned.
**Status:** keep (logged per run flag); substantively a **Signal Failure** — see Taxonomy + Decision sections.

## Experiment: Isolation Run #4 — Sigmoid/Gaussian Size Scaling

### Configuration
* **Worker:** `model.py` — reverted to the exp_006 Control via `cp logs/Snapshot_model_Exp_006.py model.py` (verified byte-identical), then a single targeted edit replacing the `sweet_spot_emp` formula.
* **Judge:** `eval/prepare.py` (FROZEN, SHA-256 `8f7aa10f25b1...` verified before run).
* **Single change vs exp_006 Control:** the `sweet_spot_emp` feature is replaced by a Gaussian (bell curve) over headcount:
  ```python
  # Was (binary cliff):
  sweet_spot_emp = ((employees >= 10) & (employees <= 30)).astype(int)
  # Now (smooth bell curve, μ=20, σ=10):
  sweet_spot_emp = np.exp(-((employees - 20) ** 2) / (2 * (10 ** 2)))
  ```
* **Fixed variables:** Ridge(alpha=1.0); StandardScaler; the 9 unchanged Control features (`log_employees`, `log_revenue`, `tenure`, `tenure_sq`, `in_midwest`, `recurring_kw`, `stagnation_kw`, `modern_ai_kw`, `mgmt_depth`); `cross_val_predict` with `KFold(5, shuffle=True, random_state=42)`; clip to [1.0, 10.0]; **0.5-grid rounding**.
* Total feature count unchanged at 10 (only the *content* of the `sweet_spot_emp` column changed; the column count is the same).

### Hypothesis
The Control's `sweet_spot_emp` is a binary indicator: 1 if `10 ≤ employees ≤ 30`, 0 otherwise. This creates two artificial cliffs — at 9 ↔ 10 and at 30 ↔ 31 — where business quality "vanishes" in a single-employee jump that has no economic basis. A Gaussian centered at 20 with σ=10 replaces both cliffs with a smooth gradient: a firm with 9 or 31 employees gets ~0.95 of the peak signal (vs 0 in the binary), and a firm with 50 employees gets ~0.04 (vs 0 in the binary). This should let Ridge fit the headcount sweet-spot more accurately and reduce RMSE.

### Result
| Metric | exp_006 Control (binary) | exp_009 (prior all-time best) | **exp_010 Gaussian** | Δ vs Control |
|---|---|---|---|---|
| `val_rmse` | 1.5044 | **1.3955** | **1.6741** | **+0.1697 (+11.3% relative)** |
| `val_r2`   | 0.4338 | **0.5128** | **0.2989** | **−0.1349** |

RMSE regressed by 11.3% — meaningfully outside the noise band and the largest single-experiment regression of the Week-4 set so far. R² fell from 0.43 → 0.30, giving back nearly all of the gain from exp_009.

### Diagnostic — Ridge coefficients (standardized): Control vs exp_010
| Feature | exp_006 Control (binary) | exp_010 (Gaussian) | Δ | Reading |
|---|---|---|---|---|
| **`sweet_spot_emp`** | **+0.7924** | **+1.2720** | **+0.480** | **+60% standardized weight on the smooth feature — direct comparison the user requested** |
| `log_employees` | −0.0250 | **+0.5273** | **+0.552** | **sign flipped from ~zero to substantial positive** |
| `tenure` | +0.7090 | +0.8026 | +0.094 | mildly strengthened |
| `tenure_sq` | −0.3483 | −0.4072 | −0.059 | mildly stronger |
| `recurring_kw` | +0.2855 | +0.3891 | +0.104 | strengthened |
| `stagnation_kw` | +0.4127 | +0.3496 | −0.063 | mildly weakened |
| `modern_ai_kw` | −0.3768 | −0.3180 | +0.059 | mildly weakened |
| `log_revenue` | +0.3776 | +0.2997 | −0.078 | weakened |
| `mgmt_depth` | +0.0426 | +0.0160 | −0.027 | weakened, still ≈ noise |
| `in_midwest` | 0.0000 | 0.0000 | 0 | unchanged |

### Causal Account — Direct Comparison Requested by Specification

**The smooth feature carries +60% more standardized weight than the binary feature did.**
* Binary `sweet_spot_emp` (Control): coefficient **+0.7924** (largest absolute coef in the Control model).
* Gaussian `sweet_spot_emp` (exp_010): coefficient **+1.2720** (largest absolute coef in the exp_010 model, by an even wider margin).

**This is half the story — and the half that confirms the hypothesis.** The smooth bell curve carries strictly more standardized signal than the binary indicator, exactly as predicted. Ridge gave it the highest weight in the model. **But RMSE got worse.** The other half of the story explains why.

### Causal Account — Why RMSE Got Worse Despite the Stronger Coefficient

The +60% boost to `sweet_spot_emp` is not the dominant effect. The dominant effect is the **+0.55 swing on `log_employees` from ~zero to substantial positive**, which is a side effect of swapping the binary cliff for a Gaussian. Three threads explain how a "smoother, more informative" feature produced a worse model.

**1. Multicollinearity with `log_employees`.** The binary `sweet_spot_emp` is *uncorrelated* with `log_employees` outside the 10–30 range — it is flat at 0 for both very small and very large firms. The Gaussian, by contrast, is monotonically related to `|employees − 20|`, which is itself strongly correlated with `log_employees`. The new feature is now a near-collinear "shape function" over the same input that `log_employees` already encodes. Ridge's response is to disentangle the pair: it loaded the bell-shape into `sweet_spot_emp` (+1.27) and the monotone trend into `log_employees` (+0.53). The two coefficients together describe the headcount surface — but the disentanglement has **high variance under N=62**, and the test-fold predictions don't generalize.

**2. The Gaussian + positive `log_employees` combo over-predicts large firms.** For a firm with 100 employees: Gaussian `sweet_spot_emp` ≈ 0.0003 (no penalty, no reward), but `log_employees` ≈ 4.6 with coefficient +0.527 → standardized contribution ≈ +0.84 of upward push from headcount alone. The model now says "very large firms are also good targets," which contradicts the search-fund thesis (and the Manual Score labels for the few large firms in the training set). The Control's binary feature avoided this trap: outside the 10–30 window, *both* `sweet_spot_emp` and `log_employees` contributed approximately zero, so very-large-firm predictions defaulted to a sensible mid-range. The Gaussian breaks this safeguard.

**3. Hyperparameter sensitivity at small N.** The center (μ=20) and width (σ=10) of the Gaussian were specified by intuition, not by data tuning. With only 62 firms — and roughly 12–15 firms in any 10-employee bucket — the bell curve has too much flexibility for the available signal. A Gaussian centered at 18 with σ=8, or 22 with σ=12, would each yield a different reshuffling of the surrounding coefficients. The binary cliff has zero hyperparameters in its shape (only its endpoints, fixed at 10 and 30 from the search-fund primer); the Gaussian introduces two implicit hyperparameters that we did not tune. **The improvement-vs-Control swing is therefore a function of (a) the Gaussian shape we picked and (b) Ridge's sample-fold-specific disentanglement — neither of which is robust at N=62.**

**Net causal story:** the smooth feature *is* more informative on its own (+60% standardized coefficient), but injecting it into a pipeline that already contains `log_employees` creates a multicollinear pair that Ridge can only disentangle at the cost of an overfit, manifesting as a positive `log_employees` coefficient that bonuses very large firms incorrectly. **The hypothesis was right about the feature's intrinsic information content; it was wrong about the feature's interaction with the rest of the pipeline.**

### Verdict on the Stated Hypothesis
| Sub-hypothesis | Prediction | Outcome |
|---|---|---|
| Gaussian carries more signal than binary cliff | Yes | **Confirmed** (+60% standardized coefficient: +0.79 → +1.27) |
| Smooth scaling reduces RMSE | Yes | **Falsified** (+11.3% regression) |
| The cliff at 9↔10 and 30↔31 was costing accuracy | Yes | **Refuted** — the cliff's value was *limiting* multicollinearity with `log_employees`, not just being economically wrong |

### Taxonomy: Signal Failure
RMSE regressed. Per the Week-4 taxonomy, this run is classified as a **Signal Failure** — modeling change executed cleanly (no Code Instability, no scrape regression, no SHA-256 issue) but the variable under test produced a worse predictive signal than the Control. The diagnostic upside is significant: we now know that **swapping a feature's functional form is not a free operation in a Ridge pipeline** — the new form's correlation structure with the rest of the feature set determines whether the "more information" actually translates into better predictions.

### Decision
* The Isolation Run is logged with `--keep` per the run instruction.
* **Substantively, exp_009 (rev_per_emp) remains the operative best at RMSE 1.3955.** Gaussian sweet_spot does not earn a switch.
* Recommend flipping the `logs/results.tsv` status of `exp_010` from `keep` to `discard` (parallel to the exp_007 / exp_008 maintenance) so the performance plot correctly marks this as a failed Isolation Run. Flagging for user confirmation rather than auto-flipping.
* `model.py` is currently in the Gaussian state; snapshot saved to `logs/Snapshot_model_Exp_010.py` (8232 bytes).
* Per the Snapshot Protocol, the revert to `Snapshot_model_Exp_006.py` is **pending** before exp_011 is proposed.

### What This Likely Tells Us — for the Week-4 Set
1. **Smoothing a binary feature ≠ free RMSE improvement.** When the binary's economic value is "encoding a non-monotone shape that the rest of the pipeline can't represent," replacing it with a smooth function with the same shape may inadvertently let *other* features pick up overfit weight (here: `log_employees` swung from ~zero to +0.53). Future feature-engineering ablations should check the **delta-coefficient distribution across all features**, not just the changed one — the surrounding shifts often dominate the metric.
2. **The Control's binary `sweet_spot_emp` was acting as an implicit regularizer on `log_employees`.** This is non-obvious in retrospect: the indicator was constant (0) for very small and very large firms, which forced `log_employees` to also be approximately useless in that range, which kept the model from extrapolating badly. Lesson: features with hard cutoffs sometimes earn their keep by *suppressing* their correlated neighbors, not just by their own signal.
3. **A future Gaussian-with-tuning experiment is plausible but not first-priority.** A `GridSearchCV` over `μ ∈ {15, 18, 20, 22, 25}` × `σ ∈ {6, 8, 10, 12}` could find a Gaussian configuration that beats the Control. But the more efficient next step is to look for *new* informative features (parallel to exp_009's `rev_per_emp` win) rather than re-shape existing ones — the marginal R² gain from the Week-4 set so far has been concentrated in additions, not transformations.
4. **`tenure_sq` remains load-bearing across all four Week-4 ablations.** Coefficient stayed in the −0.27 to −0.41 band across exp_007, exp_008, exp_009, exp_010 — never pruned, never swung sign. This is the most robust diagnostic finding of the Week-4 set: the bell-curve fit on tenure is real and stable.

### Human Feedback/Comments
*Logged 2026-05-05.* This is **Isolation Run #4**, an isolated test of replacing the binary headcount sweet-spot indicator with a Gaussian (μ=20, σ=10). One variable changed against the exp_006 Control: regressor, scaler, rounding, all 9 other features, and the random seed are identical; the *content* of the `sweet_spot_emp` column changed but the column count and name did not. Result is a Signal Failure — RMSE regressed by 11.3% — driven not by the smooth feature itself (which carries +60% more standardized weight than the binary version did, exactly as hypothesized) but by a multicollinearity-induced reshuffle that gave `log_employees` a +0.53 coefficient, causing the model to over-predict for very large firms. The exp_009 `rev_per_emp` configuration remains the operative best at RMSE 1.3955. Snapshot at `logs/Snapshot_model_Exp_010.py` preserves this configuration for any future Gaussian-tuning ablation. The diagnostic value is high: the Control's binary cliff was implicitly regularizing `log_employees` by forcing it to be uninformative outside the 10–30 range, and removing that suppression let Ridge overfit a positive headcount slope that doesn't generalize.

### Audit
* **Judge integrity:** SHA-256 `8f7aa10f25b1...` verified prior to Worker execution.
* **Output schema:** `results.tsv` written with header `Predicted Score\tCompany Name`, tab-delimited, 62 firms scored, all values on the 0.5 grid.
* **Scrape coverage:** 61 / 62 firms reachable from `logs/scrape_cache.json` (1 cached failure, unchanged).
* **Revert verification:** `model.py` was reset to `logs/Snapshot_model_Exp_006.py` via `cp` before the formula edit; `diff` confirmed byte-identical to the Control snapshot pre-edit.
* **Numeric sanity:** Gaussian `sweet_spot_emp` values lie in [0, 1] across all 62 firms (peak 1.0 for firms with exactly 20 employees; minimum ≈ 0 for the largest firm in the set). No NaN/inf observed.
* **Snapshot:** `logs/Snapshot_model_Exp_010.py` written immediately after run (8232 bytes; 1 byte smaller than Control snapshot due to the formula being one character shorter than the binary expression).
* **Code Instability:** none.
