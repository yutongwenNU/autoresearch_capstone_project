# Experiment Log: Auto-Private-Equity Search Engine
**Date:** 2026-05-05
**Experiment ID:** exp_007 (Week 4 Controlled Experiment Set — Isolation Run #1)
**System-assigned ID in `logs/results.tsv`:** `exp_007` — IDs are now aligned (the exp_006 Meeting-Demo cleanup re-synced auto-numbering).
**Status:** keep (logged per run flag); substantively a **Signal Failure** — see Taxonomy + Decision sections.

## Experiment: Isolation Run #1 — Lasso(alpha=0.1) Feature Pruning

### Configuration
* **Worker:** `model.py` — `Pipeline([StandardScaler, Lasso(alpha=0.1, max_iter=10000, random_state=42)])`. The `max_iter=10000` (vs sklearn default 1000) and the seed are the only Lasso-specific knobs touched; alpha is the variable under test.
* **Judge:** `eval/prepare.py` (FROZEN, SHA-256 `8f7aa10f25b1...` verified before run — the re-baselined hash from the exp_006 Judge update).
* **Single change vs exp_006 Control:** regressor swap **Ridge(alpha=1.0) → Lasso(alpha=0.1)**. Everything else held constant.
* **Fixed variables (held identical to Control):** 10-feature set (`log_employees`, `log_revenue`, `tenure`, `tenure_sq`, `sweet_spot_emp`, `in_midwest`, `recurring_kw`, `stagnation_kw`, `modern_ai_kw`, `mgmt_depth`); StandardScaler before regressor; `cross_val_predict` with `KFold(n_splits=5, shuffle=True, random_state=42)`; clip to [1.0, 10.0]; **0.5-grid rounding via `np.round(preds * 2) / 2`**.

### Hypothesis
L1 regularization performs automated feature selection by driving uninformative coefficients to exactly zero (unlike Ridge's L2 which only shrinks them). Two specific predictions:

1. **Lasso will prune `mgmt_depth`** — Ridge gave it +0.0426 in exp_003/exp_006, the smallest non-zero coefficient. The exp_003 diagnostic already flagged this as likely noise from coarse role-title regex counting.
2. **Lasso will prune `in_midwest`** — Ridge gave it exactly 0.0000 already; L1 will also discard it.

If pruning these noisy features simplifies the model without losing real signal, RMSE on the discretized grid should improve.

### Result
| Metric | exp_006 Control | **exp_007 Lasso** | Δ vs Control |
|---|---|---|---|
| `val_rmse` | 1.5044 | **1.7168** | **+0.2124 (+14.1% relative)** |
| `val_r2`   | 0.4338 | **0.2626** | **−0.1712** |
| Non-zero features | 9/10 | **6/10** | −3 features |

R² fell from 0.43 → 0.26: the Lasso model now explains 17 percentage points less variance than the Control. RMSE rose by ~14% — well outside the run-to-run noise band established by the rounding-step Control (which moved RMSE by only 0.0028).

### Diagnostic — Lasso coefficients (standardized)
```
sweet_spot_emp: +0.7081       tenure: +0.2602
 stagnation_kw: +0.3393  log_revenue: +0.2536
  modern_ai_kw: −0.2962 recurring_kw: +0.2264
 log_employees:  0.0000  (pruned)
     tenure_sq:  0.0000  (pruned)
    in_midwest:  0.0000  (pruned)
    mgmt_depth:  0.0000  (pruned)
```
**4 of 10 features pruned to zero: `log_employees`, `tenure_sq`, `in_midwest`, `mgmt_depth`.**

### Causal Account — Which Features Lasso Zeroed and Why It Hurt

The pruning pattern decomposes into three distinct stories — only one of which actually costs RMSE.

**1. `in_midwest` and `log_employees` — no real loss.** Ridge already had `in_midwest` at exactly 0.000 in the Control diagnostic, and `log_employees` at −0.025 (effectively zero). Lasso pruning these is consistent with what L2 had already done implicitly. These prunings carry no information loss vs the Control.

**2. `mgmt_depth` — Hypothesis #1 confirmed; cost ≈ zero.** Lasso pruned the scraped `mgmt_depth` feature (Ridge had it at +0.0426 — smallest non-zero coefficient). This empirically validates the exp_003 diagnostic that the role-title regex count was too noisy to discriminate. Whatever fraction of the +0.21 RMSE regression comes from this prune is small, because Ridge's +0.04 weight contributed only marginally to predictions in the first place.

**3. `tenure_sq` — Hypothesis-orthogonal collateral damage; this is where the RMSE went.** Ridge had `tenure_sq` at **−0.348** in the Control (the third-largest absolute coefficient). Combined with positive `tenure` at +0.71, that pair encodes the **inverted-U "established but not ancient" sweet spot** central to the search-fund thesis: predicted score rises with tenure up to a peak (~17–25 years) and then falls off for very old firms. **Lasso pruned `tenure_sq` outright and shrunk `tenure` from +0.71 to +0.26.** What was a bell curve over tenure has collapsed to a weak, monotonically-increasing line. For ~25-year-old firms the model now under-predicts; for 40+ year-old firms it over-predicts. This single pruning explains most of the +0.21 RMSE shift.

**Why alpha=0.1 was too aggressive.** With only 62 firms and 10 features, the L1 penalty's "minimum coefficient magnitude needed to survive" was set high enough to exceed `tenure_sq`'s post-standardization weight. A smaller alpha (e.g., 0.01–0.03) would likely retain `tenure_sq` and the structurally important `tenure` magnitude while still pruning the noisy features. The hypothesis was **half-validated**: Lasso did prune the predicted noise features, but it also pruned a feature carrying real bell-curve signal, and the RMSE went the wrong way.

### Verdict on the Stated Hypothesis
| Sub-hypothesis | Prediction | Outcome |
|---|---|---|
| Lasso prunes `mgmt_depth` | Yes | **Confirmed** (coef 0.0000) |
| Lasso prunes `in_midwest` | Yes | **Confirmed** (coef 0.0000) |
| Pruning improves RMSE on 0.5 grid | Yes | **Falsified** (+14.1% regression) |

The "automated feature selection helps" reasoning was correct in *direction* on the noise features but wrong about *net effect* — Lasso's L1 mechanism cannot distinguish "noisy" from "structurally necessary but smaller-magnitude after standardization," and at alpha=0.1 it culled both.

### Taxonomy: Signal Failure
RMSE did not improve. Per the Week-4 taxonomy, this run is classified as a **Signal Failure** — the modeling change was executed cleanly (no Code Instability, no scrape regression, no integrity-lock issue) but the variable under test produced a worse predictive signal than the Control. The diagnostic value (knowing exactly which features Lasso prunes at this alpha) is the recovered upside; the metric movement is the Signal Failure.

### Decision
* The Isolation Run is logged with `--keep` per the run instruction (Week-4 protocol: every Isolation Run row stays in the table for traceability — both successes and failures).
* **Substantively, exp_006 Control (Ridge + scraper + 0.5 rounding, RMSE 1.5044) remains the operative best.** Lasso(alpha=0.1) does not earn a switch.
* `model.py` is currently in the **Lasso state** and a snapshot has been saved to `logs/Snapshot_model_Exp_007.py` for reproducibility.
* **Per the Week-4 Snapshot Protocol, `model.py` will be reverted to the exp_006 Control state (`cp logs/Snapshot_model_Exp_006.py model.py`) before the next Isolation Run is proposed.** The revert has not been executed yet — flagged here so the user can confirm or override.

### What This Likely Tells Us — for the Week-4 Set
1. **Lasso is not categorically wrong here, but alpha=0.1 is too aggressive.** A natural follow-up Isolation Run would hold the regressor as Lasso and vary only alpha (e.g., alpha=0.01) — testing whether a softer L1 retains `tenure_sq` while still pruning `mgmt_depth`. The same single-variable-at-a-time discipline applies.
2. **`tenure_sq` is structurally load-bearing for this dataset.** Any future ablation that touches the tenure encoding (e.g., binning, polynomial degree change) should expect a meaningful RMSE shift. This is now a known sensitivity.
3. **The Control's Ridge coefficients are a useful "feature importance prior."** Features Ridge weighted near zero are safe to prune in future experiments; features Ridge weighted heavily (>|0.3| standardized) should not be removed without a specific hypothesis about why their signal is illusory.
4. **The hypothesis-verdict table format above isolates *which part* of the hypothesis was right vs wrong.** Recommend adopting it for subsequent Isolation Runs — it cleanly separates "prediction confirmed" from "downstream metric impact."

### Human Feedback/Comments
*Reviewed 2026-05-05.* This is an **isolated test of regularization type** (L2 → L1) — one variable changed against the exp_006 Control. The result is a clean Signal Failure: RMSE regressed by 14% because Lasso's L1 penalty pruned `tenure_sq`, a feature that encodes the bell-curve "established but not ancient" sweet spot central to the search-fund thesis. The hypothesis that L1 would help by simplifying away noise was directionally right on the noise features (`mgmt_depth`, `in_midwest`) but wrong on net because the feature-selection mechanism cannot distinguish "noisy" from "small-magnitude-but-structurally-necessary." The substantive Control (exp_006, RMSE 1.5044) is unchanged. Snapshot at `logs/Snapshot_model_Exp_007.py` preserves reproducibility for any future re-test or alpha sweep.

### Audit
* **Judge integrity:** SHA-256 `8f7aa10f25b1...` verified prior to Worker execution (the re-baselined hash post-exp_006 Judge update).
* **Output schema:** `results.tsv` written with header `Predicted Score\tCompany Name`, tab-delimited, 62 firms scored, all values on the 0.5 grid (post-write inspection).
* **Scrape coverage:** 61 / 62 firms reachable from `logs/scrape_cache.json` (1 cached failure, unchanged from exp_003 onward; mgmt_depth was ultimately pruned anyway).
* **Lasso convergence:** ran cleanly at `max_iter=10000` (default 1000 was raised proactively to avoid `ConvergenceWarning` on the standardized 10-feature problem; no warning emitted).
* **Snapshot:** `logs/Snapshot_model_Exp_007.py` written immediately after run (8426 bytes).
* **Code Instability:** none.
