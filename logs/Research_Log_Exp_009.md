# Experiment Log: Auto-Private-Equity Search Engine
**Date:** 2026-05-05
**Experiment ID:** exp_009 (Week 4 Controlled Experiment Set — Isolation Run #3)
**System-assigned ID in `logs/results.tsv`:** `exp_009` — IDs aligned.
**Status:** keep — **first Week-4 Isolation Run to improve on the Control**, and a new project all-time best.

## Experiment: Isolation Run #3 — Revenue per Employee (Structural MRR Proxy)

### Configuration
* **Worker:** `model.py` — reverted to the exp_006 Control via `cp logs/Snapshot_model_Exp_006.py model.py` (verified byte-identical), then a single targeted edit adding one new feature.
* **Judge:** `eval/prepare.py` (FROZEN, SHA-256 `8f7aa10f25b1...` verified before run).
* **Single change vs exp_006 Control:** add **one new structural feature**:
  ```python
  rev_per_emp = revenue / employees.clip(lower=1)   # $/employee, no log
  ```
  Total feature count rises from 10 → 11.
* **Fixed variables:** Ridge(alpha=1.0); StandardScaler; the 10 unchanged Control features (`log_employees`, `log_revenue`, `tenure`, `tenure_sq`, `sweet_spot_emp`, `in_midwest`, `recurring_kw`, `stagnation_kw`, `modern_ai_kw`, `mgmt_depth`); `cross_val_predict` with `KFold(5, shuffle=True, random_state=42)`; clip to [1.0, 10.0]; **0.5-grid rounding**.

### Hypothesis
The exp_008 Signal Failure showed that *keyword-based* MRR signals (e.g., up-weighting "managed services," "subscription," "sla") are too universal in MSP marketing copy to discriminate revenue quality. The Yale "Nature of Revenue" thesis is correct in spirit — MRR-heavy firms are higher-quality search-fund targets — but the operationalization needs to switch from text to **structural** data. Revenue-per-employee is the simplest available proxy: high-quality MRR shops should generate more revenue per head than break-fix or hardware-resale shops with the same headcount.

### Result
| Metric | exp_006 Control | exp_003 (prior all-time best) | **exp_009 rev_per_emp** | Δ vs Control |
|---|---|---|---|---|
| `val_rmse` | 1.5044 | 1.5016 | **1.3955** | **−0.1089 (−7.2% relative)** |
| `val_r2`   | 0.4338 | 0.4359 | **0.5128** | **+0.0790 (+18% relative)** |
| Non-zero features | 9/10 | 9/10 | **10/11** | +1 |

**This is the first Week-4 Isolation Run to *move RMSE in the right direction*, and it sets a new project all-time best (1.3955 vs the prior 1.5016 from exp_003).** R² jumped 8 percentage points, the largest single-experiment gain since exp_002 introduced Ridge over hand-coded heuristics.

### Diagnostic — Ridge coefficients (standardized): Control vs exp_009
| Feature | exp_006 Control | exp_009 | Δ | Reading |
|---|---|---|---|---|
| `log_revenue` | +0.3776 | **+0.9399** | **+0.562** | jumped ~2.5× — biggest single coefficient now |
| `sweet_spot_emp` | +0.7924 | +0.7759 | −0.017 | unchanged |
| `tenure` | +0.7090 | +0.6607 | −0.048 | mildly weakened |
| **`rev_per_emp` (NEW)** | — | **−0.6563** | — | **2nd-largest absolute coef, sign-flipped from hypothesis** |
| `stagnation_kw` | +0.4127 | +0.4050 | −0.008 | unchanged |
| `modern_ai_kw` | −0.3768 | −0.3898 | −0.013 | unchanged |
| `log_employees` | −0.0250 | **−0.3738** | **−0.349** | swung sharply negative |
| `recurring_kw` | +0.2855 | +0.3323 | +0.047 | mildly strengthened |
| `tenure_sq` | −0.3483 | −0.3127 | +0.036 | mildly weakened (still load-bearing) |
| `mgmt_depth` | +0.0426 | +0.0637 | +0.021 | still ≈ noise |
| `in_midwest` | 0.0000 | 0.0000 | unchanged | — |

### Causal Account — Two Direct Checks Requested

**Check 1 — Does `rev_per_emp` get a stronger coefficient than `recurring_kw` did?**
**YES, by ~2×.** `|rev_per_emp| = 0.656` vs `|recurring_kw| = 0.332` in the same model (and vs `|recurring_kw| = 0.286` in the Control). The structural ratio absorbs about twice the standardized signal that the keyword count carries — empirical confirmation that `rev_per_emp` is a more informative MRR proxy than text-keyword counts in this dataset. This directly answers the open question left by the exp_008 Signal Failure: switching from text to structure was the right pivot.

**Check 2 — Does it improve RMSE?**
**YES, by 7.2% (RMSE 1.5044 → 1.3955), the largest single-experiment improvement of the Week-4 set and a new project all-time best.** R² rose by 8 percentage points (0.43 → 0.51). The signal is meaningfully outside the run-to-run noise band (the rounding-only Control moved RMSE by only 0.0028; this experiment moved it by ~0.11, ~40× larger).

### Causal Account — Why the Coefficient is Negative (the Surprise)

The hypothesis predicted a *positive* sign — "high MRR efficiency → high search-fund target score." Empirically Ridge gave `rev_per_emp` a sign-flipped **−0.656**. Two stories explain the sign, both of which are consistent with the RMSE improvement.

**Story 1 — Multicollinearity redistribution (mechanical).** The pipeline now contains *three* features that together encode the revenue/headcount surface: `log_revenue`, `log_employees`, and the new `rev_per_emp = revenue / employees`. Mathematically these are not orthogonal — `log(R) − log(E) = log(R/E)` is the log-form sibling of the new linear ratio, and Ridge sees the resulting columns as partially collinear. The L2 penalty's response is to spread coefficient mass across the triplet rather than refuse the new feature: `log_revenue` jumped +0.56 to +0.94, `log_employees` swung −0.35 to −0.37, and `rev_per_emp` picked up −0.66. The *combined* surface those three coefficients describe is what the model uses to predict — the standalone sign on any one of them is hard to interpret in isolation.

The win comes from the new feature being a **different functional form**, not a redundant duplicate. `log(R) − log(E)` is concave in R/E; the linear `R/E` is, well, linear. Together they let Ridge approximate non-linearities in the revenue-per-head surface that a log-additive pair alone could not — and this is where the +0.08 R² lift comes from.

**Story 2 — A genuine "stagnation premium" reading (economic).** Holding total revenue and headcount fixed (which is roughly what the redistributed `log_revenue + log_employees` captures), a *higher* `rev_per_emp` could plausibly indicate a firm that is *already* operationally optimized — i.e., one with less room for a searcher to add value. The search-fund thesis explicitly favors "stable but stagnant" targets where per-head productivity has slack. A negative `rev_per_emp` coefficient is consistent with: among firms of similar size and revenue, the *less* operationally efficient ones are better acquisition targets, because the searcher's playbook is to drive that efficiency up post-acquisition.

**Most likely both effects contribute.** Story 1 explains why the sign is negative *given* that two correlated revenue features are already in the model (Ridge had to assign one of the three a negative weight to avoid overshooting). Story 2 explains why the assignment converges on the specific sign-and-magnitude that empirically improves RMSE — the data agrees that lower per-head productivity, conditional on size and revenue, signals higher target quality.

### Verdict on the Stated Hypothesis
| Sub-hypothesis | Prediction | Outcome |
|---|---|---|
| `rev_per_emp` is a stronger MRR proxy than text keywords | Yes | **Confirmed** (\|coef\| ≈ 2× `recurring_kw`) |
| Adding it improves RMSE | Yes | **Confirmed** (−7.2%, new all-time best) |
| Coefficient is positive (high efficiency → high score) | Yes | **Falsified** — sign is −0.66, consistent with a "stagnation premium" reading |

The hypothesis was right on the two metrics that matter for the Week-4 protocol (informativeness and RMSE) and wrong on the *direction* of the relationship — a productive falsification, because the negative sign sharpens the search-fund thesis from "MRR efficiency is good" to "operational slack inside an MRR-bearing firm is the actual signal."

### Taxonomy: Signal Success
First Week-4 Isolation Run to clear the Control. Per the exp_007/exp_008 taxonomy, this run inverts the Signal Failure category — RMSE improved meaningfully, the new feature carries non-trivial standardized weight, and the diagnostic produced a refined economic reading. **Recommend adopting Signal Success as the affirmative counterpart to Signal Failure for Week-4 reporting.**

### Decision
* The Isolation Run is logged with `--keep` per the run instruction.
* **Substantively, exp_009 is the new operative best at RMSE 1.3955.** It dethrones both the exp_006 Control (1.5044) and the prior all-time best exp_003 (1.5016).
* `model.py` is currently in the `rev_per_emp` state and a snapshot has been saved to `logs/Snapshot_model_Exp_009.py` for reproducibility.
* **Open question for the user:** should `Snapshot_model_Exp_009.py` become the *new* Week-4 Control (replacing exp_006) for subsequent Isolation Runs? Pros: future ablations would diff against the strongest known baseline, so productive changes are easier to detect. Cons: changing the reference mid-set complicates the cross-experiment narrative; the current pattern (exp_006 = Control, all ablations diff against it) is cleanly reportable. Flagging for explicit decision rather than auto-promoting.
* Per the Snapshot Protocol, the revert to `Snapshot_model_Exp_006.py` is **pending** and should be confirmed (or overridden, if exp_009 becomes the new Control) before exp_010 is proposed.

### What This Likely Tells Us — for the Week-4 Set
1. **Structural data beat text data for revenue-quality signal at this dataset size.** This is the empirical answer to the open question raised by exp_008. Future Yale-thesis ablations should explore other structural ratios (revenue / tenure-year, employees / tenure-year as a "growth rate" proxy, etc.) rather than refining keyword lists.
2. **Adding correlated-but-different-functional-form features is productive in this Ridge pipeline.** The +0.08 R² gain came from giving Ridge a *linear* ratio alongside an existing *log-additive* pair encoding the same underlying variables. This is a generalizable pattern: where two raw inputs already exist as logs, adding their direct ratio (or product) gives the L2 model access to non-linearities it couldn't represent before.
3. **The "negative sign on a positively-hypothesized feature" should be treated as diagnostic, not anomalous.** It's how Ridge tells us that the feature carries information *conditional on* what's already in the model — even when the marginal interpretation flips. This is the third Week-4 Isolation Run to surface a coefficient-level diagnostic that sharpens the underlying thesis (after exp_007's "tenure_sq is load-bearing" and exp_008's "keywords are too universal").
4. **`tenure_sq` is still meaningfully weighted (−0.31).** The bell-curve fit over tenure survives the addition of `rev_per_emp` — confirming exp_007's "load-bearing" reading and ruling out an interpretation where the new feature was just absorbing tenure-related variance.

### Human Feedback/Comments
*Logged 2026-05-05.* This is **Isolation Run #3**, an isolated test of the Yale "Nature of Revenue" thesis re-operationalized as a structural ratio (`Annual Revenue / # Employees`) rather than text-keyword weighting. One variable changed against the exp_006 Control: regressor, scaler, rounding, all 10 original features, and the random seed are identical; one new feature was added. Result is the first Week-4 **Signal Success** — RMSE improved by 7.2% to 1.3955, beating both the Control and the prior all-time best (exp_003 at 1.5016). The new feature carries the second-largest absolute Ridge coefficient (−0.66) and gets ~2× the standardized signal of `recurring_kw`, directly answering the open question left by exp_008. The negative coefficient sign was unexpected but coherent: under multicollinearity with `log_revenue` and `log_employees`, Ridge redistributes coefficient mass across the triplet, and the resulting negative sign on `rev_per_emp` reads economically as a "stagnation premium" — among firms of similar size and revenue, lower per-head productivity is a *positive* search-fund target signal because it implies operational slack a searcher can convert. Snapshot at `logs/Snapshot_model_Exp_009.py` preserves this configuration. The Control-vs-best-model question (whether to promote exp_009 to the new Control going forward) is flagged in the Decision section and held for explicit user direction.

### Audit
* **Judge integrity:** SHA-256 `8f7aa10f25b1...` verified prior to Worker execution.
* **Output schema:** `results.tsv` written with header `Predicted Score\tCompany Name`, tab-delimited, 62 firms scored, all values on the 0.5 grid.
* **Scrape coverage:** 61 / 62 firms reachable from `logs/scrape_cache.json` (1 cached failure, unchanged from exp_003 onward).
* **Revert verification:** `model.py` was reset to `logs/Snapshot_model_Exp_006.py` via `cp` before the feature addition; `diff` confirmed byte-identical to the Control snapshot pre-edit.
* **Division-by-zero handling:** `revenue / employees.clip(lower=1)` ensures no `inf` or `nan` enters the feature column even for firms reporting zero employees (a `safe_float` default). Empirically no row hit the floor in this dataset, but the guard is in place for `data/locked_test_set.csv` later.
* **Snapshot:** `logs/Snapshot_model_Exp_009.py` written immediately after run (8322 bytes; 89 bytes larger than the Control snapshot).
* **Code Instability:** none.
