# Experiment Log: Auto-Private-Equity Search Engine
**Date:** 2026-05-05
**Experiment ID:** exp_008 (Week 4 Controlled Experiment Set — Isolation Run #2)
**System-assigned ID in `logs/results.tsv`:** `exp_008` — IDs aligned.
**Status:** keep (logged per run flag); substantively a **Signal Failure** — see Taxonomy + Decision sections.

## Experiment: Isolation Run #2 — Weighted Revenue Quality (Yale "Nature of Revenue" Thesis)

### Configuration
* **Worker:** `model.py` — reverted to the exp_006 Control via `cp logs/Snapshot_model_Exp_006.py model.py` (verified byte-identical), then a single targeted edit to the keyword feature.
* **Judge:** `eval/prepare.py` (FROZEN, SHA-256 `8f7aa10f25b1...` verified before run).
* **Single change vs exp_006 Control:** the `recurring_kw` feature now uses a **weighted** count over two disjoint keyword lists:
  * **Premium MRR (weight 2.5x):** `"managed services", "managed it", "recurring", "mrr", "sla", "monthly subscription", "subscription"`
  * **Standard service (weight 1.0x):** `"help desk", "monitoring", "backup", "compliance", "business continuity"`
  * New formula: `recurring_kw = 2.5 × count(PREMIUM_MRR) + 1.0 × count(STANDARD_SERVICE)`
* **Fixed variables:** Ridge(alpha=1.0); StandardScaler; 9 unchanged features (`log_employees`, `log_revenue`, `tenure`, `tenure_sq`, `sweet_spot_emp`, `in_midwest`, `stagnation_kw`, `modern_ai_kw`, `mgmt_depth`); `cross_val_predict` with `KFold(5, shuffle=True, random_state=42)`; clip to [1.0, 10.0]; **0.5-grid rounding**.

### Hypothesis
Per Yale's "On the Nature of Revenue," recurring/contracted revenue (MRR) is a higher-quality earnings stream than project work or hardware sales — and a defining feature of attractive search-fund MSP targets. The Control's `recurring_kw` feature treated all 9 service-related keywords as equal-weight binary signals, conflating a firm whose marketing copy emphasizes "managed services" and "SLA" with one that mentions "help desk" and "backup." Up-weighting the explicitly-MRR-bearing keywords by 2.5x should let Ridge place a larger coefficient on the discriminative subset and pull predictions higher for the "true MSP" subgroup.

### Result
| Metric | exp_006 Control | **exp_008 Weighted MRR** | Δ vs Control |
|---|---|---|---|
| `val_rmse` | 1.5044 | **1.5811** | **+0.0767 (+5.1% relative)** |
| `val_r2`   | 0.4338 | **0.3746** | **−0.0592** |
| `recurring_kw` Ridge coef | +0.2855 | +0.3230 | +0.038 |

RMSE regressed by ~5% — meaningfully outside the run-to-run noise band (the rounding-only Control moved RMSE by 0.0028, two orders of magnitude smaller than this shift). R² fell by 6 percentage points.

### Diagnostic — Ridge coefficients (standardized): Control vs Weighted MRR
| Feature | exp_006 Control | exp_008 Weighted | Δ |
|---|---|---|---|
| `sweet_spot_emp` | +0.7924 | +0.7851 | −0.007 |
| `tenure` | +0.7090 | +0.6745 | −0.034 |
| `stagnation_kw` | +0.4127 | +0.4296 | +0.017 |
| `log_revenue` | +0.3776 | +0.3730 | −0.005 |
| `modern_ai_kw` | −0.3768 | −0.3733 | +0.004 |
| `tenure_sq` | **−0.3483** | **−0.2732** | **+0.075** ← weakened |
| `recurring_kw` | +0.2855 | **+0.3230** | **+0.038** ← target of this exp |
| `mgmt_depth` | +0.0426 | +0.0193 | −0.023 |
| `log_employees` | −0.0250 | −0.0816 | −0.057 |
| `in_midwest` | 0.0000 | 0.0000 | unchanged |

### Causal Account — Why the Weighted Signal Hurt

The shift is small and the diagnostic is rich. Three threads explain the +0.077 RMSE regression.

**1. The 2.5x weight got partially absorbed by StandardScaler.** The pipeline scales every feature to unit variance before Ridge sees it. Multiplying the raw `recurring_kw` count by 2.5 for premium keywords doesn't translate directly into a 2.5x larger Ridge coefficient — it changes the *distribution* of the column (mean and variance), but StandardScaler renormalizes both. The effective "boost" surfaces only as (a) a shift in the *rank ordering* of firms (premium-keyword-rich firms now rank higher within the column) and (b) a change in the column's correlational structure with `Manual Score`. Empirically the coefficient rose by only +0.038 (from +0.286 to +0.323) — about a 13% bump, far short of 2.5x. The mechanism is doing less than the framing implied.

**2. The Premium MRR keywords are near-universal in this MSP cohort.** Inspecting the labeled firms: phrases like "managed services," "managed it," and "recurring" appear in the marketing copy of nearly every firm in the training set, regardless of their actual revenue mix or `Manual Score`. Up-weighting a *near-universal* signal does not improve discrimination — it amplifies noise on a column whose between-firm variance was already low. The keywords that *would* discriminate (e.g., "monthly subscription," "sla") appear in too few firms (sparse signal) for Ridge to pick out. This is a feature-engineering instrumentation problem, not a thesis problem: the Yale "Nature of Revenue" thesis is correct, but our keyword *operationalization* of it is too coarse to capture revenue quality from website copy.

**3. Ridge redistributed weight in a costly direction.** The third-largest absolute coefficient change was on `tenure_sq`, which weakened from −0.348 to −0.273 — a 22% reduction in its contribution to the bell-curve fit over tenure. This is the same feature exp_007 demonstrated to be load-bearing. We didn't *prune* `tenure_sq` here (it's still non-zero), but the redistribution of L2-penalized coefficient mass toward `recurring_kw` and `log_employees` came partly out of the structurally-important tenure encoding. Ridge is doing what the L2 penalty asks of it: equalizing pressure across features. The hypothesis assumed "more weight on recurring_kw is free"; in reality the L2 budget is conserved, and the feature that paid the bill was the bell-curve helper. **Net: the model gained ~0.04 of standardized signal on a noisy keyword and lost ~0.08 on a clean structural feature. The trade was negative.**

### Verdict on the Stated Hypothesis
| Sub-hypothesis | Prediction | Outcome |
|---|---|---|
| Weighting changes the `recurring_kw` coefficient | Yes | **Confirmed** (+0.038, ~13% increase — much smaller than 2.5x due to StandardScaler) |
| Discriminates High-Quality MSPs from Project shops | Yes | **Falsified** — premium keywords are too universal in the cohort to discriminate |
| Improves RMSE on the discretized grid | Yes | **Falsified** (+5.1% regression) |

### Taxonomy: Signal Failure
RMSE did not improve. Per the Week-4 taxonomy established in exp_007, this run is classified as a **Signal Failure** — the modeling change executed cleanly (no Code Instability, no scrape regression, no SHA-256 issue) but the variable under test produced a worse predictive signal than the Control. The diagnostic upside is significant: we now know that *up-weighting the existing keyword set is not the right operationalization of the Yale thesis*. Future Yale-thesis ablations should focus on (a) finding rarer, more discriminative MRR keywords (e.g., "annual contract value," "renewal rate," "recurring revenue percentage"), (b) using the existing structured `Annual Revenue` field divided by employee count as an MRR-density proxy, or (c) scraping pricing pages directly rather than counting marketing keywords.

### Decision
* The Isolation Run is logged with `--keep` per the run instruction.
* **Substantively, exp_006 Control (RMSE 1.5044) remains the operative best.** Weighted Revenue Quality does not earn a switch.
* Recommend flipping the `logs/results.tsv` status of `exp_008` from `keep` to `discard` (parallel to the exp_007 maintenance step), so the performance plot correctly marks this as a failed Isolation Run. Flagging for user confirmation rather than auto-flipping.
* Per the Week-4 Snapshot Protocol, a snapshot of the current Weighted-MRR `model.py` has been saved to `logs/Snapshot_model_Exp_008.py` (8409 bytes) for reproducibility, and `model.py` will be reverted to `logs/Snapshot_model_Exp_006.py` before the next Isolation Run is proposed.

### What This Likely Tells Us — for the Week-4 Set
1. **The Yale "Nature of Revenue" thesis is operationally hard to encode from marketing copy.** Two of the three Premium MRR phrases that *might* discriminate ("monthly subscription," "sla") are too rare in this dataset to lift `recurring_kw`'s informativeness, while the dominant phrases ("managed services," "managed it") are too common to discriminate. This suggests the right instrument is structural data (revenue per employee, contract terms) rather than text features.
2. **Pre-scaler weighting is a leaky lever in a StandardScaler pipeline.** If a future experiment wants to inject a real 2.5x bump in coefficient magnitude, it must operate *post*-scaling — e.g., via a feature-specific Ridge alpha (using `ColumnTransformer` to scale features differently) or by injecting the weight into the loss directly. The current weighting only changes column rank and variance, not effective coefficient size.
3. **Ridge's L2 budget is conservative.** Adding mass to one coefficient draws it from elsewhere — and in this dataset, the "elsewhere" was the structurally-important `tenure_sq`. Future experiments that boost a feature should explicitly check whether the `tenure_sq` coefficient drops materially as a side effect.
4. **The diagnostic-first style is paying off.** Two consecutive Signal Failures (exp_007, exp_008) have produced *more* understanding of the model than the original exp_001 → exp_003 successes did. We now know `tenure_sq` is load-bearing, that `mgmt_depth` is dispensable, that L1 is too aggressive at alpha=0.1, and that text-keyword weighting can't deliver MRR discrimination in this cohort.

### Human Feedback/Comments
*Logged 2026-05-05.* This is **Isolation Run #2**, an isolated test of the Yale "Nature of Revenue" thesis operationalized as a 2.5x weight on Premium MRR keywords inside the existing `recurring_kw` feature. One variable changed against the exp_006 Control: regressor, scaler, rounding, all other 9 features, and the random seed are identical. Result is a Signal Failure — RMSE regressed by 5.1% — driven by two factors: the StandardScaler partially absorbed the weighting (so the effective amplification was ~13% in coefficient terms, not 2.5x), and the Premium MRR phrases used are too common in MSP marketing copy to discriminate within the cohort. Diagnostic value is high: the next Yale-thesis ablation should pivot from text-keyword counts to a structural MRR proxy (e.g., revenue per employee) or to rarer, more-discriminative phrases. The exp_006 Control remains the operative best at RMSE 1.5044. Snapshot at `logs/Snapshot_model_Exp_008.py` preserves this configuration for reproducibility.

### Audit
* **Judge integrity:** SHA-256 `8f7aa10f25b1...` verified prior to Worker execution.
* **Output schema:** `results.tsv` written with header `Predicted Score\tCompany Name`, tab-delimited, 62 firms scored, all values on the 0.5 grid.
* **Scrape coverage:** 61 / 62 firms reachable from `logs/scrape_cache.json` (1 cached failure, unchanged).
* **Revert verification:** `model.py` was reset to `logs/Snapshot_model_Exp_006.py` via `cp` before the keyword edit; `diff` confirmed byte-identical to the Control snapshot pre-edit.
* **Snapshot:** `logs/Snapshot_model_Exp_008.py` written immediately after run (8409 bytes; 176 bytes larger than Control snapshot due to the split keyword lists + weight constant).
* **Code Instability:** none.
