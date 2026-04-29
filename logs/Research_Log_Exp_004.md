# Experiment Log: Auto-Private-Equity Search Engine
**Date:** 2026-04-28
**Experiment ID:** exp_004
**Status:** keep (logged) → **DISCARDED** (reverted before exp_005; exp_003 remains current best)

## Experiment: Tenure × Management-Absence Interaction Term

### Configuration
* **Worker:** `research.py` — Ridge(alpha=1.0), unchanged from exp_003
* **Judge:** `eval/prepare.py` (FROZEN, SHA-256 verified before run)
* **New input:** `succession_gap = tenure × (max_depth − mgmt_depth)` interaction term added to the existing 10-feature set
* **Validation Protocol:** unchanged — `cross_val_predict` with `KFold(n_splits=5, shuffle=True, random_state=42)`

### Hypothesis
The diagnostic from exp_003 showed that `mgmt_depth` alone carried only a tiny, wrong-signed coefficient (+0.04) — Ridge couldn't extract usable signal from a standalone count. The hypothesis for exp_004: **the search-fund "succession gap" thesis is fundamentally a *conditional* signal — `(high tenure) AND (low management depth)` — that a linear model cannot represent with two independent main effects.** Encoding the conjunction as an explicit interaction term should let Ridge give it a single coefficient and learn its sign empirically.

`mgmt_absence = MAX_DEPTH − mgmt_depth` was bounded in [0, 9] using the dataset-wide maximum depth as the ceiling, so the interaction lives in [0, ~450]. The expected sign is *positive*: deep tenure × high management absence → high Manual Score.

### Result
| Metric | exp_003 | **exp_004** | Δ |
|---|---|---|---|
| `val_rmse` | 1.5016 | **1.5110** | +0.0094 (regression) |
| `val_r2`   | 0.4359 | 0.4287 | −0.0072 |

### Diagnostic — Ridge coefficients (standardized), exp_003 vs exp_004
| feature | exp_003 | **exp_004** | Δ |
|---|---|---|---|
| `mgmt_depth` | +0.0426 | **+0.2706** | +0.228 |
| `succession_gap` (new) | — | **+0.2842** | — |
| `tenure` | +0.7090 | +0.5413 | −0.168 |
| `tenure_sq` | −0.3483 | −0.3112 | +0.037 |
| `sweet_spot_emp` | +0.7924 | +0.7916 | unchanged |
| (other features) | mostly unchanged | | |

### What This Likely Tells Us — Hypothesis Supported in Coefficient Space, Not Predictive Space
The interaction is **correctly signed and ~6× the standardized magnitude** of the standalone count from exp_003 (+0.28 vs +0.04). This is a real diagnostic win: the conditional encoding flushed out a succession-gap signal that the standalone count had obscured. The hypothesis that "the thesis is conditional" is empirically supported by the model's internal weights.

But predictive accuracy did not move (in fact regressed slightly). Two readings explain this:

1. **Multicollinearity redistribution.** Adding `succession_gap` made `mgmt_depth` jump from +0.04 to +0.27, while `tenure` dropped from +0.71 to +0.54. Ridge is reshuffling weight among three highly correlated features — `tenure`, `mgmt_depth`, and their product — that all derive from the same two underlying variables. The *total* explanatory power across that group barely changed; only the distribution did.
2. **Two opposing effects are now disentangled but cancel.** `mgmt_depth = +0.27` reads as a "professionalism premium" (visible management correlates with quality), while `succession_gap = +0.28` reads as the "thesis effect" (deep tenure × no management = upside). These pull in *opposite directions* on the same scraped-depth variable. Ridge has separated them at the coefficient level, but the net signal in the data is roughly what exp_003 already captured.

### What This Means for the Research Direction
The interaction term worked exactly as a diagnostic: it confirmed the conditional structure of the signal. But to actually *move RMSE*, we need new information, not a different shape on the existing information. Two paths:

1. **Sharper scraping.** Look for explicit founder-tenure mentions (`"founded by"`, `"since 19XX by"`) on team pages, rather than counting role-title regexes that pick up boilerplate equally well.
2. **Pivot model class.** Now that we know the interaction is real, a tree-based model (`HistGradientBoostingRegressor`) can find the *threshold* version of the interaction ("tenure > 25 AND mgmt_depth ≤ 1") that Ridge approximates only crudely with a multiplicative term. **(Chosen for exp_005)**

### Decision: Discard
Although the run was logged with `--keep`, the RMSE regressed by 0.0094 — exp_003 (RMSE 1.5016) remains the current best. **`research.py` will be reverted to the exp_003 feature set before exp_005**, and exp_004 stands as a logged-but-not-merged diagnostic experiment that informed the model-class pivot.

### Audit
* **Judge integrity:** SHA-256 `570d9e2a89c8...` verified prior to Worker execution.
* **Output schema:** `results.tsv` written with header `Predicted Score\tCompany Name`, tab-delimited, 62 firms scored.
* **No new external data fetched** — used the same `logs/scrape_cache.json` from exp_003.
