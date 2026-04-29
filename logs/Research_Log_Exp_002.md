# Experiment Log: Auto-Private-Equity Search Engine
**Date:** 2026-04-28
**Experiment ID:** exp_002
**Status:** keep

## Experiment: Ridge Regression on Engineered Features (5-fold OOF)

### Configuration
* **Worker:** `research.py` — Ridge(alpha=1.0) inside a `Pipeline([StandardScaler, Ridge])`
* **Judge:** `eval/prepare.py` (FROZEN, SHA-256 verified before run)
* **Training Set:** 62 manually labeled IT MSPs from the U.S. Midwest
* **Validation Protocol:** `cross_val_predict` with `KFold(n_splits=5, shuffle=True, random_state=42)` — every prediction written to `results.tsv` was made by a model that never saw that firm's `Manual Score`.

### Engineered Features (9 total)
| Feature | Source | Rationale |
|---|---|---|
| `log_employees` | `# Employees` (log1p) | Headcount distribution is right-skewed; log dampens long-tail outliers so a linear model can use it. |
| `log_revenue` | `Annual Revenue` (log1p) | Same reasoning as above; revenue spans multiple orders of magnitude. |
| `tenure` | `2026 − Founded Year` | Search-fund thesis: established firms with ≥25 years of tenure are succession-rich. |
| `tenure_sq` | `tenure²` | Lets a linear model approximate the bell curve around the "established but not ancient" sweet spot. |
| `sweet_spot_emp` | indicator: `10 ≤ employees ≤ 30` | Stanford/Yale primer's headcount sweet spot for a single-searcher acquisition. |
| `in_midwest` | `Company State ∈ MIDWEST_STATES` | Geography-of-fit signal from the project thesis. |
| `recurring_kw` | keyword count over `Rationale + Keywords + Technologies + Short Description` | Yale "On the Nature of Revenue": MRR / managed services / monitoring / backup are the high-quality revenue signals. |
| `stagnation_kw` | keyword count, same text bundle | "Stable but stagnant" upside signal — legacy / outdated / VOIP / hardware sales mean room for a young owner to modernize. |
| `modern_ai_kw` | keyword count, same text bundle | Negative signal — firms already AI-forward have less transformation upside for a search-fund acquirer. |

### Proposal & Rationale
The V0 baseline (`exp_001`) used four hand-picked boolean rules and produced **RMSE 1.8460, R² 0.1474** — i.e., the heuristic explained ~15% of the variance in human labels. The hypothesis for `exp_002` was that the predictive signal is *real* but the encoding is too coarse: boolean thresholds discard magnitude information (a firm with 28 employees and one with 11 are both "in the sweet spot" but get identical credit), and four rules cannot represent the additive structure the search-fund literature implies.

Three design choices, each motivated by the small-N constraint (62 firms):

1. **Ridge over a tree-based model.** Tree ensembles (`GradientBoostingRegressor`, `HistGradientBoostingRegressor`) would have memorized labels at this sample size and produced misleadingly low in-sample RMSE. Ridge's L2 penalty is the right inductive bias when features ≈ rows / 6 — it pulls coefficients toward zero unless the signal is strong enough to overcome the penalty.
2. **Engineered features over raw inputs.** A linear model can only exploit signals that are already linear in the encoded feature. Sweet-spot headcount, Midwest geography, and keyword counts had to be made explicit — Ridge can't discover "10 ≤ employees ≤ 30" from a raw `# Employees` column on its own.
3. **Out-of-fold predictions, not in-sample fit.** Fitting on all 62 rows and predicting back would have produced a near-zero RMSE (label leakage). Using `cross_val_predict` ensures the RMSE the Judge computes is comparable to the V0 baseline's RMSE — both are honest validation errors on the same 62 firms.

### Result
| Metric | exp_001 (baseline) | exp_002 (Ridge) | Δ |
|---|---|---|---|
| `val_rmse` | 1.8460 | **1.5112** | **−0.3348** |
| `val_r2` | 0.1474 | **0.4287** | **+0.2813** |

Variance explained nearly tripled (0.15 → 0.43), confirming the engineered features carry real signal that the V0 boolean rules were leaving on the table. RMSE improvement (~18%) comes mainly from Ridge fitting *magnitudes* that booleans flatten.

### What This Likely Tells Us
* **The feature set is informative.** R² jumping from 0.15 to 0.43 means the engineered signals (recurring-revenue keywords, sweet-spot headcount, log-revenue, tenure) are doing real work that the four boolean rules were leaving on the table. The features themselves are most of the story; Ridge is just the simplest competent way to consume them.
* **Ridge is probably underfitting interactions.** A single global linear model treats each contribution independently — but the search-fund thesis is fundamentally about *combinations* (e.g., "old + small + Midwest + recurring revenue" should compound into a strong target, not be the linear sum of four weak ones). The largest residuals are likely on firms where these signals stack non-additively.
* **The remaining ~57% of unexplained variance is a roadmap.** Two unexplored families of signal stand out: (a) qualitative succession indicators (founder tenure vs. management team depth — currently zero features encode this), and (b) interaction terms between existing features. Exp_003 should test (a) before assuming the gap is purely about model complexity.

### Next Iteration Candidates
1. **Qualitative scraping** — add a "Management Depth / Succession Gap" feature from website content. **(chosen for exp_003)**
2. `HistGradientBoostingRegressor` with the same features and the same 5-fold OOF protocol — captures the interaction structure Ridge can't.
3. `PolynomialFeatures(degree=2, interaction_only=True)` inside the Ridge pipeline — tests whether interactions alone close the gap, while staying within the linear-model family.

### Audit
* **Judge integrity:** SHA-256 `570d9e2a89c8...` verified prior to Worker execution by `run_experiment.py` → `verify_integrity.verify_prepare()`.
* **Output schema:** `results.tsv` written with header `Predicted Score\tCompany Name`, tab-delimited, 62 firms scored, all present in training set (no missing-firm warnings).
