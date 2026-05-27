# Research Axes Summary — 20 Experiments Organized by Hypothesis Direction
**Date compiled:** 2026-05-13
**Project state:** Operative baseline is **Exp_009** (Ridge + Stagnation Premium, RMSE **1.3955**, R² 0.5128). Root `model.py` is byte-identical to `logs/Snapshot_model_Exp_009.py`.

The Week-4 → Week-5 sweep has organized into five distinct **Research Axes**. Each axis is defined by a class of hypothesis about *what makes a Midwest IT MSP a good search-fund acquisition target*, plus the data-science apparatus used to encode it.

---

## Headline Table

| Axis | Key Feature | Status | Impact on Model |
|---|---|---|---|
| **Demographics** | `tenure_sq` | **Core** | Established the *"Succession Gap"* bell curve over tenure. Load-bearing in 14/14 controlled experiments — most reliable diagnostic in the project. |
| **Operational Efficiency** | `rev_per_emp` | **Core** | Validated the *"Stagnation Premium"* — operational slack inside MRR-bearing firms is the actual signal. **Current Baseline (Exp_009, RMSE 1.3955).** |
| **Ownership / Succession** | `succession_red_flag` family | Exploratory | Correctly-signed coefficients across multiple encodings (founder-led, institutionalization, refined kill-switch); all failed RMSE due to **sparsity** (3–11% tag rates at N=62). |
| **Competitive Moat** | `has_moat` | **Discarded** | Apparent Exp_016 win (RMSE 1.3079) **disqualified by Exp_018 audit**: the lift was a `compliance`-keyword artifact, not a vertical-moat signal. See pivot section below. |
| **Model Integrity / Robustness** | Bagging / Interaction / Sigmoid scaling | **Discarded** | Confirmed simple Ridge is the right inductive bias for N=62. Bagging, multiplicative interactions, and smooth-curve replacements all regressed; Lasso pruned load-bearing features. |

---

## 1. Demographics — *Core*

**Defining feature:** `tenure_sq`
**Thesis:** The Stanford/Yale primer's "established but not ancient" sweet spot — firms with deep founder tenure are succession-rich, but very old firms hit diminishing returns. A negative `tenure_sq` coefficient combined with positive `tenure` encodes the inverted-U bell curve.

**Empirical signature** (across all 14 controlled experiments since Exp_002):
* Coefficient stays in **[−0.27, −0.45]** in healthy runs.
* Weakens or sign-flips in every Signal Failure that touched the tenure encoding.
* Single most reliable model-health diagnostic in the project.

**Member experiments**

| Exp | Variant | RMSE | Outcome |
|---|---|---|---|
| 001 | Hand-coded boolean tenure rule | 1.8460 | Baseline |
| 002 | Ridge with `tenure` + `tenure_sq` + `sweet_spot_emp` | 1.5112 | **Keep** — established the bell-curve fit |
| 010 | `sweet_spot_emp` → Gaussian (μ=20, σ=10) | 1.6741 | Discard — smoothing broke implicit regularization of `log_employees` |
| 019 | `is_hub_proximate` (city → metro hub binary) | 1.4189 | Discard (mild) — correctly signed +0.22 but redundant with existing firmographics |

---

## 2. Operational Efficiency — *Core, Current Baseline*

**Defining feature:** `rev_per_emp = Annual Revenue / # Employees`
**Thesis (refined):** The Yale "Nature of Revenue" thesis — but operationalized as a structural ratio, not text keywords. Among firms of similar size and revenue, *lower* per-head productivity reads as **operational slack** a searcher can convert — the "Stagnation Premium."

**Key code (Exp_009 — current baseline):**
```python
rev_per_emp = revenue / employees.clip(lower=1)
# Added to the feature DataFrame as the 11th column.
```
**Resulting standardized coefficient:** −0.66 (largest magnitude after `log_revenue`; sign-flipped from naïve "MRR efficiency → quality" hypothesis but coherent under the stagnation-premium reading).

**Member experiments**

| Exp | Variant | RMSE | Outcome |
|---|---|---|---|
| 008 | `recurring_kw` × 2.5 weight on Premium MRR keywords | 1.5811 | Discard — keywords too universal in MSP cohort (49/62 mention "compliance" or similar) |
| **009** | `rev_per_emp = revenue / emp` | **1.3955** | **Keep — Current Baseline; new all-time best** |
| 017 | `stagnation_ratio = (legacy+1)/(modern+1)` | 1.5811 | Discard — redundant with existing `stagnation_kw` count |

---

## 3. Ownership / Succession — *Exploratory*

**Defining family:** `succession_red_flag` / `founder_led` / `is_institutionalized` / `ownership_red_flag` — five separate attempts to flag firms where the original founder is still in charge OR the firm has been acquired / institutionalized.

**Thesis:** A 25-year-old firm where the founder still appears as a current leader is a *higher-risk* succession play (the searcher can't easily replace them). A firm explicitly acquired or PE-backed is *no longer a target*. Both should pull predicted Manual Score downward.

**Empirical signature across the family:**
* Coefficients are **correctly signed** (all negative, in [−0.11, −0.36] range).
* All five attempts **failed RMSE** at α=1.0 due to **sparsity**: tag rates of 3–11 firms out of 62 cannot overcome the variance penalty of fitting a coefficient at N=62.
* In one case (Exp_014, Exp_020), a single high-leverage false positive (World Synergy, Manual Score 8.5) dominated the squared-error contribution.

**Cross-experiment threshold finding:** binary features need ≥ ~16% tag rate at N=62 to be RMSE-positive even when correctly signed.

**Member experiments**

| Exp | Variant | Tag Rate | Coef | RMSE | Outcome |
|---|---|---|---|---|---|
| 003 | `mgmt_depth` (role-title regex count from scraped pages) | continuous | +0.04 | 1.5016 | Marginal keep — feature too coarse |
| 004 | `tenure × (max_depth − mgmt_depth)` interaction | continuous | +0.28 | 1.5110 | Discard — multicollinearity reshuffle |
| 013 | `founder_led` (5-regex NLP detection) | 4/62 | −0.11 | 1.4555 | Discard — sparse signal |
| 014 | `is_institutionalized` (10 keywords + 1 regex) | 3/62 | −0.36 | 1.4645 | Discard — sparse signal + World Synergy false positive |
| 015 | `succession_red_flag` (founder ∪ acquisition) | 7/62 | −0.31 | 1.4690 | Discard — sparsity below 16% threshold |
| 020 | `ownership_red_flag` (4 explicit acquisition phrases) | 3/62 | −0.36 | 1.4645 | Discard — *identical* to Exp_014; refinement didn't change tag set |

---

## 4. Competitive Moat — *Discarded* ⚠️

**Defining feature:** `has_moat` — binary indicator for firms whose marketing copy mentions regulated verticals.
**Thesis (user-stated):** Firms in regulated industries (HIPAA, dental, legal, manufacturing, PCI, compliance, regulated) have higher switching costs and more stable revenue — hence higher acquisition quality.

### Why the Moat Axis Was Discarded — the Exp_018 "Compliance Artifact" Audit

**The pivot moment.** Exp_016 produced an apparent **new all-time best RMSE of 1.3079** (−6.3% vs the Exp_009 baseline) with `has_moat` carrying a strong negative coefficient (−0.58). But three diagnostics raised red flags:
1. **Tag rate was 88.7%** — the binary was tagging almost every firm, meaning the predictive work was being done by the rare *untagged* minority (7 firms).
2. **Coefficient sign was opposite to the stated hypothesis.** The thesis predicted positive (moat → quality); empirically the sign was strongly negative.
3. **One keyword dominated the tag set.** `compliance` alone tagged 49/62 firms (79%) — and `compliance` was already inside `RECURRING_KW`.

**The Exp_018 ablation audit (2026-05-08):** removed `compliance` from `MOAT_KW`. **RMSE jumped from 1.3079 → 1.4049** — a +0.097 swing on a single keyword removal. The pure industry-vertical signal alone is RMSE 1.4049, *worse* than baseline 1.3955 by +0.7%.

**Conclusion:** the Exp_016 win was an artifact. The 7 untagged firms (Pinnacle 10.0, Innovative Computers 9.0, Miken 9.0, Dymin 8.5, SMaRT 8.5, One Click 7.5, Axia 6.5) happened to be high-scored — but the mechanism was "atypical marketing copy" labeler bias, not vertical-moat economics. **The audit caught a Type-I promotion before commitment to baseline.**

**Process lesson** codified going forward: any feature with > 80% tag rate or coefficient sign opposite to the hypothesis must be ablation-tested before promotion.

**Member experiments**

| Exp | Variant | Tag Rate | Coef | RMSE | Outcome |
|---|---|---|---|---|---|
| 016 | `has_moat` (8 keywords incl. `compliance`) | 55/62 (88.7%) | −0.58 | 1.3079 | **Discarded by Exp_018 audit** — artifact win |
| 018 | `has_moat` (7 keywords; `compliance` excluded) | 38/62 (61.3%) | −0.48 | 1.4049 | Discard — proved Exp_016 was a `compliance`-driven artifact |

---

## 5. Model Integrity / Robustness — *Discarded*

**Family:** changes to the *modeling apparatus* itself rather than to the features — regressor swaps, regularization changes, ensembles, post-processing.

**Cross-experiment finding:** simple Ridge(α=1.0) is the right inductive bias at N=62. Every attempt to "improve" the model class regressed:
* **Lasso** pruned the load-bearing `tenure_sq` feature → +14% RMSE.
* **HistGradientBoosting** overfit the 50-row training folds → +57% RMSE.
* **BaggingRegressor** averaged structural signal toward zero when the per-bag training size dropped to ~50 → +3.6% RMSE.
* **Multiplicative interactions** of existing features created multicollinearity traps that forced α tuning to over-shrink load-bearing features → +34% RMSE.

This axis was the *empirical motivation* for `program.md` §5 (Decoupled Isolation Rule) and §6 (5× Alpha Guardrail) — both protocols codified to prevent the failure modes observed here.

**Member experiments**

| Exp | Variant | RMSE | Outcome |
|---|---|---|---|
| 005 | HGBR + 0.5-grid rounding (bundled) | 2.1764 | Discard — HGBR overfits N=62 |
| **006** | **Week-4 Control: Ridge + 0.5-grid rounding** | **1.5044** | **Keep — established the discretization step** |
| 007 | Lasso(α=0.1) | 1.7168 | Discard — pruned `tenure_sq` |
| 011 | BaggingRegressor(50 bags, max_samples=0.8) | 1.4464 | Discard — averaged away `tenure_sq` |
| 012 | `tenure × rev_per_emp` interaction + GridSearchCV α | 1.8743 | Discard — α=10 erased load-bearing features (worse than Exp_001 baseline) |

---

## Cumulative Diagnostic Accumulator (2026-05-13)

Across the 20 controlled experiments, four cross-cutting findings are robust enough to cite without further testing:

1. **`tenure_sq` is the canary feature.** Coefficient in [−0.27, −0.45] in every healthy run; weakened or sign-flipped in every regression.
2. **`mgmt_depth` is dispensable.** Coefficient bounces in [+0.02, +0.18]; safe to keep, never costly to drop.
3. **Sparse binary features fail at N=62.** Empirical threshold: ~16% tag rate required for a correctly-signed binary to deliver RMSE improvement.
4. **Multiplicative interactions of existing Ridge features are multicollinearity traps.** Either pruned by L1, redistributed by L2 toward harmful shrinkage, or trigger the §6 5× Alpha Guardrail.

## Trajectory Headline Numbers

| Phase | Best RMSE | Improvement | Notes |
|---|---|---|---|
| Original baseline (Exp_001) | 1.8460 | — | Hand-coded boolean rules |
| End of Week 3 (Exp_003) | 1.5016 | −18.6% | Ridge + 9 features + scraped management depth |
| **End of Week 5 (Exp_009, current)** | **1.3955** | **−24.4%** vs Exp_001; **−7.1%** vs Week-3 best | Ridge + structural Stagnation Premium |

**R² progression:** Exp_001 = 0.147 → Exp_003 = 0.436 → **Exp_009 = 0.513** (~3.5× variance explained).

## Cost & Compute

* **Cumulative cost across all 20 experiments:** $1.24 (Apollo firmographics one-time charge; zero marginal cost for any modeling/feature experiment since Exp_001).
* **Cumulative wall-time across 20 controlled experiments:** ~25 seconds (warm cache).
* The cheapest controlled-experiment regime in the project's runtime budget by ~3 orders of magnitude vs. the §scalability projection for 10,000 leads.
