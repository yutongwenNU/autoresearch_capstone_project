# Experiment Log: Auto-Private-Equity Search Engine
**Date:** 2026-04-28
**Experiment ID:** exp_003
**Status:** keep (marginal)

## Experiment: Management Depth Scraper Signal

### Configuration
* **Worker:** `research.py` — Ridge(alpha=1.0) inside a `Pipeline([StandardScaler, Ridge])`, identical regressor to exp_002
* **Judge:** `eval/prepare.py` (FROZEN, SHA-256 verified before run)
* **New input:** `mgmt_depth` feature scraped per-firm from each company's website
* **Validation Protocol:** `cross_val_predict` with `KFold(n_splits=5, shuffle=True, random_state=42)` — unchanged from exp_002

### Proposal & Rationale
Exp_002 closed ~28 percentage points of R² by encoding firmographic and keyword-textual signal in a Ridge model. The remaining 57% of unexplained variance pointed to a missing family of features: **qualitative succession indicators** that the search-fund literature flags as central to target quality.

The Stanford/Yale primer (p.40, "Sourcing Acquisition Opportunities") explicitly lists "age of CEO/founder" alongside revenue and headcount as a key screening metric, and the broader thesis throughout the primer is that the *searcher* steps in to replace a thin or absent management layer. The hypothesis: **a firm with deep founder tenure but a thin/absent visible management team has a higher Succession Gap, which is a positive signal for a search-fund acquisition.**

This is the first experiment to introduce a feature derived from a data source outside `train_set.csv` — it tests whether qualitative web-scraped signals can complement the structured firmographic features.

### Scraper Design
* **Two-page fetch per firm:** homepage + first reachable team-page candidate from `[/about, /about-us, /team, /our-team, /leadership, /management, /company]`. Counting across two pages is more robust than relying solely on the homepage.
* **Distinct-category counting (not raw occurrences):** count *which* role-title patterns appear at least once across the scraped HTML — `\bceo\b`, `\bcto\b`, `vice president`, `director of`, `leadership team`, `management team`, etc. A page repeating "CEO CEO CEO" contributes 1 unit, not 3, preventing one keyword from dominating.
* **HTML stripping:** `<script>` and `<style>` blocks removed first, then a generic tag-strip regex, then lowercased before regex matching.
* **Disk cache** (`logs/scrape_cache.json`) keyed by URL: first run is the slow, deterministic establishing run; subsequent runs are instant and reproducible. The cache also makes the experiment auditable — anyone can inspect what was actually fetched per firm.
* **4-second per-request timeout** with a User-Agent header. On any HTTP exception or non-200 response, the firm gets `mgmt_depth = 0` and the failure is reported. (Known bias: this conflates "no management visible" with "couldn't fetch.")

### Result
| Metric | exp_002 | **exp_003** | Δ |
|---|---|---|---|
| `val_rmse` | 1.5112 | **1.5016** | −0.0096 |
| `val_r2`   | 0.4287 | **0.4359** | +0.0072 |
| Scraper coverage | — | 61 / 62 firms reachable | 1 failure |

### Diagnostic — Ridge coefficients (standardized)
```
sweet_spot_emp: +0.7924   tenure: +0.7090   stagnation_kw: +0.4127
   log_revenue: +0.3776  modern_ai_kw: −0.3768   tenure_sq: −0.3483
  recurring_kw: +0.2855    mgmt_depth: +0.0426  ← scraped feature
 log_employees: −0.0250    in_midwest:  0.0000
```

### What This Likely Tells Us — Hypothesis Did Not Validate Cleanly
The thesis predicted `mgmt_depth` would carry a *negative* coefficient (more visible management → lower succession gap → lower predicted score). Instead Ridge assigned it a tiny *positive* weight of +0.04, and the absolute RMSE improvement (0.01) is within run-to-run noise.

Two readings of the diagnostic, in order of likelihood:

1. **The signal is too noisy to discriminate (most likely).** Counting role-title regexes across a homepage and one team page picks up footers, generic boilerplate, and stock-photo captions equally well as it picks up a real "meet the leadership" section. Ridge's response was correct: when a feature is mostly noise, regularization shrinks its coefficient toward zero. The 0.04 is noise around 0, not a meaningful inverted sign.
2. **The thesis may be partially miscalibrated for this dataset.** Labelers may have rewarded firms that "look professional" — a visible management page partially correlates with quality, offsetting any "succession gap is good" effect. This is hard to disentangle from (1) without sharper instrumentation.

A scraped depth distribution `{0:8, 1:12, 2:11, 3:7, 4:9, 5:4, 6:6, 7:3, 9:1}` confirms the feature *does* vary across firms (it's not constant) — the issue is that the variance isn't aligned with Manual Score variance.

### What Should Come Next
The right inference from this diagnostic is **not** "abandon scraping" — it's "the standalone count is the wrong shape." The thesis is fundamentally *conditional*: the high-value case is "founder with deep tenure AND no management team," not "low management depth on its own." A Ridge model cannot represent this conjunction with two separate linear weights — the conditional needs to be encoded as an interaction term.

Three candidate next steps, in order of preference:

1. **Tenure × management-absence interaction (chosen for exp_004).** Encode the conditional explicitly so Ridge can give it a single coefficient.
2. Targeted scraping — search team pages for explicit founder-tenure mentions ("founded by", "since 1985 by") rather than counting any role title.
3. Pivot to `HistGradientBoostingRegressor` and let a tree-based model discover interactions implicitly across all features.

### Audit
* **Judge integrity:** SHA-256 `570d9e2a89c8...` verified prior to Worker execution.
* **Output schema:** `results.tsv` written with header `Predicted Score\tCompany Name`, tab-delimited, 62 firms scored.
* **Scraper artifact:** 62-entry cache persisted at `logs/scrape_cache.json` for reproducibility.
