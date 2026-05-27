# Runtime and Budget Log: Exp_012 (Week 5 Isolation Run #2 — Tenure × Rev/Emp Interaction)
**Date:** 2026-05-07

## 1. Measured Runtime
| Mode | Wall Time | Per-Firm | Notes |
|---|---|---|---|
| **Warm cache (this run)** | ~1.8 s | ~0.029 s / firm | Scrape: 0 new fetches. GridSearchCV (4 alphas × 5 inner folds = 20 sub-fits) × `cross_val_predict` (5 outer folds) = 100 sub-fits + 20 sub-fits for the diagnostic full-data refit. |
| **Cold scrape (hypothetical)** | ~262 s | ~4.2 s / firm | Same scrape budget as exp_003+; α tuning adds ~0.5 s on top. |

* **Per-firm runtime budget compliance:** 0.029 s / firm ≪ 10 s budget from `program.md`. ✓
* **Marginal runtime cost vs exp_009 baseline:** ~+0.5 s wall (1.3 s → 1.8 s). The 4× α multiplier × 5 inner folds adds 20 sub-fits per outer fold, but each Ridge fit is closed-form and trivially fast at p=12, n≈40. Far cheaper than the exp_011 BaggingRegressor wrapper (which was +2.2 s).
* **Components (warm cache):**
  * Scrape: cache hit on all 62 entries (instant)
  * Featurization: identical to exp_009 + 1 vectorized multiplication for `stagnant_legacy = tenure × rev_per_emp` (12 features, p=12, n=62)
  * Model: nested CV — outer `cross_val_predict(GridSearchCV(...), cv=KFold(5))`. Each outer training fold (~50 rows) runs an inner GridSearchCV over 4 alphas with 5 inner folds, picks the best, refits on all ~50, predicts the held-out ~12.
  * Diagnostic: 1 additional full-data GridSearchCV fit (4 alphas × 5 inner folds + 1 refit) for coefficient + best-α reporting
  * Post-process: identical clip + 0.5-grid round
  * Snapshot: post-run `cp model.py logs/Snapshot_model_Exp_012.py` (~5 ms)

## 2. Estimated API / Data Cost
* **Data Source:** no new sources. The interaction is `tenure × rev_per_emp`, both already in the exp_009 feature set.
* **Cost per Credit:** $0.02 (Apollo, unchanged)
* **Total Leads Processed:** 62
* **Marginal Cost vs. exp_009:** **$0.00** — pure feature-engineering + autonomous tuning change.
* **Cumulative Cost Through Exp_012:** **$1.24** (Apollo firmographics, unchanged across all twelve experiments).

## 3. Scalability Projection
| Workload | exp_009 (Ridge α=1) | exp_012 (Ridge w/ GridSearchCV) | Notes |
|---|---|---|---|
| 1,000 leads, warm | ~12 s | ~17 s | GridSearchCV adds a constant 4× overhead per fit; sub-linear if `n_jobs=-1` enabled. |
| 10,000 leads, warm | ~2 min | ~3 min | Same ratio. |
| 1,000 leads, cold scrape | ~7 min @ 10× concurrency | ~7 min @ 10× concurrency | Network I/O still dominates. |

* **No new bottlenecks introduced.** GridSearchCV at p=12 is trivially cheap; bottleneck for production scaling remains web scraping.
* **For broader α sweeps:** if a future Week-5 run wants to expand the grid (e.g., `[0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]`), wall time scales linearly in grid size. An 8-alpha sweep would cost ~3 s warm — still well under budget.

## 4. Cumulative Budget Through Exp_012
| Item | Amount |
|---|---|
| Apollo firmographics (62 firms × $0.02) | $1.24 |
| Web scraping (own bandwidth, cached from exp_003) | $0.00 |
| Compute (local sklearn) | $0.00 |
| **Total** | **$1.24** |

## 5. Code Instability Audit
* **Edits made to `model.py`:**
  * Added import: `from sklearn.model_selection import GridSearchCV` (replaced existing `KFold, cross_val_predict` import line to add `GridSearchCV`)
  * Added one line in `featurize()`: `stagnant_legacy = tenure * rev_per_emp`
  * Added one entry in the returned DataFrame: `"stagnant_legacy": stagnant_legacy` (12th feature)
  * Replaced `build_model()` body: previously returned a fixed-α `Pipeline([StandardScaler, Ridge(alpha=1.0)])`; now returns `GridSearchCV(estimator=Pipeline([StandardScaler, Ridge()]), param_grid={"ridge__alpha": [0.1, 1.0, 10.0, 100.0]}, cv=KFold(5, shuffle=True, random_state=42), scoring="neg_root_mean_squared_error", refit=True)`. Return type annotation updated.
  * Diagnostic block updated: prints (a) the chosen α from full-data inner CV, (b) per-α mean RMSE table with the chosen row marked, (c) coefficients of `best_estimator_` at the chosen α.
* **Pre-edit revert:** `cp logs/Snapshot_model_Exp_009.py model.py` followed by `diff` — byte-identical confirmed before any edits.
* **Frozen-file modifications:** none. `eval/prepare.py`, `run_experiment.py`, `verify_integrity.py` all unchanged.
* **SHA-256 verification:** passed at run start (`8f7aa10f25b1...`).
* **Worker exit code:** 0. Judge exit code: 0.
* **Failures, warnings, or partial outputs:** none. All 4 GridSearchCV alphas converged on every fold without `ConvergenceWarning`.
* **Code Instability classification:** **none triggered.** (Note: a process-level guardrail recommendation around α-vs-baseline divergence is discussed in the Research Log → "What This Likely Tells Us" section #4; that is a *future* improvement, not a current-run instability.)

## 6. Snapshot Protocol Compliance
* `logs/Snapshot_model_Exp_012.py` written immediately after the Worker completed (9477 bytes; 1155 bytes larger than `Snapshot_model_Exp_009.py` due to the GridSearchCV import, wrapper, expanded diagnostic block, and the new feature line).
* Snapshot directory state after this run:
  * `logs/Snapshot_model_Exp_006.py` (8233 bytes) — Week-4 Control (legacy)
  * `logs/Snapshot_model_Exp_007.py` (8426 bytes) — Lasso (discarded)
  * `logs/Snapshot_model_Exp_008.py` (8409 bytes) — Weighted MRR (discarded)
  * `logs/Snapshot_model_Exp_009.py` (8322 bytes) — **current Week-5 baseline / canonical revert reference**
  * `logs/Snapshot_model_Exp_010.py` (8232 bytes) — Gaussian sweet-spot (discarded)
  * `logs/Snapshot_model_Exp_011.py` (8933 bytes) — Bagging (discarded)
  * `logs/Snapshot_model_Exp_012.py` (9477 bytes) — Tenure×Rev/Emp interaction + α tuning (this run; **stop rule triggered**)
* **Pending action:** before exp_013, `cp logs/Snapshot_model_Exp_009.py model.py` to revert to the Week-5 baseline (per `program.md` §Week 5 Specific Plans). Held for explicit user confirmation per the established Snapshot Protocol pattern.

## 7. Notes on the Severe Signal Failure
* **+0.4788 RMSE** is the largest absolute regression vs the *current* baseline of any Week-4 or Week-5 run, and the resulting RMSE (1.8743) exceeds the original exp_001 hand-coded baseline (1.8460). The model has effectively unlearned every modeling improvement since exp_002.
* **Cumulative diagnostic accumulator (Week-4 + Week-5 to date):**
  * `tenure_sq` is structurally load-bearing. It is the single most reliable indicator of model health across all controlled experiments to date — coefficient stays negative and substantial (−0.27 to −0.41) in healthy runs, weakens or sign-flips in every regression. exp_012's sign flip to +0.04 is the most extreme observation.
  * `rev_per_emp` (Week-4 Signal Success) is the second most fragile feature: large magnitude (−0.66) when alpha is in the right range; collapses to ~zero under heavier shrinkage.
  * Multiplicative interactions of already-present Ridge features have failed twice now (exp_004 with `tenure × mgmt-absence` and exp_012 with `tenure × rev_per_emp`). Both produced nominally-correct-signed but predictively-flat coefficients on the new term, and both triggered redistribution of weight that hurt the model. **Pattern flag: "tenure × X" interactions are multicollinearity traps; future interactions should use a non-tenure axis or be filtered through Lasso.**
* **Cumulative Week-4 + Week-5 wall-time: ~12 seconds** across seven controlled experiments. **Cumulative cost: $0.00 marginal beyond the original $1.24 Apollo charge.**
