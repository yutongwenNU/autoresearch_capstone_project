# Runtime and Budget Log: Exp_011 (Week 5 Isolation Run #1 — Bootstrap Aggregation)
**Date:** 2026-05-07

## 1. Measured Runtime
| Mode | Wall Time | Per-Firm | Notes |
|---|---|---|---|
| **Warm cache (this run)** | ~3.5 s | ~0.057 s / firm | Scrape: 0 new fetches. BaggingRegressor fits 50 Ridge pipelines × 5 OOF folds = 250 sub-fits + 50 sub-fits for the diagnostic refit. |
| **Cold scrape (hypothetical)** | ~263 s | ~4.2 s / firm | Same scrape budget as exp_003+; bagging adds ~3 s on top. |

* **Per-firm runtime budget compliance:** 0.057 s / firm ≪ 10 s budget from `program.md`. ✓
* **Marginal runtime cost vs exp_009 baseline:** ~+2.2 s wall (1.3 s → 3.5 s). The 50× sub-estimator multiplier is partially absorbed because each Ridge fit is closed-form and trivially fast at p=11, n≈50; but the StandardScaler refit per bag adds non-trivial overhead. BaggingRegressor's default `n_jobs=None` runs sequentially — could be parallelized with `n_jobs=-1` if a future ablation needs the speed.
* **Components (warm cache):**
  * Scrape: cache hit on all 62 entries (instant)
  * Featurization: identical to exp_009 (11 features, p=11, n=62)
  * Model: BaggingRegressor with 50 base estimators × `cross_val_predict` 5-fold = 250 sub-fits for the OOF metric, plus 50 sub-fits for the diagnostic full-data refit
  * Diagnostic: stack 50 Ridge coefficient vectors, compute mean ± std vectorized over 11 features (sub-millisecond)
  * Post-process: identical clip + 0.5-grid round
  * Snapshot: post-run `cp model.py logs/Snapshot_model_Exp_011.py` (~5 ms)

## 2. Estimated API / Data Cost
* **Data Source:** no new sources. Bagging operates on the same featurized matrix already loaded since exp_001.
* **Cost per Credit:** $0.02 (Apollo, unchanged)
* **Total Leads Processed:** 62
* **Marginal Cost vs. exp_009:** **$0.00** — exp_011 is a pure modeling-wrapper change.
* **Cumulative Cost Through Exp_011:** **$1.24** (Apollo firmographics, unchanged across all eleven experiments).

## 3. Scalability Projection
| Workload | exp_009 (Ridge, warm) | exp_011 (Bagging-Ridge, warm) | Notes |
|---|---|---|---|
| 1,000 leads, warm | ~12 s | ~33 s | Bagging adds ~2.5–3× overhead at any N due to the 50-fit multiplier; sub-linear if `n_jobs=-1` enabled (likely 8–12 s on a 4-core laptop). |
| 10,000 leads, warm | ~2 min | ~6 min | Same ratio. |
| 1,000 leads, cold scrape | ~7 min @ 10× concurrency | ~7 min @ 10× concurrency | Network I/O still dominates. |

* **No new bottlenecks introduced.** Bagging's only resource cost is CPU and a constant-factor 50× multiplier on Ridge fit time. At p=11, n≈1k it's still sub-second per bag.
* **Note for grid-search experiments:** if a future Week-5 run sweeps `n_estimators ∈ {25, 50, 100, 200}` × `max_samples ∈ {0.5, 0.7, 0.8, 1.0}`, that's a 4×4 grid × 5 OOF folds × the chosen `n_estimators`. At `n_estimators=200` the wall-time per cell rises to ~12 s; the full grid is ~3 minutes. Still within the project's runtime budget.

## 4. Cumulative Budget Through Exp_011
| Item | Amount |
|---|---|
| Apollo firmographics (62 firms × $0.02) | $1.24 |
| Web scraping (own bandwidth, cached from exp_003) | $0.00 |
| Compute (local sklearn) | $0.00 |
| **Total** | **$1.24** |

## 5. Code Instability Audit
* **Edits made to `model.py`:**
  * Added import: `from sklearn.ensemble import BaggingRegressor`
  * Modified `build_model()` to wrap the existing `Pipeline([StandardScaler, Ridge])` in `BaggingRegressor(estimator=base, n_estimators=50, max_samples=0.8, random_state=42)`. Return type annotation updated to `BaggingRegressor`.
  * Diagnostic block updated: stacks coefficient vectors from `full_model.estimators_` and prints `mean ± std` per feature instead of a single coefficient row.
* **Pre-edit revert:** `cp logs/Snapshot_model_Exp_009.py model.py` followed by `diff` — byte-identical confirmed before any edits.
* **Frozen-file modifications:** none. `eval/prepare.py`, `run_experiment.py`, `verify_integrity.py` all unchanged.
* **SHA-256 verification:** passed at run start (`8f7aa10f25b1...`).
* **Worker exit code:** 0. Judge exit code: 0.
* **Failures, warnings, or partial outputs:** none. All 50 sub-estimators converged silently under Ridge's default solver.
* **Code Instability classification:** **none triggered.**

## 6. Snapshot Protocol Compliance
* `logs/Snapshot_model_Exp_011.py` written immediately after the Worker completed (8933 bytes; 611 bytes larger than `Snapshot_model_Exp_009.py` due to the BaggingRegressor import + wrapper + the expanded diagnostic block).
* Snapshot directory state after this run:
  * `logs/Snapshot_model_Exp_006.py` (8233 bytes) — Week-4 Control (legacy)
  * `logs/Snapshot_model_Exp_007.py` (8426 bytes) — Lasso isolation (discarded)
  * `logs/Snapshot_model_Exp_008.py` (8409 bytes) — Weighted MRR (discarded)
  * `logs/Snapshot_model_Exp_009.py` (8322 bytes) — **current Week-5 baseline / canonical revert reference**
  * `logs/Snapshot_model_Exp_010.py` (8232 bytes) — Gaussian sweet-spot (discarded)
  * `logs/Snapshot_model_Exp_011.py` (8933 bytes) — Bagging isolation (this run)
* **Pending action:** before exp_012, `cp logs/Snapshot_model_Exp_009.py model.py` to revert to the Week-5 baseline (per `program.md` §Week 5 Specific Plans). Held for explicit user confirmation.

## 7. Notes on the Signal Failure
* This is the **first Week-5 Signal Failure**. The diagnostic value is unusually high for a regression-by-only-3.6%: the per-bag std-devs make explicit which features are "real-but-fragile" (sign flips across bags) vs "real-and-stable" (consistent sign across bags). Notable instability ranking by std/|mean|:
  * `log_employees`: std/|mean| = 2.49 (most unstable)
  * `tenure`: 1.82
  * `mgmt_depth`: 1.50
  * `recurring_kw`: 1.30
  * `rev_per_emp`: 1.25 — even our Week-4 Signal Success feature flips sign across bags
* These ratios are an inversion of the cross-experiment finding that `tenure_sq` is "load-bearing": at the bag level, *no* feature except `in_midwest` (which is exactly zero) is unambiguously stable. The full-sample Ridge fit is producing coefficients that are real but rely on the entire dataset being present.
* **Cumulative Week-4 + early-Week-5 wall-time: ~10.2 seconds** across six controlled experiments. **Cumulative cost: $0.00 marginal beyond the original $1.24 Apollo charge.**
