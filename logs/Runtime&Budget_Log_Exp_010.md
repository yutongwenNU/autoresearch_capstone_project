# Runtime and Budget Log: Exp_010 (Week 4 Isolation Run #4 — Gaussian Size Scaling)
**Date:** 2026-05-05

## 1. Measured Runtime
| Mode | Wall Time | Per-Firm | Notes |
|---|---|---|---|
| **Warm cache (this run)** | ~1.3 s | ~0.021 s / firm | Scrape: 0 new fetches — all 62 firms hit `logs/scrape_cache.json`. Ridge fit ×5 folds + diagnostic refit + rounding + TSV write. |
| **Cold scrape (hypothetical)** | ~260 s | ~4.2 s / firm | Same scrape budget as exp_003+; the formula change adds zero network calls. |

* **Per-firm runtime budget compliance:** 0.021 s / firm ≪ 10 s budget from `program.md`. ✓
* **Marginal runtime cost vs exp_006 Control:** effectively zero. The Gaussian formula `np.exp(-((employees - 20) ** 2) / (2 * (10 ** 2)))` is a single vectorized operation over 62 floats — sub-millisecond.
* **Components (warm cache):**
  * Scrape: cache hit on all 62 entries (instant)
  * Featurization: identical to Control except `sweet_spot_emp` now uses `np.exp` instead of a boolean comparison; both are vectorized
  * Model: Ridge fit ×5 folds at p=10 (unchanged from Control)
  * Post-process: identical (clip + 0.5-grid round)
  * Snapshot: post-run `cp model.py logs/Snapshot_model_Exp_010.py` (~5 ms)

## 2. Estimated API / Data Cost
* **Data Source:** no new sources. The Gaussian uses the same `# Employees` column already loaded since exp_001.
* **Cost per Credit:** $0.02 (Apollo, unchanged)
* **Total Leads Processed:** 62
* **Marginal Cost vs. exp_006 Control:** **$0.00** — exp_010 is a pure feature-formula change.
* **Cumulative Cost Through exp_010:** **$1.24** (Apollo firmographics, unchanged across all ten experiments).

## 3. Scalability Projection
| Workload | exp_006 Control | exp_010 Gaussian | Notes |
|---|---|---|---|
| 1,000 leads, warm | ~12 s | ~12 s | `np.exp` over 1,000 floats is sub-millisecond. |
| 10,000 leads, warm | ~2 min | ~2 min | Same. |
| 1,000 leads, cold scrape | ~7 min @ 10× concurrency | ~7 min @ 10× concurrency | Network I/O dominates. |

* **No new bottlenecks introduced.** The Gaussian formula is O(n) and vectorized.
* **Note for any future hyperparameter-tuned variant:** a `GridSearchCV` over μ × σ with 5-fold inner CV on the Gaussian would multiply fit time by ~20× (a 4×5 grid at 5 folds), still ≪ 30 seconds at this dataset size. Mentioned for completeness; not the recommended next step.

## 4. Cumulative Budget Through Exp_010
| Item | Amount |
|---|---|
| Apollo firmographics (62 firms × $0.02) | $1.24 |
| Web scraping (own bandwidth, cached from exp_003) | $0.00 |
| Compute (local sklearn) | $0.00 |
| **Total** | **$1.24** |

## 5. Code Instability Audit
* **Edits made to `model.py`:**
  * Single-line replacement on line 184: `sweet_spot_emp = ((employees >= 10) & (employees <= 30)).astype(int)` → `sweet_spot_emp = np.exp(-((employees - 20) ** 2) / (2 * (10 ** 2)))`
* **Pre-edit revert:** `cp logs/Snapshot_model_Exp_006.py model.py` followed by `diff` — byte-identical confirmed before any edits.
* **Frozen-file modifications:** none. `eval/prepare.py`, `run_experiment.py`, `verify_integrity.py` all unchanged from the post-exp_006 re-baseline.
* **SHA-256 verification:** passed at run start (`8f7aa10f25b1...`).
* **Worker exit code:** 0. Judge exit code: 0.
* **Failures, warnings, or partial outputs:** none. `np.exp` produced no overflow / underflow on the 62 input values.
* **Code Instability classification:** **none triggered.**

## 6. Snapshot Protocol Compliance
* `logs/Snapshot_model_Exp_010.py` written immediately after the Worker completed (8232 bytes; 1 byte smaller than `Snapshot_model_Exp_006.py` due to the Gaussian expression being one character shorter than the boolean comparison).
* Snapshot directory state after this run:
  * `logs/Snapshot_model_Exp_006.py` (8233 bytes) — current Week-4 Control / canonical revert reference
  * `logs/Snapshot_model_Exp_007.py` (8426 bytes) — Lasso isolation (discarded)
  * `logs/Snapshot_model_Exp_008.py` (8409 bytes) — Weighted MRR isolation (discarded)
  * `logs/Snapshot_model_Exp_009.py` (8322 bytes) — rev_per_emp isolation (current best)
  * `logs/Snapshot_model_Exp_010.py` (8232 bytes) — Gaussian sweet-spot (this run)
* **Pending action:** before exp_011, `cp logs/Snapshot_model_Exp_006.py model.py` to revert to Control. Held for explicit user confirmation per the established Snapshot Protocol pattern.

## 7. Notes on the Signal Failure
* This is the **third Week-4 Signal Failure** out of four ablations. Pattern across the set so far:
  * exp_007 (Lasso, alpha=0.1): regressor swap → +14% RMSE — pruned `tenure_sq`, the load-bearing bell-curve helper.
  * exp_008 (Weighted MRR keywords): keyword-list weighting → +5% RMSE — premium keywords too universal in cohort to discriminate.
  * exp_009 (`rev_per_emp`): structural feature add → **−7% RMSE (Signal Success, new best)**.
  * exp_010 (Gaussian sweet-spot): feature formula swap → +11% RMSE — multicollinearity reshuffle let `log_employees` overfit very large firms.
* **Cumulative wall-time spent on Week-4 controlled experiments: ~6.7 seconds.** Four Isolation Runs across three failure modes and one structural success, with one new all-time best (exp_009 at RMSE 1.3955).
* **Cumulative cost: $0.00 marginal beyond the original $1.24 Apollo charge.** Every Week-4 experiment has been a pure modeling/feature change against already-loaded data.
* **Diagnostic accumulator (carrying forward from earlier runs):**
  * `tenure_sq` is structurally load-bearing — never pruned, never swung sign across exp_007–exp_010.
  * `mgmt_depth` is dispensable — coefficient stays in the +0.02 to +0.06 band; pruning it (exp_007 Lasso) cost roughly nothing.
  * Structural ratios (revenue / headcount) carry more discriminative signal than text keywords (exp_008 vs exp_009).
  * Smoothing a binary feature can break implicit regularization on its correlated neighbors (exp_010 → `log_employees`).
