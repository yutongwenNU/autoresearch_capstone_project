# Runtime and Budget Log: Exp_007 (Week 4 Isolation Run #1 — Lasso)
**Date:** 2026-05-05

## 1. Measured Runtime
| Mode | Wall Time | Per-Firm | Notes |
|---|---|---|---|
| **Warm cache (this run)** | ~1.4 s | ~0.023 s / firm | Scrape: 0 new fetches — all 62 firms hit `logs/scrape_cache.json`. Lasso fit ×5 folds + diagnostic refit + rounding + TSV write. |
| **Cold scrape (hypothetical)** | ~260 s | ~4.2 s / firm | Same scrape budget as exp_003+; Lasso fit cost is negligible vs network I/O. |

* **Per-firm runtime budget compliance:** 0.023 s / firm ≪ 10 s budget from `program.md`. ✓
* **Marginal runtime cost vs exp_006 Control:** ~+0.1 s wall — Lasso's coordinate-descent solver is slightly slower than Ridge's closed-form solve at p=10, but the difference is invisible at this scale. `max_iter=10000` was set proactively; the actual iterations needed for convergence at alpha=0.1 on standardized features were well under that ceiling (no `ConvergenceWarning` emitted).
* **Components (warm cache):**
  * Scrape: cache hit on all 62 entries (instant)
  * Featurization: identical to exp_006 (10 features, p=10, n=62)
  * Model: Lasso fit ×5 folds via `cross_val_predict` + 1 full-data fit for the diagnostic
  * Post-process: `np.clip` + `np.round(preds * 2) / 2` — sub-millisecond
  * Snapshot: post-run `cp model.py logs/Snapshot_model_Exp_007.py` (~5 ms)

## 2. Estimated API / Data Cost
* **Data Source:** no new sources. All inputs already loaded from `data/train_set.csv` or cached from exp_003.
* **Cost per Credit:** $0.02 (Apollo, unchanged since exp_001)
* **Total Leads Processed:** 62
* **Marginal Cost vs. exp_006 Control:** **$0.00** — exp_007 is a pure regressor swap; no data acquisition, no new external calls.
* **Cumulative Cost Through exp_007:** **$1.24** (Apollo firmographics, unchanged across all seven experiments).

## 3. Scalability Projection
| Workload | exp_006 Control (Ridge) | exp_007 (Lasso) | Notes |
|---|---|---|---|
| 1,000 leads, warm | ~12 s | ~13 s | Lasso ~10% slower at this p; both bounded by O(n × p × iter). |
| 10,000 leads, warm | ~2 min | ~2.2 min | Same ratio. |
| 1,000 leads, cold scrape | ~7 min @ 10× concurrency | ~7 min @ 10× concurrency | Network I/O dominates. |

* **No new bottlenecks introduced.** Lasso's coordinate descent stays linear-ish in features at this dimensionality.
* **Note for future alpha sweeps:** running a small grid (e.g., `LassoCV` with 5 alphas × 5 folds) would multiply fit time by ~25× — still ≪ 1 minute on this dataset. Worth considering as a follow-up Isolation Run.

## 4. Cumulative Budget Through Exp_007
| Item | Amount |
|---|---|
| Apollo firmographics (62 firms × $0.02) | $1.24 |
| Web scraping (own bandwidth, cached from exp_003) | $0.00 |
| Compute (local sklearn) | $0.00 |
| **Total** | **$1.24** |

## 5. Code Instability Audit
* **Edits made to `model.py`:**
  * `from sklearn.linear_model import Ridge` → `from sklearn.linear_model import Lasso`
  * `("ridge", Ridge(alpha=1.0))` → `("lasso", Lasso(alpha=0.1, max_iter=10000, random_state=42))`
  * Diagnostic block: `named_steps["ridge"]` → `named_steps["lasso"]`; print label updated to `"Lasso coefficients ..."` and now reports the count of pruned features inline.
* **Frozen-file modifications:** none. `eval/prepare.py`, `run_experiment.py`, `verify_integrity.py` all unchanged from the post-exp_006 re-baseline.
* **SHA-256 verification:** passed at run start (`8f7aa10f25b1...`).
* **Worker exit code:** 0. Judge exit code: 0.
* **Failures, warnings, or partial outputs:** none. Lasso converged inside `max_iter=10000` with no `ConvergenceWarning`.
* **Code Instability classification:** **none triggered.**

## 6. Snapshot Protocol Compliance
* `logs/Snapshot_model_Exp_007.py` written immediately after the Worker completed (8426 bytes; 193 bytes larger than `Snapshot_model_Exp_006.py` due to the longer Lasso config + the added `n_zero` diagnostic line).
* `logs/Snapshot_model_Exp_006.py` (8233 bytes) remains the canonical revert reference for the next Isolation Run.
* **Pending action:** before exp_008, `cp logs/Snapshot_model_Exp_006.py model.py` and verify byte-identical with `diff` before proposing the next variable change. The revert was deliberately *not* auto-executed in this run so the user can review the Lasso state before the rollback.

## 7. Notes on the Signal Failure
* The +14% RMSE regression is the largest controlled-experiment shift since exp_005. Unlike exp_005 (which bundled HGBR + rounding and confounded the cause), this Isolation Run pins the regression on a single named variable: regularization type at this alpha.
* The runtime cost of *learning* this — that Lasso(alpha=0.1) prunes `tenure_sq` and crashes the bell-curve fit — was 1.4 s of wall time. This is the explicit value proposition of the controlled-experiment protocol: cheap, attributable failures.
