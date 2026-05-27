# Runtime and Budget Log: Exp_004
**Date:** 2026-04-28

## 1. Measured Runtime
* **Total Time:** ~1.2s wall (warm scrape cache; sklearn fit + 5-fold OOF + write)
* **Average Time per Lead:** ~0.019 seconds
* **Components:**
  * Scrape: 0 new fetches — all 62 firms hit the disk cache populated in exp_003
  * Featurization: identical to exp_003 + one O(n) elementwise multiplication for the `succession_gap` term
  * Model: Ridge fit ×5 folds (≈ same cost as exp_003; one extra column at p=11 vs p=10 is negligible)
* **Per-firm runtime budget compliance:** 0.019s / firm ≪ 10s budget. ✓

## 2. Estimated API / Data Cost
* **Data Source:** no new sources. All inputs already cached or loaded from `train_set.csv`.
* **Cost per Credit:** $0.02 (Apollo, unchanged)
* **Total Leads Processed:** 62
* **Marginal Cost vs. exp_003:** $0.00 — exp_004 introduces a derived feature only, not a new data source.
* **Cumulative Cost Through exp_004:** $1.24

## 3. Scalability Projection
* **1,000 leads:** ~19 seconds (warm cache); ~7 minutes if scrape cache must be rebuilt at 10× concurrency.
* **10,000 leads:** ~3 minutes (warm cache); ~1.2 hours if cold scrape at 10× concurrency.
* **No new bottlenecks introduced** — the interaction term is pure compute on already-loaded columns.

## 4. Cumulative Budget Through Exp_004
| Item | Amount |
|---|---|
| Apollo firmographics (62 firms × $0.02) | $1.24 |
| Web scraping (own bandwidth, cached from exp_003) | $0.00 |
| Compute (local sklearn) | $0.00 |
| **Total** | **$1.24** |

## 5. Notes
* Because the run was discarded (RMSE regressed), the marginal compute cost is the only "wasted" budget — and at sub-2-second wall-clock, that's effectively zero.
* The diagnostic value (confirming the interaction signal is correctly signed) justifies the run independent of the RMSE outcome.
