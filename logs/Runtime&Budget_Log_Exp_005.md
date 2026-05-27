# Runtime and Budget Log: Exp_005
**Date:** 2026-04-28

## 1. Measured Runtime
| Mode | Wall Time | Per-Firm | Notes |
|---|---|---|---|
| **Warm cache (typical)** | 8.5s | ~0.14s / firm | Scrape cache reused; HGBR fit ×5 folds + permutation importance (10 repeats × 10 features). |
| **Cold scrape (hypothetical)** | ~268s | ~4.3s / firm | Same scrape budget as exp_003; HGBR fit cost is negligible compared to network I/O. |

* **Per-firm runtime budget compliance:** 0.14s / firm (warm) ≪ 10s budget. ✓
* **Where the time goes (warm cache):** Permutation importance dominates (~7s) — it refits-and-rescores 100 times. The actual `cross_val_predict` is ~1s; the HGBR fit on full data for the diagnostic is ~0.3s.
* **Optimization note:** `permutation_importance(n_repeats=10)` is a deliberate diagnostic cost, not a per-prediction cost. In a production sourcing pipeline this step would only run during model evaluation, not on each new lead.

## 2. Estimated API / Data Cost
* **Data Source:** no new sources. All inputs already loaded from `train_set.csv` or cached from exp_003.
* **Marginal Cost vs. exp_004:** $0.00 — exp_005 swaps the regressor and adds a post-process; no new data acquisition.
* **Cumulative Cost Through exp_005:** $1.24 (Apollo firmographics, unchanged since exp_001).

## 3. Scalability Projection
| Workload | exp_003 (Ridge, warm) | exp_005 (HGBR, warm) | Notes |
|---|---|---|---|
| 1,000 leads | ~12 sec | ~140 sec | HGBR + permutation importance is ~10× slower than Ridge, dominated by the diagnostic step. |
| 10,000 leads | ~2 min | ~24 min | Same ratio; for production *prediction* (no diagnostic) HGBR would be much closer to Ridge. |
| 1,000 leads, cold scrape | ~7 min @ 10× concurrency | ~7 min @ 10× concurrency | Network I/O dominates for both. |

* **Real bottleneck for scaling remains web scraping**, not regression. HGBR's fit time at p=10 features and any reasonable N is sub-linear in N; the diagnostic is what makes it look slow at small scale.

## 4. Cumulative Budget Through Exp_005
| Item | Amount |
|---|---|
| Apollo firmographics (62 firms × $0.02) | $1.24 |
| Web scraping (own bandwidth, cached from exp_003) | $0.00 |
| Compute (local sklearn) | $0.00 |
| **Total** | **$1.24** |

## 5. Notes on the Regression
* exp_005 is the first experiment in this dry-run series where wall-time *increased*, which a reader might assume reflects a more capable model. The opposite is true here — the wall-time increase came from the permutation-importance diagnostic, not from a more accurate model. RMSE got worse.
* The compute-vs-quality tradeoff for this dataset (N=62) is sharp: Ridge runs in ~1 second and produces RMSE 1.50; HGBR runs in ~9 seconds (with diagnostic) and produces RMSE 2.18. At small N, the simpler model wins on both axes.
