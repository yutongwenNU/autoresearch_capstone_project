# Runtime and Budget Log: Exp_006 (Week 4 Control)
**Date:** 2026-05-05

## 1. Measured Runtime
| Mode | Wall Time | Per-Firm | Notes |
|---|---|---|---|
| **Warm cache (this run)** | ~1.3 s | ~0.021 s / firm | Scrape: 0 new fetches — all 62 firms hit `logs/scrape_cache.json` populated in exp_003. Ridge fit ×5 folds + diagnostic refit + rounding + TSV write. |
| **Cold scrape (hypothetical)** | ~260 s | ~4.2 s / firm | Same scrape budget as exp_003; rounding adds <1 ms. |

* **Per-firm runtime budget compliance:** 0.021 s / firm ≪ 10 s budget from `program.md`. ✓
* **Components (warm cache):**
  * Scrape: cache hit on all 62 entries (instant)
  * Featurization: identical to exp_003 (10 features, p=10, n=62)
  * Model: Ridge fit ×5 folds via `cross_val_predict` + 1 full-data fit for the diagnostic
  * Post-process: `np.clip` + `np.round(preds * 2) / 2` — both vectorized on a length-62 array, sub-millisecond
* **Marginal runtime cost vs exp_003:** effectively zero — the rounding is a single vectorized `np.round` over 62 floats.

## 2. Estimated API / Data Cost
* **Data Source:** no new sources. All inputs already loaded from `data/train_set.csv` or cached from exp_003.
* **Cost per Credit:** $0.02 (Apollo, unchanged since exp_001)
* **Total Leads Processed:** 62
* **Marginal Cost vs. exp_003:** **$0.00** — exp_006 is a pure post-processing change; no data acquisition, no new external calls.
* **Cumulative Cost Through exp_006:** **$1.24** (Apollo firmographics, unchanged across all six experiments).

## 3. Scalability Projection
| Workload | exp_003 (Ridge, no rounding) | exp_006 (Ridge + 0.5 rounding) | Notes |
|---|---|---|---|
| 1,000 leads, warm | ~12 s | ~12 s | Rounding is O(n); negligible at any practical N. |
| 10,000 leads, warm | ~2 min | ~2 min | Same. |
| 1,000 leads, cold scrape | ~7 min @ 10× concurrency | ~7 min @ 10× concurrency | Network I/O dominates; rounding cost is invisible. |

* **No new bottlenecks introduced** — the rounding adds no compute, no I/O, no memory pressure.
* **Production note:** if this rounding were ever adopted for live scoring, it would reduce downstream consumer code complexity (predictions match the label vocabulary) at zero compute cost. The reason not to adopt it is statistical (RMSE neutral-to-slightly-worse), not operational.

## 4. Cumulative Budget Through Exp_006
| Item | Amount |
|---|---|
| Apollo firmographics (62 firms × $0.02) | $1.24 |
| Web scraping (own bandwidth, cached from exp_003) | $0.00 |
| Compute (local sklearn) | $0.00 |
| **Total** | **$1.24** |

## 5. Code Instability Audit
* **Edits made to `model.py`:** one line — `preds = np.round(preds * 2) / 2` inserted after the existing `np.clip(preds, 1.0, 10.0)`.
* **Frozen-file modifications:** none. `eval/prepare.py`, `run_experiment.py`, `verify_integrity.py` all unchanged.
* **SHA-256 verification:** passed at run start (`570d9e2a89c8...`).
* **Worker exit code:** 0. Judge exit code: 0.
* **Failures, warnings, or partial outputs:** none.
* **Code Instability classification:** **none triggered.** The Control ran cleanly end-to-end on the first attempt.

## 6. Notes on the Control
* This is the first run in the Week 4 *Controlled* set. The pattern going forward: every variable change is tested in isolation against a known reference (here, exp_003) so that the *cause* of any RMSE shift is unambiguous. exp_005 demonstrated the failure mode of bundling — two changes, one effect, no clean attribution.
* The wall-time of this Control (~1.3 s) is the lower bound for the Week-4 set. Any Week-4 follow-up that takes substantially longer (e.g., feature scraping, larger ensembles) is paying for that compute with a specific, articulated hypothesis.
