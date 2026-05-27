# Runtime and Budget Log: Exp_008 (Week 4 Isolation Run #2 — Weighted Revenue Quality)
**Date:** 2026-05-05

## 1. Measured Runtime
| Mode | Wall Time | Per-Firm | Notes |
|---|---|---|---|
| **Warm cache (this run)** | ~1.3 s | ~0.021 s / firm | Scrape: 0 new fetches — all 62 firms hit `logs/scrape_cache.json`. Ridge fit ×5 folds + diagnostic refit + rounding + TSV write. |
| **Cold scrape (hypothetical)** | ~260 s | ~4.2 s / firm | Same scrape budget as exp_003+; the keyword-weighting change adds zero network calls. |

* **Per-firm runtime budget compliance:** 0.021 s / firm ≪ 10 s budget from `program.md`. ✓
* **Marginal runtime cost vs exp_006 Control:** effectively zero. The change replaces a single-pass `count_kw` with two single-pass `count_kw` calls and a multiply — both vectorized over a length-62 column.
* **Components (warm cache):**
  * Scrape: cache hit on all 62 entries (instant)
  * Featurization: 1 extra `count_kw` pass over the same text bundle + 1 multiply (sub-millisecond)
  * Model: identical Ridge fit cost to Control (p=10 unchanged)
  * Post-process: identical (clip + 0.5-grid round)
  * Snapshot: post-run `cp model.py logs/Snapshot_model_Exp_008.py` (~5 ms)

## 2. Estimated API / Data Cost
* **Data Source:** no new sources. The keyword-weighting change uses the same text bundle (`Rationale + Keywords + Technologies + Short Description`) already loaded from `data/train_set.csv` in every run since exp_002.
* **Cost per Credit:** $0.02 (Apollo, unchanged since exp_001)
* **Total Leads Processed:** 62
* **Marginal Cost vs. exp_006 Control:** **$0.00** — exp_008 is a pure feature-engineering change; no data acquisition, no new external calls.
* **Cumulative Cost Through exp_008:** **$1.24** (Apollo firmographics, unchanged across all eight experiments).

## 3. Scalability Projection
| Workload | exp_006 Control | exp_008 Weighted MRR | Notes |
|---|---|---|---|
| 1,000 leads, warm | ~12 s | ~12 s | Doubling the keyword-list passes adds <1% wall-time overhead. |
| 10,000 leads, warm | ~2 min | ~2 min | Same. |
| 1,000 leads, cold scrape | ~7 min @ 10× concurrency | ~7 min @ 10× concurrency | Network I/O dominates. |

* **No new bottlenecks introduced.** The keyword-weighting math is O(n × |kw_list|) and the lists are small (7 + 5 = 12 keywords vs the Control's 9).
* **Note for future rare-keyword experiments:** scaling the keyword set to dozens of phrases would still keep the per-firm cost in microseconds — the cohort size (62), not the keyword count, is the cost driver.

## 4. Cumulative Budget Through Exp_008
| Item | Amount |
|---|---|
| Apollo firmographics (62 firms × $0.02) | $1.24 |
| Web scraping (own bandwidth, cached from exp_003) | $0.00 |
| Compute (local sklearn) | $0.00 |
| **Total** | **$1.24** |

## 5. Code Instability Audit
* **Edits made to `model.py`:**
  * Replaced single `RECURRING_KW` list with two disjoint lists: `PREMIUM_MRR_KW` (7 phrases) and `STANDARD_SERVICE_KW` (5 phrases), plus a `PREMIUM_MRR_WEIGHT = 2.5` constant.
  * Updated the `recurring = text.apply(...)` line in `featurize()` to compute `2.5 × count(PREMIUM_MRR) + 1.0 × count(STANDARD_SERVICE)`.
* **Pre-edit revert:** `cp logs/Snapshot_model_Exp_006.py model.py` followed by `diff` — byte-identical confirmed before any edits.
* **Frozen-file modifications:** none. `eval/prepare.py`, `run_experiment.py`, `verify_integrity.py` all unchanged from the post-exp_006 re-baseline.
* **SHA-256 verification:** passed at run start (`8f7aa10f25b1...`).
* **Worker exit code:** 0. Judge exit code: 0.
* **Failures, warnings, or partial outputs:** none.
* **Code Instability classification:** **none triggered.**

## 6. Snapshot Protocol Compliance
* `logs/Snapshot_model_Exp_008.py` written immediately after the Worker completed (8409 bytes; 176 bytes larger than `Snapshot_model_Exp_006.py` due to the split keyword lists and weight constant).
* Snapshot directory state after this run:
  * `logs/Snapshot_model_Exp_006.py` (8233 bytes) — Control / canonical revert reference
  * `logs/Snapshot_model_Exp_007.py` (8426 bytes) — Lasso isolation
  * `logs/Snapshot_model_Exp_008.py` (8409 bytes) — Weighted MRR isolation
* **Pending action:** before exp_009, `cp logs/Snapshot_model_Exp_006.py model.py` to revert to Control (with `diff` verification). The revert was deliberately not auto-executed so the user can inspect the Weighted-MRR state first.

## 7. Notes on the Signal Failure
* This is the second consecutive Signal Failure of the Week-4 set (exp_007 Lasso, exp_008 Weighted MRR). Both produced rich diagnostics — `tenure_sq` is structurally load-bearing, `mgmt_depth` is dispensable, L1 at alpha=0.1 is too aggressive, and pre-scaler keyword weighting under-delivers because StandardScaler renormalizes.
* **Cumulative wall-time spent on Week-4 controlled experiments so far: ~4.0 seconds.** The cost of "knowing precisely which levers don't work" is essentially free at this dataset size — exactly the value proposition of the controlled-experiment protocol over bundled changes.
* **Cumulative cost: $0.00 marginal beyond the original $1.24 Apollo charge** — every Week-4 experiment so far has been a pure modeling/feature change against already-loaded data.
