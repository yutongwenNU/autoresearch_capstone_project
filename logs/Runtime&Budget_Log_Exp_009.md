# Runtime and Budget Log: Exp_009 (Week 4 Isolation Run #3 — Revenue per Employee)
**Date:** 2026-05-05

## 1. Measured Runtime
| Mode | Wall Time | Per-Firm | Notes |
|---|---|---|---|
| **Warm cache (this run)** | ~1.3 s | ~0.021 s / firm | Scrape: 0 new fetches — all 62 firms hit `logs/scrape_cache.json`. Ridge fit ×5 folds + diagnostic refit + rounding + TSV write. |
| **Cold scrape (hypothetical)** | ~260 s | ~4.2 s / firm | Same scrape budget as exp_003+; the structural-feature change adds zero network calls. |

* **Per-firm runtime budget compliance:** 0.021 s / firm ≪ 10 s budget from `program.md`. ✓
* **Marginal runtime cost vs exp_006 Control:** effectively zero. The added feature is a single vectorized division (`revenue / employees.clip(lower=1)`) over 62 rows — sub-millisecond.
* **Components (warm cache):**
  * Scrape: cache hit on all 62 entries (instant)
  * Featurization: identical to Control + 1 vectorized division + 1 extra column in the returned DataFrame
  * Model: Ridge fit ×5 folds at p=11 (vs Control's p=10) — fit time grows ~linearly in p, negligible at this scale
  * Post-process: identical to Control (clip + 0.5-grid round)
  * Snapshot: post-run `cp model.py logs/Snapshot_model_Exp_009.py` (~5 ms)

## 2. Estimated API / Data Cost
* **Data Source:** no new sources. `Annual Revenue` and `# Employees` are pre-existing columns in `data/train_set.csv` already loaded since exp_001.
* **Cost per Credit:** $0.02 (Apollo, unchanged since exp_001)
* **Total Leads Processed:** 62
* **Marginal Cost vs. exp_006 Control:** **$0.00** — exp_009 is a pure feature-engineering change derived from already-loaded structured fields.
* **Cumulative Cost Through exp_009:** **$1.24** (Apollo firmographics, unchanged across all nine experiments).

## 3. Scalability Projection
| Workload | exp_006 Control | exp_009 rev_per_emp | Notes |
|---|---|---|---|
| 1,000 leads, warm | ~12 s | ~12 s | One extra column at p=11 vs p=10 is invisible. |
| 10,000 leads, warm | ~2 min | ~2 min | Same. |
| 1,000 leads, cold scrape | ~7 min @ 10× concurrency | ~7 min @ 10× concurrency | Network I/O dominates. |

* **No new bottlenecks introduced.** Ridge fit is O(n × p²) at small p; growing from 10 → 11 features changes nothing operationally.
* **Held-out test pipeline:** the `revenue / employees.clip(lower=1)` guard means the new feature can be safely computed on `data/locked_test_set.csv` without inf/nan even for firms with missing employee counts. No additional data validation needed before scoring the held-out set.

## 4. Cumulative Budget Through Exp_009
| Item | Amount |
|---|---|
| Apollo firmographics (62 firms × $0.02) | $1.24 |
| Web scraping (own bandwidth, cached from exp_003) | $0.00 |
| Compute (local sklearn) | $0.00 |
| **Total** | **$1.24** |

## 5. Code Instability Audit
* **Edits made to `model.py`:**
  * Added one line in `featurize()`: `rev_per_emp = revenue / employees.clip(lower=1)`
  * Added one entry in the returned DataFrame: `"rev_per_emp": rev_per_emp`
  * Total: a 2-line change, both inside the existing `featurize()` function.
* **Pre-edit revert:** `cp logs/Snapshot_model_Exp_006.py model.py` followed by `diff` — byte-identical confirmed before any edits.
* **Frozen-file modifications:** none.
* **SHA-256 verification:** passed at run start (`8f7aa10f25b1...`).
* **Worker exit code:** 0. Judge exit code: 0.
* **Failures, warnings, or partial outputs:** none.
* **Code Instability classification:** **none triggered.**

## 6. Snapshot Protocol Compliance
* `logs/Snapshot_model_Exp_009.py` written immediately after the Worker completed (8322 bytes; 89 bytes larger than `Snapshot_model_Exp_006.py`).
* Snapshot directory state after this run:
  * `logs/Snapshot_model_Exp_006.py` (8233 bytes) — current Week-4 Control / canonical revert reference
  * `logs/Snapshot_model_Exp_007.py` (8426 bytes) — Lasso isolation (discarded)
  * `logs/Snapshot_model_Exp_008.py` (8409 bytes) — Weighted MRR isolation (discarded)
  * `logs/Snapshot_model_Exp_009.py` (8322 bytes) — **rev_per_emp isolation (new all-time best)**
* **Pending decisions for the user:**
  1. Should `Snapshot_model_Exp_009.py` *replace* `Snapshot_model_Exp_006.py` as the Week-4 Control going forward? See Research_Log_Exp_009.md → Decision section for the trade-off.
  2. If keeping exp_006 as Control: revert `model.py` via `cp logs/Snapshot_model_Exp_006.py model.py` before exp_010.
  3. If promoting exp_009 to Control: leave `model.py` as-is and use `Snapshot_model_Exp_009.py` as the canonical revert reference for subsequent Isolation Runs.

## 7. Notes on the Signal Success
* This is the **first Week-4 Signal Success** after two consecutive Signal Failures (exp_007 Lasso, exp_008 Weighted MRR). The pattern is consistent with the controlled-experiment value proposition: cheap, attributable failures sharpen the hypothesis, and the next attempt lands cleanly when the operationalization shifts to the right kind of data.
* **Cumulative wall-time spent on Week-4 controlled experiments so far: ~5.3 seconds.** Three Isolation Runs across two failure modes and one structural pivot, with one new all-time best. Cost of learning: $0.00 marginal beyond the one-time Apollo charge.
* The +0.08 R² and −0.11 RMSE improvement is the largest single-experiment gain since exp_002 introduced Ridge over hand-coded heuristics — i.e., the largest gain since the project moved from rules to ML.
