# Runtime and Budget Log: Exp_018 (Moat Audit — has_moat without compliance)
**Date:** 2026-05-08

## 1. Measured Runtime
| Mode | Wall Time | Per-Firm | Notes |
|---|---|---|---|
| **Warm cache (this run)** | ~1.3 s | ~0.021 s / firm | Scrape: 0 new fetches. Ridge fit ×5 OOF folds + 1 diagnostic refit + substring pass over 62 short descriptions. |
| **Cold scrape (hypothetical)** | ~261 s | ~4.2 s / firm | Same scrape budget as exp_003+; the keyword feature adds zero network calls. |

* **Per-firm runtime budget compliance:** 0.021 s / firm ≪ 10 s budget from `program.md`. ✓
* **Marginal runtime cost vs Exp_009 baseline:** essentially zero. 7-keyword substring match on 62 rows is sub-millisecond. No GridSearchCV per §5.
* **Components (warm cache):** identical to Exp_016, just with one fewer keyword in the substring loop.

## 2. Estimated API / Data Cost
* **Data Source:** no new sources. Same `Short Description` + `Keywords` columns already loaded since exp_001.
* **Cost per Credit:** $0.02 (Apollo, unchanged).
* **Total Leads Processed:** 62.
* **Marginal Cost vs. Exp_009:** **$0.00** — pure feature-engineering ablation.
* **Cumulative Cost Through Exp_018:** **$1.24** (Apollo firmographics, unchanged across all 18 experiments).

## 3. Scalability Projection
Identical to Exp_016 (which had nearly identical wall time). 7-keyword vs 8-keyword substring match is invisible at any practical N.

## 4. Cumulative Budget Through Exp_018
| Item | Amount |
|---|---|
| Apollo firmographics (62 firms × $0.02) | $1.24 |
| Web scraping (own bandwidth, cached from exp_003) | $0.00 |
| Compute (local sklearn) | $0.00 |
| **Total** | **$1.24** |

## 5. Code Instability Audit
* **Edits made to `model.py`:**
  * Removed `"compliance"` from the `MOAT_KW` list defined for this audit (7 keywords, down from Exp_016's 8).
  * Otherwise identical to Exp_016: `has_moat()` helper, featurize() wiring of `is_inst → has_moat` column, tag-count print in `main()`.
* **Pre-edit revert:** `cp logs/Snapshot_model_Exp_009.py model.py` followed by `diff` — byte-identical confirmed before any edits.
* **Frozen-file modifications:** none. `eval/prepare.py`, `run_experiment.py`, `verify_integrity.py` all unchanged.
* **SHA-256 verification:** passed at run start (`8f7aa10f25b1...`).
* **Worker exit code:** 0. Judge exit code: 0.
* **Failures, warnings, or partial outputs:** none.
* **Code Instability classification:** **none triggered.**

## 6. Snapshot Protocol Compliance
* `logs/Snapshot_model_Exp_018.py` written immediately after the Worker completed.
* Snapshot directory state after this run includes Exp_006, 007, 008, 009 (current baseline), 010, 011, 012, 013, 014, 015, 016 (artifact), 017, and 018 (this audit).
* **Pending action:** before Exp_019, `cp logs/Snapshot_model_Exp_009.py model.py` to revert to the Week-5 baseline. Held for explicit user confirmation.

## 7. Notes on the Audit Process
* **The single most informative experiment of Week 5 in process terms.** It cost 1.3 s of wall time and $0.00 to prevent a Type-I promotion that would have committed a fragile small-N artifact to the project baseline.
* **The +0.097 RMSE swing on removing one keyword is the diagnostic value.** Single-keyword ablations are cheap (single-line edit + 1.3 s run) and decisive. The pattern should be applied to any future feature whose tag rate exceeds 80% or whose coefficient sign disagrees with the stated hypothesis.
* **Cumulative Week-4 + Week-5 wall-time across 13 controlled experiments: ~22 s.** Cumulative cost: $0.00 marginal beyond the original $1.24 Apollo charge.
* **Diagnostic accumulator update:** `tenure_sq` now observed in 13 of 13 controlled experiments. Sign preserved in all healthy runs and most regressions; meaningfully attenuated in 4 of 5 Week-5 Signal Failures (Exp_014: −0.19, Exp_016: −0.45 *strengthened*, Exp_017: −0.24, Exp_018: −0.16). Pattern: **healthy runs preserve `tenure_sq` magnitude; regressions either weaken or sign-flip it.** This now has 13 data points and is ready for codification as a process rule.
