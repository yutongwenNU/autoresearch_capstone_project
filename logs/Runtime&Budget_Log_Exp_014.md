# Runtime and Budget Log: Exp_014 (Week 5 Isolation Run #4 — Institutionalization Red-Flag Index)
**Date:** 2026-05-08

## 1. Measured Runtime
| Mode | Wall Time | Per-Firm | Notes |
|---|---|---|---|
| **Warm cache (this run)** | ~1.3 s | ~0.021 s / firm | Scrape: 0 new fetches. Ridge fit ×5 OOF folds + 1 diagnostic refit + substring/regex pass over 62 short descriptions. |
| **Cold scrape (hypothetical)** | ~261 s | ~4.2 s / firm | Same scrape budget as exp_003+; the keyword feature adds zero network calls. |

* **Per-firm runtime budget compliance:** 0.021 s / firm ≪ 10 s budget from `program.md`. ✓
* **Marginal runtime cost vs exp_009 baseline:** essentially zero. Substring matching over 10 keywords + 1 compiled regex on 62 rows is sub-millisecond total. No GridSearchCV in this run per §5 Decoupled Isolation Rule.
* **Components (warm cache):**
  * Scrape: cache hit on all 62 entries (instant)
  * Featurization: identical to exp_009 + 1 new vectorized binary column from substring/regex matching (12 features, p=12, n=62)
  * Model: Ridge fit ×5 OOF folds + 1 full-data refit (closed-form, trivially fast)
  * Diagnostic: identical print format to exp_009 with the new `is_institutionalized` row in the sorted coefficient output; tag count pre-print added to `main()`
  * Post-process: identical clip + 0.5-grid round
  * Snapshot: post-run `cp model.py logs/Snapshot_model_Exp_014.py` (~5 ms)

## 2. Estimated API / Data Cost
* **Data Source:** no new sources. Keyword/regex match operates on `Short Description` and `Keywords` columns already loaded since exp_001.
* **Cost per Credit:** $0.02 (Apollo, unchanged)
* **Total Leads Processed:** 62
* **Marginal Cost vs. exp_009:** **$0.00** — pure feature-engineering change derived from already-loaded text fields.
* **Cumulative Cost Through Exp_014:** **$1.24** (Apollo firmographics, unchanged across all fourteen experiments).

## 3. Scalability Projection
| Workload | exp_009 (Ridge α=1) | exp_014 (Ridge + is_institutionalized) | Notes |
|---|---|---|---|
| 1,000 leads, warm | ~12 s | ~12 s | Substring match over 1,000 rows is sub-second. |
| 10,000 leads, warm | ~2 min | ~2 min | Same. |
| 1,000 leads, cold scrape | ~7 min @ 10× concurrency | ~7 min @ 10× concurrency | Network I/O still dominates. |

* **No new bottlenecks introduced.** The feature is a sparse binary derived from already-loaded text — operationally free.

## 4. Cumulative Budget Through Exp_014
| Item | Amount |
|---|---|
| Apollo firmographics (62 firms × $0.02) | $1.24 |
| Web scraping (own bandwidth, cached from exp_003) | $0.00 |
| Compute (local sklearn) | $0.00 |
| **Total** | **$1.24** |

## 5. Code Instability Audit
* **Edits made to `model.py`:**
  * Added `INSTITUTIONALIZED_KW` (10 substring keywords) and `INSTITUTIONALIZED_REGEXES` (1 compiled regex for `part of the [X] family`) constants near the existing keyword blocks.
  * Added `is_institutionalized(text)` helper function that returns 1 if any keyword or regex matches, 0 otherwise.
  * Inside `featurize()`: added a `Short Description + Keywords` text bundle (separate from the wider `text` bundle used for keyword counts) and applied `is_institutionalized` over it; added `"is_institutionalized": is_inst` as the 12th entry of the returned DataFrame.
  * Inside `main()`: assigned `featurize(...)` to a named variable `X_df` so the tag count can be printed (`is_institutionalized tagged: N/62 firms`) before passing `X_df.values` to `cross_val_predict`.
* **Pre-edit revert:** `cp logs/Snapshot_model_Exp_009.py model.py` followed by `diff` — byte-identical confirmed before any edits.
* **Frozen-file modifications:** none. `eval/prepare.py`, `run_experiment.py`, `verify_integrity.py` all unchanged.
* **SHA-256 verification:** passed at run start (`8f7aa10f25b1...`).
* **Worker exit code:** 0. Judge exit code: 0.
* **Failures, warnings, or partial outputs:** none. Regex compiled cleanly; no NaN/inf in the new column; no `ConvergenceWarning` from Ridge.
* **Code Instability classification:** **none triggered.**

## 6. Snapshot Protocol Compliance
* `logs/Snapshot_model_Exp_014.py` written immediately after the Worker completed (10049 bytes; 1727 bytes larger than `Snapshot_model_Exp_009.py` due to the keyword/regex constants, helper function, featurize wiring, and tag-count print line).
* Snapshot directory state after this run:
  * `logs/Snapshot_model_Exp_006.py` (8233 bytes) — Week-4 Control (legacy)
  * `logs/Snapshot_model_Exp_007.py` (8426 bytes) — Lasso (discarded)
  * `logs/Snapshot_model_Exp_008.py` (8409 bytes) — Weighted MRR (discarded)
  * `logs/Snapshot_model_Exp_009.py` (8322 bytes) — **current Week-5 baseline / canonical revert reference**
  * `logs/Snapshot_model_Exp_010.py` (8232 bytes) — Gaussian sweet-spot (discarded)
  * `logs/Snapshot_model_Exp_011.py` (8933 bytes) — Bagging (discarded)
  * `logs/Snapshot_model_Exp_012.py` (9477 bytes) — Tenure×Rev/Emp (discarded; stop rule fired)
  * `logs/Snapshot_model_Exp_013.py` (10004 bytes) — NLP founder_led (discarded; sparse signal)
  * `logs/Snapshot_model_Exp_014.py` (10049 bytes) — Institutionalization red-flag (this run)
* **Pending action:** before exp_015, `cp logs/Snapshot_model_Exp_009.py model.py` to revert to the Week-5 baseline. Held for explicit user confirmation.

## 7. Notes on the Sparse-Signal + Outlier-Leverage Signal Failure
* **+0.069 RMSE** is moderate — second-mildest Week-5 Signal Failure after exp_011 (Bagging, +0.051), tied roughly with exp_013 (founder_led, +0.060).
* **Sparse-NLP-binary-feature pattern is now firmly established across exp_013 and exp_014.** Both runs:
  * Validate the user's thesis at the *population level* (mean Manual Score gap of 2.4 points between tagged and untagged firms in exp_014; tag rate × correct-direction proxy in exp_013).
  * Receive correctly-signed Ridge coefficients (founder_led −0.11, is_institutionalized −0.36).
  * Fail RMSE due to tag-rate sparsity (≤ 6.5% positive rate).
* **The single-firm outlier (World Synergy, Manual Score 8.5) is the most expensive single observation in this dataset.** With 3 tagged firms, removing or correctly-classifying the outlier alone would likely flip the run from regression to neutral or improvement. Future re-tests of `is_institutionalized` should audit the tagged firms before running.
* **Cumulative diagnostic accumulator (Week-4 + Week-5 to date, 9 controlled experiments):**
  * `tenure_sq` is structurally load-bearing. Coefficient stays in [−0.27, −0.41] in healthy runs; weakens (exp_014: −0.19) or sign-flips (exp_007, 010, 012) in regressions.
  * `mgmt_depth` is dispensable. Coefficient stays in [+0.02, +0.18] band; never costly to add or remove.
  * Sparse binary features (≤ 10% tag rate) regress RMSE even when the population-level signal direction is correct.
  * High-coefficient new features (|coef| > 0.30) tend to draw weight from `tenure` + `tenure_sq` — a generalizable "redistribution cost" pattern.
* **Cumulative Week-4 + Week-5 wall-time: ~14.6 seconds** across nine controlled experiments. **Cumulative cost: $0.00 marginal beyond the original $1.24 Apollo charge.**
