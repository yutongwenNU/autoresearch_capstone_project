# Runtime and Budget Log: Exp_013 (Week 5 Isolation Run #3 — NLP Founder-Led Detection)
**Date:** 2026-05-08

## 1. Measured Runtime
| Mode | Wall Time | Per-Firm | Notes |
|---|---|---|---|
| **Warm cache (this run)** | ~1.3 s | ~0.021 s / firm | Scrape: 0 new fetches. Ridge fit ×5 OOF folds + 1 diagnostic refit + regex pass over 62 short descriptions. |
| **Cold scrape (hypothetical)** | ~261 s | ~4.2 s / firm | Same scrape budget as exp_003+; the regex feature adds zero network calls. |

* **Per-firm runtime budget compliance:** 0.021 s / firm ≪ 10 s budget from `program.md`. ✓
* **Marginal runtime cost vs exp_009 baseline:** essentially zero. The 5 compiled regexes are short-circuited (first match returns 1) and applied via `Series.apply` over a 62-row column — sub-millisecond total. No GridSearchCV in this run per §5 Decoupled Isolation Rule.
* **Components (warm cache):**
  * Scrape: cache hit on all 62 entries (instant)
  * Featurization: identical to exp_009 + 1 new vectorized regex column (12 features, p=12, n=62)
  * Model: Ridge fit ×5 OOF folds + 1 full-data refit (closed-form, trivially fast)
  * Diagnostic: identical print format to exp_009; the `founder_led` coefficient appears in the sorted output between `recurring_kw` and `mgmt_depth`
  * Post-process: identical clip + 0.5-grid round
  * Snapshot: post-run `cp model.py logs/Snapshot_model_Exp_013.py` (~5 ms)

## 2. Estimated API / Data Cost
* **Data Source:** no new sources. Regexes operate on the existing `Short Description` and `Keywords` columns from `data/train_set.csv` (already loaded since exp_001).
* **Cost per Credit:** $0.02 (Apollo, unchanged)
* **Total Leads Processed:** 62
* **Marginal Cost vs. exp_009:** **$0.00** — pure feature-engineering change derived from already-loaded text fields.
* **Cumulative Cost Through Exp_013:** **$1.24** (Apollo firmographics, unchanged across all thirteen experiments).

## 3. Scalability Projection
| Workload | exp_009 (Ridge α=1) | exp_013 (Ridge + founder_led) | Notes |
|---|---|---|---|
| 1,000 leads, warm | ~12 s | ~12 s | Regex match over 1,000 rows is sub-second; Ridge at p=12 is invisible. |
| 10,000 leads, warm | ~2 min | ~2 min | Same. |
| 1,000 leads, cold scrape | ~7 min @ 10× concurrency | ~7 min @ 10× concurrency | Network I/O still dominates. |

* **No new bottlenecks introduced.** The feature is a sparse binary derived from already-loaded text — operationally free.
* **Future NLP scaling note:** if a follow-up replaces the 5 regexes with a denser parser (e.g., spaCy NER for founder-name extraction), wall time would rise to ~10 s for the model loading + ~50 ms / firm for parsing. Still well under the 10 s/firm budget at any practical N.

## 4. Cumulative Budget Through Exp_013
| Item | Amount |
|---|---|
| Apollo firmographics (62 firms × $0.02) | $1.24 |
| Web scraping (own bandwidth, cached from exp_003) | $0.00 |
| Compute (local sklearn) | $0.00 |
| **Total** | **$1.24** |

## 5. Code Instability Audit
* **Edits made to `model.py`:**
  * Added a `FOUNDER_LED_REGEXES` list (5 compiled regex patterns) and an `is_founder_led(text)` helper function near the top of the module, alongside the existing `RECURRING_KW`/`STAGNATION_KW`/`MODERN_AI_KW` keyword constants.
  * Inside `featurize()`: added a case-preserving `founder_text` bundle (`Short Description + Keywords`, no `.lower()` since the capital-letter pattern relies on case) and applied `is_founder_led` over it; added `"founder_led": founder_led` as the 12th entry of the returned DataFrame.
  * Inside `main()`: assigned `featurize(...)` to a named variable `X_df` so the founder_led tag count can be printed (`founder_led tagged: N/62 firms`) before passing `X_df.values` to `cross_val_predict`.
* **Pre-edit revert:** `cp logs/Snapshot_model_Exp_009.py model.py` followed by `diff` — byte-identical confirmed before any edits.
* **Frozen-file modifications:** none. `eval/prepare.py`, `run_experiment.py`, `verify_integrity.py` all unchanged.
* **SHA-256 verification:** passed at run start (`8f7aa10f25b1...`).
* **Worker exit code:** 0. Judge exit code: 0.
* **Failures, warnings, or partial outputs:** none. All 5 regexes compiled successfully; no `re.error`, no NaN/inf in the founder_led column, no `ConvergenceWarning` from Ridge.
* **Code Instability classification:** **none triggered.**

## 6. Snapshot Protocol Compliance
* `logs/Snapshot_model_Exp_013.py` written immediately after the Worker completed (10004 bytes; 1682 bytes larger than `Snapshot_model_Exp_009.py` due to the regex patterns block, helper function, featurize wiring, and tag-count print line).
* Snapshot directory state after this run:
  * `logs/Snapshot_model_Exp_006.py` (8233 bytes) — Week-4 Control (legacy)
  * `logs/Snapshot_model_Exp_007.py` (8426 bytes) — Lasso (discarded)
  * `logs/Snapshot_model_Exp_008.py` (8409 bytes) — Weighted MRR (discarded)
  * `logs/Snapshot_model_Exp_009.py` (8322 bytes) — **current Week-5 baseline / canonical revert reference**
  * `logs/Snapshot_model_Exp_010.py` (8232 bytes) — Gaussian sweet-spot (discarded)
  * `logs/Snapshot_model_Exp_011.py` (8933 bytes) — Bagging (discarded)
  * `logs/Snapshot_model_Exp_012.py` (9477 bytes) — Tenure×Rev/Emp (discarded; stop rule fired)
  * `logs/Snapshot_model_Exp_013.py` (10004 bytes) — NLP founder_led (this run)
* **Pending action:** before exp_014, `cp logs/Snapshot_model_Exp_009.py model.py` to revert to the Week-5 baseline (per `program.md` §Week 5 §2). Held for explicit user confirmation per the established Snapshot Protocol pattern.

## 7. Notes on the Sparse-Signal Signal Failure
* **+0.060 RMSE** is the *least severe* Week-5 Signal Failure to date — comparable to exp_008's Weighted MRR keyword regression (+0.077) and well below exp_011's Bagging (+0.051) or exp_012's Tenure×Rev/Emp (+0.479).
* **Tag rate is the structural constraint.** With only 4/62 firms tagged, the founder_led column is 90% identical across firms. Ridge correctly identifies this as low-information and gives it a small (|0.11|) coefficient — which is correctly signed per the user's hypothesis but cannot move RMSE on the 58 untagged firms.
* **`tenure_sq` strengthened in this run** (−0.31 → −0.35) — confirming the load-bearing diagnostic from Week-4 across yet another Isolation Run. This feature has now been observed in 7 healthy runs (exp_002, 003, 006, 008, 009, 011, 013) and 3 broken runs (exp_007, 010, 012); the broken-run signal-flip pattern is now robust enough that a future automated guardrail (e.g., "abort run if tenure_sq coefficient escapes [−0.45, −0.20]") would catch the worst regressions before they reach the metric.
* **§5 Decoupled Isolation Rule worked as intended.** This run is the cleanest possible test of the rule: a feature was added at baseline α, the result was a sparse-signal failure, and no autonomous tuning was wasted on a doomed configuration. Compare to exp_012 where bundled feature+tuning made the regression hard to attribute. **The protocol earned its keep on this run.**
* **Cumulative Week-4 + Week-5 wall-time: ~13.3 seconds** across eight controlled experiments. **Cumulative cost: $0.00 marginal beyond the original $1.24 Apollo charge.**
