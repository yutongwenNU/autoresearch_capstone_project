# Experiment Log: Auto-Private-Equity Search Engine
**Date:** 2026-05-08
**Experiment ID:** exp_014 (Week 5 Controlled Experiment Set — Isolation Run #4)
**System-assigned ID in `logs/results.tsv`:** `exp_014` — IDs aligned.
**Status:** keep (logged per run flag); substantively a **Signal Failure (sparse-signal + outlier-leverage subtype)** — see Failure Mode + Decision sections.

## Experiment: Institutionalization Red-Flag Index

### Configuration
* **Worker:** `model.py` — reset to exp_009 baseline via `cp logs/Snapshot_model_Exp_009.py model.py` (verified byte-identical), then a single targeted edit adding one new binary feature.
* **Judge:** `eval/prepare.py` (FROZEN, SHA-256 `8f7aa10f25b1...` verified before run).
* **Single change vs exp_009 baseline:** added 12th feature `is_institutionalized` (binary: 1 if any of 10 substring keywords or 1 regex match `Short Description + Keywords` text, 0 otherwise).
* **Fixed variables (per `program.md` §5 Decoupled Isolation Rule):** Ridge **α=1.0** (no GridSearchCV — fixed at baseline α to isolate the feature signal first), StandardScaler, 0.5-grid rounding, the 11 features from exp_009. **No autonomous tuning was exercised on this run.**

### Hypothesis (User-Stated)
The model currently scores some firms high based on operational structural features (high revenue, mid-tenure, sweet-spot headcount) without recognizing that those firms are *already acquired, PE-backed, or aggressively scaling* — i.e., *post-search-fund-target* states. An `is_institutionalized` red flag should let Ridge **penalize** these firms with a negative coefficient, separating "institutionally backed but operationally similar" firms from genuine acquisition targets. Predicted sign: **negative**.

### Implementation — Signal Set

| Category | Substring patterns (lowercased match on `Short Description + Keywords`) |
|---|---|
| **Explicit acquisition / ownership** | `acquired by`, `subsidiary of`, `division of` |
| **Financial / institutional backing** | `private equity`, `venture capital`, `investment from`, `funding` |
| **Modernization / growth-press** | `rapidly growing`, `inc 5000`, `award-winning growth` |
| **Regex (case-insensitive)** | `\bpart of the\s+\w+\s+family\b` |

Tag rule: `is_institutionalized = 1` if **any** signal fires; else 0.

### Result
| Metric | exp_009 baseline | **exp_014** | Δ vs baseline |
|---|---|---|---|
| `val_rmse` | **1.3955** | **1.4645** | **+0.0690 (+4.9% relative)** |
| `val_r2`   | **0.5128** | **0.4634** | −0.0494 |
| `is_institutionalized` tag rate | — | **3 / 62 firms (4.8%)** | — |

RMSE regressed by 4.9% — comparable to the exp_013 founder_led Signal Failure (+4.3%) and the exp_008 Weighted MRR keyword failure (+5.1%). This is the second consecutive sparse-binary-feature Signal Failure of the Week-5 set.

### Diagnostic — Ridge Coefficients (standardized): exp_009 vs exp_014
| Feature | exp_009 | **exp_014** | Δ | Reading |
|---|---|---|---|---|
| `log_revenue` | +0.9399 | +0.8798 | −0.060 | mildly weakened |
| `sweet_spot_emp` | +0.7759 | +0.8028 | +0.027 | mildly strengthened |
| `rev_per_emp` | −0.6563 | −0.6476 | +0.009 | unchanged |
| **`tenure`** | **+0.6607** | **+0.4684** | **−0.192** | **meaningfully weakened** |
| `modern_ai_kw` | −0.3898 | −0.4115 | −0.022 | mildly stronger |
| `log_employees` | −0.3738 | −0.3749 | −0.001 | unchanged |
| **`is_institutionalized` (NEW)** | — | **−0.3599** | — | **strong negative coef** (3.3× larger \|·\| than exp_013's `founder_led` at −0.111) |
| `stagnation_kw` | +0.4050 | +0.2927 | −0.112 | meaningfully weakened |
| `recurring_kw` | +0.3323 | +0.2888 | −0.044 | mildly weakened |
| **`tenure_sq`** | **−0.3127** | **−0.1927** | **+0.120** | **weakened by ~38%** — soft yellow flag |
| `mgmt_depth` | +0.0426 | +0.1200 | +0.077 | tripled (still small) |
| `in_midwest` | 0.0000 | 0.0000 | 0 | unchanged |

**Stop-rule check (per Week-5 §6 5× Alpha Guardrail and exp_012's tenure_sq sentinel):**
* No α tuning was exercised this run — α=1.0 fixed per §5. **§6 not triggered.**
* `tenure_sq` weakened from −0.31 to −0.19 (38% magnitude reduction) but **did not flip sign** and did not "disappear" (still meaningful negative). This is a *soft yellow flag* (the proposed −0.20 threshold from exp_012's "what this likely tells us" section is now barely crossed) but not a hard stop. Logged for transparency.
* `tenure` weakened from +0.66 to +0.47 (29% reduction) — also a soft yellow flag, parallel signal to tenure_sq. The bell-curve-over-tenure structure is *attenuated* but not destroyed (sign preserved on both halves).

### Causal Account — Does the Feature Penalize Already-Acquired Firms?

The user's question: *"Does this feature help the model 'penalize' the successful-but-already-acquired firms that were previously getting high scores?"*

**The short answer: yes at the population level (correctly-signed strong coefficient), but no for RMSE — and the diagnostic of *which* firms got tagged reveals exactly why.**

#### Audit of the 3 Tagged Firms

| Tagged Firm | Manual Score | Reading |
|---|---|---|
| **Arnet Technologies** | **3.5** | low score — thesis works (institutional signal correctly correlates with low target quality) |
| **AlignLayerNine** | **4.0** | low score — thesis works |
| **World Synergy** | **8.5** | **HIGH score — thesis FAILS for this firm** (institutional signal fired but labeler still rated this as a strong target) |

**Mean Manual Score, tagged firms: 5.33** vs **mean Manual Score, untagged firms: 7.70** — a 2.4-point gap that aligns with the negative coefficient's direction. Two of three tagged firms (Arnet, AlignLayerNine) match the thesis cleanly; the third (World Synergy) is a false positive that the model is forced to penalize.

#### Why a Strong Coefficient Cost RMSE

**1. The −0.36 standardized coefficient is fit primarily by the two correctly-tagged low-Manual-Score firms.** Ridge sees Arnet (3.5) and AlignLayerNine (4.0) and learns "institutionalized → big drop in score." When the model is then asked to predict World Synergy (Manual Score 8.5, also tagged), it pushes the prediction down by approximately the same amount — ~1 point of standardized score — getting that firm badly wrong. **A single high-leverage false positive is enough to flip the feature from "useful penalty" to "net-negative noise" at this sparsity.**

**2. The high-coefficient × low-tag-rate combination is a generalization risk.** With only 3 tagged firms split across 5 CV folds, each held-out fold sees one tagged firm whose prediction is generated by a model that fit on the other 2 — a *2-sample basis* for the coefficient that goes onto the held-out prediction. This produces high-variance OOF predictions for the tagged firms specifically, which dominate the RMSE (because their squared errors are the largest in the dataset).

**3. Coefficient redistribution attenuated the structurally-important `tenure` and `tenure_sq`.** The is_institutionalized signal correlates with tenure (older firms are more likely to have been acquired or to have grown into the institutional bracket), and Ridge's L2 budget redistributed weight from the tenure pair toward the new feature: `tenure` lost 0.19 of standardized weight, `tenure_sq` lost 0.12. This is a **collateral cost** — the new feature didn't just add information, it *replaced* some of the bell-curve-over-tenure signal that exp_007 proved load-bearing. **Compare to exp_013 (founder_led at −0.11) where tenure_sq was *strengthened* (−0.31 → −0.35); the bigger coefficient on is_institutionalized comes with a bigger redistribution cost.**

#### Verdict on the Stated Hypothesis
| Sub-hypothesis | Prediction | Outcome |
|---|---|---|
| Institutionalization is a real penalty signal | Yes | **Confirmed at population level** — tagged firms avg Manual Score 5.33 vs untagged 7.70 |
| Adding `is_institutionalized` improves RMSE at α=1.0 | Yes | **Falsified** — RMSE rose 4.9% |
| The feature "separates acquired firms from genuine targets" | Yes | **Partially** — works for 2 of 3 tagged firms; fails for World Synergy (Manual Score 8.5) |
| The feature carries a stronger signal than `founder_led` | (implicit) | **Confirmed in coefficient terms** (\|−0.36\| ≈ 3.3× \|−0.11\|), but both fail the RMSE test for the same sparse-signal reason |

### Failure Mode — per `program.md` §Logging Standards (4-Category Taxonomy)
* **1. Signal Failure (Information/Heuristic).** ✓ — applies. **Subtype: sparse-signal + outlier-leverage.** The proposed business heuristic carries real predictive direction at the population level (correctly-signed coefficient, mean-Manual-Score gap of 2.4 points between tagged/untagged), but two structural problems prevent RMSE improvement: (a) only 3/62 firms tagged so the coefficient is fit on a 2-sample basis under CV, and (b) one of the 3 tagged firms (World Synergy, Manual Score 8.5) is a false positive that the strong coefficient mis-predicts heavily. This is a Signal Failure of *encoding precision*, not of *thesis*.
* 2. Code Instability (Infrastructure). ✗ — not triggered. Worker exit 0, Judge exit 0, no warnings, SHA-256 lock held, scrape cache served all 62 entries, regex compiled and ran cleanly.
* 3. Evaluation Leakage (Validity). ✗ — not triggered. Manual Score labels untouched, train/val split unchanged, Judge metric unchanged.
* 4. Agent Misbehavior (Control). ✗ — not triggered. Exactly one new feature added; α held at 1.0 per `program.md` §5; 0.5-rounding preserved; no frozen file modified; no autonomous tuning exercised.

### Decision
* The Isolation Run is logged with `--keep` per the run instruction.
* **Substantively, exp_009 (Ridge α=1.0, 11 features) remains the operative best at RMSE 1.3955.** This experiment does not earn promotion.
* Per `program.md` §Week 5 §2: **recommend `cp logs/Snapshot_model_Exp_009.py model.py` before exp_015** and **flipping `exp_014` status from `keep` to `discard`** in `logs/results.tsv`. Both held for explicit user confirmation.
* Per `program.md` §5 Decoupled Isolation Rule: since the feature did not deliver positive RMSE signal at α=1.0, **no follow-up GridSearchCV α-tuning run is warranted** — α tuning would not rescue a feature whose failure is structural (sparsity + a false positive), not regularization.
* Snapshot at `logs/Snapshot_model_Exp_014.py` (10049 bytes) preserves the keyword/regex set for any future revisit (e.g., a re-tuned signal that excludes `funding` and `rapidly growing` to reduce false positives).

### What This Likely Tells Us — for the Week-5 Set

1. **The Week-5 NLP-feature pattern is now a clear pattern: `(positive thesis direction) ∧ (sparse tag rate) ⇒ Signal Failure`.** Both exp_013 (`founder_led`, 4/62 tagged, coef −0.11) and exp_014 (`is_institutionalized`, 3/62 tagged, coef −0.36) confirm the thesis at the coefficient level but fail the RMSE test. The structural constraint is **tag rate**, not heuristic quality. **Future single-binary-NLP-feature experiments should target ≥10 tagged firms (~16% rate) before being considered RMSE-competitive at N=62.**
2. **A combined-NLP feature is the productive next step.** `union(founder_led, is_institutionalized)` would tag 4+3 = 6–7 firms (no overlap audited yet, but the keyword sets are disjoint). A more general "thesis-flag" feature aggregating multiple low-density signals into a single denser column might pass the sparsity threshold above. This is a non-trivial design choice (which heuristics to merge) but a single new feature, so still §5-compliant.
3. **Inspect the false-positive (World Synergy, MS 8.5) before any re-test of `is_institutionalized`.** Three options:
   * **Refine the keyword set:** drop `funding` (likely over-broad — many MSPs offer "funding" services), `rapidly growing` (a marketing platitude), and `award-winning growth`. Keep only the explicit acquisition signals.
   * **Add a confirming requirement:** require *two* signals from different categories (one explicit + one financial), not just one.
   * **Inspect World Synergy's actual triggering keyword** to understand whether it's a true institutional firm rated highly (in which case the labels disagree with the thesis) or a false positive on noise text.
4. **The `tenure` + `tenure_sq` redistribution cost is a recurring pattern with high-coefficient new features.** Exp_012 (interaction term, severe regression) and exp_014 (institutionalization, moderate regression) both show that any new feature with |coef| > ~0.30 in this Ridge pipeline tends to draw weight from the bell-curve-over-tenure encoding. A future automated guardrail could be: *if a new feature's |coef| exceeds the median |coef| of existing features, check `tenure_sq` for ≥30% magnitude reduction and flag as a soft warning.* Not a hard stop, but a reliable diagnostic of "this feature is rebudgeting the model."

### Human Feedback/Comments
*Logged 2026-05-08.* This is **Week 5 Isolation Run #4**, the second consecutive sparse-NLP-binary-feature experiment. The institutionalization signal validates the user's hypothesis at the population level (mean Manual Score gap of 2.4 points between the 3 tagged firms and the 59 untagged firms), and Ridge gives it a strong negative coefficient (−0.36, 3.3× larger than founder_led's). Yet RMSE regressed by 4.9% because (a) only 3 firms tagged limits the feature's effect to 4.8% of the dataset, (b) one of the 3 tagged firms (World Synergy, Manual Score 8.5) is a false positive whose prediction is now badly mis-pushed downward, and (c) the strong coefficient redistributed weight away from the load-bearing `tenure` (+0.66 → +0.47) and `tenure_sq` (−0.31 → −0.19) — a soft yellow flag on the latter (still negative, no sign flip, but 38% magnitude reduction crosses the proposed −0.20 vigilance threshold). The exp_009 baseline holds at RMSE 1.3955. Snapshot at `logs/Snapshot_model_Exp_014.py` preserves the keyword/regex set; the most productive Week-5 next step would be a *combined* thesis-flag feature merging the institutionalization and founder-led signals to push tag density above the ~16% sparsity threshold suggested by the empirical pattern.

### Audit
* **Judge integrity:** SHA-256 `8f7aa10f25b1...` verified prior to Worker execution.
* **Output schema:** `results.tsv` written with header `Predicted Score\tCompany Name`, tab-delimited, 62 firms scored, all values on the 0.5 grid.
* **Scrape coverage:** 61 / 62 firms reachable from `logs/scrape_cache.json` (1 cached failure, unchanged).
* **Revert verification:** `model.py` was reset to `logs/Snapshot_model_Exp_009.py` via `cp` before any edits; `diff` confirmed byte-identical to baseline pre-edit.
* **Numeric sanity:** the 1 regex compiled successfully; Ridge converged silently; no NaN/inf in the new column.
* **`is_institutionalized` tag distribution:**
  * Arnet Technologies (Manual Score 3.5)
  * AlignLayerNine (Manual Score 4.0)
  * World Synergy (Manual Score 8.5) ← false positive
  * Tagged-mean Manual Score: 5.33; untagged-mean: 7.70 (2.4-point gap, validates the thesis at population level)
* **Soft yellow flags raised** (no hard stop):
  * `tenure_sq`: −0.3127 → −0.1927 (38% magnitude reduction; below the proposed −0.20 vigilance threshold from exp_012)
  * `tenure`: +0.6607 → +0.4684 (29% reduction)
  * Neither feature flipped sign or disappeared; the bell-curve-over-tenure structure is attenuated but intact.
* **Snapshot:** `logs/Snapshot_model_Exp_014.py` written immediately after run (10049 bytes; 1727 bytes larger than exp_009 snapshot due to the keyword/regex constants, helper function, featurize wiring, and tag-count print line).
* **Code Instability classification:** none triggered.
