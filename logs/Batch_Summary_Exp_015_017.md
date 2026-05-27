# Batch Summary: Week-5 Automated Run (Exp 015 → Exp 017)
**Date:** 2026-05-08
**Baseline:** Exp_009 (Ridge α=1.0, 11 features, RMSE 1.3955, R² 0.5128) — `logs/Snapshot_model_Exp_009.py`
**Protocol:** Strict isolation — each run started from a `cp Snapshot_model_Exp_009.py model.py` revert, exactly one new feature added, α held at 1.0 per `program.md` §5 Decoupled Isolation Rule. No autonomous tuning exercised.
**Maintenance applied first:** `exp_014` flipped `keep → discard` in `logs/results.tsv`.

---

## 🏆 Succession Winner — Exp_016 (Sticky Vertical Moat)

**Lowest RMSE of the batch: 1.3079** (R² 0.5721) — also a **new project all-time best**, beating Exp_009 by −0.0876 RMSE (−6.3% relative). Promotion is *recommended pending the surprise audit below* — see "Why the Winner Needs Scrutiny" before committing.

---

## Results Matrix

| exp_id | Variable Added | RMSE | R² | Δ RMSE vs Exp_009 | Coef of New Feature | Tag Rate | Verdict |
|---|---|---|---|---|---|---|---|
| Exp_009 | (baseline reference) | **1.3955** | 0.5128 | — | — | — | baseline |
| **Exp_015** | `succession_red_flag` (binary; founder regex ∪ acquisition substrings, growth-press excluded) | 1.4690 | 0.4602 | **+0.0735 (+5.3%)** | −0.3127 (correctly signed) | 7/62 (11.3%) | **Signal Failure — sparsity** |
| **Exp_016** | `has_moat` (binary; regulated-vertical keywords) | **1.3079** | **0.5721** | **−0.0876 (−6.3%)** | **−0.5832 (sign opposite to hypothesis)** | **55/62 (88.7%)** | **Signal Success — new all-time best** ⚠️ see audit |
| **Exp_017** | `stagnation_ratio` (numeric; `(legacy+1)/(modern+1)`) | 1.5811 | 0.3746 | +0.1856 (+13.3%) | −0.2284 (sign opposite to thesis) | range [0.25, 2.00], mean 0.91 | **Signal Failure — feature redistribution** |

---

## Per-Run Diagnostic

### Exp_015 — Combined Succession Flag
- **Tag rate: 7/62 (11.3%)** — the union of (a) the 5 founder-presence regexes from Exp_013 and (b) the 4 acquisition substrings from Exp_014, with the financial/growth-press keywords from Exp_014 *excluded* (which had caused the World Synergy false positive).
- **Coefficient: −0.3127 standardized**, correctly signed per the user's "succession red flag → lower target quality" thesis.
- **Why it still failed RMSE:** at 11.3% tag rate the feature can affect 7 firms' predictions; the variance penalty of fitting a 12th coefficient at N=62 outweighs the bias reduction. This is the *third consecutive* sparse-binary-NLP-feature Signal Failure (after Exp_013 founder_led and Exp_014 institutionalization), and the densest of the three — but still below the ~16% threshold the cross-experiment pattern suggests is needed for RMSE-positive signal.
- **Stop-rule check:** PASSED. `tenure_sq` held at −0.33 (vs −0.31 baseline; mildly strengthened); `tenure` weakened from +0.66 to +0.63 — within noise. Bell-curve-over-tenure structure intact.
- **Failure Mode (program.md taxonomy):** **(1) Signal Failure — sparse-signal subtype.**

### Exp_016 — Sticky Vertical Moat ⚠️ Winner with caveats
- **Tag rate: 55/62 (88.7%)** — *extremely* dense, more than any prior feature in the project.
- **Coefficient: −0.5832 standardized** — strong, but **sign-flipped from the user's "moat firms have higher quality" hypothesis**. Empirically, tagged firms have *lower* mean Manual Score (7.48) than untagged (8.43), and Ridge fits the negative direction.
- **Per-keyword tag count audit** (which keywords drove the 55 tags):
  ```
   compliance: 49 firms  (79%)  ← dominant trigger
  manufacturing: 28 firms
        legal: 17 firms
        hipaa: 11 firms
          pci:  4 firms
     law firm:  2 firms
    regulated:  2 firms
       dental:  1 firm
  ```
  **`compliance` alone tags 79% of the dataset** — and it's the same `compliance` keyword already inside `RECURRING_KW` in the existing `recurring_kw` count feature.
- **The 7 *untagged* firms** (those with none of the listed keywords): Innovative Computers (9.0), Axia Technology Partners (6.5), Dymin (8.5), SMaRT Technology Services (8.5), Miken Technologies (9.0), Pinnacle Computer Services (10.0), One Click Inc (7.5). **Mean Manual Score: 8.43** — meaningfully above the tagged-firm mean of 7.48.
- **What `has_moat` is *actually* learning:** with 88.7% positive rate, the binary is more informative as `NOT has_moat` (the 7-firm minority). Ridge's −0.58 coefficient is essentially saying *"firms whose marketing copy lacks every one of these generic vertical/compliance terms tend to be standout high-scored MSPs"*. This is a real correlation in the 62-firm training set — but it's effectively fitting a small-N artifact: 6 of 7 untagged firms happen to be Manual-Score 7.5+, and Ridge gets to lower predictions for the other 55 firms accordingly.
- **Stop-rule check:** PASSED, but with notable redistribution. `tenure_sq` at −0.45 (strengthened from −0.31), `tenure` at +0.83 (strengthened from +0.66), `recurring_kw` at +0.59 (strengthened from +0.33), `log_revenue` at +0.80 (weakened from +0.94), `rev_per_emp` at −0.53 (weakened from −0.66). The strong has_moat coefficient is reshuffling weight across the entire feature set — *not* localized to one redistribution direction.
- **Risk assessment:** the RMSE win is real on this fold, but the mechanism is fragile. With only 7 untagged firms total, removing or correctly-classifying any one of them (e.g., adding "compliance" to Pinnacle's description) could swing the coefficient and the metric. **Recommend manual review before promoting to baseline.**

### Exp_017 — Stagnation Ratio
- **Range: [0.25, 2.00], mean 0.91** — most firms cluster near balanced (ratio ≈ 1); the maximum-stagnation firm has 2× more legacy keywords than modern.
- **Coefficient: −0.2284 standardized** — opposite sign to the search-fund thesis (which predicted positive: stagnant firms = better acquisition targets). Empirically, firms with higher legacy/modern ratios are predicted to have *lower* Manual Scores.
- **Why the sign-flip:** the existing `stagnation_kw` feature (raw count) stays positive at +0.55 — *strengthened* from baseline +0.41. The ratio captures something different: it penalizes firms that are *purely* legacy with *no* modern signals, which the labelers may have rated lower because they look unfixable rather than improvable. The "stable but stagnant" thesis works for the count (some legacy + some modern), not for pure legacy.
- **Why RMSE got worse despite a non-trivial coefficient:** the ratio reshuffles weight away from `tenure` (+0.66 → +0.57) and `tenure_sq` (−0.31 → −0.24) — a 24% reduction in the bell-curve helper, the largest weakening of the batch on this load-bearing feature. The ratio effectively duplicates information already in the count features (which is why count and ratio together don't add net signal) while adding noise from the smoothing constants.
- **Stop-rule check:** SOFT YELLOW FLAG. `tenure_sq` at −0.24 — still negative, but crosses the proposed −0.20 vigilance threshold from Exp_012's recommendations. Not a hard stop; bell-curve sign preserved.
- **Failure Mode:** **(1) Signal Failure — feature-redundancy subtype.** The ratio is collinear with the existing `stagnation_kw` and `modern_ai_kw` count features; Ridge cannot extract additional signal from the same underlying counts repackaged as a ratio.

---

## What This Tells Us — Cross-Run Synthesis

1. **Tag-rate pattern updated.** Three Week-5 NLP-binary experiments (Exp_013/014/015) all failed RMSE at tag rates of 4.8–11.3%. Exp_016's 88.7% rate succeeded — but via an *inverted* mechanism (the rare untagged firms are the signal). The pattern is now: sparse positive-fraction binaries fail; densely-positive binaries can succeed by encoding the rare-negative case as a price-premium. The threshold band 12–80% remains untested in Week 5.

2. **The exp_016 mechanism warrants targeted follow-up.** Two cheap experiments would clarify whether the lift is real or a small-N artifact:
   - **Drop the `compliance` keyword from MOAT_KW.** If RMSE drops back toward 1.40, the moat lift is essentially the *NOT-compliance* signal in disguise — and the claimed "vertical moat" interpretation is wrong.
   - **Try `is_premium_msp` as a binary 1-of-7 manual flag** for the same 7 untagged firms (or a regex that targets *their* descriptive language directly, not the negation of generic terms). If that reproduces the lift, the moat-vertical framing is incidental and the real signal is "atypical/differentiated marketing copy."

3. **`tenure_sq` remains the most reliable model-health diagnostic.** Three of three batch runs preserved its sign and substantial magnitude (Exp_015: −0.33, Exp_016: −0.45 *strengthened*, Exp_017: −0.24 yellow flag). It's now been observed in 12 of 12 controlled experiments, including 6 healthy and 6 broken runs — the strongest single diagnostic in the project.

4. **§5 Decoupled Isolation Rule is paying off in batch mode too.** All three runs were single-variable changes at fixed α=1.0. No GridSearchCV confounds; results are cleanly attributable to the feature alone. If Exp_016 is to be promoted, a separate follow-up Isolation Run (with α tuning ON, per §5) can determine whether α=1.0 is still optimal *given the new feature*.

---

## Failure Mode Summary (Per `program.md` 4-Category Taxonomy)

| Run | Signal Failure | Code Instability | Evaluation Leakage | Agent Misbehavior |
|---|---|---|---|---|
| Exp_015 | ✓ — sparse-signal subtype | ✗ | ✗ | ✗ |
| Exp_016 | n/a (Signal Success) | ✗ | ✗ — but mechanism is small-N-artifact-adjacent; recommend manual audit before promoting | ✗ |
| Exp_017 | ✓ — feature-redundancy subtype | ✗ | ✗ | ✗ |

---

## Snapshots & Artifacts

| Run | Snapshot | Bytes | Notes |
|---|---|---|---|
| Exp_015 | `logs/Snapshot_model_Exp_015.py` | ~10 KB | Combined founder + acquisition regex/keyword set |
| Exp_016 | `logs/Snapshot_model_Exp_016.py` | ~9 KB | MOAT_KW list (8 keywords) + has_moat helper |
| Exp_017 | `logs/Snapshot_model_Exp_017.py` | ~9 KB | LEGACY_KW + MODERN_KW + stagnation_ratio_score |

All three snapshots reproduce their respective `model.py` state at run completion (verified via byte-for-byte cp immediately after the Worker exited).

## Runtime & Budget (Batch Cumulative)
* **Total wall time across the 3 runs:** ~4.2 s warm cache (1.4 s × 3 runs, dominated by Ridge's 5-fold OOF + diagnostic refit). All within the per-firm budget (≤ 0.025 s / firm).
* **Marginal cost vs Exp_009:** **$0.00** — three pure feature-engineering changes, zero new data acquisition.
* **Cumulative project cost:** **$1.24** (Apollo firmographics, unchanged across all 17 experiments).
* **Cumulative project wall-time across all 17 controlled experiments:** ~17 s.

## Pending User Decisions

1. **Status calibration on the two regression rows.** Recommend flipping `exp_015` and `exp_017` from `keep → discard` in `logs/results.tsv` (parallel to the Exp_011/012/013/014 maintenance pattern). Flagging for confirmation; not auto-applied.
2. **Promotion decision on Exp_016.** RMSE 1.3079 is the lowest in project history, but the negative coefficient + 88% tag rate + dominance of generic `compliance` keyword warrants the audit recommended above before committing the promotion. Two paths:
   - **Promote now**: `cp logs/Snapshot_model_Exp_016.py model.py`, update Week-5 baseline, treat the audit as a Week-6 follow-up.
   - **Audit first**: drop `compliance` from MOAT_KW (or run the inverted is_premium_msp variant) before promoting; only promote if the lift survives.
3. **Current `model.py` state.** After Exp_017 + snapshot, `model.py` is in the Exp_017 (stagnation_ratio) state — *not* the Exp_009 baseline. Recommend `cp logs/Snapshot_model_Exp_009.py model.py` to restore the canonical baseline before any non-batch experiment is proposed (or `cp logs/Snapshot_model_Exp_016.py model.py` if the promotion path is chosen). Held for explicit direction.
