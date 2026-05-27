# Experiment Log: Auto-Private-Equity Search Engine
**Date:** 2026-05-07
**Experiment ID:** exp_012 (Week 5 Controlled Experiment Set — Isolation Run #2)
**System-assigned ID in `logs/results.tsv`:** `exp_012` — IDs aligned.
**Status:** keep (logged per run flag); substantively a **Signal Failure (severe)** — see Stop Rule + Failure Mode + Decision sections.

---

## 🛑 STOP RULE TRIGGERED — User-Defined Sentinel Fired

The user-defined Week-5 stop rule for this run was: *"If the interaction term causes other key features to flip signs or disappear (like tenure_sq), stop and flag it in the log."*

**Both conditions fired:**

| Feature | exp_009 baseline coef | **exp_012 coef** | Stop-rule trigger |
|---|---|---|---|
| `tenure_sq` | **−0.3127** | **+0.0442** | **SIGN FLIP** — negative bell-curve helper has become a near-zero positive; the inverted-U over tenure is broken. |
| `rev_per_emp` | **−0.6563** | **−0.0613** | **COLLAPSE** — magnitude reduced 11×; the Week-4 Signal Success feature is now near-noise. |
| `mgmt_depth` | +0.0637 | +0.0009 | zeroed (was already small; not classified as "key" but worth noting). |

**Action:** the run is logged for traceability, but **no further variants of this configuration are proposed without explicit user direction.** The recommended path is to revert to the exp_009 snapshot and rethink the interaction encoding (see What This Likely Tells Us — Section 1).

---

## Experiment: Tenure × Rev/Emp Interaction ("Stagnant Legacy")

### Configuration
* **Worker:** `model.py` — reset to exp_009 baseline via `cp logs/Snapshot_model_Exp_009.py model.py` (verified byte-identical), then two targeted edits: a new feature `stagnant_legacy = tenure × rev_per_emp`, and a `GridSearchCV` wrapper over the Ridge α.
* **Judge:** `eval/prepare.py` (FROZEN, SHA-256 `8f7aa10f25b1...` verified before run).
* **Single substantive change vs exp_009:** added `stagnant_legacy` feature (12 features total, up from 11). The α-tuning is a Week-5 agent-autonomy mechanism, not a separate hypothesis (see Agent Autonomy disclosure below).
* **Fixed variables:** the 11 features from exp_009 (kept exactly), StandardScaler, 0.5-grid rounding, `cross_val_predict(KFold(5, shuffle=True, random_state=42))` for the outer OOF metric.

### Hypothesis (User-Stated)
A firm that is both **old** (high `tenure`) **and inefficient** (low `rev_per_emp`) is the ultimate search-fund target. The product `stagnant_legacy = tenure × rev_per_emp` should be a low value for those firms (operationalizes the "stagnant + acquirable" thesis as a single conditional signal). Adding it as a 12th feature should let Ridge fit that conditional structure with one coefficient, improving RMSE on top of exp_009's 1.3955.

### Agent Autonomy Disclosure (Per Week-5 Autonomy Clause)
This run includes an **autonomous internal tuning step**, exercised once and disclosed in full per the user's instruction.

| Item | Value |
|---|---|
| Mechanism | `sklearn.model_selection.GridSearchCV` wrapping `Pipeline([StandardScaler, Ridge])` |
| Hyperparameter tuned | `ridge__alpha` |
| Grid searched | **`[0.1, 1.0, 10.0, 100.0]`** (per user-suggested range; not expanded) |
| Inner CV | `KFold(n_splits=5, shuffle=True, random_state=42)` (matches the outer CV used by `cross_val_predict`) |
| Scoring | `neg_root_mean_squared_error` |
| Outer-CV protection | `cross_val_predict(GridSearchCV(...), X, y, cv=outer_cv)` ⇒ each outer fold runs an independent inner-CV α selection (proper nested CV; no evaluation leakage). |
| **Chosen α (full-data inner CV)** | **`10.0`** |

**Per-alpha mean RMSE (full-data 5-fold inner CV, lower is better):**
```
alpha=  0.1: RMSE=2.2300
alpha=  1.0: RMSE=2.1414
alpha= 10.0: RMSE=2.0837  <-- chosen
alpha=100.0: RMSE=2.1379
```

The chosen α=10.0 is **10× stronger regularization than exp_009's fixed α=1.0** — and that α-shift, not the interaction term itself, is the proximate cause of the regression (see Causal Account).

### Result
| Metric | exp_009 baseline | **exp_012** | Δ vs baseline | vs exp_001 baseline |
|---|---|---|---|---|
| `val_rmse` | **1.3955** | **1.8743** | **+0.4788 (+34.3%)** | exp_001 was 1.8460 — **exp_012 is worse than the original hand-coded baseline** |
| `val_r2`   | **0.5128** | **0.1211** | **−0.3917** | exp_001 R² was 0.1474 — also worse than the hand-coded baseline |

This is the **largest single-experiment regression of any Week-4 or Week-5 run** by absolute RMSE shift, exceeding even exp_005's HGBR + rounding catastrophe (+0.6748 RMSE) when measured relative to the *current* baseline rather than the original exp_001.

### Diagnostic — Ridge Coefficients at α=10 (sorted by |coef|)
```
   sweet_spot_emp: +0.5593       stagnation_kw: +0.3542
      log_revenue: +0.3151        modern_ai_kw: -0.3122
     recurring_kw: +0.2982              tenure: +0.2779
    log_employees: -0.2164         rev_per_emp: -0.0613   ← collapsed
        tenure_sq: +0.0442    stagnant_legacy: +0.0279   ← new feature, near-noise
       mgmt_depth: +0.0009          in_midwest: +0.0000
```

**Reference (exp_009 at α=1.0, sorted by |coef|):**
```
log_revenue: +0.94, sweet_spot_emp: +0.78, tenure: +0.66, rev_per_emp: -0.66,
stagnation_kw: +0.41, modern_ai_kw: -0.39, log_employees: -0.37, recurring_kw: +0.33,
tenure_sq: -0.31, mgmt_depth: +0.06, in_midwest: 0.00
```

### Causal Account — Why a Reasonable Interaction Term Catastrophically Failed

**1. The interaction term itself contributes almost nothing (+0.028 coefficient).** Despite being constructed precisely to capture a thesis-aligned conditional signal, `stagnant_legacy` ranks 10th of 12 features by absolute coefficient. The "old AND inefficient" conditional doesn't earn its keep as an explicit product — Ridge gives it 4% the weight of `sweet_spot_emp` and ~half the weight of the smallest material feature in exp_009.

**2. The damage came from α tuning, not the interaction.** Adding `stagnant_legacy` introduces strong multicollinearity (it's a deterministic function of `tenure` and `rev_per_emp`, both already present). Ridge's standard response to multicollinearity is to demand stronger shrinkage to keep coefficient variance bounded. GridSearchCV correctly identified α=10 as the inner-CV optimum on this 12-feature input — but α=10 is so strong it shrinks **all** coefficients toward zero, including the structurally load-bearing ones from exp_009:
   * `rev_per_emp` (the Week-4 Signal Success feature): −0.66 → −0.06, a **10.7× magnitude reduction**.
   * `tenure_sq` (the bell-curve helper proven load-bearing across all four Week-4 ablations): −0.31 → **+0.04** — magnitude collapsed AND **sign flipped**.
   * `tenure`: +0.66 → +0.28 — 2.4× reduction.
   * `log_revenue`: +0.94 → +0.32 — 3× reduction.

**3. The α decision was statistically defensible but operationally wrong.** Inside GridSearchCV's inner CV, α=10 produces the lowest mean RMSE on the 12-feature problem (2.084 vs α=1.0's 2.141). That decision is locally optimal *for the augmented feature set*. But the augmented feature set itself is worse than the exp_009 set: the interaction adds collinearity and almost no new signal, so the optimal α for the new set delivers a worse global RMSE than the optimal α for the old set. **GridSearchCV optimized within a worse problem space.**

**4. The α-vs-feature-set coupling is the deeper lesson.** Running α tuning *and* a feature change in the same experiment confounds the cause: the regression could be (a) the interaction term itself, (b) the higher α, or (c) the interaction between (a) and (b). The diagnostic decomposes this:
   * If we kept α=1.0 and just added `stagnant_legacy`, would the model recover? Likely partially — the interaction term would get a higher coefficient (less shrinkage) but the multicollinearity would still inflate per-coefficient variance.
   * The fact that GridSearchCV moved to α=10 *given the new feature set* tells us the new feature set is harder to fit. Forcing α=1.0 would have produced lower bias but higher variance — possibly a smaller regression than +0.48, but not better than exp_009 either.

### Verdict on the Stated Hypothesis
| Sub-hypothesis | Prediction | Outcome |
|---|---|---|
| `tenure × rev_per_emp` captures the "stagnant + inefficient" thesis | Yes | **Falsified** — Ridge gives the new feature only +0.028 standardized weight |
| Adding the interaction reduces RMSE | Yes | **Falsified** — RMSE rose 34% to 1.8743 (worse than exp_001 hand-coded baseline) |
| Stronger regularization helps the augmented model | (autonomy hypothesis) | **Confirmed locally** (α=10 beats α=1 *within* the 12-feature problem) but **falsified globally** (the 12-feature problem at α=10 is worse than the 11-feature problem at α=1) |

### Failure Mode — per `program.md` §Logging Standards (4-Category Taxonomy)
* **1. Signal Failure (Information/Heuristic).** ✓ — applies. The proposed business-heuristic interaction (`tenure × rev_per_emp`) does not have predictive power as a multiplicative product, and the autonomy-driven α tuning compounded the failure by selecting heavier shrinkage that erased load-bearing structural coefficients.
* 2. Code Instability (Infrastructure). ✗ — not triggered. Worker exit 0, Judge exit 0, no `ConvergenceWarning`, SHA-256 lock held, scrape cache served all 62 entries, GridSearchCV converged on all 4 alphas.
* 3. Evaluation Leakage (Validity). ✗ — not triggered. Nested CV is properly applied (outer `cross_val_predict` over `GridSearchCV`); the Judge's metric and split are unchanged; no peeking at the validation set.
* 4. Agent Misbehavior (Control). ✗ — not triggered. The autonomy clause was exercised exactly once on the user-suggested grid `[0.1, 1.0, 10.0, 100.0]`, disclosed in full above. The interaction term is exactly one new feature (not "100 random features"). The 0.5-rounding rule is preserved. No frozen file was modified.

### Decision
* The Isolation Run is logged with `--keep` per the run instruction.
* **Substantively, exp_009 (Ridge α=1.0, 11 features) remains the operative best at RMSE 1.3955.** This experiment does not earn promotion.
* Per `program.md` §Week 5 Specific Plans: a Controlled Experiment must be reverted to the exp_009 snapshot unless explicitly promoted. **Recommend `cp logs/Snapshot_model_Exp_009.py model.py` before exp_013** and **flipping `exp_012` status from `keep` to `discard`** in `logs/results.tsv`. Both held for explicit user confirmation.
* Snapshot at `logs/Snapshot_model_Exp_012.py` (9477 bytes) preserves the GridSearchCV + interaction code for any future revisit (e.g., trying a polynomial-feature interaction with α held at 1.0 to disentangle the two effects).

### What This Likely Tells Us — for the Week-5 Set

1. **Multiplicative interactions of already-present features are a multicollinearity trap in Ridge.** Future interaction experiments should either (a) operate on *one new dimension* (e.g., scrape a new "founder age" signal, then interact with tenure), or (b) project the interaction into an orthogonal subspace via `PolynomialFeatures(interaction_only=True)` followed by Lasso so the L1 penalty can reject collinear duplicates. A raw product of two existing features in a Ridge pipeline almost always burns budget.

2. **Decouple feature changes from α tuning in single Isolation Runs.** The agent-autonomy clause for α tuning is valuable, but in this run it confounded the diagnostic. Recommended protocol: run the feature change at the *baseline α* first to isolate the feature effect; if RMSE improves, *then* run a follow-up Isolation Run that tunes α on the new feature set. Two clean runs beat one ambiguous one — especially when the stop rule fires.

3. **`tenure_sq` is the canary in this dataset.** Across all six Week-4 + Week-5 controlled experiments so far, this feature has been the most reliable diagnostic of model health: it's negative and substantial (−0.27 to −0.41) in every run that produces a competitive RMSE, and it weakens or sign-flips in every run that regresses. A future "model health check" rule could be: *if `tenure_sq` coefficient is in [−0.20, +0.20], flag the run as suspect even before checking the metric.*

4. **The autonomy clause needs a feedback channel.** When GridSearchCV selects an α that triggers the stop rule (because heavier shrinkage erased load-bearing features), the run still completes and logs. Recommend adding a guardrail: if the chosen α differs from the baseline α by more than 5×, surface it before the run completes so a human can decide whether to proceed. This is a Code-Instability-adjacent improvement to the experimental machinery, not a Signal Failure of the model itself.

### Human Feedback/Comments
*Logged 2026-05-07.* This is **Week 5 Isolation Run #2**, exercising the new agent-autonomy clause for the first time. The stop rule fired: `tenure_sq` flipped sign (−0.31 → +0.04) and `rev_per_emp` collapsed (−0.66 → −0.06), both due to GridSearchCV selecting α=10 on the augmented 12-feature input. The interaction term `stagnant_legacy` itself contributes ~zero (+0.028 standardized weight); the damage is the α-shift's collateral, not the feature's signal. The exp_009 baseline holds at RMSE 1.3955. Snapshot at `logs/Snapshot_model_Exp_012.py` preserves the GridSearchCV + interaction configuration. The autonomy disclosure (grid `[0.1, 1.0, 10.0, 100.0]`, chosen α=10, full per-alpha CV scores) is in the Configuration section above. Recommend revert + status flip; held for confirmation.

### Audit
* **Judge integrity:** SHA-256 `8f7aa10f25b1...` verified prior to Worker execution.
* **Output schema:** `results.tsv` written with header `Predicted Score\tCompany Name`, tab-delimited, 62 firms scored, all values on the 0.5 grid.
* **Scrape coverage:** 61 / 62 firms reachable from `logs/scrape_cache.json` (1 cached failure, unchanged).
* **Revert verification:** `model.py` was reset to `logs/Snapshot_model_Exp_009.py` via `cp` before any edits; `diff` confirmed byte-identical to baseline pre-edit.
* **Numeric sanity:** all 4 GridSearchCV alphas converged silently under Ridge's default solver; nested CV ran without errors.
* **Snapshot:** `logs/Snapshot_model_Exp_012.py` written immediately after run (9477 bytes; 1155 bytes larger than exp_009 snapshot due to the GridSearchCV import + wrapper + the expanded diagnostic block + the new feature line).
* **Code Instability classification:** none triggered.
