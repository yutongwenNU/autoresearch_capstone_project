# Batch Summary: Week-5 Structural-Facts Run (Exp 019 → Exp 020)
**Date:** 2026-05-08
**Baseline:** Exp_009 (Ridge α=1.0, 11 features, RMSE **1.3955**, R² 0.5128) — `logs/Snapshot_model_Exp_009.py`
**Protocol:** Strict isolation — each run started from a `cp Snapshot_model_Exp_009.py model.py` revert; exactly one new feature added; α held at 1.0 per `program.md` §5 Decoupled Isolation Rule.
**Maintenance applied first:** `exp_016` and `exp_018` flipped `keep → discard` in `logs/results.tsv` (joining `exp_015` and `exp_017`, which were already at `discard`). All four prior Week-5 entries are now correctly marked.

---

## 🏁 Verdict — No New Winner

**Neither run beat the baseline.** Exp_009 (RMSE 1.3955) remains the operative project best. Both Exp_019 and Exp_020 are Signal Failures, but at meaningfully different magnitudes and for *different* reasons.

| Run | Variable | RMSE | R² | Δ vs Exp_009 | Tag Rate | Coef | Verdict |
|---|---|---|---|---|---|---|---|
| Exp_009 | (baseline reference) | **1.3955** | 0.5128 | — | — | — | baseline |
| **Exp_019** | `is_hub_proximate` (binary; metro hubs) | 1.4189 | 0.4964 | **+0.0234 (+1.7%)** | **30/62 (48.4%)** | **+0.2203** (correctly signed) | **Signal Failure — mild** |
| **Exp_020** | `ownership_red_flag` (binary; explicit acquisition phrases only) | 1.4645 | 0.4634 | **+0.0690 (+4.9%)** | **3/62 (4.8%)** | −0.3599 | **Signal Failure — identical to Exp_014** |

---

## Per-Run Diagnostic

### Exp_019 — Logistical Hub Proximity (Structural)
* **Tag rate: 30/62 (48.4%)** — densest single-binary tag rate of any Week-5 feature outside the `compliance`-driven Exp_016 (88.7%). The user-supplied 3 hubs (Chicago, Milwaukee, St. Louis) plus the autonomous additions (Detroit/SE Michigan, Twin Cities, Columbus, Cleveland, Cincinnati, Indianapolis, Kansas City corridors) cleanly captured ~half the cohort.
* **Hub list used (autonomy disclosure):** the 6 user-supplied Chicago suburbs, 4 Milwaukee, 4 St. Louis (with St./Saint variants for join robustness), plus autonomously-added: Detroit corridor (Livonia, Troy, Southfield, Ann Arbor, etc.), Twin Cities corridor (Minneapolis, St. Paul, Eden Prairie, etc.), Columbus, Cleveland, Cincinnati, Indianapolis, and Kansas City corridors. Total ~75 city tokens; full list in `logs/Snapshot_model_Exp_019.py` line 51 onward.
* **Coefficient: +0.2203 standardized — correctly signed per the user's hypothesis** (hub-proximate firms predicted higher than non-hub firms; consistent with "client density / margin / less windshield time" thesis). This is the *cleanest signed-correctly result* of any Week-5 run.
* **Why RMSE still got slightly worse:** at +0.22 the coefficient is the second-smallest non-zero feature in the model; the variance penalty of fitting a 12th coefficient at N=62 nudges RMSE slightly upward. The hub thesis appears to be a *real but small* effect that's already partially absorbed by `log_revenue`, `sweet_spot_emp`, and other firmographic signals — the 30 hub firms aren't systematically different *enough* from the 32 non-hub firms after controlling for the existing features.
* **Stop-rule check:** PASSED. `tenure_sq` held at −0.29 (vs −0.31 baseline; mildly weakened, well within healthy range). `tenure` at +0.70 (vs +0.66 baseline; mildly strengthened). Bell-curve-over-tenure structure intact.
* **Failure Mode (program.md taxonomy):** **(1) Signal Failure — mild redundancy.** The structural hub signal correlates with revenue and headcount in the existing features; on its own it adds insufficient orthogonal information to overcome the variance penalty.

### Exp_020 — Refined Ownership Red-Flag (High-Precision Kill Switch)
* **Tag rate: 3/62 (4.8%) — identical to Exp_014.** The 3 tagged firms are also identical: **Arnet Technologies (Manual Score 3.5), AlignLayerNine (4.0), and World Synergy (8.5)** — including the same false positive on World Synergy that was the original concern raised in Exp_014's audit.
* **Coefficient: −0.3599** — also bit-for-bit identical to Exp_014.
* **RMSE: 1.4645** — identical to Exp_014 to 4 decimal places. **The "refined" feature is mathematically equivalent to Exp_014's `is_institutionalized`** in this dataset.
* **Why the refinement didn't help:** the user's stated motivation for excluding `funding`, `private equity`, `venture capital`, `investment from`, `rapidly growing`, `inc 5000`, and `award-winning growth` was to avoid the World Synergy false positive. But empirically, *none of those keywords were firing* in the 62-firm training set — the 3 firms tagged by Exp_014 were already being tagged exclusively via the explicit acquisition phrases (`acquired by`, `subsidiary of`, `division of`, or `part of the * family`). Removing the unused keywords changes nothing. **World Synergy is matching on an explicit acquisition phrase, not on growth/inc-5000 language** — the false positive is structurally inside the user's allowed keyword set.
* **Stop-rule check:** SOFT YELLOW FLAG, identical to Exp_014. `tenure_sq` at −0.19 (vs −0.31 baseline; 38% magnitude reduction; below proposed −0.20 vigilance threshold). `tenure` at +0.47 (vs +0.66; 29% reduction). Strong-coefficient new feature → tenure-pair redistribution, same pattern as Exp_014.
* **Failure Mode:** **(1) Signal Failure — feature-equivalence subtype.** The refinement renamed the feature without changing its empirical content; the same diagnostic that disqualified Exp_014 disqualifies Exp_020.

---

## Why the Failures Are Different — and Why That Matters

The two runs fail in *opposite ends* of the feature-space:

| Axis | Exp_019 (Hub Proximity) | Exp_020 (Ownership Red-Flag) |
|---|---|---|
| Tag rate | 48.4% (dense, mid-spectrum) | 4.8% (sparse) |
| Coefficient sign | +0.22 (correctly signed per hypothesis) | −0.36 (correctly signed per hypothesis) |
| RMSE delta | +0.0234 (mild) | +0.0690 (moderate) |
| Mechanism of failure | Real-but-redundant signal already in existing features | Real-but-too-sparse signal + a structural false positive |
| `tenure_sq` impact | Stable (−0.29) | Soft yellow flag (−0.19) |
| Promising for re-test? | **Yes — possible coefficient-strengthening at lower α** | **No — World Synergy is the bottleneck, not the keyword set** |

**Exp_019 is the more salvageable of the two.** A correctly-signed +0.22 coefficient on a 48% tag-rate feature is *not* a small-N artifact pattern — it's a real structural signal that's marginally redundant with the existing firmographics. A reasonable Week-6 follow-up (per Protocol 5 → 4 escalation) is: hold the hub feature ON and run GridSearchCV α ∈ {0.1, 0.3, 1.0, 3.0, 10.0}. If a lower α delivers RMSE < 1.3955 with the hub feature, the structural-hub thesis would be vindicated.

**Exp_020 is structurally dead.** Refining the keyword list cannot fix the World Synergy false positive because that firm is matching on phrases the user wants to keep. Two non-keyword paths for a future ownership-flag experiment:
1. **Manual exclusion list:** explicitly remove specific company names (e.g., World Synergy) from the tag set after the regex match. Less elegant, but it would isolate the 2 valid red flags (Arnet, AlignLayerNine) without the false positive.
2. **Source the signal from a non-description field:** Apollo's `Subsidiary of (Organization ID)` column is a structured field that, if non-empty, is a clean indicator. Using it directly avoids the noise of marketing copy entirely.

---

## Cumulative Diagnostic — Week-4 + Week-5 Pattern

| Pattern | Observation Count | Implication |
|---|---|---|
| `tenure_sq` is structurally load-bearing | 14/14 controlled experiments | Coefficient stays in [−0.27, −0.45] in healthy runs; weakens or sign-flips in regressions. **The most reliable single diagnostic in the project.** |
| `mgmt_depth` is dispensable | 14/14 | Coefficient stays in [+0.02, +0.18]; never costly to add or remove. |
| Sparse binary features (≤ 10% tag) regress RMSE | 4/4 (Exp_013, 014, 015, 020) | Tag rate × effect size must overcome the variance of fitting a coefficient at N=62. Threshold ≈ 16% based on cross-experiment fit. |
| Dense binary features can succeed via inverse mechanism | 1/1 (Exp_016 — but later disqualified by audit as artifact) | Watch for sign-opposite-to-hypothesis: the rare-class minority is doing the predictive work. |
| Mid-density correctly-signed features produce mild regressions | 1/1 (Exp_019) | Real but redundant. Only hope is α tuning. |
| Multiplicative interactions of existing features create multicollinearity traps | 2/2 (Exp_004, Exp_012) | Codified in §6 (5× α guardrail). |
| Promotion candidates need ablation testing before commitment | 1/1 (Exp_016 audit caught the artifact) | Codified inside Exp_018 log; recommend formal `program.md` §7. |

---

## Failure Mode Summary (Per `program.md` 4-Category Taxonomy)
| Run | Signal Failure | Code Instability | Evaluation Leakage | Agent Misbehavior |
|---|---|---|---|---|
| Exp_019 | ✓ — mild redundancy subtype | ✗ | ✗ | ✗ |
| Exp_020 | ✓ — feature-equivalence subtype (duplicate of Exp_014) | ✗ | ✗ | ✗ — autonomy clause exercised only on the hub-list extension in Exp_019, fully disclosed |

---

## Snapshots & Artifacts
| Run | Snapshot | Status |
|---|---|---|
| Exp_019 | `logs/Snapshot_model_Exp_019.py` | Hub-list configuration with autonomous extensions |
| Exp_020 | `logs/Snapshot_model_Exp_020.py` | High-precision ownership keyword set |

## Runtime & Budget (Batch Cumulative)
* **Total wall time across the 2 runs:** ~2.6 s warm cache. Under per-firm budget on every run.
* **Marginal cost vs Exp_009:** **$0.00** — both runs are pure feature-engineering changes from already-loaded structured/text fields.
* **Cumulative project cost:** **$1.24** (Apollo, unchanged across all 20 experiments).
* **Cumulative project wall-time across 20 controlled experiments:** ~25 s.

## Pending User Decisions
1. **Status calibration on the two regression rows.** Recommend flipping `exp_019` and `exp_020` from `keep → discard` in `logs/results.tsv`. Held for confirmation; not auto-applied.
2. **Salvage path for Exp_019.** The hub thesis is the only Week-5 result with a correctly-signed coefficient and a non-pathological tag distribution. If you want to give it one more shot before declaring Week-5 closed, the recommended follow-up is *Decoupled Isolation §5 → §4 escalation*: a single GridSearchCV run over α ∈ {0.1, 0.3, 1.0, 3.0, 10.0} *with* `is_hub_proximate` included. Only worth doing if the §6 5× Alpha Guardrail can be respected.
3. **Current `model.py` state.** After Exp_020 + snapshot, `model.py` is in the Exp_020 (`ownership_red_flag`) state — *not* the Exp_009 baseline. Recommend `cp logs/Snapshot_model_Exp_009.py model.py` before any non-batch experiment is proposed. Held for explicit direction.
