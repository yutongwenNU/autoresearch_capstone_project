# Experiment Log: Auto-Private-Equity Search Engine
**Date:** 2026-05-08
**Experiment ID:** exp_013 (Week 5 Controlled Experiment Set — Isolation Run #3)
**System-assigned ID in `logs/results.tsv`:** `exp_013` — IDs aligned.
**Status:** keep (logged per run flag); substantively a **Signal Failure (sparse-signal subtype)** — see Failure Mode + Decision sections.

## Experiment: NLP Founder-Led Detection ("Succession Gap" Nuance)

### Configuration
* **Worker:** `model.py` — reset to exp_009 baseline via `cp logs/Snapshot_model_Exp_009.py model.py` (verified byte-identical), then a single targeted edit adding one new binary feature derived from regex matching.
* **Judge:** `eval/prepare.py` (FROZEN, SHA-256 `8f7aa10f25b1...` verified before run).
* **Single change vs exp_009 baseline:** added 12th feature `founder_led` (binary: 1 if any of five regex patterns match the firm's `Short Description + Keywords` text, 0 otherwise).
* **Fixed variables (per `program.md` §5 Decoupled Isolation Rule):** Ridge **α=1.0** (no GridSearchCV — fixed at baseline α to isolate the feature signal first), StandardScaler, 0.5-grid rounding, the 11 features from exp_009, `cross_val_predict(KFold(5, shuffle=True, random_state=42))`. **No autonomous tuning was exercised on this run** per the Decoupled Isolation Rule.

### Hypothesis (User-Stated)
A 25-year-old firm where the original founder is still mentioned as a current leader is a **higher-risk succession play** than one where a founder is no longer mentioned. The thesis predicts a **negative** coefficient on `founder_led` (founder-still-present → succession risk → lower Manual Score / lower acquisition-target quality). This is the first NLP-extraction feature in the project, operationalizing what was previously only a roadmap entry (per `program.md` §3 Future Research Roadmap).

### Implementation — Five Regex Patterns
Operating on **case-preserved** text from `Short Description + Keywords` (the existing keyword-count features use a separate lowercased text bundle; this feature is computed independently to exploit capital letters as a name signal):

| # | Pattern | Flags | Matches |
|---|---|---|---|
| 1 | `\bfounder\b.{0,50}\b(?:ceo|president|leadership|cto|coo|cfo)\b` | IGNORECASE | "founder" within 50 chars of a leadership-role token (forward direction) |
| 2 | `\b(?:ceo|president|leadership|cto|coo|cfo)\b.{0,50}\bfounder\b` | IGNORECASE | Same, reverse direction |
| 3 | `\bfounder\s*(?:and|&|/)?\s*(?:ceo|president|owner)\b` | IGNORECASE | "Founder and CEO" / "Founder & CEO" / "Founder/CEO" / "Founder CEO" / "Founder Owner" phrasings |
| 4 | `\bfounders?\s+[A-Z][a-z]+` | (case-sensitive) | "founder Matthew" or "founders Matt" — capitalized first letter as a name signal |
| 5 | `\bfounded\s+(?:in\s+\d{4}\s+)?by\s+[A-Z][a-z]+` | (case-sensitive) | "founded by John" or "founded in 2006 by Matt Kahle" |

A firm is tagged `founder_led=1` if **any** pattern matches; otherwise 0.

### Result
| Metric | exp_009 baseline | **exp_013** | Δ vs baseline |
|---|---|---|---|
| `val_rmse` | **1.3955** | **1.4555** | **+0.0600 (+4.3% relative)** |
| `val_r2`   | **0.5128** | **0.4700** | −0.0428 |
| `founder_led` tag rate | — | **4 / 62 firms (6.5%)** | — |

RMSE regressed by 4.3%. Magnitude is comparable to exp_008 (Weighted MRR keywords, +5.1%) and meaningfully smaller than exp_010 (Gaussian sweet-spot, +11.3%) or exp_012 (Tenure×Rev/Emp, +34.3%) — this is the *least severe* Week-5 Signal Failure to date, but a Signal Failure nonetheless.

### Diagnostic — Ridge Coefficients (standardized): exp_009 vs exp_013
| Feature | exp_009 | **exp_013** | Δ | Reading |
|---|---|---|---|---|
| `log_revenue` | +0.9399 | +0.9389 | −0.001 | unchanged |
| `sweet_spot_emp` | +0.7759 | +0.7657 | −0.010 | unchanged |
| `tenure` | +0.6607 | +0.6981 | +0.037 | mildly strengthened |
| `rev_per_emp` | −0.6563 | −0.6599 | −0.004 | unchanged |
| `stagnation_kw` | +0.4050 | +0.4108 | +0.006 | unchanged |
| `modern_ai_kw` | −0.3898 | −0.3944 | −0.005 | unchanged |
| `log_employees` | −0.3738 | −0.3684 | +0.005 | unchanged |
| **`tenure_sq`** | **−0.3127** | **−0.3532** | **−0.041** | **strengthened** (no stop-rule trigger) |
| `recurring_kw` | +0.3323 | +0.3374 | +0.005 | unchanged |
| **`founder_led` (NEW)** | — | **−0.1110** | — | **correctly signed per thesis** (founder-led → lower score) |
| `mgmt_depth` | +0.0426 | +0.0827 | +0.040 | doubled (still small) |
| `in_midwest` | 0.0000 | 0.0000 | 0 | unchanged |

**Stop-rule check: PASSED.** Unlike exp_012, the load-bearing features survived intact. `tenure_sq` is *more* negative (−0.31 → −0.35), `rev_per_emp` is unchanged (−0.66), and the bell-curve-over-tenure structure is preserved. The model is healthy; the new feature simply doesn't help.

### Causal Account — Why a Correctly-Signed Coefficient Failed to Improve RMSE

The user's question: *"Does the presence of a founder act as the 'Nuanced Signal' we've been looking for to separate seemingly identical MSPs?"*

**The short answer: yes in direction, no in magnitude — and the magnitude problem is decisive at N=62.** Three threads explain why a correctly-signed +0.11 coefficient nonetheless cost +0.060 RMSE.

**1. The signal is too sparse.** Only **4 of 62 firms (6.5%)** were tagged `founder_led=1`. The other 58 firms see *no change* in their feature vector vs exp_009 — their predictions can only shift by Ridge's redistribution of weight onto adjacent features (which the diagnostic shows is minimal). For the model to recoup the ~0.060 RMSE penalty added by the new feature's coefficient variance, the 4 tagged firms would need to be substantially mis-predicted in exp_009 *in a direction the founder_led signal corrects*. They are not — Ridge's `cross_val_predict` already places them at sensible mid-to-high scores driven by their tenure and revenue features. **The "nuanced signal" can only differentiate the 4 firms it touches; the other 58 dilute it.**

**2. The thesis is conditional, but the encoding is unconditional.** The user's heuristic specifies the high-risk case as *"a 25-year-old firm where the original founder is still mentioned"* — i.e., **founder-led × old-firm**. The current feature flags founder presence regardless of tenure. Of the 4 tagged firms, at least one (Real IT Solutions, Inc., tenure ≈ 20 years, "founded in 2006 by Matt Kahle and Adam Peterson," Manual Score = 9.5) is a *young-ish, high-quality* firm that is founder-led but *not* a succession-risk play. Penalizing it via a flat −0.11 founder_led coefficient pushes its prediction in the wrong direction. The thesis is right; the encoding flattens it.

**3. The "right" encoding for this thesis is the same trap exp_012 just fell into.** A natural fix is `founder_led × tenure` — but exp_012 demonstrated that multiplicative interactions of already-present features create multicollinearity in this Ridge pipeline (codified in `program.md` §6 as the 5× α guardrail). Possible safer paths:
   * **Threshold-and-multiply:** `founder_led × (tenure > 25)` as a binary AND signal — single new column, less collinear because both inputs are binary.
   * **Targeted scraping:** parse the homepage's About/Team page directly for *current-year* founder-leadership claims (vs the description's possibly-historical "founded by" language). This separates "historically founded by" from "still led by today" — a meaningful distinction the current regex cannot make.
   * **Lasso on the augmented set:** if a future experiment retries `founder_led × tenure`, wrapping in Lasso would let L1 prune the collinear duplicates rather than redistribute weight as Ridge would.

**4. Even the regex itself may be missing matches.** The 5 patterns are conservative: they require either close proximity to a leadership token or a Capitalized name. Firms whose `Short Description` says "the company is led by its founders" (no name) or "owned and operated by [first name only]" might not match. Tag rate could plausibly be 10–15% with a more permissive regex; whether that improves RMSE depends on whether the additional matches are *real* founder-led firms or false positives from generic phrasing.

### Verdict on the Stated Hypothesis
| Sub-hypothesis | Prediction | Outcome |
|---|---|---|
| Founder presence is a real signal of succession risk | Yes | **Coefficient sign confirms** (−0.11, negative as predicted) |
| The signal differentiates "seemingly identical MSPs" | Yes | **Partially** — at N=62 with 4 tagged firms, the differentiation is too sparse to move RMSE |
| Adding `founder_led` improves RMSE at α=1.0 | Yes | **Falsified** — RMSE rose 4.3% |

### Failure Mode — per `program.md` §Logging Standards (4-Category Taxonomy)
* **1. Signal Failure (Information/Heuristic).** ✓ — applies. **Subtype: sparse-signal.** The proposed business heuristic carries real predictive direction (correctly-signed coefficient), but the feature operationalization tags too few firms (4/62) for the signal to overcome the variance of an additional fitted coefficient at N=62. This is a Signal Failure of *encoding*, not of *thesis*.
* 2. Code Instability (Infrastructure). ✗ — not triggered. Worker exit 0, Judge exit 0, no warnings, SHA-256 lock held, scrape cache served all 62 entries, regex compilation succeeded.
* 3. Evaluation Leakage (Validity). ✗ — not triggered. Manual Score labels untouched, train/val split unchanged, Judge metric unchanged. The regex operates on `Short Description` and `Keywords` columns which were already part of the training feature space in exp_002 onward (via the `text` keyword bundle).
* 4. Agent Misbehavior (Control). ✗ — not triggered. Exactly one new feature added; α held at 1.0 per `program.md` §5 (Decoupled Isolation Rule); 0.5-rounding preserved; no frozen file modified; no autonomous tuning exercised. **This run is the cleanest possible test of the Decoupled Isolation Rule's intent: see whether the feature delivers signal at baseline α before considering tuning.**

### Decision
* The Isolation Run is logged with `--keep` per the run instruction.
* **Substantively, exp_009 (Ridge α=1.0, 11 features) remains the operative best at RMSE 1.3955.** This experiment does not earn promotion.
* Per `program.md` §Week 5 §2: a Controlled Experiment must be reverted to the exp_009 snapshot unless explicitly promoted. **Recommend `cp logs/Snapshot_model_Exp_009.py model.py` before exp_014** and **flipping `exp_013` status from `keep` to `discard`** in `logs/results.tsv`. Both held for explicit user confirmation.
* **`program.md` §5 Decoupled Isolation Rule was correctly applied.** Per the rule, since the feature did not demonstrate a positive signal at baseline α, **no follow-up GridSearchCV run is warranted** — α tuning would not rescue a feature with insufficient signal density. The rule prevented exp_012's α-tuning trap from recurring.
* Snapshot at `logs/Snapshot_model_Exp_013.py` (10004 bytes) preserves the regex patterns + feature wiring for future revisits.

### What This Likely Tells Us — for the Week-5 Set

1. **The "nuance" thesis is correct but needs higher tag density to surface.** Rather than chasing a multiplicative interaction (which exp_012 showed is a multicollinearity trap in Ridge), the productive next step is **denser scraping** of *current* founder-leadership signals — specifically the team/leadership pages already cached in `logs/scrape_cache.json` (used by `mgmt_depth`), but parsed for founder-name + current-role pairings instead of role-title counts. The data is already on disk; only the parser changes.
2. **Sparse binary features at N=62 need a "cost-benefit" check.** A useful pre-flight test before a binary-feature experiment: compute (tag_rate × |expected_effect_size|) and compare to the baseline RMSE noise band (~0.003). For founder_led at 4/62 × ~1 score-point effect ≈ 0.065 — barely detectable above noise even if the effect is fully real. Future binary features should aim for 15–30% tag rate or have an effect size ≥ 1.5 score-points to justify inclusion.
3. **The Decoupled Isolation Rule is delivering on its purpose.** This run completed cleanly, the diagnostic is interpretable, and we have a clean answer: founder_led-as-currently-encoded does not help at α=1.0. No confounding from autonomous tuning. Compare to exp_012 where the α-feature confound made it impossible to attribute the regression cleanly. **Recommend §5 stays in `program.md` as a non-negotiable for feature-addition runs.**
4. **The `Short Description` text is the right *source* but possibly the wrong *granularity*.** The patterns above match historical phrasings (e.g., "founded in 2006 by") that don't actually answer the user's heuristic question (*"is the founder still in the company today?"*). A future NLP experiment could distinguish "historical founder mention" from "current founder mention" using verb tense or temporal markers — though that's a much larger NLP investment. Cheap intermediate: parse the scraped homepage text (case-preserved) for founder-name occurrences, not just the description.

### Human Feedback/Comments
*Logged 2026-05-08.* This is **Week 5 Isolation Run #3**, the first feature-only Isolation Run since `program.md` §5 Decoupled Isolation Rule was codified. The run executed cleanly under the new protocol: feature added at fixed α=1.0, no autonomous tuning, single-variable change against exp_009. Result is a sparse-signal Signal Failure — the founder-led coefficient is correctly signed (−0.11, matching the user's "succession risk" thesis) but the 4/62 tag rate is too low to overcome the variance penalty at N=62. The exp_009 baseline holds at RMSE 1.3955. Snapshot at `logs/Snapshot_model_Exp_013.py` preserves the regex patterns; if a future run refines the tag rate (e.g., parses the scraped team pages for current-leadership founder mentions), the regex can be lifted from this snapshot directly. Per §5 Decoupled Isolation Rule, **no follow-up α-tuning run is warranted**; the feature does not have signal at baseline α to merit tuning.

### Audit
* **Judge integrity:** SHA-256 `8f7aa10f25b1...` verified prior to Worker execution.
* **Output schema:** `results.tsv` written with header `Predicted Score\tCompany Name`, tab-delimited, 62 firms scored, all values on the 0.5 grid.
* **Scrape coverage:** 61 / 62 firms reachable from `logs/scrape_cache.json` (1 cached failure, unchanged).
* **Revert verification:** `model.py` was reset to `logs/Snapshot_model_Exp_009.py` via `cp` before any edits; `diff` confirmed byte-identical to baseline pre-edit.
* **Numeric sanity:** all 5 regexes compiled successfully; Ridge converged silently under default solver; no NaN/inf in the founder_led column.
* **founder_led tag distribution:** 4/62 firms tagged (6.5% positive rate). No false-positive audit performed in this run; recommended for any follow-up that revisits the regex.
* **Snapshot:** `logs/Snapshot_model_Exp_013.py` written immediately after run (10004 bytes; 1682 bytes larger than exp_009 snapshot due to the regex patterns block + helper function + featurize wiring + the founder_led tag-count print line).
* **Code Instability classification:** none triggered.
