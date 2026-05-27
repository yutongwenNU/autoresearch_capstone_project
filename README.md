# Auto-Private-Equity Search Engine — IT MSP Sourcing

An AutoResearch project that automates the identification of "stable but stagnant" IT Managed Service Providers (MSPs) in the U.S. Midwest as Search Fund acquisition targets. The project applies the agent loop **propose → edit → run → compare → keep/discard** to iteratively reduce a validation RMSE against manually labeled "Investment Grade" scores.

---

## Problem

Predict a continuous **1.0–10.0 Investment Grade score** for each IT MSP based on the Stanford / Yale search-fund thesis (deep founder tenure, succession-gap upside, recurring-revenue quality, technical-stagnation upside, geography-of-fit).

* **Metric:** validation RMSE against 62 manually labeled Midwest MSPs (lower is better). Current best: **RMSE 1.3955, R² 0.5128** (Exp_009 — Ridge + Stagnation Premium, 2026-05-07).
* **Ground truth:** `Manual Score` column in `data/train_set.csv`, labels quantized to 0.5 increments.
* **Data:** Apollo.io firmographic export + per-firm web scraping for qualitative signals.

---

## Key Architecture: Worker / Judge with a "One-Way Valve"

This project extends the basic AutoResearch loop with a **tamper-evidence layer** that protects evaluation integrity. The agent may freely modify the Worker (`model.py`); the Judge (`eval/prepare.py`) is locked by SHA-256 baseline and verified before every run.

```
┌──────────────────────────┐         ┌───────────────────────────┐
│  model.py (Worker)    │ writes  │  results.tsv              │
│  EDITABLE                │ ──────▶ │  Predicted Score \t Name  │
│  (features + regressor)  │         └───────────────────────────┘
└──────────────────────────┘                       │
                                                   ▼
              ┌──────────────────────────────────────────────────┐
              │  run_experiment.py  (FROZEN orchestrator)        │
              │   1. verify_integrity.py — SHA-256 of Judge      │
              │      ├── match → continue                        │
              │      └── mismatch → ABORT "Tamper Detected"      │
              │   2. python model.py        (Worker)          │
              │   3. python eval/prepare.py    (Judge)           │
              └──────────────────────────────────────────────────┘
                                                   │
                                                   ▼
                                       logs/results.tsv (RMSE, R², status)
                                       logs/performance.png
```

**Key rule:** the agent may only modify `model.py`. `eval/prepare.py`, `run_experiment.py`, and `verify_integrity.py` are FROZEN and any modification trips the SHA-256 check.

---

## Project Structure

```
capstone_project/
├── model.py                      # EDITABLE — Worker: features + regressor
├── run_experiment.py                # FROZEN  — orchestrator (verify → Worker → Judge)
├── verify_integrity.py              # FROZEN  — SHA-256 verifier
├── eval/
│   ├── prepare.py                   # FROZEN  — Judge: data join + RMSE + log append
│   └── prepare.sha256               # locked baseline checksum
├── data/
│   ├── train_set.csv                # 62 manually labeled Midwest IT MSPs
│   └── locked_test_set.csv          # held-out test set (post-iteration evaluation)
├── program.md                       # AutoResearch agent instructions
├── results.tsv                      # current Worker output (per-firm scores)
├── logs/
│   ├── results.tsv                  # rolling experiment log (one row per run)
│   ├── performance.png              # RMSE / R² over iterations
│   ├── Research_Log_Exp_NNN.md      # detailed per-experiment writeups
│   ├── Runtime&Budget_Log_*.md      # runtime + cost log per experiment
│   └── scrape_cache.json            # per-URL cache from the website scraper
├── 2020-Search-Fund-Primer.pdf      # thesis source
├── On the Nature of Revenue.pdf     # thesis source (revenue quality)
└── The Arc of a 10x Outcome.pdf     # thesis source (succession upside)
```

---

## Quick Start (For Grader)

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd capstone_project

# Python 3.10+ required
pip install pandas numpy scikit-learn requests matplotlib
```

The Worker scrapes 62 company websites on first run, so an **active internet connection** is required for the cold path. Subsequent runs hit `logs/scrape_cache.json` and complete in seconds.

### 2. (Optional) Verify the Judge is unmodified

```bash
python verify_integrity.py
# Expected: "Integrity Verified: <sha-256-hash>"
```

### 3. Run the experiment loop

The canonical entry point is `run_experiment.py`, which enforces the verify → Worker → Judge sequence and refuses to run if the Judge has been tampered with.

```bash
# Standard form: <description> + status flag
python run_experiment.py "Reproducing best-known model" --keep

# Example output:
# === Step 1/3: Verifying Judge integrity ===
# Judge integrity OK (SHA-256: 570d9e2a89c8...)
# === Step 2/3: Running Worker (model.py) ===
# Scrape complete: 61/62 firms reachable, 1 failures.
# === Step 3/3: Running Judge (eval/prepare.py) ===
# Evaluation Complete | RMSE: 1.5016 | Status: keep
```

Status flags (passed through to the experiment log):
* `--baseline` — establishing run, no comparison to prior best
* `--keep` — change is being adopted (RMSE improved or signal validated)
* `--discard` — change is being rolled back (RMSE regressed)

### 4. Inspect results

```bash
cat logs/results.tsv          # rolling experiment log
open logs/performance.png     # RMSE / R² over iterations
ls logs/Research_Log_*.md     # per-experiment writeups
```

---

## How to Run the AutoResearch Loop

### Quick-start agent prompt

```
Read program.md for your instructions, then read model.py.
The current best is in logs/results.tsv. Then enter the AutoResearch loop:

1. Propose ONE modification to model.py grounded in either:
   - a Data Science angle (regularization, feature engineering, model class), OR
   - a Business Heuristic from the search-fund literature (succession gap,
     revenue quality, technical stagnation, size sweet spot)
2. Edit model.py.
3. Run: python run_experiment.py "<description>" --keep|--baseline|--discard
4. Compare new val_rmse vs current best:
   - If improved → keep change, write logs/Research_Log_Exp_NNN.md
   - If regressed → revert model.py, log as --discard
5. Always include a model-internals diagnostic (Ridge coefficients or
   tree feature importances) and "what this likely tells us" interpretation.
6. Repeat. Try at least 4 different ideas.
```

### Constraints (from `program.md`)

* Agent may only modify `model.py`.
* `eval/prepare.py`, `run_experiment.py`, `verify_integrity.py` are FROZEN — any modification trips the SHA-256 check and aborts the run.
* Each loop must complete in **< 10 seconds per firm** (scraping + scoring).
* `results.tsv` schema: header row `Predicted Score\tCompany Name`, tab-delimited, every firm in `train_set.csv` scored.

---

## The Five Research Axes

By Week 5 the 20 controlled experiments had organized cleanly into five hypothesis directions. Each axis is a class of *what makes a Midwest IT MSP a good search-fund acquisition target*, plus the encoding used to test it.

| Axis | Key Feature | Status | Impact on Model |
|---|---|---|---|
| **Demographics** | `tenure_sq` | **Core** | Established the *"Succession Gap"* bell curve over tenure. Load-bearing in 14/14 controlled experiments. |
| **Operational Efficiency** | `rev_per_emp` | **Core** | Validated the *"Stagnation Premium."* **Current baseline — Exp_009, RMSE 1.3955.** |
| **Ownership / Succession** | `succession_red_flag` family | Exploratory | Correctly-signed across five encodings; all failed RMSE due to sparsity (3–11% tag rates at N=62). |
| **Competitive Moat** | `has_moat` | **Discarded** ⚠️ | Apparent Exp_016 win disqualified by **Exp_018 "Compliance Artifact" audit**. |
| **Model Integrity / Robustness** | Bagging / Interaction / Sigmoid | **Discarded** | Confirmed simple Ridge(α=1.0) is the right inductive bias for N=62. |

Full per-experiment groupings, diagnostics, and the Exp_018 audit narrative are in `logs/Research_Axes_Summary.md`.

---

## Experimental Trajectory (Axis-Organized, 20 Experiments)

Headline progression: **RMSE 1.8460 → 1.3955 (−24.4%)** and **R² 0.147 → 0.513 (~3.5× variance explained)** across the 20 runs. The two axis-defining wins are shown below with their code; everything else is summarized in the tables.

### Core Axis 1 — Demographics: the `tenure_sq` Bell Curve (Exp_002)

Replaced the hand-coded boolean rules with Ridge over 9 engineered features, the most consequential of which was the **inverted-U over founder tenure** encoded as a `tenure` + `tenure_sq` pair:

```python
# model.py — featurize()
tenure = (CURRENT_YEAR - founded).clip(lower=0)
# ...
return pd.DataFrame({
    "tenure":         tenure,
    "tenure_sq":      tenure ** 2,         # negative coefficient → bell curve
    "sweet_spot_emp": ((employees >= 10) & (employees <= 30)).astype(int),
    # ... 6 other features ...
})
```
```
val_rmse: 1.5112   val_r2: 0.4287   status: keep   (Δ −0.335, +28pp R²)
```
**Why it stuck:** `tenure_sq` coefficient has held in [−0.27, −0.45] across every healthy run since. Negative on `tenure_sq` plus positive on `tenure` = "established but not ancient" search-fund sweet spot. It is the single most reliable diagnostic of model health in the project.

### Core Axis 2 — Operational Efficiency: the Stagnation Premium (Exp_009, **current baseline**)

After 6 failed attempts to encode the Yale "Nature of Revenue" thesis via marketing keywords (Exp_008 weighted Premium MRR was the cleanest failure), Exp_009 switched from text to a **structural revenue ratio**:

```python
# model.py — featurize()
rev_per_emp = revenue / employees.clip(lower=1)   # $ per head, no log
# Added as the 11th column of the returned feature DataFrame.
```
```
val_rmse: 1.3955   val_r2: 0.5128   status: keep   ← new all-time best, current baseline
```
**The surprise:** Ridge gave `rev_per_emp` a **negative** coefficient (−0.66) — opposite to the naïve "more efficiency = better firm" reading. The coherent interpretation is the **Stagnation Premium**: among firms of similar size and revenue, *lower* per-head productivity signals operational slack a searcher can convert. The new feature is the largest non-`log_revenue` absolute coefficient and earns the −7.1% RMSE improvement over the prior best.

### Exploratory — Ownership / Succession (5 attempts, all failed RMSE)

A series of NLP and keyword features attempted to flag *firms where the founder is still in charge or that have already been acquired*. Every attempt produced a **correctly-signed coefficient** (in [−0.11, −0.36]) but **none beat baseline** — all suffered from sparse tag rates (3–11 firms out of 62).

| Exp | Variant | Tag rate | Coef | RMSE |
|---|---|---|---|---|
| 003 | `mgmt_depth` (scraped role-title count) | continuous | +0.04 | 1.5016 (marginal) |
| 013 | `founder_led` (5 regex patterns) | 4/62 | −0.11 | 1.4555 |
| 014 | `is_institutionalized` (10 keywords + 1 regex) | 3/62 | −0.36 | 1.4645 |
| 015 | `succession_red_flag` (founder ∪ acquisition) | 7/62 | −0.31 | 1.4690 |
| 020 | `ownership_red_flag` (4 explicit phrases) | 3/62 | −0.36 | 1.4645 (= Exp_014) |

**Cross-experiment threshold finding:** at N=62, a single binary feature needs ≥ ~16% tag rate to be RMSE-positive even when correctly signed. This thesis direction remains worthwhile but needs a denser data source (e.g., parsing the cached team pages directly rather than relying on description copy).

### Discarded — Competitive Moat: the Compliance Artifact ⚠️ (Exp_016 → Exp_018 Audit)

**The pivot moment of Week 5.** Exp_016 added a binary `has_moat = 1` if the description contained any of `hipaa, dental, legal, law firm, manufacturing, pci, compliance, regulated`:

```python
MOAT_KW = ["hipaa", "dental", "legal", "law firm", "manufacturing", "pci", "compliance", "regulated"]
```
```
val_rmse: 1.3079   val_r2: 0.5721   status: keep (provisional)   ← apparent new best
```
But three diagnostics raised flags: tag rate **88.7%**, coefficient **sign opposite** to the stated hypothesis (−0.58), and **`compliance` alone tagged 79% of firms**. The Exp_018 ablation removed `compliance`:

```python
MOAT_KW = ["hipaa", "dental", "legal", "law firm", "manufacturing", "pci", "regulated"]  # compliance OUT
```
```
val_rmse: 1.4049   val_r2: 0.5062   status: discard   ← entire lift was the compliance keyword
```
**Verdict:** the +0.097 RMSE swing on removing one keyword proved Exp_016 was an artifact, not a vertical-moat signal. The 7 firms that *did not* mention `compliance` (Pinnacle, Innovative Computers, Miken, etc.) happened to be high-Manual-Score outliers — Ridge was fitting "atypical marketing copy" labeler bias, not business economics. The audit caught a Type-I promotion before commitment. **The Moat axis is parked until a non-keyword encoding is proposed.**

### Discarded — Model Integrity / Robustness (5 attempts, all failed RMSE)

Every attempt to "improve" the model class regressed:

| Exp | Change | RMSE | Failure Mode |
|---|---|---|---|
| 005 | HGBR + 0.5 rounding (bundled) | 2.1764 | HGBR overfits 50-row CV folds; N=62 too small |
| 007 | Lasso(α=0.1) | 1.7168 | L1 pruned the load-bearing `tenure_sq` |
| 011 | BaggingRegressor(n=50, max_samples=0.8) | 1.4464 | Averaging shrinks `tenure_sq` mean to ~0 across bags |
| 012 | `tenure × rev_per_emp` + GridSearchCV α | 1.8743 | Multicollinearity → α=10 erased load-bearing coefs |
| 010 | `sweet_spot_emp` → Gaussian (μ=20, σ=10) | 1.6741 | Smoothing broke implicit regularization of `log_employees` |

This axis is the empirical motivation for `program.md` §5 (Decoupled Isolation Rule) and §6 (5× Alpha Guardrail).

### Summary — All 20 Experiments

| # | Axis | Variable | RMSE | Decision |
|---|---|---|---|---|
| 001 | (origin) | Hand-coded heuristics | 1.8460 | baseline |
| 002 | Demographics | Ridge + 9 engineered features | 1.5112 | keep |
| 003 | Ownership | + `mgmt_depth` scraper | 1.5016 | keep (marginal) |
| 004 | Ownership | + `tenure × mgmt-absence` | 1.5110 | discard |
| 005 | Robustness | HGBR + 0.5 rounding | 2.1764 | discard |
| 006 | Robustness | 0.5 rounding (Ridge Control) | 1.5044 | keep |
| 007 | Robustness | Lasso(α=0.1) | 1.7168 | discard |
| 008 | Operational | Weighted Premium MRR keywords | 1.5811 | discard |
| **009** | **Operational** | **`rev_per_emp` (Stagnation Premium)** | **1.3955** | **keep ← current baseline** |
| 010 | Demographics | Gaussian `sweet_spot_emp` | 1.6741 | discard |
| 011 | Robustness | BaggingRegressor | 1.4464 | discard |
| 012 | Robustness | `tenure × rev_per_emp` + α tuning | 1.8743 | discard |
| 013 | Ownership | `founder_led` (NLP regex) | 1.4555 | discard |
| 014 | Ownership | `is_institutionalized` | 1.4645 | discard |
| 015 | Ownership | `succession_red_flag` (combined) | 1.4690 | discard |
| 016 | Moat | `has_moat` (incl. `compliance`) | 1.3079 | **discard (audit ⚠️)** |
| 017 | Operational | `stagnation_ratio` | 1.5811 | discard |
| 018 | Moat | `has_moat` (no `compliance`) | 1.4049 | discard (audit) |
| 019 | Demographics | `is_hub_proximate` | 1.4189 | discard |
| 020 | Ownership | `ownership_red_flag` (refined) | 1.4645 | discard |

**Per-experiment writeups:** `logs/Research_Log_Exp_NNN.md` for every run, with hypothesis, Ridge-coefficient diagnostic, "what this likely tells us" interpretation, and (from Exp_011 onward) a 4-category Failure Mode entry per `program.md` §Logging Standards.

---

## How the One-Way Valve Works

The Judge (`eval/prepare.py`) is the source of truth for the RMSE metric. To prevent silent tampering — e.g., the agent quietly relaxing the evaluation logic to make its scores look better — the Judge is locked by SHA-256 and verified before every Worker run.

```python
# verify_integrity.py
EXPECTED_HASH = "8f7aa10f25b1..."  # full 64-hex hash in the file
                                   # re-baselined after the Week-4 plotting fix to eval/prepare.py;
                                   # originally 570d9e2a89c8... pre-rebaseline.

def verify_prepare():
    actual = sha256(eval/prepare.py)
    if actual != EXPECTED_HASH:
        raise RuntimeError("[TAMPER DETECTED]")
```

`run_experiment.py` calls `verify_prepare()` as Step 1/3 and aborts the entire pipeline (Worker never runs, no log entry written) if the hash doesn't match. The Judge file is also `chmod 444` on the local filesystem as a speed-bump against accidental edits — though the SHA-256 check is the real lock.

To intentionally update the Judge: re-`chmod 644`, edit, re-`chmod 444`, recompute the hash with `shasum -a 256 eval/prepare.py`, and update `EXPECTED_HASH` in `verify_integrity.py`. This is a deliberate action that produces a visible diff in version control.

---

## Reproducing the Best Result

`model.py` is checked in at the **Exp_009 configuration** — Ridge(α=1.0) + StandardScaler + 11 engineered features (10 from Exp_006 + `rev_per_emp`) + 0.5-grid rounding. The same code is preserved in `logs/Snapshot_model_Exp_009.py` as the canonical reproducibility reference. Running the pipeline should reproduce the best-known RMSE of **1.3955** directly:

```bash
# View the rolling experiment log
cat logs/results.tsv

# Run via the wrapper (recommended — verifies Judge integrity first)
python run_experiment.py "Reproducing exp_009 best" --keep

# Or run the Worker and Judge by hand:
python model.py            # writes results.tsv
python eval/prepare.py     # prints RMSE

# Expected:
# Evaluation Complete | RMSE: 1.3955 | Status: keep
```

Note: web scraping introduces external dependencies. The `logs/scrape_cache.json` file is committed to the repo so the cold-scrape path is only needed for fresh forks; reruns from cache are deterministic and finish in ~1 second.

---

## Adapting This Structure for Your Own Project

The Worker / Judge / one-way-valve pattern is reusable for any AutoResearch task where evaluation integrity matters:

1. **`eval/prepare.py`** — your data loading, evaluation metric, plotting, log-append. Frozen.
2. **`model.py`** — the agent's modifiable surface (model definition, feature engineering, hyperparameters).
3. **`verify_integrity.py`** — SHA-256 check on the Judge, with a hardcoded `EXPECTED_HASH` constant.
4. **`run_experiment.py`** — orchestrator that calls verify → Worker → Judge in that order, aborting on tamper.
5. **`program.md`** — the agent's rules and search ideas.
6. **`logs/`** — rolling experiment log + per-experiment markdown writeups + plot.

The principle: **separate what changes (Worker) from what measures (Judge), and make the boundary cryptographically auditable.**
