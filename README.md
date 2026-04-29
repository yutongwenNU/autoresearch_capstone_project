# Auto-Private-Equity Search Engine — IT MSP Sourcing

An AutoResearch project that automates the identification of "stable but stagnant" IT Managed Service Providers (MSPs) in the U.S. Midwest as Search Fund acquisition targets. The project applies the agent loop **propose → edit → run → compare → keep/discard** to iteratively reduce a validation RMSE against manually labeled "Investment Grade" scores.

---

## Problem

Predict a continuous **1.0–10.0 Investment Grade score** for each IT MSP based on the Stanford / Yale search-fund thesis (deep founder tenure, succession-gap upside, recurring-revenue quality, technical-stagnation upside, geography-of-fit).

* **Metric:** validation RMSE against 62 manually labeled Midwest MSPs (lower is better).
* **Ground truth:** `Manual Score` column in `data/train_set.csv`, labels quantized to 0.5 increments.
* **Data:** Apollo.io firmographic export + per-firm web scraping for qualitative signals.

---

## Key Architecture: Worker / Judge with a "One-Way Valve"

This project extends the basic AutoResearch loop with a **tamper-evidence layer** that protects evaluation integrity. The agent may freely modify the Worker (`research.py`); the Judge (`eval/prepare.py`) is locked by SHA-256 baseline and verified before every run.

```
┌──────────────────────────┐         ┌───────────────────────────┐
│  research.py (Worker)    │ writes  │  results.tsv              │
│  EDITABLE                │ ──────▶ │  Predicted Score \t Name  │
│  (features + regressor)  │         └───────────────────────────┘
└──────────────────────────┘                       │
                                                   ▼
              ┌──────────────────────────────────────────────────┐
              │  run_experiment.py  (FROZEN orchestrator)        │
              │   1. verify_integrity.py — SHA-256 of Judge      │
              │      ├── match → continue                        │
              │      └── mismatch → ABORT "Tamper Detected"      │
              │   2. python research.py        (Worker)          │
              │   3. python eval/prepare.py    (Judge)           │
              └──────────────────────────────────────────────────┘
                                                   │
                                                   ▼
                                       logs/results.tsv (RMSE, R², status)
                                       logs/performance.png
```

**Key rule:** the agent may only modify `research.py`. `eval/prepare.py`, `run_experiment.py`, and `verify_integrity.py` are FROZEN and any modification trips the SHA-256 check.

---

## Project Structure

```
capstone_project/
├── research.py                      # EDITABLE — Worker: features + regressor
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
# === Step 2/3: Running Worker (research.py) ===
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
Read program.md for your instructions, then read research.py.
The current best is in logs/results.tsv. Then enter the AutoResearch loop:

1. Propose ONE modification to research.py grounded in either:
   - a Data Science angle (regularization, feature engineering, model class), OR
   - a Business Heuristic from the search-fund literature (succession gap,
     revenue quality, technical stagnation, size sweet spot)
2. Edit research.py.
3. Run: python run_experiment.py "<description>" --keep|--baseline|--discard
4. Compare new val_rmse vs current best:
   - If improved → keep change, write logs/Research_Log_Exp_NNN.md
   - If regressed → revert research.py, log as --discard
5. Always include a model-internals diagnostic (Ridge coefficients or
   tree feature importances) and "what this likely tells us" interpretation.
6. Repeat. Try at least 4 different ideas.
```

### Constraints (from `program.md`)

* Agent may only modify `research.py`.
* `eval/prepare.py`, `run_experiment.py`, `verify_integrity.py` are FROZEN — any modification trips the SHA-256 check and aborts the run.
* Each loop must complete in **< 10 seconds per firm** (scraping + scoring).
* `results.tsv` schema: header row `Predicted Score\tCompany Name`, tab-delimited, every firm in `train_set.csv` scored.

---

## Experimental Trajectory (Actually Executed)

Below is the real agent session — 5 experiments, 4 keeps (3 productive, 1 logged-but-reverted), 1 model regression with high diagnostic value.

### exp_001 — Baseline: Hand-coded Heuristics

```python
# research.py — 4 boolean rules from the search-fund primer
score = 5.0
if 10 <= employees <= 30: score += 1.5
if (2026 - founded) >= 25: score += 1.5
if revenue >= 5_000_000: score += 1.5
if "managed" in desc or "recurring" in desc: score += 0.5
```
```
val_rmse: 1.846048   val_r2: 0.1474   status: baseline
```

### exp_002 — Ridge on 9 Engineered Features

Replaced hand rules with Ridge regression on engineered firmographics + keyword counts. Predictions written **out-of-fold** via `cross_val_predict(KFold=5)` so the RMSE the Judge computes is honest validation, not training fit.

```
val_rmse: 1.511196   val_r2: 0.4287   status: keep   (Δ −0.335, +28pp R²)
```

**Diagnostic:** Ridge coefficients showed `sweet_spot_emp = +0.79` and `tenure = +0.71` as the dominant signals — the engineered features carried real signal that the boolean rules were leaving on the table.

### exp_003 — Add "Management Depth" Web Scraper

Scraped each firm's homepage + first reachable team page (`/about`, `/team`, `/leadership`, ...) and counted distinct role-title regex patterns (CEO, CTO, "leadership team", etc.). Disk-cached in `logs/scrape_cache.json` for reproducibility.

```
val_rmse: 1.501594   val_r2: 0.4359   status: keep   (Δ −0.010, marginal)
```

**Diagnostic:** `mgmt_depth` coefficient was a tiny `+0.04` — wrong-signed per the search-fund thesis (which predicted a **negative** coefficient: more management = less succession gap = lower score). The marginal RMSE win was within noise. Conclusion: the standalone count was too coarse to extract the conditional structure of the thesis.

### exp_004 — Tenure × Management-Absence Interaction

Encoded the conditional explicitly: `succession_gap = tenure × (max_depth − mgmt_depth)`. Hypothesis: "deep tenure × no management" should compound into a single positive signal that Ridge can fit with one coefficient.

```
val_rmse: 1.510976   val_r2: 0.4288   status: discard
```

**Diagnostic — the experiment that "succeeded by failing":** the new interaction coefficient was `+0.28` and *correctly signed* (vs the noisy +0.04 standalone) — confirming the thesis is conditional, not additive. But predictive power didn't move because Ridge was redistributing weight across collinear features (tenure, mgmt_depth, succession_gap) without adding new information. **Reverted to exp_003.**

### exp_005 — HistGradientBoostingRegressor + 0.5-Grid Discretization

Two changes bundled: (a) swap Ridge → HGBR to capture non-linear thresholds natively; (b) snap predictions to 0.5 grid via `round(x*2)/2` to align with the discretization of the Manual Score labels.

```
val_rmse: 2.176429   val_r2: -0.1850   status: discard
```

**Diagnostic — informative regression:** HGBR drove R² *negative* — the model performs worse than predicting the mean. Permutation importance revealed the cause: HGBR zeroed out `sweet_spot_emp`, `tenure_sq`, `in_midwest`, and `modern_ai_kw` — features Ridge had relied on heavily. A tree finds those thresholds internally on the raw `log_employees` and `tenure`, so the engineered features are redundant; **but** N=62 is too small for boosted trees to generalize.

A follow-up showed that the bundled 0.5-rounding step also doesn't generalize — applying it to the exp_003 Ridge pipeline pushes RMSE from 1.50 to 2.22. Random rounding to a 0.5 grid adds ~√(1/12) × 0.5 ≈ 0.14 of expected RMSE just from quantization noise, which is only "earned back" when the model is accurate enough to land ≥30% of predictions in their correct bin. **Both bundled changes reverted; research.py is restored to pure exp_003.**

### Summary

| # | Worker change | RMSE | R² | Decision |
|---|---|---|---|---|
| 001 | Hand-coded heuristics | 1.8460 | 0.147 | baseline |
| 002 | Ridge + 9 engineered features | 1.5112 | 0.429 | keep |
| 003 | + `mgmt_depth` from web scraper | **1.5016** | **0.436** | keep ← **best (active)** |
| 004 | + tenure × mgmt-absence interaction | 1.5110 | 0.429 | discard |
| 005 | HGBR + 0.5-grid rounding | 2.1764 | −0.185 | discard |

**Total improvement: RMSE 1.8460 → 1.5016 (−18.6%); R² 0.147 → 0.436 (~3× variance explained).**

The detailed per-experiment writeups in `logs/Research_Log_Exp_NNN.md` document the proposal, rationale (citing the search-fund primer where applicable), result, model-internals diagnostic, and "what this likely tells us" interpretation for each run.

---

## How the One-Way Valve Works

The Judge (`eval/prepare.py`) is the source of truth for the RMSE metric. To prevent silent tampering — e.g., the agent quietly relaxing the evaluation logic to make its scores look better — the Judge is locked by SHA-256 and verified before every Worker run.

```python
# verify_integrity.py
EXPECTED_HASH = "570d9e2a89c805992323f21c350a4048fc947aa8df274ff935772bef641a4243"

def verify_prepare():
    actual = sha256(eval/prepare.py)
    if actual != EXPECTED_HASH:
        raise RuntimeError("[TAMPER DETECTED]")
```

`run_experiment.py` calls `verify_prepare()` as Step 1/3 and aborts the entire pipeline (Worker never runs, no log entry written) if the hash doesn't match. The Judge file is also `chmod 444` on the local filesystem as a speed-bump against accidental edits — though the SHA-256 check is the real lock.

To intentionally update the Judge: re-`chmod 644`, edit, re-`chmod 444`, recompute the hash with `shasum -a 256 eval/prepare.py`, and update `EXPECTED_HASH` in `verify_integrity.py`. This is a deliberate action that produces a visible diff in version control.

---

## Reproducing the Best Result

`research.py` is checked in at the exp_003 configuration — Ridge + 9 engineered features + scraped `mgmt_depth`, continuous output, no rounding. Running it should reproduce the best-known RMSE of **1.5016** directly:

```bash
# View the rolling experiment log
cat logs/results.tsv

# Run via the wrapper (recommended — verifies Judge integrity first)
python run_experiment.py "Reproducing exp_003 best" --keep

# Or run the Worker and Judge by hand:
python research.py            # writes results.tsv
python eval/prepare.py        # prints RMSE

# Expected:
# Evaluation Complete | RMSE: 1.5016 | Status: keep
```

Note: web scraping introduces external dependencies. The `logs/scrape_cache.json` file is committed to the repo so the cold-scrape path is only needed for fresh forks; reruns from cache are deterministic and finish in ~1 second.

---

## Adapting This Structure for Your Own Project

The Worker / Judge / one-way-valve pattern is reusable for any AutoResearch task where evaluation integrity matters:

1. **`eval/prepare.py`** — your data loading, evaluation metric, plotting, log-append. Frozen.
2. **`research.py`** — the agent's modifiable surface (model definition, feature engineering, hyperparameters).
3. **`verify_integrity.py`** — SHA-256 check on the Judge, with a hardcoded `EXPECTED_HASH` constant.
4. **`run_experiment.py`** — orchestrator that calls verify → Worker → Judge in that order, aborting on tamper.
5. **`program.md`** — the agent's rules and search ideas.
6. **`logs/`** — rolling experiment log + per-experiment markdown writeups + plot.

The principle: **separate what changes (Worker) from what measures (Judge), and make the boundary cryptographically auditable.**
