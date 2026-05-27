# AutoResearch Agent Instructions: MSP Sourcing

## Core Objective & ScopeTask 
- Type: Supervised Machine Learning Ordinal Regression (Scaled 1.0–10.0, constrained to strict 0.5-grid increments).
- Target Domain: IT Managed Service Providers (MSPs).
- Objective Function: Minimize Validation RMSE via 5-Fold Cross-Validation on a human-labeled training set (N=62).
- Historical Baseline: Exp_001 Hand-Coded Baseline (RMSE: 1.8460).
- Operative Baseline: Experiment 009 (Ridge Regression `alpha=1.0`, 11 Features, RMSE: 1.3955, R²: 0.5128).
- Final Presentation Goal: Achieve an out-of-sample Test Set RMSE < 2.0 on the locked 28-firm "blind exam" in Week 8.


## Protected Guardrails & System Rules
- Permitted Modifications: You may ONLY modify `model.py`.
- Protected Components: `eval/prepare.py`, `run_experiment.py`, and `verify_integrity.py` are FROZEN. Do not attempt to modify the evaluation logic or the data join.
- Data Handling: `model.py` must handle the 'Company Name' column (string data) by dropping it or using a `ColumnTransformer` before passing data to the regressor.
- Runtime Budget: Complete the entire scraping and scoring loop in under 20 seconds per firm.


## General Workflow
- Read: Analyze the current scoring logic in model.py (which is the baseline model), the current results.tsv (outputs), and the target heuristics in this document.
- Propose: Identify a specific Data Science or Business Heuristic to test (e.g., "Lasso Regression" or "Succession Gap Interaction").
- Edit: Modify model.py with the proposed regression logic. Ensure that the output format in model.py remains consistent (results.tsv with firm_id and predicted score). Ensure your build_model() returns an sklearn-compatible pipeline.
- Run: Execute the experiment using the unified wrapper: `python run_experiment.py "Short description of change" --keep` (Use `--baseline` for initial runs or `--discard` if you are intentionally testing a negative hypothesis.)
- Compare: Check the output val_rmse.
    - **If $RMSE$ < best:** Mark as `keep`. Commit the change: `git add model.py && git commit -m "feat: [Description]"`
    - **If $RMSE$ >= best:** Mark as `discard`. Revert `model.py` to the last known good state.
- Repeat: Continue iterating through failure modes.
    - **Document Failure:** If the run crashes or regresses significantly, record the specific failure mode in the logs subfolder.
- **Outcome Recording**: Every run must append to results.tsv. Required Fields: experiment_id, val_rmse, status (keep/discard/baseline), and a descriptive reason for the change.
    - model.py must produce a file named results.tsv in the project root with
        - Header Row: Predicted Score\tCompany Name
        - Delimiter: Tab-separated.
        - Content: Every firm in the training set must be scored.
    - Each experiment's model code must be saved down: `Snapshot_model_Exp_XXX.py` in the logs folder.


### Logging Standards
- **Summary Log (`logs/results.tsv`):** Every single run—whether it is a baseline, a keep, or a discard—must append a row here.
- **Detailed Trace (`logs/`):** For every experiment, the `run_experiment.py` wrapper should generate a detailed text file (e.g., `logs/exp_001_trace.txt`) containing:
    - The raw predictions vs. manual labels for the 63 firms.
    - Any scraper errors or "Unknown" values encountered.
    - The agent's internal "Chain of Thought" explaining why this specific regressor was chosen.
- **One-Way Valve Audit:** Before every log entry, the system must confirm that the SHA-256 checksum of `eval/prepare.py` is unchanged.  
- **Logging Files**: You will see in /logs that there are a number of files you need to produce following each experiment as research logs:
    - `Research_Log_ExpXXX.md`: See previous versions for format. But future logs should also include a "Failure Mode" section:
        - Every experiment must be categorized into one of the following four failure modes (if summarized as `discard` or a regression (i.e. disimprovement) from previous experiments):
            1. **Signal Failure (Information/Heuristic):** The loop runs successfully, but metrics (RMSE and R²) do not improve. In the MSP context, this could be due to a proposed heuristic (e.g., "Succession Gap") not having predictive power, or a data science method (e.g., "Lasso Regression") not effectively capturing the signal in the training data. Includes Sparse Signal Subtypes (tag rate <16%) and Feature Redundancy Subtypes (collinear variables).
            2. **Code Instability (Infrastructure):** Crashes or pipeline breaks. Strict environment exceptions, pipeline breaks, runtime budget overruns (>20 seconds), or syntax exceptions. In the MSP context, this could be due to Scraper 403 Forbidden errors, formatting errors in `results.tsv` (commas/tabs), or `ValueError` in data preprocessing, etc.
            3. **Evaluation Leakage (Validity):** Metric improvements that are "fake" because the setup shifted. In the MSP context, you as the Coding Agent attempts to modify the `Manual Score` labels, the training/test split, or the evaluation math in `prepare.py`.
            4. **Agent Misbehavior (Control):** You as the coding agent ignores constraints or makes uncontrolled changes. Disregarding structural system rules, failing to round outputs to the required 0.5-ordinal increment, or deploying stacked features without a baseline reset. In the MSP context, this could be due to the agent adding 100 random features when asked for one isolated heuristic, or ignoring the 0.5-interval rounding rule for ordinal regression.
    - `Runtime&Budget_Log_ExpXXX.md`: See previous versions for format.


### Workflow Flowchart at Operational Baseline (Exp_009)

[Start Experiment]
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
┌──────────────────┐       ┌────────────────────┐
│ CONTROLLED RUN   │       │ BATCH RUN (AXIS)   │
└────────┬─────────┘       └─────────┬──────────┘
         │                           │
         ▼                           ▼
  Revert to Exp_009           Revert to Exp_009 
  Snapshot via `cp`           Pre-Run Every Test
         │                           │
         ▼                           ▼
   Test 1 Feature              Test Sibling Group
   At Baseline α=1.0           At Baseline α=1.0
         │                           │
         └─────────────┬─────────────┘
                       │
                       ▼
             [Abnormal Shift Check]
             - Tag Rate < 16% ?
             - Coef Sign Flip ?
             - tenure_sq Weakens > 30% ?
                       │
              ┌────────┴────────┐
              ▼                 ▼
          (Passed)           (Failed)
              │                 │
              ▼                 ▼
     [Optional HPT Loop]   [STOP RULE FIRED]
     GridSearchCV Over α   Log Discard & Flag

- **Protocol A:** Controlled Single Experiments
    - Definition: An isolated test of exactly one explicit variable change against the current operative baseline (Exp_009).
    - Isolation Rule: Before running, you must execute `cp logs/Snapshot_model_Exp_009.py model.py` to ensure a clean baseline.
    - The Decoupled Rule: New features must be tested first at the baseline alpha (`alpha=1.0`). Never mix a new feature and a hyperparameter tuning pass in the same initial run.
    
- **Protocol B:** Grouped Batch Experiments
    - Definition: Running a sibling group of distinct feature combinations under a unified thematic research axis as defined below.
    - Batch Isolation Rule: You must iterate a hard loop that executes a clean `cp logs/Snapshot_model_Exp_009.py model.py` prior to each individual test within the batch. Never stack untested features sequentially.

- **Protocol C:** Hyperparameter Tuning (HPT) Escalation
    - Trigger condition: You are permitted to execute an autonomous tuning loop (GridSearchCV or RandomizedSearchCV, etc.) if and only if a feature has already demonstrated a positive standalone RMSE improvement at the baseline alpha (`alpha=1.0`).
    - The 5x Alpha Guardrail: If an internal search loop selects an optimal `alpha` that is more than 5x higher or lower than the baseline value (e.g., jumping from 1.0 to 10.0), flag the run for manual review. High shifts typically indicate the model is dampening collinear noise rather than reading a stable signal.


## The Five Sourcing Research Axes

| Research Axis | Anchor Metric | Feature Name | System Function / Description | Project Status & Baseline | Promotion Rule / Next Step |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Axis 1:<br>Demographics** | Succession Gap Window | `tenure_sq` | Models a non-linear inverted quadratic U-curve for company age. It recognizes that target quality peaks between 17 and 25 years when a founder is hitting retirement velocity. | **Core Baseline**<br>(Retained) | Must preserve an inverted-U curve shape with a strict negative coefficient across all future iterations. |
| **Axis 2:<br>Operational Efficiency** | Stagnation Premium | `rev_per_emp` | Tracks raw company revenue divided by headcount to calculate operational slack. It penalizes highly optimized, modern firms and rewards stable, unoptimized legacy businesses. | **Current Baseline**<br>(Exp_009 Champion:<br>RMSE 1.3955) | All future isolation runs are mathematically benchmarked against this structural feature configuration. |
| **Axis 3:<br>Ownership Status** | Corporate Institutionalization | `ownership_red_flag`<br>family | Parses text data for active acquisition flags (`acquired by`, `subsidiary of`) to act as a target filter. | *Exploratory*<br>(All runs discarded) | Features must achieve a dataset tag density of `>= 16%` to avoid high-variance outlier leverage at `N=62`. |
| **Axis 4:<br>Competitive Moats** | Vertical Specialization | `has_moat` | Scans for industry infrastructure stickiness (e.g., Legal tech, HIPAA compliance, Dental software integrations). | *Suspended*<br>(Parked via Exp_018) | Text keyword-matching is banned due to proxy leakage (the `compliance` buzzword artifact). Requires non-keyword inputs (e.g., NAICS codes). |
| **Axis 5:<br>Model Integrity** | Structural Robustness | Regularization<br>Pipeline Constraints | Enforces a simple linear inductive bias (Ridge `alpha=1.0` with a strict `StandardScaler` wrapper) over complex trees. | **Core Framework**<br>(Enforced) | Protects the pipeline from overfitting or memorizing noise. Activates the 5x Alpha Guardrail for hyperparameter loops. |

You are welcomed to read or re-read through the literatures located under /Search Fund Literature, but you should only implement features that can be operationalized within the 20-second runtime budget and the current data schema. Prioritize features that are directly supported by the existing dataset or can be engineered from it without external data sources.
