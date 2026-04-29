# AutoResearch Agent Instructions: MSP Sourcing

## Objective
The goal is a **Regression Task**. We are predicting a continuous 1.0–10.0 "Investment Grade" score for IT Managed Service Providers. Minimize the **Validation RMSE** against the manually labeled training set with 63 IT MSPs. Success is defined as decreasing the RMSE from the current baseline of 1.8460.


## Rules
- Permitted Modifications: You may ONLY modify `research.py`.
- Protected Components: As prepared earlier, `eval/prepare.py`, `run_experiment.py`, and `verify_integrity.py` are FROZEN. Do not attempt to modify the evaluation logic or the data join.
- Data Handling: `research.py` must handle the 'Company Name' column (string data) by dropping it or using a `ColumnTransformer` before passing data to the regressor.
- Efficiency: Each research loop (scraping + scoring) must complete in under 10 seconds per firm to respect the runtime budget.
- Environment: Use the existing .env for API keys. Do not hard-code credentials or ground truth labels.


## Workflow
- Read: Analyze the current scoring logic in research.py (which is the baseline model), the current results.tsv (outputs), and the target heuristics in this document.
- Propose: Identify a specific Data Science or Business Heuristic to test (e.g., "Lasso Regression" or "Succession Gap Interaction").
- Edit: Modify research.py with the proposed regression logic. Ensure that the output format in research.py remains consistent (results.tsv with firm_id and predicted score). Ensure your build_model() returns an sklearn-compatible pipeline.
- Run: Execute the experiment using the unified wrapper: `python run_experiment.py "Short description of change" --keep` (Use `--baseline` for initial runs or `--discard` if you are intentionally testing a negative hypothesis.)
- Compare: Check the output val_rmse.
    - **If $RMSE$ < best:** Mark as `keep`. Commit the change: `git add research.py && git commit -m "feat: [Description]"`
    - **If $RMSE$ >= best:** Mark as `discard`. Revert `research.py` to the last known good state.
- Repeat: Continue iterating through failure modes.
    - **Document Failure:** If the run crashes or regresses significantly, record the specific failure mode in the logs subfolder.
- **Outcome Recording**: Every run must append to results.tsv. Required Fields: experiment_id, val_rmse, status (keep/discard/baseline), and a descriptive reason for the change.
    - research.py must produce a file named results.tsv in the project root with
        - Header Row: Predicted Score\tCompany Name
        - Delimiter: Tab-separated (\t).
        - Content: Every firm in the training set must be scored.


## Logging Standards
- **Summary Log (`logs/results.tsv`):** Every single run—whether it is a baseline, a keep, or a discard—must append a row here.
- **Detailed Trace (`logs/`):** For every experiment, the `run_experiment.py` wrapper should generate a detailed text file (e.g., `logs/exp_001_trace.txt`) containing:
    - The raw predictions vs. manual labels for the 63 firms.
    - Any scraper errors or "Unknown" values encountered.
    - The agent's internal "Chain of Thought" explaining why this specific regressor was chosen.
- **One-Way Valve Audit:** Before every log entry, the system must confirm that the SHA-256 checksum of `eval/prepare.py` is unchanged.  

## Ideas to Explore (Both Data Science and Business Heuristics Aspects)

### Data Science Toolkits
1. Regression Models & Ensemble Methods
- **Linear Models:** Explore `Ridge`, `Lasso`, or `ElasticNet` to handle potential multicollinearity between features like headcount and revenue.
- **Support Vector Machines:** Test `SVR` with different kernels (RBF, linear) for non-linear relationships.
- **Tree-Based Ensembles:** Implement `RandomForestRegressor`, `GradientBoostingRegressor`, or `HistGradientBoostingRegressor` to capture complex interactions between features.

2. Preprocessing & Feature Engineering
- **Scaling:** Use `StandardScaler`, `RobustScaler` (if data has outliers), or `QuantileTransformer` to normalize input features.
- **Interactions:** Use `PolynomialFeatures` to create interaction terms (e.g., Tenure x Employee Count) to find specific "Sweet Spots" in the search fund rubric.
- **Target Transformation:** Experiment with `TransformedTargetRegressor` using a `log` transform if the 1-10 scores are heavily skewed.

### Business Heuristics
- **Succession Gaps:** Look for "About Us" pages where founders have 20+ years of tenure with no clear "Next Gen" leadership listed.
    - Data Science Perspective: Explore interactions between Founder Tenure and Team Hierarchy depth to refine "Succession Potential."
- **Revenue Quality (Yale "Nature of Revenue"):** Search for signals of Managed Services (MRR) versus one-off project work or hardware sales.
    - Data Science Perspective: Weight "Recurring Revenue" keywords 2x higher than "Hardware Sales" keywords within your feature extraction logic.
- **Technical Stagnation:** Look for "Legacy Tech Debt" indicators—outdated website design, mention of EOL (End of Life) software, or lack of modern cybersecurity focus.
- **Size Sweet Spot:** Refine the headcount logic to distinguish between "Technician-heavy" firms and those with professional management.
    - Data Science Perspective: Implement bell-curve or Sigmoid scaling for the "Size Sweet Spot" (10-30 employees) to reward targets in the middle range.
- **Further Literature Review:** Besides the above heuristics, feel free to propose and test any other signal that you believe could be predictive of a fitting company, based on the training data and the three literature sources in the folder (`On the Nature of Revenue.pdf`, `The Arc of a 10x outcome.pdf`, `2020-Search-Fund-Primer.pdf`)