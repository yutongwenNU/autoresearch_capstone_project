# Project Title Auto-Private-Equity Search Engine (MSP firms sourcing)

## Overview
This project automates the identification of "stable but stagnant" IT Managed Service Providers (MSPs) in the U.S. Midwest for Search Fund acquisition.

## Quick Start (For Grader)
To reproduce the baseline results, follow these steps:

1. **Environment Setup:**
   * Clone the repository.
   * Install requirements: `pip install pandas` (and any other necessary libraries).
   * Ensure your `.env` file contains valid API keys for Apollo.io. (See `.env.example`). (note that the current baseline does not make API calls, but this is required for future iterations).

2. **Run the Baseline:**
   * Execute: `python research.py`
   * This generates `results.tsv` based on the 62 firms in the training set.

3. **Verify the Metric:**
   * Execute: `python prepare.py`
   * This will output the **RMSE** score comparing the baseline code against manual human labels.

## Project Structure
* `/data/train_set.csv`: Graded training data (62 firms).
* `prepare.py`: **FROZEN** evaluation instrument (RMSE).
* `research.py`: **EDITABLE** worker script for the agent.
* `results.tsv`: The current output of the search agent.