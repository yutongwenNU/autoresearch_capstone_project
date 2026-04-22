# Experiment Log: Auto-Private-Equity Search Engine
**Date:** 2026-04-21

## Experiment: Baseline V0 (Reproducibility Gate)

### Configuration
* **Worker:** `research.py` (V0 - Simple Heuristic)
* **Judge:** `prepare.py` (FROZEN - RMSE Metric)
* **Training Set:** 63 manually labeled IT MSPs from the U.S. Midwest.

### Result
* **Initial RMSE Score:** 2.316665
* **Note:** This score represents the alignment between the baseline Python heuristic and human investment intuition derived from the Stanford/Yale search fund literature.

### Analysis
* This baseline creates the "Ground Truth" for the project. 
* It confirms that the evaluator returns a stable output and that the data pipeline is fully functional.