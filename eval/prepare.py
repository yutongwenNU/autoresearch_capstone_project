"""
FROZEN - Judge Script for MSP Sourcing.
Handles data loading, RMSE calculation, logging, and plotting.
"""

import os
import sys
import csv
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# --- Configuration ---
RANDOM_SEED = 42
VAL_FRACTION = 0.3  # 70/30 split as per Week 2 deliverables

# Paths relative to this script (inside eval/)
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "train_set.csv"  # Manual Ground Truth
PREDICTIONS_PATH = PROJECT_ROOT / "results.tsv"      # Output from research.py
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_HISTORY = LOGS_DIR / "results.tsv"
PLOT_PATH = LOGS_DIR / "performance.png"

def normalize_name(name):
    """Standardizes company names for reliable joining."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())

def load_data():
    """Loads the manual training set and splits it for validation."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH, quotechar='"', skipinitialspace=True)
    # 70/30 Split matches the baseline reproducibility gate
    train_df, val_df = train_test_split(
        df, test_size=VAL_FRACTION, random_state=RANDOM_SEED
    )
    return val_df

def evaluate():
    """Compares agent predictions in root results.tsv to manual ground truth."""
    val_df = load_data()
    
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError("Agent output 'results.tsv' not found in root.")
    
    pred_df = pd.read_csv(PREDICTIONS_PATH, sep='\t')
    
    # Normalize names to join dataframes
    val_df['norm_name'] = val_df['Company Name'].apply(normalize_name)
    pred_df['norm_name'] = pred_df['Company Name'].apply(normalize_name)
    
    # Join on normalized name
    merged = pd.merge(
        val_df, 
        pred_df[['norm_name', 'Predicted Score']], 
        on='norm_name', 
        how='inner'
    )
    
    if merged.empty:
        raise ValueError("Could not match any companies between manual labels and agent output.")

    y_true = merged['Manual Score'].values
    y_pred = merged['Predicted Score'].values
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return float(rmse), float(r2)

def log_result(description, status, rmse, r2):
    """Appends the experiment result to the long-term log."""
    LOGS_DIR.mkdir(exist_ok=True)
    file_exists = RESULTS_HISTORY.exists()
    
    with open(RESULTS_HISTORY, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if not file_exists:
            writer.writerow(["experiment_id", "val_rmse", "val_r2", "status", "description"])
        
        # Determine next experiment ID
        exp_num = sum(1 for _ in open(RESULTS_HISTORY)) if file_exists else 1
        exp_id = f"exp_{exp_num:03d}"
        
        writer.writerow([exp_id, f"{rmse:.6f}", f"{r2:.4f}", status, description])

def plot_results():
    """Generates the performance trajectory chart."""
    if not RESULTS_HISTORY.exists():
        return
        
    df = pd.read_csv(RESULTS_HISTORY, sep="\t")
    plt.figure(figsize=(10, 6))
    
    # Color mapping for statuses
    colors = {"baseline": "#3498db", "keep": "#2ecc71", "discard": "#e74c3c"}
    
    for status, color in colors.items():
        subset = df[df['status'] == status]
        plt.scatter(subset.index, subset['val_rmse'], c=color, label=status, s=100, edgecolors='black')

    # Best-so-far line
    plt.plot(df.index, df['val_rmse'].cummin(), color='#2ecc71', linestyle='--', alpha=0.6, label="Best RMSE")
    
    plt.title("MSP Sourcing: Experiment Trajectory", fontsize=14)
    plt.ylabel("Validation RMSE (Lower is Better)", fontsize=12)
    plt.xlabel("Experiment Index", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(PLOT_PATH)
    print(f"Update: Performance plot saved to {PLOT_PATH}")

def main():
    # Handle sys.argv from run_experiment.py
    # sys.argv[1] = description, sys.argv[2] = status (e.g. --keep)
    description = sys.argv[1] if len(sys.argv) > 1 else "unlabeled_run"
    status_raw = sys.argv[2] if len(sys.argv) > 2 else "--keep"
    status = status_raw.replace("--", "")

    try:
        rmse, r2 = evaluate()
        log_result(description, status, rmse, r2)
        plot_results()
        print(f"Evaluation Complete | RMSE: {rmse:.4f} | Status: {status}")
    except Exception as e:
        print(f"Evaluation Failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()