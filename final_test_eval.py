"""
final_test_eval.py — Week 7 single-pass holdout-set evaluation.

This is a one-shot inference script, NOT a Worker / Judge experiment. It does
not modify eval/prepare.py (FROZEN), does not call run_experiment.py, and does
not append to logs/results.tsv. The SHA-256 lock on the Judge is untouched.

Pipeline:
  1. Fit the Exp_009 champion pipeline on the FULL 62-firm training set.
  2. Predict on the 28 labeled holdout firms in data/locked_test_set.csv.
  3. Apply the same post-processing as Exp_009: clip [1.0, 10.0] + round to 0.5 grid.
  4. Compute Test RMSE and Test R² against the held-out Manual Score labels.
  5. Write predictions to logs/final_test_predictions.tsv.

Featurization, scaler, and regressor are imported directly from model.py, which
must be byte-identical to logs/Snapshot_model_Exp_009.py before this script runs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from model import (
    build_model,
    featurize,
    load_cache,
    normalize_url,
    save_cache,
    scrape_management_depth,
)

PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_CSV = PROJECT_ROOT / "data" / "train_set.csv"
TEST_CSV = PROJECT_ROOT / "data" / "locked_test_set.csv"
OUTPUT_TSV = PROJECT_ROOT  / "final_test_predictions.tsv"


def featurize_with_scrape(df: pd.DataFrame, cache: dict) -> tuple[pd.DataFrame, int]:
    depths = []
    failures = 0
    for _, row in df.iterrows():
        url = normalize_url(row.get("Website", ""))
        result = scrape_management_depth(url, cache)
        depths.append(result["depth"])
        if result["error"]:
            failures += 1
    return featurize(df, pd.Series(depths)), failures


def main():
    # 1. Training set (62 firms, full labels) — used to FIT the pipeline.
    train_df = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    cache = load_cache()
    X_train_df, train_fail = featurize_with_scrape(train_df, cache)
    X_train = X_train_df.values
    y_train = train_df["Manual Score"].astype(float).values
    print(f"Train: {len(train_df)} firms loaded, {train_fail} scrape failures")

    # 2. Test set — filter to rows with a Manual Score label (the 28 holdout firms).
    test_df_all = pd.read_csv(TEST_CSV, quotechar='"', skipinitialspace=True)
    if "Manual Score" not in test_df_all.columns:
        raise RuntimeError("locked_test_set.csv missing Manual Score column — cannot compute Test RMSE.")
    test_df = test_df_all[test_df_all["Manual Score"].notna()].reset_index(drop=True)
    print(f"Test:  {len(test_df)} labeled firms (filtered from {len(test_df_all)} rows total)")

    X_test_df, test_fail = featurize_with_scrape(test_df, cache)
    save_cache(cache)
    print(f"Test scrape: {test_fail} failures (cache: logs/scrape_cache.json)")

    y_test = test_df["Manual Score"].astype(float).values
    X_test = X_test_df.values

    # 3. Fit champion pipeline on ALL training data, predict on holdout.
    #    No cross-validation here — this is the final inference pass.
    model = build_model()
    model.fit(X_train, y_train)
    preds_continuous = model.predict(X_test)
    preds_clipped = np.clip(preds_continuous, 1.0, 10.0)
    preds = np.round(preds_clipped * 2) / 2  # 0.5-grid rounding per Exp_009 protocol

    # 4. Metrics.
    test_rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    test_r2 = float(r2_score(y_test, preds))
    print()
    print("=" * 50)
    print(" FINAL TEST METRICS (Holdout, 28 firms)")
    print("=" * 50)
    print(f" Test RMSE: {test_rmse:.4f}")
    print(f" Test R^2:  {test_r2:.4f}")
    print("=" * 50)

    # 5. Write predictions TSV.
    OUTPUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TSV, "w") as f:
        f.write("Predicted Score\tCompany Name\n")
        for score, name in zip(preds, test_df["Company Name"].values):
            f.write(f"{round(float(score), 4)}\t{name}\n")
    print(f"\nPredictions written: {OUTPUT_TSV}")

    # 6. Diagnostic: Ridge coefficients from full-data fit, plus per-firm errors.
    coefs = model.named_steps["ridge"].coef_
    feature_names = list(X_train_df.columns)
    print("\nFitted Ridge coefficients (standardized, trained on all 62 firms):")
    for name, c in sorted(zip(feature_names, coefs), key=lambda x: -abs(x[1])):
        print(f"  {name:>16s}: {c:+.4f}")

    abs_errors = np.abs(preds - y_test)
    print(f"\nAbsolute error distribution (n=28):")
    print(f"  min={abs_errors.min():.2f}, median={np.median(abs_errors):.2f}, max={abs_errors.max():.2f}")
    print(f"  firms within 1.0 point: {int((abs_errors <= 1.0).sum())}/{len(abs_errors)}")
    print(f"  firms within 1.5 points: {int((abs_errors <= 1.5).sum())}/{len(abs_errors)}")
    print(f"  firms within 2.0 points: {int((abs_errors <= 2.0).sum())}/{len(abs_errors)}")

    # Save metrics summary to a small companion file for the ledger to ingest.
    summary_path = PROJECT_ROOT /  "_final_test_metrics.txt"
    with open(summary_path, "w") as f:
        f.write(f"test_rmse\t{test_rmse:.6f}\n")
        f.write(f"test_r2\t{test_r2:.6f}\n")
        f.write(f"n_test\t{len(test_df)}\n")
        f.write(f"abs_err_min\t{float(abs_errors.min()):.4f}\n")
        f.write(f"abs_err_median\t{float(np.median(abs_errors)):.4f}\n")
        f.write(f"abs_err_max\t{float(abs_errors.max()):.4f}\n")
        f.write(f"within_1.0\t{int((abs_errors <= 1.0).sum())}\n")
        f.write(f"within_1.5\t{int((abs_errors <= 1.5).sum())}\n")
        f.write(f"within_2.0\t{int((abs_errors <= 2.0).sum())}\n")
        for name, c in sorted(zip(feature_names, coefs), key=lambda x: -abs(x[1])):
            f.write(f"coef\t{name}\t{c:+.6f}\n")


if __name__ == "__main__":
    main()
