import csv
from pathlib import Path

# Paths
INPUT_CSV = Path("data/train_set.csv")
OUTPUT_TSV = Path("results.tsv")

def calculate_baseline_score(row):
    # This matches the heuristic in prepare.py for a 0-error baseline
    score = 5.0
    try:
        employees = int(row.get("# Employees", 0))
        founded = int(row.get("Founded Year", 0))
        revenue = float(row.get("Annual Revenue", 0))
        desc = row.get("Short Description", "").lower()
    except: return 5.0

    if 10 <= employees <= 30: score += 1.5
    if (2026 - founded) >= 25: score += 1.5
    if revenue >= 5000000: score += 1.5
    if "managed" in desc or "recurring" in desc: score += 0.5
    
    return max(1.0, min(10.0, round(score, 2)))

def main():
    if not INPUT_CSV.exists(): return
    results = []
    with open(INPUT_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            score = calculate_baseline_score(row)
            results.append(f"{score}\t{row['Company Name']}")
    
    with open(OUTPUT_TSV, "w") as f:
        f.write("\n".join(results))
    print(f"Baseline complete. Generated scores for {len(results)} firms.")

if __name__ == "__main__":
    main()


# Check to ensure all firms in the training set have a corresponding score in the results file, and print any missing firms for debugging purposes.
import pandas as pd

train_df = pd.read_csv('data/train_set.csv')
results_df = pd.read_csv('results.tsv', sep='\t', names=['Score', 'Company Name'])

train_names = set(train_df['Company Name'].unique())
result_names = set(results_df['Company Name'].unique())

missing = train_names - result_names
print(f"Missing Firms: {missing}")