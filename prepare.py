from __future__ import annotations

import csv
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

BASE_DIR = Path(__file__).resolve().parent
TRAIN_SET_PATH = BASE_DIR / "data" / "train_set.csv"
RESULTS_PATH = BASE_DIR / "results.tsv"

MANUAL_SCORE_COLUMN = "Manual Score"
COMPANY_COLUMN = "Company Name"
RATIONALE_COLUMN = "Rationale"
KEYWORDS_COLUMN = "Keywords"
TECHNOLOGIES_COLUMN = "Technologies"
DESCRIPTION_COLUMN = "Short Description"
EMPLOYEES_COLUMN = "# Employees"
REVENUE_COLUMN = "Annual Revenue"
FOUNDED_YEAR_COLUMN = "Founded Year"
STATE_COLUMN = "Company State"
INDUSTRY_COLUMN = "Industry"

MIDWEST_STATES = {
    "illinois",
    "indiana",
    "iowa",
    "kansas",
    "michigan",
    "minnesota",
    "missouri",
    "nebraska",
    "north dakota",
    "ohio",
    "south dakota",
    "wisconsin",
}


def normalize_company_name(name: str) -> str:
    """
    Standardizes company names into a 'slug' format (lowercase, alphanumeric).
    This prevents matching failures caused by inconsistent suffixes (Inc, LLC), 
    special characters, or extra whitespace between the manual labels and agent output.
    """
    cleaned = re.sub(r"[^a-z0-9]+", " ", (name or "").strip().lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def safe_float(value: str, default: float = 0.0) -> float:
    """
    Safely converts string-based numeric data (like Annual Revenue or # Employees) to floats.
    It handles common CSV 'noise' like commas, null values, and empty strings, 
    returning a default value rather than raising a ValueError during the RMSE calculation.
    """
    if value is None:
        return default
    text = str(value).strip().replace(",", "")
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def load_training_rows(path: Path) -> List[Dict[str, str]]:
    """
    Loads the manually labeled 'Ground Truth' data from the training CSV. 
    This is to prevent errors when reading CSVs exported from Excel, 
    which often adds a hidden "Byte Order Mark" (BOM) that can break standard Python scripts.
    Uses 'utf-8-sig' to handle Excel-generated BOM issues and verifies that 
    required columns (Company Name and Manual Score) exist before proceeding, 
    ensuring the evaluation script is robust.
    """
    if not path.exists():
        raise FileNotFoundError(f"Training set not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    required = {COMPANY_COLUMN, MANUAL_SCORE_COLUMN}
    missing = required.difference(reader.fieldnames or [])
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Training set is missing required columns: {missing_str}")

    return rows


def load_predictions(path: Path) -> Dict[str, float]:
    """
    Loads predicted scores from a TSV file.
    Each line should be in the format '<score>\\t<company>'.
    Returns a dictionary mapping normalized company names to their predicted scores.
    """
    predictions: Dict[str, float] = {}
    if not path.exists():
        return predictions

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                raise ValueError(
                    f"Invalid results.tsv format on line {line_number}: expected '<score>\\t<company>'"
                )

            score_text = parts[0].strip()
            company = "\t".join(parts[1:]).strip()
            if not company:
                raise ValueError(f"Invalid results.tsv format on line {line_number}: missing company name")

            score = safe_float(score_text, default=float("nan"))
            if math.isnan(score):
                raise ValueError(
                    f"Invalid predicted score on line {line_number} of results.tsv: {score_text!r}"
                )

            predictions[normalize_company_name(company)] = clip_score(score)

    return predictions


def clip_score(score: float) -> float:
    """
    Ensures that predicted scores are within the valid range of 1.0 to 10.0.
    """
    return max(1.0, min(10.0, float(score)))


def contains_any(text: str, phrases: Iterable[str]) -> bool:
    """
    Acts as a flexible keyword scanner to detect qualitative 'Success' or 'Stagnation' signals.
    By converting both the source text and search phrases to lowercase, it ensures 
    robust matching across inconsistent data fields (like 'Short Description' or 'Keywords')
    without requiring exact case alignment.
    """
    lowered = (text or "").lower()
    return any(phrase in lowered for phrase in phrases)


def deterministic_baseline_prediction(row: Dict[str, str], current_year: int = 2026) -> float:
    """
    Generates a deterministic baseline prediction score for a given company row. The determined value is 5.0 to start with.
    The score is influenced by factors such as company size, revenue, tenure, 
    industry, and specific keywords in the company's description.
    The heuristic is designed to reflect the investment intuition derived from the Stanford/Yale search fund primers/literature (attached in Github repo)
    """
    score = 5.0

    employees = safe_float(row.get(EMPLOYEES_COLUMN, ""))
    revenue = safe_float(row.get(REVENUE_COLUMN, ""))
    founded_year = int(safe_float(row.get(FOUNDED_YEAR_COLUMN, ""), default=0.0) or 0)
    tenure = current_year - founded_year if founded_year > 0 else 0

    state = (row.get(STATE_COLUMN, "") or "").strip().lower()
    industry = (row.get(INDUSTRY_COLUMN, "") or "").strip().lower()
    rationale = row.get(RATIONALE_COLUMN, "") or ""
    keywords = row.get(KEYWORDS_COLUMN, "") or ""
    technologies = row.get(TECHNOLOGIES_COLUMN, "") or ""
    description = row.get(DESCRIPTION_COLUMN, "") or ""

    combined_text = " ".join([rationale, keywords, technologies, description]).lower()

    # Geography / scope fit
    if state in MIDWEST_STATES:
        score += 0.5

    # Size sweet spot from the thesis baseline
    if 10 <= employees <= 30:
        score += 1.5
    elif 31 <= employees <= 40:
        score += 0.25
    else:
        score -= 0.75

    # Established tenure / succession potential
    if tenure >= 25:
        score += 1.5
    elif tenure >= 15:
        score += 1.0
    elif tenure >= 10:
        score += 0.25
    else:
        score -= 0.5

    # Recurring revenue / switching cost style signals
    if revenue >= 2_000_000:
        score += 1.0
    if revenue >= 5_000_000:
        score += 0.5

    if contains_any(combined_text, [
        "managed services",
        "managed it services",
        "managed it",
        "recurring",
        "mrr",
        "help desk",
        "monitoring",
        "backup",
        "business continuity",
        "compliance",
        "vendor management",
        "outsourcing",
    ]):
        score += 0.75

    if contains_any(combined_text, [
        "proprietary",
        "platform",
        "connect product",
        "switching cost",
        "sticky",
    ]):
        score += 0.75

    # Stable-but-stagnant upside signals
    if contains_any(combined_text, [
        "outdated",
        "legacy",
        "onsite support",
        "voip",
        "hardware",
        "hardware/software sales",
        "hardwaresoftware sales",
        "stagnation",
        "under-optimized",
        "stable but stagnant",
    ]):
        score += 1.0

    # Penalize firms that look too modern / frontier-forward for the thesis
    # We want firms that are not-to-modern so that a new, young owner can take over with his latest expertise to transform the business
    if contains_any(combined_text, [
        "ai-integrated",
        "ai-forward",
        "advanced ai",
        "machine learning",
        "cutting-edge ai",
    ]):
        score -= 0.75

    # Reward MSP-like positioning, lightly penalize non-fit industries
    if contains_any(industry, ["information technology", "computer networking", "managed", "it services"]):
        score += 0.5

    return clip_score(round(score, 4))


def collect_scored_pairs(
    training_rows: List[Dict[str, str]], predictions: Dict[str, float]
) -> List[Tuple[float, float, str]]:
    """
    This is where the manual "vibe check" and the agent's code finally meet to be graded against each other.
    Aligns human-labeled "Manual Scores" with the agent's predicted scores.
    This function performs a data join between the training set and the results file.
    If the agent fails to provide a prediction for a specific company, it defaults 
    to a deterministic baseline to ensure the final RMSE remains a complete and 
    honest reflection of the entire training set. This alignment is critical for 
    measuring 'Human-AI Alignment'.
    """
    pairs: List[Tuple[float, float, str]] = []

    for row in training_rows:
        company = row.get(COMPANY_COLUMN, "")
        company_key = normalize_company_name(company)
        actual = clip_score(safe_float(row.get(MANUAL_SCORE_COLUMN, "")))
        predicted = predictions.get(company_key)
        if predicted is None:
            predicted = deterministic_baseline_prediction(row)
        pairs.append((actual, predicted, company))

    return pairs


def rmse(pairs: Iterable[Tuple[float, float, str]]) -> float:
    """"
    This represents the mathematical distance between human intuition (Manual Score) 
    and agent logic (Predicted Score). By squaring the errors, we heavily penalize 
    large 'misses' on high-value targets, forcing the agent to align strictly with 
    the search fund investment thesis.
    """
    pairs = list(pairs)
    if not pairs:
        raise ValueError("No scored pairs were available for RMSE calculation")
    mse = sum((actual - predicted) ** 2 for actual, predicted, _ in pairs) / len(pairs)
    return math.sqrt(mse)

# Below, the prepare.py script exits with printing the RMSE score in the terminal

def main() -> int:
    training_rows = load_training_rows(TRAIN_SET_PATH)
    predictions = load_predictions(RESULTS_PATH)
    pairs = collect_scored_pairs(training_rows, predictions)
    score = rmse(pairs)
    print(f"{score:.6f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
