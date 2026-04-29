"""
Worker (research.py) — Ridge regression with engineered features
plus a "Management Depth" feature scraped from each firm's website.

Validation strategy: 5-fold OOF predictions via cross_val_predict, so the
RMSE the Judge computes is an honest validation error, not a training fit.

Search-fund thesis (per the Stanford/Yale primers): a target with deep
founder tenure but a thin/absent visible management team is high "Succession
Gap" — i.e., a desirable acquisition. Ridge will learn the sign of the
mgmt_depth coefficient empirically; we expect it to be negative (more
visible management → lower Manual Score).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

INPUT_CSV = Path("data/train_set.csv")
OUTPUT_TSV = Path("results.tsv")
SCRAPE_CACHE = Path("logs/scrape_cache.json")
CURRENT_YEAR = 2026

MIDWEST = {
    "illinois", "indiana", "iowa", "kansas", "michigan", "minnesota",
    "missouri", "nebraska", "north dakota", "ohio", "south dakota", "wisconsin",
}

RECURRING_KW = [
    "managed services", "managed it", "recurring", "mrr", "help desk",
    "monitoring", "backup", "compliance", "business continuity",
]
STAGNATION_KW = [
    "legacy", "outdated", "onsite support", "voip", "hardware", "stagnation",
    "under-optimized",
]
MODERN_AI_KW = [
    "ai-integrated", "ai-forward", "advanced ai", "machine learning",
    "cutting-edge ai",
]

# Distinct role-title categories. We count how many of these patterns appear
# at least once across the scraped pages (not raw occurrences) — so a page
# spamming "CEO CEO CEO" still contributes 1 unit, not 3.
MGMT_PATTERNS = [
    r"\bceo\b", r"\bcto\b", r"\bcfo\b", r"\bcoo\b", r"\bcio\b",
    r"\bvice president\b", r"\bvp\b", r"\bpresident\b",
    r"\bdirector of\b", r"\bchief\s+\w+\s+officer\b",
    r"leadership team", r"executive team", r"management team",
    r"meet the team", r"our team", r"our leadership",
]

TEAM_PATHS = ["about", "about-us", "team", "our-team", "leadership", "management", "company"]

USER_AGENT = "Mozilla/5.0 (compatible; capstone-research/1.0)"
FETCH_TIMEOUT = 4  # seconds per HTTP request


def safe_float(x, default=0.0):
    if x is None:
        return default
    s = str(x).strip().replace(",", "")
    if not s:
        return default
    try:
        return float(s)
    except ValueError:
        return default


def count_kw(text: str, kw_list) -> int:
    t = (text or "").lower()
    return sum(1 for k in kw_list if k in t)


def normalize_url(raw: str) -> str | None:
    if not raw:
        return None
    u = raw.strip()
    if not u:
        return None
    if not u.startswith(("http://", "https://")):
        u = "http://" + u
    return u


def strip_html(html: str) -> str:
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return text.lower()


def fetch(url: str) -> str:
    try:
        r = requests.get(
            url,
            timeout=FETCH_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
            allow_redirects=True,
        )
        if r.status_code == 200 and r.text:
            return r.text
    except requests.RequestException:
        return ""
    return ""


def scrape_management_depth(url: str | None, cache: dict) -> dict:
    """Returns {'depth': int, 'pages_fetched': int, 'error': str|None}."""
    if not url:
        return {"depth": 0, "pages_fetched": 0, "error": "no_url"}
    if url in cache:
        return cache[url]

    home_html = fetch(url)
    if not home_html:
        result = {"depth": 0, "pages_fetched": 0, "error": "homepage_fetch_failed"}
        cache[url] = result
        return result

    pages_text = strip_html(home_html)
    pages_fetched = 1

    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    for path in TEAM_PATHS:
        candidate = urljoin(base + "/", path)
        page_html = fetch(candidate)
        if page_html:
            pages_text += " " + strip_html(page_html)
            pages_fetched += 1
            break  # one team-style page is enough

    depth = sum(1 for pat in MGMT_PATTERNS if re.search(pat, pages_text))
    result = {"depth": depth, "pages_fetched": pages_fetched, "error": None}
    cache[url] = result
    return result


def load_cache() -> dict:
    if SCRAPE_CACHE.exists():
        try:
            return json.loads(SCRAPE_CACHE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_cache(cache: dict) -> None:
    SCRAPE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SCRAPE_CACHE.write_text(json.dumps(cache, indent=2))


def featurize(df: pd.DataFrame, mgmt_depth: pd.Series) -> pd.DataFrame:
    employees = df["# Employees"].apply(safe_float)
    revenue = df["Annual Revenue"].apply(safe_float)
    founded = df["Founded Year"].apply(safe_float)
    tenure = (CURRENT_YEAR - founded).clip(lower=0)

    state = df["Company State"].fillna("").str.strip().str.lower()
    in_midwest = state.isin(MIDWEST).astype(int)

    text = (
        df.get("Rationale", pd.Series([""] * len(df))).fillna("").astype(str) + " "
        + df.get("Keywords", pd.Series([""] * len(df))).fillna("").astype(str) + " "
        + df.get("Technologies", pd.Series([""] * len(df))).fillna("").astype(str) + " "
        + df.get("Short Description", pd.Series([""] * len(df))).fillna("").astype(str)
    )
    recurring = text.apply(lambda t: count_kw(t, RECURRING_KW))
    stagnation = text.apply(lambda t: count_kw(t, STAGNATION_KW))
    modern_ai = text.apply(lambda t: count_kw(t, MODERN_AI_KW))

    sweet_spot_emp = ((employees >= 10) & (employees <= 30)).astype(int)

    return pd.DataFrame({
        "log_employees": np.log1p(employees),
        "log_revenue": np.log1p(revenue),
        "tenure": tenure,
        "tenure_sq": tenure ** 2,
        "sweet_spot_emp": sweet_spot_emp,
        "in_midwest": in_midwest,
        "recurring_kw": recurring,
        "stagnation_kw": stagnation,
        "modern_ai_kw": modern_ai,
        "mgmt_depth": mgmt_depth.values,
    })


def build_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])


def main():
    if not INPUT_CSV.exists():
        return
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

    cache = load_cache()
    depths = []
    failures = 0
    for _, row in df.iterrows():
        url = normalize_url(row.get("Website", ""))
        result = scrape_management_depth(url, cache)
        depths.append(result["depth"])
        if result["error"]:
            failures += 1
    save_cache(cache)
    print(f"Scrape complete: {len(df) - failures}/{len(df)} firms reachable, {failures} failures.")

    X = featurize(df, pd.Series(depths)).values
    y = df["Manual Score"].astype(float).values

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    preds = cross_val_predict(build_model(), X, y, cv=cv)
    preds = np.clip(preds, 1.0, 10.0)

    # Diagnostic: standardized Ridge coefficients from a model fit on all
    # data (just for logging — predictions above are OOF and label-honest).
    full_model = build_model().fit(X, y)
    coefs = full_model.named_steps["ridge"].coef_
    feature_names = list(featurize(df, pd.Series(depths)).columns)
    coef_pairs = sorted(zip(feature_names, coefs), key=lambda x: -abs(x[1]))
    print("Ridge coefficients (standardized):")
    for name, c in coef_pairs:
        print(f"  {name:>16s}: {c:+.4f}")

    with open(OUTPUT_TSV, "w") as f:
        f.write("Predicted Score\tCompany Name\n")
        for score, name in zip(preds, df["Company Name"].values):
            f.write(f"{round(float(score), 4)}\t{name}\n")
    print(f"Ridge OOF complete. Generated scores for {len(df)} firms.")


if __name__ == "__main__":
    main()
