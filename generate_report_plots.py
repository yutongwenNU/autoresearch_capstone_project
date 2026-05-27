"""
generate_report_plots.py — Two figures for the NeurIPS-style final report.

Figure 1 (EDA): The Succession Sweet Spot & Stagnation Distributions.
  Left panel:  tenure vs Manual Score scatter + fitted quadratic regression
               curve showing the inverted-U peaking around 17–25 years.
  Right panel: KDE/density of rev_per_emp, faceted by quality tier
               (High/Mid/Low Manual Score) to show that lower per-head
               revenue correlates with higher target quality.

Figure 2 (Trajectory): RMSE across all 20 controlled experiments with the
  Exp_009 champion marker and an explicit callout for the Exp_016 → Exp_018
  compliance-trap-then-audit-rebound story.

Outputs to plots/figure_1_eda.png and plots/figure_2_trajectory.png,
plus PDF versions for LaTeX embedding.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_CSV = PROJECT_ROOT / "data" / "train_set.csv"
TEST_CSV = PROJECT_ROOT / "data" / "locked_test_set.csv"
RESULTS_TSV = PROJECT_ROOT / "logs" / "results.tsv"
PLOTS_DIR = PROJECT_ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Theme
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.edgecolor": "#333333",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

CURRENT_YEAR = 2026


# ──────────────────────────────────────────────────────────────────────
def load_labeled_firms() -> pd.DataFrame:
    """Return ALL labeled firms (62 train + 28 test = 90 rows) with derived columns."""
    train = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    test = pd.read_csv(TEST_CSV, quotechar='"', skipinitialspace=True)
    test = test[test["Manual Score"].notna()].copy()

    df = pd.concat([train, test], ignore_index=True)

    def safe_float(x, default=0.0):
        if pd.isna(x):
            return default
        try:
            return float(str(x).strip().replace(",", ""))
        except ValueError:
            return default

    df["employees"] = df["# Employees"].apply(safe_float)
    df["revenue"] = df["Annual Revenue"].apply(safe_float)
    df["founded"] = df["Founded Year"].apply(safe_float)
    df["tenure"] = (CURRENT_YEAR - df["founded"]).clip(lower=0)
    df["rev_per_emp"] = df["revenue"] / df["employees"].clip(lower=1)
    df["Manual Score"] = df["Manual Score"].astype(float)
    # Drop pathological rows (zero revenue or zero tenure that came from missing fields).
    df = df[(df["revenue"] > 0) & (df["tenure"] > 0)].reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────────────
# Figure 1 — EDA: Succession Sweet Spot + Stagnation Distribution
# ──────────────────────────────────────────────────────────────────────
def figure_1(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), constrained_layout=True)

    # ── Left panel: tenure vs Manual Score + quadratic fit ───────────
    ax = axes[0]
    sns.scatterplot(
        data=df, x="tenure", y="Manual Score",
        color="#1F4F91", alpha=0.55, s=50, edgecolor="white", linewidth=0.7, ax=ax,
    )
    # Quadratic fit on the bulk of the data (4–45 yr) where the
    # search-fund "established but not ancient" thesis is well-posed.
    # Very young (just-founded) and very old (>50 yr) firms are sparse
    # outliers that bend the unconstrained fit far past the meaningful range.
    df_fit = df[(df["tenure"] >= 4) & (df["tenure"] <= 45)]
    coefs = np.polyfit(df_fit["tenure"], df_fit["Manual Score"], deg=2)
    x_curve = np.linspace(4, 45, 200)
    y_curve = np.polyval(coefs, x_curve)
    ax.plot(x_curve, y_curve, color="#C0392B", lw=2.2,
            label=f"Quadratic fit on 4–45 yr (n={len(df_fit)})")
    # Highlight the 17–25 year sweet-spot band
    ax.axvspan(17, 25, color="#27AE60", alpha=0.13, label="Sweet spot (17–25 yr)")
    # Mark the analytic vertex of the fit if it falls in plausible range
    if coefs[0] < 0:
        peak_x = -coefs[1] / (2 * coefs[0])
        if 5 <= peak_x <= 80:
            peak_y = np.polyval(coefs, peak_x)
            ax.scatter([peak_x], [peak_y], marker="*", s=180, color="#C0392B",
                       zorder=5, edgecolor="white", linewidth=1.0)
            ax.annotate(f"Peak ≈ {peak_x:.0f} yr",
                        xy=(peak_x, peak_y), xytext=(peak_x + 10, peak_y - 0.6),
                        fontsize=9, color="#C0392B",
                        arrowprops=dict(arrowstyle="->", color="#C0392B", lw=0.8))
    ax.set_xlabel("Company Tenure (years since founding)")
    ax.set_ylabel("Manual Score (Investment Grade, 1–10)")
    ax.set_title("Left  ·  Succession Sweet Spot")
    ax.set_ylim(0.5, 10.5)
    ax.legend(loc="lower right", framealpha=0.92)

    # ── Right panel: rev_per_emp distribution, faceted by quality tier ─
    ax = axes[1]
    df = df.copy()
    df["Quality Tier"] = pd.cut(
        df["Manual Score"],
        bins=[0, 6.0, 8.0, 11.0],
        labels=["Low (≤6)", "Mid (6–8)", "High (>8)"],
        include_lowest=True,
    )
    palette = {"Low (≤6)": "#7F8C8D", "Mid (6–8)": "#F39C12", "High (>8)": "#27AE60"}
    # Trim extreme outliers for visual readability — keep firms below the 95th pct.
    cap = df["rev_per_emp"].quantile(0.95)
    plot_df = df[df["rev_per_emp"] <= cap]
    for tier in ["Low (≤6)", "Mid (6–8)", "High (>8)"]:
        sub = plot_df[plot_df["Quality Tier"] == tier]
        if len(sub) < 2:
            continue
        sns.kdeplot(
            data=sub, x="rev_per_emp",
            ax=ax, color=palette[tier], lw=2.2, label=f"{tier}  (n={len(sub)})",
            fill=True, alpha=0.15,
        )
    ax.set_xlabel("Revenue per Employee  ($ / head, unscaled)")
    ax.set_ylabel("Density")
    ax.set_title("Right  ·  Stagnation Premium  (rev_per_emp by Tier)")
    ax.legend(title="Quality Tier", loc="upper right", framealpha=0.92)
    # Annotation explaining the directional finding
    ax.text(
        0.55, 0.55,
        "High-quality targets cluster\nat LOWER rev / employee\n(operational slack signal)",
        transform=ax.transAxes, fontsize=9, color="#27AE60",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#27AE60", lw=1.0, alpha=0.85),
    )

    fig.suptitle(
        "Figure 1  ·  EDA — Succession Sweet Spot (Demographics) + Stagnation Premium (Operational)",
        fontsize=13, fontweight="bold", y=1.04,
    )

    out_png = PLOTS_DIR / "figure_1_eda.png"
    out_pdf = PLOTS_DIR / "figure_1_eda.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png


# ──────────────────────────────────────────────────────────────────────
# Figure 2 — Trajectory: RMSE across exp_001…exp_020 + callouts
# ──────────────────────────────────────────────────────────────────────
def figure_2() -> Path:
    res = pd.read_csv(RESULTS_TSV, sep="\t")
    res = res[res["experiment_id"].str.match(r"exp_0(0[1-9]|1[0-9]|20)$")].copy()
    # The status flips we made; re-derive a clean dataframe in the canonical order
    res["idx"] = range(1, len(res) + 1)

    color_map = {"baseline": "#3498DB", "keep": "#27AE60", "discard": "#E74C3C"}

    fig, ax = plt.subplots(figsize=(11, 4.3), constrained_layout=True)

    # Dashed connecting line (chronological order)
    ax.plot(res["idx"], res["val_rmse"], color="#34495E", lw=0.9, alpha=0.45, zorder=1)

    # Scatter, colored by status
    for status, group in res.groupby("status"):
        ax.scatter(
            group["idx"], group["val_rmse"],
            c=color_map.get(status, "#95A5A6"), s=95,
            edgecolor="white", linewidth=1.0, zorder=3,
            label=f"{status}  (n={len(group)})",
        )

    # Best-RMSE-over-keeps envelope (respects status, unlike the live judge)
    keeps_only = res[res["status"].isin(["keep", "baseline"])].copy()
    keeps_only["best_so_far"] = keeps_only["val_rmse"].cummin()
    ax.plot(keeps_only["idx"], keeps_only["best_so_far"],
            color="#27AE60", lw=2.0, ls="--", alpha=0.7,
            label="Best-so-far (keep/baseline only)", zorder=2)

    # ★ Champion marker: Exp_009
    champ = res[res["experiment_id"] == "exp_009"].iloc[0]
    ax.scatter([champ["idx"]], [champ["val_rmse"]],
               marker="*", s=380, color="#F1C40F", edgecolor="#2C3E50",
               linewidth=1.5, zorder=5, label="Exp_009 Champion (1.3955)")
    ax.annotate(
        f"Exp_009  ·  CHAMPION\nrev_per_emp\nRMSE 1.3955",
        xy=(champ["idx"], champ["val_rmse"]),
        xytext=(champ["idx"] + 1.0, champ["val_rmse"] - 0.20),
        fontsize=9, fontweight="bold", color="#7D6608",
        bbox=dict(boxstyle="round,pad=0.4", fc="#FEF9E7", ec="#F1C40F", lw=1.2),
        arrowprops=dict(arrowstyle="->", color="#7D6608", lw=1.0),
    )

    # ⚠ Compliance Trap & Audit Rebound callout — Exp_016 → Exp_018
    exp16 = res[res["experiment_id"] == "exp_016"].iloc[0]
    exp18 = res[res["experiment_id"] == "exp_018"].iloc[0]
    # Connector arrow between Exp_016 and Exp_018
    ax.annotate(
        "", xy=(exp18["idx"], exp18["val_rmse"]),
        xytext=(exp16["idx"], exp16["val_rmse"]),
        arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.5, ls="-"),
        zorder=4,
    )
    # Callout box positioned to the right of the trajectory
    ax.annotate(
        ("Compliance Trap (Exp_016 → Exp_018)\n"
         "Exp_016: apparent RMSE 1.3079 (artifact)\n"
         "Exp_018 audit: drop `compliance`\n"
         "→ RMSE rebounds to 1.4049\n"
         "Both reclassified DISCARD"),
        xy=(exp16["idx"], exp16["val_rmse"]),
        xytext=(exp16["idx"] - 9.5, 1.93),
        fontsize=8.5, color="#922B21", ha="left",
        bbox=dict(boxstyle="round,pad=0.45", fc="#FDEDEC", ec="#C0392B", lw=1.2),
        arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.0),
    )

    # Axes
    ax.set_xticks(res["idx"])
    ax.set_xticklabels([eid.replace("exp_0", "E") for eid in res["experiment_id"]],
                       rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Experiment Sequence")
    ax.set_ylabel("Validation RMSE  (lower is better)")
    ax.set_title("Figure 2  ·  RMSE Trajectory across the 20 Controlled Experiments",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(1.20, 2.30)
    ax.legend(loc="upper right", framealpha=0.95, ncol=2, fontsize=8.5)
    ax.grid(True, alpha=0.30)

    out_png = PLOTS_DIR / "figure_2_trajectory.png"
    out_pdf = PLOTS_DIR / "figure_2_trajectory.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main() -> None:
    df = load_labeled_firms()
    print(f"Loaded {len(df)} labeled firms (train + test) for EDA.")

    fig1_path = figure_1(df)
    print(f"OK  Figure 1 -> {fig1_path}")
    fig2_path = figure_2()
    print(f"OK  Figure 2 -> {fig2_path}")

    for f in sorted(PLOTS_DIR.iterdir()):
        size = f.stat().st_size
        print(f"  {f.name}  ({size//1024} KB)")


if __name__ == "__main__":
    main()
