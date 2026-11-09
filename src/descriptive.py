"""Step 2 - Descriptive analytics: who answered, and what do they say?

Before any model, we describe the sample: demographics, spend, how often
people compare prices, which pain points and features are most common, and
how adoption intent varies across groups. These answer "know your customer"
questions directly and also tell us which variables are worth modelling.

Run:  python src/descriptive.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
from utils import PALETTE, banner, save_results, save_table, savefig


def adoption_by(df: pd.DataFrame, col: str, order=None) -> pd.DataFrame:
    t = (df.groupby(col)
           .agg(respondents=("is_adopter", "size"),
                adoption_rate=("is_adopter", "mean"),
                avg_spend_aed=("monthly_grocery_spend_aed", "mean"),
                avg_wtp_aed=("wtp_aed_month", "mean"))
           .round(3))
    if order is not None:
        t = t.reindex([o for o in order if o in t.index])
    return t.reset_index()


def main() -> dict:
    banner("STEP 2  Descriptive analytics")
    df = pd.read_csv(C.CLEAN_FILE)

    # --- Summary statistics -------------------------------------------------
    num_cols = ["age", "household_size", "monthly_grocery_spend_aed", "spend_per_person_aed",
                "trips_per_week", "stores_used_regularly", "minutes_comparing_week",
                "expected_saving_aed_month", "wtp_aed_month", "nps_concept",
                "challenge_count", "feature_count"]
    summary = df[num_cols].describe(percentiles=[.25, .5, .75]).T.round(1)
    save_table(summary, "descriptive_numeric_summary", index=True)

    # --- Intent & NPS -------------------------------------------------------
    intent = df["intent_download"].value_counts(normalize=True).reindex(C.INTENT_LEVELS[::-1])
    promoters = (df["nps_concept"] >= 9).mean()
    detractors = (df["nps_concept"] <= 6).mean()
    nps = 100 * (promoters - detractors)

    # --- Segment cuts -------------------------------------------------------
    cuts = {
        "income_band_aed": adoption_by(df, "income_band_aed", C.INCOME_BANDS),
        "emirate": adoption_by(df, "emirate", C.EMIRATES),
        "age_group": adoption_by(df, "age_group", ["16-24", "25-34", "35-44", "45-54", "55+"]),
        "occupation": adoption_by(df, "occupation", C.OCCUPATIONS),
        "primary_channel": adoption_by(df, "primary_channel", C.CHANNELS),
    }
    for k, t in cuts.items():
        save_table(t, f"adoption_by_{k}")

    # --- Pain points and features -----------------------------------------
    prev = pd.DataFrame({
        "item": C.CHALLENGE_COLS + C.FEATURE_COLS,
        "label": list(C.CHALLENGES.values()) + list(C.FEATURES.values()),
        "type": ["challenge"] * len(C.CHALLENGE_COLS) + ["feature"] * len(C.FEATURE_COLS),
        "all_respondents": df[C.CHALLENGE_COLS + C.FEATURE_COLS].mean().values,
        "adopters": df.loc[df.is_adopter == 1, C.CHALLENGE_COLS + C.FEATURE_COLS].mean().values,
        "non_adopters": df.loc[df.is_adopter == 0, C.CHALLENGE_COLS + C.FEATURE_COLS].mean().values,
    }).round(3)
    prev["adopter_lift"] = (prev["adopters"] / prev["non_adopters"]).round(2)
    save_table(prev, "challenge_feature_prevalence")

    # --- Correlations with adoption ----------------------------------------
    corr_cols = ["age", "household_size", "income_ord", "monthly_grocery_spend_aed",
                 "trips_per_week", "stores_used_regularly", "minutes_comparing_week",
                 *C.LIKERT_COLS, "challenge_count", "feature_count", "shops_online",
                 "price_seeker_index", "wtp_aed_month", "nps_concept", "is_adopter"]
    corr = df[corr_cols].corr(numeric_only=True).round(2)
    save_table(corr, "correlation_matrix", index=True)

    # ======================= FIGURES =======================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    intent.plot.bar(ax=axes[0], color=[PALETTE[2], PALETTE[4], PALETTE[1]])
    axes[0].set_title("Q25 Would you download BasketWise?")
    axes[0].set_ylabel("Share of respondents"); axes[0].tick_params(axis="x", rotation=0)
    for i, v in enumerate(intent.values):
        axes[0].text(i, v + .01, f"{v:.0%}", ha="center")
    axes[1].hist(df["monthly_grocery_spend_aed"], bins=40, color=PALETTE[0])
    axes[1].axvline(df["monthly_grocery_spend_aed"].median(), color="k", ls="--", lw=1)
    axes[1].set_title("Monthly grocery spend (AED)"); axes[1].set_xlabel("AED / month")
    nps_counts = df["nps_concept"].value_counts().sort_index()
    axes[2].bar(nps_counts.index, nps_counts.values,
                color=[PALETTE[1] if i <= 6 else PALETTE[4] if i <= 8 else PALETTE[2] for i in nps_counts.index])
    axes[2].set_title(f"Concept NPS = {nps:+.0f}"); axes[2].set_xlabel("Likelihood to recommend (0-10)")
    savefig(fig, "01_intent_spend_nps")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    for ax, key, title in [(axes[0], "income_band_aed", "Adoption rate by monthly household income (AED)"),
                           (axes[1], "age_group", "Adoption rate by age group")]:
        t = cuts[key]
        ax.bar(t[key].astype(str), t["adoption_rate"], color=PALETTE[0])
        ax.axhline(df["is_adopter"].mean(), color=PALETTE[1], ls="--", lw=1, label="overall")
        ax.set_title(title); ax.set_ylabel("Share answering 'Yes'"); ax.legend()
        ax.tick_params(axis="x", rotation=15)
    savefig(fig, "02_adoption_by_income_age")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, typ, title in [(axes[0], "challenge", "Q15 Challenges (share who ticked)"),
                           (axes[1], "feature", "Q16 Features wanted (share who ticked)")]:
        t = prev[prev.type == typ].sort_values("all_respondents")
        y = np.arange(len(t))
        ax.barh(y - .2, t["adopters"], .4, label="Adopters (Yes)", color=PALETTE[2])
        ax.barh(y + .2, t["non_adopters"], .4, label="Maybe / No", color=PALETTE[8])
        ax.set_yticks(y, t["item"].str.replace("ch_", "").str.replace("f_", ""))
        ax.set_title(title); ax.legend(loc="lower right")
    savefig(fig, "03_challenges_features_prevalence")

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90)
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.1f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(corr.values[i, j]) > .5 else "black")
    fig.colorbar(im, ax=ax, shrink=.7); ax.set_title("Correlation matrix (numeric variables)")
    savefig(fig, "04_correlation_heatmap")

    top_corr = corr["is_adopter"].drop(["is_adopter"]).sort_values(key=abs, ascending=False)
    res = dict(
        n=len(df),
        intent_share=intent.round(3).to_dict(),
        concept_nps=round(nps, 1),
        median_spend_aed=float(df["monthly_grocery_spend_aed"].median()),
        median_minutes_comparing=float(df["minutes_comparing_week"].median()),
        share_compare_3plus_stores=round(float((df["stores_used_regularly"] >= 3).mean()), 3),
        mean_wtp_aed=round(float(df["wtp_aed_month"].mean()), 2),
        share_wtp_zero=round(float((df["wtp_aed_month"] == 0).mean()), 3),
        share_prefer_only_big_savings_alerts=round(float((df["alert_preference"] == C.ALERT_PREFS[0]).mean()), 3),
        share_share_receipts=round(float(df["shares_receipts"].mean()), 3),
        top_challenges=prev[prev.type == "challenge"].nlargest(3, "all_respondents")[["item", "all_respondents"]].values.tolist(),
        top_features=prev[prev.type == "feature"].nlargest(4, "all_respondents")[["item", "all_respondents"]].values.tolist(),
        adoption_by_income=cuts["income_band_aed"].set_index("income_band_aed")["adoption_rate"].to_dict(),
        top_correlates_with_adoption=top_corr.head(6).to_dict(),
    )
    print(f"Respondents: {res['n']} | Intent: {res['intent_share']} | Concept NPS: {res['concept_nps']:+.0f}")
    print("Strongest correlates of adoption:", res["top_correlates_with_adoption"])
    save_results("descriptive", res)
    return res


if __name__ == "__main__":
    main()
