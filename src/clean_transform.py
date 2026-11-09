"""Step 1 - Data engineering: clean, validate and transform the raw export.

Every rule is applied in a fixed order and its effect is counted, so the
cleaning log (outputs/tables/cleaning_log.csv) shows exactly how 2,040 raw
rows became the analysis sample. The same script runs unchanged on a real
survey export with the same column names.

Outputs
  data/processed/basketwise_survey_clean.csv   readable, one row per valid respondent
  data/processed/basketwise_model_ready.csv    encoded + engineered features for ML
  outputs/tables/cleaning_log.csv

Run:  python src/clean_transform.py
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

import config as C
from utils import banner, save_results, save_table

EMIRATE_MAP = {
    "dubai": "Dubai", "dxb": "Dubai",
    "abudhabi": "Abu Dhabi", "auh": "Abu Dhabi",
    "sharjah": "Sharjah", "sharja": "Sharjah", "shj": "Sharjah",
    "ajman": "Ajman",
    "rasalkhaimah": "Ras Al Khaimah", "rak": "Ras Al Khaimah",
    "fujairah": "Fujairah", "fujeirah": "Fujairah",
    "ummalquwain": "Umm Al Quwain", "uaq": "Umm Al Quwain",
}
GENDER_MAP = {"m": "Male", "male": "Male", "f": "Female", "female": "Female",
              "femal": "Female", "n/a": "Prefer not to say",
              "prefer not to say": "Prefer not to say"}
INCOME_MAP = {"below 5000": "<5,000", "5000-9999": "5,000-9,999",
              "aed 10k-20k": "10,000-19,999", "20000 - 34999": "20,000-34,999",
              "35k and above": "35,000+"}
INCOME_MID = dict(zip(C.INCOME_BANDS, [3500, 7500, 15000, 27500, 45000]))

# Validity ranges used to separate data-entry errors from genuine extremes
VALID = {"age": (16, 80), "household_size": (1, 15),
         "monthly_grocery_spend_aed": (100, 50000)}
SPEEDER_SEC = 120          # 25 questions in under 2 minutes is not credible


def _key(s: str) -> str:
    return re.sub(r"[^a-z]", "", str(s).lower())


def iqr_cap(s: pd.Series, k: float = 3.0) -> tuple[pd.Series, int, float]:
    """Cap values above Q3 + k*IQR (k=3 keeps real heavy spenders, trims absurd ones)."""
    q1, q3 = s.quantile([.25, .75])
    upper = q3 + k * (q3 - q1)
    n = int((s > upper).sum())
    return s.clip(upper=upper), n, float(upper)


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = raw.copy()
    log = []

    def step(name, rule, rows_affected):
        log.append(dict(step=len(log) + 1, action=name, rule=rule,
                        rows_affected=int(rows_affected), rows_remaining=len(df)))

    # 1-2. Duplicates
    n = df.duplicated().sum()
    df = df.drop_duplicates()
    step("Drop exact duplicates", "identical on every column", n)
    answer_cols = [c for c in df.columns if c != "respondent_id"]
    n = df.duplicated(subset=answer_cols).sum()
    df = df.drop_duplicates(subset=answer_cols)
    step("Drop re-submissions under a new ID", "identical answers + timestamp, different respondent_id", n)

    # 3. Standardise text categories
    before = df["emirate"].copy()
    df["emirate"] = df["emirate"].map(lambda v: EMIRATE_MAP.get(_key(v), v))
    step("Standardise emirate spellings", "e.g. 'DXB', 'dubai ' -> 'Dubai'", (before != df["emirate"]).sum())
    before = df["gender"].copy()
    # pandas reads the text "N/A" as missing, which here means "Prefer not to say"
    df["gender"] = df["gender"].fillna("N/A").map(lambda v: GENDER_MAP.get(str(v).strip().lower(), v))
    step("Standardise gender labels", "e.g. 'M', 'male' -> 'Male'", (before != df["gender"]).sum())
    before = df["income_band_aed"].copy()
    df["income_band_aed"] = df["income_band_aed"].map(
        lambda v: INCOME_MAP.get(str(v).strip().lower(), v) if pd.notna(v) else v)
    step("Standardise income band formats", "e.g. 'AED 10k-20k' -> '10,000-19,999'",
         (before.fillna("") != df["income_band_aed"].fillna("")).sum())

    # 4-5. Low-quality respondents
    mask = df["completion_time_sec"] < SPEEDER_SEC
    df = df[~mask]
    step("Remove speeders", f"completion_time_sec < {SPEEDER_SEC}", mask.sum())
    mask = df[C.ATTENTION_COL] != 4
    df = df[~mask]
    step("Remove failed attention checks / straight-liners",
         "instructed item 'select Agree (4)' not answered 4", mask.sum())

    # 6. Contradictory answers
    mask = ((df["intent_download"] == "No") & (df["preferred_plan"] == C.PLANS[2])
            & (df["wtp_aed_month"] >= 20))
    df = df[~mask]
    step("Remove contradictory respondents",
         "would NOT download, yet picks Premium and WTP >= AED 20", mask.sum())

    # 7. Impossible values -> missing (then imputed below)
    for col, (lo, hi) in VALID.items():
        bad = df[col].notna() & ~df[col].between(lo, hi)
        df.loc[bad, col] = np.nan
        step(f"Set impossible {col} to missing", f"outside {lo}-{hi}", bad.sum())

    # 8. Genuine-but-extreme outliers -> capped (IQR x3)
    for col in ["monthly_grocery_spend_aed", "minutes_comparing_week", "wtp_aed_month"]:
        capped, n, upper = iqr_cap(df[col].dropna())
        df.loc[capped.index, col] = capped
        step(f"Cap extreme {col}", f"> Q3 + 3*IQR ({upper:,.0f})", n)

    # 9. Missing values
    miss_before = df.isna().sum()
    df["household_size"] = df["household_size"].fillna(df["household_size"].median())
    df["age"] = df["age"].fillna(df.groupby("occupation")["age"].transform("median")).fillna(df["age"].median())
    df["monthly_grocery_spend_aed"] = df["monthly_grocery_spend_aed"].fillna(
        df.groupby("household_size")["monthly_grocery_spend_aed"].transform("median"))
    df["minutes_comparing_week"] = df["minutes_comparing_week"].fillna(df["minutes_comparing_week"].median())
    df["occupation"] = df["occupation"].fillna(df["occupation"].mode()[0])
    df["income_band_aed"] = df["income_band_aed"].fillna(
        df.groupby("occupation")["income_band_aed"].transform(lambda s: s.mode()[0]))
    df["nps_concept"] = df["nps_concept"].fillna(df["nps_concept"].median())
    # wtp_aed_month is the regression target: left missing, dropped only in regression
    step("Impute missing values",
         "median (numeric, by group where sensible) / mode (categorical); regression target left as-is",
         int(miss_before.drop("wtp_aed_month").sum()))

    df = df.reset_index(drop=True)
    for c in ["age", "household_size", "monthly_grocery_spend_aed", "minutes_comparing_week"]:
        df[c] = df[c].round().astype(int)
    return df, pd.DataFrame(log)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Derived features that the algorithms and the business questions need."""
    out = df.copy()
    out["income_ord"] = out["income_band_aed"].map({b: i + 1 for i, b in enumerate(C.INCOME_BANDS)})
    out["income_mid_aed"] = out["income_band_aed"].map(INCOME_MID)
    out["spend_per_person_aed"] = (out["monthly_grocery_spend_aed"] / out["household_size"]).round(0)
    out["spend_share_of_income"] = (out["monthly_grocery_spend_aed"] / out["income_mid_aed"]).clip(upper=1.5).round(3)
    out["challenge_count"] = out[C.CHALLENGE_COLS].sum(axis=1)
    out["feature_count"] = out[C.FEATURE_COLS].sum(axis=1)
    out["price_seeker_index"] = ((out["lk_compare_often"] + out["lk_switch_for_savings"]
                                  + (6 - out["lk_brand_loyal"])) / 3).round(2)
    out["shops_online"] = out["primary_channel"].isin(["Online/app delivery", "Mix of both"]).astype(int)
    out["promo_notif_ord"] = out["promo_notifs_per_week"].map({b: i + 1 for i, b in enumerate(C.NOTIF_BANDS)})
    out["shares_receipts"] = (out["share_receipts_for_rewards"] == "Yes").astype(int)
    out["age_group"] = pd.cut(out["age"], [15, 24, 34, 44, 54, 80],
                              labels=["16-24", "25-34", "35-44", "45-54", "55+"]).astype(str)
    out["intent_ord"] = out["intent_download"].map({"No": 0, "Maybe": 1, "Yes": 2})
    out[C.CLASS_TARGET_BINARY] = (out["intent_download"] == "Yes").astype(int)
    out["nps_group"] = pd.cut(out["nps_concept"], [-1, 6, 8, 10],
                              labels=["Detractor", "Passive", "Promoter"]).astype(str)
    return out


def model_ready(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode nominal categories (drop_first avoids the dummy trap)."""
    nominal = ["gender", "emirate", "nationality_group", "occupation", "primary_channel"]
    enc = pd.get_dummies(df, columns=nominal, drop_first=True, dtype=int)
    return enc


def main() -> pd.DataFrame:
    banner("STEP 1  Data engineering: cleaning & transformation")
    raw = pd.read_csv(C.RAW_FILE)
    clean_df, log = clean(raw)
    feat = engineer(clean_df)
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    feat.to_csv(C.CLEAN_FILE, index=False)
    model_ready(feat).to_csv(C.MODEL_FILE, index=False)
    save_table(log, "cleaning_log")
    print(log[["step", "action", "rows_affected", "rows_remaining"]].to_string(index=False))
    print(f"\nRaw rows {len(raw)}  ->  clean rows {len(feat)}  ({len(raw) - len(feat)} removed)")
    save_results("cleaning", dict(raw_rows=len(raw), clean_rows=len(feat),
                                  removed=len(raw) - len(feat),
                                  steps={r.action: r.rows_affected for r in log.itertuples()}))
    return feat


if __name__ == "__main__":
    main()
