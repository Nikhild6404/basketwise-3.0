"""Step 0 - Generate the synthetic BasketWise survey dataset.

Simulates 2,000 UAE residents answering the 25-question survey in
survey/survey_questionnaire.md. Each respondent is drawn from one of five
hidden personas; their answers follow from that persona plus individual
randomness. After clean answers are generated, realistic data-quality
problems (duplicates, typos, impossible values, outliers, speeders,
straight-liners, missing values, contradictions, flipped labels) are
injected and logged so the cleaning step can be checked against them.

Outputs
  data/raw/basketwise_survey_raw.csv     messy survey export (what you analyse)
  data/raw/answer_key_true_personas.csv  hidden persona per respondent
  data/raw/noise_injection_log.csv       every injected problem, row by row

Run:  python src/generate_data.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C

rng = np.random.default_rng(C.RANDOM_STATE)

# ---------------------------------------------------------------------------
# 1. Hidden personas (never shown to the analysis, only in the answer key)
# ---------------------------------------------------------------------------
PERSONAS = {
    "Budget Families": dict(
        share=0.26, age=(38, 7, 25, 60), hh=[0, 0, .05, .2, .3, .25, .2],
        income=[.12, .38, .35, .12, .03], per_person=520,
        occ=[.0, .55, .08, .1, .25, .02], trips=2.6, stores=3.2, minutes=55,
        likert=dict(compare=4.0, switch=4.2, loyal=2.4, app=3.4, annoyed=3.6),
        channel=[.55, .1, .35], notif=[.15, .35, .35, .15],
        ch=dict(ch_price_inconsistency=.70, ch_hidden_delivery_fees=.25,
                ch_expired_offers=.55, ch_out_of_stock=.35, ch_promo_overload=.45,
                ch_time_comparing=.60, ch_bad_substitutions=.15),
        f=dict(f_price_compare=.72, f_smart_basket=.78, f_price_history=.35,
               f_wishlist_alerts=.45, f_promo_digest=.55, f_delivery_fee_compare=.20,
               f_loyalty_wallet=.40, f_expiry_tracker=.35, f_budget_planner=.55,
               f_shared_list=.60),
        intent_base=0.35, wtp_bonus=0.0),
    "Busy Professionals": dict(
        share=0.22, age=(33, 6, 23, 55), hh=[0, .35, .35, .2, .1, 0, 0],
        income=[.0, .08, .35, .37, .20], per_person=780,
        occ=[.0, .85, .03, .12, .0, .0], trips=1.4, stores=2.4, minutes=20,
        likert=dict(compare=2.9, switch=3.0, loyal=3.0, app=4.5, annoyed=4.2),
        channel=[.10, .60, .30], notif=[.05, .20, .40, .35],
        ch=dict(ch_price_inconsistency=.40, ch_hidden_delivery_fees=.72,
                ch_expired_offers=.20, ch_out_of_stock=.50, ch_promo_overload=.65,
                ch_time_comparing=.55, ch_bad_substitutions=.55),
        f=dict(f_price_compare=.45, f_smart_basket=.60, f_price_history=.25,
               f_wishlist_alerts=.62, f_promo_digest=.25, f_delivery_fee_compare=.75,
               f_loyalty_wallet=.45, f_expiry_tracker=.25, f_budget_planner=.20,
               f_shared_list=.25),
        intent_base=0.15, wtp_bonus=4.0),
    "Students & Flat-sharers": dict(
        share=0.18, age=(22, 2.5, 17, 30), hh=[0, .1, .25, .3, .2, .15, 0],
        income=[.55, .35, .08, .02, .0], per_person=330,
        occ=[.70, .05, .20, .03, .0, .02], trips=2.2, stores=2.8, minutes=40,
        likert=dict(compare=3.8, switch=4.1, loyal=2.0, app=4.6, annoyed=3.4),
        channel=[.45, .25, .30], notif=[.10, .40, .35, .15],
        ch=dict(ch_price_inconsistency=.60, ch_hidden_delivery_fees=.45,
                ch_expired_offers=.40, ch_out_of_stock=.20, ch_promo_overload=.30,
                ch_time_comparing=.45, ch_bad_substitutions=.15),
        f=dict(f_price_compare=.75, f_smart_basket=.50, f_price_history=.30,
               f_wishlist_alerts=.35, f_promo_digest=.60, f_delivery_fee_compare=.40,
               f_loyalty_wallet=.20, f_expiry_tracker=.20, f_budget_planner=.70,
               f_shared_list=.65),
        intent_base=0.25, wtp_bonus=-3.0),
    "Loyal Convenience Shoppers": dict(
        share=0.20, age=(47, 9, 28, 72), hh=[0, .15, .35, .25, .15, .1, 0],
        income=[.0, .07, .28, .35, .30], per_person=950,
        occ=[.0, .50, .05, .20, .10, .15], trips=2.0, stores=1.4, minutes=8,
        likert=dict(compare=1.9, switch=1.8, loyal=4.4, app=3.0, annoyed=3.8),
        channel=[.55, .20, .25], notif=[.35, .35, .20, .10],
        ch=dict(ch_price_inconsistency=.15, ch_hidden_delivery_fees=.20,
                ch_expired_offers=.10, ch_out_of_stock=.45, ch_promo_overload=.40,
                ch_time_comparing=.10, ch_bad_substitutions=.30),
        f=dict(f_price_compare=.15, f_smart_basket=.15, f_price_history=.10,
               f_wishlist_alerts=.15, f_promo_digest=.10, f_delivery_fee_compare=.15,
               f_loyalty_wallet=.55, f_expiry_tracker=.30, f_budget_planner=.10,
               f_shared_list=.25),
        intent_base=-1.6, wtp_bonus=-1.0),
    "Deal Hunters": dict(
        share=0.14, age=(36, 9, 21, 65), hh=[0, .15, .25, .25, .2, .15, 0],
        income=[.08, .30, .37, .20, .05], per_person=600,
        occ=[.05, .60, .10, .12, .10, .03], trips=3.4, stores=4.3, minutes=90,
        likert=dict(compare=4.7, switch=4.6, loyal=1.8, app=4.1, annoyed=2.8),
        channel=[.35, .20, .45], notif=[.05, .20, .35, .40],
        ch=dict(ch_price_inconsistency=.85, ch_hidden_delivery_fees=.45,
                ch_expired_offers=.80, ch_out_of_stock=.40, ch_promo_overload=.35,
                ch_time_comparing=.70, ch_bad_substitutions=.20),
        f=dict(f_price_compare=.88, f_smart_basket=.75, f_price_history=.82,
               f_wishlist_alerts=.80, f_promo_digest=.50, f_delivery_fee_compare=.40,
               f_loyalty_wallet=.50, f_expiry_tracker=.25, f_budget_planner=.35,
               f_shared_list=.30),
        intent_base=0.75, wtp_bonus=2.0),
}
PERSONA_NAMES = list(PERSONAS)
T_YES, T_NO = 2.6, 0.75  # cut-points on the latent adoption score


def likert(mean: float, size: int = 1) -> np.ndarray:
    """1-5 Likert answer around a persona mean with individual spread."""
    return np.clip(np.rint(rng.normal(mean, 0.85, size)), 1, 5).astype(int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---------------------------------------------------------------------------
# 2. Clean responses
# ---------------------------------------------------------------------------
def generate_clean(n: int) -> pd.DataFrame:
    shares = np.array([p["share"] for p in PERSONAS.values()])
    persona_idx = rng.choice(len(PERSONAS), size=n, p=shares / shares.sum())
    rows = []
    for i, pi in enumerate(persona_idx):
        name = PERSONA_NAMES[pi]
        p = PERSONAS[name]
        r: dict = {"respondent_id": f"BW{i + 1:05d}", "_persona": name}

        # --- Section A: demographics (Q1-Q7)
        m, sd, lo, hi = p["age"]
        r["age"] = int(np.clip(rng.normal(m, sd), lo, hi))
        r["gender"] = rng.choice(["Male", "Female", "Prefer not to say"],
                                 p=[.56, .41, .03])
        r["emirate"] = rng.choice(C.EMIRATES, p=[.40, .25, .18, .07, .04, .03, .03])
        r["nationality_group"] = rng.choice(C.NATIONALITY,
                                            p=[.12, .38, .20, .10, .14, .06])
        r["household_size"] = int(rng.choice(range(7), p=p["hh"])) + 1
        inc = int(rng.choice(5, p=p["income"]))
        r["income_band_aed"] = C.INCOME_BANDS[inc]
        r["occupation"] = rng.choice(C.OCCUPATIONS, p=p["occ"])

        # --- Section B: shopping behaviour (Q8-Q14)
        r["trips_per_week"] = int(np.clip(rng.poisson(p["trips"]), 0, 10))
        spend = (r["household_size"] * p["per_person"] * (1 + 0.12 * (inc - 2))
                 * rng.lognormal(0, 0.22))
        r["monthly_grocery_spend_aed"] = int(round(spend / 10) * 10)
        r["primary_channel"] = rng.choice(C.CHANNELS, p=p["channel"])
        r["stores_used_regularly"] = int(np.clip(rng.poisson(p["stores"]), 1, 8))
        r["minutes_comparing_week"] = int(np.clip(rng.gamma(2.0, p["minutes"] / 2), 0, 300))
        lk = p["likert"]
        r["lk_compare_often"] = likert(lk["compare"])[0]
        r["lk_switch_for_savings"] = likert(lk["switch"])[0]
        r["lk_brand_loyal"] = likert(lk["loyal"])[0]
        r["lk_app_comfort"] = likert(lk["app"])[0]
        r["lk_notif_annoyed"] = likert(lk["annoyed"])[0]
        r["lk_attention_check"] = 4 if rng.random() > 0.01 else int(rng.integers(1, 6))
        r["promo_notifs_per_week"] = rng.choice(C.NOTIF_BANDS, p=p["notif"])

        # --- Section C: challenges (Q15) with built-in co-occurrence
        ch = {k: rng.random() < v for k, v in p["ch"].items()}
        if ch["ch_price_inconsistency"] and ch["ch_expired_offers"] and rng.random() < .55:
            ch["ch_time_comparing"] = True
        if ch["ch_hidden_delivery_fees"] and rng.random() < .45:
            ch["ch_bad_substitutions"] = True
        if r["promo_notifs_per_week"] in ("16-30", "30+") and rng.random() < .35:
            ch["ch_promo_overload"] = True
        r.update({k: int(v) for k, v in ch.items()})

        # --- Features (Q16): pain points pull matching features in
        f = {k: rng.random() < v for k, v in p["f"].items()}
        if ch["ch_hidden_delivery_fees"] and rng.random() < .55:
            f["f_delivery_fee_compare"] = True
        if ch["ch_time_comparing"] and rng.random() < .50:
            f["f_smart_basket"] = True
        if ch["ch_expired_offers"] and rng.random() < .40:
            f["f_wishlist_alerts"] = True
        if f["f_price_compare"] and f["f_price_history"] and rng.random() < .70:
            f["f_wishlist_alerts"] = True
        if ch["ch_promo_overload"] and rng.random() < .35:
            f["f_promo_digest"] = True
        if r["household_size"] >= 4 and rng.random() < .30:
            f["f_shared_list"] = True
        r.update({k: int(v) for k, v in f.items()})

        chosen = [k for k, v in f.items() if v]
        pool = chosen or list(p["f"])
        w = np.array([p["f"][k] for k in pool]); w = w / w.sum()
        r["most_important_feature"] = rng.choice(pool, p=w)

        # --- Latent adoption propensity (drives Q23 & Q25)
        n_ch, n_f = sum(ch.values()), sum(f.values())
        logit = (p["intent_base"]
                 + 0.40 * (r["lk_compare_often"] - 3)
                 + 0.35 * (r["lk_switch_for_savings"] - 3)
                 - 0.35 * (r["lk_brand_loyal"] - 3)
                 + 0.30 * (r["lk_app_comfort"] - 3)
                 + 0.18 * (n_ch - 3)
                 + 0.45 * f["f_smart_basket"] + 0.35 * f["f_price_compare"]
                 + 0.30 * f["f_wishlist_alerts"] + 0.20 * f["f_price_history"]
                 + 0.004 * r["minutes_comparing_week"]
                 + 0.00012 * (r["monthly_grocery_spend_aed"] - 2000)
                 - 0.15 * (inc - 2)
                 + rng.normal(0, 0.9))
        # Ordinal response: one latent score, two cut-points -> No / Maybe / Yes
        z = logit + rng.logistic(0, 0.45)
        r["_z"] = z
        r["intent_download"] = "Yes" if z > T_YES else ("No" if z < T_NO else "Maybe")

        # --- Section D: notifications, savings, pricing (Q18-Q22)
        if r["lk_notif_annoyed"] >= 4:
            r["alert_preference"] = rng.choice(C.ALERT_PREFS, p=[.62, .25, .08, .05])
        else:
            r["alert_preference"] = rng.choice(C.ALERT_PREFS, p=[.30, .30, .22, .18])
        exp_save = 0.065 * r["monthly_grocery_spend_aed"] * (1 + 0.08 * (r["lk_switch_for_savings"] - 3))
        r["expected_saving_aed_month"] = int(max(0, round(rng.normal(exp_save, exp_save * 0.25 + 5) / 10) * 10))

        wtp = (-2.0 + 2.0 * inc + 0.0011 * r["monthly_grocery_spend_aed"]
               + 1.6 * f["f_wishlist_alerts"] + 1.3 * f["f_smart_basket"]
               + 1.4 * f["f_delivery_fee_compare"] + 0.9 * f["f_price_history"]
               + 0.8 * (r["lk_app_comfort"] - 3) + 0.6 * (r["lk_notif_annoyed"] - 3)
               + p["wtp_bonus"] + rng.normal(0, 3.2))
        if r["intent_download"] == "No":
            wtp *= 0.25
        elif r["intent_download"] == "Maybe":
            wtp *= 0.7
        r["wtp_aed_month"] = int(round(wtp)) if wtp >= 2.5 else 0

        if r["wtp_aed_month"] >= 8 and r["intent_download"] == "Yes" and rng.random() < .7:
            r["preferred_plan"] = C.PLANS[2]
        elif r["intent_download"] == "No":
            r["preferred_plan"] = rng.choice(C.PLANS[:2], p=[.35, .65])
        else:
            r["preferred_plan"] = rng.choice(C.PLANS, p=[.55, .30, .15])
        r["share_receipts_for_rewards"] = rng.choice(
            ["Yes", "No"], p=[min(.85, .30 + .10 * (r["lk_app_comfort"] - 3) + .20 * (name == "Deal Hunters")),
                              1 - min(.85, .30 + .10 * (r["lk_app_comfort"] - 3) + .20 * (name == "Deal Hunters"))])

        # --- Section E: intent (Q23-Q25)
        r["nps_concept"] = int(np.clip(np.rint(6.3 + 1.2 * (logit - 2.0) + rng.normal(0, 1.4)), 0, 10))
        ios_p = [.10, .25, .40, .55, .65][inc]
        r["preferred_platform"] = rng.choice(C.PLATFORMS, p=[.90 - ios_p, ios_p, .05, .05])

        # --- Survey metadata
        r["completion_time_sec"] = int(max(150, rng.normal(430, 110)))
        r["submitted_at"] = (pd.Timestamp("2026-10-05")
                             + pd.Timedelta(minutes=int(rng.integers(0, 21 * 24 * 60)))
                             ).strftime("%Y-%m-%d %H:%M")
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Noise injection (every change is logged)
# ---------------------------------------------------------------------------
def inject_noise(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    log: list[dict] = []

    def pick(k, exclude=()):
        pool = df.index.difference(list(exclude))
        return rng.choice(pool, size=k, replace=False)

    def note(idx, col, kind, old, new):
        log.append(dict(respondent_id=df.at[idx, "respondent_id"], column=col,
                        noise_type=kind, original=old, injected=new))

    # Label noise first (undetectable by design -> sets a performance ceiling)
    for idx in pick(int(0.05 * len(df))):
        old = df.at[idx, "intent_download"]
        new = rng.choice([x for x in C.INTENT_LEVELS if x != old])
        df.at[idx, "intent_download"] = new
        note(idx, "intent_download", "label_flip (undetectable)", old, new)

    # Inconsistent spellings / formats
    variants = {"Dubai": ["dubai", "DXB", "Dubai ", "DUBAI"],
                "Abu Dhabi": ["abu dhabi", "AbuDhabi", "AUH", "Abu-Dhabi"],
                "Sharjah": ["sharjah", "SHJ", "Sharja"],
                "Ras Al Khaimah": ["RAK", "Ras al-Khaimah"],
                "Umm Al Quwain": ["UAQ"], "Ajman": ["ajman "], "Fujairah": ["Fujeirah"]}
    for idx in pick(int(0.08 * len(df))):
        old = df.at[idx, "emirate"]
        new = rng.choice(variants[old])
        df.at[idx, "emirate"] = new
        note(idx, "emirate", "inconsistent_spelling", old, new)
    gmap = {"Male": ["M", "male", "MALE"], "Female": ["F", "female", "Femal"],
            "Prefer not to say": ["N/A"]}
    for idx in pick(int(0.05 * len(df))):
        old = df.at[idx, "gender"]
        new = rng.choice(gmap[old])
        df.at[idx, "gender"] = new
        note(idx, "gender", "inconsistent_spelling", old, new)
    imap = {"<5,000": "below 5000", "5,000-9,999": "5000-9999",
            "10,000-19,999": "AED 10k-20k", "20,000-34,999": "20000 - 34999",
            "35,000+": "35k and above"}
    for idx in pick(int(0.03 * len(df))):
        old = df.at[idx, "income_band_aed"]
        df.at[idx, "income_band_aed"] = imap[old]
        note(idx, "income_band_aed", "inconsistent_format", old, imap[old])

    # Impossible values (entry errors)
    for idx in pick(14):
        old = df.at[idx, "age"]
        new = int(rng.choice([0, 3, 7, 112, 150, 999, -1]))
        df.at[idx, "age"] = new
        note(idx, "age", "impossible_value", old, new)
    for idx in pick(10):
        old = df.at[idx, "monthly_grocery_spend_aed"]
        new = int(rng.choice([-250, 0, 95000, 120000, 250000]))
        df.at[idx, "monthly_grocery_spend_aed"] = new
        note(idx, "monthly_grocery_spend_aed", "impossible_value", old, new)
    for idx in pick(8):
        old = df.at[idx, "household_size"]
        df.at[idx, "household_size"] = 0
        note(idx, "household_size", "impossible_value", old, 0)

    # Plausible-but-extreme outliers (real people, keep but cap)
    for idx in pick(18):
        old = df.at[idx, "monthly_grocery_spend_aed"]
        new = int(old * rng.uniform(3, 5) // 10 * 10)
        df.at[idx, "monthly_grocery_spend_aed"] = new
        note(idx, "monthly_grocery_spend_aed", "extreme_outlier", old, new)
    for idx in pick(12):
        old = df.at[idx, "minutes_comparing_week"]
        new = int(rng.integers(600, 2000))
        df.at[idx, "minutes_comparing_week"] = new
        note(idx, "minutes_comparing_week", "extreme_outlier", old, new)
    for idx in pick(10):
        old = df.at[idx, "wtp_aed_month"]
        new = int(rng.choice([150, 200, 300, 500]))
        df.at[idx, "wtp_aed_month"] = new
        note(idx, "wtp_aed_month", "extreme_outlier", old, new)

    # Missing values (item non-response)
    for col, frac in [("income_band_aed", .04), ("monthly_grocery_spend_aed", .03),
                      ("household_size", .02), ("nps_concept", .03),
                      ("wtp_aed_month", .02), ("occupation", .02),
                      ("minutes_comparing_week", .02)]:
        for idx in pick(int(frac * len(df))):
            note(idx, col, "missing_value", df.at[idx, col], None)
            df.at[idx, col] = np.nan

    # Contradictory answers: says "No" to app but WTP high + premium plan
    no_idx = df.index[df["intent_download"] == "No"]
    for idx in rng.choice(no_idx, size=int(0.02 * len(df)), replace=False):
        old = df.at[idx, "wtp_aed_month"]
        df.at[idx, "wtp_aed_month"] = int(rng.integers(20, 40))
        df.at[idx, "preferred_plan"] = C.PLANS[2]
        note(idx, "wtp_aed_month/preferred_plan", "contradiction", old, "high WTP + Premium")

    # Straight-liners: same answer on every grid row incl. the attention check
    grid = C.LIKERT_COLS + [C.ATTENTION_COL]
    for idx in pick(int(0.03 * len(df))):
        v = int(rng.choice([1, 3, 5]))
        for c in grid:
            df.at[idx, c] = v
        df.at[idx, "completion_time_sec"] = int(rng.integers(160, 260))
        note(idx, "lk_* grid", "straight_liner", None, v)

    # Speeders: finished the 25 questions implausibly fast
    for idx in pick(int(0.03 * len(df))):
        old = df.at[idx, "completion_time_sec"]
        new = int(rng.integers(35, 95))
        df.at[idx, "completion_time_sec"] = new
        note(idx, "completion_time_sec", "speeder", old, new)

    # Duplicates: exact re-submits and same answers under a new id
    dup_exact = df.loc[pick(25)].copy()
    for rid in dup_exact["respondent_id"]:
        log.append(dict(respondent_id=rid, column="(row)", noise_type="exact_duplicate",
                        original=None, injected="row repeated"))
    dup_newid = df.loc[pick(15)].copy()
    start = len(df) + 1
    new_ids = [f"BW{start + j:05d}" for j in range(len(dup_newid))]
    for old_id, new_id in zip(dup_newid["respondent_id"], new_ids):
        log.append(dict(respondent_id=new_id, column="respondent_id",
                        noise_type="duplicate_new_id", original=old_id, injected=new_id))
    dup_newid["respondent_id"] = new_ids
    df = pd.concat([df, dup_exact, dup_newid], ignore_index=True)
    df = df.sample(frac=1, random_state=C.RANDOM_STATE).reset_index(drop=True)
    return df, pd.DataFrame(log)


def main() -> None:
    clean = generate_clean(C.N_RESPONDENTS)
    raw, log = inject_noise(clean)

    key = raw[["respondent_id", "_persona"]].drop_duplicates("respondent_id")
    key = key.rename(columns={"_persona": "true_persona"})
    raw = raw.drop(columns=["_persona", "_z"])

    C.DATA_RAW.mkdir(parents=True, exist_ok=True)
    raw.to_csv(C.RAW_FILE, index=False)
    key.to_csv(C.ANSWER_KEY_FILE, index=False)
    log.to_csv(C.NOISE_LOG_FILE, index=False)
    print(f"Raw survey export : {C.RAW_FILE.name}  ({len(raw)} rows x {raw.shape[1]} cols)")
    print(f"Noise log         : {len(log)} injected issues")
    print(log["noise_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
