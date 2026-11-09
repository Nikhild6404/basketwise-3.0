# Data dictionary

## Files

| File | Rows | What it is |
|---|---|---|
| `data/raw/basketwise_survey_raw.csv` | 2,040 | Messy survey export (2,000 respondents + 40 duplicates), 48 columns. **Start here.** |
| `data/raw/answer_key_true_personas.csv` | 2,000 | Hidden persona of each respondent. Used only to validate clustering; a real survey would not have it. |
| `data/raw/noise_injection_log.csv` | 1,052 | Every data problem deliberately injected, with the original and injected value. |
| `data/processed/basketwise_survey_clean.csv` | 1,825 | Cleaned, validated data with engineered features (readable categories). |
| `data/processed/basketwise_model_ready.csv` | 1,825 | Same, with nominal categories one-hot encoded for machine learning. |
| `data/processed/segments.csv` | 1,825 | Cluster id and segment name per respondent. |

## Raw columns (survey export)

| Column | Q | Type | Values / range |
|---|---|---|---|
| `respondent_id` | — | id | BW00001 … |
| `age` | Q1 | int | 16–80 (raw contains impossible values) |
| `gender` | Q2 | category | Male, Female, Prefer not to say (raw has spelling variants) |
| `emirate` | Q3 | category | 7 emirates (raw has variants such as DXB, AUH, Sharja) |
| `nationality_group` | Q4 | category | Emirati, South Asian, Arab expat, Western, Southeast Asian, Other |
| `household_size` | Q5 | int | 1–7 (raw contains 0) |
| `income_band_aed` | Q6 | ordinal | <5,000 … 35,000+ AED/month (raw has format variants) |
| `occupation` | Q7 | category | 6 options |
| `trips_per_week` | Q8 | int | 0–10 |
| `monthly_grocery_spend_aed` | Q9 | int | AED/month (raw has negatives and 250,000) |
| `primary_channel` | Q10 | category | In-store, Online/app delivery, Mix of both |
| `stores_used_regularly` | Q11 | int | 1–8 |
| `minutes_comparing_week` | Q12 | int | minutes/week |
| `lk_compare_often` … `lk_notif_annoyed` | Q13a–e | Likert | 1 = strongly disagree … 5 = strongly agree |
| `lk_attention_check` | Q13f | Likert | correct answer = 4 |
| `promo_notifs_per_week` | Q14 | ordinal | 0-5, 6-15, 16-30, 30+ |
| `ch_*` (7 columns) | Q15 | binary | 1 = faced this challenge |
| `f_*` (10 columns) | Q16 | binary | 1 = would use this feature |
| `most_important_feature` | Q17 | category | one `f_*` name |
| `alert_preference` | Q18 | category | 4 options |
| `expected_saving_aed_month` | Q19 | int | AED/month |
| `wtp_aed_month` | Q20 | int | AED/month for Premium, 0 = would not pay |
| `preferred_plan` | Q21 | category | Free with ads, Free no alerts, Premium |
| `share_receipts_for_rewards` | Q22 | Yes/No | |
| `nps_concept` | Q23 | int | 0–10 |
| `preferred_platform` | Q24 | category | Android, iOS, Web, WhatsApp bot |
| `intent_download` | Q25 | category | Yes, Maybe, No |
| `completion_time_sec` | meta | int | seconds to finish |
| `submitted_at` | meta | datetime | Oct 2026 |

## Engineered columns (added in `clean_transform.py`)

| Column | Formula | Why |
|---|---|---|
| `income_ord` | band → 1..5 | Ordinal income usable by every model |
| `income_mid_aed` | band midpoint | For spend-to-income ratio |
| `spend_per_person_aed` | spend / household size | Separates rich singles from large families |
| `spend_share_of_income` | spend / income midpoint (capped 1.5) | Grocery burden |
| `challenge_count`, `feature_count` | row sums of `ch_*`, `f_*` | Breadth of pain and need |
| `price_seeker_index` | (compare + switch + (6 − loyal)) / 3 | One attitude score |
| `shops_online` | channel is online or mix | Binary |
| `promo_notif_ord` | band → 1..4 | Ordinal |
| `shares_receipts` | Yes → 1 | Binary |
| `age_group` | 16-24 … 55+ | Descriptive cuts |
| `intent_ord` | No 0, Maybe 1, Yes 2 | 3-class model |
| `is_adopter` | Q25 = Yes | **Classification target** |
| `nps_group` | 0-6 Detractor, 7-8 Passive, 9-10 Promoter | NPS reporting |

## Injected data problems and how each is handled

| Problem | Count | Detected by | Treatment |
|---|---|---|---|
| Exact duplicate rows | 25 | `duplicated()` | Dropped |
| Re-submission under a new ID | 15 | duplicate on all answer columns | Dropped |
| Inconsistent emirate / gender spelling | 160 / 100 | value map after lower-casing and stripping | Standardised |
| Inconsistent income band format | 60 | value map | Standardised (56 remain after other drops; 4 sat on already-removed rows) |
| Speeders (< 120 s for 25 questions) | 60 | `completion_time_sec` | Dropped |
| Straight-liners | 60 | failed attention check (plus ~1% genuine failures) | Dropped (76 total) |
| Contradictory answers (No, but Premium and WTP ≥ 20) | 40 | rule | Dropped (39) |
| Impossible age / household / spend | 14 / 8 / 10 | validity ranges | Set to missing, then imputed |
| Extreme but plausible outliers | 40 | Q3 + 3×IQR | Capped (not dropped: real heavy spenders exist) |
| Missing values | ~360 cells | `isna()` | Median / group median / mode; WTP target left missing and those rows skipped in regression |
| Randomly flipped Q25 labels | 100 | **cannot be detected** | Kept. Sets a ceiling on achievable accuracy, as in real surveys |
