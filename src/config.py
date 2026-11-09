"""Shared settings: paths, column groups and category orders.

Every script imports from here so the column-to-algorithm mapping lives in
one place (see docs/column_algorithm_map.md for the written rationale).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIG_DIR = ROOT / "outputs" / "figures"
TAB_DIR = ROOT / "outputs" / "tables"

RAW_FILE = DATA_RAW / "basketwise_survey_raw.csv"
ANSWER_KEY_FILE = DATA_RAW / "answer_key_true_personas.csv"
NOISE_LOG_FILE = DATA_RAW / "noise_injection_log.csv"
CLEAN_FILE = DATA_PROCESSED / "basketwise_survey_clean.csv"
MODEL_FILE = DATA_PROCESSED / "basketwise_model_ready.csv"
RESULTS_FILE = ROOT / "outputs" / "results_summary.json"

RANDOM_STATE = 42
N_RESPONDENTS = 2000

# ---- Category orders ---------------------------------------------------
EMIRATES = ["Dubai", "Abu Dhabi", "Sharjah", "Ajman",
            "Ras Al Khaimah", "Fujairah", "Umm Al Quwain"]
INCOME_BANDS = ["<5,000", "5,000-9,999", "10,000-19,999",
                "20,000-34,999", "35,000+"]
NOTIF_BANDS = ["0-5", "6-15", "16-30", "30+"]
CHANNELS = ["In-store", "Online/app delivery", "Mix of both"]
OCCUPATIONS = ["Student", "Employed full-time", "Employed part-time",
               "Self-employed", "Homemaker", "Retired/Not working"]
NATIONALITY = ["Emirati", "South Asian", "Arab expat", "Western",
               "Southeast Asian", "Other"]
ALERT_PREFS = ["Only big savings on my list", "Weekly digest",
               "Daily digest", "Real-time for every deal"]
PLANS = ["Free with ads", "Free, no alerts",
         "Premium (no ads + smart alerts)"]
PLATFORMS = ["Android", "iOS", "Web", "WhatsApp bot"]
INTENT_LEVELS = ["No", "Maybe", "Yes"]

# ---- Multi-select items (Q15 challenges, Q16 features) ------------------
CHALLENGES = {
    "ch_price_inconsistency": "Same product priced very differently across stores",
    "ch_hidden_delivery_fees": "Delivery/service fees make online look cheaper than it is",
    "ch_expired_offers": "Offers have expired or are out of stock when I arrive",
    "ch_out_of_stock": "Items I want are often out of stock",
    "ch_promo_overload": "Too many promotional notifications and flyers",
    "ch_time_comparing": "Comparing prices takes too much time",
    "ch_bad_substitutions": "Online orders arrive with poor substitutions",
}
FEATURES = {
    "f_price_compare": "Compare one product's price across stores",
    "f_smart_basket": "Find the cheapest store for my whole basket",
    "f_price_history": "Price history and 'is this a real deal?' signal",
    "f_wishlist_alerts": "Alerts only when items on my list drop in price",
    "f_promo_digest": "One weekly digest of the best offers",
    "f_delivery_fee_compare": "Total cost incl. delivery fees across apps",
    "f_loyalty_wallet": "All loyalty cards and points in one wallet",
    "f_expiry_tracker": "Pantry/expiry tracker to cut food waste",
    "f_budget_planner": "Monthly grocery budget and meal planner",
    "f_shared_list": "Shared family/flatmate shopping list",
}
CHALLENGE_COLS = list(CHALLENGES)
FEATURE_COLS = list(FEATURES)

LIKERT_COLS = ["lk_compare_often", "lk_switch_for_savings",
               "lk_brand_loyal", "lk_app_comfort", "lk_notif_annoyed"]
ATTENTION_COL = "lk_attention_check"   # instructed answer = 4 ("Agree")

# ---- Targets ------------------------------------------------------------
CLASS_TARGET = "intent_download"        # Q25 Yes / Maybe / No
CLASS_TARGET_BINARY = "is_adopter"      # engineered: 1 = Yes
REG_TARGET = "wtp_aed_month"            # Q20 willingness to pay
