# BasketWise UAE — Consumer Survey (25 questions)

**Purpose:** validate demand for BasketWise, a grocery price-comparison app for the UAE, and collect data that supports four analytics techniques: classification, clustering, association rule mining and regression.

**Length:** 25 questions (within the 20–25 limit), about 6–8 minutes.
**Audience:** UAE residents aged 16+ who buy groceries for their household.
**Quality controls:** one instructed attention-check row inside Q13, a recorded completion time (to flag speeders), and duplicate detection on submission.

Each question lists the **dataset column(s)** it fills and the **technique(s)** it serves:
`CLS` classification · `CLU` clustering · `ARM` association rules · `REG` regression · `DES` descriptive only.

> **Concept shown before Section C:** *"BasketWise compares grocery prices across Carrefour, Lulu, Union Coop, Spinneys, Choithrams and delivery apps, tells you which store is cheapest for your whole basket, and only notifies you when items on your list genuinely drop in price."*
> (Retailer names describe where prices would be compared; this is a student concept, not affiliated with any of them.)

---

## Section A — About you (demographics)

| # | Question | Answer format | Column | Used in |
|---|---|---|---|---|
| Q1 | What is your age? | Number (16–80) | `age` | CLS, CLU, REG |
| Q2 | Gender | Male / Female / Prefer not to say | `gender` | CLS, REG, DES |
| Q3 | Which emirate do you live in? | Dubai / Abu Dhabi / Sharjah / Ajman / Ras Al Khaimah / Fujairah / Umm Al Quwain | `emirate` | CLS, REG, DES |
| Q4 | Which group best describes your nationality? | Emirati / South Asian / Arab expat / Western / Southeast Asian / Other | `nationality_group` | CLS, REG, DES |
| Q5 | How many people live in your household (including you)? | Number (1–15) | `household_size` | CLS, CLU, REG |
| Q6 | Monthly household income (AED) | <5,000 / 5,000–9,999 / 10,000–19,999 / 20,000–34,999 / 35,000+ | `income_band_aed` → `income_ord` | CLS, CLU, REG |
| Q7 | Current occupation | Student / Employed full-time / Employed part-time / Self-employed / Homemaker / Retired or not working | `occupation` | CLS, REG, DES |

## Section B — How you shop today

| # | Question | Answer format | Column | Used in |
|---|---|---|---|---|
| Q8 | How many grocery trips or orders do you make per week? | Number (0–10) | `trips_per_week` | CLS, CLU, REG |
| Q9 | Roughly how much does your household spend on groceries per month (AED)? | Number | `monthly_grocery_spend_aed` | CLS, CLU (per person), REG |
| Q10 | How do you mostly buy groceries? | In-store / Online or app delivery / Mix of both | `primary_channel` → `shops_online` | CLS, CLU, REG |
| Q11 | How many different stores or grocery apps do you use regularly? | Number (1–8) | `stores_used_regularly` | CLS, CLU, REG |
| Q12 | About how many minutes per week do you spend comparing grocery prices (flyers, apps, websites)? | Number | `minutes_comparing_week` | CLS, CLU, REG |
| Q13 | How much do you agree with each statement? *(1 = Strongly disagree … 5 = Strongly agree)* | Likert grid, 6 rows | see below | CLS, CLU, REG |
| | a) I regularly compare prices before buying groceries | 1–5 | `lk_compare_often` | |
| | b) I would switch stores to save 5% on my basket | 1–5 | `lk_switch_for_savings` | |
| | c) I stick to the same store and brands whatever the price | 1–5 | `lk_brand_loyal` | |
| | d) I am comfortable managing my shopping in an app | 1–5 | `lk_app_comfort` | |
| | e) Promotional notifications from shopping apps annoy me | 1–5 | `lk_notif_annoyed` | |
| | f) *Attention check:* please select "Agree" (4) for this row | 1–5 | `lk_attention_check` | data quality |
| Q14 | How many promotional notifications do you receive from grocery apps per week? | 0–5 / 6–15 / 16–30 / 30+ | `promo_notifs_per_week` → `promo_notif_ord` | CLS, REG |

## Section C — Challenges and features *(tick all that apply)*

| # | Question | Options → columns (1 = ticked, 0 = not) | Used in |
|---|---|---|---|
| Q15 | Which of these problems have you faced when buying groceries in the UAE? | Same product priced very differently across stores → `ch_price_inconsistency` · Delivery/service fees make online look cheaper than it is → `ch_hidden_delivery_fees` · Offers have expired or are out of stock when I arrive → `ch_expired_offers` · Items I want are often out of stock → `ch_out_of_stock` · Too many promotional notifications and flyers → `ch_promo_overload` · Comparing prices takes too much time → `ch_time_comparing` · Online orders arrive with poor substitutions → `ch_bad_substitutions` | **ARM**, CLS, REG (count → CLU) |
| Q16 | Which features would you actually use in BasketWise? | Compare one product's price across stores → `f_price_compare` · Find the cheapest store for my whole basket → `f_smart_basket` · Price history and "is this a real deal?" → `f_price_history` · Alerts only when items on my list drop in price → `f_wishlist_alerts` · One weekly digest of the best offers → `f_promo_digest` · Total cost incl. delivery fees across apps → `f_delivery_fee_compare` · All loyalty cards and points in one wallet → `f_loyalty_wallet` · Pantry/expiry tracker to cut food waste → `f_expiry_tracker` · Monthly grocery budget and meal planner → `f_budget_planner` · Shared family/flatmate shopping list → `f_shared_list` | **ARM**, CLS, REG (count → CLU) |
| Q17 | Which ONE feature matters most to you? | Single choice from the Q16 list | `most_important_feature` | DES (cross-check of Q16) |

## Section D — Alerts, savings and price

| # | Question | Answer format | Column | Used in |
|---|---|---|---|---|
| Q18 | How would you like BasketWise to notify you? | Only big savings on items on my list / Weekly digest / Daily digest / Real-time for every deal | `alert_preference` | DES (product design) |
| Q19 | How much would the app need to save you per month (AED) to be worth using? | Number | `expected_saving_aed_month` | DES (value proposition) |
| Q20 | What is the most you would pay per month for BasketWise Premium (no ads + smart alerts)? Enter 0 if nothing. | Number (AED) | `wtp_aed_month` | **REG target** |
| Q21 | Which plan would you choose? | Free with ads / Free, no alerts / Premium | `preferred_plan` | DES, data quality (contradiction check) |
| Q22 | Would you share digital receipts in exchange for rewards? | Yes / No | `share_receipts_for_rewards` → `shares_receipts` | CLS, REG (partner-data strategy) |

## Section E — Overall reaction

| # | Question | Answer format | Column | Used in |
|---|---|---|---|---|
| Q23 | How likely are you to recommend BasketWise to a friend? | 0–10 | `nps_concept` | DES (concept NPS) |
| Q24 | Where would you prefer to use BasketWise? | Android / iOS / Web / WhatsApp bot | `preferred_platform` | DES (build priority) |
| Q25 | If BasketWise launched today, would you download and use it? | Yes / Maybe / No | `intent_download` → `is_adopter` | **CLS target** |

---

### Design notes

- **Why the classification target is last.** Q25 comes after respondents have seen the concept and thought about their own problems and features, so their answer is more considered. For the same reason Q20–Q25 are *not* used as classification inputs: they are answered in the same moment as Q25 and would leak the answer.
- **Why Q15 and Q16 are tick-all-that-apply.** Association rule mining needs "baskets" of yes/no items. Tick boxes give exactly that, so we can test rules like *"people who face challenge 1 and 3 also face challenge 6"* and *"people who want features A and B also want C"*.
- **Why numbers, not bands, for spend, minutes and WTP.** Regression and clustering work better on continuous values, and numbers let us detect outliers and impossible values during cleaning.
- **Why an attention check and timing.** Straight-liners and speeders are common in online panels. Removing them is part of the data-engineering step.
