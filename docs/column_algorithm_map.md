# Which columns feed which algorithm, and why

This is the map between every dataset column and the four techniques. ✅ = used as an input, 🎯 = target, 📊 = used only to describe results afterwards, — = not used.

| Column (survey Q) | Classification | Clustering | Association rules | Regression | Why |
|---|---|---|---|---|---|
| `age` (Q1) | ✅ | ✅ | — | ✅ | Life stage shapes shopping habits and tech comfort |
| `gender` (Q2) | ✅ (one-hot) | — | — | ✅ | Control variable; not a behaviour, so kept out of segments |
| `emirate` (Q3) | ✅ (one-hot) | — | — | ✅ | Store coverage differs by emirate; needed for rollout order |
| `nationality_group` (Q4) | ✅ (one-hot) | — | — | ✅ | Control variable; shopping culture |
| `household_size` (Q5) | ✅ | ✅ | — | ✅ | Bigger baskets = bigger absolute savings |
| `income_ord` (Q6) | ✅ | ✅ | — | ✅ | Price sensitivity and ability to pay pull in opposite directions |
| `occupation` (Q7) | ✅ (one-hot) | 📊 | — | ✅ | Students behave differently; used to name the student segment |
| `trips_per_week` (Q8) | ✅ | ✅ | — | ✅ | Frequency of purchase decisions |
| `monthly_grocery_spend_aed` (Q9) | ✅ | 📊 | — | ✅ | Size of the prize; clustering uses spend *per person* instead |
| `spend_per_person_aed` (Q9/Q5) | ✅ | ✅ (log) | — | ✅ | Separates affluent small households from large budget families |
| `spend_share_of_income` (Q9/Q6) | ✅ | — | — | ✅ | Grocery burden: high share = stronger reason to save |
| `shops_online` (Q10) | ✅ | ✅ | — | ✅ | Online shoppers care about delivery fees and substitutions |
| `stores_used_regularly` (Q11) | ✅ | ✅ | — | ✅ | Multi-store shoppers already compare, so the app saves them time |
| `minutes_comparing_week` (Q12) | ✅ | ✅ (log) | — | ✅ | Time currently spent = time the app can save |
| `lk_compare_often` (Q13a) | ✅ | ✅ | — | ✅ | Core attitude: price awareness |
| `lk_switch_for_savings` (Q13b) | ✅ | ✅ | — | ✅ | Will they act on a cheaper price? |
| `lk_brand_loyal` (Q13c) | ✅ | ✅ | — | ✅ | Loyalists ignore comparisons; strongest negative signal |
| `lk_app_comfort` (Q13d) | ✅ | ✅ | — | ✅ | Barrier to adopting any app |
| `lk_notif_annoyed` (Q13e) | ✅ | ✅ | — | ✅ | Validates the "quiet alerts" positioning |
| `price_seeker_index` (Q13a,b,c) | ✅ | — | — | ✅ (trees only) | Composite attitude score; dropped from linear coefficients because it is an exact combination of three items |
| `promo_notif_ord` (Q14) | ✅ | — | — | ✅ | Notification fatigue |
| `ch_*` 7 columns (Q15) | ✅ | count only | ✅ | ✅ | Pain points; ARM finds which co-occur |
| `f_*` 10 columns (Q16) | ✅ | count only | ✅ | ✅ | Wanted features; ARM finds which must ship together |
| `challenge_count`, `feature_count` | ✅ | ✅ | — | ✅ | Breadth of need; keeps clustering from being dominated by 17 binary columns |
| `shares_receipts` (Q22) | ✅ | — | — | ✅ | Openness to data-sharing partnerships |
| `most_important_feature` (Q17) | — | — | — | — | Cross-check of Q16 only |
| `alert_preference` (Q18) | — | 📊 | — | — | Product design input, answered after the concept |
| `expected_saving_aed_month` (Q19) | — | — | — | — | Answered after the concept; describes the value bar |
| `wtp_aed_month` (Q20) | — (leakage) | 📊 | — | 🎯 | Regression target |
| `preferred_plan` (Q21) | — (leakage) | 📊 | — | — (leakage) | Nearly restates intent and WTP; used to catch contradictions |
| `nps_concept` (Q23) | — (leakage) | 📊 | — | — (leakage) | Same moment as Q25; correlation 0.57 with adoption |
| `preferred_platform` (Q24) | — | — | — | — | Build-priority question only |
| `intent_download` / `is_adopter` (Q25) | 🎯 | 📊 | (adopter-only rules) | — (leakage) | Classification target |
| `completion_time_sec`, `lk_attention_check` | cleaning | cleaning | cleaning | cleaning | Data quality filters |

## How the four algorithms connect

```
            Survey (25 Q)  ──►  Data engineering (clean, cap, impute, encode)
                                         │
     ┌───────────────────┬───────────────┼───────────────────┬──────────────────┐
     ▼                   ▼               ▼                   ▼                  ▼
 Descriptive       Classification    Clustering        Association rules    Regression
 who answered?     WILL they adopt?  WHO are they?     WHAT do they need    HOW MUCH will
                   (lead score)      (5 segments)      together? (MVP)      they pay? (price)
     │                   │               │                   │                  │
     └─────────► Target segments with high lead scores ◄─────┘                  │
                 get the MVP features that rules say must ship together,        │
                 at the price the regression supports  ◄────────────────────────┘
                                         │
                                         ▼
                           Business case (finance) & rollout (operations)
```

- **Classification → Clustering:** segments are built without the target, then adoption rate per segment checks that the segments matter commercially (2% to 77%).
- **Clustering → Association rules:** the MVP is sized on *adopters*; segment feature rates show which segment each feature wins.
- **Association rules → Regression:** the regression shows which features raise willingness to pay (delivery-fee comparison, smart basket, wishlist alerts), which decides what goes behind the Premium paywall.
- **Regression → Business case:** the revenue-maximising price feeds the 3-year projection.

## Which algorithm "works better" and how we know

Each technique answers a different question, so they are not ranked against each other. Within each technique several models are compared on the same split:

| Technique | Models compared | How accuracy is checked |
|---|---|---|
| Classification | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN, Naive Bayes | 5-fold CV ROC-AUC (selection), then held-out accuracy, precision, recall, F1, AUC and a **confusion matrix** for every model, vs a majority-class baseline |
| Clustering | K-Means (k = 2–9) vs Agglomerative (Ward) | Elbow, silhouette, Davies-Bouldin, Calinski-Harabasz; K-Means vs Ward agreement (ARI); recovery of the hidden personas (ARI, synthetic data only) |
| Association rules | Apriori | Support, confidence, lift, leverage, conviction; redundant rules pruned |
| Regression | OLS, Ridge, Lasso, Random Forest, Gradient Boosting | 5-fold CV R² (selection), then held-out R², MAE, RMSE vs a predict-the-mean baseline |
