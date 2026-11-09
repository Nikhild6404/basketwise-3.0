"""Step 3 - Classification: who will adopt BasketWise?

Business question: before spending on marketing, can we predict whether a
person will download and use the app (Q25 = "Yes")? If yes, we can score
any new lead and spend acquisition budget on the high-probability ones.

Target  : is_adopter (1 = answered "Yes" to Q25, 0 = "Maybe" or "No")
Inputs  : demographics, shopping behaviour, attitudes (Likert), challenges
          (Q15) and wanted features (Q16).
Excluded: nps_concept, wtp_aed_month, preferred_plan, most_important_feature,
          preferred_platform, alert_preference. These are answered after the
          concept is pitched, at the same moment as Q25, so they would leak the
          answer and could not be known for a new lead.

Six algorithms are compared on the same 75/25 stratified split with 5-fold
cross-validation; accuracy, precision, recall, F1, ROC-AUC and a confusion
matrix are reported for each. A 3-class model (No/Maybe/Yes) is added as an
extension.

Run:  python src/classification.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import config as C
from utils import PALETTE, banner, save_results, save_table, savefig

LEAKAGE_COLS = ["nps_concept", "nps_group", "wtp_aed_month", "preferred_plan",
                "most_important_feature", "preferred_platform", "alert_preference",
                "expected_saving_aed_month"]
ID_TARGET_COLS = ["respondent_id", "submitted_at", "completion_time_sec", C.ATTENTION_COL,
                  "intent_download", "intent_ord", C.CLASS_TARGET_BINARY]
# Raw categoricals already represented by engineered numeric versions
REDUNDANT = ["income_band_aed", "promo_notifs_per_week", "share_receipts_for_rewards",
             "age_group", "income_mid_aed"]


def feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[c for c in LEAKAGE_COLS + ID_TARGET_COLS + REDUNDANT if c in df])
    return X.astype(float), df[C.CLASS_TARGET_BINARY]


def models() -> dict:
    rs = C.RANDOM_STATE
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=rs),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=5, random_state=rs, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=250, learning_rate=0.05,
                                                        max_depth=3, random_state=rs),
        "K-Nearest Neighbours": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)),
        "Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
    }


def main() -> dict:
    banner("STEP 3  Classification: predicting adopters")
    df = pd.read_csv(C.MODEL_FILE)
    X, y = feature_matrix(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y,
                                              random_state=C.RANDOM_STATE)
    cv = StratifiedKFold(5, shuffle=True, random_state=C.RANDOM_STATE)
    baseline_acc = max(y_te.mean(), 1 - y_te.mean())

    rows, fitted, probas = [], {}, {}
    for name, m in models().items():
        cv_auc = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="roc_auc")
        m.fit(X_tr, y_tr)
        p = m.predict_proba(X_te)[:, 1]
        pred = (p >= 0.5).astype(int)
        fitted[name], probas[name] = m, p
        rows.append(dict(model=name, cv_auc_mean=cv_auc.mean(), cv_auc_std=cv_auc.std(),
                         test_accuracy=accuracy_score(y_te, pred),
                         test_precision=precision_score(y_te, pred),
                         test_recall=recall_score(y_te, pred),
                         test_f1=f1_score(y_te, pred),
                         test_auc=roc_auc_score(y_te, p)))
    comp = pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False).round(3)
    save_table(comp, "classification_model_comparison")
    print(comp.to_string(index=False))
    print(f"Baseline accuracy (always predict majority class): {baseline_acc:.3f}")

    best_name = comp.iloc[0]["model"]
    best = fitted[best_name]
    best_pred = (probas[best_name] >= 0.5).astype(int)
    report = classification_report(y_te, best_pred, target_names=["Maybe/No", "Yes"], output_dict=True)
    save_table(pd.DataFrame(report).T.round(3), "classification_best_model_report", index=True)

    # --- Confusion matrices for every model --------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    for ax, (name, p) in zip(axes.ravel(), probas.items()):
        cm = confusion_matrix(y_te, (p >= .5).astype(int))
        ConfusionMatrixDisplay(cm, display_labels=["Maybe/No", "Yes"]).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d")
        acc = (cm[0, 0] + cm[1, 1]) / cm.sum()
        ax.set_title(f"{name}\nacc {acc:.2f}" + ("  (best)" if name == best_name else ""))
    fig.suptitle("Confusion matrices on the held-out test set (n = %d)" % len(y_te), fontweight="bold")
    fig.tight_layout()
    savefig(fig, "05_confusion_matrices")

    # --- ROC curves -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_te, p)
        ax.plot(fpr, tpr, lw=2.2 if name == best_name else 1.2,
                label=f"{name} (AUC {roc_auc_score(y_te, p):.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves - adopter classifier"); ax.legend(fontsize=8, loc="lower right")
    savefig(fig, "06_roc_curves")

    # --- What drives adoption? (permutation importance, model-agnostic) ----
    imp = permutation_importance(best, X_te, y_te, scoring="roc_auc", n_repeats=15,
                                 random_state=C.RANDOM_STATE, n_jobs=-1)
    imp_df = (pd.DataFrame({"feature": X.columns, "importance": imp.importances_mean,
                            "std": imp.importances_std})
              .sort_values("importance", ascending=False).round(4))
    save_table(imp_df, "classification_feature_importance")
    lr = fitted["Logistic Regression"]
    coef = pd.DataFrame({"feature": X.columns,
                         "std_coefficient": lr[-1].coef_[0]}).sort_values("std_coefficient", key=abs,
                                                                          ascending=False).round(3)
    save_table(coef, "classification_logit_coefficients")

    top = imp_df.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color=PALETTE[0])
    ax.set_title(f"Top drivers of adoption ({best_name}, permutation importance)")
    ax.set_xlabel("Drop in test AUC when the column is shuffled")
    savefig(fig, "07_adoption_drivers")

    # --- Business use: lead scoring tiers on the whole sample --------------
    best.fit(X, y)  # refit on all data for scoring
    score = best.predict_proba(X)[:, 1]
    tiers = pd.cut(score, [0, .35, .65, 1], labels=["Low", "Medium", "High"], include_lowest=True)
    leads = pd.DataFrame({"respondent_id": df["respondent_id"], "adoption_probability": score.round(3),
                          "lead_tier": tiers, "actual_intent": df["intent_download"]})
    save_table(leads, "lead_scores")
    tier_summary = (leads.groupby("lead_tier", observed=False)
                    .agg(people=("respondent_id", "size"),
                         share_of_sample=("respondent_id", lambda s: len(s) / len(leads)),
                         actual_yes_rate=("actual_intent", lambda s: (s == "Yes").mean()))
                    .round(3).reset_index())
    save_table(tier_summary, "lead_tier_summary")
    print("\nLead tiers:\n", tier_summary.to_string(index=False))

    # --- Extension: 3-class model (No / Maybe / Yes) --------------------------
    y3 = df["intent_download"]
    X3_tr, X3_te, y3_tr, y3_te = train_test_split(X, y3, test_size=.25, stratify=y3,
                                                  random_state=C.RANDOM_STATE)
    m3 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
    m3.fit(X3_tr, y3_tr)
    p3 = m3.predict(X3_te)
    acc3 = accuracy_score(y3_te, p3)
    cm3 = confusion_matrix(y3_te, p3, labels=C.INTENT_LEVELS)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ConfusionMatrixDisplay(cm3, display_labels=C.INTENT_LEVELS).plot(ax=ax, cmap="Purples",
                                                                    colorbar=False, values_format="d")
    ax.set_title(f"3-class intent (No/Maybe/Yes)\nmultinomial logistic, acc {acc3:.2f}")
    savefig(fig, "08_confusion_matrix_3class")
    print(f"3-class accuracy: {acc3:.3f} (baseline {y3_te.value_counts(normalize=True).max():.3f})")

    best_row = comp.iloc[0].to_dict()
    cm_best = confusion_matrix(y_te, best_pred)
    res = dict(target="is_adopter (Q25 = Yes)", n_features=X.shape[1], n_train=len(y_tr), n_test=len(y_te),
               positive_rate=round(float(y.mean()), 3), baseline_accuracy=round(float(baseline_acc), 3),
               best_model=best_name, best=best_row,
               best_confusion_matrix=dict(tn=cm_best[0, 0], fp=cm_best[0, 1], fn=cm_best[1, 0], tp=cm_best[1, 1]),
               comparison=comp.to_dict("records"),
               top_drivers=imp_df.head(8)[["feature", "importance"]].values.tolist(),
               lead_tiers=tier_summary.to_dict("records"),
               three_class_accuracy=round(acc3, 3),
               three_class_baseline=round(float(y3_te.value_counts(normalize=True).max()), 3))
    save_results("classification", res)
    return res


if __name__ == "__main__":
    main()
