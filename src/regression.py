"""Step 6 - Regression: how much will people pay, and what drives it?

Business question: what should BasketWise Premium cost, and which customer
traits and features raise willingness to pay (WTP)? This feeds the finance
side: price point, expected premium revenue and which features to put
behind the paywall.

Target : wtp_aed_month (Q20, maximum AED per month for Premium; 0 = would not pay)
Inputs : the same pre-pitch variables as the classifier (demographics,
         behaviour, attitudes, challenges, features).
Excluded: intent, NPS and preferred plan (answered at the same moment as
          Q20, so they would leak the answer).

Five models are compared with 5-fold CV and a held-out test set (R2, MAE,
RMSE) against a "predict the mean" baseline. The best model's drivers are
reported, and a simple price-response curve picks the Premium price.

Run:  python src/regression.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as C
from classification import feature_matrix
from utils import PALETTE, banner, save_results, save_table, savefig

PRICE_POINTS = np.arange(5, 31, 1)


def models() -> dict:
    rs = C.RANDOM_STATE
    return {
        "Linear Regression (OLS)": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))),
        "Lasso": make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=rs, max_iter=200000, tol=1e-3)),
        "Random Forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=8, random_state=rs, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.04, max_depth=3,
                                                       subsample=0.9, random_state=rs),
    }


def main() -> dict:
    banner("STEP 6  Regression: willingness to pay for Premium")
    df = pd.read_csv(C.MODEL_FILE)
    df = df[df[C.REG_TARGET].notna()].reset_index(drop=True)
    X, _ = feature_matrix(df)
    y = df[C.REG_TARGET].astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=C.RANDOM_STATE)
    cv = KFold(5, shuffle=True, random_state=C.RANDOM_STATE)
    base_pred = np.full(len(y_te), y_tr.mean())

    rows, fitted, preds = [], {}, {}
    for name, m in models().items():
        cv_r2 = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="r2")
        m.fit(X_tr, y_tr)
        p = np.clip(m.predict(X_te), 0, None)
        fitted[name], preds[name] = m, p
        rows.append(dict(model=name, cv_r2_mean=cv_r2.mean(), cv_r2_std=cv_r2.std(),
                         test_r2=r2_score(y_te, p), test_mae=mean_absolute_error(y_te, p),
                         test_rmse=mean_squared_error(y_te, p) ** .5))
    rows.append(dict(model="Baseline (predict mean)", cv_r2_mean=0, cv_r2_std=0, test_r2=r2_score(y_te, base_pred),
                     test_mae=mean_absolute_error(y_te, base_pred),
                     test_rmse=mean_squared_error(y_te, base_pred) ** .5))
    comp = pd.DataFrame(rows).sort_values("cv_r2_mean", ascending=False).round(3)
    save_table(comp, "regression_model_comparison")
    print(comp.to_string(index=False))
    best_name = comp.iloc[0]["model"]
    best = fitted[best_name]

    # --- Drivers: linear coefficients (direction) + permutation importance -----
    # price_seeker_index is an exact combination of three Likert items, which
    # makes plain OLS coefficients unstable (multicollinearity). For reading
    # the direction and size of effects we refit Ridge without that composite.
    Xi = X.drop(columns=["price_seeker_index"])
    interp = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))).fit(Xi, y)
    coef = (pd.DataFrame({"feature": Xi.columns,
                          "aed_per_1sd": interp[-1].coef_})
            .sort_values("aed_per_1sd", key=abs, ascending=False).round(4))
    # unscaled effect for binary features = AED change when the box is ticked
    sd = Xi.std()
    coef["aed_if_ticked_or_per_unit"] = (coef["aed_per_1sd"] / coef["feature"].map(sd)).round(4)
    save_table(coef, "regression_ridge_coefficients")
    imp = permutation_importance(best, X_te, y_te, scoring="r2", n_repeats=15,
                                 random_state=C.RANDOM_STATE, n_jobs=-1)
    imp_df = (pd.DataFrame({"feature": X.columns, "importance": imp.importances_mean, "std": imp.importances_std})
              .sort_values("importance", ascending=False).round(4))
    save_table(imp_df, "regression_feature_importance")
    feat_premium = coef[coef.feature.isin(C.FEATURE_COLS)].sort_values("aed_if_ticked_or_per_unit", ascending=False)

    # --- Pricing: revenue curve among likely adopters -----------------------
    raw = pd.read_csv(C.CLEAN_FILE)
    pool = raw[(raw.is_adopter == 1) & raw[C.REG_TARGET].notna()]
    curve = pd.DataFrame({"price_aed": PRICE_POINTS})
    curve["share_willing"] = [(pool[C.REG_TARGET] >= p).mean() for p in PRICE_POINTS]
    curve["revenue_index"] = curve["price_aed"] * curve["share_willing"]
    best_price = int(curve.loc[curve.revenue_index.idxmax(), "price_aed"])
    curve = curve.round(3)
    save_table(curve, "pricing_revenue_curve")
    print(f"\nBest model: {best_name} | revenue-maximising Premium price among adopters: AED {best_price}/month")

    # --- Figures -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    p = preds[best_name]
    axes[0].scatter(y_te, p, s=10, alpha=.5, color=PALETTE[0])
    lim = max(y_te.max(), p.max()) + 2
    axes[0].plot([0, lim], [0, lim], "k--", lw=.8)
    axes[0].set_xlabel("Actual WTP (AED/month)"); axes[0].set_ylabel("Predicted WTP")
    axes[0].set_title(f"{best_name}: actual vs predicted (test R2 {r2_score(y_te, p):.2f})")
    axes[1].scatter(p, y_te - p, s=10, alpha=.5, color=PALETTE[1])
    axes[1].axhline(0, color="k", lw=.8)
    axes[1].set_xlabel("Predicted WTP"); axes[1].set_ylabel("Residual (actual - predicted)")
    axes[1].set_title("Residuals: wider spread at higher WTP, many exact zeros")
    savefig(fig, "16_regression_actual_vs_predicted")

    top = imp_df.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color=PALETTE[3])
    ax.set_title(f"Top drivers of willingness to pay ({best_name})")
    ax.set_xlabel("Drop in test R2 when the column is shuffled")
    savefig(fig, "17_wtp_drivers")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(curve.price_aed, curve.revenue_index, "o-", color=PALETTE[2], label="price x share willing")
    ax2 = ax.twinx(); ax2.plot(curve.price_aed, curve.share_willing, "s--", color=PALETTE[8], ms=4,
                               label="share of adopters willing")
    ax.axvline(best_price, color="k", ls=":", lw=1)
    ax.set_xlabel("Premium price (AED / month)"); ax.set_ylabel("Revenue per adopter (AED)")
    ax2.set_ylabel("Share willing to pay"); ax2.spines["right"].set_visible(True)
    ax.set_title(f"Price-response curve: revenue peaks at AED {best_price}")
    ax.legend(loc="upper right"); ax2.legend(loc="center right")
    savefig(fig, "18_pricing_revenue_curve")

    res = dict(target="wtp_aed_month (Q20)", n=len(y), n_train=len(y_tr), n_test=len(y_te),
               mean_wtp=round(float(y.mean()), 2), median_wtp=float(y.median()),
               share_zero=round(float((y == 0).mean()), 3),
               best_model=best_name, best=comp.iloc[0].to_dict(), comparison=comp.to_dict("records"),
               top_drivers=imp_df.head(8)[["feature", "importance"]].values.tolist(),
               features_adding_wtp=feat_premium.head(5)[["feature", "aed_if_ticked_or_per_unit"]].values.tolist(),
               income_effect_aed_per_band=float(coef.set_index("feature").loc["income_ord", "aed_if_ticked_or_per_unit"]),
               best_price_aed=best_price,
               share_adopters_willing_at_best_price=float(curve.set_index("price_aed").loc[best_price, "share_willing"]),
               mean_wtp_adopters=round(float(pool[C.REG_TARGET].mean()), 2))
    save_results("regression", res)
    return res


if __name__ == "__main__":
    main()
