"""BasketWise UAE - interactive analytics dashboard.

Run locally:   streamlit run streamlit_app.py
Deploy:        Streamlit Community Cloud -> New app -> main file: streamlit_app.py

The app reuses the project's own modules in src/, so every number here
matches REPORT.md. Models are trained once per session and cached.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import association as A  # noqa: E402
import business_case as B  # noqa: E402
import clean_transform as CT  # noqa: E402
import config as C  # noqa: E402
from classification import feature_matrix  # noqa: E402
from classification import models as clf_models  # noqa: E402
from clustering import CLUSTER_FEATURES, name_segments  # noqa: E402
from utils import PALETTE  # noqa: E402

st.set_page_config(page_title="BasketWise UAE Analytics", page_icon="🛒", layout="wide")

# Informational deltas (e.g. "serve 94% of adopters") should not show an up/down arrow.
# `delta_arrow` exists only in newer Streamlit releases, so use it when available.
import inspect  # noqa: E402

NOARROW = {"delta_arrow": "off"} if "delta_arrow" in inspect.signature(st.metric).parameters else {}

LABELS = {**{k: v for k, v in C.CHALLENGES.items()}, **{k: v for k, v in C.FEATURES.items()}}
SHORT = lambda c: c.replace("ch_", "").replace("f_", "").replace("_", " ")  # noqa: E731
SEG_COLORS = {"Deal Hunters": PALETTE[0], "Students & Flat-sharers": PALETTE[1],
              "Budget Families": PALETTE[2], "Busy Professionals": PALETTE[3],
              "Loyal Convenience Shoppers": PALETTE[8]}


# =============================================================================
# Data & models (cached)
# =============================================================================
@st.cache_data(show_spinner="Loading survey data...")
def load_data():
    if not C.CLEAN_FILE.exists() or not C.MODEL_FILE.exists():
        if not C.RAW_FILE.exists():
            import generate_data
            generate_data.main()
        CT.main()
    raw = pd.read_csv(C.RAW_FILE)
    clean = pd.read_csv(C.CLEAN_FILE)
    model = pd.read_csv(C.MODEL_FILE)
    noise = pd.read_csv(C.NOISE_LOG_FILE) if C.NOISE_LOG_FILE.exists() else pd.DataFrame()
    _, log = CT.clean(raw)
    results = json.loads(C.RESULTS_FILE.read_text()) if C.RESULTS_FILE.exists() else {}
    return raw, clean, model, noise, log, results


@st.cache_resource(show_spinner="Training classifiers...")
def train_classifiers():
    from sklearn.model_selection import train_test_split
    _, _, model, *_ = load_data()
    X, y = feature_matrix(model)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=C.RANDOM_STATE)
    fitted = {}
    for name, m in clf_models().items():
        fitted[name] = m.fit(X_tr, y_tr)
    return fitted, X_tr, X_te, y_tr, y_te, list(X.columns)


@st.cache_resource(show_spinner="Training willingness-to-pay models...")
def train_regressors():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    _, _, model, *_ = load_data()
    df = model[model[C.REG_TARGET].notna()].reset_index(drop=True)
    X, _ = feature_matrix(df)
    y = df[C.REG_TARGET].astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=C.RANDOM_STATE)
    rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=8, random_state=C.RANDOM_STATE,
                               n_jobs=-1).fit(X_tr, y_tr)
    ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))).fit(X_tr, y_tr)
    return {"Random Forest": rf, "Ridge": ridge}, X_te, y_te


def cluster_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df[CLUSTER_FEATURES].astype(float).copy()
    X["minutes_comparing_week"] = np.log1p(X["minutes_comparing_week"])
    X["spend_per_person_aed"] = np.log(X["spend_per_person_aed"].clip(lower=1))
    return X


@st.cache_resource(show_spinner="Clustering respondents...")
def fit_clusters(k: int):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    _, clean, *_ = load_data()
    scaler = StandardScaler().fit(cluster_matrix(clean))
    Xs = scaler.transform(cluster_matrix(clean))
    km = KMeans(n_clusters=k, n_init=20, random_state=C.RANDOM_STATE).fit(Xs)
    prof = clean.assign(cluster_id=km.labels_).groupby("cluster_id")[CLUSTER_FEATURES].mean()
    names = name_segments(prof)
    pca = PCA(2, random_state=C.RANDOM_STATE).fit(Xs)
    return dict(scaler=scaler, km=km, names=names, labels=km.labels_,
                pcs=pca.transform(Xs), var=pca.explained_variance_ratio_,
                silhouette=silhouette_score(Xs, km.labels_), Xs=Xs)


@st.cache_data(show_spinner="Mining association rules...")
def get_rules(min_support: float, min_conf: float, min_lift: float, adopters_only: bool):
    _, clean, *_ = load_data()
    d = clean[clean.is_adopter == 1] if adopters_only else clean
    rules, freq = A.mine(d[C.CHALLENGE_COLS + C.FEATURE_COLS], min_support, min_conf, min_lift)
    return rules, len(freq)


def build_profile(inputs: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn simulator inputs into (engineered row, model-ready row)."""
    _, clean, *_ = load_data()
    base_cols = list(pd.read_csv(C.RAW_FILE, nrows=1).columns)
    row = clean.iloc[[0]][[c for c in base_cols if c in clean.columns]].copy()
    for k, v in inputs.items():
        row[k] = v
    row["respondent_id"] = "NEW"
    row["intent_download"] = "Maybe"  # placeholder; never used as an input
    eng = CT.engineer(row)
    both = CT.model_ready(pd.concat([clean, eng], ignore_index=True))
    return eng, both.iloc[[-1]]


def fmt_aed(x: float) -> str:
    return f"AED {x:,.0f}"


def plot(fig, height: int | None = None):
    fig.update_layout(margin=dict(l=10, r=10, t=80, b=10), height=height, legend_title_text="",
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
                      title=dict(y=0.98, yanchor="top"))
    st.plotly_chart(fig, width="stretch")


raw, clean, model_df, noise, clean_log, R = load_data()

# =============================================================================
# Sidebar
# =============================================================================
st.sidebar.title("🛒 BasketWise UAE")
st.sidebar.caption("Validating a grocery price-comparison app with data analytics")
PAGES = ["Overview", "Data & cleaning", "Descriptive analytics", "Classification",
         "Clustering", "Association rules", "Regression & pricing", "Business case",
         "Customer simulator"]
page = st.sidebar.radio("Go to", PAGES, label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.info("Synthetic data: 2,000 simulated respondents with deliberate noise. "
                "Results show the method works, not real market demand.", icon="ℹ️")
st.sidebar.caption("Code & report: see the GitHub repository README.")


# =============================================================================
# Pages
# =============================================================================
if page == "Overview":
    st.title("BasketWise UAE — business idea validation")
    st.markdown(
        "**The idea:** an app that tells UAE households which store is cheapest for their *whole* "
        "grocery basket, including delivery fees, and only notifies them when items on their own "
        "list genuinely drop in price.\n\n"
        "**The question:** is it worth building, for whom, with which features, at what price? "
        "A 25-question survey was designed so that every answer feeds a specific algorithm.")
    c1, c2, c3, c4, c5 = st.columns(5)
    clf, clu, asc, reg = (R.get(k, {}) for k in ["classification", "clustering", "association", "regression"])
    c1.metric("Valid respondents", f"{len(clean):,}", f"−{len(raw) - len(clean)} removed in cleaning",
              delta_color="off", **NOARROW)
    c2.metric("Would download", f"{clean.is_adopter.mean():.0%}")
    c3.metric("Adopter model AUC", f"{clf.get('best', {}).get('test_auc', 0):.2f}",
              f"{clf.get('best_model', '')}", delta_color="off", **NOARROW)
    c4.metric("MVP features", f"{asc.get('mvp_size', 4)}", "serve 94% of adopters", delta_color="off", **NOARROW)
    c5.metric("Premium price", f"AED {reg.get('best_price_aed', 9)}/mo")

    st.subheader("How the analysis fits together")
    steps = pd.DataFrame([
        ["Data engineering", "Is the data usable?", "Clean 2,040 raw rows → 1,825 valid", "Same script runs on the real survey"],
        ["Descriptive", "Is the problem real?", "Median spend AED 2,350; 47% want only big-saving alerts", "'Quiet savings' positioning"],
        ["Classification", "Who will adopt?", "AUC 0.85; high lead tier converts at 82%", "Spend acquisition money on high-tier leads"],
        ["Clustering", "What shopper types exist?", "5 segments; adoption from 2% to 77%", "Launch with Deal Hunters"],
        ["Association rules", "Which features go together?", "price compare + history → alerts (90%)", "4-feature MVP"],
        ["Regression", "What will people pay?", "R² 0.44; revenue peaks at AED 9", "Premium AED 9/mo"],
        ["Business case", "Is it sustainable?", "Base case breaks even only with retailer partnerships", "Pilot with kill criteria"],
    ], columns=["Step", "Business question", "Finding", "Decision"])
    st.dataframe(steps, hide_index=True, width="stretch")

    st.subheader("Five customer segments")
    seg = pd.DataFrame(clu.get("segments", []))
    if not seg.empty:
        fig = px.scatter(seg, x="adoption_rate", y="avg_wtp_aed", size="size", color="segment",
                         color_discrete_map=SEG_COLORS, size_max=60, text="segment",
                         labels={"adoption_rate": "Adoption rate", "avg_wtp_aed": "Avg willingness to pay (AED/mo)"})
        fig.update_traces(textposition="top center", showlegend=False)
        fig.update_xaxes(tickformat=".0%", range=[-0.05, 0.95])
        plot(fig, 420)
        st.caption("Top-right is where to start: segments that both adopt and pay.")

elif page == "Data & cleaning":
    st.title("Data & cleaning")
    st.markdown("The raw export was deliberately made messy — the way real survey data arrives. "
                "Each cleaning rule below is applied in order and logged.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Raw rows", f"{len(raw):,}")
    c2.metric("Injected data problems", f"{len(noise):,}")
    c3.metric("Valid respondents after cleaning", f"{len(clean):,}")

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Cleaning log")
        st.dataframe(clean_log[["step", "action", "rule", "rows_affected", "rows_remaining"]],
                     hide_index=True, width="stretch", height=560)
    with right:
        st.subheader("What was injected")
        if not noise.empty:
            nt = noise["noise_type"].value_counts().reset_index()
            nt.columns = ["problem", "count"]
            fig = px.bar(nt.sort_values("count"), x="count", y="problem", orientation="h",
                         color_discrete_sequence=[PALETTE[1]])
            plot(fig, 420)
        st.caption("Flipped labels cannot be detected, so no model can be perfect — as with careless "
                   "answers in a real survey.")

    st.subheader("Explore the data")
    which = st.segmented_control("Dataset", ["Clean", "Raw (messy)"], default="Clean")
    data = clean if which != "Raw (messy)" else raw
    c1, c2 = st.columns(2)
    em = c1.multiselect("Emirate", sorted(data["emirate"].dropna().astype(str).unique()))
    it = c2.multiselect("Intent (Q25)", C.INTENT_LEVELS[::-1])
    view = data
    if em:
        view = view[view["emirate"].astype(str).isin(em)]
    if it:
        view = view[view["intent_download"].isin(it)]
    st.dataframe(view, width="stretch", height=320)
    st.caption(f"{len(view):,} rows shown")
    d1, d2 = st.columns(2)
    d1.download_button("Download raw CSV", raw.to_csv(index=False), "basketwise_survey_raw.csv", "text/csv")
    d2.download_button("Download clean CSV", clean.to_csv(index=False), "basketwise_survey_clean.csv", "text/csv")

elif page == "Descriptive analytics":
    st.title("Descriptive analytics — knowing the customer")
    d = R.get("descriptive", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median grocery spend", fmt_aed(clean.monthly_grocery_spend_aed.median()) + "/mo")
    c2.metric("Median time comparing", f"{clean.minutes_comparing_week.median():.0f} min/week")
    c3.metric("Use 3+ stores", f"{(clean.stores_used_regularly >= 3).mean():.0%}")
    c4.metric("Concept NPS", f"{d.get('concept_nps', 0):+.0f}")

    st.subheader("Adoption rate by group")
    opts = {"Income band": ("income_band_aed", C.INCOME_BANDS),
            "Age group": ("age_group", ["16-24", "25-34", "35-44", "45-54", "55+"]),
            "Occupation": ("occupation", C.OCCUPATIONS), "Shopping channel": ("primary_channel", C.CHANNELS),
            "Emirate": ("emirate", C.EMIRATES), "Household size": ("household_size", None),
            "Notification preference": ("alert_preference", C.ALERT_PREFS)}
    choice = st.selectbox("Break down by", list(opts))
    col, order = opts[choice]
    g = clean.groupby(col).agg(respondents=("is_adopter", "size"), adoption=("is_adopter", "mean"),
                               wtp=("wtp_aed_month", "mean")).reset_index()
    if order:
        g[col] = pd.Categorical(g[col], order, ordered=True)
        g = g.sort_values(col)
    fig = px.bar(g, x=col, y="adoption", text=g["adoption"].map("{:.0%}".format),
                 hover_data={"respondents": True, "wtp": ":.1f"}, color_discrete_sequence=[PALETTE[0]],
                 labels={"adoption": "Share answering Yes", col: choice})
    fig.add_hline(y=clean.is_adopter.mean(), line_dash="dash", line_color=PALETTE[1],
                  annotation_text="overall")
    fig.update_yaxes(tickformat=".0%")
    plot(fig, 380)
    if col == "emirate":
        st.caption("Emirate was generated independently of intent in this synthetic data — differences are noise.")

    st.subheader("Pain points and wanted features")
    kind = st.segmented_control("Show", ["Challenges (Q15)", "Features (Q16)"], default="Features (Q16)")
    cols = C.CHALLENGE_COLS if kind == "Challenges (Q15)" else C.FEATURE_COLS
    p = pd.DataFrame({"item": [LABELS[c] for c in cols],
                      "Adopters (Yes)": clean.loc[clean.is_adopter == 1, cols].mean().values,
                      "Maybe / No": clean.loc[clean.is_adopter == 0, cols].mean().values})
    p = p.sort_values("Adopters (Yes)")
    fig = go.Figure()
    fig.add_bar(y=p["item"], x=p["Adopters (Yes)"], orientation="h", name="Adopters (Yes)", marker_color=PALETTE[2])
    fig.add_bar(y=p["item"], x=p["Maybe / No"], orientation="h", name="Maybe / No", marker_color="#b0b0b0")
    fig.update_xaxes(tickformat=".0%", title="Share who ticked")
    plot(fig, 440)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(clean, x="monthly_grocery_spend_aed", nbins=40, color_discrete_sequence=[PALETTE[0]],
                           labels={"monthly_grocery_spend_aed": "Monthly grocery spend (AED)"},
                           title="Monthly grocery spend")
        plot(fig, 320)
    with c2:
        n = clean["nps_concept"].value_counts().sort_index().reset_index()
        n.columns = ["score", "count"]
        n["group"] = np.where(n.score <= 6, "Detractor", np.where(n.score <= 8, "Passive", "Promoter"))
        fig = px.bar(n, x="score", y="count", color="group", title="Likelihood to recommend (0–10)",
                     color_discrete_map={"Detractor": PALETTE[1], "Passive": PALETTE[4], "Promoter": PALETTE[2]})
        plot(fig, 320)

elif page == "Classification":
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                                 recall_score, roc_auc_score, roc_curve)
    st.title("Classification — who will adopt?")
    st.markdown("**Target:** Q25 = *Yes* (41% of respondents). **Inputs:** 57 pre-pitch answers. "
                "NPS, willingness to pay and plan choice are excluded because they are answered at the "
                "same moment as Q25 and would leak the answer.")
    fitted, X_tr, X_te, y_tr, y_te, cols = train_classifiers()

    comp_path = C.TAB_DIR / "classification_model_comparison.csv"
    if comp_path.exists():
        st.subheader("Model comparison")
        st.dataframe(pd.read_csv(comp_path), hide_index=True, width="stretch")
        st.caption(f"Model chosen on 5-fold cross-validated AUC (training data only). "
                   f"Baseline accuracy (always predict 'not adopter'): {1 - y_te.mean():.3f}.")

    st.subheader("Confusion matrix — try a different model or threshold")
    c1, c2 = st.columns([1, 2])
    name = c1.selectbox("Model", list(fitted), index=0)
    thr = c1.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05,
                    help="Probability above which a person is predicted as an adopter. Lower = catch more adopters, "
                         "but more false alarms.")
    p = fitted[name].predict_proba(X_te)[:, 1]
    pred = (p >= thr).astype(int)
    cm = confusion_matrix(y_te, pred)
    c1.metric("Accuracy", f"{accuracy_score(y_te, pred):.3f}")
    c1.metric("Precision", f"{precision_score(y_te, pred, zero_division=0):.3f}")
    c1.metric("Recall", f"{recall_score(y_te, pred):.3f}")
    c1.metric("F1", f"{f1_score(y_te, pred):.3f}")
    c1.metric("ROC-AUC", f"{roc_auc_score(y_te, p):.3f}")
    with c2:
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=["Predicted Maybe/No", "Predicted Yes"], y=["Actual Maybe/No", "Actual Yes"])
        fig.update_layout(coloraxis_showscale=False, title=f"{name} — test set (n = {len(y_te)})")
        plot(fig, 380)
        st.caption(f"True positives {cm[1, 1]} · True negatives {cm[0, 0]} · "
                   f"False positives {cm[0, 1]} (wasted ad spend) · False negatives {cm[1, 0]} (missed adopters)")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for n, m in fitted.items():
            pp = m.predict_proba(X_te)[:, 1]
            fpr, tpr, _ = roc_curve(y_te, pp)
            fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{n} ({roc_auc_score(y_te, pp):.2f})",
                            line=dict(width=3 if n == name else 1.3))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="grey"), showlegend=False)
        fig.update_layout(title="ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate",
                          legend=dict(orientation="v", x=0.45, y=0.05, yanchor="bottom"))
        st.plotly_chart(fig, width="stretch")
    with c2:
        imp_path = C.TAB_DIR / "classification_feature_importance.csv"
        if imp_path.exists():
            imp = pd.read_csv(imp_path).head(12).iloc[::-1]
            fig = px.bar(imp, x="importance", y="feature", orientation="h", error_x="std",
                         color_discrete_sequence=[PALETTE[0]], title="Top drivers of adoption (permutation importance)")
            plot(fig, 450)

    st.subheader("Lead scoring — the business use")
    lt = C.TAB_DIR / "lead_tier_summary.csv"
    if lt.exists():
        t = pd.read_csv(lt)
        cols3 = st.columns(len(t))
        for col_, r in zip(cols3, t.itertuples()):
            col_.metric(f"{r.lead_tier} tier", f"{r.actual_yes_rate:.0%} say Yes", f"{r.share_of_sample:.0%} of people",
                        delta_color="off", **NOARROW)
        st.markdown("**Decision:** paid acquisition targets only High-tier profiles; Medium gets low-cost nurturing; "
                    "Low is not targeted.")

elif page == "Clustering":
    st.title("Clustering — what kinds of shoppers exist?")
    st.markdown("K-Means on 15 standardised behaviour and attitude variables. Adoption, NPS and willingness to "
                "pay are **not** inputs — they are used afterwards to see which segments matter.")
    ksel = C.TAB_DIR / "clustering_k_selection.csv"
    c1, c2 = st.columns([1, 2])
    k = c1.slider("Number of clusters (k)", 2, 8, 5,
                  help="k = 5 was chosen: acceptable silhouette, every segment ≥ 12% of people, ≤ 6 campaigns.")
    res = fit_clusters(k)
    c1.metric("Silhouette", f"{res['silhouette']:.3f}")
    seg_names = pd.Series(res["labels"]).map(res["names"])
    c1.metric("Smallest segment", f"{seg_names.value_counts(normalize=True).min():.0%} of people")
    if ksel.exists():
        kd = pd.read_csv(ksel)
        with c2:
            fig = go.Figure()
            fig.add_scatter(x=kd.k, y=kd.silhouette, mode="lines+markers", name="Silhouette (higher better)")
            fig.add_scatter(x=kd.k, y=kd.davies_bouldin / 10, mode="lines+markers",
                            name="Davies-Bouldin ÷ 10 (lower better)")
            fig.add_vline(x=k, line_dash="dash")
            fig.update_layout(title="Choosing k", xaxis_title="k")
            plot(fig, 300)

    pc = pd.DataFrame(res["pcs"], columns=["PC1", "PC2"]).assign(segment=seg_names.values)
    fig = px.scatter(pc, x="PC1", y="PC2", color="segment", opacity=0.6,
                     color_discrete_map=SEG_COLORS,
                     title=f"Segments projected on 2 principal components "
                           f"({res['var'][0]:.0%} + {res['var'][1]:.0%} of variance)")
    fig.update_traces(marker=dict(size=5))
    plot(fig, 480)

    prof = clean.assign(segment=seg_names.values).groupby("segment").agg(
        size=("respondent_id", "size"), age=("age", "mean"), household=("household_size", "mean"),
        spend_aed=("monthly_grocery_spend_aed", "mean"), minutes_comparing=("minutes_comparing_week", "median"),
        brand_loyal=("lk_brand_loyal", "mean"), shops_online=("shops_online", "mean"),
        adoption=("is_adopter", "mean"), avg_wtp_aed=("wtp_aed_month", "mean")).round(2)
    prof["share"] = prof["size"] / prof["size"].sum()
    prof = prof.sort_values("adoption", ascending=False)
    st.subheader("Segment profiles")
    st.dataframe(prof.style.format({"share": "{:.0%}", "adoption": "{:.0%}", "shops_online": "{:.0%}",
                                    "spend_aed": "{:,.0f}", "age": "{:.0f}", "household": "{:.1f}",
                                    "avg_wtp_aed": "{:.1f}", "minutes_comparing": "{:.0f}",
                                    "brand_loyal": "{:.1f}"}), width="stretch")
    z = pd.DataFrame(res["Xs"], columns=CLUSTER_FEATURES).assign(segment=seg_names.values).groupby("segment").mean()
    fig = px.imshow(z.loc[prof.index].round(1), text_auto=True, color_continuous_scale="RdBu_r",
                    zmin=-1.5, zmax=1.5, aspect="auto", title="Standardised profile (red = above average)")
    plot(fig, 360)

    if k == 5 and C.ANSWER_KEY_FILE.exists():
        from sklearn.metrics import adjusted_rand_score
        key = pd.read_csv(C.ANSWER_KEY_FILE)
        m = clean[["respondent_id"]].assign(segment=seg_names.values).merge(key, on="respondent_id")
        st.success(f"Validation (synthetic data only): clusters recover the hidden personas with "
                   f"ARI = {adjusted_rand_score(m.true_persona, m.segment):.3f}; "
                   f"{(m.true_persona == m.segment).mean():.0%} land in the matching segment.")

elif page == "Association rules":
    st.title("Association rules — what goes together?")
    st.markdown("Apriori on the **17 tick-box items only** (7 challenges from Q15, 10 features from Q16). "
                "Rules whose extra items add less than 3 points of confidence over a simpler rule are pruned.")
    c1, c2, c3, c4 = st.columns(4)
    sup = c1.slider("Min support", 0.05, 0.30, 0.08, 0.01)
    conf = c2.slider("Min confidence", 0.40, 0.95, 0.60, 0.05)
    lift = c3.slider("Min lift", 1.0, 2.0, 1.10, 0.05)
    adopters_only = c4.toggle("Adopters only", False, help="Mine rules only among people who said Yes")
    rules, n_items = get_rules(sup, conf, lift, adopters_only)
    fams = ["challenge -> challenge", "challenge -> feature", "feature -> feature"]
    fam = st.multiselect("Rule family", fams + ["mixed -> feature", "mixed -> challenge", "feature -> challenge"],
                         default=fams)
    contains = st.selectbox("Rules involving item", ["(any)"] + [SHORT(c) for c in C.CHALLENGE_COLS + C.FEATURE_COLS])
    view = rules[rules.family.isin(fam)] if len(rules) else rules
    if contains != "(any)" and len(view):
        view = view[view.rule.str.contains(contains.replace(" ", "_"), regex=False)]
    st.caption(f"{n_items:,} frequent itemsets · {len(rules):,} rules after pruning · {len(view):,} shown")
    st.dataframe(view.head(200), hide_index=True, width="stretch", height=380)
    if len(view):
        fig = px.scatter(view, x="support", y="confidence", size=(view.lift - 0.95).clip(lower=0.05),
                         color="family", hover_name="rule", title="Support vs confidence (bubble size = lift)")
        plot(fig, 420)

    st.subheader("How many features should the MVP launch with?")
    cov = A.greedy_coverage(clean, C.FEATURE_COLS)
    fig = go.Figure()
    fig.add_bar(x=[f"{s}. {SHORT(f)}" for s, f in zip(cov.step, cov.feature_added)], y=cov.marginal_gain,
                name="Marginal gain", marker_color=PALETTE[4])
    fig.add_scatter(x=[f"{s}. {SHORT(f)}" for s, f in zip(cov.step, cov.feature_added)],
                    y=cov.adopters_well_served, mode="lines+markers", name="Adopters well served",
                    line=dict(color=PALETTE[2], width=3))
    fig.update_yaxes(tickformat=".0%")
    fig.add_vline(x=3.5, line_dash="dash")
    plot(fig, 380)
    st.markdown("**Decision:** launch with **smart basket, price compare, wishlist alerts and shared list** — "
                "94% of adopters get at least two features they asked for. Price history ships together with "
                "wishlist alerts (90% of people wanting compare + history also want alerts).")

elif page == "Regression & pricing":
    from sklearn.metrics import mean_absolute_error, r2_score
    st.title("Regression — how much will people pay?")
    st.markdown("**Target:** Q20, the most someone would pay per month for Premium (AED; 0 = nothing).")
    regs, X_te, y_te = train_regressors()
    comp = C.TAB_DIR / "regression_model_comparison.csv"
    if comp.exists():
        st.dataframe(pd.read_csv(comp), hide_index=True, width="stretch")
    c1, c2 = st.columns([1, 2])
    rname = c1.radio("Model", list(regs))
    pr = np.clip(regs[rname].predict(X_te), 0, None)
    c1.metric("Test R²", f"{r2_score(y_te, pr):.3f}")
    c1.metric("Test MAE", f"AED {mean_absolute_error(y_te, pr):.2f}")
    c1.metric("Baseline MAE (predict mean)", f"AED {mean_absolute_error(y_te, np.full(len(y_te), y_te.mean())):.2f}")
    with c2:
        fig = px.scatter(x=y_te, y=pr, opacity=0.5, labels={"x": "Actual WTP (AED)", "y": "Predicted WTP (AED)"},
                         title="Actual vs predicted (test set)", color_discrete_sequence=[PALETTE[0]])
        mx = float(max(y_te.max(), pr.max()))
        fig.add_scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(dash="dash", color="grey"), showlegend=False)
        plot(fig, 380)

    coef = C.TAB_DIR / "regression_ridge_coefficients.csv"
    if coef.exists():
        cf = pd.read_csv(coef).head(14).sort_values("aed_per_1sd")
        fig = px.bar(cf, x="aed_per_1sd", y="feature", orientation="h",
                     color=np.where(cf.aed_per_1sd > 0, "raises WTP", "lowers WTP"),
                     color_discrete_map={"raises WTP": PALETTE[2], "lowers WTP": PALETTE[1]},
                     title="What moves willingness to pay (Ridge, AED per 1 standard deviation)")
        plot(fig, 440)

    st.subheader("Pricing simulator")
    pool = clean[(clean.is_adopter == 1) & clean.wtp_aed_month.notna()]
    seg_df = pd.read_csv(C.DATA_PROCESSED / "segments.csv") if (C.DATA_PROCESSED / "segments.csv").exists() else None
    segs = ["All adopters"] + (sorted(seg_df.segment.unique()) if seg_df is not None else [])
    sp = st.selectbox("Audience", segs)
    if sp != "All adopters" and seg_df is not None:
        pool = pool.merge(seg_df, on="respondent_id")
        pool = pool[pool.segment == sp]
    price = st.slider("Premium price (AED / month)", 3, 30, 9)
    prices = np.arange(3, 31)
    share = np.array([(pool.wtp_aed_month >= p).mean() for p in prices])
    best = int(prices[np.argmax(prices * share)])
    c1, c2, c3 = st.columns(3)
    c1.metric("Share willing at this price", f"{(pool.wtp_aed_month >= price).mean():.0%}")
    c2.metric("Revenue per 100 people", f"AED {100 * price * (pool.wtp_aed_month >= price).mean():,.0f}/mo")
    c3.metric("Revenue-maximising price", f"AED {best}")
    fig = go.Figure()
    fig.add_scatter(x=prices, y=prices * share, mode="lines+markers", name="Revenue per person (AED)",
                    line=dict(color=PALETTE[2], width=3))
    fig.add_scatter(x=prices, y=share * (prices * share).max(), mode="lines", name="Share willing (scaled)",
                    line=dict(dash="dot", color="grey"))
    fig.add_vline(x=price, line_dash="dash")
    fig.update_layout(xaxis_title="Price (AED / month)", title=f"Price-response curve — {sp} (n = {len(pool)})")
    plot(fig, 380)
    st.caption("Stated willingness to pay overstates real payment; the business case assumes only 2–6% of active "
               "users subscribe.")

elif page == "Business case":
    st.title("Business case — is it sustainable?")
    st.markdown("Every assumption below is a **placeholder** you can change. The price comes from the regression step.")
    defaults = B.ASSUMPTIONS
    c1, c2, c3 = st.columns(3)
    price = c1.number_input("Premium price (AED/mo)", 3, 40, int(R.get("regression", {}).get("best_price_aed", 9)))
    base_dl = defaults["downloads_per_year"]["Base"]
    dl_cols = c1.columns(3)
    dls = [dl_cols[i].number_input(f"Y{i + 1} downloads", 1_000, 1_000_000, base_dl[i], 5_000)
           for i in range(3)]
    active = c2.slider("Monthly active rate", 0.10, 0.70, defaults["monthly_active_rate"], 0.05)
    conv = c2.slider("Premium conversion of active users", 0.005, 0.15, 0.04, 0.005, format="%.3f")
    ads = c2.slider("Ad revenue per free user (AED/mo)", 0.0, 2.0, defaults["ad_revenue_per_free_mau_aed_month"], 0.05)
    partner = c3.slider("Retailer partner revenue per user (AED/mo)", 0.0, 2.5, 0.45, 0.05)
    cac = c3.slider("Acquisition cost per download (AED)", 0.5, 15.0, defaults["cac_aed_per_download"], 0.5)
    team_scale = c3.slider("Team & data cost (× base plan)", 0.5, 2.0, 1.0, 0.1)
    plan = B.project("Base", price, {
        "downloads_per_year": {"Base": dls}, "monthly_active_rate": active,
        "premium_conversion_of_active": {"Base": conv}, "ad_revenue_per_free_mau_aed_month": ads,
        "partner_revenue_per_mau_aed_month": {"Base": partner}, "cac_aed_per_download": cac,
        "team_cost": [x * team_scale for x in defaults["team_cost"]],
        "data_and_infra_cost": [x * team_scale for x in defaults["data_and_infra_cost"]]})
    m1, m2, m3, m4 = st.columns(4)
    y3 = plan.iloc[-1]
    m1.metric("Year-3 active users", f"{y3.monthly_active_users:,.0f}")
    m2.metric("Year-3 Premium subscribers", f"{y3.premium_subscribers:,.0f}")
    m3.metric("Year-3 revenue", fmt_aed(y3.total_revenue))
    m4.metric("Year-3 operating profit", fmt_aed(y3.operating_profit),
              "profitable" if y3.operating_profit > 0 else "-loss-making")
    long = plan.melt(id_vars="year", value_vars=["revenue_premium", "revenue_ads", "revenue_partners"],
                     var_name="stream", value_name="AED")
    long["stream"] = long["stream"].str.replace("revenue_", "").str.title()
    fig = px.bar(long, x="year", y="AED", color="stream", title="Revenue by stream vs total cost",
                 color_discrete_sequence=[PALETTE[3], PALETTE[4], PALETTE[2]])
    fig.add_scatter(x=plan.year, y=plan.total_cost, mode="lines+markers", name="Total cost",
                    line=dict(color=PALETTE[1], width=3))
    fig.update_xaxes(tickvals=[1, 2, 3], ticktext=["Year 1", "Year 2", "Year 3"])
    plot(fig, 420)
    st.dataframe(plan.drop(columns="scenario"), hide_index=True, width="stretch")
    gap = y3.total_cost - y3.revenue_premium - y3.revenue_ads
    st.info(f"With these settings, Year 3 breaks even when retailer partners pay about "
            f"**AED {gap / (y3.monthly_active_users * 12):.2f} per active user per month**.")

elif page == "Customer simulator":
    st.title("Customer simulator")
    st.markdown("Describe a shopper. The trained models predict whether they would adopt BasketWise, "
                "what they would pay, and which segment they belong to.")
    with st.form("sim"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", 16, 80, 34)
        hh = c1.number_input("Household size", 1, 15, 4)
        inc = c1.selectbox("Monthly household income (AED)", C.INCOME_BANDS, index=2)
        occ = c1.selectbox("Occupation", C.OCCUPATIONS, index=1)
        emi = c1.selectbox("Emirate", C.EMIRATES)
        spend = c2.number_input("Monthly grocery spend (AED)", 200, 20_000, 2_500, 100)
        trips = c2.slider("Grocery trips per week", 0, 10, 3)
        stores = c2.slider("Stores / apps used regularly", 1, 8, 4)
        minutes = c2.slider("Minutes per week comparing prices", 0, 300, 60, 5)
        chan = c2.selectbox("Main channel", C.CHANNELS, index=0)
        lk1 = c3.slider("I regularly compare prices", 1, 5, 4)
        lk2 = c3.slider("I'd switch stores to save 5%", 1, 5, 4)
        lk3 = c3.slider("I stick to the same store & brands", 1, 5, 2)
        lk4 = c3.slider("I'm comfortable shopping in an app", 1, 5, 4)
        lk5 = c3.slider("Promo notifications annoy me", 1, 5, 4)
        ch = st.multiselect("Challenges faced (Q15)", C.CHALLENGE_COLS, format_func=lambda c: LABELS[c],
                            default=["ch_price_inconsistency", "ch_expired_offers", "ch_time_comparing"])
        ft = st.multiselect("Features they'd use (Q16)", C.FEATURE_COLS, format_func=lambda c: LABELS[c],
                            default=["f_smart_basket", "f_price_compare", "f_wishlist_alerts"])
        go_ = st.form_submit_button("Predict", type="primary")
    inputs = dict(age=age, household_size=hh, income_band_aed=inc, occupation=occ, emirate=emi,
                  monthly_grocery_spend_aed=spend, trips_per_week=trips, stores_used_regularly=stores,
                  minutes_comparing_week=minutes, primary_channel=chan, lk_compare_often=lk1,
                  lk_switch_for_savings=lk2, lk_brand_loyal=lk3, lk_app_comfort=lk4, lk_notif_annoyed=lk5,
                  **{c: int(c in ch) for c in C.CHALLENGE_COLS}, **{f: int(f in ft) for f in C.FEATURE_COLS})
    eng, mrow = build_profile(inputs)
    fitted, *_, cols = train_classifiers()
    Xrow = mrow.reindex(columns=cols, fill_value=0).astype(float)
    prob = float(fitted["Logistic Regression"].predict_proba(Xrow)[:, 1][0])
    regs, *_ = train_regressors()
    wtp = float(max(0, regs["Random Forest"].predict(Xrow)[0]))
    cres = fit_clusters(5)
    seg = cres["names"][int(cres["km"].predict(cres["scaler"].transform(cluster_matrix(eng)))[0])]
    tier = "High" if prob > 0.65 else "Medium" if prob > 0.35 else "Low"
    c1, c2, c3 = st.columns(3)
    c1.metric("Adoption probability", f"{prob:.0%}", f"{tier} lead tier", delta_color="off", **NOARROW)
    c2.metric("Predicted willingness to pay", f"AED {wtp:.1f}/mo",
              "would likely buy Premium at AED 9" if wtp >= 9 else "free tier user", delta_color="off", **NOARROW)
    c3.metric("Segment", seg)
    advice = {
        "Deal Hunters": "Launch audience. Lead with all-time-low alerts and the cheapest-basket finder.",
        "Budget Families": "Volume segment. Message: 'save on the weekly family shop', shared list.",
        "Students & Flat-sharers": "Adopts but rarely pays. Monetise with ads and partner offers; shared list.",
        "Busy Professionals": "Phase-2 upsell. Message: 'true total across delivery apps' (Premium).",
        "Loyal Convenience Shoppers": "Do not spend acquisition budget here.",
    }
    st.info(f"**Recommended action:** {advice.get(seg, '')}")
