"""Step 4 - Clustering: what kinds of grocery shoppers exist?

Business question: one marketing message will not work for a student
sharing a flat and a high-income family loyal to one store. Clustering
groups respondents by behaviour and attitude (no target column), so each
segment gets its own message, offer and channel.

Inputs : age, household size, income, spend, shopping frequency, number of
         stores, time spent comparing, the 5 attitude statements, online
         shopping, number of challenges and features ticked.
Not used: intent, NPS and WTP. They are kept aside to describe each
          cluster afterwards, so the segments are not built around the answer.

Method : standardise -> K-Means for k = 2..9 (elbow, silhouette,
         Davies-Bouldin, Calinski-Harabasz) -> choose k -> cross-check with
         Agglomerative (Ward) clustering -> profile and name each segment.
Validation only: the hidden personas in answer_key_true_personas.csv are
         compared with the clusters (Adjusted Rand Index). Real survey data
         will not have this key.

Run:  python src/clustering.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.preprocessing import StandardScaler

import config as C
from utils import PALETTE, banner, save_results, save_table, savefig

CLUSTER_FEATURES = ["age", "household_size", "income_ord", "spend_per_person_aed",
                    "trips_per_week", "stores_used_regularly", "minutes_comparing_week",
                    *C.LIKERT_COLS, "shops_online", "challenge_count", "feature_count"]
K_RANGE = range(2, 10)


def name_segments(profile: pd.DataFrame) -> dict:
    """Assign readable names from each cluster's profile, one name per cluster.

    Rules are applied in order of how distinctive the trait is; each rule
    takes the best remaining cluster.
    """
    remaining = list(profile.index)
    names = {}
    rules = [
        ("Loyal Convenience Shoppers", "lk_brand_loyal", True),
        ("Deal Hunters", "minutes_comparing_week", True),
        ("Students & Flat-sharers", "age", False),
        ("Busy Professionals", "shops_online", True),
        ("Budget Families", "household_size", True),
    ]
    for label, col, high in rules:
        if not remaining:
            break
        s = profile.loc[remaining, col]
        cid = s.idxmax() if high else s.idxmin()
        names[cid] = label
        remaining.remove(cid)
    for cid in remaining:
        names[cid] = f"Segment {cid}"
    return names


def main() -> dict:
    banner("STEP 4  Clustering: customer segments")
    df = pd.read_csv(C.CLEAN_FILE)
    X = df[CLUSTER_FEATURES].copy()
    X["minutes_comparing_week"] = np.log1p(X["minutes_comparing_week"])   # tame skew
    X["spend_per_person_aed"] = np.log(X["spend_per_person_aed"])
    Xs = StandardScaler().fit_transform(X)

    # --- Choosing k --------------------------------------------------------
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=20, random_state=C.RANDOM_STATE).fit(Xs)
        rows.append(dict(k=k, inertia=km.inertia_,
                         smallest_cluster_share=np.bincount(km.labels_).min() / len(Xs),
                         silhouette=silhouette_score(Xs, km.labels_),
                         davies_bouldin=davies_bouldin_score(Xs, km.labels_),
                         calinski_harabasz=calinski_harabasz_score(Xs, km.labels_)))
    kdf = pd.DataFrame(rows).round(3)
    save_table(kdf, "clustering_k_selection")
    print(kdf.to_string(index=False))

    # Decision rule (statistics + business):
    #  1. k >= 3: k=2 only splits "price-driven" vs "not", too coarse to act on
    #  2. silhouette within 0.01 of the best k>=3 (survey data rarely separates sharply)
    #  3. every segment >= 12% of respondents (big enough for its own campaign)
    #  4. k <= 6: a small team cannot run more distinct campaigns
    #  -> take the largest k that passes, i.e. the most detail we can still act on
    inertia = kdf.set_index("k")["inertia"]
    elbow_k = int((inertia.diff() - inertia.diff().shift(-1)).iloc[1:-1].idxmax())
    cand = kdf[(kdf.k >= 3) & (kdf.k <= 6)]
    best_sil = cand.silhouette.max()
    ok = cand[(cand.silhouette >= best_sil - 0.01) & (cand.smallest_cluster_share >= 0.12)]
    k_final = int(ok.k.max()) if len(ok) else int(cand.loc[cand.silhouette.idxmax(), "k"])
    print(f"Elbow suggests k={elbow_k}; k passing all rules: {ok.k.tolist()} -> chosen k={k_final}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(kdf.k, kdf.inertia, "o-"); axes[0].set_title("Elbow (inertia)")
    axes[1].plot(kdf.k, kdf.silhouette, "o-", color=PALETTE[2]); axes[1].set_title("Silhouette (higher = better)")
    axes[2].plot(kdf.k, kdf.davies_bouldin, "o-", color=PALETTE[1]); axes[2].set_title("Davies-Bouldin (lower = better)")
    for ax in axes:
        ax.axvline(k_final, color="k", ls="--", lw=.8); ax.set_xlabel("k (number of clusters)")
    savefig(fig, "09_cluster_k_selection")

    # --- Final model + cross-check ----------------------------------------
    km = KMeans(n_clusters=k_final, n_init=50, random_state=C.RANDOM_STATE).fit(Xs)
    labels = km.labels_
    agg = AgglomerativeClustering(n_clusters=k_final, linkage="ward").fit(Xs)
    agreement = adjusted_rand_score(labels, agg.labels_)

    df["cluster_id"] = labels
    profile = df.groupby("cluster_id")[CLUSTER_FEATURES].mean()
    names = name_segments(profile)
    df["segment"] = df["cluster_id"].map(names)

    seg = (df.groupby("segment")
             .agg(size=("respondent_id", "size"),
                  age=("age", "mean"), household_size=("household_size", "mean"),
                  income_ord=("income_ord", "mean"),
                  monthly_spend_aed=("monthly_grocery_spend_aed", "mean"),
                  stores_used=("stores_used_regularly", "mean"),
                  minutes_comparing=("minutes_comparing_week", "median"),
                  brand_loyal=("lk_brand_loyal", "mean"),
                  app_comfort=("lk_app_comfort", "mean"),
                  notif_annoyed=("lk_notif_annoyed", "mean"),
                  shops_online=("shops_online", "mean"),
                  student_share=("occupation", lambda s: (s == "Student").mean()),
                  adoption_rate=("is_adopter", "mean"),
                  avg_wtp_aed=("wtp_aed_month", "mean"),
                  premium_share=("preferred_plan", lambda s: (s == C.PLANS[2]).mean()),
                  nps=("nps_concept", lambda s: 100 * ((s >= 9).mean() - (s <= 6).mean())))
             .round(2))
    seg["share"] = (seg["size"] / seg["size"].sum()).round(3)
    seg = seg.sort_values("adoption_rate", ascending=False)
    feat_rate = df.groupby("segment")[C.FEATURE_COLS + C.CHALLENGE_COLS].mean().round(2)
    seg["top_features"] = [", ".join(feat_rate.loc[s, C.FEATURE_COLS].nlargest(3).index.str.replace("f_", ""))
                           for s in seg.index]
    seg["top_challenges"] = [", ".join(feat_rate.loc[s, C.CHALLENGE_COLS].nlargest(2).index.str.replace("ch_", ""))
                             for s in seg.index]
    save_table(seg.reset_index(), "cluster_profiles")
    save_table(feat_rate.reset_index(), "cluster_feature_challenge_rates")
    df[["respondent_id", "cluster_id", "segment"]].to_csv(C.DATA_PROCESSED / "segments.csv", index=False)
    print(seg[["size", "share", "age", "monthly_spend_aed", "minutes_comparing", "brand_loyal",
               "adoption_rate", "avg_wtp_aed", "top_features"]].to_string())

    # --- Validation against the hidden personas (synthetic data only) -------
    ari_true = None
    if C.ANSWER_KEY_FILE.exists():
        key = pd.read_csv(C.ANSWER_KEY_FILE)
        m = df.merge(key, on="respondent_id", how="left")
        ari_true = adjusted_rand_score(m["true_persona"], m["segment"])
        ct = pd.crosstab(m["segment"], m["true_persona"])
        save_table(ct.reset_index(), "cluster_vs_true_persona")
        name_match = float((m["segment"] == m["true_persona"]).mean())
        print(f"ARI vs hidden personas: {ari_true:.3f} | label match {name_match:.1%}")

    # --- Figures -------------------------------------------------------------
    pcs = PCA(2, random_state=C.RANDOM_STATE).fit(Xs)
    P = pcs.transform(Xs)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, s in enumerate(seg.index):
        mk = df["segment"] == s
        ax.scatter(P[mk, 0], P[mk, 1], s=9, alpha=.6, color=PALETTE[i], label=f"{s} ({mk.mean():.0%})")
    ax.set_xlabel(f"PC1 ({pcs.explained_variance_ratio_[0]:.0%} var)")
    ax.set_ylabel(f"PC2 ({pcs.explained_variance_ratio_[1]:.0%} var)")
    ax.set_title(f"K-Means segments (k={k_final}) projected on 2 principal components")
    ax.legend(fontsize=8, markerscale=2)
    savefig(fig, "10_cluster_pca_scatter")

    z = pd.DataFrame(Xs, columns=CLUSTER_FEATURES).assign(segment=df["segment"]).groupby("segment").mean()
    z = z.loc[seg.index]
    fig, ax = plt.subplots(figsize=(12, 4.2))
    im = ax.imshow(z.values, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(range(z.shape[1]), z.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(z)), z.index)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, f"{z.values[i, j]:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="standardised mean (z)")
    ax.set_title("Segment profiles (red = above average, blue = below)")
    savefig(fig, "11_cluster_profile_heatmap")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(seg.index, seg["adoption_rate"], color=PALETTE[:len(seg)])
    ax2 = ax.twinx(); ax2.plot(seg.index, seg["avg_wtp_aed"], "ko-", label="Avg WTP (AED/mo)")
    ax.set_ylabel("Adoption rate (Q25 = Yes)"); ax2.set_ylabel("Avg willingness to pay (AED)")
    ax.set_title("Which segments to target first"); ax.tick_params(axis="x", rotation=15)
    ax2.legend(loc="upper right"); ax2.spines["right"].set_visible(True)
    savefig(fig, "12_segment_adoption_wtp")

    res = dict(features=CLUSTER_FEATURES, k_selection=kdf.to_dict("records"), elbow_k=elbow_k,
               chosen_k=k_final, silhouette=float(kdf.set_index("k").loc[k_final, "silhouette"]),
               kmeans_vs_ward_ari=round(agreement, 3),
               ari_vs_hidden_personas=None if ari_true is None else round(ari_true, 3),
               segments=seg.reset_index()[["segment", "size", "share", "adoption_rate", "avg_wtp_aed",
                                           "premium_share", "nps", "top_features", "top_challenges"]]
               .to_dict("records"))
    save_results("clustering", res)
    return res


if __name__ == "__main__":
    main()
