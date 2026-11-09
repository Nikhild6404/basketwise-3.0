"""Step 5 - Association rule mining: which pains and features go together?

Business question: we cannot build 10 features at launch. Which small set
should the MVP include, and which features must ship together because
people who want A and B also expect C? And which pain points travel
together (if a shopper faces challenge 1 and 3, do they also face 5)?

Inputs : ONLY the tick-all-that-apply items - Q15 challenges (7 columns) and
         Q16 features (10 columns). Association rules need yes/no "basket"
         data; age, spend or Likert scores are not items and are excluded.
Method : Apriori (min support 8%) -> rules with confidence >= 60% and
         lift > 1.1 -> split into three rule families:
           challenge -> challenge   (pain points that co-occur)
           challenge -> feature     (which feature answers which pain)
           feature   -> feature     (which features must ship together)
         Then a greedy coverage analysis picks the MVP size.

Run:  python src/association.py
"""
from __future__ import annotations

import re

import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

import config as C
from utils import PALETTE, banner, save_results, save_table, savefig

MIN_SUPPORT, MIN_CONF, MIN_LIFT = 0.08, 0.60, 1.10


def _fmt(items) -> str:
    return " + ".join(sorted(re.sub(r"^(ch_|f_)", "", i) for i in items))


def _family(row) -> str:
    a = "challenge" if all(i.startswith("ch_") for i in row.antecedents) else (
        "feature" if all(i.startswith("f_") for i in row.antecedents) else "mixed")
    c = "challenge" if all(i.startswith("ch_") for i in row.consequents) else (
        "feature" if all(i.startswith("f_") for i in row.consequents) else "mixed")
    return f"{a} -> {c}"


def prune_redundant(rules: pd.DataFrame, tol: float = 0.03) -> pd.DataFrame:
    """Drop a rule if a simpler rule (subset of its antecedents, same
    consequent) already reaches within `tol` of its confidence: the extra
    items add nothing a manager could act on."""
    keep = []
    by_cons = {c: g for c, g in rules.groupby(rules["consequents"])}
    for idx, r in rules.iterrows():
        simpler = by_cons[r.consequents]
        simpler = simpler[simpler.antecedents.apply(lambda a: a < r.antecedents)]
        keep.append(not (simpler.confidence >= r.confidence - tol).any())
    return rules[keep]


def top_diverse(t: pd.DataFrame, n: int = 15, per_consequent: int = 3) -> pd.DataFrame:
    """Top rules by lift, at most `per_consequent` rules per right-hand side."""
    cons = t["rule"].str.split("->").str[1].str.strip()
    return t[cons.groupby(cons).cumcount() < per_consequent].head(n)


def mine(basket: pd.DataFrame, min_support: float = MIN_SUPPORT, min_conf: float = MIN_CONF,
         min_lift: float = MIN_LIFT, prune: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    freq = apriori(basket.astype(bool), min_support=min_support, use_colnames=True, max_len=4)
    cols = ["family", "rule", "support", "confidence", "lift", "leverage", "conviction"]
    if freq.empty:
        return pd.DataFrame(columns=cols), freq
    rules = association_rules(freq, num_itemsets=len(basket), metric="confidence", min_threshold=min_conf)
    rules = rules[(rules["lift"] > min_lift) & (rules["consequents"].apply(len) == 1)].copy()
    if rules.empty:
        return pd.DataFrame(columns=cols), freq
    if prune:
        rules = prune_redundant(rules)
    rules["family"] = rules.apply(_family, axis=1)
    rules["rule"] = rules["antecedents"].apply(_fmt) + "  ->  " + rules["consequents"].apply(_fmt)
    cols = ["family", "rule", "support", "confidence", "lift", "leverage", "conviction"]
    return rules.sort_values(["lift", "confidence"], ascending=False)[cols].round(3), freq


def greedy_coverage(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Add features one at a time, each time picking the one that leaves the
    most adopters 'well served'.

    Well served = the MVP contains at least 2 of the features the adopter
    ticked (or their only one, if they ticked just one). Needing two keeps
    the test honest: almost everyone wants *something* from a 1-feature app.
    The first pick is the feature most adopters want (with a 2-feature rule
    no single feature serves anyone, so the greedy step has no signal yet).
    """
    adopters = df[df.is_adopter == 1]
    need = adopters[feats].sum(axis=1).clip(upper=2)
    popularity = adopters[feats].mean()
    chosen, rows, prev = [], [], 0.0
    for step in range(1, len(feats) + 1):
        if not chosen:
            f = popularity.idxmax()
        else:
            gains = {f: (adopters[chosen + [f]].sum(axis=1) >= need).mean()
                     for f in feats if f not in chosen}
            f = max(gains, key=gains.get)
        chosen.append(f)
        served = float((adopters[chosen].sum(axis=1) >= need).mean())
        rows.append(dict(step=step, feature_added=f, adopters_well_served=round(served, 3),
                         marginal_gain=round(served - prev, 3),
                         share_of_adopters_wanting_it=round(float(popularity[f]), 3)))
        prev = served
    return pd.DataFrame(rows)


def main() -> dict:
    banner("STEP 5  Association rule mining: challenges & features")
    df = pd.read_csv(C.CLEAN_FILE)
    basket = df[C.CHALLENGE_COLS + C.FEATURE_COLS]

    rules, freq = mine(basket)
    save_table(rules, "association_rules_all")
    fam_tables = {}
    for fam in ["challenge -> challenge", "challenge -> feature", "feature -> feature"]:
        t = top_diverse(rules[rules.family == fam])
        fam_tables[fam] = t
        save_table(t, "association_rules_" + fam.replace(" -> ", "_to_"))
        print(f"\nTop {fam} rules:\n" + t.head(6)[["rule", "support", "confidence", "lift"]].to_string(index=False))

    # Rules among adopters only - what the people we will actually win want
    rules_adopt, _ = mine(df.loc[df.is_adopter == 1, C.FEATURE_COLS])
    save_table(rules_adopt.head(20), "association_rules_adopters_features")

    # MVP sizing
    cov = greedy_coverage(df, C.FEATURE_COLS)
    save_table(cov, "mvp_feature_coverage")
    # Stop adding features once the next one serves < 5% more adopters
    late = cov[(cov.step >= 2) & (cov.marginal_gain < 0.05)]
    mvp_n = int(late.step.min() - 1) if len(late) else 4
    mvp = cov.head(mvp_n)["feature_added"].tolist()
    # Completeness check: add any feature that MVP features strongly imply
    implied = []
    for r in rules_adopt.itertuples():
        ants = [a.strip() for a in r.rule.split("->")[0].split(" + ")]
        cons = r.rule.split("->")[1].strip()
        if (all(f"f_{a}" in mvp for a in ants) and f"f_{cons}" not in mvp
                and r.confidence >= 0.75 and f"f_{cons}" not in implied):
            implied.append(f"f_{cons}")
    print(f"\nGreedy coverage suggests an MVP of {mvp_n} features: {mvp}")
    if implied:
        print(f"Rules say MVP users also expect: {implied}")

    # --- Figures ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, fam in enumerate(["challenge -> challenge", "challenge -> feature", "feature -> feature", "mixed -> feature"]):
        t = rules[rules.family == fam]
        if len(t):
            ax.scatter(t.support, t.confidence, s=(t.lift - 1) * 250 + 10, alpha=.55,
                       color=PALETTE[i], label=f"{fam} ({len(t)})")
    ax.set_xlabel("Support"); ax.set_ylabel("Confidence")
    ax.set_title("Association rules (bubble size = lift)"); ax.legend(fontsize=8)
    savefig(fig, "13_association_rules_scatter")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for ax, fam, col in [(axes[0], "challenge -> feature", PALETTE[1]), (axes[1], "feature -> feature", PALETTE[0])]:
        t = fam_tables[fam].head(10).iloc[::-1]
        ax.barh(t["rule"], t["lift"], color=col)
        for y, (l, c) in enumerate(zip(t["lift"], t["confidence"])):
            ax.text(l + .01, y, f"conf {c:.0%}", va="center", fontsize=8)
        ax.set_title(f"Top {fam} rules by lift"); ax.set_xlabel("Lift")
        ax.set_xlim(1, t["lift"].max() * 1.18)
    fig.tight_layout()
    savefig(fig, "14_top_rules_by_lift")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(cov.step, cov.adopters_well_served, "o-", color=PALETTE[2],
            label="adopters well served (>= 2 of their wanted features)")
    ax.bar(cov.step, cov.marginal_gain, color=PALETTE[4], alpha=.6, label="marginal gain")
    ax.axvline(mvp_n + .5, color="k", ls="--", lw=.8)
    ax.text(mvp_n + .6, .55, f"MVP = first {mvp_n}", fontsize=9, fontweight="bold")
    ax.set_xticks(cov.step, [f"{s}. {f.replace('f_', '')}" for s, f in zip(cov.step, cov.feature_added)],
                  rotation=35, ha="right", fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_title("How many features should the MVP launch with?"); ax.legend(loc="lower right")
    ax.set_xlabel("Features in the order the greedy search adds them")
    savefig(fig, "15_mvp_feature_coverage")

    def top(fam, n=5):
        return fam_tables[fam].head(n)[["rule", "support", "confidence", "lift"]].values.tolist()

    res = dict(items_used=C.CHALLENGE_COLS + C.FEATURE_COLS, min_support=MIN_SUPPORT,
               min_confidence=MIN_CONF, min_lift=MIN_LIFT, n_frequent_itemsets=len(freq),
               n_rules=len(rules), rules_by_family=rules.family.value_counts().to_dict(),
               top_challenge_to_challenge=top("challenge -> challenge"),
               top_challenge_to_feature=top("challenge -> feature"),
               top_feature_to_feature=top("feature -> feature"),
               mvp_size=mvp_n, mvp_features=mvp, mvp_implied_features=implied,
               coverage_curve=cov[["step", "feature_added", "adopters_well_served", "marginal_gain"]].values.tolist())
    save_results("association", res)
    return res


if __name__ == "__main__":
    main()
