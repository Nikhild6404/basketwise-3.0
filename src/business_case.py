"""Step 7 - Business case: turn the model outputs into a 3-year plan.

Combines what the algorithms found (who adopts, how big each segment is,
what Premium can cost) with clearly labelled planning ASSUMPTIONS to
produce conservative / base / optimistic projections. Every assumption is
in the ASSUMPTIONS dict below so it can be challenged and changed.

Nothing here is a forecast. The survey is synthetic, and the market,
conversion and cost figures are placeholders to be replaced with real
quotes and real survey results.

Run:  python src/business_case.py
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

import config as C
from utils import PALETTE, banner, save_results, save_table, savefig

ASSUMPTIONS = {
    # Funnel - downloads reached through marketing each year (placeholders)
    "downloads_per_year": {"Conservative": [25_000, 60_000, 110_000],
                           "Base": [50_000, 120_000, 220_000],
                           "Optimistic": [90_000, 200_000, 350_000]},
    "monthly_active_rate": 0.35,         # share of cumulative downloads still active
    # Stated WTP overstates real paying behaviour, so Premium conversion is set
    # far below the survey's "share willing" figure (see regression output)
    "premium_conversion_of_active": {"Conservative": .02, "Base": .04, "Optimistic": .06},
    "ad_revenue_per_free_mau_aed_month": 0.40,
    # Retailer partnerships: sponsored deals / referral fee per tracked basket
    "partner_revenue_per_mau_aed_month": {"Conservative": 0.20, "Base": 0.45, "Optimistic": 0.70},
    # Costs (AED per year)
    "team_cost": [420_000, 780_000, 1_150_000],
    "data_and_infra_cost": [120_000, 200_000, 300_000],   # price feeds, cloud, maps
    "cac_aed_per_download": 6.0,                          # paid + referral acquisition
    "app_store_fee_on_premium": 0.15,
}


def project(scenario: str, price: float, overrides: dict | None = None) -> pd.DataFrame:
    A = {**ASSUMPTIONS, **(overrides or {})}
    rows, cum = [], 0
    for yr in range(3):
        dl = A["downloads_per_year"][scenario][yr]
        cum += dl
        mau = cum * A["monthly_active_rate"]
        premium = mau * A["premium_conversion_of_active"][scenario]
        free = mau - premium
        rev_premium = premium * price * 12 * (1 - A["app_store_fee_on_premium"])
        rev_ads = free * A["ad_revenue_per_free_mau_aed_month"] * 12
        rev_partner = mau * A["partner_revenue_per_mau_aed_month"][scenario] * 12
        revenue = rev_premium + rev_ads + rev_partner
        cost = A["team_cost"][yr] + A["data_and_infra_cost"][yr] + dl * A["cac_aed_per_download"]
        rows.append(dict(scenario=scenario, year=yr + 1, downloads=dl, monthly_active_users=round(mau),
                         premium_subscribers=round(premium), revenue_premium=round(rev_premium),
                         revenue_ads=round(rev_ads), revenue_partners=round(rev_partner),
                         total_revenue=round(revenue), total_cost=round(cost),
                         operating_profit=round(revenue - cost)))
    return pd.DataFrame(rows)


def main() -> dict:
    banner("STEP 7  Business case (scenario projections)")
    results = json.loads(C.RESULTS_FILE.read_text())
    price = results["regression"]["best_price_aed"]
    plan = pd.concat([project(s, price) for s in ["Conservative", "Base", "Optimistic"]], ignore_index=True)
    save_table(plan, "business_case_3yr")
    print(f"Premium price used: AED {price}/month (from regression step)")
    print(plan[["scenario", "year", "monthly_active_users", "premium_subscribers",
                "total_revenue", "total_cost", "operating_profit"]].to_string(index=False))

    # --- Sensitivity: which levers make the Base case sustainable? ---------
    cacs = [3.0, 4.5, 6.0, 9.0]
    partner = [0.20, 0.45, 0.70, 1.00, 1.30]
    grid = []
    for cac in cacs:
        for pr in partner:
            t = project("Base", price, {"cac_aed_per_download": cac,
                                        "partner_revenue_per_mau_aed_month": {"Base": pr}})
            grid.append(dict(cac_aed=cac, partner_rev_per_mau_aed=pr,
                             year3_profit=int(t.iloc[-1].operating_profit),
                             cumulative_3yr_profit=int(t.operating_profit.sum())))
    sens = pd.DataFrame(grid)
    save_table(sens, "business_case_sensitivity")
    pivot = sens.pivot(index="cac_aed", columns="partner_rev_per_mau_aed", values="year3_profit")
    print("\nBase case, Year-3 operating profit (AED) by CAC (rows) and partner revenue per MAU (cols):")
    print(pivot.to_string())
    y3 = plan[(plan.scenario == "Base") & (plan.year == 3)].iloc[0]
    gap = y3.total_cost - y3.revenue_premium - y3.revenue_ads
    breakeven_partner = round(gap / (y3.monthly_active_users * 12), 2)
    print(f"Base Year 3 breaks even at partner revenue of AED {breakeven_partner} per MAU per month")

    fig, ax = plt.subplots(figsize=(7.5, 4))
    im = ax.imshow(pivot.values / 1000, cmap="RdYlGn", vmin=-1500, vmax=1500, aspect="auto")
    ax.set_xticks(range(len(partner)), [f"{p:.2f}" for p in partner])
    ax.set_yticks(range(len(cacs)), [f"{c:.1f}" for c in cacs])
    for i in range(len(cacs)):
        for j in range(len(partner)):
            ax.text(j, i, f"{pivot.values[i, j] / 1000:,.0f}k", ha="center", va="center", fontsize=8)
    ax.set_xlabel("Retailer partner revenue per active user (AED / month)")
    ax.set_ylabel("Acquisition cost per download (AED)")
    ax.set_title("Base case Year-3 operating profit (AED)")
    fig.colorbar(im, ax=ax, label="AED '000")
    savefig(fig, "20_business_case_sensitivity")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, s in enumerate(["Conservative", "Base", "Optimistic"]):
        t = plan[plan.scenario == s]
        ax.plot(t.year, t.operating_profit / 1000, "o-", color=PALETTE[i], label=s)
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks([1, 2, 3]); ax.set_xlabel("Year"); ax.set_ylabel("Operating profit (AED '000)")
    ax.set_title("3-year operating profit by scenario (planning assumptions)")
    ax.legend()
    savefig(fig, "19_business_case_profit")

    mix = plan[(plan.scenario == "Base") & (plan.year == 3)][["revenue_premium", "revenue_ads", "revenue_partners"]].iloc[0]
    res = dict(price_aed=price, assumptions=ASSUMPTIONS,
               year3_base=plan[(plan.scenario == "Base") & (plan.year == 3)].iloc[0].to_dict(),
               year3_revenue_mix_base={k: round(v / mix.sum(), 3) for k, v in mix.items()},
               breakeven_partner_rev_per_mau_year3_base=breakeven_partner,
               sensitivity=sens.to_dict("records"),
               first_profitable_year={s: (int(plan[(plan.scenario == s) & (plan.operating_profit > 0)].year.min())
                                          if (plan[(plan.scenario == s)].operating_profit > 0).any() else None)
                                      for s in ["Conservative", "Base", "Optimistic"]})
    save_results("business_case", res)
    return res


if __name__ == "__main__":
    main()
