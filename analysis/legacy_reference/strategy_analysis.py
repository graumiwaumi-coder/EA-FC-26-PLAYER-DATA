"""
Deeper strategy analysis, addressing:
1. Weekly cycle: is there a real, tighter pattern in price vs a player's own weekly
   average, by day of week (the Champs/rewards/sell-off cycle), especially for
   higher-liquidity (more actively traded) cards?
2. Promo/event release timing: for cards that appear mid-season (price history starts
   well after the dataset's start date -- our best available proxy for "this is a new
   special/promo card release" since we have no explicit promo-name field), what's the
   typical price trajectory after release, and which (buy-day-after-release, hold-length)
   combination historically produced the best forward returns?
3. Month-of-year seasonality (single season only -- explicitly caveated, not yet
   proven to repeat next year).

Run: python3 strategy_analysis.py
"""
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

DATA_DIR = "../data"


def weekly_cycle():
    print("=" * 70)
    print("1. WEEKLY CYCLE (price vs player's own 7-day average, by day of week)")
    print("=" * 70)
    dataset = ds.dataset(f"{DATA_DIR}/prices_features.parquet", format="parquet")
    cols = ["player_id", "platform", "date", "rating", "price_clean", "ma_7",
            "liquidity_14", "day_of_week"]
    df = dataset.to_table(columns=cols, filter=(ds.field("rating") >= 75) & (ds.field("platform") == "pc")).to_pandas()
    df = df.dropna(subset=["price_clean", "ma_7", "liquidity_14"])
    df["dev_from_own_weekly_avg"] = df["price_clean"] / df["ma_7"] - 1

    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    df["dow_name"] = df["day_of_week"].map(dict(enumerate(dow_names)))

    liq_median = df["liquidity_14"].median()
    high_liq = df[df["liquidity_14"] >= df["liquidity_14"].quantile(0.75)]
    low_liq = df[df["liquidity_14"] <= df["liquidity_14"].quantile(0.25)]

    print(f"\nALL rating>=75 PC players (n={len(df)}): avg deviation from own trailing 7-day avg, by day:")
    print((df.groupby("dow_name")["dev_from_own_weekly_avg"].mean().reindex(dow_names) * 100).round(2).astype(str) + "%")

    print(f"\nHIGH liquidity (top quartile, most actively traded, n={len(high_liq)}):")
    print((high_liq.groupby("dow_name")["dev_from_own_weekly_avg"].mean().reindex(dow_names) * 100).round(2).astype(str) + "%")

    print(f"\nLOW liquidity (bottom quartile, n={len(low_liq)}):")
    print((low_liq.groupby("dow_name")["dev_from_own_weekly_avg"].mean().reindex(dow_names) * 100).round(2).astype(str) + "%")


def promo_release_timing():
    print("\n" + "=" * 70)
    print("2. PROMO/RELEASE TIMING (cards whose price history starts mid-season)")
    print("=" * 70)
    dataset = ds.dataset(f"{DATA_DIR}/prices_features.parquet", format="parquet")
    cols = ["player_id", "platform", "date", "rating", "price_clean"]
    df = dataset.to_table(columns=cols, filter=(ds.field("platform") == "pc") & (ds.field("rating") >= 75)).to_pandas()
    df = df.dropna(subset=["price_clean"])
    df = df.sort_values(["player_id", "date"])

    first_date_overall = df["date"].min()
    first_seen = df.groupby("player_id")["date"].min().rename("release_date")
    df = df.merge(first_seen, on="player_id")

    # "new release" = first appears more than 14 days after the dataset's own start
    # (so we're not just picking up cards that were simply always there)
    cutoff = first_date_overall + pd.Timedelta(days=14)
    releases = df[df["release_date"] > cutoff].copy()
    releases["days_since_release"] = (releases["date"] - releases["release_date"]).dt.days

    n_release_players = releases["player_id"].nunique()
    print(f"\nFound {n_release_players} players (rating>=75) whose price history starts mid-season "
          f"-- used as a proxy for 'new special/promo card' since there's no explicit promo-name field.")

    # normalize each player's price to their own release-day price
    day0 = releases[releases["days_since_release"] == 0][["player_id", "price_clean"]].rename(
        columns={"price_clean": "release_day_price"}
    )
    releases = releases.merge(day0, on="player_id")
    releases["price_ratio_to_release"] = releases["price_clean"] / releases["release_day_price"]

    print("\nAverage price trajectory after release (price as multiple of release-day price), rating>=75:")
    traj = releases[releases["days_since_release"].between(0, 45)].groupby("days_since_release")[
        "price_ratio_to_release"
    ].agg(["mean", "median", "count"])
    print(traj.loc[[0, 1, 2, 3, 5, 7, 10, 14, 21, 28, 35, 45]].round(3))

    print("\nSame, split by rating tier (85+ vs 75-84), since elite promos may behave differently:")
    for label, mask in [("85+ rated", releases["rating"] >= 85), ("75-84 rated", releases["rating"] < 85)]:
        sub = releases[mask]
        t = sub[sub["days_since_release"].between(0, 45)].groupby("days_since_release")["price_ratio_to_release"].median()
        idx = [d for d in [0, 3, 7, 14, 21, 30] if d in t.index]
        print(f"  {label} (n_players={sub['player_id'].nunique()}): " +
              ", ".join(f"day{d}={t[d]:.2f}x" for d in idx))

    # grid search: for each buy day (days since release) and hold length, what was the
    # average/median forward return? uses only players with enough history after release.
    print("\nGrid search: buy X days after release, hold Y days -- median return (rating>=75, all release cohorts):")
    pivot = releases.pivot_table(index=["player_id"], columns="days_since_release", values="price_clean")
    results = []
    for buy_day in (0, 3, 7, 14, 21):
        if buy_day not in pivot.columns:
            continue
        for hold in (7, 14, 21, 30):
            sell_day = buy_day + hold
            if sell_day not in pivot.columns:
                continue
            buy_price = pivot[buy_day]
            sell_price = pivot[sell_day]
            valid = buy_price.notna() & sell_price.notna() & (buy_price > 0)
            if valid.sum() < 20:
                continue
            ret = (sell_price[valid] * 0.95 - buy_price[valid]) / buy_price[valid]
            results.append((buy_day, hold, valid.sum(), ret.mean(), ret.median(), (ret > 0).mean()))

    res_df = pd.DataFrame(results, columns=["buy_day", "hold_days", "n", "mean_ret", "median_ret", "win_rate"])
    print(res_df.sort_values("median_ret", ascending=False).to_string(index=False))


def month_seasonality():
    print("\n" + "=" * 70)
    print("3. MONTH-OF-YEAR SEASONALITY (single season only -- not yet cross-validated)")
    print("=" * 70)
    dataset = ds.dataset(f"{DATA_DIR}/prices_features.parquet", format="parquet")
    cols = ["player_id", "platform", "date", "rating", "fwd_return_21d_net_tax"]
    df = dataset.to_table(columns=cols, filter=(ds.field("platform") == "pc") & (ds.field("rating") >= 75)).to_pandas()
    df = df.dropna(subset=["fwd_return_21d_net_tax"])
    df["month"] = df["date"].dt.to_period("M").astype(str)
    print("\nMedian 21-day net-of-tax forward return by calendar month of purchase (rating>=75, PC):")
    print(df.groupby("month")["fwd_return_21d_net_tax"].agg(["count", "mean", "median"]))
    print("\nCAVEAT: this is ONE season's data. A month that looks good/bad here might be a one-off "
          "tied to a specific promo calendar event, not a repeatable seasonal rule, until validated "
          "against next year's export.")


if __name__ == "__main__":
    weekly_cycle()
    promo_release_timing()
    month_seasonality()
