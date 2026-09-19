"""
Bootstrap Monte Carlo: what does compounding actually look like using the REAL
historical trade outcomes we've already validated, not a hopeful point estimate?

Two strategies, both grounded in real backtested episodes:
  A) "Steady model" -- 21-day HistGradientBoosting classifier, confident (prob>=0.5)
     picks, de-duplicated/independent episodes (from evaluate_robustness.py)
  B) "Promo dip" -- buy special/promo cards 3 days after release, hold 30 days
     (from strategy_analysis.py's grid search)

For each, resample real historical returns (with replacement) to simulate ~11 cycles
(roughly one season's worth of sequential 21-30 day holds), compounding a starting
bankroll, repeated thousands of times to show the actual DISTRIBUTION of outcomes --
including the realistic chance of ending up with less than you started.

Run: python3 simulate_bankroll.py
"""
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import joblib

from train_model import ALL_FEATURES, HORIZON, EMBARGO_DAYS, load_data

N_SIMULATIONS = 20000
N_CYCLES = 11  # ~one season / 21-30 day cycles
STARTING_BANKROLLS = [100_000, 500_000, 1_000_000]
MIN_TRADE_PRICE = 5_000  # excludes near-worthless fodder where "10x return" = a few thousand coins


def get_steady_model_returns():
    """Real de-duplicated 21-day returns for confident (prob>=0.5) classifier picks."""
    df = load_data()
    target_reg = f"fwd_return_{HORIZON}d_net_tax"
    target_clf = f"fwd_up_{HORIZON}d_net_tax"
    df = df.dropna(subset=[target_reg, target_clf])

    dates = df["date"]
    split_date = dates.quantile(0.75)
    test_mask = dates >= (split_date + pd.Timedelta(days=EMBARGO_DAYS))

    clf = joblib.load("../data/models/clf_21d.joblib")
    test_df = df.loc[test_mask].copy()
    test_df["proba"] = clf.predict_proba(test_df[ALL_FEATURES])[:, 1]

    test_df = test_df.sort_values(["player_id", "platform", "date"])
    test_df["day_rank"] = test_df.groupby(["player_id", "platform"]).cumcount()
    dedup = test_df[test_df["day_rank"] % HORIZON == 0]

    confident = dedup[dedup["proba"] >= 0.5]
    n_before = len(confident)
    # exclude near-worthless fodder (ma_7 as a price proxy) -- these produce the
    # "32x return" outliers that are a few thousand coins of real profit, not
    # representative of a trade you'd actually size real money into
    confident = confident[confident["ma_7"] >= MIN_TRADE_PRICE]
    print(f"  (excluded {n_before - len(confident)} sub-{MIN_TRADE_PRICE:,}-coin fodder episodes "
          f"of {n_before} total -- these were driving unrealistic outlier returns)")
    returns = confident[target_reg].dropna().to_numpy()
    print(f"Steady model: {len(returns)} real historical episodes, "
          f"mean={returns.mean():.1%}, median={np.median(returns):.1%}, win_rate={(returns>0).mean():.1%}")
    return returns


def get_promo_dip_returns():
    """Real returns: buy 3 days after a promo/special release, hold 30 days, net of tax."""
    dataset = ds.dataset("../data/prices_features.parquet", format="parquet")
    cols = ["player_id", "platform", "date", "rating", "price_clean"]
    df = dataset.to_table(columns=cols, filter=(ds.field("platform") == "pc") & (ds.field("rating") >= 75)).to_pandas()
    df = df.dropna(subset=["price_clean"]).sort_values(["player_id", "date"])

    first_date_overall = df["date"].min()
    first_seen = df.groupby("player_id")["date"].min().rename("release_date")
    df = df.merge(first_seen, on="player_id")
    cutoff = first_date_overall + pd.Timedelta(days=14)
    releases = df[df["release_date"] > cutoff].copy()
    releases["days_since_release"] = (releases["date"] - releases["release_date"]).dt.days

    pivot = releases.pivot_table(index="player_id", columns="days_since_release", values="price_clean")
    buy_day, hold = 3, 30
    sell_day = buy_day + hold
    if buy_day not in pivot.columns or sell_day not in pivot.columns:
        raise SystemExit("required days not in pivot -- check data range")
    buy_price = pivot[buy_day]
    sell_price = pivot[sell_day]
    valid = buy_price.notna() & sell_price.notna() & (buy_price >= MIN_TRADE_PRICE)
    returns = ((sell_price[valid] * 0.95 - buy_price[valid]) / buy_price[valid]).to_numpy()
    print(f"  (excluded sub-{MIN_TRADE_PRICE:,}-coin fodder releases)")
    print(f"Promo dip: {len(returns)} real historical episodes, "
          f"mean={returns.mean():.1%}, median={np.median(returns):.1%}, win_rate={(returns>0).mean():.1%}")
    return returns


def simulate(returns, starting_bankroll, reinvest_fraction, rng, n_positions=1):
    """n_positions=1 reproduces the unrealistic 'all-in on one card' version.
    n_positions>1 diversifies each cycle's investable bankroll across that many
    simultaneous independent positions (bootstrap-sampled), which is what real
    trading with a meaningful bankroll actually requires -- market depth on any
    single Gold+ card is nowhere near large enough to absorb a multi-million-coin
    single bet without moving the price against you."""
    bankroll = np.full(N_SIMULATIONS, float(starting_bankroll))
    for _ in range(N_CYCLES):
        sampled = rng.choice(returns, size=(N_SIMULATIONS, n_positions), replace=True)
        portfolio_return = sampled.mean(axis=1)
        invested = bankroll * reinvest_fraction
        bankroll = (bankroll - invested) + invested * (1 + portfolio_return)
    return bankroll


def report(label, bankrolls, starting):
    p10, p50, p90 = np.percentile(bankrolls, [10, 50, 90])
    pct_below_start = (bankrolls < starting).mean()
    pct_millionaire = (bankrolls >= 1_000_000).mean()
    print(f"  {label}: start={starting:,.0f}  ->  p10={p10:,.0f}  median={p50:,.0f}  p90={p90:,.0f}  "
          f"| chance of ending BELOW start: {pct_below_start:.1%}  | chance of reaching 1M+: {pct_millionaire:.1%}")


def main():
    rng = np.random.default_rng(42)

    print("=" * 70)
    print("Gathering REAL historical trade outcome distributions...")
    print("=" * 70)
    steady_returns = get_steady_model_returns()
    promo_returns = get_promo_dip_returns()
    blended_returns = np.concatenate([steady_returns, promo_returns])

    print(f"\nSimulating {N_SIMULATIONS} possible {N_CYCLES}-cycle seasons (bootstrap resampling "
          f"real trade outcomes, not assumed averages)...\n")

    print("=" * 70)
    print("WHY CONCENTRATION IS RECKLESS (illustrative only -- do not use this):")
    print("Betting 100% of bankroll on ONE card per cycle, full reinvestment:")
    print("=" * 70)
    b = simulate(steady_returns, 500_000, 1.0, rng, n_positions=1)
    report("steady model, 1 position/cycle (unrealistic)", b, 500_000)
    print("This is dominated by rare huge outlier trades and assumes you can dump an\n"
          "arbitrarily large bankroll into a single ~40k-coin card with no market impact.\n"
          "No real single Gold+ card has that depth. Ignore this number.\n")

    print("=" * 70)
    print("REALISTIC VERSION: bankroll split across 20 simultaneous positions per cycle")
    print("(diversified, the way you'd actually have to trade a meaningful bankroll)")
    print("=" * 70)
    N_POS = 20
    for reinvest_fraction, style in [(1.0, "FULL reinvestment every cycle (aggressive)"),
                                      (0.5, "HALF reinvestment, half banked each cycle (conservative)")]:
        print(f"\n--- Strategy: STEADY MODEL ONLY | {style} ---")
        for start in STARTING_BANKROLLS:
            b = simulate(steady_returns, start, reinvest_fraction, rng, n_positions=N_POS)
            report(f"start {start:,}", b, start)

        print(f"\n--- Strategy: PROMO DIP ONLY | {style} ---")
        for start in STARTING_BANKROLLS:
            b = simulate(promo_returns, start, reinvest_fraction, rng, n_positions=N_POS)
            report(f"start {start:,}", b, start)

        print(f"\n--- Strategy: BLENDED (both pooled) | {style} ---")
        for start in STARTING_BANKROLLS:
            b = simulate(blended_returns, start, reinvest_fraction, rng, n_positions=N_POS)
            report(f"start {start:,}", b, start)


if __name__ == "__main__":
    main()
