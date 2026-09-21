"""
Task C: equity-curve simulation with Kelly-criterion position sizing, Sharpe
ratio, and max drawdown -- on top of the bootstrap Monte Carlo framework this
already had (real historical trade outcomes, not assumed point estimates).

Kelly criterion, in plain terms: given a bet with win probability p, a typical
gain of fraction b if you win, and a typical loss of fraction a if you lose,
there's a mathematically optimal fraction of your bankroll to stake that
maximizes long-run compound growth. Bet less than that and you leave growth on
the table over many bets; bet more and volatility eats you even though each
individual bet still has positive edge. The classic textbook formula (f* = p -
q/b) assumes a TOTAL loss (a=1) on failure -- wrong here, since a losing trade
in this market is usually just the 5% EA sell tax on a flat price, not a
wipeout. Using the correct generalized formula for partial losses instead:

    f* = p/a - q/b     (derived below in estimate_kelly_by_bucket)

Real trading practice uses a FRACTION of full Kelly (this script uses half-
Kelly) because full Kelly is mathematically optimal for growth but extremely
high-variance in practice -- half-Kelly gives up relatively little long-run
growth for a much smoother ride, which is the standard risk-management
tradeoff professional bettors and quant traders actually use.

Per-trade probability and return are bucketed by classifier confidence (not
fed through the regressor directly) since the regressor's raw point estimates
are known to be skewed by rare extreme outliers (see train_model.py's top-
decile check) -- Kelly sizing is dangerously sensitive to overestimated edge,
so bucket-level empirical win rate and MEDIAN (not mean) win/loss magnitude
are used instead, which is far more robust to that skew.

Run: python3 simulate_bankroll.py
"""
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import joblib

from train_model import ALL_FEATURES, HORIZON, MIN_RATING, load_data, make_folds

N_SIMULATIONS = 20000
N_CYCLES = 11  # ~one season / 21-30 day cycles
STARTING_BANKROLLS = [100_000, 500_000, 1_000_000]
MIN_TRADE_PRICE = 5_000  # excludes near-worthless fodder where "10x return" = a few thousand coins
KELLY_FRACTION = 0.5     # half-Kelly -- see module docstring
MAX_POSITION_FRACTION = 0.15  # never stake more than this much of bankroll on one position,
                               # regardless of what Kelly says -- a hard risk cap
CONFIDENCE_BUCKETS = [0.5, 0.6, 0.7, 0.8, 1.01]  # bucket edges on classifier proba
N_POS = 20  # simultaneous diversified positions per cycle


def get_trade_episodes():
    """Real de-duplicated 21-day (proba, actual_return) pairs for confident
    (prob>=0.5) classifier picks, evaluated on the SAME held-out fold the
    deployed model actually reports its numbers against (train_model.py's
    make_folds with fold_edges=[0.55,0.65,0.75,0.85,0.95]) -- not the old
    single 75/25 split, which predates the rolling-window fix and would be
    inconsistent with everything else in this project now."""
    df = load_data()
    target_reg = f"fwd_return_{HORIZON}d_net_tax"
    target_clf = f"fwd_up_{HORIZON}d_net_tax"
    df = df.dropna(subset=[target_reg, target_clf])

    fold_edges = [0.55, 0.65, 0.75, 0.85, 0.95]
    folds = make_folds(df, fold_edges)
    _, test_mask = folds[-1]  # same held-out period train_model.py's deployed model uses

    clf = joblib.load("../data/models/clf_21d.joblib")
    test_df = df.loc[test_mask].copy()
    test_df["proba"] = clf.predict_proba(test_df[ALL_FEATURES])[:, 1]

    test_df = test_df.sort_values(["player_id", "platform", "date"])
    test_df["day_rank"] = test_df.groupby(["player_id", "platform"]).cumcount()
    dedup = test_df[test_df["day_rank"] % HORIZON == 0]

    confident = dedup[dedup["proba"] >= 0.5]
    n_before = len(confident)
    confident = confident[confident["ma_7"] >= MIN_TRADE_PRICE]
    print(f"  (excluded {n_before - len(confident)} sub-{MIN_TRADE_PRICE:,}-coin fodder episodes "
          f"of {n_before} total -- these were driving unrealistic outlier returns)")

    episodes = confident[["proba", target_reg]].rename(columns={target_reg: "actual_return"}).dropna()
    print(f"Steady model: {len(episodes)} real historical episodes, "
          f"mean={episodes['actual_return'].mean():.1%}, median={episodes['actual_return'].median():.1%}, "
          f"win_rate={(episodes['actual_return']>0).mean():.1%}")
    return episodes


def get_promo_dip_returns():
    """Real returns: buy 3 days after a promo/special release, hold 30 days, net of tax."""
    dataset = ds.dataset("../data/prices_features.parquet", format="parquet")
    cols = ["player_id", "platform", "date", "rating", "price_clean"]
    df = dataset.to_table(columns=cols, filter=(ds.field("platform") == "pc") & (ds.field("rating") >= MIN_RATING)).to_pandas()
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


def estimate_kelly_by_bucket(episodes):
    """f* = p/a - q/b for each confidence bucket (see module docstring for the
    derivation -- this is the generalized Kelly formula for a bet that risks
    losing fraction 'a' with probability q and gaining fraction 'b' with
    probability p, NOT the textbook f*=p-q/b version which assumes a full
    wipeout on a loss). p, a, b are all estimated empirically per bucket using
    MEDIANS (robust to the regressor-style outlier skew), never a single
    trade's own predicted return."""
    print("\n=== Kelly fraction by confidence bucket ===")
    kelly_by_bucket = {}
    for lo, hi in zip(CONFIDENCE_BUCKETS[:-1], CONFIDENCE_BUCKETS[1:]):
        bucket = episodes[(episodes["proba"] >= lo) & (episodes["proba"] < hi)]
        if len(bucket) < 30:
            print(f"  [{lo:.1f}, {hi:.1f}): only {len(bucket)} episodes, too few to size -- skipping (f*=0)")
            kelly_by_bucket[(lo, hi)] = 0.0
            continue
        wins = bucket[bucket["actual_return"] > 0]["actual_return"]
        losses = bucket[bucket["actual_return"] <= 0]["actual_return"]
        p = len(wins) / len(bucket)
        q = 1 - p
        b = wins.median() if len(wins) > 0 else 0.0
        a = abs(losses.median()) if len(losses) > 0 else 0.05  # fallback: just the EA tax
        if b <= 0 or a <= 0:
            kelly_by_bucket[(lo, hi)] = 0.0
            continue
        f_star = p / a - q / b
        f_star = max(0.0, min(f_star, 1.0))  # never negative (no edge -> don't bet), never >100%
        f_half = min(f_star * KELLY_FRACTION, MAX_POSITION_FRACTION)
        kelly_by_bucket[(lo, hi)] = f_half
        print(f"  [{lo:.1f}, {hi:.1f}): n={len(bucket)}, p={p:.1%}, typical win=+{b:.1%}, typical loss={-a:.1%}, "
              f"full Kelly f*={f_star:.1%}, half-Kelly (capped)={f_half:.1%}")
    return kelly_by_bucket


def assign_kelly_fraction(episodes, kelly_by_bucket):
    episodes = episodes.copy()
    episodes["kelly_fraction"] = 0.0
    for (lo, hi), f in kelly_by_bucket.items():
        mask = (episodes["proba"] >= lo) & (episodes["proba"] < hi)
        episodes.loc[mask, "kelly_fraction"] = f
    return episodes


def simulate_equity_curve(returns_pool, starting_bankroll, rng, n_positions=N_POS,
                           kelly_fractions_pool=None, fixed_fraction=None):
    """Returns (n_simulations, n_cycles+1) array: bankroll value at the start
    and after each cycle, for every simulated path -- the actual equity curve,
    not just the final number. returns_pool is a plain array of real historical
    returns to bootstrap-sample; kelly_fractions_pool (same length, aligned by
    index) gives each historical episode's own Kelly stake, or pass
    fixed_fraction for uniform sizing instead (used for the promo-dip strategy,
    which has no per-trade confidence score to bucket by)."""
    n = len(returns_pool)
    idx_pool = np.arange(n)
    curve = np.zeros((N_SIMULATIONS, N_CYCLES + 1))
    curve[:, 0] = starting_bankroll
    bankroll = np.full(N_SIMULATIONS, float(starting_bankroll))

    for cycle in range(1, N_CYCLES + 1):
        sampled_idx = rng.choice(idx_pool, size=(N_SIMULATIONS, n_positions), replace=True)
        sampled_returns = returns_pool[sampled_idx]
        if kelly_fractions_pool is not None:
            sampled_fractions = kelly_fractions_pool[sampled_idx]
        else:
            sampled_fractions = np.full_like(sampled_returns, fixed_fraction / n_positions)

        # Each position's OWN Kelly fraction can be up to MAX_POSITION_FRACTION, but
        # nothing stops all n_positions from hitting that cap simultaneously -- with no
        # leverage/borrowing available, total stake across every simultaneous position
        # still can't exceed the bankroll actually in hand. Scale each row down
        # proportionally whenever the fractions would sum past 100%; leave rows alone
        # where the total is already under 100% (no need to force full deployment).
        row_totals = sampled_fractions.sum(axis=1, keepdims=True)
        scale = np.where(row_totals > 1.0, 1.0 / row_totals, 1.0)
        sampled_fractions = sampled_fractions * scale

        # each position stakes its own (now capacity-scaled) kelly_fraction of CURRENT bankroll
        per_position_stake = bankroll[:, None] * sampled_fractions
        pnl = (per_position_stake * sampled_returns).sum(axis=1)
        bankroll = bankroll + pnl
        bankroll = np.maximum(bankroll, 0)  # can't go negative
        curve[:, cycle] = bankroll

    return curve


def estimate_single_kelly(returns):
    """Same generalized f*=p/a-q/b formula as estimate_kelly_by_bucket, but for
    a strategy with no per-trade confidence score (promo-dip) -- one aggregate
    bucket over all its historical episodes instead of confidence-segmented."""
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    p = len(wins) / len(returns)
    q = 1 - p
    b = np.median(wins) if len(wins) > 0 else 0.0
    a = abs(np.median(losses)) if len(losses) > 0 else 0.05
    if b <= 0 or a <= 0:
        return 0.0
    f_star = max(0.0, min(p / a - q / b, 1.0))
    return min(f_star * KELLY_FRACTION, MAX_POSITION_FRACTION)


def compute_risk_metrics(curve):
    """Per-simulated-path Sharpe ratio and max drawdown, summarized across all paths."""
    cycle_returns = curve[:, 1:] / np.maximum(curve[:, :-1], 1e-9) - 1
    mean_r = cycle_returns.mean(axis=1)
    std_r = cycle_returns.std(axis=1)
    sharpe_per_cycle = np.divide(mean_r, std_r, out=np.zeros_like(mean_r), where=std_r > 0)
    # rough annualization assuming ~21-day independent cycles (~17.4 cycles/year) --
    # a simplifying assumption (real cycles aren't independent or identically timed),
    # reported alongside the raw per-cycle number rather than in place of it
    sharpe_annualized = sharpe_per_cycle * np.sqrt(365 / HORIZON)

    running_peak = np.maximum.accumulate(curve, axis=1)
    drawdown = (curve - running_peak) / np.maximum(running_peak, 1e-9)
    max_drawdown = drawdown.min(axis=1)  # most negative value per path

    return sharpe_per_cycle, sharpe_annualized, max_drawdown


def report(label, curve):
    final = curve[:, -1]
    starting = curve[0, 0]
    sharpe, sharpe_ann, max_dd = compute_risk_metrics(curve)
    p10, p50, p90 = np.percentile(final, [10, 50, 90])
    pct_below_start = (final < starting).mean()
    pct_millionaire = (final >= 1_000_000).mean()
    print(f"  {label}: start={starting:,.0f} -> p10={p10:,.0f} median={p50:,.0f} p90={p90:,.0f} "
          f"| below start: {pct_below_start:.1%} | 1M+: {pct_millionaire:.1%}")
    print(f"    Sharpe (per-cycle): median={np.median(sharpe):.2f}  "
          f"| Sharpe (annualized, cycles assumed independent): median={np.median(sharpe_ann):.2f}  "
          f"| Max drawdown: median={np.median(max_dd):.1%}, worst 10%={np.percentile(max_dd, 10):.1%}")


def main():
    rng = np.random.default_rng(42)

    print("=" * 70)
    print("Gathering REAL historical trade outcome distributions...")
    print("=" * 70)
    episodes = get_trade_episodes()
    promo_returns = get_promo_dip_returns()

    kelly_by_bucket = estimate_kelly_by_bucket(episodes)
    episodes = assign_kelly_fraction(episodes, kelly_by_bucket)
    steady_returns = episodes["actual_return"].to_numpy()
    steady_kelly = episodes["kelly_fraction"].to_numpy()

    promo_kelly_f = estimate_single_kelly(promo_returns)
    print(f"\nPromo-dip aggregate half-Kelly fraction: {promo_kelly_f:.1%} "
          f"(no per-trade confidence score to bucket by, so one fraction for all its episodes)")

    print(f"\nSimulating {N_SIMULATIONS} possible {N_CYCLES}-cycle seasons "
          f"(bootstrap resampling real trade outcomes, Kelly-sized per position)...\n")

    print("=" * 70)
    print(f"STEADY MODEL, Kelly-sized ({int(KELLY_FRACTION*100)}%-Kelly, capped at "
          f"{MAX_POSITION_FRACTION:.0%}/position, {N_POS} diversified positions/cycle)")
    print("=" * 70)
    for start in STARTING_BANKROLLS:
        curve = simulate_equity_curve(steady_returns, start, rng, n_positions=N_POS,
                                       kelly_fractions_pool=steady_kelly)
        report(f"start {start:,}", curve)

    print("\n" + "=" * 70)
    print("PROMO DIP, Kelly-sized (single aggregate fraction)")
    print("=" * 70)
    for start in STARTING_BANKROLLS:
        curve = simulate_equity_curve(promo_returns, start, rng, n_positions=N_POS,
                                       fixed_fraction=promo_kelly_f * N_POS)
        report(f"start {start:,}", curve)

    print("\n" + "=" * 70)
    print("COMPARISON: steady model, same trades, FIXED 50% total reinvestment (no Kelly sizing)")
    print("=" * 70)
    for start in STARTING_BANKROLLS:
        curve = simulate_equity_curve(steady_returns, start, rng, n_positions=N_POS, fixed_fraction=0.5)
        report(f"start {start:,}", curve)

    print("\n" + "=" * 70)
    print("COMPARISON: steady model, same trades, FIXED 100% total reinvestment (no Kelly, aggressive)")
    print("=" * 70)
    for start in STARTING_BANKROLLS:
        curve = simulate_equity_curve(steady_returns, start, rng, n_positions=N_POS, fixed_fraction=1.0)
        report(f"start {start:,}", curve)


if __name__ == "__main__":
    main()
