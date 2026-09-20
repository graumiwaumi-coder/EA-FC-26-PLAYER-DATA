"""
Task 1: foundational feature engineering pass 2 -- finance-inspired features plus
previously planned additions. Builds on top of the existing prices_features.parquet
(Gold+ rated players only, matching the Colab notebook's memory-safe scoping) rather
than regenerating the whole base pipeline.

New features:
  - days_since_release: proxy for "how long has this specific card existed" (capped)
  - trend_slope_14d / trend_slope_30d: proper OLS price trend (not just moving averages)
  - ma_crossover / days_since_crossover: golden/death cross signal
  - cross_sectional_rank: where this player's price sits vs SAME-RATING peers today
    (a genuinely different comparison than the futbin-index-based ones we already have)
  - beta_60d: market-beta (finance concept) -- price sensitivity to Index100 daily moves,
    proper rolling regression, not just a ratio
  - autocorr_30d: lag-1 autocorrelation of daily returns -- positive = momentum/trending,
    negative = mean-reverting (a real quant-finance regime indicator)
  - relative_strength: band index momentum minus Index100 momentum (is this rating tier
    outperforming or lagging the broad market)

Run: python3 build_features_v2.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from build_features import _fc26_only_join_keys, _tag_fc26

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MIN_RATING = 75
RELEASE_CAP_DAYS = 60


def load_gold_plus():
    players = pd.read_parquet(DATA_DIR / "players_features.parquet")
    players = players[players["rating"] >= MIN_RATING]

    cols = ["player_id", "platform", "date", "rating", "price_clean", "ma_7", "ma_30",
            "pct_change_1d", "band", "band_index_value", "band_idx_pct_change_7d",
            "index100_index_value", "index100_idx_pct_change_7d",
            "fwd_return_21d_net_tax", "fwd_up_21d_net_tax"]
    dataset = ds.dataset(DATA_DIR / "prices_features.parquet", format="parquet")
    # game_version has to survive this column selection -- it's not just metadata, it's
    # needed below (cross_sectional_rank, market beta, relative_strength) to stop an FC27
    # row from being grouped/joined against FC26 rows that happen to share a date.
    if "game_version" in dataset.schema.names:
        cols = cols + ["game_version"]
    table = dataset.to_table(columns=cols, filter=ds.field("rating") >= MIN_RATING)
    df = table.to_pandas(split_blocks=True, self_destruct=True)
    del table
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype("float32")
    df = df.sort_values(["player_id", "platform", "date"]).reset_index(drop=True)
    print(f"Loaded {len(df)} rows, {df['player_id'].nunique()} players (rating>={MIN_RATING})")
    return df, players


def add_release_timing(df):
    first_date_overall = df["date"].min()
    first_seen = df.groupby("player_id")["date"].min().rename("release_date")
    df = df.merge(first_seen, on="player_id", how="left")
    cutoff = first_date_overall + pd.Timedelta(days=14)
    is_release_cohort = df["release_date"] > cutoff
    days_since = (df["date"] - df["release_date"]).dt.days.clip(upper=RELEASE_CAP_DAYS)
    df["days_since_release"] = np.where(is_release_cohort, days_since, RELEASE_CAP_DAYS)
    df = df.drop(columns=["release_date"])
    return df


def add_trend_slope(df, window, col="price_clean"):
    def group_slope(s):
        t = pd.Series(np.arange(len(s)), index=s.index, dtype="float32")
        cov = s.rolling(window, min_periods=max(5, window // 2)).cov(t)
        var = t.rolling(window, min_periods=max(5, window // 2)).var()
        slope = cov / var
        return slope / s.rolling(window, min_periods=max(5, window // 2)).mean()  # normalize by price level

    return df.groupby(["player_id", "platform"], sort=False)[col].apply(group_slope).reset_index(level=[0, 1], drop=True)


def add_ma_crossover(df):
    cross = df.groupby(["player_id", "platform"], sort=False).apply(
        lambda g: (g["ma_7"] > g["ma_30"]).ne((g["ma_7"] > g["ma_30"]).shift(1))
    ).reset_index(level=[0, 1], drop=True)
    df["ma_crossover"] = cross.fillna(False)

    df["_pos"] = df.groupby(["player_id", "platform"], sort=False).cumcount()
    df["_cross_pos"] = df["_pos"].where(df["ma_crossover"])
    df["_last_cross_pos"] = df.groupby(["player_id", "platform"], sort=False)["_cross_pos"].ffill()
    df["days_since_crossover"] = (df["_pos"] - df["_last_cross_pos"]).fillna(999).clip(upper=999)
    df = df.drop(columns=["_pos", "_cross_pos", "_last_cross_pos"])
    return df


def add_cross_sectional_rank(df):
    # rank against peers on the same date/platform/rating -- without game_version an FC27
    # card could get ranked against FC26 cards that happen to share a calendar date, which
    # is a different economy entirely.
    group_cols = ["date", "platform", "rating"]
    if "game_version" in df.columns:
        group_cols = group_cols + ["game_version"]
    df["cross_sectional_rank"] = df.groupby(group_cols, sort=False)["price_clean"].rank(pct=True)
    return df


def add_market_beta(df, window=60):
    # index100_index_value is duplicated across every player row on a given (platform,
    # date[, game_version]); dedup back down to one market-return series per group before
    # computing returns, keeping FC26 and FC27's market series separate if both are present.
    key_cols = ["platform", "game_version"] if "game_version" in df.columns else ["platform"]
    dedup_cols = key_cols + ["date"]
    mkt = df[dedup_cols + ["index100_index_value"]].drop_duplicates(subset=dedup_cols).sort_values(dedup_cols)
    mkt["mkt_ret_1d"] = mkt.groupby(key_cols, sort=False)["index100_index_value"].pct_change(1)
    mkt["mkt_var_w"] = mkt.groupby(key_cols, sort=False)["mkt_ret_1d"].transform(lambda s: s.rolling(window, min_periods=window // 3).var())
    df = df.merge(mkt[dedup_cols + ["mkt_ret_1d", "mkt_var_w"]], on=dedup_cols, how="left")

    def group_cov(g):
        return g["pct_change_1d"].rolling(window, min_periods=window // 3).cov(g["mkt_ret_1d"])

    cov = df.groupby(["player_id", "platform"], sort=False).apply(group_cov).reset_index(level=[0, 1], drop=True)
    # same near-zero-variance instability as autocorrelation -- clip defensively.
    # real market betas are rarely beyond a few, so this only catches numerical
    # artifacts, not legitimate signal
    df["beta_60d"] = (cov / df["mkt_var_w"]).clip(-10.0, 10.0)
    df = df.drop(columns=["mkt_ret_1d", "mkt_var_w"])
    return df


def add_autocorr(df, window=30):
    def group_autocorr(g):
        s = g["pct_change_1d"]
        s_lag = s.shift(1)
        cov = s.rolling(window, min_periods=window // 2).cov(s_lag)
        var = s.rolling(window, min_periods=window // 2).var()
        return cov / var

    ac = df.groupby(["player_id", "platform"], sort=False).apply(group_autocorr).reset_index(level=[0, 1], drop=True)
    # true autocorrelation is mathematically bounded in [-1, 1] -- values outside that
    # range are a numerical artifact of dividing by a rolling variance that's too close
    # to zero (a player whose price barely moved in the window), not a real signal
    df["autocorr_30d"] = ac.clip(-1.0, 1.0)
    return df


def add_relative_strength(df):
    # indices_features.parquet is fc26-only and predates the FC26/FC27 distinction (same
    # issue as build_features.py's add_index_context) -- tag it and require game_version
    # in the join key so an FC27 row gets NaN instead of a wrong FC26 match.
    idx = _tag_fc26(pd.read_parquet(DATA_DIR / "indices_features.parquet",
                                     columns=["band", "platform", "date", "idx_pct_change_7d"]))
    idx = idx.rename(columns={"idx_pct_change_7d": "band_momentum_check"})
    join_keys = _fc26_only_join_keys(df, ["band", "platform", "date"])
    df = df.merge(idx[join_keys + ["band_momentum_check"]], on=join_keys, how="left")
    df["relative_strength"] = df["band_idx_pct_change_7d"] - df["index100_idx_pct_change_7d"]
    df = df.drop(columns=["band_momentum_check"])
    return df


def main():
    df, players = load_gold_plus()

    print("Adding release timing...")
    df = add_release_timing(df)

    print("Adding trend slope (14d, 30d)...")
    df["trend_slope_14d"] = add_trend_slope(df, 14)
    df["trend_slope_30d"] = add_trend_slope(df, 30)

    print("Adding MA crossover signal...")
    df = add_ma_crossover(df)

    print("Adding cross-sectional price rank...")
    df = add_cross_sectional_rank(df)

    print("Adding market beta (60d rolling)...")
    df = add_market_beta(df)

    print("Adding autocorrelation / mean-reversion-vs-momentum regime...")
    df = add_autocorr(df)

    print("Adding relative strength (band vs broad market momentum)...")
    df = add_relative_strength(df)

    out_path = DATA_DIR / "prices_features_v2.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved {out_path}: {len(df)} rows, {len(df.columns)} columns")
    print("New columns:", [c for c in df.columns if c not in
          ["player_id", "platform", "date", "rating", "price_clean", "ma_7", "ma_30",
           "pct_change_1d", "band", "band_index_value", "band_idx_pct_change_7d",
           "index100_index_value", "index100_idx_pct_change_7d",
           "fwd_return_21d_net_tax", "fwd_up_21d_net_tax"]])

    # sanity checks
    print("\n=== Sanity check: new feature summary stats ===")
    new_cols = ["days_since_release", "trend_slope_14d", "trend_slope_30d",
                "days_since_crossover", "cross_sectional_rank", "beta_60d", "autocorr_30d",
                "relative_strength"]
    print(df[new_cols].describe())
    for c in new_cols:
        n_inf = np.isinf(df[c]).sum()
        if n_inf > 0:
            print(f"WARNING: {c} has {n_inf} infinite values")


if __name__ == "__main__":
    main()
