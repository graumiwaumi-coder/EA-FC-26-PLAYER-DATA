"""
Phase 1 (expanded): full feature set for model training.

Reads data/players_features.parquet, data/prices_long.parquet, data/indices.parquet,
     data/macro_wide.parquet
Writes data/prices_features.parquet -- one row per (player_id, platform, date) with a
large set of technical/momentum features, cross-platform features, macro/index context,
and multi-horizon forward returns (training targets).

Run: python3 build_player_features.py && python3 build_index_features.py && python3 build_features.py
(player + index features must be built first)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

MIN_TRADEABLE_PRICE = 1
SELL_TAX = 0.05
FORWARD_HORIZONS = (7, 14, 21, 30)


def rsi(price, window=14):
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window, min_periods=window // 2).mean()
    avg_loss = loss.rolling(window, min_periods=window // 2).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def build_price_features(prices, players):
    prices = prices.sort_values(["player_id", "platform", "date"]).reset_index(drop=True)
    prices["price_clean"] = prices["price"].where(prices["price"] >= MIN_TRADEABLE_PRICE)

    g = prices.groupby(["player_id", "platform"], sort=False)["price_clean"]

    for w in (3, 7, 14, 30, 60, 90):
        prices[f"ma_{w}"] = g.transform(lambda s, w=w: s.rolling(w, min_periods=max(3, w // 3)).mean())
    for w in (7, 14, 30, 60):
        prices[f"std_{w}"] = g.transform(lambda s, w=w: s.rolling(w, min_periods=max(3, w // 3)).std())
        prices[f"volatility_{w}"] = prices[f"std_{w}"] / prices[f"ma_{w}"]
    for w in (1, 3, 7, 14, 30, 60, 90):
        prices[f"pct_change_{w}d"] = g.transform(lambda s, w=w: s.pct_change(periods=w))

    prices["roll_min_30"] = g.transform(lambda s: s.rolling(30, min_periods=7).min())
    prices["roll_max_30"] = g.transform(lambda s: s.rolling(30, min_periods=7).max())
    prices["roll_min_90"] = g.transform(lambda s: s.rolling(90, min_periods=14).min())
    prices["roll_max_90"] = g.transform(lambda s: s.rolling(90, min_periods=14).max())

    prices["pct_off_90d_high"] = (prices["roll_max_90"] - prices["price_clean"]) / prices["roll_max_90"]
    prices["pct_above_90d_low"] = (prices["price_clean"] - prices["roll_min_90"]) / prices["roll_min_90"]
    denom_30 = (prices["roll_max_30"] - prices["roll_min_30"]).replace(0, np.nan)
    prices["season_position_30"] = (prices["price_clean"] - prices["roll_min_30"]) / denom_30

    prices["rsi_14"] = prices.groupby(["player_id", "platform"], sort=False)["price_clean"].transform(
        lambda s: rsi(s, 14)
    )
    prices["macd_norm"] = (prices["ma_7"] - prices["ma_30"]) / prices["ma_30"]

    prices["skew_30"] = prices.groupby(["player_id", "platform"], sort=False)["pct_change_1d"].transform(
        lambda s: s.rolling(30, min_periods=10).skew()
    )

    prices["tradeable"] = (prices["price"] > 0).astype(float)
    prices["liquidity_14"] = prices.groupby(["player_id", "platform"], sort=False)["tradeable"].transform(
        lambda s: s.rolling(14, min_periods=5).mean()
    )

    prices["bollinger_z_30"] = (prices["price_clean"] - prices["ma_30"]) / prices["std_30"]

    return prices


def add_cross_platform(prices):
    pivot = prices.pivot_table(index=["player_id", "date"], columns="platform", values="price_clean")
    pivot = pivot.rename(columns={"pc": "pc_price_same_day", "console": "console_price_same_day"})
    pivot = pivot.reset_index()

    mom = prices.pivot_table(index=["player_id", "date"], columns="platform", values="pct_change_7d")
    mom = mom.rename(columns={"pc": "pc_pct_change_7d_x", "console": "console_pct_change_7d_x"}).reset_index()

    prices = prices.merge(pivot, on=["player_id", "date"], how="left")
    prices = prices.merge(mom, on=["player_id", "date"], how="left")

    is_pc = prices["platform"] == "pc"
    prices["other_platform_price"] = np.where(is_pc, prices["console_price_same_day"], prices["pc_price_same_day"])
    prices["other_platform_pct_change_7d"] = np.where(
        is_pc, prices["console_pct_change_7d_x"], prices["pc_pct_change_7d_x"]
    )
    prices["own_to_other_platform_ratio"] = prices["price_clean"] / prices["other_platform_price"]

    prices = prices.drop(columns=["pc_price_same_day", "console_price_same_day",
                                   "pc_pct_change_7d_x", "console_pct_change_7d_x"])
    return prices


def _fc26_only_join_keys(prices, base_keys):
    """indices.parquet/indices_features.parquet/macro_wide.parquet predate the FC26/FC27
    distinction and have no FC27 equivalent yet -- band/platform/date alone isn't a safe
    join key once two games' data coexist in the same table (an FC27 row could silently
    borrow an FC26 index value just because the calendar date happens to coincide).
    Tags the index side as game_version="fc26" and, if `prices` has a game_version
    column, requires it in the join key too, so an FC27 row simply gets no match (NaN,
    correct) instead of a wrong cross-game one. If `prices` has no game_version column
    at all (pre-FC27 pipeline runs), joins on the base keys alone -- unambiguous since
    only FC26 data exists in that case."""
    if "game_version" in prices.columns:
        return base_keys + ["game_version"]
    return base_keys


def _tag_fc26(df):
    df = df.copy()
    df["game_version"] = "fc26"
    return df


def add_index_context(prices, players):
    prices = prices.merge(players[["id", "rating", "band"]], left_on="player_id", right_on="id", how="left")
    prices = prices.drop(columns=["id"])

    indices = _tag_fc26(pd.read_parquet(DATA_DIR / "indices_features.parquet"))
    base_keys = ["band", "platform", "date"]
    join_keys = _fc26_only_join_keys(prices, base_keys)
    idx_cols = ["band", "platform", "date", "game_version", "index_value", "idx_pct_change_7d",
                "idx_pct_change_14d", "idx_pct_change_30d", "idx_momentum_pctile_90"]
    prices = prices.merge(
        indices[idx_cols].rename(columns={c: f"band_{c}" for c in idx_cols if c not in join_keys}),
        on=join_keys, how="left",
    )
    prices["price_to_band_index_ratio"] = prices["price_clean"] / prices["band_index_value"]
    prices["band_ratio_zscore_60d"] = prices.groupby(["player_id", "platform"], sort=False)[
        "price_to_band_index_ratio"
    ].transform(lambda s: (s - s.rolling(60, min_periods=14).mean()) / s.rolling(60, min_periods=14).std())

    macro = _tag_fc26(pd.read_parquet(DATA_DIR / "macro_wide.parquet"))
    join_keys_macro = _fc26_only_join_keys(prices, ["platform", "date"])
    prices = prices.merge(macro, on=join_keys_macro, how="left")
    prices["price_to_index100_ratio"] = prices["price_clean"] / prices["index100_index_value"]
    prices["index100_ratio_zscore_60d"] = prices.groupby(["player_id", "platform"], sort=False)[
        "price_to_index100_ratio"
    ].transform(lambda s: (s - s.rolling(60, min_periods=14).mean()) / s.rolling(60, min_periods=14).std())

    return prices


def add_forward_returns(prices):
    prices = prices.sort_values(["player_id", "platform", "date"])
    g = prices.groupby(["player_id", "platform"], sort=False)["price_clean"]
    for h in FORWARD_HORIZONS:
        fwd_price = g.transform(lambda s, h=h: s.shift(-h))
        prices[f"fwd_return_{h}d"] = (fwd_price - prices["price_clean"]) / prices["price_clean"]
        prices[f"fwd_return_{h}d_net_tax"] = (fwd_price * (1 - SELL_TAX) - prices["price_clean"]) / prices["price_clean"]
        prices[f"fwd_up_{h}d_net_tax"] = (prices[f"fwd_return_{h}d_net_tax"] > 0).astype("Int8")
        prices.loc[prices[f"fwd_return_{h}d"].isna(), f"fwd_up_{h}d_net_tax"] = pd.NA
    return prices


def add_calendar(prices):
    prices["day_of_week"] = prices["date"].dt.dayofweek
    prices["is_weekend"] = prices["day_of_week"].isin([5, 6])
    min_date = prices["date"].min()
    prices["days_since_start"] = (prices["date"] - min_date).dt.days
    return prices


def main():
    players = pd.read_parquet(DATA_DIR / "players_features.parquet")
    prices = pd.read_parquet(DATA_DIR / "prices_long.parquet")

    print("Building technical/momentum features...")
    prices = build_price_features(prices, players)

    print("Adding cross-platform features...")
    prices = add_cross_platform(prices)

    print("Joining index/macro context...")
    prices = add_index_context(prices, players)

    print("Computing multi-horizon forward returns (training targets)...")
    prices = add_forward_returns(prices)

    print("Adding calendar features...")
    prices = add_calendar(prices)

    prices.to_parquet(DATA_DIR / "prices_features.parquet", index=False)
    print(f"\nSaved prices_features.parquet: {len(prices)} rows, {len(prices.columns)} columns")
    print("Columns:", list(prices.columns))


if __name__ == "__main__":
    main()
