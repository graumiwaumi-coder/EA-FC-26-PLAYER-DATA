"""
Phase 1 part 2: feature engineering on top of the parsed tables.

Reads data/players.parquet, data/prices_long.parquet, data/indices.parquet
Writes data/prices_features.parquet -- one row per (player_id, platform, date) with:
  price, rolling 7/14/30-day mean, volatility (rolling std / mean), pct change over
  7/14/30 days, matching band index value, price-vs-index relative ratio and its
  z-score vs the player's own history.

Run: python3 build_features.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

MIN_TRADEABLE_PRICE = 1  # price of 0 treated as "no market data that day"


def main():
    players = pd.read_parquet(DATA_DIR / "players.parquet")
    prices = pd.read_parquet(DATA_DIR / "prices_long.parquet")
    indices = pd.read_parquet(DATA_DIR / "indices.parquet")

    prices = prices.sort_values(["player_id", "platform", "date"]).reset_index(drop=True)

    # treat 0 (and negative, just in case) as missing/untradeable rather than a real price
    prices["price_clean"] = prices["price"].where(prices["price"] >= MIN_TRADEABLE_PRICE)

    grp = prices.groupby(["player_id", "platform"], sort=False)["price_clean"]

    for window in (7, 14, 30):
        prices[f"ma_{window}"] = grp.transform(
            lambda s: s.rolling(window, min_periods=max(3, window // 3)).mean()
        )
        prices[f"std_{window}"] = grp.transform(
            lambda s: s.rolling(window, min_periods=max(3, window // 3)).std()
        )
        prices[f"pct_change_{window}d"] = grp.transform(
            lambda s: s.pct_change(periods=window)
        )

    prices["volatility_30"] = prices["std_30"] / prices["ma_30"]

    # attach rating band per player, then join matching index value per (band, platform, date)
    prices = prices.merge(players[["id", "rating", "band"]], left_on="player_id", right_on="id", how="left")
    prices = prices.drop(columns=["id"])

    indices_renamed = indices.rename(columns={"band": "band", "index_value": "band_index_value"})
    prices = prices.merge(
        indices_renamed,
        left_on=["band", "platform", "date"],
        right_on=["band", "platform", "date"],
        how="left",
    )

    # price relative to its own rating-band index: player_price / band_index (only meaningful where band is set)
    prices["price_to_index_ratio"] = prices["price_clean"] / prices["band_index_value"]

    # z-score of that ratio against the player's own trailing 60-day history --
    # tells us if the player is currently cheap/expensive relative to ITS OWN normal
    # relationship to the index, not an absolute cross-player comparison.
    def zscore_ratio(s):
        roll_mean = s.rolling(60, min_periods=14).mean()
        roll_std = s.rolling(60, min_periods=14).std()
        return (s - roll_mean) / roll_std

    prices["ratio_zscore_60d"] = prices.groupby(["player_id", "platform"], sort=False)[
        "price_to_index_ratio"
    ].transform(zscore_ratio)

    out_cols = [
        "player_id", "platform", "date", "price", "price_clean",
        "ma_7", "ma_14", "ma_30",
        "pct_change_7d", "pct_change_14d", "pct_change_30d",
        "volatility_30",
        "band", "band_index_value", "price_to_index_ratio", "ratio_zscore_60d",
    ]
    prices[out_cols].to_parquet(DATA_DIR / "prices_features.parquet", index=False)
    print(f"Saved prices_features.parquet: {len(prices)} rows")

    # quick sanity summary
    have_band = prices["band"].notna().sum()
    print(f"Rows with a matching index band: {have_band} ({have_band/len(prices):.1%})")
    print(prices[["pct_change_30d", "volatility_30", "ratio_zscore_60d"]].describe())


if __name__ == "__main__":
    main()
