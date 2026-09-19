"""
Phase 1 (expanded): static / slow-moving player-level features.

Reads data/players.parquet, data/prices_long.parquet
Writes data/players_features.parquet

Adds: position_group, is_icon, and demand-proxy scores per league/club/nation
(derived from the data itself -- median non-zero PC price and liquidity, i.e.
fraction of days actually tradeable -- as a stand-in for "how popular/demanded
is this league/club/nation"). These are computed from the FULL history, so
they're a simplification (see caveat printed at the end) that Phase 2's
model training should redo fold-safe to avoid leakage.

Run: python3 build_player_features.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

POSITION_GROUP = {
    "GK": "GK",
    "CB": "DEF", "LB": "DEF", "RB": "DEF", "LWB": "DEF", "RWB": "DEF",
    "CDM": "MID", "CM": "MID", "CAM": "MID", "LM": "MID", "RM": "MID",
    "LW": "FWD", "RW": "FWD", "ST": "FWD", "CF": "FWD",
}


def main():
    players = pd.read_parquet(DATA_DIR / "players.parquet")
    prices = pd.read_parquet(DATA_DIR / "prices_long.parquet")

    pc = prices[prices["platform"] == "pc"].copy()
    pc["tradeable"] = pc["price"] > 0
    pc["price_nonzero"] = pc["price"].where(pc["price"] > 0)

    per_player = pc.groupby("player_id").agg(
        median_price_pc=("price_nonzero", "median"),
        liquidity_pc=("tradeable", "mean"),
    ).reset_index()

    players = players.merge(per_player, left_on="id", right_on="player_id", how="left")
    players = players.drop(columns=["player_id"])

    players["position_group"] = players["position"].map(POSITION_GROUP)
    players["is_icon"] = players["league"] == "Icons"

    for col in ("league", "club", "nation"):
        agg = players.groupby(col).agg(
            **{
                f"{col}_price_score": ("median_price_pc", "median"),
                f"{col}_liquidity_score": ("liquidity_pc", "mean"),
                f"{col}_n_players": ("id", "count"),
            }
        ).reset_index()
        players = players.merge(agg, on=col, how="left")

    players.to_parquet(DATA_DIR / "players_features.parquet", index=False)
    print(f"Saved players_features.parquet: {len(players)} rows, {len(players.columns)} columns")
    print("Columns:", list(players.columns))
    print("\nCAVEAT: league/club/nation popularity scores use the full season's history "
          "(not just data available up to each point in time). Fine for Phase 1 exploration; "
          "Phase 2 training will need to recompute these using only trailing/historical data "
          "at each training example to avoid look-ahead leakage.")


if __name__ == "__main__":
    main()
