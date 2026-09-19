#!/usr/bin/env python3
"""One-off sanity check on the merged live dataset. Run: python3 check_live_dataset.py"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

players = pd.read_parquet(DATA_DIR / "players.parquet")
prices = pd.read_parquet(DATA_DIR / "prices_long.parquet")
live = pd.read_parquet(DATA_DIR / "live_snapshots.parquet")
sales = pd.read_parquet(DATA_DIR / "sales_history.parquet")

print("=== players.parquet ===")
print(players["game_version"].value_counts())
fc27 = players[players["game_version"] == "fc27"]
print(f"fc27 rows: {len(fc27)}")
print(f"fc27 rows with null rating: {fc27['rating'].isna().sum()} / {len(fc27)}")
print("fc27 rating distribution:\n", fc27["rating"].describe())
print("\nsample fc27 rows:")
print(fc27[["id", "rating", "position", "league", "current_price"]].sample(min(5, len(fc27)), random_state=1))

print("\n=== prices_long.parquet ===")
print(prices["game_version"].value_counts())
fc27p = prices[prices["game_version"] == "fc27"]
print(f"fc27 date range: {fc27p['date'].min()} -> {fc27p['date'].max()}")
print(f"fc27 price range: {fc27p['price'].min()} -> {fc27p['price'].max()}")
print(fc27p["platform"].value_counts())

print("\n=== live_snapshots.parquet ===")
print(f"rows: {len(live)}, date range: {live['date'].min()} -> {live['date'].max()}")
print(f"price nulls (expected for chart-only rows): {live['price'].isna().sum()} / {len(live)}")
print(live["source"].value_counts())

print("\n=== sales_history.parquet ===")
print(f"rows: {len(sales)}, unique players: {sales['player_id'].nunique()}")
print("columns:", list(sales.columns))
print(sales.sample(min(5, len(sales)), random_state=1).to_string())
