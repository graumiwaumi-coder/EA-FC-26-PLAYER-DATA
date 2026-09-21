#!/usr/bin/env python3
"""
Writes a compact, human-readable data-quality report after every full
pipeline run -- row counts, date ranges, null rates, and specific checks
for the failure modes we've actually hit building this (the FC26/FC27 ID
collision, duplicate rows, the index-player rating/price parsing issues).

Written to reports/latest_quality_report.txt (NOT under data/, which is
gitignored) so it can be pushed and reviewed directly instead of pasting
large amounts of terminal output back and forth.

Run: python3 data_quality_report.py
"""
import glob
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SCRAPES_DIR = ROOT / "scraper" / "scrapes"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
REPORT_PATH = REPORTS_DIR / "latest_quality_report.txt"

lines = []


def log(s=""):
    lines.append(str(s))


def section(title):
    log()
    log(f"=== {title} ===")


def load(name):
    path = DATA_DIR / name
    if not path.exists():
        log(f"  NOT FOUND: {path}")
        return None
    return pd.read_parquet(path)


def check_players():
    section("players.parquet")
    df = load("players.parquet")
    if df is None:
        return
    log(f"  total rows: {len(df)}")
    if "game_version" in df.columns:
        log(f"  by game_version: {df['game_version'].value_counts().to_dict()}")
        n_missing_gv = df["game_version"].isna().sum()
        log(f"  rows with MISSING game_version (should be 0 -- risk of FC26/FC27 id collision): {n_missing_gv}")
    else:
        log("  WARNING: no game_version column at all")
    if "rating" in df.columns:
        log(f"  rating null rate: {df['rating'].isna().mean():.1%}")
        below75 = (df["rating"] < 75).sum()
        log(f"  rows with rating < 75 (out of Gold+ scope, should be rare/0): {below75}")
    fc27 = df[df.get("game_version") == "fc27"] if "game_version" in df.columns else df
    if "current_price" in fc27.columns and len(fc27):
        log(f"  FC27 rows with current_price populated: {fc27['current_price'].notna().sum()}/{len(fc27)}")


def check_prices_long():
    section("prices_long.parquet")
    df = load("prices_long.parquet")
    if df is None:
        return
    log(f"  total rows: {len(df)}")
    if "game_version" in df.columns:
        log(f"  by game_version: {df['game_version'].value_counts().to_dict()}")
    dup_cols = [c for c in ("player_id", "platform", "date", "game_version") if c in df.columns]
    if dup_cols:
        n_dupes = df.duplicated(subset=dup_cols).sum()
        log(f"  duplicate rows on {dup_cols} (should be 0 -- dedup key): {n_dupes}")
    if "date" in df.columns and len(df):
        log(f"  date range: {df['date'].min()} to {df['date'].max()}")
    if "price" in df.columns and len(df):
        bad_price = ((df["price"] <= 0) | (df["price"] > 20_000_000)).sum()
        log(f"  rows with implausible price (<=0 or >20M, should be rare/0): {bad_price}")


def check_live_snapshots():
    section("live_snapshots.parquet")
    df = load("live_snapshots.parquet")
    if df is None:
        return
    log(f"  total rows: {len(df)}")
    if "source" in df.columns:
        log(f"  by source: {df['source'].value_counts().to_dict()}")
    if "date" in df.columns and len(df):
        log(f"  date range: {df['date'].min()} to {df['date'].max()}")


def check_sales_history():
    section("sales_history.parquet")
    df = load("sales_history.parquet")
    if df is None:
        return
    log(f"  total rows: {len(df)}")
    if "sold_for" in df.columns:
        log(f"  rows with missing sold_for: {df['sold_for'].isna().sum()}")
    content_cols = [c for c in df.columns if c != "_first_seen_scraped_at"]
    n_dupes = df.duplicated(subset=content_cols).sum()
    log(f"  duplicate rows on sale content (should be 0 -- dedup key): {n_dupes}")


def check_market_indices():
    section("market_indices.parquet")
    df = load("market_indices.parquet")
    if df is None:
        return
    log(f"  total rows: {len(df)}")
    if "index_tier" in df.columns:
        log(f"  by index_tier: {df['index_tier'].value_counts().to_dict()}")
    if "platform" in df.columns:
        log(f"  by platform: {df['platform'].value_counts().to_dict()}")
    dup_cols = [c for c in ("index_tier", "platform", "date") if c in df.columns]
    if dup_cols:
        n_dupes = df.duplicated(subset=dup_cols).sum()
        log(f"  duplicate rows on {dup_cols} (should be 0 -- dedup key): {n_dupes}")
    if "date" in df.columns and len(df):
        log(f"  date range: {df['date'].min()} to {df['date'].max()}")
    if "price" in df.columns and len(df):
        bad = ((df["price"] <= 0) | (df["price"] > 100_000)).sum()
        log(f"  rows with implausible index value (<=0 or >100k, should be rare/0): {bad}")


def check_index_players():
    section("index_players_*.jsonl (latest raw file -- player-discovery stream)")
    files = sorted(glob.glob(str(SCRAPES_DIR / "index_players_*.jsonl"))) + \
        sorted(glob.glob(str(SCRAPES_DIR / "processed" / "index_players_*.jsonl")))
    if not files:
        log("  NOT FOUND")
        return
    path = sorted(files)[-1]
    df = pd.read_json(path, lines=True)
    log(f"  file: {path}")
    log(f"  total rows: {len(df)}")
    log(f"  unique player_id: {df['player_id'].nunique()}")
    if "section" in df.columns:
        log(f"  by section: {df['section'].value_counts().to_dict()}")
    if "rating" in df.columns:
        n_null = df["rating"].isna().sum()
        out_of_range = df["rating"].dropna().apply(lambda r: not (75 <= r <= 99)).sum()
        log(f"  rating: {n_null} null, {out_of_range} outside 75-99 (should be 0 after the plausibility filter)")
    if "price" in df.columns:
        n_nonnull_price = df["price"].notna().sum()
        log(f"  price non-null (should be 0 -- intentionally dropped as unreliable): {n_nonnull_price}")
    if "pct_change" in df.columns:
        log(f"  pct_change null rate: {df['pct_change'].isna().mean():.1%}")


def main():
    log(f"Data quality report -- generated {datetime.now(timezone.utc).isoformat()}")
    check_players()
    check_prices_long()
    check_live_snapshots()
    check_sales_history()
    check_market_indices()
    check_index_players()
    log()

    text = "\n".join(lines)
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
