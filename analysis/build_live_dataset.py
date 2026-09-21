"""
Merge the FC27 live scrapes (scraper/scrapes/*.jsonl) into the training
data, incrementally and without duplicating anything on repeat runs.

Three separate input streams, three separate merge strategies, because they
behave differently:

  1. player_history_*.jsonl (scrape_player_history.py -- meta + whatever
     price-history array the player page exposes, one snapshot per player
     per run)
       -> upserted into data/players.parquet (latest scraped_at wins per
          player id) and anti-joined into data/prices_long.parquet (a
          price point for a (player_id, platform, date) we've already seen
          is skipped, only genuinely new dates get appended).
       Tagged with game_version="fc27" so ids can never collide with the
       existing verified FC26 rebuild data (game_version="fc26", backfilled
       onto the existing rows the first time this runs) -- FC26 and FC27
       are different item pools, so the same numeric id in both means
       nothing and must not be merged as if it were one player.

  2. market_list_*.jsonl (scrape_market_list.py -- squad + price-bucket
     sweeps, wide row per player with both ps and pc columns at once)
       -> reshaped to long format (one row per player x platform x
          scraped_at) and appended to data/live_snapshots.parquet. Each
          run's scraped_at is a genuinely new point in time, so this is
          append-only, EXCEPT a single run's squad pass and price-bucket
          pass can both re-return the same player -- those in-run repeats
          are dropped (same player+platform+scraped_at+price = redundant).

  3. player_sales_*.jsonl (scrape_player_details.py -- daily/live chart +
     up to ~1000 recent sales-history rows per player x platform)
       -> daily/live chart snapshots go into data/live_snapshots.parquet
          the same way as #2 (append-only, one point in time per run).
          The sales_history rows are different: the SAME real-world sale
          reappears in every run's "most recent sales" window until it
          ages out, so those are merged into data/sales_history.parquet
          keyed on their own content (player_id, platform, date, listed
          for, sold for, ea tax, net price, type) -- re-scraping the same
          sale a second time is a no-op, only sales we haven't seen before
          get appended.

Safe to re-run any time: each step only reads scrape files and only adds
rows that don't already exist by that stream's dedup key. Run this after
any scraper run (or all of them).

Run: python3 build_live_dataset.py
"""
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from build_dataset import band_for

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SCRAPES_DIR = ROOT / "scraper" / "scrapes"
PROCESSED_DIR = SCRAPES_DIR / "processed"

PLAYERS_PATH = DATA_DIR / "players.parquet"
PRICES_LONG_PATH = DATA_DIR / "prices_long.parquet"
LIVE_SNAPSHOTS_PATH = DATA_DIR / "live_snapshots.parquet"
SALES_HISTORY_PATH = DATA_DIR / "sales_history.parquet"
MARKET_INDICES_PATH = DATA_DIR / "market_indices.parquet"


def parse_money(s):
    if s is None:
        return None
    s = str(s).strip().upper().replace(",", "")
    if s in ("", "-", "N/A", "NIL", "NONE"):
        return None
    mult = 1
    if s.endswith("K"):
        mult, s = 1_000, s[:-1]
    elif s.endswith("M"):
        mult, s = 1_000_000, s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return None


def parse_pct(s):
    if s is None:
        return None
    m = re.search(r"-?[\d.]+", str(s))
    return float(m.group(0)) if m else None


def read_jsonl_files(pattern):
    for path in sorted(glob.glob(str(SCRAPES_DIR / pattern))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                rec["_source_file"] = Path(path).name
                yield rec


def parse_height_cm(s):
    if not s:
        return None
    m = re.search(r"(\d+)\s*cm", s)
    return int(m.group(1)) if m else None


def parse_age(s):
    if not s:
        return None
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else None


def ensure_game_version_column(df, existing_default):
    if "game_version" not in df.columns:
        df["game_version"] = existing_default
    return df


# ---------------------------------------------------------------- stream 1
def merge_player_history():
    records = list(read_jsonl_files("player_history_*.jsonl"))
    if not records:
        print("[player_history] no player_history_*.jsonl files found, skipping")
        return

    print(f"[player_history] {len(records)} snapshot rows across all runs")

    player_rows, price_rows = [], []
    for rec in records:
        # rating is usually just "86", but an over-broad DOM selector can grab
        # a whole card's text ("86\n  CM\n  ++") -- take the leading digits
        # either way rather than requiring an exact int-parseable string.
        rating = None
        m = re.match(r"\s*(\d+)", str(rec.get("rating") or ""))
        if m:
            rating = int(m.group(1))
        player_rows.append({
            "id": rec["id"],
            "url": rec.get("url"),
            "scraped_at": rec.get("scraped_at"),
            "rating": rating,
            "position": rec.get("position"),
            "nation": rec.get("nation"),
            "league": rec.get("league"),
            "club": rec.get("club"),
            "squad": rec.get("squad"),
            "skills": int(rec["skills"]) if str(rec.get("skills") or "").strip().isdigit() else None,
            "weak_foot": int(rec["weak_foot"]) if str(rec.get("weak_foot") or "").strip().isdigit() else None,
            "height_cm": parse_height_cm(rec.get("height")),
            "foot": rec.get("foot"),
            "body_type": rec.get("btype"),
            "age": parse_age(rec.get("age")),
            "playstyles": "|".join(rec.get("playstyles") or []),
            "n_playstyles": len(rec.get("playstyles") or []),
            "roles_text": rec.get("roles_text"),
            "band": band_for(rating, rec.get("league")),
            "current_price": parse_money(rec.get("current_price")),
            "current_price_platform": rec.get("current_price_platform"),
            "game_version": "fc27",
        })

        for platform_label, key in (("pc", "pc"), ("console", "ps")):
            raw = rec.get(key)
            if not raw:
                continue
            try:
                series = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            for point in series:
                if len(point) != 2:
                    continue
                ts, price = point
                price_rows.append((rec["id"], platform_label, ts, price, "fc27"))

    new_players = pd.DataFrame(player_rows)
    if not new_players.empty:
        new_players["scraped_at"] = pd.to_datetime(new_players["scraped_at"])
        new_players = new_players.sort_values("scraped_at").drop_duplicates(
            subset=["id", "game_version"], keep="last"
        )

    if PLAYERS_PATH.exists():
        players_df = pd.read_parquet(PLAYERS_PATH)
        players_df = ensure_game_version_column(players_df, "fc26")
    else:
        players_df = pd.DataFrame(columns=list(new_players.columns))

    if not new_players.empty:
        combined = pd.concat([players_df, new_players], ignore_index=True)
        if "scraped_at" in combined.columns:
            combined["scraped_at"] = pd.to_datetime(combined["scraped_at"])
            combined = combined.sort_values("scraped_at", na_position="first")
        combined = combined.drop_duplicates(subset=["id", "game_version"], keep="last")
        combined.to_parquet(PLAYERS_PATH, index=False)
        print(f"[player_history] players.parquet: {len(players_df)} -> {len(combined)} rows "
              f"({len(combined) - len(players_df)} net new/updated)")

    if PRICES_LONG_PATH.exists():
        prices_df = pd.read_parquet(PRICES_LONG_PATH)
        prices_df = ensure_game_version_column(prices_df, "fc26")
    else:
        prices_df = pd.DataFrame(columns=["player_id", "platform", "date", "price", "game_version"])

    new_prices = pd.DataFrame(price_rows, columns=["player_id", "platform", "ts_ms", "price", "game_version"])
    if new_prices.empty:
        print("[player_history] no price-history points found in any player_history file "
              "(FC27 pages may not expose a price-history array -- see scrape_player_history.py notes)")
        return

    new_prices["date"] = pd.to_datetime(new_prices["ts_ms"], unit="ms")
    new_prices = new_prices.drop(columns=["ts_ms"]).drop_duplicates(
        subset=["player_id", "platform", "date", "game_version"]
    )

    existing_keys = prices_df[["player_id", "platform", "date", "game_version"]].drop_duplicates()
    merged_keys = new_prices.merge(
        existing_keys, on=["player_id", "platform", "date", "game_version"],
        how="left", indicator=True,
    )
    truly_new = new_prices[merged_keys["_merge"].values == "left_only"]

    if truly_new.empty:
        print(f"[player_history] {len(new_prices)} price points scraped, all already present -- 0 new")
    else:
        combined_prices = pd.concat([prices_df, truly_new], ignore_index=True)
        combined_prices.to_parquet(PRICES_LONG_PATH, index=False)
        print(f"[player_history] prices_long.parquet: +{len(truly_new)} new price points "
              f"({len(new_prices) - len(truly_new)} were already present)")


# ---------------------------------------------------------------- stream 2
def _melt_platform_row(rec, platform_label, suffix):
    price = parse_money(rec.get(f"price_{suffix}"))
    if price is None:
        return None
    return {
        "player_id": rec.get("player_id"),
        "platform": platform_label,
        "date": rec.get("scraped_at"),
        "price": price,
        "trend_pct": parse_pct(rec.get(f"trend_{suffix}")),
        "ea_average": parse_money(rec.get(f"ea_average_{suffix}")),
        "difference": parse_money(rec.get(f"difference_{suffix}")),
        "ea_tax": parse_money(rec.get(f"ea_tax_{suffix}")),
        # "source" was added to the scraper after some early runs -- infer
        # it from "squad" for files that predate that field instead of
        # leaving it null.
        "source": rec.get("source") or ("squad" if rec.get("squad") else "price_range"),
        "squad": rec.get("squad"),
    }


def merge_live_snapshots():
    market_records = list(read_jsonl_files("market_list_*.jsonl"))
    sales_records = list(read_jsonl_files("player_sales_*.jsonl"))
    if not market_records and not sales_records:
        print("[live_snapshots] no market_list_*.jsonl or player_sales_*.jsonl files found, skipping")
        return

    rows = []
    for rec in market_records:
        for platform_label, suffix in (("console", "ps"), ("pc", "pc")):
            row = _melt_platform_row(rec, platform_label, suffix)
            if row:
                rows.append(row)

    for rec in sales_records:
        platform_label = "console" if rec.get("platform") == "ps" else "pc"
        rows.append({
            "player_id": rec.get("player_id"),
            "platform": platform_label,
            "date": rec.get("scraped_at"),
            "price": None,
            "trend_pct": None,
            "ea_average": None,
            "difference": None,
            "ea_tax": None,
            "source": "sales_page_chart",
            "squad": None,
            "daily_high": parse_money(rec.get("daily_high")),
            "daily_low": parse_money(rec.get("daily_low")),
            "live_high": parse_money(rec.get("live_high")),
            "live_low": parse_money(rec.get("live_low")),
            "live_avg": parse_money(rec.get("live_avg")),
        })

    new_df = pd.DataFrame(rows)
    if new_df.empty:
        print("[live_snapshots] nothing to merge")
        return
    new_df["date"] = pd.to_datetime(new_df["date"])
    # dedupe on the actual data, not the "source"/"squad" bookkeeping tags --
    # the same player caught by both the squad pass and a price-bucket pass
    # in one run has identical price/trend/etc but different source/squad
    # labels, and is still just one real snapshot, not two.
    dedup_cols = [c for c in new_df.columns if c not in ("source", "squad")]
    new_df = new_df.sort_values("source").drop_duplicates(subset=dedup_cols, keep="first")

    if LIVE_SNAPSHOTS_PATH.exists():
        existing = pd.read_parquet(LIVE_SNAPSHOTS_PATH)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.sort_values("source").drop_duplicates(subset=dedup_cols, keep="first")
    else:
        existing = pd.DataFrame()
        combined = new_df

    combined.to_parquet(LIVE_SNAPSHOTS_PATH, index=False)
    print(f"[live_snapshots] live_snapshots.parquet: {len(existing)} -> {len(combined)} rows "
          f"(+{len(combined) - len(existing)} net new)")


# ---------------------------------------------------------------- stream 3
def merge_sales_history():
    sales_records = list(read_jsonl_files("player_sales_*.jsonl"))
    if not sales_records:
        print("[sales_history] no player_sales_*.jsonl files found, skipping")
        return

    rows = []
    for rec in sales_records:
        platform_label = "console" if rec.get("platform") == "ps" else "pc"
        for sale in rec.get("sales_history") or []:
            row = dict(sale)  # dynamic columns straight from the table header
            row["player_id"] = rec.get("player_id")
            row["platform"] = platform_label
            row["_first_seen_scraped_at"] = rec.get("scraped_at")
            rows.append(row)

    if not rows:
        print("[sales_history] no sales_history rows found in any run")
        return

    new_df = pd.DataFrame(rows)
    # dedupe on every column that describes the sale itself, i.e. everything
    # except our own bookkeeping (_first_seen_scraped_at) -- the same
    # real-world sale reappearing in a later scrape must not duplicate.
    content_cols = [c for c in new_df.columns if c != "_first_seen_scraped_at"]
    new_df = new_df.drop_duplicates(subset=content_cols)

    if SALES_HISTORY_PATH.exists():
        existing = pd.read_parquet(SALES_HISTORY_PATH)
        # align columns in case the table's headers ever change/add a column
        all_cols = sorted(set(existing.columns) | set(new_df.columns))
        existing = existing.reindex(columns=all_cols)
        new_df = new_df.reindex(columns=all_cols)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=[c for c in content_cols if c in all_cols], keep="first")
    else:
        existing = pd.DataFrame()
        combined = new_df

    combined.to_parquet(SALES_HISTORY_PATH, index=False)
    print(f"[sales_history] sales_history.parquet: {len(existing)} -> {len(combined)} rows "
          f"(+{len(combined) - len(existing)} genuinely new sales, "
          f"{len(new_df) - (len(combined) - len(existing))} were re-scrapes of sales we already had)")


# ---------------------------------------------------------------- stream 4
def merge_market_indices():
    """market_indices_*.jsonl (scrape_market_indices.py -- one daily-price
    array per rating-tier index x platform per run) -> reshaped to one row
    per (index, platform, date) and appended to market_indices.parquet,
    same dedup-by-date pattern as the player price-history stream."""
    records = list(read_jsonl_files("market_indices_*.jsonl"))
    if not records:
        print("[market_indices] no market_indices_*.jsonl files found, skipping")
        return

    rows = []
    for rec in records:
        index_name = rec.get("index")
        platform = rec.get("platform")
        for point in rec.get("data") or []:
            if len(point) != 2:
                continue
            ts, price = point
            rows.append((index_name, platform, ts, price))

    if not rows:
        print("[market_indices] no data points found in any run")
        return

    new_df = pd.DataFrame(rows, columns=["index_tier", "platform", "ts_ms", "price"])
    new_df["date"] = pd.to_datetime(new_df["ts_ms"], unit="ms")
    new_df = new_df.drop(columns=["ts_ms"]).drop_duplicates(
        subset=["index_tier", "platform", "date"]
    )

    if MARKET_INDICES_PATH.exists():
        existing = pd.read_parquet(MARKET_INDICES_PATH)
    else:
        existing = pd.DataFrame(columns=["index_tier", "platform", "date", "price"])

    existing_keys = existing[["index_tier", "platform", "date"]].drop_duplicates()
    merged_keys = new_df.merge(
        existing_keys, on=["index_tier", "platform", "date"],
        how="left", indicator=True,
    )
    truly_new = new_df[merged_keys["_merge"].values == "left_only"]

    if truly_new.empty:
        print(f"[market_indices] {len(new_df)} index points scraped, all already present -- 0 new")
    else:
        combined = pd.concat([existing, truly_new], ignore_index=True)
        combined.to_parquet(MARKET_INDICES_PATH, index=False)
        print(f"[market_indices] market_indices.parquet: +{len(truly_new)} new index points "
              f"({len(new_df) - len(truly_new)} were already present)")


def archive_processed_files():
    """Move raw scrape files out of scraper/scrapes/ once they're merged in,
    so the NEXT run's glob doesn't re-read (and re-hold-in-memory) every
    scrape ever taken. Confirmed live via dmesg: this was silently making
    every run more memory-hungry than the last, and is the real cause of
    repeated OOM kills today -- one python3 process alone was killed while
    holding nearly 12GB, on an 11GB VPS with no swap. The parquet files are
    the durable record from here on; the raw jsonl served its purpose."""
    PROCESSED_DIR.mkdir(exist_ok=True)
    moved = 0
    for pattern in ("market_list_*.jsonl", "player_history_*.jsonl", "player_sales_*.jsonl",
                     "market_indices_*.jsonl"):
        for path in glob.glob(str(SCRAPES_DIR / pattern)):
            p = Path(path)
            p.rename(PROCESSED_DIR / p.name)
            moved += 1
    if moved:
        print(f"[archive] moved {moved} processed scrape file(s) to {PROCESSED_DIR}")


def main():
    DATA_DIR.mkdir(exist_ok=True)
    merge_player_history()
    merge_live_snapshots()
    merge_sales_history()
    merge_market_indices()
    archive_processed_files()


if __name__ == "__main__":
    main()
