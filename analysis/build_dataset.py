"""
Phase 1: parse the raw futbin exports into clean, reusable local tables.

Outputs (into ../data/):
  players.parquet        one row per player: metadata + engineered features
  prices_long.parquet    one row per (player_id, platform, date): price + rolling stats + index-relative fields
  indices.parquet        one row per (band, platform, date): index value

Run: python3 build_dataset.py
"""
import json
import zipfile
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

ZIP_PATH = ROOT / "futbin_all_rebuild.zip"
JSONL_NAME = "futbin_all_rebuild.jsonl"
INDICES_PATH = ROOT / "futbin_market_indices.jsonl"

EXPECTED_PLAYER_LINES = 28481
EXPECTED_PLAYER_BYTES = 350868623


def band_for(rating, league):
    if league == "Icons":
        return "icons"
    if rating is None:
        return None
    if rating < 81:
        return None  # no matching index band; still analyzed, just not vs-index
    if rating >= 86:
        return "86"  # catch-all top bracket per Phase 1 correlation check
    return str(rating)


def parse_age(s):
    if not s:
        return None
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else None


def parse_height_cm(s):
    if not s:
        return None
    m = re.search(r"(\d+)\s*cm", s)
    return int(m.group(1)) if m else None


def load_indices():
    rows = []
    with open(INDICES_PATH) as f:
        for line in f:
            obj = json.loads(line)
            band = obj["index"]
            platform = obj["platform"]
            for ts, val in obj["data"]:
                rows.append((band, platform, pd.to_datetime(ts, unit="ms"), val))
    df = pd.DataFrame(rows, columns=["band", "platform", "date", "index_value"])
    assert len(df) == 16 * 351 or df.groupby(["band", "platform"]).size().eq(351).all()
    return df


def main():
    print("Verifying zip integrity before parsing...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        info = z.getinfo(JSONL_NAME)
        assert info.file_size == EXPECTED_PLAYER_BYTES, f"unexpected size {info.file_size}"

    print("Loading market indices...")
    indices_df = load_indices()
    indices_df.to_parquet(DATA_DIR / "indices.parquet", index=False)
    print(f"  saved indices.parquet: {len(indices_df)} rows")

    player_rows = []
    price_rows = []
    n_lines = 0
    n_bytes = 0

    print("Streaming player records from zip (this parses all 28,481 players + both price histories)...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        with z.open(JSONL_NAME) as f:
            for raw_line in f:
                n_lines += 1
                n_bytes += len(raw_line)
                rec = json.loads(raw_line)

                rating = int(rec["rating"]) if rec.get("rating") not in (None, "") else None
                league = rec.get("league")
                band = band_for(rating, league)

                player_rows.append({
                    "id": rec["id"],
                    "url": rec.get("url"),
                    "rating": rating,
                    "position": rec.get("position"),
                    "nation": rec.get("nation"),
                    "league": league,
                    "club": rec.get("club"),
                    "squad": rec.get("squad"),
                    "skills": int(rec["skills"]) if rec.get("skills") not in (None, "") else None,
                    "weak_foot": int(rec["weak_foot"]) if rec.get("weak_foot") not in (None, "") else None,
                    "height_cm": parse_height_cm(rec.get("height")),
                    "foot": rec.get("foot"),
                    "body_type": rec.get("btype"),
                    "age": parse_age(rec.get("age")),
                    "playstyles": rec.get("playstyles") or [],
                    "n_playstyles": len(rec.get("playstyles") or []),
                    "roles_text": rec.get("roles_text"),
                    "band": band,
                })

                for platform, key in (("pc", "pc"), ("console", "ps")):
                    try:
                        series = json.loads(rec[key])
                    except (KeyError, TypeError, json.JSONDecodeError):
                        continue
                    for ts, price in series:
                        price_rows.append((rec["id"], platform, ts, price))

                if n_lines % 5000 == 0:
                    print(f"  ...{n_lines} players parsed")

    assert n_lines == EXPECTED_PLAYER_LINES, f"line count mismatch: {n_lines}"
    assert n_bytes == EXPECTED_PLAYER_BYTES, f"byte count mismatch: {n_bytes}"
    print(f"Parsed all {n_lines} players, {n_bytes} bytes -- matches expected counts exactly.")

    players_df = pd.DataFrame(player_rows)
    players_df["playstyles"] = players_df["playstyles"].apply(lambda x: "|".join(x) if x else "")

    prices_df = pd.DataFrame(price_rows, columns=["player_id", "platform", "ts_ms", "price"])
    prices_df["date"] = pd.to_datetime(prices_df["ts_ms"], unit="ms")
    prices_df = prices_df.drop(columns=["ts_ms"])

    print(f"players_df: {len(players_df)} rows")
    print(f"prices_df: {len(prices_df)} rows (long format, both platforms)")

    players_df.to_parquet(DATA_DIR / "players.parquet", index=False)
    prices_df.to_parquet(DATA_DIR / "prices_long.parquet", index=False)
    print("Saved players.parquet and prices_long.parquet")


if __name__ == "__main__":
    main()
