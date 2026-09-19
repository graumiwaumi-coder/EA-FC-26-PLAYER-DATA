"""Generates EA_FC_Trading_Model.ipynb -- run this locally to (re)build the notebook."""
import json

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                   "outputs": [], "source": text.splitlines(keepends=True)})


# ---------------------------------------------------------------------------
md("""# EA FC 26 Trading Model

**How to use this (no coding needed):**
1. `Runtime` menu -> `Run all`
2. When a file-upload box appears, upload your two exported files:
   `futbin_all_rebuild.zip` and `futbin_market_indices.jsonl`
3. Wait for it to finish (the heaviest step, feature engineering, takes the longest).
4. Scroll to the bottom for a ranked list of players and a downloadable CSV.

**Important: this is NOT live pricing.** The ranked list at the end reflects market
conditions as of the LAST DATE PRESENT IN YOUR UPLOADED FILE -- not a real-time price
feed, and not a "buy this exact listing right now" signal. If you upload an export from
a month ago, the list reflects that month-old snapshot. The closer you run this to when
you exported the data, the more current the list is. Live, real-time buy/sell decisions
are a later, not-yet-built phase -- this notebook is the periodic retrain-and-review step:
re-export your data whenever you want a refresh, then Run All again.

**To retrain on a new export later:** just come back to this notebook and do the same
thing (`Run all`, upload the new files). Everything rebuilds from scratch each time,
so it's always using your latest data, not something stale.

This notebook mirrors the analysis built in the original session, with one upgrade:
**multi-fold walk-forward validation** (Step 5) instead of a single train/test split,
so we can see whether the model's edge is consistent across different periods of the
season rather than trusting one lucky/unlucky window.
""")

# ---------------------------------------------------------------------------
md("## Setup")

code("""
import warnings
warnings.filterwarnings("ignore")

import gc
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, accuracy_score

def free_memory():
    # gc.collect() alone often doesn't return freed memory to the OS in a long-running
    # process (glibc keeps it for reuse) -- malloc_trim forces an actual release, which
    # matters here since this notebook runs everything in one continuous session instead
    # of separate scripts that would each exit and free everything automatically.
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


MIN_RATING = 75  # your stated trading focus: gold and above
MIN_TRADE_PRICE = 5_000  # excludes near-worthless fodder (a "10x return" on a 900-coin card is not a real strategy)

ROOT = Path("/content")
DATA_DIR = ROOT / "data"
MODEL_DIR = DATA_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

ZIP_PATH = ROOT / "futbin_all_rebuild.zip"
JSONL_PATH = ROOT / "futbin_all_rebuild.jsonl"
INDICES_PATH = ROOT / "futbin_market_indices.jsonl"
JSONL_NAME = "futbin_all_rebuild.jsonl"

import contextlib
import zipfile


@contextlib.contextmanager
def open_players_jsonl():
    # accepts EITHER the zipped export (futbin_all_rebuild.zip) OR the raw
    # .jsonl file uploaded directly -- some export workflows produce one,
    # some the other, and there's no reason to force a re-zip for this.
    if ZIP_PATH.exists():
        with zipfile.ZipFile(ZIP_PATH) as z:
            with z.open(JSONL_NAME) as f:
                yield f
    elif JSONL_PATH.exists():
        with open(JSONL_PATH, "rb") as f:
            yield f
    else:
        raise FileNotFoundError("Neither futbin_all_rebuild.zip nor futbin_all_rebuild.jsonl was found in /content")


def players_jsonl_size():
    if ZIP_PATH.exists():
        with zipfile.ZipFile(ZIP_PATH) as z:
            return z.getinfo(JSONL_NAME).file_size
    elif JSONL_PATH.exists():
        return JSONL_PATH.stat().st_size
    raise FileNotFoundError("Neither futbin_all_rebuild.zip nor futbin_all_rebuild.jsonl was found in /content")


pd.set_option("display.width", 160)
print("Setup complete.")
""")

# ---------------------------------------------------------------------------
md("## Step 1: Upload your data files\n\nRun this cell, then click **Choose Files** and select "
   "your player export (either `futbin_all_rebuild.zip` OR the unzipped `futbin_all_rebuild.jsonl` "
   "-- both work) and `futbin_market_indices.jsonl` (you can select both files at once).")

code("""
from google.colab import files

print("Please upload futbin_all_rebuild.zip (or futbin_all_rebuild.jsonl) AND futbin_market_indices.jsonl")
uploaded = files.upload()

for name in uploaded:
    dest = ROOT / name
    with open(dest, "wb") as f:
        f.write(uploaded[name])
    print(f"Saved {name} ({len(uploaded[name]):,} bytes) -> {dest}")

assert ZIP_PATH.exists() or JSONL_PATH.exists(), \\
    "Neither futbin_all_rebuild.zip nor futbin_all_rebuild.jsonl found -- did the upload include one of them?"
assert INDICES_PATH.exists(), "futbin_market_indices.jsonl not found -- did the upload include it?"
""")

# ---------------------------------------------------------------------------
md("## Step 2: Verify the data before trusting anything downstream\n\n"
   "Same philosophy as the original session: prove we're reading the *whole* file, not a "
   "truncated sample, before doing any analysis.")

code("""
print(f"{JSONL_NAME}: {players_jsonl_size():,} bytes (per file/zip metadata)")

n_lines = 0
n_bytes = 0
with open_players_jsonl() as f:
    for line in f:
        n_lines += 1
        n_bytes += len(line)
print(f"Streamed count: {n_lines:,} lines, {n_bytes:,} bytes")

n_idx_lines = sum(1 for _ in open(INDICES_PATH))
print(f"{INDICES_PATH.name}: {n_idx_lines} lines")

combos = set()
with open(INDICES_PATH) as f:
    for line in f:
        obj = json.loads(line)
        combos.add((obj["index"], obj["platform"]))
print(f"Index/platform combos found: {len(combos)} -> {sorted(combos)}")
print("\\nIf these numbers don't match what you expect from your export, STOP and check the files "
      "before trusting anything below.")
""")

# ---------------------------------------------------------------------------
md("## Step 3: Parse into clean tables")

code("""
def band_for(rating, league):
    if league == "Icons":
        return "icons"
    if rating is None:
        return None
    if rating < 81:
        return None
    if rating >= 86:
        return "86"
    return str(rating)


def parse_age(s):
    if not s:
        return None
    m = re.search(r"(\\d+)", s)
    return int(m.group(1)) if m else None


def parse_height_cm(s):
    if not s:
        return None
    m = re.search(r"(\\d+)\\s*cm", s)
    return int(m.group(1)) if m else None


# --- market indices ---
idx_rows = []
with open(INDICES_PATH) as f:
    for line in f:
        obj = json.loads(line)
        for ts, val in obj["data"]:
            idx_rows.append((obj["index"], obj["platform"], pd.to_datetime(ts, unit="ms"), val))
indices_df = pd.DataFrame(idx_rows, columns=["band", "platform", "date", "index_value"])
indices_df.to_parquet(DATA_DIR / "indices.parquet", index=False)
print(f"indices.parquet: {len(indices_df)} rows")

# --- players + price history ---
player_rows, price_rows = [], []
n_lines = 0
with open_players_jsonl() as f:
    for raw_line in f:
        n_lines += 1
        rec = json.loads(raw_line)
        rating = int(rec["rating"]) if rec.get("rating") not in (None, "") else None
        league = rec.get("league")
        band = band_for(rating, league)
        player_rows.append({
            "id": rec["id"], "url": rec.get("url"), "rating": rating,
            "position": rec.get("position"), "nation": rec.get("nation"),
            "league": league, "club": rec.get("club"), "squad": rec.get("squad"),
            "skills": int(rec["skills"]) if rec.get("skills") not in (None, "") else None,
            "weak_foot": int(rec["weak_foot"]) if rec.get("weak_foot") not in (None, "") else None,
            "height_cm": parse_height_cm(rec.get("height")), "foot": rec.get("foot"),
            "body_type": rec.get("btype"), "age": parse_age(rec.get("age")),
            "playstyles": "|".join(rec.get("playstyles") or []),
            "n_playstyles": len(rec.get("playstyles") or []),
            "roles_text": rec.get("roles_text"), "band": band,
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

players_df = pd.DataFrame(player_rows)
prices_df = pd.DataFrame(price_rows, columns=["player_id", "platform", "ts_ms", "price"])
prices_df["date"] = pd.to_datetime(prices_df["ts_ms"], unit="ms")
prices_df = prices_df.drop(columns=["ts_ms"])

players_df.to_parquet(DATA_DIR / "players.parquet", index=False)
prices_df.to_parquet(DATA_DIR / "prices_long.parquet", index=False)
print(f"\\nParsed {n_lines} players -> players.parquet ({len(players_df)} rows), "
      f"prices_long.parquet ({len(prices_df)} rows)")

# free the raw parsing lists -- these are large (millions of Python tuples) and are
# no longer needed now that the data is saved; keeping the whole pipeline in one
# continuous notebook session (unlike separate scripts) means memory doesn't get
# freed automatically between steps, so we free it explicitly to avoid running out.
del player_rows, price_rows
free_memory()
""")

# ---------------------------------------------------------------------------
md("## Step 4: Engineer features\n\n"
   "Player-level demand proxies (computed from the FULL ~28k-player pool) and market "
   "index momentum, then the full technical feature set (moving averages, RSI, volatility, "
   "cross-platform ratios, index-relative z-scores, multi-horizon forward returns) -- this "
   "last part is restricted to Gold+ (rating>=75) players, since that's your stated trading "
   "focus and every later step only uses Gold+ anyway. This is the slowest step.")

code("""
# --- player-level features ---
pc = prices_df[prices_df["platform"] == "pc"].copy()
pc["tradeable"] = pc["price"] > 0
pc["price_nonzero"] = pc["price"].where(pc["price"] > 0)
per_player = pc.groupby("player_id").agg(
    median_price_pc=("price_nonzero", "median"), liquidity_pc=("tradeable", "mean")
).reset_index()

players = players_df.merge(per_player, left_on="id", right_on="player_id", how="left").drop(columns=["player_id"])

POSITION_GROUP = {"GK": "GK", "CB": "DEF", "LB": "DEF", "RB": "DEF", "LWB": "DEF", "RWB": "DEF",
                  "CDM": "MID", "CM": "MID", "CAM": "MID", "LM": "MID", "RM": "MID",
                  "LW": "FWD", "RW": "FWD", "ST": "FWD", "CF": "FWD"}
players["position_group"] = players["position"].map(POSITION_GROUP)
players["is_icon"] = players["league"] == "Icons"

for col in ("league", "club", "nation"):
    agg = players.groupby(col).agg(**{
        f"{col}_price_score": ("median_price_pc", "median"),
        f"{col}_liquidity_score": ("liquidity_pc", "mean"),
        f"{col}_n_players": ("id", "count"),
    }).reset_index()
    players = players.merge(agg, on=col, how="left")

players.to_parquet(DATA_DIR / "players_features.parquet", index=False)
print(f"players_features.parquet: {len(players)} rows, {len(players.columns)} columns")

del players_df, pc, per_player, agg
free_memory()
""")

code("""
# --- index momentum features ---
idx = indices_df.sort_values(["band", "platform", "date"]).copy()
grp = idx.groupby(["band", "platform"], sort=False)["index_value"]
idx["idx_pct_change_7d"] = grp.transform(lambda s: s.pct_change(7))
idx["idx_pct_change_14d"] = grp.transform(lambda s: s.pct_change(14))
idx["idx_pct_change_30d"] = grp.transform(lambda s: s.pct_change(30))
idx["idx_pct_change_1d"] = grp.transform(lambda s: s.pct_change(1))


def momentum_percentile(s, window=90):
    def pct_rank_last(x):
        if len(x) < 5:
            return np.nan
        return (x < x[-1]).sum() / (len(x) - 1) * 100
    return s.rolling(window, min_periods=10).apply(pct_rank_last, raw=True)


idx["idx_momentum_pctile_90"] = idx.groupby(["band", "platform"], sort=False)["idx_pct_change_1d"].transform(momentum_percentile)
idx.to_parquet(DATA_DIR / "indices_features.parquet", index=False)

macro_cols = ["platform", "date", "index_value", "idx_pct_change_7d", "idx_pct_change_14d", "idx_pct_change_30d", "idx_momentum_pctile_90"]
idx100 = idx[idx["band"] == "100"][macro_cols].rename(columns={c: f"index100_{c}" for c in macro_cols if c not in ("platform", "date")})
idxicon = idx[idx["band"] == "icons"][macro_cols].rename(columns={c: f"iconsidx_{c}" for c in macro_cols if c not in ("platform", "date")})
macro = idx100.merge(idxicon, on=["platform", "date"], how="outer")
macro.to_parquet(DATA_DIR / "macro_wide.parquet", index=False)
print(f"indices_features.parquet: {len(idx)} rows, macro_wide.parquet: {len(macro)} rows")
""")

code("""
# --- full price/technical feature set ---
# Restricted to Gold+ (rating>=75) from here on. Every downstream step (model
# training, strategy analysis, today's picks, the bankroll simulation) only ever
# uses Gold+ players anyway, matching your stated trading focus -- and computing
# the heaviest feature-engineering step across the full ~28k-player pool (including
# ~21k sub-75-rated fodder cards nobody here trades) was pushing memory past what
# this notebook can reliably run within on Colab's free tier. players_features.parquet
# still covers the full pool; only the expensive price-history features are scoped down.
_gold_ids = players.loc[players["rating"] >= MIN_RATING, "id"]
_n_before, _n_players_before = len(prices_df), prices_df["player_id"].nunique()
prices_df = prices_df[prices_df["player_id"].isin(_gold_ids)].copy()
print(f"Restricting price-feature computation to rating>={MIN_RATING}: "
      f"{prices_df['player_id'].nunique()} players / {len(prices_df)} rows "
      f"(full pool was {_n_players_before} players / {_n_before} rows)")
free_memory()

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


prices = prices_df.sort_values(["player_id", "platform", "date"]).reset_index(drop=True)
del prices_df
free_memory()
prices["price_clean"] = (prices["price"].where(prices["price"] >= MIN_TRADEABLE_PRICE)).astype("float32")
# downcast early (not just at the end) -- this table has 16.8M rows and ~50 numeric
# columns, so computing everything at float64 roughly doubles peak memory for no benefit
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

prices["rsi_14"] = prices.groupby(["player_id", "platform"], sort=False)["price_clean"].transform(lambda s: rsi(s, 14))
prices["macd_norm"] = (prices["ma_7"] - prices["ma_30"]) / prices["ma_30"]
prices["skew_30"] = prices.groupby(["player_id", "platform"], sort=False)["pct_change_1d"].transform(lambda s: s.rolling(30, min_periods=10).skew())
prices["tradeable"] = (prices["price"] > 0).astype(float)
prices["liquidity_14"] = prices.groupby(["player_id", "platform"], sort=False)["tradeable"].transform(lambda s: s.rolling(14, min_periods=5).mean())
prices["bollinger_z_30"] = (prices["price_clean"] - prices["ma_30"]) / prices["std_30"]

_float_cols = prices.select_dtypes(include=["float64"]).columns
prices[_float_cols] = prices[_float_cols].astype("float32")
free_memory()
print("Base technical features done. Adding cross-platform + index context + forward returns...")
""")

code("""
# cross-platform
pivot = prices.pivot_table(index=["player_id", "date"], columns="platform", values="price_clean")
pivot = pivot.rename(columns={"pc": "pc_price_same_day", "console": "console_price_same_day"}).reset_index()
mom = prices.pivot_table(index=["player_id", "date"], columns="platform", values="pct_change_7d")
mom = mom.rename(columns={"pc": "pc_pct_change_7d_x", "console": "console_pct_change_7d_x"}).reset_index()
prices = prices.merge(pivot, on=["player_id", "date"], how="left").merge(mom, on=["player_id", "date"], how="left")
del pivot, mom
free_memory()
is_pc = prices["platform"] == "pc"
prices["other_platform_price"] = np.where(is_pc, prices["console_price_same_day"], prices["pc_price_same_day"])
prices["other_platform_pct_change_7d"] = np.where(is_pc, prices["console_pct_change_7d_x"], prices["pc_pct_change_7d_x"])
prices["own_to_other_platform_ratio"] = prices["price_clean"] / prices["other_platform_price"]
prices = prices.drop(columns=["pc_price_same_day", "console_price_same_day", "pc_pct_change_7d_x", "console_pct_change_7d_x"])

# index context
prices = prices.merge(players[["id", "rating", "band"]], left_on="player_id", right_on="id", how="left").drop(columns=["id"])
idx_cols = ["band", "platform", "date", "index_value", "idx_pct_change_7d", "idx_pct_change_14d", "idx_pct_change_30d", "idx_momentum_pctile_90"]
prices = prices.merge(idx[idx_cols].rename(columns={c: f"band_{c}" for c in idx_cols if c not in ("band", "platform", "date")}),
                       on=["band", "platform", "date"], how="left")
prices["price_to_band_index_ratio"] = prices["price_clean"] / prices["band_index_value"]
prices["band_ratio_zscore_60d"] = prices.groupby(["player_id", "platform"], sort=False)["price_to_band_index_ratio"].transform(
    lambda s: (s - s.rolling(60, min_periods=14).mean()) / s.rolling(60, min_periods=14).std())

prices = prices.merge(macro, on=["platform", "date"], how="left")
prices["price_to_index100_ratio"] = prices["price_clean"] / prices["index100_index_value"]
prices["index100_ratio_zscore_60d"] = prices.groupby(["player_id", "platform"], sort=False)["price_to_index100_ratio"].transform(
    lambda s: (s - s.rolling(60, min_periods=14).mean()) / s.rolling(60, min_periods=14).std())

# forward returns (training targets)
prices = prices.sort_values(["player_id", "platform", "date"])
g2 = prices.groupby(["player_id", "platform"], sort=False)["price_clean"]
for h in FORWARD_HORIZONS:
    fwd_price = g2.transform(lambda s, h=h: s.shift(-h))
    prices[f"fwd_return_{h}d"] = (fwd_price - prices["price_clean"]) / prices["price_clean"]
    prices[f"fwd_return_{h}d_net_tax"] = (fwd_price * (1 - SELL_TAX) - prices["price_clean"]) / prices["price_clean"]
    up = (prices[f"fwd_return_{h}d_net_tax"] > 0).astype("Int8")
    up[prices[f"fwd_return_{h}d"].isna()] = pd.NA
    prices[f"fwd_up_{h}d_net_tax"] = up

prices["day_of_week"] = prices["date"].dt.dayofweek
prices["is_weekend"] = prices["day_of_week"].isin([5, 6])
prices["days_since_start"] = (prices["date"] - prices["date"].min()).dt.days

prices.to_parquet(DATA_DIR / "prices_features.parquet", index=False)
print(f"prices_features.parquet: {len(prices)} rows, {len(prices.columns)} columns")
del prices  # free memory before modeling -- Step 5+ reload only what they need from disk
free_memory()
""")

# ---------------------------------------------------------------------------
md("## Step 5: Train + validate with MULTIPLE walk-forward folds\n\n"
   "This is the upgrade from the local session. Instead of one train/test split, we roll "
   "the split point forward through several positions in the season, training fresh each "
   "time on the trailing ~200 days before that point and testing on the data right after "
   "(with a 21-day embargo gap so no training label's forward-return window overlaps the "
   "test period). The training window is a fixed ROLLING size, not \"everything since the "
   "start\" -- that keeps every fold's memory cost about the same instead of the last fold "
   "always being the biggest and most likely to run out of RAM. If the win rate is similar "
   "across folds, the edge is real and stable, not a fluke of one lucky window.")

code("""
HORIZON = 21
EMBARGO_DAYS = HORIZON

NUMERIC_FEATURES = [
    "rating", "ma_3", "ma_7", "ma_14", "ma_30", "ma_60", "ma_90",
    "std_7", "std_14", "std_30", "std_60",
    "volatility_7", "volatility_14", "volatility_30", "volatility_60",
    "pct_change_1d", "pct_change_3d", "pct_change_7d", "pct_change_14d",
    "pct_change_30d", "pct_change_60d", "pct_change_90d",
    "pct_off_90d_high", "pct_above_90d_low", "season_position_30",
    "rsi_14", "macd_norm", "skew_30", "liquidity_14", "bollinger_z_30",
    "other_platform_price", "other_platform_pct_change_7d", "own_to_other_platform_ratio",
    "band_index_value", "band_idx_pct_change_7d", "band_idx_pct_change_14d",
    "band_idx_pct_change_30d", "band_idx_momentum_pctile_90",
    "price_to_band_index_ratio", "band_ratio_zscore_60d",
    "index100_index_value", "index100_idx_pct_change_7d", "index100_idx_pct_change_14d",
    "index100_idx_pct_change_30d", "index100_idx_momentum_pctile_90",
    "iconsidx_index_value", "iconsidx_idx_pct_change_7d", "iconsidx_idx_pct_change_14d",
    "iconsidx_idx_pct_change_30d", "iconsidx_idx_momentum_pctile_90",
    "price_to_index100_ratio", "index100_ratio_zscore_60d",
    "day_of_week", "days_since_start",
    "skills", "weak_foot", "height_cm", "age", "n_playstyles",
    "league_price_score", "league_liquidity_score", "club_price_score",
    "club_liquidity_score", "nation_price_score", "nation_liquidity_score",
]
CATEGORICAL_FEATURES = ["platform", "position", "position_group", "league", "nation", "foot", "body_type", "is_icon"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

PLAYER_LEVEL_NUMERIC = ["skills", "weak_foot", "height_cm", "age", "n_playstyles",
                        "league_price_score", "league_liquidity_score", "club_price_score",
                        "club_liquidity_score", "nation_price_score", "nation_liquidity_score"]
player_cols = ["id"] + [c for c in CATEGORICAL_FEATURES if c != "platform"] + PLAYER_LEVEL_NUMERIC
players_sub = players[players["rating"] >= MIN_RATING][player_cols]

# only columns that actually live in prices_features.parquet -- player-level columns
# (skills, league_price_score, etc.) come from players_sub via the merge below instead
price_cols = [c for c in NUMERIC_FEATURES if c != "rating" and c not in PLAYER_LEVEL_NUMERIC] + \\
    ["player_id", "platform", "date", "rating", f"fwd_return_{HORIZON}d_net_tax", f"fwd_up_{HORIZON}d_net_tax"]
price_cols = list(dict.fromkeys(price_cols))

dataset = ds.dataset(DATA_DIR / "prices_features.parquet", format="parquet")
table = dataset.to_table(columns=price_cols, filter=ds.field("rating") >= MIN_RATING)
df = table.to_pandas(split_blocks=True, self_destruct=True)
del table
float_cols = df.select_dtypes(include=["float64"]).columns
df[float_cols] = df[float_cols].astype("float32")
df = df.merge(players_sub, left_on="player_id", right_on="id", how="left").drop(columns=["id"])
for c in CATEGORICAL_FEATURES:
    df[c] = df[c].astype("category")

target_reg = f"fwd_return_{HORIZON}d_net_tax"
target_clf = f"fwd_up_{HORIZON}d_net_tax"
df = df.dropna(subset=[target_reg, target_clf])
print(f"Training data: {len(df)} rows, {df['player_id'].nunique()} players (rating>=75)")
""")

code("""
cat_idx = [ALL_FEATURES.index(c) for c in CATEGORICAL_FEATURES]

min_date, max_date = df["date"].min(), df["date"].max()
total_days = (max_date - min_date).days
print(f"Date range: {min_date.date()} to {max_date.date()} ({total_days} days)\\n")

# rolling-forward folds: test windows at 55-65%, 65-75%, 75-85%, 85-95% of the season
fold_edges = [0.55, 0.65, 0.75, 0.85, 0.95]
fold_results = []
# also collect genuinely OUT-OF-SAMPLE, de-duplicated (non-overlapping), fodder-filtered
# confident-episode returns across all folds -- this feeds the bankroll simulation later.
# Using the final model's predictions on its OWN training data (in-sample) would look
# far rosier than reality; this way the simulation only ever sees returns from test
# periods the predicting model never trained on, matching the rigor used locally.
oos_confident_returns = []

TRAIN_WINDOW_DAYS = 200  # rolling (not expanding) window -- keeps every fold's memory
# cost roughly the same. An expanding "everything since the start" window made each
# fold's training set bigger than the last, so the biggest fold (last one) was also
# the most likely to run out of RAM -- exactly where this crashed on Colab.

for i in range(len(fold_edges) - 1):
    test_start_q, test_end_q = fold_edges[i], fold_edges[i + 1]
    test_start = min_date + pd.Timedelta(days=int(total_days * test_start_q))
    test_end = min_date + pd.Timedelta(days=int(total_days * test_end_q))
    train_end = test_start - pd.Timedelta(days=EMBARGO_DAYS)
    train_start = train_end - pd.Timedelta(days=TRAIN_WINDOW_DAYS)

    train_mask = (df["date"] > train_start) & (df["date"] <= train_end)
    test_mask = (df["date"] >= test_start) & (df["date"] < test_end)
    if train_mask.sum() < 1000 or test_mask.sum() < 200:
        print(f"Fold {i+1}: skipped, not enough data")
        continue

    X_train, y_train = df.loc[train_mask, ALL_FEATURES], df.loc[train_mask, target_clf].astype(int)
    X_test, y_test = df.loc[test_mask, ALL_FEATURES], df.loc[test_mask, target_clf].astype(int)
    y_test_reg = df.loc[test_mask, target_reg]

    clf = HistGradientBoostingClassifier(categorical_features=cat_idx, max_iter=300, learning_rate=0.05,
                                          early_stopping=True, validation_fraction=0.15, random_state=42)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)

    mask_conf = proba >= 0.5
    win_rate = y_test.to_numpy()[mask_conf].mean() if mask_conf.sum() > 0 else np.nan
    med_ret = np.median(y_test_reg.to_numpy()[mask_conf]) if mask_conf.sum() > 0 else np.nan

    print(f"Fold {i+1}: train {train_start.date()}->{train_end.date()}, test {test_start.date()}->{test_end.date()} "
          f"| n_train={train_mask.sum()} n_test={test_mask.sum()} "
          f"| AUC={auc:.3f} | confident(>=0.5) n={mask_conf.sum()} win_rate={win_rate:.1%} median_return={med_ret:.1%}")
    fold_results.append({"fold": i+1, "auc": auc, "win_rate": win_rate, "median_return": med_ret, "n_confident": mask_conf.sum()})

    # de-duplicate this fold's confident test rows to ~independent episodes (daily
    # snapshots of the same player are heavily autocorrelated, not separate trades)
    # and stash the fodder-filtered returns for the bankroll simulation
    fold_test_meta = df.loc[test_mask, ["player_id", "platform", "ma_7"]].copy()
    fold_test_meta["proba"] = proba
    fold_test_meta[target_reg] = y_test_reg.to_numpy()
    fold_test_meta = fold_test_meta.sort_values(["player_id", "platform"])
    fold_test_meta["day_rank"] = fold_test_meta.groupby(["player_id", "platform"]).cumcount()
    dedup_fold = fold_test_meta[(fold_test_meta["day_rank"] % HORIZON == 0) &
                                 (fold_test_meta["ma_7"] >= MIN_TRADE_PRICE) &
                                 (fold_test_meta["proba"] >= 0.5)]
    oos_confident_returns.extend(dedup_fold[target_reg].dropna().tolist())
    del fold_test_meta, dedup_fold

    # free each fold's data/model before the next iteration -- otherwise 4 folds'
    # worth of train/test copies and fitted models stay resident simultaneously
    del X_train, y_train, X_test, y_test, y_test_reg, clf, proba, mask_conf
    free_memory()

oos_confident_returns = np.array(oos_confident_returns)
print(f"\\nCollected {len(oos_confident_returns)} genuinely out-of-sample, de-duplicated, "
      f"fodder-filtered confident episodes across all folds: mean={oos_confident_returns.mean():.1%}, "
      f"median={np.median(oos_confident_returns):.1%}, win_rate={(oos_confident_returns>0).mean():.1%}")
print("(this is what the bankroll simulation in Step 8 uses -- NOT the final model's predictions on")
print("its own training data, which would look unrealistically good)")

fold_df = pd.DataFrame(fold_results)
print("\\n=== Summary across folds ===")
print(fold_df)
print(f"\\nAUC: mean={fold_df['auc'].mean():.3f} std={fold_df['auc'].std():.3f}")
print(f"Win rate: mean={fold_df['win_rate'].mean():.1%} std={fold_df['win_rate'].std():.1%}")
print("\\nIf win_rate is fairly consistent across folds (similar mean, small std), the edge looks real.")
print("If one fold is wildly different from the others, treat the overall edge with more caution --")
print("it may be tied to a specific period's market conditions rather than a stable pattern.")
""")

code("""
# train the FINAL production model on ALL available data (for making live predictions).
# Free memory before AND between the two fits -- fitting two large boosted models
# back-to-back without releasing the first's internal buffers was the last OOM point.
free_memory()

X_all = df[ALL_FEATURES]
y_all_clf = df[target_clf].astype(int)

clf_final = HistGradientBoostingClassifier(categorical_features=cat_idx, max_iter=300, learning_rate=0.05,
                                            early_stopping=True, validation_fraction=0.15, random_state=42)
clf_final.fit(X_all, y_all_clf)
joblib.dump(clf_final, MODEL_DIR / f"clf_{HORIZON}d.joblib")
print("Classifier trained and saved.")

del y_all_clf
free_memory()

y_all_reg = df[target_reg]
reg_final = HistGradientBoostingRegressor(categorical_features=cat_idx, max_iter=300, learning_rate=0.05,
                                           early_stopping=True, validation_fraction=0.15, random_state=42)
reg_final.fit(X_all, y_all_reg)
joblib.dump(reg_final, MODEL_DIR / f"reg_{HORIZON}d.joblib")
print("Regressor trained and saved.")

joblib.dump(ALL_FEATURES, MODEL_DIR / "feature_list.joblib")
del X_all, y_all_reg
free_memory()
print("Final production models trained on all available data and saved.")
""")

# ---------------------------------------------------------------------------
md("## Step 6: Strategy analysis -- weekly cycle, promo/release timing, month seasonality")

code("""
# weekly cycle
dataset2 = ds.dataset(DATA_DIR / "prices_features.parquet", format="parquet")
wk_cols = ["player_id", "platform", "date", "rating", "price_clean", "ma_7", "day_of_week"]
wkdf = dataset2.to_table(columns=wk_cols, filter=(ds.field("rating") >= 75) & (ds.field("platform") == "pc")).to_pandas()
wkdf = wkdf.dropna(subset=["price_clean", "ma_7"])
wkdf["dev_from_own_weekly_avg"] = wkdf["price_clean"] / wkdf["ma_7"] - 1
dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
wkdf["dow_name"] = wkdf["day_of_week"].map(dict(enumerate(dow_names)))
print("Average deviation from own trailing 7-day price average, by day of week:")
print((wkdf.groupby("dow_name")["dev_from_own_weekly_avg"].mean().reindex(dow_names) * 100).round(2).astype(str) + "%")
""")

code("""
# promo/release timing
rel_cols = ["player_id", "platform", "date", "rating", "price_clean"]
reldf = dataset2.to_table(columns=rel_cols, filter=(ds.field("platform") == "pc") & (ds.field("rating") >= 75)).to_pandas()
reldf = reldf.dropna(subset=["price_clean"]).sort_values(["player_id", "date"])
first_date_overall = reldf["date"].min()
first_seen = reldf.groupby("player_id")["date"].min().rename("release_date")
reldf = reldf.merge(first_seen, on="player_id")
cutoff = first_date_overall + pd.Timedelta(days=14)
releases = reldf[reldf["release_date"] > cutoff].copy()
releases["days_since_release"] = (releases["date"] - releases["release_date"]).dt.days

pivot_r = releases.pivot_table(index="player_id", columns="days_since_release", values="price_clean")
results = []
for buy_day in (0, 3, 7, 14, 21):
    if buy_day not in pivot_r.columns:
        continue
    for hold in (7, 14, 21, 30):
        sell_day = buy_day + hold
        if sell_day not in pivot_r.columns:
            continue
        bp, sp = pivot_r[buy_day], pivot_r[sell_day]
        valid = bp.notna() & sp.notna() & (bp > 0)
        if valid.sum() < 20:
            continue
        ret = (sp[valid] * 0.95 - bp[valid]) / bp[valid]
        results.append((buy_day, hold, valid.sum(), ret.mean(), ret.median(), (ret > 0).mean()))
res_df = pd.DataFrame(results, columns=["buy_day", "hold_days", "n", "mean_ret", "median_ret", "win_rate"])
print("Buy-day / hold-length grid (median return, net of tax), best first:")
print(res_df.sort_values("median_ret", ascending=False).to_string(index=False))
""")

# ---------------------------------------------------------------------------
md("## Step 7: Ranked picks as of your data's most recent snapshot date\n\n"
   "Ranked by ABSOLUTE expected coin profit (not %). **Read the date this cell prints "
   "before trusting the list below** -- it's the last date in the file you uploaded, "
   "not necessarily today's real-world date.")

code("""
BUDGET_TIERS = [("small (<=200k)", 0, 200_000), ("medium (200k-1M)", 200_000, 1_000_000), ("large (>1M)", 1_000_000, float("inf"))]

latest_date = dataset2.to_table(columns=["date"]).to_pandas()["date"].max()
days_old = (pd.Timestamp.now().normalize() - latest_date).days
print(f"*** These picks reflect market conditions as of {latest_date.date()} -- the last date in your")
print(f"*** uploaded export -- which is {days_old} day(s) before today's real date. This is NOT a live")
print(f"*** price feed. Re-export fresh data and re-run this notebook for a more current read.")

pick_price_native = [c for c in NUMERIC_FEATURES if c != "rating" and c not in PLAYER_LEVEL_NUMERIC]
pick_cols = list(dict.fromkeys(pick_price_native + ["player_id", "date", "price_clean", "platform", "rating"]))
filt = (ds.field("rating") >= MIN_RATING) & (ds.field("date") == pd.Timestamp(latest_date)) & (ds.field("platform") == "pc")
today = dataset2.to_table(columns=pick_cols, filter=filt).to_pandas()
today = today.dropna(subset=["price_clean"])
today = today.merge(players_sub, left_on="player_id", right_on="id", how="left").drop(columns=["id"])
for c in CATEGORICAL_FEATURES:
    today[c] = today[c].astype("category")

today["proba"] = clf_final.predict_proba(today[ALL_FEATURES])[:, 1]

# expected return per confidence bucket, using THIS run's fold results as the empirical estimate
bucket_return = fold_df["median_return"].mean() if len(fold_df) else 0.10
today["expected_return_pct"] = np.where(today["proba"] >= 0.7, bucket_return * 1.2,
                                 np.where(today["proba"] >= 0.6, bucket_return * 1.05,
                                 np.where(today["proba"] >= 0.5, bucket_return, -0.05)))
today["expected_absolute_profit"] = today["price_clean"] * today["expected_return_pct"]

meta = players[["id", "club"]].rename(columns={"id": "player_id"})
today = today.merge(meta, on="player_id", how="left")

all_picks = []
for label, lo, hi in BUDGET_TIERS:
    print(f"\\n=== BUDGET TIER: {label} ===")
    tier = today[(today["price_clean"] > lo) & (today["price_clean"] <= hi) & (today["proba"] >= 0.5)]
    top = tier.sort_values("expected_absolute_profit", ascending=False).head(10)
    if len(top) == 0:
        print("  (no confident picks in this tier today)")
        continue
    display_cols = ["rating", "position", "club", "price_clean", "proba", "expected_return_pct", "expected_absolute_profit"]
    print(top[display_cols].to_string(index=False))
    top = top.copy()
    top["budget_tier"] = label
    all_picks.append(top)

if all_picks:
    picks_out = pd.concat(all_picks)[["budget_tier", "rating", "position", "club", "price_clean", "proba", "expected_return_pct", "expected_absolute_profit"]]
    picks_out.to_csv(DATA_DIR / "todays_picks.csv", index=False)
    print(f"\\nSaved {len(picks_out)} picks to todays_picks.csv")
""")

# ---------------------------------------------------------------------------
md("## Step 8: Bankroll simulation -- realistic range of outcomes, not a hopeful guess\n\n"
   "Bootstrap-resamples REAL historical trade outcomes (filtered to exclude near-worthless "
   "fodder cards, which produce misleading 10-30x outlier swings on a few thousand coins) "
   "and simulates compounding across ~11 three-week cycles (roughly one season), diversified "
   "across 20 simultaneous positions per cycle.")

code("""
N_SIMULATIONS = 20000
N_CYCLES = 11
STARTING_BANKROLLS = [100_000, 500_000, 1_000_000]

# IMPORTANT: use the out-of-sample returns collected during Step 5's walk-forward
# folds (oos_confident_returns), NOT clf_final's predictions on its own training
# data -- evaluating a model on data it was trained on looks far rosier than
# reality (a fitted model partially "remembers" its own training examples).
confident_returns = oos_confident_returns
print(f"Using {len(confident_returns)} out-of-sample confident episodes from Step 5's folds: "
      f"mean={confident_returns.mean():.1%}, median={np.median(confident_returns):.1%}, "
      f"win_rate={(confident_returns>0).mean():.1%}")


def simulate(returns, starting_bankroll, reinvest_fraction, n_positions, rng):
    bankroll = np.full(N_SIMULATIONS, float(starting_bankroll))
    for _ in range(N_CYCLES):
        sampled = rng.choice(returns, size=(N_SIMULATIONS, n_positions), replace=True)
        portfolio_return = sampled.mean(axis=1)
        invested = bankroll * reinvest_fraction
        bankroll = (bankroll - invested) + invested * (1 + portfolio_return)
    return bankroll


rng = np.random.default_rng(42)
for start in STARTING_BANKROLLS:
    b = simulate(confident_returns, start, reinvest_fraction=1.0, n_positions=20, rng=rng)
    p10, p50, p90 = np.percentile(b, [10, 50, 90])
    print(f"Start {start:,}: median={p50:,.0f}  p10={p10:,.0f}  p90={p90:,.0f}  "
          f"chance below start={ (b<start).mean():.1%}  chance of 1M+={ (b>=1_000_000).mean():.1%}")

print("\\nReminder: this assumes the historical edge holds, you can always find ~20 fresh confident")
print("opportunities each cycle, and execution is disciplined. Treat it as a plausible RANGE, not a promise.")
""")

# ---------------------------------------------------------------------------
md("## Step 9: Download your results\n\n"
   "This saves the trained model files and today's picks so you can keep them, and offers "
   "a direct download of the picks CSV.")

code("""
from google.colab import files as gfiles

print("Files saved in this Colab session's /content/data/models/:")
for f in sorted(MODEL_DIR.glob("*")):
    print(" ", f)

if (DATA_DIR / "todays_picks.csv").exists():
    gfiles.download(str(DATA_DIR / "todays_picks.csv"))
    print("\\nDownloading todays_picks.csv...")
else:
    print("\\nNo picks CSV was generated (no confident picks today).")
""")

with open("EA_FC_Trading_Model.ipynb", "w") as fh:
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": [], "name": "EA_FC_Trading_Model.ipynb"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    json.dump(notebook, fh, indent=1)

print(f"Wrote EA_FC_Trading_Model.ipynb with {len(cells)} cells")
