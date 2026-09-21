#!/usr/bin/env python3
"""
Stage 2: builds the full engineered feature + label panel used to train the
model, from the merged tables in data/. Ground-up rebuild (see PROJECT.md) --
every feature here is new, not ported from the old build.

Feature groups:
  - price technicals: multi-horizon returns, moving averages, RSI, MACD,
    Bollinger-style range position
  - volatility: rolling vol at several windows, a short-vs-long "regime"
    ratio, a shock/anomaly z-score for today's move
  - relative strength: player return vs. both their own rating-tier index
    and the overall market index (from market_indices.parquet)
  - liquidity/order-flow: sell-through rate, discount to listed price,
    buy-now vs. bid mix, EA tax turnover (from real sales_history rows)
  - market context: gap to EA's own suggested average price, daily
    high/low range (from live_snapshots.parquet)
  - cross-sectional: how a card's price/momentum ranks against same
    rating-tier peers on the same day
  - cyclical: day of week, weekend flag, days since this game's season
    started, and (when data/indices.parquet -- the old FC26 historical
    index export -- is present) an explicit FC26-vs-FC27 same-week lookup
  - card metadata: rating, position, league, nation, skills, etc.

Labels: forward return + a tax-aware win flag (net of EA's 5% sell tax,
not just any gross increase) at HORIZONS calendar days ahead, per
(player_id, game_version, platform).

Known simplification: rolling/return windows are computed over ROWS, not
strict calendar days -- i.e. they assume close to one row per calendar day.
`days_since_prev_point` is emitted specifically so this assumption can be
checked (and fed to the model) rather than silently trusted; the run
summary prints how common a real gap is.

Run: python3 build_features.py
Writes: data/feature_panel.parquet
"""
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

BLACKLIST_PATH = DATA_DIR / "no_market_blacklist.json"
OLD_INDICES_PATH = DATA_DIR / "indices.parquet"  # FC26-era historical index export, optional
OUT_PATH = DATA_DIR / "feature_panel.parquet"

HORIZONS = [1, 3, 7, 14, 21, 30]
ROLL_WINDOWS = [3, 7, 14, 21, 30]
EA_SELL_TAX = 0.05
WIN_BREAKEVEN = 1 / (1 - EA_SELL_TAX) - 1  # a trade must clear this to be a net win, not just gross-up

KEYS = ["player_id", "game_version", "platform"]

PLATFORM_NORMALIZE = {"ps": "console", "pc": "pc", "console": "console"}


def log(msg):
    print(msg, flush=True)


def mem_mb(df):
    return df.memory_usage(deep=True).sum() / 1e6


# ---------------------------------------------------------------- loading

def load_blacklist():
    if not BLACKLIST_PATH.exists():
        return set()
    records = json.loads(BLACKLIST_PATH.read_text())
    return {(r["player_id"], r["game_version"]) for r in records}


def load_players():
    df = pd.read_parquet(DATA_DIR / "players.parquet")
    keep = ["id", "game_version", "rating", "position", "nation", "league", "club",
            "squad", "skills", "weak_foot", "height_cm", "foot", "body_type", "age",
            "n_playstyles", "band"]
    df = df[keep].rename(columns={"id": "player_id"})
    df = df.sort_values("player_id").drop_duplicates(["player_id", "game_version"], keep="last")
    for c in ["rating", "skills", "weak_foot", "height_cm", "age"]:
        df[c] = df[c].astype("float32")
    df["n_playstyles"] = df["n_playstyles"].astype("float32")
    for c in ["position", "nation", "league", "club", "squad", "foot", "body_type", "band", "game_version"]:
        df[c] = df[c].astype("category")
    df["is_icon"] = (df["league"].astype(str) == "Icons") | (df["band"].astype(str) == "icons")
    return df


def load_prices(blacklist):
    df = pd.read_parquet(DATA_DIR / "prices_long.parquet")
    df["price"] = df["price"].replace(0, np.nan).astype("float32")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.drop_duplicates(KEYS + ["date"])
    if blacklist:
        mask = ~df.set_index(["player_id", "game_version"]).index.isin(blacklist)
        before = len(df)
        df = df[mask]
        log(f"[prices] dropped {before - len(df)} rows for blacklisted (no-market) players")
    return df


def load_indices_daily():
    """Live FC27 index data -- resampled from ~minutely to one value/day (the
    day's mean) so it lines up with the daily player-price panel."""
    df = pd.read_parquet(DATA_DIR / "market_indices.parquet")
    df["platform"] = df["platform"].map(PLATFORM_NORMALIZE).fillna(df["platform"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    daily = (df.groupby(["index_tier", "platform", "date"], as_index=False)["price"]
             .mean().rename(columns={"price": "index_value", "index_tier": "band"}))
    daily["index_value"] = daily["index_value"].astype("float32")
    daily = daily.sort_values(["band", "platform", "date"])
    g = daily.groupby(["band", "platform"], sort=False)["index_value"]
    daily["index_return_1d"] = g.pct_change().astype("float32")
    for w in ROLL_WINDOWS:
        daily[f"index_return_{w}d"] = g.pct_change(w).astype("float32")
    ret1 = daily.groupby(["band", "platform"], sort=False)["index_return_1d"]
    daily["index_vol_7d"] = ret1.transform(lambda s: s.rolling(7, min_periods=3).std()).astype("float32")
    return daily


def load_old_fc26_indices():
    if not OLD_INDICES_PATH.exists():
        log("[fc26_indices] data/indices.parquet not found -- skipping FC26-vs-FC27 "
            "same-week lookup feature (safe to skip, everything else still works)")
        return None
    df = pd.read_parquet(OLD_INDICES_PATH)
    df["platform"] = df["platform"].map(PLATFORM_NORMALIZE).fillna(df["platform"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["index_value"] = df["index_value"].astype("float32")
    df = df.sort_values(["band", "platform", "date"])
    df["day_of_season"] = (df["date"] - df.groupby(["band", "platform"])["date"].transform("min")).dt.days
    g = df.groupby(["band", "platform"], sort=False)["index_value"]
    df["index_return_7d"] = g.pct_change(7).astype("float32")
    return df[["band", "platform", "day_of_season", "index_value", "index_return_7d"]].rename(
        columns={"index_value": "fc26_index_value_same_week", "index_return_7d": "fc26_index_return_7d_same_week"}
    )


def _parse_money(s):
    if s is None:
        return np.nan
    s = str(s).strip().replace(",", "")
    if s in ("", "-", "N/A", "None", "none", "nan", "NaN"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _parse_sale_dates(date_strs, scraped_at):
    """'Sep 19, 8:48 PM' has no year -- infer it from the scrape timestamp
    (the site always shows recent history), then correct any sale that
    lands AFTER its own scrape time by rolling it back a year (a Dec sale
    seen by a Jan scrape, not a future sale)."""
    years = scraped_at.dt.year.fillna(pd.Timestamp.now().year).astype(int)
    combined = date_strs.fillna("") + ", " + years.astype(str)
    parsed = pd.to_datetime(combined, format="%b %d, %I:%M %p, %Y", errors="coerce")
    future_mask = parsed > scraped_at
    parsed = parsed.where(~future_mask, parsed - pd.DateOffset(years=1))
    return parsed


def load_sales_daily():
    path = DATA_DIR / "sales_history.parquet"
    if not path.exists():
        log("[sales] sales_history.parquet not found -- skipping liquidity features")
        return None
    df = pd.read_parquet(path)
    if df.empty:
        return None

    df["sold_for_num"] = df["sold_for"].map(_parse_money)
    df["listed_for_num"] = df["listed_for"].map(_parse_money)
    df["ea_tax_num"] = df["ea_tax"].map(_parse_money)

    scraped_at = pd.to_datetime(df["_first_seen_scraped_at"], errors="coerce")
    sale_dt = _parse_sale_dates(df["date"], scraped_at)
    df["date"] = sale_dt.dt.normalize()
    df = df.dropna(subset=["date"])

    df["is_sold"] = df["sold_for_num"].fillna(0) > 0
    df["is_buy_now"] = df["type"].fillna("").str.strip().str.lower().eq("buy now")
    df["discount_pct"] = np.where(
        df["is_sold"] & (df["listed_for_num"] > 0),
        (df["listed_for_num"] - df["sold_for_num"]) / df["listed_for_num"],
        np.nan,
    )
    df["sold_price_if_sold"] = np.where(df["is_sold"], df["sold_for_num"], np.nan)
    df["buy_now_and_sold"] = df["is_buy_now"] & df["is_sold"]

    daily = df.groupby(["player_id", "platform", "date"]).agg(
        n_listings=("is_sold", "size"),
        n_sold=("is_sold", "sum"),
        n_buy_now_sold=("buy_now_and_sold", "sum"),
        avg_sold_price=("sold_price_if_sold", "mean"),
        avg_discount_pct=("discount_pct", "mean"),
        total_ea_tax_paid=("ea_tax_num", "sum"),
    ).reset_index()

    daily["sell_through_rate"] = (daily["n_sold"] / daily["n_listings"]).astype("float32")
    daily["buy_now_share"] = (daily["n_buy_now_sold"] / daily["n_sold"].replace(0, np.nan)).astype("float32")
    for c in ["n_listings", "n_sold", "avg_sold_price", "avg_discount_pct", "total_ea_tax_paid"]:
        daily[c] = daily[c].astype("float32")
    daily["platform"] = daily["platform"].map(PLATFORM_NORMALIZE).fillna(daily["platform"])
    return daily.drop(columns=["n_buy_now_sold"])


def load_snapshots_daily():
    path = DATA_DIR / "live_snapshots.parquet"
    if not path.exists():
        log("[snapshots] live_snapshots.parquet not found -- skipping EA-average/range features")
        return None
    df = pd.read_parquet(path)
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    agg = df.groupby(["player_id", "platform", "date"]).agg(
        snap_price_last=("price", "last"),
        ea_average_last=("ea_average", "last"),
        trend_pct_last=("trend_pct", "last"),
        daily_high=("daily_high", "max"),
        daily_low=("daily_low", "min"),
        live_high=("live_high", "max"),
        live_low=("live_low", "min"),
        live_avg=("live_avg", "mean"),
        n_snapshots=("price", "size"),
    ).reset_index()
    agg["price_vs_ea_avg_pct"] = (
        (agg["snap_price_last"] - agg["ea_average_last"]) / agg["ea_average_last"]
    )
    agg["daily_range_pct"] = (
        (agg["daily_high"] - agg["daily_low"]) / agg["daily_low"]
    )
    num_cols = ["snap_price_last", "ea_average_last", "trend_pct_last", "daily_high", "daily_low",
                "live_high", "live_low", "live_avg", "price_vs_ea_avg_pct", "daily_range_pct"]
    for c in num_cols:
        agg[c] = agg[c].astype("float32")
    agg["platform"] = agg["platform"].map(PLATFORM_NORMALIZE).fillna(agg["platform"])
    return agg


# ---------------------------------------------------------------- technicals

def build_price_technicals(prices):
    df = prices.sort_values(KEYS + ["date"]).reset_index(drop=True)

    def grouped(col):
        return df.groupby(KEYS, sort=False)[col]

    prev_date = grouped("date").transform(lambda s: s.diff().dt.days)
    df["days_since_prev_point"] = prev_date.astype("float32")

    df["return_1d"] = grouped("price").transform(lambda s: s.pct_change(1)).astype("float32")
    for h in HORIZONS:
        if h == 1:
            continue
        df[f"return_{h}d"] = grouped("price").transform(lambda s: s.pct_change(h)).astype("float32")

    for w in ROLL_WINDOWS:
        mp = max(2, w // 2)
        roll_mean = grouped("price").transform(lambda s: s.rolling(w, min_periods=mp).mean())
        roll_std = grouped("price").transform(lambda s: s.rolling(w, min_periods=mp).std())
        roll_min = grouped("price").transform(lambda s: s.rolling(w, min_periods=mp).min())
        roll_max = grouped("price").transform(lambda s: s.rolling(w, min_periods=mp).max())
        df[f"sma_{w}d"] = roll_mean.astype("float32")
        df[f"vol_{w}d"] = roll_std.astype("float32")
        df[f"zscore_{w}d"] = ((df["price"] - roll_mean) / roll_std).astype("float32")
        rng = (roll_max - roll_min).replace(0, np.nan)
        df[f"pct_of_range_{w}d"] = ((df["price"] - roll_min) / rng).astype("float32")

    df["vol_regime_7_30"] = (df["vol_7d"] / df["vol_30d"].replace(0, np.nan)).astype("float32")
    # today's move sized against 30d volatility, both expressed as returns --
    # how many "typical days" today's move is worth
    vol_30d_pct = (df["vol_30d"] / df["sma_30d"].replace(0, np.nan))
    df["shock_score"] = (df["return_1d"] / vol_30d_pct.replace(0, np.nan)).astype("float32")

    df["ema_fast"] = grouped("price").transform(lambda s: s.ewm(span=7, min_periods=3).mean()).astype("float32")
    df["ema_slow"] = grouped("price").transform(lambda s: s.ewm(span=21, min_periods=5).mean()).astype("float32")
    df["macd"] = (df["ema_fast"] - df["ema_slow"]).astype("float32")
    df["macd_pct"] = (df["macd"] / df["ema_slow"].replace(0, np.nan)).astype("float32")

    df["_delta"] = grouped("price").transform(lambda s: s.diff())
    df["_gain"] = df["_delta"].clip(lower=0)
    df["_loss"] = (-df["_delta"]).clip(lower=0)
    avg_gain = df.groupby(KEYS, sort=False)["_gain"].transform(lambda s: s.rolling(14, min_periods=5).mean())
    avg_loss = df.groupby(KEYS, sort=False)["_loss"].transform(lambda s: s.rolling(14, min_periods=5).mean())
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14d"] = (100 - 100 / (1 + rs)).astype("float32")
    df = df.drop(columns=["_delta", "_gain", "_loss"])

    # dozens of columns were added one at a time above -- consolidate the
    # underlying memory layout now rather than carrying that fragmentation
    # (and pandas' own PerformanceWarning about it) through every later merge
    return df.copy()


def add_cross_platform_features(df):
    """Not true 15-minute/1-hour lead-lag -- we only scrape a handful of
    times a day, so that resolution doesn't exist in our data -- but the
    same underlying idea at the granularity we actually have: daily. Two
    signals: how far the PC/console price ratio has drifted from its own
    recent normal (mean-reversion), and each platform's most recent move
    fed as a lagged feature onto the OTHER platform's row (does yesterday's
    console move predict today's PC move) -- lagged by a day specifically
    so this is a genuine leading indicator, not same-day lookahead."""
    wide = df.pivot_table(index=["player_id", "game_version", "date"], columns="platform",
                           values=["price", "return_1d"])
    if ("price", "pc") not in wide.columns or ("price", "console") not in wide.columns:
        return df
    wide.columns = [f"{v}_{p}" for v, p in wide.columns]
    wide = wide.reset_index().sort_values(["player_id", "game_version", "date"])

    wide["pc_console_ratio"] = (wide["price_pc"] / wide["price_console"].replace(0, np.nan)).astype("float32")
    g = wide.groupby(["player_id", "game_version"], sort=False)["pc_console_ratio"]
    ratio_sma = g.transform(lambda s: s.rolling(7, min_periods=3).mean())
    ratio_std = g.transform(lambda s: s.rolling(7, min_periods=3).std())
    wide["pc_console_ratio_zscore_7d"] = ((wide["pc_console_ratio"] - ratio_sma) / ratio_std).astype("float32")

    lag_keys = ["player_id", "game_version"]
    wide["console_return_1d_prevday"] = wide.groupby(lag_keys, sort=False)["return_1d_console"].shift(1)
    wide["pc_return_1d_prevday"] = wide.groupby(lag_keys, sort=False)["return_1d_pc"].shift(1)

    keep = ["player_id", "game_version", "date", "pc_console_ratio", "pc_console_ratio_zscore_7d",
            "console_return_1d_prevday", "pc_return_1d_prevday"]
    wide = wide[keep]
    for c in ["console_return_1d_prevday", "pc_return_1d_prevday"]:
        wide[c] = wide[c].astype("float32")

    df = df.merge(wide, on=["player_id", "game_version", "date"], how="left")
    df["other_platform_return_1d_prevday"] = np.where(
        df["platform"] == "pc", df["console_return_1d_prevday"], df["pc_return_1d_prevday"]
    ).astype("float32")
    df = df.drop(columns=["console_return_1d_prevday", "pc_return_1d_prevday"])
    return df.copy()


def add_relative_strength(df, indices_daily):
    own = indices_daily.rename(columns={"band": "band"})
    df = df.merge(
        own.add_prefix("idx_own_").rename(
            columns={"idx_own_band": "band", "idx_own_platform": "platform", "idx_own_date": "date"}
        ),
        on=["band", "platform", "date"], how="left",
    )
    overall = indices_daily[indices_daily["band"] == "100"].drop(columns=["band"])
    df = df.merge(
        overall.add_prefix("idx_all_").rename(columns={"idx_all_platform": "platform", "idx_all_date": "date"}),
        on=["platform", "date"], how="left",
    )
    for h in HORIZONS:
        own_col = f"idx_own_index_return_{h}d" if h != 1 else "idx_own_index_return_1d"
        all_col = f"idx_all_index_return_{h}d" if h != 1 else "idx_all_index_return_1d"
        ret_col = f"return_{h}d" if h != 1 else "return_1d"
        if own_col in df.columns:
            df[f"rel_strength_own_{h}d"] = (df[ret_col] - df[own_col]).astype("float32")
        if all_col in df.columns:
            df[f"rel_strength_all_{h}d"] = (df[ret_col] - df[all_col]).astype("float32")
    return df.copy()


def add_cross_sectional(df):
    grp = df.groupby(["band", "platform", "date"], sort=False)
    df["price_rank_in_band"] = grp["price"].rank(pct=True).astype("float32")
    df["return_7d_rank_in_band"] = grp["return_7d"].rank(pct=True).astype("float32")
    df["vol_7d_rank_in_band"] = grp["vol_7d"].rank(pct=True).astype("float32")
    return df


def add_cyclical(df, old_fc26_indices):
    df["day_of_week"] = df["date"].dt.dayofweek.astype("int8")
    df["is_weekend"] = (df["day_of_week"] >= 5)
    season_start = df.groupby("game_version")["date"].transform("min")
    df["days_since_season_start"] = (df["date"] - season_start).dt.days.astype("float32")

    if old_fc26_indices is not None:
        fc27_mask = df["game_version"] == "fc27"
        lookup = old_fc26_indices.rename(columns={"day_of_season": "days_since_season_start"})
        merged = df.loc[fc27_mask].merge(
            lookup, on=["band", "platform", "days_since_season_start"], how="left"
        )
        df.loc[fc27_mask, "fc26_index_value_same_week"] = merged["fc26_index_value_same_week"].values
        df.loc[fc27_mask, "fc26_index_return_7d_same_week"] = merged["fc26_index_return_7d_same_week"].values
    return df


def add_labels(df):
    df = df.sort_values(KEYS + ["date"]).reset_index(drop=True)
    g = df.groupby(KEYS, sort=False)["price"]
    for h in HORIZONS:
        fwd = g.transform(lambda s: s.shift(-h))
        ret = (fwd - df["price"]) / df["price"]
        df[f"label_return_{h}d"] = ret.astype("float32")
        win = np.where(ret.isna(), np.nan, (ret > WIN_BREAKEVEN).astype(float))
        df[f"label_win_{h}d"] = win.astype("float32")
    return df.copy()


def main():
    DATA_DIR.mkdir(exist_ok=True)
    blacklist = load_blacklist()

    log("Loading players...")
    players = load_players()
    log(f"  {len(players)} players, {mem_mb(players):.1f} MB")

    log("Loading prices...")
    prices = load_prices(blacklist)
    log(f"  {len(prices)} price rows, {mem_mb(prices):.1f} MB")

    log("Building price technicals (momentum, volatility, RSI, MACD)...")
    df = build_price_technicals(prices)
    del prices
    gc.collect()
    log(f"  {len(df)} rows, {mem_mb(df):.1f} MB")

    gap_rate = (df["days_since_prev_point"] > 1).mean()
    log(f"  NOTE: {gap_rate:.1%} of rows have a >1-day gap since the previous "
        f"price point for that player (affects how exact horizon/window math is)")

    log("Adding cross-platform (PC vs console) lead-lag + ratio features...")
    df = add_cross_platform_features(df)
    log(f"  {len(df)} rows, {mem_mb(df):.1f} MB")

    log("Merging player metadata (rating, band, league, position, ...)...")
    df = df.merge(players, on=["player_id", "game_version"], how="left")
    log(f"  {len(df)} rows, {mem_mb(df):.1f} MB")

    log("Loading + merging market indices (relative strength vs. rating tier)...")
    indices_daily = load_indices_daily()
    df = add_relative_strength(df, indices_daily)
    del indices_daily
    gc.collect()
    log(f"  {len(df)} rows, {mem_mb(df):.1f} MB")

    log("Loading + merging real sales history (liquidity/order-flow features)...")
    sales_daily = load_sales_daily()
    if sales_daily is not None:
        df = df.merge(sales_daily, on=["player_id", "platform", "date"], how="left")
        del sales_daily
        gc.collect()
    log(f"  {len(df)} rows, {mem_mb(df):.1f} MB")

    log("Loading + merging live snapshots (EA-average gap, daily range)...")
    snaps_daily = load_snapshots_daily()
    if snaps_daily is not None:
        df = df.merge(snaps_daily, on=["player_id", "platform", "date"], how="left")
        del snaps_daily
        gc.collect()
    log(f"  {len(df)} rows, {mem_mb(df):.1f} MB")

    log("Adding cross-sectional rank-within-tier features...")
    df = add_cross_sectional(df)

    log("Adding cyclical + FC26/FC27 season-alignment features...")
    old_fc26_indices = load_old_fc26_indices()
    df = add_cyclical(df, old_fc26_indices)

    log("Computing forward-looking labels (tax-aware win flags, 6 horizons)...")
    df = add_labels(df)
    log(f"  final: {len(df)} rows, {len(df.columns)} columns, {mem_mb(df):.1f} MB")

    log(f"Saving to {OUT_PATH}...")
    df.to_parquet(OUT_PATH, index=False)
    log("Done.")

    log("\n=== Column summary ===")
    with pd.option_context("display.max_rows", None):
        print(df.dtypes)
    log("\n=== Null rates (top 30 by null %) ===")
    null_rates = df.isna().mean().sort_values(ascending=False)
    print(null_rates.head(30))


if __name__ == "__main__":
    main()
