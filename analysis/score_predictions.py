#!/usr/bin/env python3
"""
Stage 3: turns the trained 12-model set (models/latest) into an actionable
BUY list, a SELL list for currently-held cards, and exact position sizes
against a manually-entered bankroll. Design comes from a 20-question
intake on 2026-09-21 (verbatim answers live in chat history, not repeated
here) -- see recommendations_db.py's docstring for how results are stored.

Scope, exactly as specified: PC market only, FC27 only (the only game
actually being traded), Gold+ (rating >= 75), no-market (untradeable)
cards excluded. Console and FC26 rows are still real signal INSIDE the
models (see train_model.py) but are never something to buy or sell here.
SBC fodder is deliberately NOT filtered out separately -- if a Gold+ card
clears the bar, it's shown, whatever it's normally bought for.

Per PC/FC27 candidate, all 6 horizons are scored, but only ONE is shown
per card: whichever horizon clears the confidence bar (default 60% win
probability) with the best OPPORTUNITY-COST-ADJUSTED daily return, not
just the highest headline number -- a 7-day play that's slow to actually
sell can lose to a faster 3-day play with a smaller predicted return, and
that tradeoff is handled here automatically rather than left to the
raw prediction. "Slow to sell" is estimated from real sales_history
turnover in the last 14 days (build_features.py's own liquidity
aggregation, reused rather than re-implemented), not modeled by the
price-prediction models themselves.

Position sizing: fractional Kelly using the CORRECTED formula from
PROJECT.md -- f* = p/a - q/b (p=win probability, q=1-p, a=average loss
magnitude, b=expected win magnitude) -- not the textbook f* = p - q/b,
which assumes total loss on failure and is wrong for FUT trades (a losing
trade usually still recovers most of its value). "a" is computed once per
horizon at train time from real historical losses and stored in
meta.json. Full Kelly is known to be high-variance in practice; a 0.75x
fractional multiplier is used as "aggressive but not reckless" for the
stated risk tolerance, with a per-position cap (see size_positions()).
Each row is sized independently against the FULL bankroll -- the buy list
is a menu to pick from, not a portfolio pre-allocated across every row
shown, so sizing is never diluted by how many other cards also cleared
the confidence bar. Predicted returns are also capped (MAX_REALISTIC_RETURN)
before they drive ranking or sizing, since a card with almost no real
price history -- true of every FC27 card in its first couple of weeks --
can make the regressor extrapolate into triple-digit "returns" that
aren't a real trading edge, just noise from launch-week price corrections
in the FC26 training data.

Run: python3 score_predictions.py [--bankroll 15000]
(bankroll is optional -- omit it to reuse whatever was last saved, either
by a previous --bankroll or via the dashboard; defaults to 15000 coins the
very first time this has ever been run.)

Writes: BUY + SELL rows into data/recommendations.db, refreshes every open
holding's cached re-score (data/recommendations.db's holdings table), and
grades any past BUY recommendation whose horizon has now elapsed against
real subsequent prices.
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import recommendations_db as rdb
from build_features import EA_SELL_TAX, WIN_BREAKEVEN, load_sales_daily

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
FEATURE_PANEL_PATH = DATA_DIR / "feature_panel.parquet"
PRICES_LONG_PATH = DATA_DIR / "prices_long.parquet"
BLACKLIST_PATH = DATA_DIR / "no_market_blacklist.json"

CONFIDENCE_BAR = 0.60           # minimum win probability to appear on the BUY list at all
SELL_WIN_PROB_THRESHOLD = 0.50  # a held card whose best remaining horizon can't clear this -> SELL
KELLY_MULTIPLIER = 0.75         # fractional Kelly -- "aggressive" without full-Kelly's blowup risk
MAX_POSITION_FRACTION = 0.25    # no single card eats more than this share of bankroll
MAX_REALISTIC_RETURN = 1.0      # predicted returns are clipped here before ranking/sizing -- a
                                 # brand-new card with almost no price history (everything right
                                 # now, FC27 being 5 days old) can make the regressor extrapolate
                                 # into triple/quadruple-digit "returns" it learned from FC26
                                 # launch-week corrections; those aren't real trading edges, and
                                 # left uncapped they'd both dominate the ranking (via
                                 # effective_daily_return) and blow up the Kelly fraction.
LIQUIDITY_WINDOW_DAYS = 14
MIN_HISTORY_POINTS = 10         # fewer real PC/FC27 price points than this -> "thin_history" flag
TOP_RATIONALE_N = 5             # only the top N buy picks get a written rationale, to save compute
GRADE_MATCH_TOLERANCE_DAYS = 3  # how far from eval_date a real price point may be to grade against


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ loading

def load_models():
    latest_dir = MODELS_DIR / "latest"
    if not latest_dir.exists():
        raise SystemExit(f"{latest_dir} not found -- run train_model.py at least once first.")
    meta = json.loads((latest_dir / "meta.json").read_text())
    models = {}
    for h in meta["horizons"]:
        models[h] = {
            "clf": lgb.Booster(model_file=str(latest_dir / f"clf_{h}d.txt")),
            "reg": lgb.Booster(model_file=str(latest_dir / f"reg_{h}d.txt")),
        }
    return meta, models


def load_blacklist():
    if not BLACKLIST_PATH.exists():
        return set()
    records = json.loads(BLACKLIST_PATH.read_text())
    return {(r["player_id"], r["game_version"]) for r in records}


def load_candidate_panel(meta):
    """PC, FC27, rating>=75 rows only -- everything else in feature_panel.parquet
    (FC26 history, console rows) exists purely as training signal for the
    models, never as something to score/trade here. Loads every date for
    each series (not just the latest) so real history-length can be
    counted for the thin-history flag; the caller reduces to one row per
    player afterward."""
    feature_cols = meta["feature_cols"]
    load_cols = list(dict.fromkeys(
        ["player_id", "game_version", "platform", "date", "price"] + feature_cols
    ))
    schema_names = {f.name for f in pq.ParquetFile(str(FEATURE_PANEL_PATH)).schema_arrow}
    load_cols = [c for c in load_cols if c in schema_names]
    df = pd.read_parquet(
        FEATURE_PANEL_PATH, columns=load_cols,
        filters=[("rating", ">=", meta["min_rating"]), ("game_version", "=", "fc27"),
                 ("platform", "=", "pc")],
    )
    return df, feature_cols


def apply_blacklist(df, blacklist):
    """Defense in depth: build_features.py already drops blacklisted
    players' price rows entirely before the feature panel is built, so
    this should normally be a no-op. Kept explicit here in case a player
    gets blacklisted between the last feature build and this scoring run."""
    if not blacklist or df.empty:
        return df
    keys = list(zip(df["player_id"].tolist(), df["game_version"].astype(str).tolist()))
    mask = [k not in blacklist for k in keys]
    return df[mask].reset_index(drop=True)


def latest_snapshot(df):
    """The most recent row per player that actually has a known price --
    a row can exist for a date with price=NaN (a scrape gap on that exact
    day), and picking that as "the" current row would recommend a card at
    an unknown price, which isn't actionable. Preferring the latest PRICED
    row means price_at_snapshot is always real."""
    priced = df[df["price"].notna()]
    idx = priced.groupby("player_id")["date"].idxmax()
    return priced.loc[idx].reset_index(drop=True)


def attach_urls(latest):
    """feature_panel.parquet doesn't carry the futbin url (build_features.py's
    load_players() keeps only modeling-relevant columns), so it's pulled
    straight from players.parquet -- purely for display/linking, never
    used as a feature."""
    players_path = DATA_DIR / "players.parquet"
    if not players_path.exists():
        latest["url"] = None
        return latest
    lookup = pd.read_parquet(players_path, columns=["id", "game_version", "url"])
    lookup = lookup.rename(columns={"id": "player_id"})
    lookup = lookup.sort_values("player_id").drop_duplicates(["player_id", "game_version"], keep="last")
    latest = latest.merge(lookup, on=["player_id", "game_version"], how="left")
    return latest


def loss_magnitudes(meta):
    return {s["horizon"]: s.get("avg_loss_magnitude", 0.10) for s in meta["summaries"]}


def attach_liquidity(latest):
    """Estimated days to sell ONE copy at current market activity, from
    real sales_history turnover over the trailing LIQUIDITY_WINDOW_DAYS --
    the direct opportunity-cost signal the person asked for ("if it's
    going to take me three days to sell a card... it should choose
    faster-moving cards instead"). Falls back to the market-wide median
    for cards with no recent sales data of their own (new/thin cards),
    which is a reasonable default and also exactly the population the
    thin_history flag separately warns about."""
    daily = load_sales_daily()
    if daily is None or daily.empty:
        latest["est_days_to_sell"] = 5.0
        return latest
    daily = daily[daily["platform"] == "pc"]
    if daily.empty:
        latest["est_days_to_sell"] = 5.0
        return latest
    cutoff = daily["date"].max() - pd.Timedelta(days=LIQUIDITY_WINDOW_DAYS)
    recent = daily[daily["date"] > cutoff]
    agg = recent.groupby("player_id")["n_sold"].sum().rename("total_sold_recent").reset_index()
    agg["avg_daily_sold"] = agg["total_sold_recent"] / LIQUIDITY_WINDOW_DAYS

    latest = latest.merge(agg[["player_id", "avg_daily_sold"]], on="player_id", how="left")
    has_data = latest["avg_daily_sold"] > 0
    fallback = float((1 / latest.loc[has_data, "avg_daily_sold"]).median()) if has_data.any() else 5.0
    latest["est_days_to_sell"] = np.where(has_data, 1 / latest["avg_daily_sold"], np.nan)
    latest["est_days_to_sell"] = latest["est_days_to_sell"].fillna(fallback)
    return latest.drop(columns=["avg_daily_sold"])


# ------------------------------------------------------------------ scoring

def cat_to_plain(s):
    """Turns a pandas category Series into plain python str/None values --
    sqlite3's DB-API can't bind pandas Categorical/NaN directly."""
    if isinstance(s.dtype, pd.CategoricalDtype):
        s = s.astype(object)
    return s.where(s.notna(), None)


def prepare_for_predict(df, feature_cols, cat_cols):
    X = df[feature_cols].copy()
    for c in feature_cols:
        if c in cat_cols:
            X[c] = X[c].astype("category")
        elif pd.api.types.is_bool_dtype(X[c].dtype):
            X[c] = X[c].astype("float32")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")
    return X


def predict_all_horizons(df, feature_cols, models, meta):
    X = prepare_for_predict(df, feature_cols, meta["categorical_cols"])
    preds = {}
    for h, m in models.items():
        win_prob = np.asarray(m["clf"].predict(X), dtype="float64")
        exp_return = np.asarray(m["reg"].predict(X), dtype="float64")
        # see MAX_REALISTIC_RETURN -- guards against the regressor
        # extrapolating wildly on cards with almost no real price history
        exp_return = np.clip(exp_return, None, MAX_REALISTIC_RETURN)
        preds[h] = (win_prob, exp_return)
    return preds


def _base_rows(rows, extra=None):
    out = {
        "player_id": rows["player_id"].values,
        "platform": rows["platform"].values,
        "game_version": rows["game_version"].values,
        "url": rows["url"].values,
        "rating": rows["rating"].values,
        "position": cat_to_plain(rows["position"]).values,
        "club": cat_to_plain(rows["club"]).values,
        "league": cat_to_plain(rows["league"]).values,
        "squad": cat_to_plain(rows["squad"]).values,
        "promo_family": cat_to_plain(rows["promo_family"]).values,
        "price_at_snapshot": rows["price"].values,
        "snapshot_date": pd.to_datetime(rows["date"]).values,
    }
    if extra:
        out.update(extra)
    return out


def build_buy_candidates(latest, feature_cols, models, meta):
    preds = predict_all_horizons(latest, feature_cols, models, meta)
    rationale_cols = ["pct_of_range_30d", "return_7d", "price_rank_in_band", "price_rank_in_promo"]
    extra_common = {c: (latest[c].values if c in latest.columns else np.full(len(latest), np.nan))
                     for c in rationale_cols}
    extra_common["est_days_to_sell"] = latest["est_days_to_sell"].values
    extra_common["n_history_points"] = latest["n_history_points"].values

    frames = []
    for h, (win_prob, exp_return) in preds.items():
        sub = pd.DataFrame(_base_rows(latest, extra_common))
        sub["horizon_days"] = h
        sub["predicted_win_prob"] = win_prob
        sub["predicted_return_pct"] = exp_return
        frames.append(sub)
    all_h = pd.concat(frames, ignore_index=True)

    all_h = all_h[(all_h["predicted_win_prob"] >= CONFIDENCE_BAR) & (all_h["predicted_return_pct"] > 0)]
    if all_h.empty:
        return all_h

    all_h["eval_date"] = all_h["snapshot_date"] + pd.to_timedelta(all_h["horizon_days"], unit="D")
    all_h["effective_daily_return"] = (
        all_h["predicted_return_pct"] / (all_h["horizon_days"] + all_h["est_days_to_sell"])
    )

    best_idx = all_h.groupby("player_id")["effective_daily_return"].idxmax()
    best = all_h.loc[best_idx].reset_index(drop=True)
    best["confidence_flag"] = np.where(best["n_history_points"] < MIN_HISTORY_POINTS,
                                        "thin_history", "normal")
    return best.sort_values("effective_daily_return", ascending=False).reset_index(drop=True)


def kelly_fraction(p, b, a):
    """Corrected Kelly (PROJECT.md): f* = p/a - q/b, for a trade that can
    partially lose rather than lose everything. Negative results (a poor
    risk/reward even at a qualifying win probability) are clipped to 0,
    not shorted -- this system only ever recommends buying."""
    q = 1 - p
    with np.errstate(divide="ignore", invalid="ignore"):
        f = p / a - q / b
    return np.clip(np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)


def size_positions(buy, loss_mags, bankroll):
    """Each row is sized independently against the FULL bankroll, capped at
    MAX_POSITION_FRACTION per card -- "if you buy this one, here's how much
    to put on it." This does NOT assume you'll buy every card shown (the
    confidence-bar list can easily run to 100+ rows); it's a menu you pick
    from, not a portfolio pre-allocated across everything on the page. An
    earlier version rescaled every position down so the WHOLE displayed
    list summed to <=1x bankroll, which silently collapsed every single
    recommended_quantity to 0 whenever more than a handful of cards
    cleared the confidence bar -- exactly what happened on the first real
    run. If you go through and buy several of these one after another,
    your bankroll number in the dashboard is what you update between
    picks (rerunning re-sizes everything left against what's actually
    still available), not something this function tracks for you."""
    if buy.empty:
        buy = buy.copy()
        buy["kelly_fraction"] = pd.Series(dtype="float64")
        buy["recommended_coins"] = pd.Series(dtype="float64")
        buy["recommended_quantity"] = pd.Series(dtype="int64")
        return buy

    buy = buy.copy()
    a = buy["horizon_days"].map(loss_mags).fillna(0.10).to_numpy()
    raw_f = kelly_fraction(buy["predicted_win_prob"].to_numpy(),
                            buy["predicted_return_pct"].to_numpy(), a)
    raw_f = np.minimum(raw_f * KELLY_MULTIPLIER, MAX_POSITION_FRACTION)

    buy["kelly_fraction"] = raw_f
    coins_allocated = raw_f * bankroll
    quantity = np.floor(
        np.where(buy["price_at_snapshot"] > 0, coins_allocated / buy["price_at_snapshot"], 0)
    ).astype(int)
    buy["recommended_quantity"] = quantity
    # actual coins spent reflects whole copies only -- a fractional-Kelly
    # allocation that can't buy even one copy at the current price rounds
    # down to 0, and is still shown (ranked normally), just unaffordable
    # right now rather than hidden.
    buy["recommended_coins"] = quantity * buy["price_at_snapshot"]
    return buy


def make_rationale(row):
    bullets = []
    if pd.notna(row.get("pct_of_range_30d")) and row["pct_of_range_30d"] < 0.30:
        bullets.append(f"Near the bottom of its 30-day price range ({row['pct_of_range_30d']:.0%} of "
                        f"range) -- more room to rise before it's expensive again")
    if pd.notna(row.get("return_7d")):
        if row["return_7d"] > 0.02:
            bullets.append(f"Already trending up over the last 7 days ({row['return_7d']:+.1%})")
        elif row["return_7d"] < -0.02:
            bullets.append(f"Down {row['return_7d']:.1%} over the last 7 days -- the model expects "
                            f"a bounce here, not a continued slide")
    if pd.notna(row.get("price_rank_in_promo")) and row.get("promo_family"):
        bullets.append(f"Cheap relative to other {row['promo_family']} cards right now "
                        f"({row['price_rank_in_promo']:.0%} price percentile within that promo)")
    elif pd.notna(row.get("price_rank_in_band")):
        bullets.append(f"{row['price_rank_in_band']:.0%} price percentile among same-rating-tier cards")
    if row.get("confidence_flag") == "thin_history":
        bullets.append("Thin price history so far -- treat this one as less certain than the raw "
                        "probability suggests")
    if pd.notna(row.get("est_days_to_sell")):
        bullets.append(f"~{row['est_days_to_sell']:.1f} day(s) estimated to sell at current market activity")
    return " • ".join(bullets[:3]) if bullets else None


def attach_rationale(buy):
    buy = buy.copy()
    buy["rationale"] = None
    if buy.empty:
        return buy
    top = buy.head(TOP_RATIONALE_N).index
    buy.loc[top, "rationale"] = [make_rationale(buy.loc[i]) for i in top]
    return buy


def build_sell_list(holdings, candidate_latest, feature_cols, models, meta):
    """Re-scores every open holding with the SAME 12 models used for BUY
    candidates -- the person's spec explicitly asked for this to reuse the
    model logic rather than a separate heuristic. Returns one row per
    holding with its best remaining horizon's outlook and current
    unrealized P&L; the caller decides which of these actually cross the
    SELL threshold."""
    held_ids = holdings["player_id"].unique().tolist()
    held_rows = candidate_latest[candidate_latest["player_id"].isin(held_ids)].copy()
    missing = set(held_ids) - set(held_rows["player_id"].tolist())
    if missing:
        log(f"  WARNING: {len(missing)} held player_id(s) have no current PC/FC27 price data to "
            f"re-score: {sorted(missing)}")
    if held_rows.empty:
        return pd.DataFrame()

    preds = predict_all_horizons(held_rows, feature_cols, models, meta)
    frames = []
    for h, (win_prob, exp_return) in preds.items():
        sub = pd.DataFrame(_base_rows(held_rows))
        sub["horizon_days"] = h
        sub["predicted_win_prob"] = win_prob
        sub["predicted_return_pct"] = exp_return
        frames.append(sub)
    all_h = pd.concat(frames, ignore_index=True)

    best_idx = all_h.groupby("player_id")["predicted_win_prob"].idxmax()
    best = all_h.loc[best_idx].reset_index(drop=True)
    best["eval_date"] = best["snapshot_date"] + pd.to_timedelta(best["horizon_days"], unit="D")

    merged = holdings.merge(best, on="player_id", how="inner", suffixes=("_hold", ""))
    merged["unrealized_return_pct"] = (
        (merged["price_at_snapshot"] * (1 - EA_SELL_TAX) - merged["buy_price_per_unit"])
        / merged["buy_price_per_unit"]
    )
    return merged


# ------------------------------------------------------------------ grading

def grade_due_predictions():
    """Checks every past BUY recommendation whose horizon has elapsed
    against the real price that actually happened, and stores the
    outcome -- the self-grading track record the person asked for so the
    model's live accuracy can be trusted (or not) instead of only ever
    citing the historical backtest."""
    due = rdb.load_unevaluated_due()
    if due.empty:
        log("  nothing due for grading.")
        return

    if not PRICES_LONG_PATH.exists():
        log("  prices_long.parquet not found -- can't grade yet.")
        return
    prices = pd.read_parquet(PRICES_LONG_PATH, columns=["player_id", "platform", "game_version", "date", "price"])
    prices = prices[(prices["platform"] == "pc") & (prices["game_version"] == "fc27")].copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices = prices.dropna(subset=["price"])
    prices = prices[prices["price"] > 0]

    graded_rows = []
    for r in due.itertuples():
        if pd.isna(r.price_at_snapshot) or r.price_at_snapshot <= 0:
            continue
        eval_date = pd.Timestamp(r.eval_date)
        window = prices[
            (prices["player_id"] == r.player_id) &
            (prices["date"] >= eval_date - pd.Timedelta(days=GRADE_MATCH_TOLERANCE_DAYS)) &
            (prices["date"] <= eval_date + pd.Timedelta(days=GRADE_MATCH_TOLERANCE_DAYS))
        ]
        if window.empty:
            continue
        dist = (window["date"] - eval_date).abs()
        actual_price = float(window.loc[dist.idxmin(), "price"])
        actual_return = (actual_price * (1 - EA_SELL_TAX) - r.price_at_snapshot) / r.price_at_snapshot
        graded_rows.append({
            "id": r.id, "actual_price_at_eval": actual_price, "actual_return": float(actual_return),
            "actual_win": int(actual_return > WIN_BREAKEVEN),
            "evaluated_at": pd.Timestamp.now().isoformat(),
        })

    if graded_rows:
        rdb.mark_evaluated(pd.DataFrame(graded_rows))
        log(f"  graded {len(graded_rows)} of {len(due)} due recommendations "
            f"({len(due) - len(graded_rows)} still lack a nearby real price point)")
    else:
        log(f"  {len(due)} recommendations due for grading, but none had a nearby price point yet")


# ------------------------------------------------------------------ main

def _stringify_dates(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c]).dt.strftime("%Y-%m-%d")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bankroll", type=float, default=None,
                     help="Coins available to invest right now. Saved for next time; omit to reuse "
                          "the last value (defaults to 15000 the very first run).")
    args = ap.parse_args()

    if args.bankroll is not None:
        rdb.set_bankroll(args.bankroll)
    bankroll = rdb.get_bankroll()
    log(f"Bankroll: {bankroll:,.0f} coins")

    log("Loading models/latest ...")
    meta, models = load_models()
    log(f"  version={meta['version']}, horizons={meta['horizons']}")
    loss_mags = loss_magnitudes(meta)

    log("Loading candidate panel (PC, FC27, rating>=75)...")
    panel, feature_cols = load_candidate_panel(meta)
    log(f"  {len(panel)} rows, {panel['player_id'].nunique()} distinct players")
    if panel.empty:
        log("No PC/FC27 Gold+ price history yet -- nothing to score.")
        return

    blacklist = load_blacklist()
    before = panel["player_id"].nunique()
    panel = apply_blacklist(panel, blacklist)
    log(f"  blacklist check: {before - panel['player_id'].nunique()} player(s) excluded (no-market)")

    history_counts = panel.groupby("player_id").size().rename("n_history_points")
    latest = latest_snapshot(panel)
    latest = latest.merge(history_counts, on="player_id", how="left")
    latest = attach_urls(latest)

    log("Estimating liquidity (days-to-sell) from real sales history...")
    latest = attach_liquidity(latest)

    log("Scoring BUY candidates across all horizons...")
    buy = build_buy_candidates(latest, feature_cols, models, meta)
    log(f"  {len(buy)} candidates clear the {CONFIDENCE_BAR:.0%} confidence bar")

    buy = size_positions(buy, loss_mags, bankroll)
    buy = attach_rationale(buy)

    scored_at = pd.Timestamp.now().isoformat()
    if not buy.empty:
        buy["scored_at"] = scored_at
        buy["action"] = "BUY"
        buy["holding_id"] = None
        buy["unrealized_return_pct"] = None
        buy = _stringify_dates(buy, ["snapshot_date", "eval_date"])
        n_ins = rdb.insert_recommendations(buy)
        log(f"  inserted {n_ins} new BUY recommendations (repeats of an already-scored snapshot "
            f"are skipped)")

    log("Re-scoring open holdings...")
    holdings = rdb.list_holdings(open_only=True)
    if holdings.empty:
        log("  no open holdings.")
    else:
        sell_all = build_sell_list(holdings, latest, feature_cols, models, meta)
        if sell_all.empty:
            log("  none of the held players had current PC/FC27 price data to re-score.")
        else:
            status = pd.DataFrame({
                "id": sell_all["id"],
                "last_scored_at": scored_at,
                "last_price": sell_all["price_at_snapshot"],
                "last_win_prob": sell_all["predicted_win_prob"],
                "last_horizon_days": sell_all["horizon_days"],
                "last_predicted_return_pct": sell_all["predicted_return_pct"],
                "last_unrealized_return_pct": sell_all["unrealized_return_pct"],
                "last_sell_signal": (sell_all["predicted_win_prob"] < SELL_WIN_PROB_THRESHOLD).astype(int),
            })
            rdb.update_holdings_status(status)

            sell_flagged = sell_all[sell_all["predicted_win_prob"] < SELL_WIN_PROB_THRESHOLD].copy()
            log(f"  {len(sell_flagged)} of {len(sell_all)} open holding(s) flagged SELL "
                f"(best remaining win probability < {SELL_WIN_PROB_THRESHOLD:.0%})")
            if not sell_flagged.empty:
                sell_flagged["scored_at"] = scored_at
                sell_flagged["action"] = "SELL"
                sell_flagged["holding_id"] = sell_flagged["id"]
                sell_flagged["confidence_flag"] = None
                sell_flagged["kelly_fraction"] = None
                sell_flagged["recommended_coins"] = None
                sell_flagged["recommended_quantity"] = None
                sell_flagged["rationale"] = None
                sell_flagged["est_days_to_sell"] = None
                sell_flagged["effective_daily_return"] = None
                sell_flagged = _stringify_dates(sell_flagged, ["snapshot_date", "eval_date"])
                n_sell = rdb.insert_recommendations(sell_flagged)
                log(f"  inserted {n_sell} new SELL recommendations")

    log("Grading past BUY recommendations whose horizon has elapsed...")
    grade_due_predictions()

    log("Done.")


if __name__ == "__main__":
    main()
