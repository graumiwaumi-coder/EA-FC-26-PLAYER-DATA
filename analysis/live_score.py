"""
Step 3 of applying the trained model to live FC27 data: score today's FC27 market
through the model trained on the mature FC26 season (clf_21d.joblib / reg_21d.joblib).

Deliberately NOT a model retrained on FC27 data -- FC27's season just started and there
isn't enough history yet to train anything on it directly. Instead this applies the
FC26-trained model to FC27's current feature snapshot, on the working assumption that
price DYNAMICS (how technical/momentum features relate to forward returns) generalize
across game versions even though the absolute economy (price levels, specific meta)
differs. That's a real assumption, not a guarantee -- sanity-checking this output against
intuition (does it single out cards that make sense, does it avoid obviously bad picks)
is the next step, not something this script itself can validate.

Scores each currently-tradeable FC27 (player, platform)'s MOST RECENT snapshot only --
this is "what does the model think right now", not a backtest. Many young cards will
have NaN in several rolling-window features (not enough price history yet for a 30d/60d
window) -- HistGradientBoosting handles that natively as a missing value, same as it
does during training; expect lower-confidence predictions for very recently released
cards, not crashes.

Run: python3 live_score.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import pyarrow.dataset as ds

from train_model import (NUMERIC_FEATURES, NEW_FEATURES, SIMILARITY_FEATURES, HORIZON, MIN_RATING)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = DATA_DIR / "models"

PLAYER_LEVEL_NUMERIC = ["skills", "weak_foot", "height_cm", "age", "n_playstyles",
                         "league_price_score", "league_liquidity_score", "club_price_score",
                         "club_liquidity_score", "nation_price_score", "nation_liquidity_score"]


def load_latest_fc27_snapshot():
    """One row per (player_id, platform): whichever date is most recent for that pair."""
    price_native = [c for c in NUMERIC_FEATURES if c not in PLAYER_LEVEL_NUMERIC]
    price_cols = list(dict.fromkeys(price_native + ["player_id", "platform", "date", "rating"]))

    dataset = ds.dataset(DATA_DIR / "prices_features.parquet", format="parquet")
    if "game_version" not in dataset.schema.names:
        raise RuntimeError("prices_features.parquet has no game_version column -- rebuild the pipeline "
                            "against combined FC26+FC27 data first (build_features.py etc.)")
    row_filter = (ds.field("rating") >= MIN_RATING) & (ds.field("game_version") == "fc27")
    table = dataset.to_table(columns=price_cols, filter=row_filter)
    prices = table.to_pandas()
    if prices.empty:
        raise RuntimeError("No FC27 rows found in prices_features.parquet -- has the pipeline been "
                            "rebuilt against combined FC26+FC27 data yet?")

    latest_idx = prices.groupby(["player_id", "platform"])["date"].idxmax()
    latest = prices.loc[latest_idx].reset_index(drop=True)

    v2 = pd.read_parquet(DATA_DIR / "prices_features_v2.parquet",
                          columns=["player_id", "platform", "date"] + NEW_FEATURES)
    latest = latest.merge(v2, on=["player_id", "platform", "date"], how="left")

    sim = pd.read_parquet(DATA_DIR / "peer_divergence.parquet",
                           columns=["player_id", "platform", "date"] + SIMILARITY_FEATURES)
    latest = latest.merge(sim, on=["player_id", "platform", "date"], how="left")

    # market_volatility_regime.parquet is entirely fc26-derived (built from the fc26-only
    # indices.parquet) -- there's no FC27 equivalent yet, so this always comes back NaN for
    # every FC27 row. That's fine: HistGradientBoosting treats NaN as missing, same as every
    # other young-card feature gap this live-scoring path was always expected to have.
    regime = pd.read_parquet(DATA_DIR / "market_volatility_regime.parquet",
                              columns=["platform", "date", "rolling_vol"])
    regime = regime.rename(columns={"rolling_vol": "market_volatility_regime"})
    latest = latest.merge(regime, on=["platform", "date"], how="left")

    players = pd.read_parquet(DATA_DIR / "players_features.parquet")
    players = players[players["game_version"] == "fc27"]
    player_cols = (["id", "url", "position", "position_group", "league", "club", "nation",
                     "foot", "body_type", "is_icon"] + PLAYER_LEVEL_NUMERIC)
    players = players[player_cols]
    latest = latest.merge(players, left_on="player_id", right_on="id", how="left").drop(columns=["id"])

    return latest


def score(df, clf, reg, feature_list, cat_idx):
    categorical_cols = [feature_list[i] for i in cat_idx]
    missing = [c for c in feature_list if c not in df.columns]
    if missing:
        raise RuntimeError(f"Live snapshot is missing expected model features: {missing}")

    X = df.copy()
    for c in categorical_cols:
        X[c] = X[c].astype("category")
    X = X[feature_list]

    proba = clf.predict_proba(X)[:, 1]
    pred_return = reg.predict(X)
    return proba, pred_return


def main():
    print("Loading trained FC26 model artifacts...")
    clf = joblib.load(MODEL_DIR / f"clf_{HORIZON}d.joblib")
    reg = joblib.load(MODEL_DIR / f"reg_{HORIZON}d.joblib")
    feature_list = joblib.load(MODEL_DIR / "feature_list.joblib")
    cat_idx = joblib.load(MODEL_DIR / "cat_idx.joblib")

    print("Loading latest FC27 market snapshot...")
    latest = load_latest_fc27_snapshot()
    print(f"{len(latest)} (player, platform) live FC27 rows to score "
          f"({latest['player_id'].nunique()} distinct players)")

    proba, pred_return = score(latest, clf, reg, feature_list, cat_idx)
    latest["predicted_win_prob"] = proba
    latest["predicted_return_21d"] = pred_return

    result = latest[["player_id", "url", "platform", "rating", "position", "club", "league",
                      "date", "predicted_win_prob", "predicted_return_21d"]].copy()
    result = result.sort_values("predicted_win_prob", ascending=False)

    out_path = DATA_DIR / "live_fc27_predictions.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")

    print("\n=== Top 20 by predicted win probability ===")
    print(result.head(20).to_string(index=False))

    print("\n=== Bottom 10 by predicted win probability (sanity check: should look like weak picks) ===")
    print(result.tail(10).to_string(index=False))

    print("\n=== Prediction distribution ===")
    print(result["predicted_win_prob"].describe())
    for thresh in (0.5, 0.6, 0.7, 0.8):
        n = (result["predicted_win_prob"] >= thresh).sum()
        print(f"Predicted win prob >= {thresh}: {n} / {len(result)} ({n/len(result):.1%})")

    n_missing_heavy = latest[feature_list].isna().mean(axis=1)
    print(f"\nAvg fraction of features missing per row: {n_missing_heavy.mean():.1%} "
          f"(expect this higher for very recently released cards -- short price history "
          f"means many rolling-window features can't be computed yet)")


if __name__ == "__main__":
    main()
