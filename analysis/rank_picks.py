"""
Rank TODAY's Gold+ players by expected ABSOLUTE coin profit (not %), split into your
three budget/risk pools. Uses the trained classifier's probability, but converts it to
an expected % return using the EMPIRICALLY MEASURED de-duplicated returns per confidence
bucket (from evaluate_robustness.py) rather than the noisy regressor, since the regressor
was shown to be unreliable for ranking by magnitude.

Run: python3 rank_picks.py
"""
import numpy as np
import pandas as pd
import joblib
import pyarrow.dataset as ds

from train_model import ALL_FEATURES, MIN_RATING

DATA_DIR = "../data"

# from evaluate_robustness.py's de-duplicated (non-overlapping) holdout results --
# the most trustworthy estimate we have of REAL expected return per confidence level
BUCKET_RETURNS = [
    (0.7, 0.223),
    (0.6, 0.208),
    (0.5, 0.174),
    (0.0, -0.05),  # below 0.5 confidence: assume roughly breakeven-to-negative after tax
]

BUDGET_TIERS = [
    ("small (<=200k)", 0, 200_000),
    ("medium (200k-1M)", 200_000, 1_000_000),
    ("large (>1M)", 1_000_000, float("inf")),
]


def expected_return(prob):
    for thresh, ret in BUCKET_RETURNS:
        if prob >= thresh:
            return ret
    return -0.05


PLAYER_LEVEL_COLS = ["position", "position_group", "league", "nation", "foot", "body_type",
                     "is_icon", "skills", "weak_foot", "height_cm", "age", "n_playstyles",
                     "league_price_score", "league_liquidity_score", "club_price_score",
                     "club_liquidity_score", "nation_price_score", "nation_liquidity_score"]


def main():
    players = pd.read_parquet(f"{DATA_DIR}/players_features.parquet")
    players = players[players["rating"] >= MIN_RATING]

    dataset = ds.dataset(f"{DATA_DIR}/prices_features.parquet", format="parquet")
    latest_date = dataset.to_table(columns=["date"]).to_pandas()["date"].max()
    print(f"Using latest available date: {latest_date.date()}")

    price_native_cols = [c for c in ALL_FEATURES if c not in PLAYER_LEVEL_COLS]
    price_cols = list(dict.fromkeys(price_native_cols + ["player_id", "date", "price_clean", "platform"]))
    filt = (ds.field("rating") >= MIN_RATING) & (ds.field("date") == pd.Timestamp(latest_date)) & (ds.field("platform") == "pc")
    df = dataset.to_table(columns=price_cols, filter=filt).to_pandas()
    df = df.dropna(subset=["price_clean"])

    df = df.merge(players[["id"] + PLAYER_LEVEL_COLS], left_on="player_id", right_on="id", how="left").drop(columns=["id"])
    for c in ["platform", "position", "position_group", "league", "nation", "foot", "body_type", "is_icon"]:
        df[c] = df[c].astype("category")

    clf = joblib.load(f"{DATA_DIR}/models/clf_21d.joblib")
    df["proba"] = clf.predict_proba(df[ALL_FEATURES])[:, 1]
    df["expected_return_pct"] = df["proba"].apply(expected_return)
    df["expected_absolute_profit"] = df["price_clean"] * df["expected_return_pct"]

    meta = players[["id", "url", "club"]].rename(columns={"id": "player_id"})
    df = df.merge(meta, on="player_id")

    for label, lo, hi in BUDGET_TIERS:
        print(f"\n{'='*70}\nBUDGET TIER: {label}\n{'='*70}")
        tier = df[(df["price_clean"] > lo) & (df["price_clean"] <= hi)]
        tier = tier[tier["proba"] >= 0.5]  # only show players the model is actually confident about
        top = tier.sort_values("expected_absolute_profit", ascending=False).head(10)
        if len(top) == 0:
            print("  (no confident picks in this tier today)")
            continue
        for _, r in top.iterrows():
            print(f"  rating={int(r['rating'])} {str(r['position']):<4} {str(r['club'])[:22]:<22} "
                  f"price={r['price_clean']:>10,.0f}  prob={r['proba']:.2f}  "
                  f"est_return={r['expected_return_pct']:+.1%}  est_profit={r['expected_absolute_profit']:>+10,.0f} coins")


if __name__ == "__main__":
    main()
