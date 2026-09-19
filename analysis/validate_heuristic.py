"""
Phase 1 part 3: sanity-check the data and backtest a simple heuristic.

Heuristic under test: "price_to_index_ratio z-score" -- is a player currently priced
low/high relative to its OWN normal relationship to its rating-band index, over a
trailing 60-day window? If that's a real signal, players flagged "cheap" (very negative
z-score) should, on average, see better forward price returns than players flagged
"expensive" (very positive z-score).

This does NOT use any future information to compute the z-score (only trailing 60 days),
so comparing it to a later forward return is a legitimate backtest, not data leakage.
We also split by an early vs late period of the season to check the effect isn't just
one lucky one-off market event.

Run: python3 validate_heuristic.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

FORWARD_DAYS = 21  # ~3 weeks, inside your stated 2-6 week long-term window
MIN_RATING = 75    # your stated trading focus: gold and above


def main():
    players = pd.read_parquet(DATA_DIR / "players.parquet")
    feats = pd.read_parquet(DATA_DIR / "prices_features.parquet")

    feats = feats.merge(players[["id", "rating", "position", "club"]], left_on="player_id", right_on="id")
    feats = feats[feats["rating"] >= MIN_RATING].copy()

    # restrict to rows that actually have a band + z-score (81+ rated or icons)
    scored = feats[feats["ratio_zscore_60d"].notna()].copy()
    scored = scored.sort_values(["player_id", "platform", "date"])

    # forward return: price FORWARD_DAYS later vs price now, per player+platform
    def forward_return(g):
        g = g.set_index("date")
        future_price = g["price_clean"].reindex(g.index + pd.Timedelta(days=FORWARD_DAYS), method=None)
        return None

    # vectorized approach: shift within group using merge_asof-free shift since daily data is regular
    scored["fwd_price"] = scored.groupby(["player_id", "platform"])["price_clean"].shift(-FORWARD_DAYS)
    scored["fwd_return"] = (scored["fwd_price"] - scored["price_clean"]) / scored["price_clean"]
    # EA charges a 5% tax on coins received when you SELL -- a trade is only actually
    # profitable if the price move clears that. This is the return net of that tax.
    SELL_TAX = 0.05
    scored["fwd_return_net_of_tax"] = (scored["fwd_price"] * (1 - SELL_TAX) - scored["price_clean"]) / scored["price_clean"]

    valid = scored.dropna(subset=["fwd_return", "ratio_zscore_60d"])
    print(f"Rows usable for backtest (have both z-score and {FORWARD_DAYS}-day forward return): {len(valid)}")
    overall_mean = valid["fwd_return"].mean()
    overall_median = valid["fwd_return"].median()
    print(f"Baseline (ALL rating>=75 rows, any bucket): mean fwd return={overall_mean:.1%}, median={overall_median:.1%}")
    print("(this is the general market drift over the period -- buckets should be judged against THIS, not against zero)")

    # bucket by z-score
    def bucket(z):
        if z <= -1.5:
            return "very cheap (z<=-1.5)"
        if z <= -0.5:
            return "cheap (-1.5<z<=-0.5)"
        if z < 0.5:
            return "neutral (-0.5<z<0.5)"
        if z < 1.5:
            return "expensive (0.5<=z<1.5)"
        return "very expensive (z>=1.5)"

    valid["bucket"] = valid["ratio_zscore_60d"].apply(bucket)

    order = ["very cheap (z<=-1.5)", "cheap (-1.5<z<=-0.5)", "neutral (-0.5<z<0.5)",
             "expensive (0.5<=z<1.5)", "very expensive (z>=1.5)"]

    def summarize(df):
        g = df.groupby("bucket")
        out = pd.DataFrame({
            "count": g["fwd_return"].count(),
            "mean_return": g["fwd_return"].mean(),
            "median_return": g["fwd_return"].median(),
            "win_rate": g["fwd_return"].apply(lambda s: (s > 0).mean()),
            "mean_return_net_of_5pct_tax": g["fwd_return_net_of_tax"].mean(),
            "median_return_net_of_5pct_tax": g["fwd_return_net_of_tax"].median(),
            "beats_baseline_win_rate": g["fwd_return"].apply(lambda s: (s > overall_median).mean()),
        }).reindex(order)
        return out

    print("\n=== FULL PERIOD: forward 21-day return by z-score bucket (rating>=75) ===")
    print(f"(win_rate = % of rows with ANY positive return; beats_baseline_win_rate = % beating the {overall_median:.1%} baseline median)")
    summary = summarize(valid)
    print(summary)

    # de-duplicate overlapping daily windows: sample at most 1 row per player+platform per FORWARD_DAYS-day block
    # to approximate independent, non-overlapping episodes rather than heavily autocorrelated daily snapshots
    valid_sorted = valid.sort_values(["player_id", "platform", "date"]).copy()
    valid_sorted["day_rank"] = valid_sorted.groupby(["player_id", "platform"]).cumcount()
    non_overlapping = valid_sorted[valid_sorted["day_rank"] % FORWARD_DAYS == 0]
    print(f"\n=== Same thing, de-duplicated to ~non-overlapping {FORWARD_DAYS}-day episodes (n={len(non_overlapping)}) ===")
    print(summarize(non_overlapping)[["count", "mean_return", "median_return", "win_rate"]])

    # split into early vs late half of the season to check consistency
    median_date = valid["date"].median()
    early = valid[valid["date"] < median_date]
    late = valid[valid["date"] >= median_date]

    print(f"\n=== EARLY period (before {median_date.date()}) ===")
    print(early.groupby("bucket")["fwd_return"].agg(["count", "mean", "median"]).reindex(order))

    print(f"\n=== LATE period (from {median_date.date()}) ===")
    print(late.groupby("bucket")["fwd_return"].agg(["count", "mean", "median"]).reindex(order))

    # concrete current examples: latest date available, most extreme z-scores
    latest_date = feats["date"].max()
    latest = feats[(feats["date"] == latest_date) & feats["ratio_zscore_60d"].notna()].copy()
    latest = latest.merge(players[["id", "url"]], left_on="player_id", right_on="id", suffixes=("", "_p"))

    print(f"\n=== Latest date in data: {latest_date.date()} ===")
    print("\nTop 10 'cheapest vs own history' (most negative z-score), rating>=75, PC:")
    cheap = latest[latest["platform"] == "pc"].nsmallest(10, "ratio_zscore_60d")
    print(cheap[["player_id", "rating", "position", "club", "price_clean", "band", "band_index_value", "ratio_zscore_60d"]].to_string(index=False))

    print("\nTop 10 'most expensive vs own history' (most positive z-score), rating>=75, PC:")
    expensive = latest[latest["platform"] == "pc"].nlargest(10, "ratio_zscore_60d")
    print(expensive[["player_id", "rating", "position", "club", "price_clean", "band", "band_index_value", "ratio_zscore_60d"]].to_string(index=False))


if __name__ == "__main__":
    main()
