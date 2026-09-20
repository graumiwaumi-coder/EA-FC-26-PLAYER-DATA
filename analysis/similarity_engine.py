"""
Task A: player similarity / pairs-trading engine, based on individual stats --
NOT club/league/nation popularity (per the explicit steer: "I don't think club,
league matters all that much its about the players individual stats").

Two stages:

  1. Find each Gold+ player's K nearest statistical neighbors -- same
     position_group and a tight rating window (hard gates, since a 65-rated CB
     and a 90-rated CB trading at wildly different prices aren't "similar" for
     trading purposes no matter how alike their playstyles are), then ranked
     by a weighted distance over skills/weak_foot/height/age/foot/body_type
     plus playstyle overlap (Jaccard).

     v1 note: ideally this would include PAC/SHO/PAS/DRI/DEF/PHY (the six core
     attribute stats) -- we never scraped those, only rating/position/skills/
     weak_foot/height/age/body_type/playstyles. Explicit call made to ship v1
     with what we have rather than block Task A on a large re-scrape; revisit
     if this proves too weak a proxy for real playing-style similarity.

  2. The actual pairs-trading signal: for every (player, platform, date), take
     that player's K neighbors and compute how much the player's own recent
     price move has DIVERGED from its peer group's average move on the same
     date. A player that's dropped 12% while its 15 closest statistical peers
     only dropped 2% is a classic pairs-trading mean-reversion candidate --
     IF the market here actually mean-reverts. That's tested empirically
     below rather than assumed: does divergence correlate with, and predict,
     forward returns?

Run: python3 similarity_engine.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

MIN_RATING = 75
RATING_WINDOW = 3      # neighbors must be within +/- this many rating points
K_NEIGHBORS = 15
DIVERGENCE_WINDOW = "pct_change_7d"   # which return column defines "recent move"
FORWARD_TARGET = "fwd_return_21d_net_tax"

NUMERIC_SIM_FEATURES = ["rating", "skills", "weak_foot", "height_cm", "age"]
PLAYSTYLE_WEIGHT = 1.5   # relative weight of playstyle-overlap vs numeric distance


def load_players():
    df = pd.read_parquet(DATA_DIR / "players_features.parquet")
    df = df[df["rating"] >= MIN_RATING].copy()
    df["playstyle_set"] = df["playstyles"].fillna("").apply(
        lambda s: frozenset(s.split("|")) if s else frozenset()
    )
    for c in NUMERIC_SIM_FEATURES:
        df[c] = df[c].astype(float)
    return df.reset_index(drop=True)


def pairwise_euclidean(features):
    """||a-b|| via the dot-product identity, avoiding an (n, n, d) intermediate
    array -- for a few thousand players that 3D array would run into the
    hundreds of MB to low GB, all to compute something an (n, n) matrix does."""
    sq_norms = np.sum(features ** 2, axis=1)
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * (features @ features.T)
    return np.sqrt(np.maximum(dist_sq, 0))


def pairwise_playstyle_distance(playstyle_sets):
    """1 - Jaccard similarity, vectorized as a binary-matrix dot product instead
    of a pure-Python nested loop over frozensets (which would be O(n^2) Python-level
    set operations -- fine at a few hundred players, slow at a few thousand)."""
    all_styles = sorted(set().union(*playstyle_sets)) if playstyle_sets else []
    if not all_styles:
        n = len(playstyle_sets)
        return np.zeros((n, n))
    style_idx = {s: i for i, s in enumerate(all_styles)}
    n = len(playstyle_sets)
    binary = np.zeros((n, len(all_styles)), dtype=np.float32)
    for i, styles in enumerate(playstyle_sets):
        for s in styles:
            binary[i, style_idx[s]] = 1.0

    intersection = binary @ binary.T
    row_sums = binary.sum(axis=1)
    union = row_sums[:, None] + row_sums[None, :] - intersection
    with np.errstate(invalid="ignore", divide="ignore"):
        jaccard = np.where(union > 0, intersection / union, 0.0)
    return 1 - jaccard


def find_neighbors(players):
    """For every player, the K nearest players within the same position_group
    and rating window, ranked by weighted numeric distance + playstyle overlap."""
    # standardize numeric features once, globally, so distances are comparable.
    # Fill missing values (e.g. ~450 players have no recorded age) with the
    # column median rather than leaving NaN, which would silently poison every
    # distance calculation involving that player.
    stats = players[NUMERIC_SIM_FEATURES].copy()
    stats = stats.fillna(stats.median())
    means, stds = stats.mean(), stats.std().replace(0, 1)
    standardized = (stats - means) / stds

    body_dummies = pd.get_dummies(players["body_type"].fillna("unknown"), prefix="body")
    foot_dummies = pd.get_dummies(players["foot"].fillna("unknown"), prefix="foot")
    # .to_numpy() on a mix of float64 + bool columns returns dtype=object, not a
    # numeric dtype -- force float32 explicitly or downstream matrix math breaks.
    feature_matrix = pd.concat([standardized, body_dummies, foot_dummies], axis=1).to_numpy(dtype=np.float32)

    ids = players["id"].to_numpy()
    ratings = players["rating"].to_numpy()
    groups = players["position_group"].to_numpy()
    playstyle_sets = players["playstyle_set"].tolist()

    neighbor_rows = []
    for group in np.unique(groups):
        idx = np.where(groups == group)[0]
        sub_features = feature_matrix[idx]
        sub_ratings = ratings[idx]
        sub_ids = ids[idx]
        sub_playstyles = [playstyle_sets[i] for i in idx]

        dist = pairwise_euclidean(sub_features)
        dist += PLAYSTYLE_WEIGHT * pairwise_playstyle_distance(sub_playstyles)

        rating_diff = np.abs(sub_ratings[:, None] - sub_ratings[None, :])
        out_of_window = rating_diff > RATING_WINDOW
        dist[out_of_window] = np.inf
        np.fill_diagonal(dist, np.inf)

        # argsort once per row, take the first K non-inf entries
        order = np.argsort(dist, axis=1)[:, :K_NEIGHBORS]
        for i in range(len(idx)):
            for rank, j in enumerate(order[i]):
                if dist[i, j] == np.inf:
                    break
                neighbor_rows.append((sub_ids[i], sub_ids[j], rank + 1, dist[i, j]))

    return pd.DataFrame(neighbor_rows, columns=["player_id", "neighbor_id", "rank", "distance"])


def compute_peer_divergence(neighbors):
    """For every (player, platform, date) in the price panel, the gap between
    the player's own recent return and its K neighbors' average recent return
    on that same date."""
    price_cols = ["player_id", "platform", "date", "rating", DIVERGENCE_WINDOW, FORWARD_TARGET]
    dataset = ds.dataset(DATA_DIR / "prices_features.parquet", format="parquet")
    table = dataset.to_table(columns=price_cols, filter=ds.field("rating") >= MIN_RATING)
    prices = table.to_pandas(split_blocks=True, self_destruct=True)
    del table
    float_cols = prices.select_dtypes(include=["float64"]).columns
    prices[float_cols] = prices[float_cols].astype("float32")

    own = prices[["player_id", "platform", "date", DIVERGENCE_WINDOW, FORWARD_TARGET]].rename(
        columns={DIVERGENCE_WINDOW: "own_return"}
    )

    neighbor_returns = neighbors.merge(
        prices[["player_id", "platform", "date", DIVERGENCE_WINDOW]].rename(
            columns={"player_id": "neighbor_id", DIVERGENCE_WINDOW: "neighbor_return"}
        ),
        on="neighbor_id", how="inner",
    )
    peer_avg = (
        neighbor_returns.groupby(["player_id", "platform", "date"])["neighbor_return"]
        .agg(peer_avg_return="mean", n_neighbors_with_data="count")
        .reset_index()
    )

    merged = own.merge(peer_avg, on=["player_id", "platform", "date"], how="inner")
    merged["peer_divergence"] = merged["own_return"] - merged["peer_avg_return"]
    return merged.dropna(subset=["peer_divergence", FORWARD_TARGET])


def validate(merged):
    """Does peer_divergence actually predict forward returns? A mean-reversion
    market would show a NEGATIVE relationship (diverged down -> reverts up), a
    momentum market a POSITIVE one. Don't assume -- check."""
    corr = merged["peer_divergence"].corr(merged[FORWARD_TARGET])
    print(f"\nCorrelation(peer_divergence, {FORWARD_TARGET}): {corr:.4f}  "
          f"({'mean-reversion' if corr < 0 else 'momentum'} signal, n={len(merged)})")

    merged["divergence_decile"] = pd.qcut(merged["peer_divergence"], 10, labels=False, duplicates="drop")
    by_decile = merged.groupby("divergence_decile")[FORWARD_TARGET].agg(["mean", "median", "count"])
    by_decile.columns = ["fwd_return_mean", "fwd_return_median", "n"]
    print("\nForward return by peer-divergence decile (0=most underperformed peers, 9=most outperformed):")
    print(by_decile)


def main():
    print("Loading Gold+ players...")
    players = load_players()
    print(f"{len(players)} players")

    print(f"\nFinding {K_NEIGHBORS} nearest neighbors per player "
          f"(same position_group, rating +/-{RATING_WINDOW})...")
    neighbors = find_neighbors(players)
    print(f"{len(neighbors)} neighbor pairs found for {neighbors['player_id'].nunique()} players")
    avg_k = neighbors.groupby("player_id").size().mean()
    print(f"Average neighbors found per player: {avg_k:.1f} (target {K_NEIGHBORS})")
    neighbors.to_parquet(DATA_DIR / "player_neighbors.parquet", index=False)

    print("\nComputing peer-group divergence against the price panel...")
    merged = compute_peer_divergence(neighbors)
    print(f"{len(merged)} (player, platform, date) rows with a valid peer_divergence")
    merged.to_parquet(DATA_DIR / "peer_divergence.parquet", index=False)

    validate(merged)


if __name__ == "__main__":
    main()
