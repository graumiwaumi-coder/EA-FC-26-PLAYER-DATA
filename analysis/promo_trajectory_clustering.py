"""
Task B: promo/release trajectory clustering.

FC26's price panel has 2,635 genuinely promo-versioned cards (squad set and
different from club -- i.e. not the club-fallback case), covering ~60+ distinct
squads (TOTW1-N, FUTBirthday, Thunderstruck, PathToGlory, Futties, GreatsOfTheGame,
WinterWildcards, FutureStars, ...), every one with full day-by-day price coverage
from its release date (confirmed: all 5,270 card x platform trajectories have
>=25/31 days of coverage in the first month, day-0 price is never 0).

Two stages:

  1. Cluster the FIRST 30 DAYS of each promo card's price trajectory (normalized
     to day-0 = 0 on a log scale, so a 10k->50k card and a 1M->5M card with the
     same shape land in the same cluster) into archetypal patterns -- e.g. quick
     pump-then-dump vs. steady climb vs. flat/no interest vs. slow decline.

  2. The actually useful part: can we tell which archetype a BRAND NEW release
     is heading toward using only its first few days, before the full 30-day
     shape is known? Train a small classifier on days 0-5 (+ rating/position)
     to predict full-trajectory cluster membership, and report how well that
     actually works -- the whole point of clustering historical releases is to
     recognize a new one's pattern early enough to act on it.

Run: python3 promo_trajectory_clustering.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

TRAJECTORY_DAYS = 30       # cluster on days 0..30
EARLY_DAYS = 5             # "new release" simulation: only days 0..5 known
N_CLUSTERS = 5


def load_promo_trajectories():
    players = pd.read_parquet(DATA_DIR / "players_features.parquet")
    promo = players[(players["squad"].notna()) & (players["squad"] != players["club"])]
    promo_ids = promo["id"].unique().tolist()
    print(f"{len(promo_ids)} distinct promo-squad player ids")

    dataset = ds.dataset(DATA_DIR / "prices_features.parquet", format="parquet")
    table = dataset.to_table(
        columns=["player_id", "platform", "date", "price_clean", "rating"],
        filter=ds.field("player_id").isin(promo_ids),
    )
    df = table.to_pandas(split_blocks=True, self_destruct=True)
    del table

    release = df.groupby("player_id")["date"].min().rename("release_date")
    df = df.merge(release, on="player_id")
    df["day"] = (df["date"] - df["release_date"]).dt.days
    df = df[df["day"].between(0, TRAJECTORY_DAYS)]

    meta_cols = ["id", "squad", "position", "position_group", "rating"]
    df = df.merge(promo[meta_cols].rename(columns={"id": "player_id", "rating": "rating_meta"}),
                  on="player_id", how="left")
    return df


def build_trajectory_matrix(df):
    """(player_id, platform) rows x day(0..30) columns, log-ratio to day 0."""
    day0 = df[df["day"] == 0][["player_id", "platform", "price_clean"]].rename(
        columns={"price_clean": "day0_price"}
    )
    df = df.merge(day0, on=["player_id", "platform"], how="inner")
    df = df[df["day0_price"] > 0]
    df["log_ratio"] = np.log(df["price_clean"].clip(lower=1) / df["day0_price"])

    wide = df.pivot_table(index=["player_id", "platform"], columns="day", values="log_ratio")
    wide = wide.reindex(columns=range(0, TRAJECTORY_DAYS + 1))
    # forward/back-fill the rare gap -- coverage was already confirmed near-complete
    wide = wide.ffill(axis=1).bfill(axis=1)
    wide = wide.dropna()
    return wide


def cluster_trajectories(wide):
    X = wide.to_numpy()
    print(f"\nClustering {len(X)} trajectories into {N_CLUSTERS} archetypes...")
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    print(f"Silhouette score: {sil:.3f} (higher = more distinct clusters, "
          f"0.1-0.3 is typical/reasonable for real-world trajectory shapes, not tight blobs)")
    return labels, km


def describe_clusters(wide, labels, meta):
    result = wide.copy()
    result["cluster"] = labels
    result = result.reset_index().merge(meta, on=["player_id", "platform"], how="left")

    print("\n=== Cluster archetypes ===")
    for c in sorted(result["cluster"].unique()):
        sub = result[result["cluster"] == c]
        day_cols = [d for d in range(0, TRAJECTORY_DAYS + 1)]
        mean_path = sub[day_cols].mean()
        peak_day = mean_path.idxmax()
        peak_val = mean_path.max()
        day30_val = mean_path[TRAJECTORY_DAYS]
        print(f"\nCluster {c}: n={len(sub)} ({len(sub)/len(result):.1%})")
        print(f"  Shape: day0=0%, peak at day {peak_day} ({(np.exp(peak_val)-1):+.1%}), "
              f"day{TRAJECTORY_DAYS}={(np.exp(day30_val)-1):+.1%}")
        print(f"  Path (every 5 days): " +
              ", ".join(f"d{d}={(np.exp(mean_path[d])-1):+.1%}" for d in range(0, TRAJECTORY_DAYS + 1, 5)))
        print(f"  Rating: mean={sub['rating_meta'].mean():.1f}, "
              f"position_group breakdown: {sub['position_group'].value_counts(normalize=True).round(2).to_dict()}")
        top_squads = sub["squad"].value_counts().head(3).to_dict()
        print(f"  Most common squads: {top_squads}")
    return result


def early_shape_classifier(result):
    """The actually useful test: using only the first EARLY_DAYS days (which is
    all you'd have for a brand-new release) plus rating/position, how well can
    we predict which of the full-30-day clusters a card is heading toward?"""
    early_cols = list(range(0, EARLY_DAYS + 1))
    feature_cols = early_cols + ["rating_meta"]
    pos_dummies = pd.get_dummies(result["position_group"], prefix="pos")
    X = pd.concat([result[feature_cols].reset_index(drop=True), pos_dummies.reset_index(drop=True)], axis=1)
    X.columns = X.columns.astype(str)
    y = result["cluster"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    baseline = y_test.value_counts(normalize=True).max()
    print(f"\n=== Early-shape classifier (days 0-{EARLY_DAYS} + rating + position only) ===")
    print(f"Test accuracy: {acc:.1%}  (baseline/always-guess-largest-cluster: {baseline:.1%})")
    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("Top predictive features:")
    print(importances.head(8))
    return clf


def main():
    print("Loading promo trajectories...")
    df = load_promo_trajectories()

    print("Building normalized trajectory matrix...")
    wide = build_trajectory_matrix(df)
    print(f"{len(wide)} usable (player_id, platform) trajectories after cleaning")

    labels, km = cluster_trajectories(wide)

    meta = df[["player_id", "platform", "squad", "position_group", "rating_meta"]].drop_duplicates(
        subset=["player_id", "platform"]
    )
    result = describe_clusters(wide, labels, meta)

    clf = early_shape_classifier(result)

    result[["player_id", "platform", "cluster"]].to_parquet(DATA_DIR / "promo_clusters.parquet", index=False)
    import joblib
    joblib.dump(clf, (DATA_DIR / "models" / "promo_early_shape_clf.joblib"))
    print("\nSaved promo_clusters.parquet and models/promo_early_shape_clf.joblib")


if __name__ == "__main__":
    main()
