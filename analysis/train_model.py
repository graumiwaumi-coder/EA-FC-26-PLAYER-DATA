"""
Phase 2: train a real model (gradient-boosted trees) on the engineered features.

Predicts, for a Gold+ (rating>=75) player on a given day, what happens over the next
21 days (the middle of your stated 2-6 week long-term window):
  - classifier: will holding be profitable after EA's 5% sell tax? (yes/no + confidence)
  - regressor: what % return should we expect?

Uses scikit-learn's HistGradientBoosting models (native categorical + missing-value
support, no extra dependency beyond sklearn, memory-efficient histogram algorithm --
the same family of algorithm as LightGBM/XGBoost).

Validation is rolling-window walk-forward (4 folds, same layout as
tune_hyperparameters.py --full): each fold trains only on the 200 days immediately
before its test period, not an expanding window of everything seen so far, with a
21-day embargo gap so no training label's forward-return window overlaps the test
period. The deployed model that actually gets saved is then trained on the most
recent 200-day window, which is what it would look like in production -- evaluated
on all 4 folds to know how good the approach is, but not what gets shipped.

Run: python3 train_model.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

HORIZON = 21
MIN_RATING = 75
EMBARGO_DAYS = HORIZON
TRAIN_WINDOW_DAYS = 200  # rolling window, not expanding -- see make_folds()

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
CATEGORICAL_FEATURES = ["platform", "position", "position_group", "league", "nation",
                         "foot", "body_type", "is_icon"]
# The finance-inspired features from build_features_v2.py -- these were part of the
# 82-feature set tune_hyperparameters.py --full actually tuned against, so they need
# to be here too or the tuned hyperparameters are being applied to a different
# (smaller) feature set than the one they were tuned for.
NEW_FEATURES = ["days_since_release", "trend_slope_14d", "trend_slope_30d", "ma_crossover",
                "days_since_crossover", "cross_sectional_rank", "beta_60d", "autocorr_30d",
                "relative_strength"]
# Task A -- similarity_engine.py's pairs-trading signal: how far a player's recent
# return has diverged from its K nearest statistical peers' average recent return.
SIMILARITY_FEATURES = ["peer_divergence", "n_neighbors_with_data"]
# Task B's promo_cluster/promo_cluster_confidence (see promo_trajectory_clustering.py)
# were tried here and reverted: averaged AUC was flat but the deployed model's
# high-confidence predictions got meaningfully less reliable (prob>=0.8 win rate
# dropped from 83.6% to 46.1%) -- likely because the cluster label is assigned once,
# early in a promo card's life, and goes stale by the time later test periods roll
# around. The clustering + early-shape-prediction analysis itself is still valid and
# kept in promo_trajectory_clustering.py; it just isn't fed into this model.
ALL_FEATURES = NUMERIC_FEATURES + NEW_FEATURES + SIMILARITY_FEATURES + CATEGORICAL_FEATURES


def load_data():
    players = pd.read_parquet(DATA_DIR / "players_features.parquet")
    players = players[players["rating"] >= MIN_RATING]
    player_cols = ["id", "position", "position_group", "league", "nation", "foot", "body_type",
                   "is_icon", "skills", "weak_foot", "height_cm", "age", "n_playstyles",
                   "league_price_score", "league_liquidity_score", "club_price_score",
                   "club_liquidity_score", "nation_price_score", "nation_liquidity_score"]
    players = players[player_cols]

    import pyarrow.dataset as ds
    price_cols = ["player_id", "platform", "date", "rating",
                  "ma_3", "ma_7", "ma_14", "ma_30", "ma_60", "ma_90",
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
                  f"fwd_return_{HORIZON}d_net_tax", f"fwd_up_{HORIZON}d_net_tax"]

    # filter at the Arrow level (rating >= MIN_RATING) BEFORE materializing a pandas
    # frame -- filtering after a full pandas load is what crashed this on 16.8M rows.
    dataset = ds.dataset(DATA_DIR / "prices_features.parquet", format="parquet")
    table = dataset.to_table(columns=price_cols, filter=ds.field("rating") >= MIN_RATING)
    prices = table.to_pandas(split_blocks=True, self_destruct=True)
    del table

    v2 = pd.read_parquet(DATA_DIR / "prices_features_v2.parquet",
                          columns=["player_id", "platform", "date"] + NEW_FEATURES)
    prices = prices.merge(v2, on=["player_id", "platform", "date"], how="left")

    sim = pd.read_parquet(DATA_DIR / "peer_divergence.parquet",
                           columns=["player_id", "platform", "date"] + SIMILARITY_FEATURES)
    prices = prices.merge(sim, on=["player_id", "platform", "date"], how="left")

    # downcast to float32 to keep memory manageable
    float_cols = prices.select_dtypes(include=["float64"]).columns
    prices[float_cols] = prices[float_cols].astype("float32")

    prices = prices.merge(players, left_on="player_id", right_on="id", how="left").drop(columns=["id"])
    for c in CATEGORICAL_FEATURES:
        prices[c] = prices[c].astype("category")

    return prices


def make_folds(df, fold_edges):
    """Rolling-window walk-forward folds -- NOT an expanding window. Each fold trains
    only on the TRAIN_WINDOW_DAYS immediately before its test period (with an embargo
    gap so no training label's forward-return window overlaps the test period), the
    same approach already used by tune_hyperparameters.py and the Colab notebook.
    An expanding window (train on everything from the start) was the original design
    here but was never brought in line with that fix -- it trains on stale,
    increasingly-irrelevant early-season data and a single split can land entirely in
    an unusually hard period, which is what was producing misleadingly low holdout
    scores."""
    min_date, max_date = df["date"].min(), df["date"].max()
    total_days = (max_date - min_date).days
    folds = []
    for i in range(len(fold_edges) - 1):
        test_start = min_date + pd.Timedelta(days=int(total_days * fold_edges[i]))
        test_end = min_date + pd.Timedelta(days=int(total_days * fold_edges[i + 1]))
        train_end = test_start - pd.Timedelta(days=EMBARGO_DAYS)
        train_start = train_end - pd.Timedelta(days=TRAIN_WINDOW_DAYS)
        train_mask = (df["date"] > train_start) & (df["date"] <= train_end)
        test_mask = (df["date"] >= test_start) & (df["date"] < test_end)
        if train_mask.sum() < 1000 or test_mask.sum() < 200:
            continue
        folds.append((train_mask, test_mask))
    return folds


def make_clf(cat_idx):
    # Hyperparameters below come from tune_hyperparameters.py --full: a 40-combo x
    # 4-fold walk-forward search, best result AUC=0.768 (vs ~0.75-0.76 with the
    # previous guessed defaults). Only the classifier was tuned directly (the
    # search optimized target_clf); the same combo is applied to the regressor
    # too since a dedicated regression tuning pass hasn't been run yet.
    return HistGradientBoostingClassifier(
        categorical_features=cat_idx, max_iter=300, learning_rate=0.02,
        max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0, max_bins=255,
        early_stopping=True, validation_fraction=0.15, random_state=42,
    )


def make_reg(cat_idx):
    return HistGradientBoostingRegressor(
        categorical_features=cat_idx, max_iter=300, learning_rate=0.02,
        max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0, max_bins=255,
        early_stopping=True, validation_fraction=0.15, random_state=42,
    )


def main():
    print("Loading and merging data (Gold+ only, memory-safe column selection)...")
    df = load_data()
    print(f"Loaded {len(df)} rows for {df['player_id'].nunique()} players")

    target_reg = f"fwd_return_{HORIZON}d_net_tax"
    target_clf = f"fwd_up_{HORIZON}d_net_tax"
    df = df.dropna(subset=[target_reg, target_clf])
    print(f"Rows with valid {HORIZON}-day forward target: {len(df)}")

    cat_idx = [ALL_FEATURES.index(c) for c in CATEGORICAL_FEATURES]

    # Same fold layout as tune_hyperparameters.py --full, so this evaluation is
    # directly comparable to the AUC=0.768 the hyperparameters were chosen against.
    fold_edges = [0.55, 0.65, 0.75, 0.85, 0.95]
    folds = make_folds(df, fold_edges)
    print(f"\n=== Rolling-window walk-forward evaluation: {len(folds)} folds ===")

    aucs, win_rates_50 = [], []
    for i, (train_mask, test_mask) in enumerate(folds):
        X_train = df.loc[train_mask, ALL_FEATURES]
        X_test = df.loc[test_mask, ALL_FEATURES]
        y_train_clf = df.loc[train_mask, target_clf].astype(int)
        y_test_clf = df.loc[test_mask, target_clf].astype(int)

        clf_fold = make_clf(cat_idx)
        clf_fold.fit(X_train, y_train_clf)
        proba = clf_fold.predict_proba(X_test)[:, 1]
        fold_auc = roc_auc_score(y_test_clf, proba)
        mask_conf = proba >= 0.5
        fold_win_rate = y_test_clf.to_numpy()[mask_conf].mean() if mask_conf.sum() > 0 else np.nan
        aucs.append(fold_auc)
        win_rates_50.append(fold_win_rate)
        print(f"  Fold {i+1}/{len(folds)}: test dates {df.loc[test_mask,'date'].min().date()} -> "
              f"{df.loc[test_mask,'date'].max().date()}, AUC={fold_auc:.3f}, win_rate@0.5={fold_win_rate:.1%}")

    print(f"\nAveraged across folds: AUC={np.mean(aucs):.3f} (std={np.std(aucs):.3f}), "
          f"win_rate@0.5={np.nanmean(win_rates_50):.1%}")

    # The evaluation above tells us how good the SYSTEM is; the deployed model itself
    # should be trained on the most recent window so it reflects the current game
    # economy, not stale data from a year ago -- same train window as the last fold.
    last_train_mask, _ = folds[-1]
    print(f"\n=== Training final deployed model on the most recent {TRAIN_WINDOW_DAYS}-day window "
          f"({last_train_mask.sum()} rows) ===")
    X_train = df.loc[last_train_mask, ALL_FEATURES]
    y_train_clf = df.loc[last_train_mask, target_clf].astype(int)
    y_train_reg = df.loc[last_train_mask, target_reg]

    clf = make_clf(cat_idx)
    clf.fit(X_train, y_train_clf)
    reg = make_reg(cat_idx)
    reg.fit(X_train, y_train_reg)

    # held-out sanity check for the deployed model: the last fold's test period,
    # which it never trained on
    _, last_test_mask = folds[-1]
    X_test = df.loc[last_test_mask, ALL_FEATURES]
    y_test_clf = df.loc[last_test_mask, target_clf].astype(int)
    y_test_reg = df.loc[last_test_mask, target_reg]
    proba_test = clf.predict_proba(X_test)[:, 1]
    pred_reg = reg.predict(X_test)
    pred_test = (proba_test >= 0.5).astype(int)
    print(f"Deployed model holdout AUC (most recent fold's test period): "
          f"{roc_auc_score(y_test_clf, proba_test):.3f}")
    print(f"Deployed model holdout accuracy @0.5 threshold: {accuracy_score(y_test_clf, pred_test):.3f} "
          f"(baseline/majority-class accuracy: {max(y_test_clf.mean(), 1 - y_test_clf.mean()):.3f})")
    print(f"Deployed model regressor MAE: {mean_absolute_error(y_test_reg, pred_reg):.3f}, "
          f"correlation(predicted, actual): {np.corrcoef(pred_reg, y_test_reg)[0,1]:.3f}")

    for thresh in (0.5, 0.6, 0.7, 0.8):
        mask = proba_test >= thresh
        n = mask.sum()
        if n == 0:
            continue
        actual_win_rate = y_test_clf.to_numpy()[mask].mean()
        avg_return = y_test_reg.to_numpy()[mask].mean()
        med_return = np.median(y_test_reg.to_numpy()[mask])
        print(f"  Predicted prob>={thresh}: n={n} ({n/len(proba_test):.1%} of test), "
              f"actual win rate={actual_win_rate:.1%}, mean net return={avg_return:.1%}, median net return={med_return:.1%}")

    # top-decile check: if we only acted on the highest-predicted-return players, how did they do?
    order = np.argsort(-pred_reg)
    top_decile = order[: len(order)//10]
    print(f"Top decile by predicted return: n={len(top_decile)}, "
          f"actual mean return={y_test_reg.to_numpy()[top_decile].mean():.1%}, "
          f"actual median return={np.median(y_test_reg.to_numpy()[top_decile]):.1%}, "
          f"win rate={ (y_test_reg.to_numpy()[top_decile] > 0).mean():.1%}")
    bottom_decile = order[-len(order)//10:]
    print(f"Bottom decile by predicted return: n={len(bottom_decile)}, "
          f"actual mean return={y_test_reg.to_numpy()[bottom_decile].mean():.1%}, "
          f"actual median return={np.median(y_test_reg.to_numpy()[bottom_decile]):.1%}, "
          f"win rate={ (y_test_reg.to_numpy()[bottom_decile] > 0).mean():.1%}")

    print("\n=== Feature importance (permutation, classifier, sampled for speed) ===")
    from sklearn.inspection import permutation_importance
    sample_idx = np.random.RandomState(42).choice(len(X_test), size=min(20000, len(X_test)), replace=False)
    perm = permutation_importance(clf, X_test.iloc[sample_idx], y_test_clf.iloc[sample_idx],
                                   n_repeats=3, random_state=42, n_jobs=-1)
    importances = pd.Series(perm.importances_mean, index=ALL_FEATURES).sort_values(ascending=False)
    print(importances.head(20))

    joblib.dump(clf, MODEL_DIR / f"clf_{HORIZON}d.joblib")
    joblib.dump(reg, MODEL_DIR / f"reg_{HORIZON}d.joblib")
    joblib.dump(ALL_FEATURES, MODEL_DIR / "feature_list.joblib")
    joblib.dump(cat_idx, MODEL_DIR / "cat_idx.joblib")
    print("\nSaved model artifacts to data/models/")


if __name__ == "__main__":
    main()
