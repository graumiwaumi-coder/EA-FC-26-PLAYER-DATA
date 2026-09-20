"""
Phase 2: train a real model (gradient-boosted trees) on the engineered features.

Predicts, for a Gold+ (rating>=75) player on a given day, what happens over the next
21 days (the middle of your stated 2-6 week long-term window):
  - classifier: will holding be profitable after EA's 5% sell tax? (yes/no + confidence)
  - regressor: what % return should we expect?

Uses scikit-learn's HistGradientBoosting models (native categorical + missing-value
support, no extra dependency beyond sklearn, memory-efficient histogram algorithm --
the same family of algorithm as LightGBM/XGBoost).

Validation is walk-forward / out-of-time: the model is trained ONLY on the earlier
~75% of the season and tested ONLY on the later period it never saw, with a 21-day
embargo gap on both sides of the split so no training label's forward-return window
overlaps the test period. This is the same honest standard used in Phase 1's backtest.

Run: python3 train_model.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, mean_absolute_error

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

HORIZON = 21
MIN_RATING = 75
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
CATEGORICAL_FEATURES = ["platform", "position", "position_group", "league", "nation",
                         "foot", "body_type", "is_icon"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


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

    # downcast to float32 to keep memory manageable
    float_cols = prices.select_dtypes(include=["float64"]).columns
    prices[float_cols] = prices[float_cols].astype("float32")

    prices = prices.merge(players, left_on="player_id", right_on="id", how="left").drop(columns=["id"])
    for c in CATEGORICAL_FEATURES:
        prices[c] = prices[c].astype("category")

    return prices


def main():
    print("Loading and merging data (Gold+ only, memory-safe column selection)...")
    df = load_data()
    print(f"Loaded {len(df)} rows for {df['player_id'].nunique()} players")

    target_reg = f"fwd_return_{HORIZON}d_net_tax"
    target_clf = f"fwd_up_{HORIZON}d_net_tax"
    df = df.dropna(subset=[target_reg, target_clf])
    print(f"Rows with valid {HORIZON}-day forward target: {len(df)}")

    dates = df["date"]
    split_date = dates.quantile(0.75)
    train_mask = dates <= (split_date - pd.Timedelta(days=EMBARGO_DAYS))
    test_mask = dates >= (split_date + pd.Timedelta(days=EMBARGO_DAYS))
    print(f"Split date: {split_date.date()}  (embargo {EMBARGO_DAYS} days each side)")
    print(f"Train rows: {train_mask.sum()}  (dates up to {dates[train_mask].max().date()})")
    print(f"Test rows: {test_mask.sum()}  (dates from {dates[test_mask].min().date()})")

    X_train = df.loc[train_mask, ALL_FEATURES]
    X_test = df.loc[test_mask, ALL_FEATURES]
    y_train_reg = df.loc[train_mask, target_reg]
    y_test_reg = df.loc[test_mask, target_reg]
    y_train_clf = df.loc[train_mask, target_clf].astype(int)
    y_test_clf = df.loc[test_mask, target_clf].astype(int)

    cat_idx = [ALL_FEATURES.index(c) for c in CATEGORICAL_FEATURES]

    print("\nTraining classifier (profitable after tax: yes/no)...")
    # Hyperparameters below come from tune_hyperparameters.py --full: a 40-combo x
    # 4-fold walk-forward search, best result AUC=0.768 (vs ~0.75-0.76 with the
    # previous guessed defaults). Only the classifier was tuned directly (the
    # search optimized target_clf); the same combo is applied to the regressor
    # below too since a dedicated regression tuning pass hasn't been run yet.
    clf = HistGradientBoostingClassifier(
        categorical_features=cat_idx, max_iter=300, learning_rate=0.02,
        max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0, max_bins=255,
        early_stopping=True, validation_fraction=0.15, random_state=42,
    )
    clf.fit(X_train, y_train_clf)

    proba_test = clf.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)
    print(f"Holdout AUC: {roc_auc_score(y_test_clf, proba_test):.3f}")
    print(f"Holdout accuracy @0.5 threshold: {accuracy_score(y_test_clf, pred_test):.3f}")
    print(f"Baseline (predict majority class) accuracy: {max(y_test_clf.mean(), 1 - y_test_clf.mean()):.3f}")

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

    print("\nTraining regressor (expected % return)...")
    reg = HistGradientBoostingRegressor(
        categorical_features=cat_idx, max_iter=300, learning_rate=0.02,
        max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0, max_bins=255,
        early_stopping=True, validation_fraction=0.15, random_state=42,
    )
    reg.fit(X_train, y_train_reg)
    pred_reg = reg.predict(X_test)
    print(f"Holdout MAE: {mean_absolute_error(y_test_reg, pred_reg):.3f}")
    print(f"Correlation(predicted, actual): {np.corrcoef(pred_reg, y_test_reg)[0,1]:.3f}")

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
