"""
Phase 2 robustness checks:
1. De-duplicate the test set to ~independent 21-day episodes (daily snapshots of the
   same player are heavily autocorrelated, not independent trades) and recheck win rate.
2. Retrain WITHOUT the league/club/nation popularity features (which were built from
   the full season, including the test period -- a leakage risk) and compare AUC/win
   rate, to see how much of the edge (if any) depends on that leakage.

Run: python3 evaluate_robustness.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from train_model import (
    load_data, ALL_FEATURES, CATEGORICAL_FEATURES, HORIZON, EMBARGO_DAYS,
)

LEAKY_FEATURES = ["league_price_score", "league_liquidity_score", "club_price_score",
                   "club_liquidity_score", "nation_price_score", "nation_liquidity_score"]


def main():
    print("Reloading data...")
    df = load_data()
    target_reg = f"fwd_return_{HORIZON}d_net_tax"
    target_clf = f"fwd_up_{HORIZON}d_net_tax"
    df = df.dropna(subset=[target_reg, target_clf])

    dates = df["date"]
    split_date = dates.quantile(0.75)
    train_mask = dates <= (split_date - pd.Timedelta(days=EMBARGO_DAYS))
    test_mask = dates >= (split_date + pd.Timedelta(days=EMBARGO_DAYS))

    # ---- Check 1: de-duplicated, ~independent episodes ----
    import joblib
    clf = joblib.load("../data/models/clf_21d.joblib")
    test_df = df.loc[test_mask].copy()
    test_df["proba"] = clf.predict_proba(test_df[ALL_FEATURES])[:, 1]

    test_df = test_df.sort_values(["player_id", "platform", "date"])
    test_df["day_rank"] = test_df.groupby(["player_id", "platform"]).cumcount()
    dedup = test_df[test_df["day_rank"] % HORIZON == 0]

    print(f"\n=== Check 1: de-duplicated test set (n={len(dedup)} vs {len(test_df)} raw rows) ===")
    for thresh in (0.5, 0.6, 0.7):
        mask = dedup["proba"] >= thresh
        n = mask.sum()
        if n < 20:
            print(f"  prob>={thresh}: n={n} (too few for a reliable estimate)")
            continue
        win_rate = (dedup.loc[mask, target_clf] == 1).mean()
        med_ret = dedup.loc[mask, target_reg].median()
        print(f"  prob>={thresh}: n={n}, win rate={win_rate:.1%}, median net return={med_ret:.1%}")

    baseline_win_rate = (dedup[target_clf] == 1).mean()
    print(f"  (baseline win rate across ALL de-duplicated test rows: {baseline_win_rate:.1%})")

    # ---- Check 2: retrain without potentially-leaky popularity features ----
    print("\n=== Check 2: retrain WITHOUT full-season league/club/nation popularity features ===")
    clean_features = [f for f in ALL_FEATURES if f not in LEAKY_FEATURES]
    clean_cat_idx = [clean_features.index(c) for c in CATEGORICAL_FEATURES]

    X_train = df.loc[train_mask, clean_features]
    X_test = df.loc[test_mask, clean_features]
    y_train = df.loc[train_mask, target_clf].astype(int)
    y_test = df.loc[test_mask, target_clf].astype(int)

    # Same tuned hyperparameters as train_model.py (see tune_hyperparameters.py
    # --full results) so this comparison isn't biased by comparing a tuned
    # full-feature model against an untuned reduced-feature one.
    clf2 = HistGradientBoostingClassifier(
        categorical_features=clean_cat_idx, max_iter=300, learning_rate=0.02,
        max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0, max_bins=255,
        early_stopping=True, validation_fraction=0.15, random_state=42,
    )
    clf2.fit(X_train, y_train)
    proba2 = clf2.predict_proba(X_test)[:, 1]
    print(f"AUC without popularity features: {roc_auc_score(y_test, proba2):.3f}")

    y_test_reg = df.loc[test_mask, target_reg]
    for thresh in (0.5, 0.6, 0.7):
        mask = proba2 >= thresh
        n = mask.sum()
        if n == 0:
            continue
        win_rate = y_test.to_numpy()[mask].mean()
        med_ret = np.median(y_test_reg.to_numpy()[mask])
        print(f"  prob>={thresh}: n={n} ({n/len(proba2):.1%} of test), win rate={win_rate:.1%}, median net return={med_ret:.1%}")


if __name__ == "__main__":
    main()
