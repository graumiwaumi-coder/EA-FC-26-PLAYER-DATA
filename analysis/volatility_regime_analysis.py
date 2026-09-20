"""
Task E: volatility-regime conditioning.

The per-player volatility features (volatility_7/14/30/60, autocorr_30d) already
in the model describe how choppy an INDIVIDUAL card's own price has been. This
is a different question: how choppy is the WHOLE MARKET right now, and does the
model's edge actually depend on that broader regime?

Directly motivated by something observed repeatedly this session: fold 4
(the most recent test period, roughly July-August) has consistently scored far
worse (AUC ~0.67) than folds 1-3 (~0.78-0.81) across every run so far --
hyperparameter tuning, the plain retrain, the Task A/B integration attempts, the
Task F ablation. That consistency across totally different model variants
suggests something structural about THAT PERIOD, not noise -- a natural
volatility-regime hypothesis to actually test rather than just note.

Regime is defined from Index100 (the overall market index, band="100" in
indices.parquet) -- a rolling 14-day standard deviation of daily index returns,
split into LOW/MEDIUM/HIGH terciles by that measure's own historical
distribution. This is a genuinely different signal from the existing per-player
volatility features (it's market-wide, not player-specific), so worth testing
for real incremental value the way Tasks A and B were, rather than assuming.

Run: python3 volatility_regime_analysis.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from train_model import ALL_FEATURES, CATEGORICAL_FEATURES, HORIZON, DATA_DIR, load_data, make_folds

ROLLING_WINDOW = 14


def build_regime_series():
    idx = pd.read_parquet(DATA_DIR / "indices.parquet")
    idx100 = idx[idx["band"] == "100"].sort_values(["platform", "date"]).copy()
    idx100["daily_return"] = idx100.groupby("platform")["index_value"].pct_change()
    idx100["rolling_vol"] = (
        idx100.groupby("platform")["daily_return"]
        .rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW // 2)
        .std()
        .reset_index(level=0, drop=True)
    )
    idx100 = idx100.dropna(subset=["rolling_vol"])

    terciles = idx100["rolling_vol"].quantile([1 / 3, 2 / 3]).to_numpy()
    idx100["regime"] = pd.cut(idx100["rolling_vol"], bins=[-np.inf, terciles[0], terciles[1], np.inf],
                               labels=["LOW", "MEDIUM", "HIGH"])
    print(f"Regime tercile cutoffs (rolling {ROLLING_WINDOW}d std of Index100 daily return): "
          f"LOW<{terciles[0]:.4f}, MEDIUM<{terciles[1]:.4f}, HIGH>={terciles[1]:.4f}")
    return idx100[["platform", "date", "rolling_vol", "regime"]]


def make_clf(cat_idx):
    return HistGradientBoostingClassifier(
        categorical_features=cat_idx, max_iter=300, learning_rate=0.02,
        max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0, max_bins=255,
        early_stopping=True, validation_fraction=0.15, random_state=42,
    )


def main():
    print("Building market volatility regime series from Index100...")
    regime = build_regime_series()

    print("Loading data and merging regime labels...")
    df = load_data()
    target_clf = f"fwd_up_{HORIZON}d_net_tax"
    target_reg = f"fwd_return_{HORIZON}d_net_tax"
    df = df.dropna(subset=[target_clf, target_reg])
    df = df.merge(regime, on=["platform", "date"], how="left")
    print(f"Rows with a regime label: {df['regime'].notna().sum()} / {len(df)}")

    cat_idx = [ALL_FEATURES.index(c) for c in CATEGORICAL_FEATURES]
    fold_edges = [0.55, 0.65, 0.75, 0.85, 0.95]
    folds = make_folds(df, fold_edges)

    print(f"\n=== What regime was each fold's TEST period predominantly in? ===")
    all_preds = []
    for i, (train_mask, test_mask) in enumerate(folds):
        test_regime_dist = df.loc[test_mask, "regime"].value_counts(normalize=True)
        print(f"  Fold {i+1}: {dict(test_regime_dist.round(2))}")

        clf = make_clf(cat_idx)
        clf.fit(df.loc[train_mask, ALL_FEATURES], df.loc[train_mask, target_clf].astype(int))
        proba = clf.predict_proba(df.loc[test_mask, ALL_FEATURES])[:, 1]
        fold_result = pd.DataFrame({
            "fold": i + 1,
            "proba": proba,
            "actual": df.loc[test_mask, target_clf].astype(int).to_numpy(),
            "regime": df.loc[test_mask, "regime"].to_numpy(),
        })
        all_preds.append(fold_result)

    preds = pd.concat(all_preds, ignore_index=True)

    print(f"\n=== Model performance BY REGIME (pooled across all 4 folds' test predictions) ===")
    for regime_label in ["LOW", "MEDIUM", "HIGH"]:
        sub = preds[preds["regime"] == regime_label]
        if len(sub) < 100:
            print(f"  {regime_label}: n={len(sub)}, too few rows")
            continue
        auc = roc_auc_score(sub["actual"], sub["proba"])
        confident = sub[sub["proba"] >= 0.5]
        win_rate = confident["actual"].mean() if len(confident) > 0 else np.nan
        print(f"  {regime_label}: n={len(sub)}, AUC={auc:.4f}, win_rate@0.5={win_rate:.1%} "
              f"(n_confident={len(confident)})")

    print(f"\n=== Same breakdown, PER FOLD (checks whether the regime effect is consistent "
          f"or driven by one fold) ===")
    for i in range(len(folds)):
        fold_preds = preds[preds["fold"] == i + 1]
        print(f"  Fold {i+1}:")
        for regime_label in ["LOW", "MEDIUM", "HIGH"]:
            sub = fold_preds[fold_preds["regime"] == regime_label]
            if len(sub) < 50:
                continue
            auc = roc_auc_score(sub["actual"], sub["proba"])
            print(f"    {regime_label}: n={len(sub)}, AUC={auc:.4f}")

    preds.to_csv(DATA_DIR / "volatility_regime_predictions.csv", index=False)
    print(f"\nSaved {DATA_DIR / 'volatility_regime_predictions.csv'}")


if __name__ == "__main__":
    main()
