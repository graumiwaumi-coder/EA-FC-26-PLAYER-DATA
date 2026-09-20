"""
Task F: does league/club/nation popularity add value beyond individual stats?

Directly tests the standing hypothesis from the start of this project ("I don't
think club, league matters all that much its about the players individual
stats") -- and the tension Task D's SHAP analysis just raised (club_price_score
ranked #5 in SHAP importance despite not showing up in the permutation-
importance top 20 at all).

Trains two models on IDENTICAL data/hyperparameters/folds, differing only in
one thing: the "clean" model has every club/league/nation-popularity signal
removed -- not just the 6 numeric popularity SCORE features
(evaluate_robustness.py's existing check already did that), but also the raw
"league" and "nation" CATEGORICAL columns themselves, since the model can
split directly on those even without a popularity score attached. ("club" was
never a feature here -- confirmed against train_model.py's CATEGORICAL_FEATURES
and NUMERIC_FEATURES; only league_price_score/club_price_score etc. and the raw
league/nation columns carry any club/league/nation information at all.)

Run across all 4 rolling-window folds (not a single split) so the comparison
has some statistical grounding rather than resting on one split's luck.

Run: python3 league_club_ablation.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from train_model import ALL_FEATURES, CATEGORICAL_FEATURES, HORIZON, load_data, make_folds

POPULARITY_SCORE_FEATURES = ["league_price_score", "league_liquidity_score",
                              "club_price_score", "club_liquidity_score",
                              "nation_price_score", "nation_liquidity_score"]
POPULARITY_CATEGORICAL_FEATURES = ["league", "nation"]  # "club" isn't a feature here at all
ABLATED_FEATURES = POPULARITY_SCORE_FEATURES + POPULARITY_CATEGORICAL_FEATURES


def make_clf(cat_idx):
    # same tuned hyperparameters used everywhere else in this project, so
    # neither side of the comparison is handicapped by worse tuning
    return HistGradientBoostingClassifier(
        categorical_features=cat_idx, max_iter=300, learning_rate=0.02,
        max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0, max_bins=255,
        early_stopping=True, validation_fraction=0.15, random_state=42,
    )


def main():
    print("Loading data...")
    df = load_data()
    target_clf = f"fwd_up_{HORIZON}d_net_tax"
    target_reg = f"fwd_return_{HORIZON}d_net_tax"
    df = df.dropna(subset=[target_clf, target_reg])

    clean_features = [f for f in ALL_FEATURES if f not in ABLATED_FEATURES]
    clean_cat = [c for c in CATEGORICAL_FEATURES if c not in ABLATED_FEATURES]
    print(f"Full feature set: {len(ALL_FEATURES)} features")
    print(f"Clean (individual-stats-only) feature set: {len(clean_features)} features")
    print(f"Removed: {ABLATED_FEATURES}")

    full_cat_idx = [ALL_FEATURES.index(c) for c in CATEGORICAL_FEATURES]
    clean_cat_idx = [clean_features.index(c) for c in clean_cat]

    fold_edges = [0.55, 0.65, 0.75, 0.85, 0.95]
    folds = make_folds(df, fold_edges)
    print(f"\n=== {len(folds)}-fold comparison: full feature set vs. individual-stats-only ===")

    results = []
    for i, (train_mask, test_mask) in enumerate(folds):
        y_train = df.loc[train_mask, target_clf].astype(int)
        y_test = df.loc[test_mask, target_clf].astype(int)

        clf_full = make_clf(full_cat_idx)
        clf_full.fit(df.loc[train_mask, ALL_FEATURES], y_train)
        proba_full = clf_full.predict_proba(df.loc[test_mask, ALL_FEATURES])[:, 1]
        auc_full = roc_auc_score(y_test, proba_full)
        wr_full = y_test.to_numpy()[proba_full >= 0.5].mean() if (proba_full >= 0.5).sum() > 0 else np.nan

        clf_clean = make_clf(clean_cat_idx)
        clf_clean.fit(df.loc[train_mask, clean_features], y_train)
        proba_clean = clf_clean.predict_proba(df.loc[test_mask, clean_features])[:, 1]
        auc_clean = roc_auc_score(y_test, proba_clean)
        wr_clean = y_test.to_numpy()[proba_clean >= 0.5].mean() if (proba_clean >= 0.5).sum() > 0 else np.nan

        print(f"  Fold {i+1}/{len(folds)}: FULL auc={auc_full:.4f} win_rate={wr_full:.1%}  |  "
              f"CLEAN auc={auc_clean:.4f} win_rate={wr_clean:.1%}  |  "
              f"delta(full-clean)={auc_full-auc_clean:+.4f}")
        results.append({"fold": i + 1, "auc_full": auc_full, "auc_clean": auc_clean,
                         "wr_full": wr_full, "wr_clean": wr_clean, "delta_auc": auc_full - auc_clean})

    results_df = pd.DataFrame(results)
    mean_delta = results_df["delta_auc"].mean()
    std_delta = results_df["delta_auc"].std()
    n_positive = (results_df["delta_auc"] > 0).sum()
    print(f"\nAveraged: AUC full={results_df['auc_full'].mean():.4f}, "
          f"AUC clean={results_df['auc_clean'].mean():.4f}")
    print(f"Delta (full - clean): mean={mean_delta:+.4f}, std={std_delta:.4f}, "
          f"full-set won {n_positive}/{len(results_df)} folds")
    print(f"\n(n={len(folds)} folds is too small for a formal significance test -- treat this as a "
          f"directional read, not a p-value. Consistent sign across all folds is the strongest "
          f"signal available at this sample size; a mixed sign means no reliable conclusion either way.)")

    if abs(mean_delta) < 0.005:
        verdict = ("Removing league/club/nation signal barely moved AUC (<0.005) -- consistent with "
                   "the standing hypothesis that individual stats drive most of the value, not "
                   "club/league popularity.")
    elif mean_delta > 0 and n_positive == len(results_df):
        verdict = ("The full feature set consistently beat the clean one across every fold -- "
                   "league/club/nation DO appear to carry real signal beyond individual stats, "
                   "contradicting the standing hypothesis. Worth reconciling with Task D's SHAP "
                   "finding that club_price_score ranked #5 in importance.")
    elif mean_delta < 0:
        verdict = ("The clean (individual-stats-only) model did AS WELL OR BETTER than the full "
                   "set -- league/club/nation features add no value and may even be adding noise.")
    else:
        verdict = "Mixed result across folds -- no reliable directional conclusion at this sample size."
    print(f"\nVerdict: {verdict}")

    out_path = "../data/league_club_ablation_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
