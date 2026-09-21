"""
Task D: SHAP interpretability analysis.

Permutation importance (already in train_model.py) tells you WHICH features
matter on average. SHAP (Shapley values, from cooperative game theory) goes
further: for every INDIVIDUAL prediction, it splits the model's output into
each feature's exact contribution, so you can see not just "rating matters"
but "rating pushed THIS prediction up by 2 points because it's unusually high
for this player's price level" -- and, in aggregate across many predictions,
whether a feature's effect is monotonic (always higher=better) or more
complex (e.g. a sweet spot in the middle).

Uses shap.TreeExplainer, which computes EXACT Shapley values for tree
ensembles (not an approximation) by walking the model's actual tree
structure -- fast and precise for HistGradientBoostingClassifier.

Explains the SAME held-out period train_model.py's deployed model reports
its numbers against (the last fold's test set), on a sample (not the full
set) since SHAP computation scales with rows x features x trees and there's
no need for millions of rows to get a stable picture.

Run: python3 shap_analysis.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import shap

from train_model import ALL_FEATURES, CATEGORICAL_FEATURES, HORIZON, MODEL_DIR, load_data, make_folds

SAMPLE_SIZE = 20000
TOP_N = 20


def prepare_sample():
    df = load_data()
    target_reg = f"fwd_return_{HORIZON}d_net_tax"
    target_clf = f"fwd_up_{HORIZON}d_net_tax"
    df = df.dropna(subset=[target_reg, target_clf])

    fold_edges = [0.55, 0.65, 0.75, 0.85, 0.95]
    folds = make_folds(df, fold_edges)
    _, test_mask = folds[-1]
    test_df = df.loc[test_mask]

    if len(test_df) > SAMPLE_SIZE:
        test_df = test_df.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"Explaining {len(test_df)} rows from the deployed model's held-out test period")

    X = test_df[ALL_FEATURES].copy()
    # shap's TreeExplainer wants plain numeric arrays -- convert the pandas
    # 'category' dtype columns to their integer codes, which is exactly what
    # HistGradientBoostingClassifier's native categorical support already
    # treats them as internally, so this doesn't change what the model sees.
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].cat.codes
    return X, test_df


def main():
    clf = joblib.load(MODEL_DIR / "clf_21d.joblib")
    X, test_df = prepare_sample()

    print("Computing exact SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer(X)

    # binary classifier -- shap_values.values may be (n, features) or (n, features, 2)
    # depending on shap/sklearn version; normalize to the positive-class contribution
    values = shap_values.values
    if values.ndim == 3:
        values = values[:, :, 1]

    mean_abs_shap = pd.Series(np.abs(values).mean(axis=0), index=ALL_FEATURES).sort_values(ascending=False)
    print(f"\n=== Top {TOP_N} features by mean |SHAP value| (global importance) ===")
    print(mean_abs_shap.head(TOP_N))

    print(f"\n=== Direction of effect for top {TOP_N} features ===")
    print("(correlation between the feature's own value and its SHAP contribution --")
    print(" positive = higher feature value pushes the prediction up, negative = pushes it down,")
    print(" near-zero = not a simple monotonic relationship, check the quantile breakdown instead)")
    for feat in mean_abs_shap.head(TOP_N).index:
        col_idx = ALL_FEATURES.index(feat)
        feat_vals = X[feat].to_numpy()
        shap_vals = values[:, col_idx]
        valid = ~np.isnan(feat_vals.astype(float)) if feat not in CATEGORICAL_FEATURES else np.ones(len(feat_vals), dtype=bool)
        if valid.sum() > 10:
            corr = np.corrcoef(feat_vals[valid].astype(float), shap_vals[valid])[0, 1]
        else:
            corr = np.nan
        print(f"  {feat}: corr(value, shap)={corr:+.3f}")

    print(f"\n=== Quantile breakdown for top 8 NUMERIC features (avoids assuming linearity) ===")
    for feat in [f for f in mean_abs_shap.head(TOP_N).index if f not in CATEGORICAL_FEATURES][:8]:
        col_idx = ALL_FEATURES.index(feat)
        feat_vals = X[feat].to_numpy().astype(float)
        shap_vals = values[:, col_idx]
        valid = ~np.isnan(feat_vals)
        if valid.sum() < 50:
            continue
        try:
            q = pd.qcut(feat_vals[valid], 5, labels=["Q1(low)", "Q2", "Q3", "Q4", "Q5(high)"], duplicates="drop")
        except ValueError:
            continue
        by_q = pd.Series(shap_vals[valid]).groupby(q, observed=True).mean()
        print(f"  {feat}: " + ", ".join(f"{k}={v:+.3f}" for k, v in by_q.items()))

    # save plots for the person to actually look at -- SHAP is fundamentally a
    # visual technique, a text table alone loses most of what makes it useful
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = Path(__file__).resolve().parent.parent / "data" / "shap_plots"
    plots_dir.mkdir(exist_ok=True)

    plt.figure()
    shap.summary_plot(values, X, feature_names=ALL_FEATURES, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(plots_dir / "shap_summary_beeswarm.png", dpi=150)
    plt.close()

    plt.figure()
    shap.summary_plot(values, X, feature_names=ALL_FEATURES, max_display=20, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(plots_dir / "shap_summary_bar.png", dpi=150)
    plt.close()

    top4 = mean_abs_shap.head(4).index.tolist()
    for feat in top4:
        col_idx = ALL_FEATURES.index(feat)
        plt.figure()
        shap.dependence_plot(col_idx, values, X, feature_names=ALL_FEATURES, show=False)
        plt.tight_layout()
        safe_name = feat.replace("/", "_")
        plt.savefig(plots_dir / f"shap_dependence_{safe_name}.png", dpi=150)
        plt.close()

    print(f"\nSaved plots to {plots_dir}/")

    np.save(plots_dir / "shap_values_sample.npy", values)
    X.to_parquet(plots_dir / "shap_sample_features.parquet", index=False)
    print("Saved raw SHAP values + feature sample for further analysis if needed")


if __name__ == "__main__":
    main()
