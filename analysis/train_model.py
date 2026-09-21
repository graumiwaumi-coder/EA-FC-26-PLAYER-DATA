#!/usr/bin/env python3
"""
Stage 2: trains the baseline model(s) on data/feature_panel.parquet.

Model: LightGBM (free, open-source, native missing-value + categorical
support -- the right fit for FC26/FC27's very different feature richness
and the rich categorical card metadata, and fast enough to retrain
Wed/Sun nights on this VPS's CPU, no GPU needed).

Two outputs per prediction horizon: a win-probability CLASSIFIER
(label_win_Xd, already tax-aware -- see build_features.py's
WIN_BREAKEVEN) and an expected-return REGRESSOR (label_return_Xd),
trained separately -- 6 horizons x 2 tasks = 12 models. CatBoost/XGBoost
ensembling and Optuna hyperparameter tuning are deliberately deferred
until this baseline's real walk-forward numbers justify the added
complexity -- this is step one, not the final model.

Validation: rolling walk-forward with an embargo gap (never a naive
train/test split -- PROJECT.md flags this as a real failure mode from
the previous build: it gave misleadingly optimistic, unstable results).
The embargo is sized to the largest horizon (30 days) so a test-set
row's forward-looking label window can never reach back across the
train/test boundary.

Training scope: rating >= 75 only (Gold+, per project scope). Fodder
cards trade on different, SBC-demand-driven dynamics and would just
dilute what the model learns about the cards this system actually
recommends.

Memory: loads only the columns actually needed (not all ~123) via
read_parquet(columns=...), and the rating filter cuts the row count
further before any training happens -- both lessons from build_features.py
needing the same care on this VPS's 11GB, no-swap ceiling.

Run: python3 train_model.py
Writes: models/<timestamp>/{clf,reg}_<horizon>d.txt (12 LightGBM model
files) + meta.json (feature list, categorical columns, walk-forward
metrics), and updates the models/latest symlink to point at the new
version -- rollback is just repointing that symlink back to an older
timestamp, nothing gets overwritten or deleted.
"""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import mean_absolute_error, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
FEATURE_PANEL_PATH = DATA_DIR / "feature_panel.parquet"

HORIZONS = [1, 3, 7, 14, 21, 30]
MIN_RATING = 75.0
FC27_SAMPLE_WEIGHT = 5.0  # FC27 rows count for more than FC26 rows during
                          # training -- a fixed multiplier for now; revisit
                          # (increase further) as FC27 builds up its own
                          # history and needs less help from FC26's patterns
EMBARGO_DAYS = 30  # >= the largest horizon, so no test-set label's forward
                   # window can reach back into the train set
TEST_DAYS = 30
N_SPLITS = 4
MIN_TRAIN_DAYS = 90
WIN_THRESHOLD = 0.5  # probability above which we'd call it a "buy" signal,
                     # for the precision/profit-style reporting below

LGB_PARAMS = dict(
    num_leaves=31, learning_rate=0.05, n_estimators=300,
    min_child_samples=50, subsample=0.8, colsample_bytree=0.8, verbosity=-1,
)


def log(msg):
    print(msg, flush=True)


def split_columns(all_cols):
    label_cols = [c for c in all_cols if c.startswith("label_")]
    exclude = {"player_id", "date"} | set(label_cols)
    feature_cols = [c for c in all_cols if c not in exclude]
    return feature_cols, label_cols


def load_training_data():
    """Confirmed live: loading all columns and THEN filtering to rating >=
    MIN_RATING in pandas got the process killed (OOM) -- filtering after
    load never actually reduces peak memory, since every excluded row was
    already fully materialized by that point. Passing `filters=` to
    read_parquet pushes the rating filter into the parquet read itself, so
    the ~70% of rows below Gold+ never get loaded into memory at all.
    (player_id was dropped from load_cols entirely -- it's carried through
    build_features.py for bookkeeping/joins but nothing here actually uses
    it as a feature or for splitting, so there's no reason to load it.)"""
    schema = pq.ParquetFile(str(FEATURE_PANEL_PATH)).schema_arrow
    all_cols = [f.name for f in schema]
    feature_cols, label_cols = split_columns(all_cols)
    load_cols = list(dict.fromkeys(["date", "rating"] + feature_cols + label_cols))

    log(f"Loading {len(load_cols)} of {len(all_cols)} columns from {FEATURE_PANEL_PATH}, "
        f"filtered to rating >= {MIN_RATING} at read time...")
    df = pd.read_parquet(
        FEATURE_PANEL_PATH, columns=load_cols, filters=[("rating", ">=", MIN_RATING)],
    )
    log(f"  {len(df)} rows loaded (already rating-filtered)")

    return df, feature_cols, label_cols


def prepare_features(df, feature_cols):
    """Cast every feature to either 'category' (for anything string/object-
    like, regardless of whether it happened to survive as a pandas category
    dtype through the build_features.py merges) or float32. Doing this
    explicitly here rather than trusting the saved dtype makes this script
    self-contained and correct even if an upstream dtype quietly changes."""
    cat_cols = []
    for c in feature_cols:
        dtype = df[c].dtype
        if isinstance(dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(dtype) \
                or pd.api.types.is_string_dtype(dtype):
            df[c] = df[c].astype("category")
            cat_cols.append(c)
        elif pd.api.types.is_bool_dtype(dtype):
            df[c] = df[c].astype("float32")
        elif pd.api.types.is_numeric_dtype(dtype) and dtype != np.dtype("float32"):
            df[c] = df[c].astype("float32")
    return df, cat_cols


def walk_forward_splits(dates, horizon, n_splits=N_SPLITS, test_days=TEST_DAYS,
                         embargo_days=EMBARGO_DAYS, min_train_days=MIN_TRAIN_DAYS):
    """Rolling-window folds, walking backward from the most recent data.
    Each fold's test window is preceded by an embargo gap so a label's
    forward-looking span (up to `embargo_days`) can never leak across the
    train/test boundary. Returns (train_mask, test_mask) numpy boolean
    array pairs, oldest fold first.

    Confirmed live: without the `horizon` adjustment below, the LAST fold at
    long horizons (21d, 30d) was mostly or entirely unusable -- a test row
    dated within `horizon` days of the most recent data we have literally
    cannot have a valid label yet (there's no future price to check it
    against), so that fold's test window was largely wasted on unlabelable
    rows. 30d's last fold had fewer than 10 valid rows and got silently
    dropped; 21d's had ~43K (thin enough that its AUC of 0.43 was likely
    noise/an artifact of the thin, date-narrow sample rather than a real
    signal). Capping the latest possible test_end at
    (max_date - horizon days) keeps every fold's test window inside the
    region that can actually be labeled."""
    dates = pd.to_datetime(dates)
    min_date, max_date = dates.min(), dates.max()
    folds = []
    test_end = max_date - pd.Timedelta(days=horizon)
    for _ in range(n_splits):
        test_start = test_end - pd.Timedelta(days=test_days)
        train_end = test_start - pd.Timedelta(days=embargo_days)
        if (train_end - min_date).days < min_train_days:
            break
        train_mask = (dates <= train_end).to_numpy()
        test_mask = ((dates > test_start) & (dates <= test_end)).to_numpy()
        folds.append((train_mask, test_mask))
        test_end = test_start
    folds.reverse()
    return folds


def sample_weights(df):
    return np.where(df["game_version"].astype(str) == "fc27", FC27_SAMPLE_WEIGHT, 1.0)


def evaluate_classifier(model, X_test, y_test, returns_test):
    if len(X_test) == 0:
        return None
    proba = model.predict_proba(X_test)[:, 1]
    valid = ~np.isnan(y_test)
    if valid.sum() < 10:
        return None
    y_valid = y_test[valid]
    auc = float(roc_auc_score(y_valid, proba[valid])) if len(np.unique(y_valid)) > 1 else float("nan")
    picks = valid & (proba >= WIN_THRESHOLD)
    n_picks = int(picks.sum())
    if n_picks > 0:
        avg_return_of_picks = float(np.nanmean(returns_test[picks]))
        win_rate_of_picks = float(np.nanmean(y_test[picks]))
    else:
        avg_return_of_picks = float("nan")
        win_rate_of_picks = float("nan")
    return {"auc": auc, "n_test": int(valid.sum()), "n_picks": n_picks,
            "win_rate_of_picks": win_rate_of_picks, "avg_return_of_picks": avg_return_of_picks}


def evaluate_regressor(model, X_test, y_test):
    """Called with rows already restricted to actual wins (see
    train_horizon), so y_test is always positive here -- directional_accuracy
    now checks whether the model correctly predicts a positive move rather
    than underpredicting into negative territory, not "up vs down" in
    general (that question belongs to the classifier)."""
    if len(X_test) == 0:
        return None
    valid = ~np.isnan(y_test)
    if valid.sum() < 10:
        return None
    pred = model.predict(X_test[valid])
    mae = float(mean_absolute_error(y_test[valid], pred))
    directional_acc = float((np.sign(pred) == np.sign(y_test[valid])).mean())
    return {"mae": mae, "directional_accuracy": directional_acc, "n_test": int(valid.sum())}


def train_horizon(df, feature_cols, cat_cols, horizon, weights, is_pc):
    win_col = f"label_win_{horizon}d"
    ret_col = f"label_return_{horizon}d"
    X_all = df[feature_cols]
    y_win = df[win_col].to_numpy()
    y_ret = df[ret_col].to_numpy()

    folds = walk_forward_splits(df["date"], horizon)
    log(f"  {len(folds)} walk-forward folds for horizon={horizon}d")

    fold_clf_results, fold_reg_results = [], []
    for i, (train_mask, test_mask) in enumerate(folds):
        win_train_mask = train_mask & ~np.isnan(y_win)
        clf = lgb.LGBMClassifier(objective="binary", **LGB_PARAMS)
        clf.fit(X_all[win_train_mask], y_win[win_train_mask],
                sample_weight=weights[win_train_mask], categorical_feature=cat_cols)
        # Trained on BOTH platforms (console data is a real, useful signal --
        # see the cross-platform lead-lag features), but graded ONLY on PC
        # rows -- the person trading this only trades on PC, so a walk-forward
        # "win rate" that secretly includes console outcomes would overstate
        # or understate what they'd actually experience.
        pc_test_mask = test_mask & is_pc
        res = evaluate_classifier(clf, X_all[pc_test_mask], y_win[pc_test_mask], y_ret[pc_test_mask])
        if res:
            fold_clf_results.append(res)

        # Regressor trains (and is evaluated) ONLY on rows that were actual
        # confirmed wins, not on the full mix of wins/losses/flat moves.
        # Confirmed live: FUT price moves are heavily skewed (most cards
        # drift flat-to-down, a few spike hard), so a regressor fit on
        # everything gets pulled toward the skewed mean and ends up guessing
        # the wrong sign more often than not (38-45% directional accuracy,
        # worse than a coin flip). Restricting it to "given this IS a real
        # winner, how big is the move" is a different, better-posed question
        # -- the classifier above already answers "is this a winner at all."
        ret_train_mask = train_mask & (y_win == 1)
        reg = lgb.LGBMRegressor(objective="regression", **LGB_PARAMS)
        reg.fit(X_all[ret_train_mask], y_ret[ret_train_mask],
                sample_weight=weights[ret_train_mask], categorical_feature=cat_cols)
        ret_test_mask = pc_test_mask & (y_win == 1)
        res_r = evaluate_regressor(reg, X_all[ret_test_mask], y_ret[ret_test_mask])
        if res_r:
            fold_reg_results.append(res_r)
        log(f"    fold {i + 1}/{len(folds)}: clf={res}, reg={res_r}")

    # Final deployed model: trained on ALL rows with a valid target, no
    # held-out test set -- the walk-forward folds above exist purely to
    # measure honest out-of-sample performance, not to pick this model.
    final_win_mask = ~np.isnan(y_win)
    final_clf = lgb.LGBMClassifier(objective="binary", **LGB_PARAMS)
    final_clf.fit(X_all[final_win_mask], y_win[final_win_mask],
                  sample_weight=weights[final_win_mask], categorical_feature=cat_cols)

    final_ret_mask = y_win == 1
    final_reg = lgb.LGBMRegressor(objective="regression", **LGB_PARAMS)
    final_reg.fit(X_all[final_ret_mask], y_ret[final_ret_mask],
                  sample_weight=weights[final_ret_mask], categorical_feature=cat_cols)

    summary = {"horizon": horizon, "n_folds": len(folds),
               "clf_folds": fold_clf_results, "reg_folds": fold_reg_results}
    return final_clf, final_reg, summary


def main():
    df, feature_cols, label_cols = load_training_data()
    df, cat_cols = prepare_features(df, feature_cols)
    log(f"{len(feature_cols)} feature columns ({len(cat_cols)} categorical: {cat_cols})")

    weights = sample_weights(df)
    # Trained on console + PC together (console is a real leading-indicator
    # signal, not just noise -- see the cross-platform features), but every
    # walk-forward success metric is graded on PC rows only, since PC is the
    # only market actually being traded.
    is_pc = (df["platform"].astype(str) == "pc").to_numpy()
    log(f"Evaluation scope: {is_pc.sum()} of {len(df)} rows are PC (graded on these only; "
        f"console rows still used for training)")

    version = time.strftime("%Y%m%d_%H%M%S")
    version_dir = MODELS_DIR / version
    version_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for h in HORIZONS:
        log(f"\n=== Training horizon {h}d ===")
        clf, reg, summary = train_horizon(df, feature_cols, cat_cols, h, weights, is_pc)
        clf.booster_.save_model(str(version_dir / f"clf_{h}d.txt"))
        reg.booster_.save_model(str(version_dir / f"reg_{h}d.txt"))
        all_summaries.append(summary)

    meta = {
        "version": version, "feature_cols": feature_cols, "categorical_cols": cat_cols,
        "horizons": HORIZONS, "min_rating": MIN_RATING, "fc27_sample_weight": FC27_SAMPLE_WEIGHT,
        "n_rows_trained": len(df), "n_pc_rows_trained": int(is_pc.sum()),
        "eval_scope": "pc_only (trained on console+pc combined; all fold metrics are PC-only)",
        "summaries": all_summaries,
    }
    (version_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    latest_link = MODELS_DIR / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(version, target_is_directory=True)

    log(f"\nSaved 12 models to {version_dir}")
    log(f"models/latest -> {version}")


if __name__ == "__main__":
    main()
