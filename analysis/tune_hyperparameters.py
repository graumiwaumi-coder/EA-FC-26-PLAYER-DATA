"""
Task 2: real hyperparameter tuning, using the same rolling walk-forward folds as the
Colab notebook (not a plain train/test split), so results are honest and comparable
to everything validated so far.

Combines the original feature set (train_model.py's ALL_FEATURES) with the new
finance-inspired features from build_features_v2.py.

LOCAL_TEST=True runs a small combo x fold grid to prove the mechanics are correct
before handing the full search to the VPS. Set LOCAL_TEST=False (or pass --full) for
the real search.

Run: python3 tune_hyperparameters.py [--full]
"""
import gc
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from train_model import NUMERIC_FEATURES, NEW_FEATURES, CATEGORICAL_FEATURES, MIN_RATING, HORIZON

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EMBARGO_DAYS = HORIZON
TRAIN_WINDOW_DAYS = 200

PLAYER_LEVEL_NUMERIC = ["skills", "weak_foot", "height_cm", "age", "n_playstyles",
                        "league_price_score", "league_liquidity_score", "club_price_score",
                        "club_liquidity_score", "nation_price_score", "nation_liquidity_score"]

ALL_FEATURES = NUMERIC_FEATURES + NEW_FEATURES + CATEGORICAL_FEATURES


def load_combined_data():
    players = pd.read_parquet(DATA_DIR / "players_features.parquet")
    players = players[players["rating"] >= MIN_RATING]
    player_cols = ["id"] + [c for c in CATEGORICAL_FEATURES if c != "platform"] + PLAYER_LEVEL_NUMERIC
    players_sub = players[player_cols]

    price_native = [c for c in NUMERIC_FEATURES if c != "rating" and c not in PLAYER_LEVEL_NUMERIC]
    price_cols = list(dict.fromkeys(price_native + ["player_id", "platform", "date", "rating",
                      f"fwd_return_{HORIZON}d_net_tax", f"fwd_up_{HORIZON}d_net_tax"]))
    dataset = ds.dataset(DATA_DIR / "prices_features.parquet", format="parquet")
    table = dataset.to_table(columns=price_cols, filter=ds.field("rating") >= MIN_RATING)
    df = table.to_pandas(split_blocks=True, self_destruct=True)
    del table

    v2 = pd.read_parquet(DATA_DIR / "prices_features_v2.parquet",
                          columns=["player_id", "platform", "date"] + NEW_FEATURES)
    df = df.merge(v2, on=["player_id", "platform", "date"], how="left")

    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype("float32")
    df = df.merge(players_sub, left_on="player_id", right_on="id", how="left").drop(columns=["id"])
    for c in CATEGORICAL_FEATURES:
        df[c] = df[c].astype("category")

    target_reg = f"fwd_return_{HORIZON}d_net_tax"
    target_clf = f"fwd_up_{HORIZON}d_net_tax"
    df = df.dropna(subset=[target_reg, target_clf])
    print(f"Combined training data: {len(df)} rows, {len(ALL_FEATURES)} features "
          f"({len(NEW_FEATURES)} new), {df['player_id'].nunique()} players")
    return df, target_reg, target_clf


def make_folds(df, fold_edges):
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


class ProgressTracker:
    """Prints a % progress bar with elapsed time and an ETA, updated after every
    individual model fit (not just every combo) so a long unattended run never
    looks stalled."""

    def __init__(self, total_fits):
        self.total = total_fits
        self.done = 0
        self.start_time = time.time()

    def tick(self, label=""):
        self.done += 1
        elapsed = time.time() - self.start_time
        rate = elapsed / self.done
        remaining = rate * (self.total - self.done)
        pct = self.done / self.total
        bar_len = 30
        filled = int(bar_len * pct)
        bar = "#" * filled + "-" * (bar_len - filled)

        def fmt(secs):
            m, s = divmod(int(secs), 60)
            h, m = divmod(m, 60)
            return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"

        print(f"  [{bar}] {pct:.0%} ({self.done}/{self.total}) "
              f"elapsed={fmt(elapsed)} ETA={fmt(remaining)} {label}", flush=True)


def evaluate_combo(df, folds, target_clf, params, cat_idx, progress=None):
    aucs, win_rates = [], []
    for fold_i, (train_mask, test_mask) in enumerate(folds):
        X_train = df.loc[train_mask, ALL_FEATURES]
        y_train = df.loc[train_mask, target_clf].astype(int)
        X_test = df.loc[test_mask, ALL_FEATURES]
        y_test = df.loc[test_mask, target_clf].astype(int)

        clf = HistGradientBoostingClassifier(categorical_features=cat_idx, early_stopping=True,
                                              validation_fraction=0.15, random_state=42, **params)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        fold_auc = roc_auc_score(y_test, proba)
        aucs.append(fold_auc)
        mask_conf = proba >= 0.5
        if mask_conf.sum() > 0:
            win_rates.append(y_test.to_numpy()[mask_conf].mean())
        if progress is not None:
            progress.tick(f"(fold {fold_i+1}/{len(folds)}, AUC={fold_auc:.3f})")
    return np.mean(aucs), np.std(aucs), np.mean(win_rates) if win_rates else np.nan


def main():
    full_run = "--full" in sys.argv

    df, target_reg, target_clf = load_combined_data()
    cat_idx = [ALL_FEATURES.index(c) for c in CATEGORICAL_FEATURES]

    if full_run:
        fold_edges = [0.55, 0.65, 0.75, 0.85, 0.95]
        param_grid = {
            "max_iter": [200, 300, 400],
            "learning_rate": [0.02, 0.05, 0.08],
            "max_leaf_nodes": [15, 31, 63],
            "min_samples_leaf": [20, 50, 100],
            "l2_regularization": [0.0, 0.5, 1.0],
            "max_bins": [128, 255],
        }
        rng = np.random.default_rng(42)
        keys = list(param_grid.keys())
        all_combos = list(itertools.product(*param_grid.values()))
        n_sample = min(40, len(all_combos))
        sampled = [all_combos[i] for i in rng.choice(len(all_combos), size=n_sample, replace=False)]
        combos = [dict(zip(keys, c)) for c in sampled]
        print(f"FULL SEARCH: {len(combos)} combos x {len(fold_edges)-1} folds = "
              f"{len(combos)*(len(fold_edges)-1)} model fits. This will take a while.")
    else:
        fold_edges = [0.65, 0.80, 0.95]  # only 2 folds for the local smoke test
        combos = [
            {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 0.0, "max_bins": 255},
            {"max_iter": 200, "learning_rate": 0.08, "max_leaf_nodes": 15, "min_samples_leaf": 50, "l2_regularization": 0.5, "max_bins": 255},
            {"max_iter": 400, "learning_rate": 0.02, "max_leaf_nodes": 63, "min_samples_leaf": 100, "l2_regularization": 1.0, "max_bins": 128},
        ]
        print(f"LOCAL SMOKE TEST: {len(combos)} combos x {len(fold_edges)-1} folds = "
              f"{len(combos)*(len(fold_edges)-1)} model fits -- just proving the mechanics work.")

    folds = make_folds(df, fold_edges)
    print(f"Using {len(folds)} folds")

    # Checkpoint to disk after every combo (not just at the end) so a crash --
    # OOM kill, dropped SSH session, anything -- loses at most one in-flight
    # combo instead of the whole run. Resuming (same out file already on
    # disk) skips combos already checkpointed rather than redoing them.
    out_name = "tuning_results_full.csv" if full_run else "tuning_results_smoketest.csv"
    out_path = DATA_DIR / out_name
    param_keys = list(combos[0].keys())
    done_params = set()
    if out_path.exists():
        existing_df = pd.read_csv(out_path)
        done_params = set(tuple(row) for row in existing_df[param_keys].itertuples(index=False, name=None))
        print(f"Resuming: {len(done_params)} combos already checkpointed in {out_name}, skipping those")

    remaining = [p for p in combos if tuple(p[k] for k in param_keys) not in done_params]
    total_fits = len(remaining) * len(folds)
    progress = ProgressTracker(total_fits)
    print(f"Total model fits this run: {total_fits} ({len(remaining)}/{len(combos)} combos remaining)\n")

    for i, params in enumerate(remaining):
        print(f"Combo {i+1}/{len(remaining)}: {params}")
        auc_mean, auc_std, win_rate = evaluate_combo(df, folds, target_clf, params, cat_idx, progress=progress)
        print(f"  -> AUC={auc_mean:.4f} (std={auc_std:.4f}) win_rate={win_rate:.1%}\n")
        row_df = pd.DataFrame([{**params, "auc_mean": auc_mean, "auc_std": auc_std, "win_rate": win_rate}])
        row_df.to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
        gc.collect()

    print(f"\nAll combos checkpointed to {out_name}")
    results_df = pd.read_csv(out_path).sort_values("auc_mean", ascending=False)
    print("\n=== Top result ===")
    print(results_df.iloc[0])


if __name__ == "__main__":
    main()
