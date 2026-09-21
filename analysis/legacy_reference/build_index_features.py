"""
Phase 1 (expanded): market index momentum features.

Reads data/indices.parquet
Writes:
  data/indices_features.parquet  -- long format, all 16 series with momentum stats
  data/macro_wide.parquet        -- one row per (platform, date) with Index100 and
                                     IndexIcon columns, for joining onto EVERY player
                                     row regardless of rating band (broad market context)

The "momentum_pctile_90" column mimics futbin's own 0-100 "Market Momentum" gauge idea:
where does today's day-over-day index change rank against the last 90 days of changes.
~0 = deeply negative move relative to recent history, ~100 = strongly positive.

Run: python3 build_index_features.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def momentum_percentile(s, window=90):
    def pct_rank_last(x):
        if len(x) < 5:
            return np.nan
        return (x < x[-1]).sum() / (len(x) - 1) * 100
    return s.rolling(window, min_periods=10).apply(pct_rank_last, raw=True)


def main():
    idx = pd.read_parquet(DATA_DIR / "indices.parquet").sort_values(["band", "platform", "date"])

    grp = idx.groupby(["band", "platform"], sort=False)["index_value"]
    idx["idx_pct_change_1d"] = grp.transform(lambda s: s.pct_change(1))
    idx["idx_pct_change_7d"] = grp.transform(lambda s: s.pct_change(7))
    idx["idx_pct_change_14d"] = grp.transform(lambda s: s.pct_change(14))
    idx["idx_pct_change_30d"] = grp.transform(lambda s: s.pct_change(30))
    idx["idx_ma_7"] = grp.transform(lambda s: s.rolling(7, min_periods=3).mean())
    idx["idx_ma_30"] = grp.transform(lambda s: s.rolling(30, min_periods=7).mean())
    idx["idx_momentum_pctile_90"] = idx.groupby(["band", "platform"], sort=False)["idx_pct_change_1d"].transform(
        momentum_percentile
    )

    idx.to_parquet(DATA_DIR / "indices_features.parquet", index=False)
    print(f"Saved indices_features.parquet: {len(idx)} rows")

    macro_cols = ["platform", "date", "index_value", "idx_pct_change_7d", "idx_pct_change_14d",
                  "idx_pct_change_30d", "idx_momentum_pctile_90"]

    idx100 = idx[idx["band"] == "100"][macro_cols].rename(
        columns={c: f"index100_{c}" for c in macro_cols if c not in ("platform", "date")}
    )
    idxicon = idx[idx["band"] == "icons"][macro_cols].rename(
        columns={c: f"iconsidx_{c}" for c in macro_cols if c not in ("platform", "date")}
    )

    macro = idx100.merge(idxicon, on=["platform", "date"], how="outer")
    macro.to_parquet(DATA_DIR / "macro_wide.parquet", index=False)
    print(f"Saved macro_wide.parquet: {len(macro)} rows, columns: {list(macro.columns)}")


if __name__ == "__main__":
    main()
