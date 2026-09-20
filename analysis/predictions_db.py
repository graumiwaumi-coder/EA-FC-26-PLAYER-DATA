"""
Shared SQLite store for every prediction live_score.py has ever made, so the model's
real live track record can be graded later (once enough time has passed to know what
actually happened) instead of only ever trusting historical backtests. This is the
"remember every recommendation and check it later" piece the dashboard needs.

One row per (player_id, platform, game_version, snapshot_date) -- re-running
live_score.py on a day where the underlying data hasn't moved forward just re-scores
the same snapshot and is silently skipped (INSERT OR IGNORE against the UNIQUE
constraint below), so the table only grows when there's genuinely new data to predict
from, not once per button click.
"""
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "predictions.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scored_at TEXT NOT NULL,
    player_id INTEGER NOT NULL,
    platform TEXT NOT NULL,
    game_version TEXT NOT NULL,
    url TEXT,
    rating REAL,
    position TEXT,
    club TEXT,
    league TEXT,
    snapshot_date TEXT NOT NULL,
    horizon_days INTEGER NOT NULL,
    eval_date TEXT NOT NULL,
    price_at_snapshot REAL,
    predicted_win_prob REAL NOT NULL,
    predicted_return_21d REAL NOT NULL,
    actual_price_at_eval REAL,
    actual_return REAL,
    actual_up INTEGER,
    evaluated_at TEXT,
    UNIQUE(player_id, platform, game_version, snapshot_date)
);
CREATE INDEX IF NOT EXISTS idx_predictions_snapshot ON predictions(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_predictions_eval ON predictions(eval_date, evaluated_at);
"""


def get_conn():
    DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    return conn


def insert_predictions(df):
    """df columns must match the schema, minus id/actual_*/evaluated_at. Rows whose
    (player_id, platform, game_version, snapshot_date) already exists are silently
    skipped -- that's a re-score of a snapshot we've already recorded a prediction for,
    not a new prediction."""
    conn = get_conn()
    cols = ["scored_at", "player_id", "platform", "game_version", "url", "rating", "position",
            "club", "league", "snapshot_date", "horizon_days", "eval_date", "price_at_snapshot",
            "predicted_win_prob", "predicted_return_21d"]
    rows = df[cols].itertuples(index=False, name=None)
    placeholders = ", ".join(["?"] * len(cols))
    cur = conn.cursor()
    cur.executemany(
        f"INSERT OR IGNORE INTO predictions ({', '.join(cols)}) VALUES ({placeholders})",
        rows,
    )
    n_inserted = cur.rowcount
    conn.commit()
    conn.close()
    return n_inserted


def load_latest_batch():
    """The most recent snapshot_date's predictions -- the "what does the model think
    right now" view the dashboard's opportunities table shows."""
    conn = get_conn()
    df = pd.read_sql(
        "SELECT * FROM predictions WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM predictions) "
        "ORDER BY predicted_win_prob DESC",
        conn, parse_dates=["scored_at", "snapshot_date", "eval_date", "evaluated_at"],
    )
    conn.close()
    return df


def load_all(limit=None):
    conn = get_conn()
    q = "SELECT * FROM predictions ORDER BY snapshot_date DESC"
    if limit:
        q += f" LIMIT {int(limit)}"
    df = pd.read_sql(q, conn, parse_dates=["scored_at", "snapshot_date", "eval_date", "evaluated_at"])
    conn.close()
    return df


def load_graded():
    conn = get_conn()
    df = pd.read_sql(
        "SELECT * FROM predictions WHERE evaluated_at IS NOT NULL ORDER BY snapshot_date DESC",
        conn, parse_dates=["scored_at", "snapshot_date", "eval_date", "evaluated_at"],
    )
    conn.close()
    return df


def load_unevaluated_due(as_of=None):
    """Predictions whose eval_date has passed and haven't been graded yet."""
    conn = get_conn()
    as_of = as_of or pd.Timestamp.now().normalize()
    df = pd.read_sql(
        "SELECT * FROM predictions WHERE evaluated_at IS NULL AND eval_date <= ?",
        conn, params=[str(as_of.date())],
        parse_dates=["scored_at", "snapshot_date", "eval_date"],
    )
    conn.close()
    return df


def mark_evaluated(rows):
    """rows: DataFrame with id, actual_price_at_eval, actual_return, actual_up, evaluated_at."""
    conn = get_conn()
    cur = conn.cursor()
    cur.executemany(
        "UPDATE predictions SET actual_price_at_eval=?, actual_return=?, actual_up=?, evaluated_at=? "
        "WHERE id=?",
        [(r.actual_price_at_eval, r.actual_return, int(r.actual_up), r.evaluated_at, int(r.id))
         for r in rows.itertuples()],
    )
    conn.commit()
    conn.close()


def counts():
    conn = get_conn()
    total = pd.read_sql("SELECT COUNT(*) AS n FROM predictions", conn)["n"].iloc[0]
    graded = pd.read_sql("SELECT COUNT(*) AS n FROM predictions WHERE evaluated_at IS NOT NULL", conn)["n"].iloc[0]
    conn.close()
    return {"total": int(total), "graded": int(graded), "pending": int(total) - int(graded)}
