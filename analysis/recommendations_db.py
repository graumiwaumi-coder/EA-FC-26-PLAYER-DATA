"""
Shared SQLite store backing score_predictions.py and the dashboard: every
BUY/SELL recommendation the model has ever made (so its real live track
record can be graded later, per PROJECT.md's "remember and check it later"
requirement), the person's manually-entered holdings, and small persistent
settings (currently just the bankroll figure, since bankroll is typed in by
hand rather than tracked automatically -- see the 20-question scoring-script
intake, answers 8 and 20).

Replaces the old predictions_db.py -- that schema was single-horizon
(predicted_return_21d only) and had no notion of BUY vs SELL, position
sizing, or holdings, all of which the rebuilt 12-model system and the
scoring script's spec need. No live rows existed under the new model
before this file, so this is a clean schema, not a migration.

One recommendation row per (player_id, platform, game_version,
snapshot_date, horizon_days, action) -- re-running score_predictions.py
against a snapshot_date it's already scored is silently skipped (INSERT OR
IGNORE), so the table only grows on genuinely new data, not once per run.
"""
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "recommendations.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scored_at TEXT NOT NULL,
    action TEXT NOT NULL,              -- 'BUY' or 'SELL'
    player_id INTEGER NOT NULL,
    platform TEXT NOT NULL,
    game_version TEXT NOT NULL,
    url TEXT,
    rating REAL,
    position TEXT,
    club TEXT,
    league TEXT,
    squad TEXT,
    promo_family TEXT,
    snapshot_date TEXT NOT NULL,
    horizon_days INTEGER NOT NULL,
    eval_date TEXT NOT NULL,
    price_at_snapshot REAL,
    predicted_win_prob REAL NOT NULL,
    predicted_return_pct REAL,         -- expected return IF it wins (conditional regressor)
    est_days_to_sell REAL,
    effective_daily_return REAL,       -- opportunity-cost-adjusted ranking metric
    confidence_flag TEXT,              -- NULL/'normal' or 'thin_history'
    kelly_fraction REAL,
    recommended_coins REAL,
    recommended_quantity INTEGER,
    rationale TEXT,
    holding_id INTEGER,                -- for SELL rows: which holdings row this re-scores
    unrealized_return_pct REAL,        -- for SELL rows: P&L if sold right now, net of tax
    actual_price_at_eval REAL,
    actual_return REAL,
    actual_win INTEGER,
    evaluated_at TEXT,
    UNIQUE(player_id, platform, game_version, snapshot_date, horizon_days, action)
);
CREATE INDEX IF NOT EXISTS idx_reco_snapshot ON recommendations(snapshot_date, action);
CREATE INDEX IF NOT EXISTS idx_reco_eval ON recommendations(eval_date, evaluated_at);

CREATE TABLE IF NOT EXISTS holdings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    platform TEXT NOT NULL,
    game_version TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    buy_price_per_unit REAL NOT NULL,
    buy_date TEXT NOT NULL,
    notes TEXT,
    closed INTEGER NOT NULL DEFAULT 0,
    sell_price_per_unit REAL,
    closed_at TEXT,
    -- cached results of the most recent score_predictions.py re-score, so
    -- the dashboard can show every holding's current outlook without
    -- reloading the models/feature panel itself -- refreshed every run
    -- regardless of whether that run's outlook triggers a SELL flag.
    last_scored_at TEXT,
    last_price REAL,
    last_win_prob REAL,
    last_horizon_days INTEGER,
    last_predicted_return_pct REAL,
    last_unrealized_return_pct REAL,
    last_sell_signal INTEGER
);
CREATE INDEX IF NOT EXISTS idx_holdings_open ON holdings(closed);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""

REC_COLS = [
    "scored_at", "action", "player_id", "platform", "game_version", "url", "rating",
    "position", "club", "league", "squad", "promo_family", "snapshot_date", "horizon_days",
    "eval_date", "price_at_snapshot", "predicted_win_prob", "predicted_return_pct",
    "est_days_to_sell", "effective_daily_return", "confidence_flag", "kelly_fraction",
    "recommended_coins", "recommended_quantity", "rationale", "holding_id",
    "unrealized_return_pct",
]


def get_conn():
    DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    return conn


# ------------------------------------------------------------------ recommendations

def insert_recommendations(df):
    """df columns must be a superset of REC_COLS. Rows whose (player_id,
    platform, game_version, snapshot_date, horizon_days, action) already
    exists are silently skipped."""
    if df.empty:
        return 0
    conn = get_conn()
    rows = df[REC_COLS].where(pd.notnull(df[REC_COLS]), None).itertuples(index=False, name=None)
    placeholders = ", ".join(["?"] * len(REC_COLS))
    cur = conn.cursor()
    cur.executemany(
        f"INSERT OR IGNORE INTO recommendations ({', '.join(REC_COLS)}) VALUES ({placeholders})",
        list(rows),
    )
    n_inserted = cur.rowcount
    conn.commit()
    conn.close()
    return n_inserted


def load_latest_buy_list():
    conn = get_conn()
    df = pd.read_sql(
        "SELECT * FROM recommendations WHERE action = 'BUY' AND snapshot_date = "
        "(SELECT MAX(snapshot_date) FROM recommendations WHERE action = 'BUY') "
        "ORDER BY effective_daily_return DESC",
        conn, parse_dates=["scored_at", "snapshot_date", "eval_date", "evaluated_at"],
    )
    conn.close()
    return df


def load_latest_sell_list():
    conn = get_conn()
    df = pd.read_sql(
        "SELECT * FROM recommendations WHERE action = 'SELL' AND snapshot_date = "
        "(SELECT MAX(snapshot_date) FROM recommendations WHERE action = 'SELL') "
        "ORDER BY predicted_win_prob ASC",
        conn, parse_dates=["scored_at", "snapshot_date", "eval_date", "evaluated_at"],
    )
    conn.close()
    return df


def load_all(limit=None):
    conn = get_conn()
    q = "SELECT * FROM recommendations ORDER BY snapshot_date DESC"
    if limit:
        q += f" LIMIT {int(limit)}"
    df = pd.read_sql(q, conn, parse_dates=["scored_at", "snapshot_date", "eval_date", "evaluated_at"])
    conn.close()
    return df


def load_graded():
    conn = get_conn()
    df = pd.read_sql(
        "SELECT * FROM recommendations WHERE action = 'BUY' AND evaluated_at IS NOT NULL "
        "ORDER BY snapshot_date DESC",
        conn, parse_dates=["scored_at", "snapshot_date", "eval_date", "evaluated_at"],
    )
    conn.close()
    return df


def load_unevaluated_due(as_of=None):
    """BUY predictions whose eval_date has passed and haven't been graded
    yet -- SELL rows aren't graded the same way (a "sell" call isn't a
    return prediction to check), so this only looks at action='BUY'."""
    conn = get_conn()
    as_of = as_of or pd.Timestamp.now().normalize()
    df = pd.read_sql(
        "SELECT * FROM recommendations WHERE action = 'BUY' AND evaluated_at IS NULL "
        "AND eval_date <= ?",
        conn, params=[str(as_of.date())],
        parse_dates=["scored_at", "snapshot_date", "eval_date"],
    )
    conn.close()
    return df


def mark_evaluated(rows):
    """rows: DataFrame with id, actual_price_at_eval, actual_return, actual_win, evaluated_at."""
    if rows.empty:
        return
    conn = get_conn()
    cur = conn.cursor()
    cur.executemany(
        "UPDATE recommendations SET actual_price_at_eval=?, actual_return=?, actual_win=?, "
        "evaluated_at=? WHERE id=?",
        [(r.actual_price_at_eval, r.actual_return, int(r.actual_win), r.evaluated_at, int(r.id))
         for r in rows.itertuples()],
    )
    conn.commit()
    conn.close()


def counts():
    conn = get_conn()
    total = pd.read_sql("SELECT COUNT(*) AS n FROM recommendations WHERE action='BUY'", conn)["n"].iloc[0]
    graded = pd.read_sql(
        "SELECT COUNT(*) AS n FROM recommendations WHERE action='BUY' AND evaluated_at IS NOT NULL",
        conn)["n"].iloc[0]
    conn.close()
    return {"total": int(total), "graded": int(graded), "pending": int(total) - int(graded)}


# ------------------------------------------------------------------ holdings

def add_holding(player_id, platform, game_version, quantity, buy_price_per_unit, buy_date, notes=None):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO holdings (player_id, platform, game_version, quantity, buy_price_per_unit, "
        "buy_date, notes, closed) VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
        (player_id, platform, game_version, quantity, buy_price_per_unit, str(buy_date), notes),
    )
    conn.commit()
    holding_id = cur.lastrowid
    conn.close()
    return holding_id


def list_holdings(open_only=True):
    conn = get_conn()
    q = "SELECT * FROM holdings"
    if open_only:
        q += " WHERE closed = 0"
    q += " ORDER BY buy_date DESC"
    df = pd.read_sql(q, conn, parse_dates=["buy_date", "closed_at"])
    conn.close()
    return df


def close_holding(holding_id, sell_price_per_unit=None):
    conn = get_conn()
    conn.execute(
        "UPDATE holdings SET closed=1, sell_price_per_unit=?, closed_at=? WHERE id=?",
        (sell_price_per_unit, pd.Timestamp.now().isoformat(), int(holding_id)),
    )
    conn.commit()
    conn.close()


def delete_holding(holding_id):
    conn = get_conn()
    conn.execute("DELETE FROM holdings WHERE id=?", (int(holding_id),))
    conn.commit()
    conn.close()


def update_holdings_status(rows):
    """rows: DataFrame with id, last_scored_at, last_price, last_win_prob,
    last_horizon_days, last_predicted_return_pct, last_unrealized_return_pct,
    last_sell_signal. Called once per score_predictions.py run for every
    open holding that could be matched to current price/feature data."""
    if rows.empty:
        return
    conn = get_conn()
    cur = conn.cursor()
    cur.executemany(
        "UPDATE holdings SET last_scored_at=?, last_price=?, last_win_prob=?, last_horizon_days=?, "
        "last_predicted_return_pct=?, last_unrealized_return_pct=?, last_sell_signal=? WHERE id=?",
        [(r.last_scored_at, r.last_price, r.last_win_prob, int(r.last_horizon_days),
          r.last_predicted_return_pct, r.last_unrealized_return_pct, int(r.last_sell_signal), int(r.id))
         for r in rows.itertuples()],
    )
    conn.commit()
    conn.close()


# ------------------------------------------------------------------ settings

def get_setting(key, default=None):
    conn = get_conn()
    row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    conn.close()
    return row[0] if row else default


def set_setting(key, value):
    conn = get_conn()
    conn.execute("INSERT INTO settings (key, value) VALUES (?, ?) "
                 "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)))
    conn.commit()
    conn.close()


def get_bankroll(default=15000.0):
    val = get_setting("bankroll")
    return float(val) if val is not None else default


def set_bankroll(value):
    set_setting("bankroll", float(value))
