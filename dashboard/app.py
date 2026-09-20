"""
FC27 live-trading dashboard: what the model currently likes, the buttons to refresh
market data and re-run the pipeline, and the model's own track record graded against
real outcomes over time.

Password-gated (DASHBOARD_PASSWORD env var) since this runs on a public port and one
of its buttons can kick off real, resource-heavy work on the VPS. Note: this is a
plain-HTTP password check, a speed bump against casual access, not real security --
if that's ever a concern, put this behind HTTPS (e.g. a Caddy reverse proxy) later.

Run (from the repo root, with the venv active):
    streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501
"""
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))

import predictions_db as pdb  # noqa: E402
import job_runner as jr  # noqa: E402

st.set_page_config(page_title="FC27 Live Trading", layout="wide")


def check_password():
    app_password = os.environ.get("DASHBOARD_PASSWORD")
    if not app_password:
        st.error("DASHBOARD_PASSWORD environment variable is not set on the server -- "
                  "refusing to start without a password configured. Set it and restart "
                  "streamlit (see dashboard/README.md).")
        st.stop()
    if st.session_state.get("authenticated"):
        return
    st.title("FC27 Live Trading")
    pw = st.text_input("Password", type="password")
    if st.button("Log in") or pw:
        if pw == app_password:
            st.session_state["authenticated"] = True
            st.rerun()
        elif pw:
            st.error("Incorrect password")
    st.stop()


check_password()

st.title("FC27 Live Trading")

REFRESH_CMD = ["bash", "-c",
    "cd scraper && xvfb-run -a python3 scrape_player_history.py && "
    "cd ../analysis && python3 -u build_live_dataset.py"]
REBUILD_CMD = ["bash", "-c",
    "cd analysis && "
    "python3 -u build_player_features.py && "
    "python3 -u build_index_features.py && "
    "python3 -u build_features.py && "
    "python3 -u build_features_v2.py && "
    "python3 -u similarity_engine.py && "
    "python3 -u volatility_regime_analysis.py && "
    "python3 -u live_score.py && "
    "python3 -u evaluate_predictions.py"]

# ------------------------------------------------------------------ action buttons
job = jr.current_job()

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    refresh_disabled = job is not None
    if st.button("Refresh market data", disabled=refresh_disabled, help="Scrapes the live FC27 "
                 "market (Chrome automation, takes a while) and merges it into players.parquet "
                 "/ prices_long.parquet. Doesn't rebuild features or rescore by itself."):
        try:
            jr.start_job("refresh_market_data", REFRESH_CMD)
            st.rerun()
        except RuntimeError as e:
            st.error(str(e))
with col2:
    rebuild_disabled = job is not None
    if st.button("Rebuild features & rescore", disabled=rebuild_disabled, help="Rebuilds the "
                 "whole feature pipeline against the latest scraped data and re-scores the "
                 "live FC27 market. Run 'Refresh market data' first if you want fresher prices "
                 "included. Also grades any past predictions whose 21-day horizon has elapsed."):
        try:
            jr.start_job("rebuild_features_rescore", REBUILD_CMD)
            st.rerun()
        except RuntimeError as e:
            st.error(str(e))
with col3:
    if job is not None:
        st.info(f"Running: **{job['job_name']}** (started "
                f"{pd.Timestamp.fromtimestamp(job['started_at']):%Y-%m-%d %H:%M:%S})")
    else:
        st.caption("No job currently running.")

if job is not None:
    with st.expander("Live log", expanded=True):
        st.code(jr.job_log_tail(job["job_name"]) or "(no output yet)", language=None)
        st.caption("This page doesn't auto-refresh -- reload to see progress.")
elif (ROOT / "data" / "jobs" / "rebuild_features_rescore.log").exists() or \
     (ROOT / "data" / "jobs" / "refresh_market_data.log").exists():
    with st.expander("Last job's log"):
        for name in ("refresh_market_data", "rebuild_features_rescore"):
            tail = jr.job_log_tail(name)
            if tail:
                st.markdown(f"**{name}**")
                st.code(tail, language=None)

st.divider()

# ------------------------------------------------------------------ live opportunities
st.header("Current opportunities")

latest = pdb.load_latest_batch()
if latest.empty:
    st.warning("No predictions recorded yet. Run 'Rebuild features & rescore' at least once.")
else:
    as_of = latest["snapshot_date"].max()
    st.caption(f"As of {as_of:%Y-%m-%d} -- {len(latest)} (player, platform) predictions")

    c1, c2, c3 = st.columns(3)
    with c1:
        min_prob = st.slider("Minimum predicted win probability", 0.0, 1.0, 0.5, 0.05)
    with c2:
        platforms = st.multiselect("Platform", options=sorted(latest["platform"].unique()),
                                    default=sorted(latest["platform"].unique()))
    with c3:
        positions = st.multiselect("Position", options=sorted(latest["position"].dropna().unique()),
                                    default=sorted(latest["position"].dropna().unique()))

    view = latest[
        (latest["predicted_win_prob"] >= min_prob) &
        (latest["platform"].isin(platforms)) &
        (latest["position"].isin(positions))
    ].sort_values("predicted_win_prob", ascending=False)

    st.dataframe(
        view[["url", "platform", "rating", "position", "club", "league",
              "price_at_snapshot", "predicted_win_prob", "predicted_return_21d"]],
        use_container_width=True, hide_index=True,
        column_config={
            "predicted_win_prob": st.column_config.ProgressColumn(
                "Win probability", min_value=0.0, max_value=1.0, format="%.0f%%"),
            "predicted_return_21d": st.column_config.NumberColumn(
                "Predicted 21d return", format="%.1f%%"),
            "price_at_snapshot": st.column_config.NumberColumn("Price", format="%d"),
        },
    )
    st.download_button("Download as CSV", view.to_csv(index=False), file_name="fc27_opportunities.csv")

st.divider()

# ------------------------------------------------------------------ track record
st.header("Track record")

counts = pdb.counts()
c1, c2, c3 = st.columns(3)
c1.metric("Total predictions made", counts["total"])
c2.metric("Graded so far", counts["graded"])
c3.metric("Pending (horizon not reached yet)", counts["pending"])

graded = pdb.load_graded()
if graded.empty:
    st.caption("Nothing graded yet -- predictions are only gradeable 21 days after they're made. "
               "Check back once the earliest live predictions have had time to play out.")
else:
    overall_win_rate = graded["actual_up"].mean()
    overall_mean_return = graded["actual_return"].mean()
    c1, c2 = st.columns(2)
    c1.metric("Actual win rate (all graded)", f"{overall_win_rate:.1%}")
    c2.metric("Actual mean return (all graded)", f"{overall_mean_return:.1%}")

    st.subheader("Calibration: does predicted confidence match actual outcomes?")
    bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01]
    labels = ["<30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80%+"]
    graded = graded.copy()
    graded["predicted_bucket"] = pd.cut(graded["predicted_win_prob"], bins=bins, labels=labels, right=False)
    calib = graded.groupby("predicted_bucket", observed=True).agg(
        n=("actual_up", "size"), actual_win_rate=("actual_up", "mean"),
        actual_mean_return=("actual_return", "mean"),
    ).reset_index()
    st.dataframe(calib, use_container_width=True, hide_index=True,
                 column_config={"actual_win_rate": st.column_config.NumberColumn(format="%.1%"),
                                 "actual_mean_return": st.column_config.NumberColumn(format="%.1%")})
    st.caption("A well-calibrated model's 'actual_win_rate' column should roughly track its "
               "predicted bucket (the 60-70% row winning close to 60-70% of the time, etc). "
               "This is graded on live FC27 outcomes, not a historical backtest -- treat early "
               "readings (few graded predictions) with real caution, not full confidence.")

    with st.expander("All graded predictions"):
        st.dataframe(graded[["url", "platform", "snapshot_date", "predicted_win_prob",
                              "predicted_return_21d", "actual_return", "actual_up"]],
                     use_container_width=True, hide_index=True)
