"""
FC27 live-trading dashboard: the model's current BUY list (with exact
position sizes against a manually-entered bankroll), a separate SELL list
for currently-held cards, holdings entry/tracking, and the model's own
live track record graded against real outcomes over time.

No login -- runs open on whatever port it's started on. Low-value target
(nothing sensitive is exposed, worst case a stranger who finds the port
clicks a button that kicks off a scrape), traded off deliberately for not
having to re-enter a password on every refresh. Ask for a password gate
back anytime if that tradeoff ever stops making sense (e.g. a lighter
option is a token in the URL query string instead of a login form, so it
survives reloads).

Run (from the repo root, with the venv active):
    streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))

import recommendations_db as rdb  # noqa: E402
import job_runner as jr  # noqa: E402
from build_features import EA_SELL_TAX  # noqa: E402


def build_plan(buy, bankroll, max_positions=10):
    """Turns the ranked BUY list into an actual spending plan: walk down by
    opportunity-cost-adjusted return, buy what's affordable with whatever
    of the bankroll is still left, stop once funds run out or
    max_positions is hit. This is deliberately different from each row's
    own recommended_quantity/recommended_coins (which sizes that ONE card
    as if it had the whole bankroll to itself, so every row can be judged
    on its own merits) -- this is "spend your real, finite bankroll across
    these specific picks, in this order," which is what a person deciding
    what to actually go buy needs."""
    if buy.empty or bankroll <= 0:
        return pd.DataFrame(), 0.0
    ranked = buy.sort_values("effective_daily_return", ascending=False).reset_index(drop=True)
    remaining = bankroll
    picks = []
    for _, row in ranked.iterrows():
        if len(picks) >= max_positions:
            break
        price = row["price_at_snapshot"]
        if pd.isna(price) or price <= 0 or price > remaining:
            continue
        cap_coins = min(row["kelly_fraction"] * bankroll, remaining)
        qty = int(cap_coins // price)
        if qty < 1:
            continue
        cost = qty * price
        remaining -= cost
        expected_sell_price = price * (1 + row["predicted_return_pct"])
        expected_profit_per_copy = expected_sell_price * (1 - EA_SELL_TAX) - price
        picks.append({
            "url": row["url"], "position": row["position"], "club": row["club"],
            "horizon_days": row["horizon_days"], "buy_price": price, "quantity": qty,
            "coins_spent": cost, "predicted_win_prob": row["predicted_win_prob"],
            "expected_sell_price": expected_sell_price,
            "expected_profit": qty * expected_profit_per_copy,
            "confidence_flag": row["confidence_flag"],
        })
    return pd.DataFrame(picks), remaining

st.set_page_config(page_title="FC27 Live Trading", layout="wide")

st.title("FC27 Live Trading")

# "Refresh market data" runs the whole pipeline through to a fresh BUY/SELL
# list -- scoring happens automatically after every scrape, per spec, not
# as a separate manual step. "Rebuild features & rescore" re-derives
# features/predictions from whatever's already scraped, without spending
# time on a fresh scrape (e.g. after a bankroll change or a code fix).
REFRESH_CMD = ["bash", "-c",
    "cd scraper && xvfb-run -a python3 scrape_player_history.py && "
    "cd ../analysis && python3 -u build_live_dataset.py && "
    "python3 -u build_features.py && "
    "python3 -u score_predictions.py"]
REBUILD_CMD = ["bash", "-c",
    "cd analysis && "
    "python3 -u build_features.py && "
    "python3 -u score_predictions.py"]

# ------------------------------------------------------------------ action buttons
job = jr.current_job()

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    refresh_disabled = job is not None
    if st.button("Refresh market data", disabled=refresh_disabled, help="Scrapes the live FC27 "
                 "market, merges it in, rebuilds features, and re-scores -- the full pipeline, "
                 "end to end. Takes a while."):
        try:
            jr.start_job("refresh_market_data", REFRESH_CMD)
            st.rerun()
        except RuntimeError as e:
            st.error(str(e))
with col2:
    rebuild_disabled = job is not None
    if st.button("Rebuild features & rescore", disabled=rebuild_disabled, help="Rebuilds features "
                 "and re-scores against whatever's already scraped, without a fresh scrape. Use "
                 "this after changing the bankroll or adding/closing holdings."):
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
        st.caption("No job currently running. Model retraining runs separately, "
                   "on its own Wed/Sun schedule (see PROJECT.md) -- not a button here.")

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

# ------------------------------------------------------------------ bankroll
st.header("Bankroll")
st.caption("Entered manually and reused by every scoring run until changed here -- position "
           "sizes below are fully optimized against whatever this is currently set to.")
current_bankroll = rdb.get_bankroll()
c1, c2 = st.columns([1, 3])
with c1:
    new_bankroll = st.number_input("Available coins", min_value=0.0, value=float(current_bankroll),
                                    step=1000.0, format="%.0f")
    if st.button("Save bankroll"):
        rdb.set_bankroll(new_bankroll)
        st.success(f"Saved -- {new_bankroll:,.0f} coins. Click 'Rebuild features & rescore' to "
                   f"re-size the BUY list against it.")

st.divider()

# ------------------------------------------------------------------ today's plan
st.header("Today's plan")
st.caption("An actual spend plan, not a menu: your real bankroll, spent on the best-ranked picks "
           "it can afford, in order, until the money runs out. Buy these, hold for the horizon "
           "shown, then sell.")

buy_for_plan = rdb.load_latest_buy_list()
if buy_for_plan.empty:
    st.info("No BUY recommendations yet -- nothing to plan around.")
else:
    max_positions = st.slider("Max number of different cards in the plan", 1, 20, 10)
    plan, leftover = build_plan(buy_for_plan, current_bankroll, max_positions)
    if plan.empty:
        st.warning(f"Nothing in the current buy list is affordable at a {current_bankroll:,.0f}-coin "
                   f"bankroll. Cheapest candidate: {buy_for_plan['price_at_snapshot'].min():,.0f} coins.")
    else:
        st.dataframe(
            plan[["url", "position", "club", "horizon_days", "buy_price", "quantity", "coins_spent",
                  "predicted_win_prob", "expected_sell_price", "expected_profit", "confidence_flag"]],
            use_container_width=True, hide_index=True,
            column_config={
                "buy_price": st.column_config.NumberColumn("Buy at", format="%d"),
                "quantity": st.column_config.NumberColumn("Copies"),
                "coins_spent": st.column_config.NumberColumn("Coins spent", format="%d"),
                "predicted_win_prob": st.column_config.ProgressColumn(
                    "Win probability", min_value=0.0, max_value=1.0, format="%.0f%%"),
                "expected_sell_price": st.column_config.NumberColumn("Expected sell price", format="%d"),
                "expected_profit": st.column_config.NumberColumn("Expected profit (net of tax)", format="%d"),
                "horizon_days": st.column_config.NumberColumn("Hold for (days)"),
            },
        )
        total_spend = plan["coins_spent"].sum()
        total_profit = plan["expected_profit"].sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total coins spent", f"{total_spend:,.0f}")
        c2.metric("Left over", f"{leftover:,.0f}")
        c3.metric("Expected total profit (if it plays out)", f"{total_profit:,.0f}",
                   f"{(total_profit / total_spend):.1%}" if total_spend else None)
        st.caption("\"Expected profit\" assumes every pick hits its predicted return -- in reality "
                   "some will and some won't, which is exactly what the win probability column "
                   "tells you. Treat this as the plan's best case, weighted by how confident the "
                   "model is in each pick, not a guarantee.")

st.divider()

# ------------------------------------------------------------------ buy list
st.header("Buy list")
st.caption("Every PC/FC27 Gold+ card clearing the 60% confidence bar, ranked by opportunity-cost-"
           "adjusted daily return (predicted return divided by horizon + estimated days-to-sell) -- "
           "not just the highest headline number. Position sizes use fractional (0.75x) Kelly "
           "against the bankroll above.")

buy = rdb.load_latest_buy_list()
if buy.empty:
    st.warning("No BUY recommendations yet. Run 'Refresh market data' or 'Rebuild features & "
               "rescore' at least once.")
else:
    as_of = buy["snapshot_date"].max()
    st.caption(f"As of {as_of:%Y-%m-%d} -- {len(buy)} candidate(s)")

    c1, c2, c3 = st.columns(3)
    with c1:
        horizons_avail = sorted(buy["horizon_days"].unique().tolist())
        horizons_sel = st.multiselect("Horizon (days)", options=horizons_avail, default=horizons_avail)
    with c2:
        thin_ok = st.checkbox("Include thin-history (less certain) cards", value=True)
    with c3:
        affordable_only = st.checkbox("Only cards I can afford right now", value=False)

    view = buy[buy["horizon_days"].isin(horizons_sel)]
    if not thin_ok:
        view = view[view["confidence_flag"] != "thin_history"]
    if affordable_only:
        view = view[view["recommended_quantity"] > 0]
    view = view.sort_values("effective_daily_return", ascending=False)

    st.dataframe(
        view[["url", "rating", "position", "club", "league", "squad", "horizon_days",
              "price_at_snapshot", "predicted_win_prob", "predicted_return_pct",
              "est_days_to_sell", "effective_daily_return", "recommended_quantity",
              "recommended_coins", "confidence_flag", "rationale"]],
        use_container_width=True, hide_index=True,
        column_config={
            "predicted_win_prob": st.column_config.ProgressColumn(
                "Win probability", min_value=0.0, max_value=1.0, format="%.0f%%"),
            "predicted_return_pct": st.column_config.NumberColumn(
                "Predicted return (if it wins)", format="%.1f%%"),
            "effective_daily_return": st.column_config.NumberColumn(
                "Opportunity-cost-adjusted daily return", format="%.2f%%"),
            "price_at_snapshot": st.column_config.NumberColumn("Price", format="%d"),
            "recommended_coins": st.column_config.NumberColumn("Coins to invest", format="%d"),
            "est_days_to_sell": st.column_config.NumberColumn("Est. days to sell", format="%.1f"),
            "horizon_days": st.column_config.NumberColumn("Horizon (days)"),
        },
    )
    st.caption(f"Total coins across this filtered list: {view['recommended_coins'].sum():,.0f} "
               f"of {current_bankroll:,.0f} bankroll")
    st.download_button("Download as CSV", view.to_csv(index=False), file_name="fc27_buy_list.csv")

st.divider()

# ------------------------------------------------------------------ holdings + sell list
st.header("Holdings & sell list")

with st.expander("Add a holding (after you manually buy a card)"):
    with st.form("add_holding"):
        c1, c2, c3 = st.columns(3)
        with c1:
            h_player_id = st.number_input("Player ID (from the futbin URL)", min_value=0, step=1)
            h_quantity = st.number_input("Quantity bought", min_value=1, step=1, value=1)
        with c2:
            h_price = st.number_input("Price paid per copy", min_value=0.0, step=100.0)
            h_date = st.date_input("Buy date", value=pd.Timestamp.now().date())
        with c3:
            h_platform = st.selectbox("Platform", options=["pc"], index=0)
            h_notes = st.text_input("Notes (optional)")
        submitted = st.form_submit_button("Add holding")
        if submitted:
            rdb.add_holding(int(h_player_id), h_platform, "fc27", int(h_quantity), float(h_price),
                             str(h_date), h_notes or None)
            st.success("Holding added. Run 'Rebuild features & rescore' to get its current outlook.")
            st.rerun()

holdings = rdb.list_holdings(open_only=True)
if holdings.empty:
    st.caption("No open holdings.")
else:
    holdings = holdings.copy()
    holdings["cost_basis"] = holdings["quantity"] * holdings["buy_price_per_unit"]
    holdings["sell_signal"] = holdings["last_sell_signal"].map({1: "SELL", 0: "hold"}).fillna("not yet scored")
    st.dataframe(
        holdings[["id", "player_id", "quantity", "buy_price_per_unit", "buy_date", "last_price",
                  "last_win_prob", "last_horizon_days", "last_unrealized_return_pct", "sell_signal", "notes"]],
        use_container_width=True, hide_index=True,
        column_config={
            "last_win_prob": st.column_config.ProgressColumn(
                "Best remaining win probability", min_value=0.0, max_value=1.0, format="%.0f%%"),
            "last_unrealized_return_pct": st.column_config.NumberColumn(
                "Unrealized P&L (net of tax)", format="%.1f%%"),
            "buy_price_per_unit": st.column_config.NumberColumn("Bought at", format="%d"),
            "last_price": st.column_config.NumberColumn("Current price", format="%d"),
        },
    )

    with st.expander("Mark a holding as sold"):
        with st.form("close_holding"):
            close_id = st.number_input("Holding ID", min_value=0, step=1)
            close_price = st.number_input("Actual sell price per copy (optional)", min_value=0.0, step=100.0)
            close_submitted = st.form_submit_button("Mark sold / remove from holdings")
            if close_submitted:
                rdb.close_holding(int(close_id), close_price if close_price > 0 else None)
                st.success(f"Holding {int(close_id)} closed.")
                st.rerun()

sell_list = rdb.load_latest_sell_list()
st.subheader("Sell signals")
if sell_list.empty:
    st.caption("No holdings currently flagged SELL (best remaining win probability below 50%).")
else:
    as_of = sell_list["snapshot_date"].max()
    st.caption(f"As of {as_of:%Y-%m-%d} -- re-scored with the same 12 models used for the buy list.")
    st.dataframe(
        sell_list[["url", "holding_id", "price_at_snapshot", "predicted_win_prob", "horizon_days",
                   "unrealized_return_pct"]],
        use_container_width=True, hide_index=True,
        column_config={
            "predicted_win_prob": st.column_config.ProgressColumn(
                "Best remaining win probability", min_value=0.0, max_value=1.0, format="%.0f%%"),
            "unrealized_return_pct": st.column_config.NumberColumn(
                "Unrealized P&L (net of tax)", format="%.1f%%"),
            "price_at_snapshot": st.column_config.NumberColumn("Current price", format="%d"),
            "horizon_days": st.column_config.NumberColumn("Best remaining horizon (days)"),
        },
    )

st.divider()

# ------------------------------------------------------------------ track record
st.header("Track record")

counts = rdb.counts()
c1, c2, c3 = st.columns(3)
c1.metric("Total BUY recommendations made", counts["total"])
c2.metric("Graded so far", counts["graded"])
c3.metric("Pending (horizon not reached yet)", counts["pending"])

graded = rdb.load_graded()
if graded.empty:
    st.caption("Nothing graded yet -- a recommendation is only gradeable once its own horizon has "
               "elapsed and a real price point exists near that date. Check back as live "
               "predictions have time to play out.")
else:
    overall_win_rate = graded["actual_win"].mean()
    overall_mean_return = graded["actual_return"].mean()
    c1, c2 = st.columns(2)
    c1.metric("Actual win rate (all graded)", f"{overall_win_rate:.1%}")
    c2.metric("Actual mean return (all graded)", f"{overall_mean_return:.1%}")

    st.subheader("Calibration: does predicted confidence match actual outcomes?")
    bins = [0, 0.6, 0.7, 0.8, 0.9, 1.01]
    labels = ["60-70%", "70-80%", "80-90%", "90-100%", "100%"]
    graded = graded.copy()
    graded["predicted_bucket"] = pd.cut(graded["predicted_win_prob"], bins=bins, labels=labels, right=False)
    calib = graded.groupby("predicted_bucket", observed=True).agg(
        n=("actual_win", "size"), actual_win_rate=("actual_win", "mean"),
        actual_mean_return=("actual_return", "mean"),
    ).reset_index()
    st.dataframe(calib, use_container_width=True, hide_index=True,
                 column_config={"actual_win_rate": st.column_config.NumberColumn(format="%.1%"),
                                 "actual_mean_return": st.column_config.NumberColumn(format="%.1%")})
    st.caption("A well-calibrated model's 'actual_win_rate' column should roughly track its "
               "predicted bucket. This is graded on live FC27 outcomes, not a historical backtest -- "
               "treat early readings (few graded rows) with real caution, not full confidence.")

    with st.expander("All graded BUY recommendations"):
        st.dataframe(graded[["url", "snapshot_date", "horizon_days", "predicted_win_prob",
                              "predicted_return_pct", "actual_return", "actual_win"]],
                     use_container_width=True, hide_index=True)
