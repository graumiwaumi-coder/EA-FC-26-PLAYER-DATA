"""
Grades past live_score.py predictions once enough time has passed to know the actual
outcome -- this is the model's real, live track record on FC27 data, not a backtest.
Meant to be re-run periodically (the dashboard's "Rebuild features & rescore" job runs
this automatically at the end; it's also safe to run standalone/on a cron).

For every stored prediction whose 21-day horizon has elapsed and hasn't been graded
yet, looks up the actual price the same (player_id, platform, game_version) had at (or
nearest before) eval_date in prices_long.parquet -- the raw, always-current price
table, not the derived feature files, so grading never depends on the heavier feature
pipeline having been rebuilt recently.

Run: python3 evaluate_predictions.py
"""
import pandas as pd

from predictions_db import load_unevaluated_due, mark_evaluated, DATA_DIR

SELL_TAX = 0.05


def main():
    due = load_unevaluated_due()
    if due.empty:
        print("No predictions due for evaluation.")
        return
    print(f"{len(due)} predictions due for evaluation (eval_date has passed, not yet graded)")

    prices = pd.read_parquet(DATA_DIR / "prices_long.parquet",
                              columns=["player_id", "platform", "date", "price", "game_version"])
    prices = prices.sort_values("date")

    graded_rows = []
    skipped_no_data = 0
    for pred in due.itertuples():
        sub = prices[
            (prices["player_id"] == pred.player_id) &
            (prices["platform"] == pred.platform) &
            (prices["game_version"] == pred.game_version) &
            (prices["date"] <= pred.eval_date) &
            (prices["price"] > 0)
        ]
        if sub.empty:
            skipped_no_data += 1
            continue
        actual_price = sub.iloc[-1]["price"]
        actual_return = (actual_price * (1 - SELL_TAX) - pred.price_at_snapshot) / pred.price_at_snapshot
        graded_rows.append({
            "id": pred.id,
            "actual_price_at_eval": float(actual_price),
            "actual_return": float(actual_return),
            "actual_up": int(actual_return > 0),
            "evaluated_at": pd.Timestamp.now().isoformat(),
        })

    if skipped_no_data:
        print(f"{skipped_no_data} predictions skipped -- no price data available yet at/before "
              f"their eval_date (card may have gone untradeable, or the price panel hasn't caught "
              f"up to that date yet)")

    if not graded_rows:
        print("None of the due predictions had usable price data to grade.")
        return

    graded_df = pd.DataFrame(graded_rows)
    mark_evaluated(graded_df)
    print(f"\nGraded {len(graded_df)} predictions.")
    print(f"Actual win rate (of newly graded): {graded_df['actual_up'].mean():.1%}")
    print(f"Actual mean return (of newly graded): {graded_df['actual_return'].mean():.1%}")


if __name__ == "__main__":
    main()
