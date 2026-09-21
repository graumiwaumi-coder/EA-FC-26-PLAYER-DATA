#!/bin/bash
# One entry point for a full data refresh: runs all three scrapers in
# sequence, then merges everything into the parquet files the model will
# train on. Sequential on purpose -- never run these concurrently with each
# other or anything else heavy (the 11GB VPS has no swap; see PROJECT.md).
#
# Run: bash run_full_scrape.sh [num_tabs]   (defaults to 3)
set -e
cd "$(dirname "$0")"
NUM_TABS="${1:-3}"

echo "=== 1/6: market list (player universe + live prices) ==="
xvfb-run -a python3 scrape_market_list.py "$NUM_TABS"

echo "=== 2/6: market indices (rating-tier benchmarks + mover discovery) ==="
xvfb-run -a python3 scrape_market_indices.py

echo "=== 3/6: player history (metadata + daily-average price array) ==="
xvfb-run -a python3 scrape_player_history.py "$NUM_TABS"

echo "=== 4/6: player sales (real sales-history transactions) ==="
xvfb-run -a python3 scrape_player_details.py "$NUM_TABS"

echo "=== 5/6: merging everything into data/*.parquet ==="
cd ../analysis
python3 -u build_live_dataset.py

echo "=== 6/6: data quality report ==="
python3 -u data_quality_report.py

echo "=== DONE ==="
