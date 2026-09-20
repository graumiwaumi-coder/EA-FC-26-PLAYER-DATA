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

echo "=== 1/4: market list (player universe + live prices) ==="
xvfb-run -a python3 scrape_market_list.py "$NUM_TABS"

echo "=== 2/4: player history (metadata + daily-average price array) ==="
xvfb-run -a python3 scrape_player_history.py "$NUM_TABS"

echo "=== 3/4: player sales (real sales-history transactions) ==="
xvfb-run -a python3 scrape_player_details.py "$NUM_TABS"

echo "=== 4/4: merging everything into data/*.parquet ==="
cd ../analysis
python3 -u build_live_dataset.py

echo "=== DONE ==="
