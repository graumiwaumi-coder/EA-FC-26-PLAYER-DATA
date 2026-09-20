# FC27 Live Trading Dashboard

A small Streamlit app that shows what the model currently likes in the live FC27
market, lets you trigger a market-data refresh (scraper) and a full feature
rebuild + rescore, and tracks the model's real prediction accuracy over time.

## One-time setup on the VPS

```bash
cd ~/EA-FC-26-PLAYER-DATA
source venv/bin/activate
pip install streamlit
```

Open the port in the VPS firewall if one is active (Contabo VPSes often ship with
`ufw`):

```bash
sudo ufw allow 8501/tcp
```

## Running it

```bash
cd ~/EA-FC-26-PLAYER-DATA
source venv/bin/activate
streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501
```

Then open `http://173.249.46.127:8501` in a browser (phone or laptop). No login --
it runs open.

**This only runs while that terminal session is open.** If you close your SSH
session, the dashboard stops. To keep it running in the background:

```bash
nohup streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501 \
    > ~/EA-FC-26-PLAYER-DATA/data/dashboard.log 2>&1 &
disown
```

That starts it detached from your terminal, so it survives you disconnecting.
Check `~/EA-FC-26-PLAYER-DATA/data/dashboard.log` if something looks wrong, and
`pkill -f streamlit` to stop it.

## What the two buttons do

- **Refresh market data** — runs the FC27 scraper (real Chrome automation,
  takes a while) and merges the results into `players.parquet` /
  `prices_long.parquet`. Doesn't touch features or predictions by itself.
- **Rebuild features & rescore** — reruns the whole feature-engineering
  pipeline against whatever data currently exists, re-scores the live FC27
  market, and grades any past predictions whose 21-day horizon has passed.
  Run "Refresh market data" first if you want the freshest prices folded in.

Only one of these can run at a time (a shared lock prevents it) — running the
scraper and the feature pipeline at once caused an out-of-memory crash earlier
in this project on this same VPS.

## Security note

This dashboard has no login and runs open on a public port -- a deliberate
tradeoff (made 2026-09-20) to avoid re-entering a password on every page
refresh. Anyone who finds `173.249.46.127:8501` can view it and click its
buttons, including the ones that trigger scraping. Nothing sensitive (money,
credentials) is exposed through it, so the worst case is someone kicking off
an unwanted scrape run. If that tradeoff ever stops making sense, ask for a
password gate back, or for a lighter option like a token in the URL query
string (survives reloads, harder to guess than nothing).
