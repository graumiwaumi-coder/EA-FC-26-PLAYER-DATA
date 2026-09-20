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

Pick a password and set it as an environment variable every time before starting
the app (the app refuses to start without one):

```bash
export DASHBOARD_PASSWORD='choose-something-only-you-know'
```

To avoid re-typing that every session, add it to `~/.bashrc` instead (replace the
placeholder first):

```bash
echo "export DASHBOARD_PASSWORD='choose-something-only-you-know'" >> ~/.bashrc
source ~/.bashrc
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

Then open `http://173.249.46.127:8501` in a browser (phone or laptop) and log in
with the password you set above.

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

The password check is plain HTTP with no encryption — good enough to keep
casual access out, but not real security (a password sent over an
unencrypted connection can in principle be intercepted). If that ever
matters, putting this behind a proper HTTPS reverse proxy (e.g. Caddy, which
gets you automatic free certificates with a couple of config lines) is a
reasonable next step, not something this v1 needed to solve.
