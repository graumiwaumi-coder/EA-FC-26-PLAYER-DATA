# EA FC Ultimate Team Trading Assistant — Project Context

**Read this before doing anything else.** This is a living document, not a
one-time briefing — keep it updated as the project evolves. It exists so a
new chat (or a future you) can pick this project up without re-deriving
context the hard way.

**The person you're working with has no coding background.** Every
explanation needs to be plain English — no unexplained jargon, no assuming
familiarity with ML/statistics/trading terminology. This has held for the
entire project so far and should not change.

## What this project is

A personal tool that recommends Ultimate Team player-card trades in EA FC —
predicting which cards are likely to rise in price over a ~3-week window, so
the person can buy low and sell high. **The person always executes trades
manually** — this system recommends, it never places trades itself, and
that should not change without them explicitly asking for it.

## Goals and constraints (from a 20-question intake on 2026-09-20)

- **Success metric**: consistent % return on bankroll over time — not a
  jackpot swing, a steady edge.
- **Starting bankroll**: small, roughly 10-20k coins, expected to scale up
  from there. Position-sizing logic needs to work sensibly at this small
  scale, not just at a large hypothetical bankroll.
- **Risk tolerance**: aggressive. (Worth designing this as an explicit,
  visible choice — e.g. how aggressive Kelly-criterion position sizing gets
  — rather than silently picking a number.)
- **Trade execution**: always manual, never automated. The system's job
  stops at "here's what I'd do" — the person clicks the buttons in the EA
  app themselves.
- **New scope vs. the previous build**: the person wants **portfolio/holdings
  tracking and sell-timing recommendations**, not just buy recommendations.
  The system should know what they currently hold and advise when to sell
  it, not only what to buy next.
- **Platform**: PC only. (Console can be dropped from scope entirely —
  simplifies a lot of what the previous build had to handle.)
- **Card scope**: Gold+ rated cards (rating >= 75), including Icons/promos —
  let the model's own predictions sort quality rather than filtering card
  types out upfront.
- **Data refresh**: automated, a few times a day (not just an on-demand
  button — needs real scheduling, e.g. cron, on the VPS).
- **Retraining**: automated on a fixed schedule — **Wednesday and Sunday
  nights**.
- **Notifications/alerts**: not wanted right now. The person checks the
  dashboard about once a day; a push/email alert system is explicitly
  out of scope for now.
- **Data retention**: no concern about database size growing over time —
  storage is cheap, no cleanup policy needed.

## What to KEEP from the previous build

- **FC26 historical data**: millions of price data points across the full
  FC26 season. FC26 itself isn't being traded anymore, but this is the
  richest dataset available and is explicitly meant to be the foundation
  the new model learns from — don't discard it or treat it as irrelevant
  just because FC26 trading has stopped.
- **The FC27 scraper**: `scraper/scrape_player_history.py` — a Chrome-automation
  scraper (via the `nodriver` library, run through `xvfb-run` since the VPS
  has no display) that pulls FC27 player metadata + price history from
  futbin.com. This is the live data source going forward. Two other
  scrapers exist (`scrape_market_list.py`, `scrape_player_details.py`) but
  were never wired into the live pipeline — worth a fresh look at whether
  they're useful this time.
- **The VPS itself** and its constraints (see "Known risks" below).
- Possibly reusable as infrastructure (not "features and models," so likely
  fine to keep, but re-evaluate fit once the new design exists): the
  Streamlit dashboard shell, the background-job runner pattern
  (`analysis/job_runner.py`), and the SQLite-for-predictions pattern
  (`analysis/predictions_db.py`). These are UI/ops plumbing, not the
  model itself.

## What to REBUILD from scratch

**All feature engineering and the trained model itself.** The person
explicitly wants this built fresh, not inherited from the previous
pipeline's specific design choices (which technical indicators, which
model family, which validation scheme, etc.) — even if the new design
independently arrives at similar answers, it should get there on its own
reasoning, not by copying forward.

This means essentially everything currently under `analysis/` that builds
features or trains the model is a candidate for a ground-up redesign:
`build_player_features.py`, `build_features.py`, `build_features_v2.py`,
`similarity_engine.py`, `volatility_regime_analysis.py`, `train_model.py`,
`live_score.py`, and the hyperparameter tuning script. Treat the existing
versions as reference material to learn from, not a base to patch.

## Known risks from the previous build

**Important — the person specifically asked that these be verified against
whatever gets designed this time, not assumed to still apply.** They're
documented here as real, hard-won lessons, but a fresh architecture might
sidestep some of them entirely or reintroduce them in a different shape.
Check each one against the actual new design rather than copying the old
fix mechanically.

1. **FC26/FC27 player-ID collision.** The scraper assigns numeric player IDs
   per game, and they are **not** globally unique — roughly 85% of FC27
   card IDs happen to reuse an FC26 card's ID. Any code that joins or
   groups data by player ID across combined FC26+FC27 data needs to check
   game version too, or it will silently blend two completely unrelated
   cards' price histories together. This was a real, previously-undetected
   bug this session (caused by an incorrect assumption that IDs never
   collide) — verify whatever new design does before trusting its output.

2. **VPS memory ceiling.** The VPS has 11GB RAM and no swap. Two real
   out-of-memory crashes happened this session: once from running the
   scraper and a heavy Python job concurrently, once from a feature-building
   script using 64-bit floats throughout instead of 32-bit. Both are fixable
   (never run heavy jobs concurrently; downcast to float32 where precision
   allows), but any new pipeline working with the full combined price panel
   needs to be memory-conscious from the start, not discover this the hard
   way again.

3. **Kelly criterion formula.** If position sizing gets built again: the
   CORRECT generalized formula for a trade with a partial-loss outcome (not
   "lose everything") is `f* = p/a - q/b` where `p`=win probability,
   `q`=1-p, `a`=magnitude of a loss, `b`=magnitude of a win — NOT the
   textbook `f* = p - q/b` (which assumes 100% loss on failure, wrong for
   this domain where a losing trade usually still recovers most of its
   value). This was verified against known textbook cases during the
   previous build.

4. **Validation methodology.** An expanding-window or single train/test
   split produced misleadingly optimistic and inconsistent backtest
   results. Rolling-window walk-forward validation (train only on a fixed
   recent window, with an embargo gap so forward-looking labels don't leak
   across the train/test boundary) gave honest, stable numbers. Whatever
   validation approach gets designed this time should be checked for this
   same failure mode.

5. **EA's sell tax (5%)** applies to every sale and must be netted out of
   any return calculation — a trade that looks profitable gross can be a
   loser net of tax.

## Practical VPS / repo details

- **VPS**: Contabo, IP `173.249.46.127`, SSH as `root`.
- **Repo location on the VPS**: `~/EA-FC-26-PLAYER-DATA` (i.e.
  `/root/EA-FC-26-PLAYER-DATA`).
- **Python virtual environment**: `~/EA-FC-26-PLAYER-DATA/venv` — activate
  with `source venv/bin/activate` from the repo root (or
  `source ../venv/bin/activate` from inside `analysis/`).
- **GitHub repo**: `graumiwaumi-coder/EA-FC-26-PLAYER-DATA`.
- **Data files** live in `data/` (gitignored — VPS-only, not in the repo
  itself) as parquet files, plus a SQLite predictions database.
- Access is normally via SSH from a phone (Termius) or laptop terminal —
  expect to walk through commands step by step, confirming output at each
  stage, the same way this session operated.

## How the new chat should start

**Do not start coding immediately.** Propose a rebuild plan first — what
the new feature set and model will look like, what gets kept vs. rebuilt,
how portfolio/sell-tracking fits in, how the scheduled refresh/retrain will
be implemented — and get explicit approval before writing code. This
matches how the person wants to work: reviewed and agreed before it's
built, not built then explained.
