#!/usr/bin/env python3
"""
Per-player sales-history + daily/live sale-price chart scraper for every
player discovered by scrape_market_list.py, plus the FC27 Popular players
list (https://www.futbin.com/27/popular) for anything not already covered
by a market-list price bucket or squad.

(Meta + full price-history + current price live on the player's own page --
that's scrape_player_history.py, run separately. This script is purely the
/27/sales/ page: sales-history table + daily/live chart high/low/avg.)

Confirmed page structure (inspected directly in DevTools, 2026-09):

  /27/sales/{id}/{slug}?platform=ps|pc
    Both the "Live" and "Daily" chart sections are present in the DOM at
    once (just CSS-toggled by the on-page Live/Daily switch), so we don't
    need to click anything -- read both directly:
      .sales-graph-wrapper.live  -> "High: X" / "Low: Y" / "AVG: Z" text
      .sales-graph-wrapper.daily -> "High: X" / "Low: Y" text
    table.auctions-table -> the sales-history rows (date, listed for, sold
      for, ea tax, net price, buy-now/bid type -- exact columns read
      dynamically from <thead> so we don't hardcode an order that might be
      wrong). This is however many rows the page renders on load; we do not
      currently simulate scrolling the scrollable history box to force more
      to load (unconfirmed whether that's needed -- flag if row counts look
      suspiciously low vs. what the site shows visually).

  /27/popular
    a.playercard-wrapper[href^="/27/player/"] -> id + slug, same pattern as
    the market-list player links.

NOTE: this has NOT been confirmed to hit a JSON API under the hood -- it
scrapes the rendered page directly for every player x platform, which is
slow (expect this to take a while across 700-1500+ unique players). If a
faster JSON endpoint is found later (check DevTools Network/XHR tab while
loading a sales page), only fetch_sales_page() below needs to change.

Run: xvfb-run python3 scrape_player_details.py [num_tabs]
Requires: pip install nodriver tqdm
"""
import asyncio
import glob
import json
import re
import sys
import time
from collections import deque
from pathlib import Path

import nodriver as uc
from nodriver import cdp
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent.resolve()
USER_DATA = SCRIPT_DIR / "chrome_profile"
SCRAPES_DIR = SCRIPT_DIR / "scrapes"
SCRAPES_DIR.mkdir(exist_ok=True)

BASE = "https://www.futbin.com"
POPULAR_URL = f"{BASE}/27/popular"

NUM_TABS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
PLATFORMS = ["ps", "pc"]

MAX_WAIT_SECONDS = 15
POLL_INTERVAL = 0.5
PAGE_DELAY = (0.5, 1.0)

BLOCKED = [
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp", "*.svg",
    "*.woff", "*.woff2", "*.ttf", "*.otf", "*.mp4", "*.mp3", "*.css",
    "*googletagmanager*", "*google-analytics*", "*doubleclick*",
    "*facebook.net*", "*hotjar*", "*segment.io*", "*adsystem*",
    "*adnxs*", "*taboola*", "*outbrain*", "*criteo*", "*amazon-adsystem*",
]

SALES_EXTRACT_JS = """
(() => {
    function parseSection(sel) {
        const el = document.querySelector(sel);
        if (!el) return null;
        const text = el.innerText || '';
        const high = text.match(/High:\\s*([\\d,]+)/i);
        const low = text.match(/Low:\\s*([\\d,]+)/i);
        const avg = text.match(/AVG:\\s*([\\d,]+)/i);
        return {
            high: high ? high[1].replace(/,/g, '') : null,
            low: low ? low[1].replace(/,/g, '') : null,
            avg: avg ? avg[1].replace(/,/g, '') : null,
        };
    }
    const daily = parseSection('.sales-graph-wrapper.daily');
    const live = parseSection('.sales-graph-wrapper.live');

    const table = document.querySelector('table.auctions-table');
    let history = [];
    if (table) {
        const headers = Array.from(table.querySelectorAll('thead th'))
            .map(th => th.textContent.trim().toLowerCase().replace(/\\s+/g, '_') || null);
        const rows = table.querySelectorAll('tbody tr');
        history = Array.from(rows).map(tr => {
            const cells = Array.from(tr.querySelectorAll('td'));
            const obj = {};
            cells.forEach((td, i) => {
                const key = headers[i] || `col${i}`;
                obj[key] = td.textContent.trim();
            });
            return obj;
        });
    }
    return JSON.stringify({ daily, live, history, n_history_rows: history.length });
})()
"""

POPULAR_EXTRACT_JS = """
(() => {
    const links = document.querySelectorAll('a.playercard-wrapper[href^="/27/player/"]');
    return JSON.stringify(Array.from(links).map(a => a.getAttribute('href')));
})()
"""

out_lock = asyncio.Lock()
pbar = None


def kill_chrome():
    import subprocess as _sp
    _sp.run("pkill -f chrome_profile", shell=True)


async def block_resources(tab):
    try:
        await tab.send(cdp.network.enable())
        await tab.send(cdp.network.set_blocked_urls(BLOCKED))
    except Exception:
        pass


GET_PAGE_TIMEOUT = 25


async def get_page(tab, url):
    """tab.get() with no outer timeout can hang forever if a page load never
    fires its load event -- confirmed live: a run got stuck on its very last
    job with no error, no progress, nothing to Ctrl+C into except a full
    kill. Wrapping it lets a single bad page raise into the caller's
    existing try/except instead of freezing the whole run."""
    await asyncio.wait_for(tab.get(url), timeout=GET_PAGE_TIMEOUT)
    await dismiss_cookie_banner(tab)


async def dismiss_cookie_banner(tab):
    """Click through the cookie-consent banner if it's showing. Confirmed
    live via VNC on scrape_market_list.py's identical pattern: a page's
    real content doesn't finish rendering underneath this banner, which
    silently looked like an empty/blocked page rather than what it was."""
    try:
        await asyncio.wait_for(tab.evaluate("""
        (() => {
            const btn = document.querySelector('#onetrust-reject-all-handler') ||
                        Array.from(document.querySelectorAll('button')).find(b => b.textContent.trim() === 'Reject All') ||
                        Array.from(document.querySelectorAll('button')).find(b => b.textContent.trim() === 'I Accept');
            if (btn) { btn.click(); return true; }
            return false;
        })()
        """), timeout=EVALUATE_TIMEOUT)
    except Exception:
        pass


def parse_id_slug(url):
    if not url:
        return None, None
    m = re.search(r"/27/player/(\d+)/([^/?]+)", url)
    if not m:
        return None, None
    return int(m.group(1)), m.group(2)


def latest_market_list_file():
    files = sorted(glob.glob(str(SCRAPES_DIR / "market_list_*.jsonl")))
    return files[-1] if files else None


def load_market_players():
    path = latest_market_list_file()
    players = {}
    if not path:
        print("WARNING: no market_list_*.jsonl found -- run scrape_market_list.py first")
        return players
    print(f"Loading players from {path}")
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            pid, slug = parse_id_slug(r.get("player_url"))
            if pid is not None:
                players[pid] = slug
    print(f"  {len(players)} unique players from market list")
    return players


EVALUATE_TIMEOUT = 10


async def evaluate_json(tab, js, default):
    """tab.evaluate() with no outer timeout was the ACTUAL cause of the
    original hang (confirmed by that run's own traceback: it was stuck
    here, not in tab.get()) -- a page can load fine and still leave the
    tab unresponsive to CDP evaluate calls. Bounded the same way get_page()
    bounds navigation: a timeout here is treated as "not ready this
    iteration" so the caller's own poll loop keeps trying instead of the
    whole run freezing on one wedged tab."""
    try:
        raw = await asyncio.wait_for(tab.evaluate(js), timeout=EVALUATE_TIMEOUT)
    except asyncio.TimeoutError:
        return default
    if isinstance(raw, dict):
        raw = raw.get("value")
    try:
        return json.loads(raw)
    except Exception:
        return default


async def poll_for(tab, js, is_ready, default, max_wait=MAX_WAIT_SECONDS):
    elapsed = 0.0
    result = default
    while elapsed < max_wait:
        result = await evaluate_json(tab, js, default)
        if is_ready(result):
            return result
        await asyncio.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
    return result


async def scrape_popular_list(tab):
    await get_page(tab, POPULAR_URL)
    hrefs = await poll_for(tab, POPULAR_EXTRACT_JS, lambda r: bool(r), [])
    players = {}
    for href in hrefs:
        pid, slug = parse_id_slug(href)
        if pid is not None:
            players[pid] = slug
    print(f"  {len(players)} unique players from /27/popular")
    return players


async def fetch_sales_page(tab, pid, slug, platform):
    """One player x one platform's sales-history page. Swap this out first
    if a JSON API is ever confirmed to exist -- see module docstring."""
    url = f"{BASE}/27/sales/{pid}/{slug}?platform={platform}"
    await get_page(tab, url)
    data = await poll_for(
        tab, SALES_EXTRACT_JS,
        lambda r: r and (r.get("daily") or r.get("live") or r.get("history")),
        {"daily": None, "live": None, "history": [], "n_history_rows": 0},
    )
    return {
        "player_id": pid,
        "slug": slug,
        "platform": platform,
        "daily_high": data["daily"]["high"] if data.get("daily") else None,
        "daily_low": data["daily"]["low"] if data.get("daily") else None,
        "live_high": data["live"]["high"] if data.get("live") else None,
        "live_low": data["live"]["low"] if data.get("live") else None,
        "live_avg": data["live"]["avg"] if data.get("live") else None,
        "sales_history": data.get("history", []),
        "n_sales_history_rows": data.get("n_history_rows", 0),
    }


async def worker(name, tab, queue, sales_f, scraped_at):
    n_sales = 0
    while True:
        async with out_lock:
            if not queue:
                return n_sales
            job = queue.popleft()
        try:
            rec = await fetch_sales_page(tab, job["player_id"], job["slug"], job["platform"])
            rec["scraped_at"] = scraped_at
            async with out_lock:
                sales_f.write(json.dumps(rec) + "\n")
                sales_f.flush()
            n_sales += 1
        except Exception as e:
            async with out_lock:
                print(f"  tab{name} ERROR on {job}: {str(e)[:80]}")
        async with out_lock:
            pbar.update(1)
        await asyncio.sleep(sum(PAGE_DELAY) / 2)


async def launch_browser():
    kill_chrome()
    await asyncio.sleep(3)
    config = uc.Config(user_data_dir=str(USER_DATA))
    browser = await uc.start(config=config, headless=False)
    tabs = [browser.main_tab]
    for _ in range(NUM_TABS - 1):
        try:
            t = await browser.get("about:blank", new_tab=True)
            tabs.append(t)
        except Exception as e:
            print(f"[launch] tab failed: {str(e)[:60]}")
            break
    for t in tabs:
        await block_resources(t)
    return browser, tabs


async def main():
    global pbar
    browser, tabs = await launch_browser()
    print(f"Launched {len(tabs)} tabs")

    market_players = load_market_players()
    all_players = dict(market_players)
    try:
        popular_players = await scrape_popular_list(tabs[0])
        all_players.update(popular_players)
    except Exception as e:
        print(f"WARNING: /27/popular fetch failed ({str(e)[:80]}) -- "
              f"continuing with just the market-list players")
    print(f"Total unique players: {len(all_players)}")

    jobs = deque()
    for pid, slug in all_players.items():
        for platform in PLATFORMS:
            jobs.append({"player_id": pid, "slug": slug, "platform": platform})

    print(f"Total jobs: {len(jobs)} ({len(all_players)} players x {len(PLATFORMS)} platforms)")

    scraped_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    sales_path = SCRAPES_DIR / f"player_sales_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    print(f"Writing sales data to {sales_path}")

    pbar = tqdm(total=len(jobs), desc="Player jobs", unit="job")
    with open(sales_path, "w", encoding="utf-8") as sales_f:
        results = await asyncio.gather(
            *[worker(i, tabs[i], jobs, sales_f, scraped_at) for i in range(len(tabs))]
        )
    pbar.close()

    print(f"\nDone: {sum(results)} sales pages -> {sales_path}")
    try:
        browser.stop()
    except Exception:
        pass
    kill_chrome()


if __name__ == "__main__":
    asyncio.run(main())
