#!/usr/bin/env python3
"""
Per-player detail scraper: sales history + daily/live sale-price charts for
every player discovered by scrape_market_list.py, plus the FC27 Popular
players list (https://www.futbin.com/27/popular) for anything not already
covered by a market-list price bucket or squad.

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

  /27/player/{id}/{slug}  (used only for "popular-only" players, i.e. found
    on the Popular list but not caught by any market-list price bucket or
    squad -- for those we don't already have a price)
    Live price: .price.inline-with-icon.lowest-price-1
    Platform is a page-wide toggle (POST /change-platform, not a URL param
    like the sales page): a <button name="platform" value="ps|pc"> inside
    a <form>, with the currently-active one's child .og-radio carrying
    class "checked". We read the price, note which platform is currently
    active from that "checked" marker, click the other button (submits the
    form -> full reload), then read price again for the other platform.

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

PLAYER_PRICE_JS = """
(() => {
    const priceEl = document.querySelector('.price.inline-with-icon.lowest-price-1');
    const price = priceEl ? priceEl.textContent.trim() : null;
    let activePlatform = null;
    document.querySelectorAll('form[action="/change-platform"] button[name="platform"]').forEach(btn => {
        const radio = btn.querySelector('.og-radio');
        if (radio && radio.classList.contains('checked')) activePlatform = btn.getAttribute('value');
    });
    return JSON.stringify({ price, activePlatform });
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


async def evaluate_json(tab, js, default):
    raw = await tab.evaluate(js)
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
    await tab.get(POPULAR_URL)
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
    await tab.get(url)
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


async def fetch_popular_only_price(tab, pid, slug):
    """Player-page live price via the POST /change-platform toggle, for
    players we have no market-list price for at all."""
    url = f"{BASE}/27/player/{pid}/{slug}"
    await tab.get(url)
    first = await poll_for(tab, PLAYER_PRICE_JS, lambda r: r and r.get("price"), {"price": None, "activePlatform": None})
    prices = {}
    if first.get("activePlatform"):
        prices[first["activePlatform"]] = first.get("price")

    other = "pc" if first.get("activePlatform") == "ps" else "ps"
    clicked = await tab.evaluate(
        f"""(() => {{
            const btn = document.querySelector('form[action="/change-platform"] button[value="{other}"]');
            if (btn) {{ btn.click(); return true; }}
            return false;
        }})()"""
    )
    if clicked:
        await asyncio.sleep(1.5)
        second = await poll_for(tab, PLAYER_PRICE_JS, lambda r: r and r.get("price"), {"price": None, "activePlatform": None})
        if second.get("activePlatform"):
            prices[second["activePlatform"]] = second.get("price")

    return {"player_id": pid, "slug": slug, "price_ps": prices.get("ps"), "price_pc": prices.get("pc")}


async def worker(name, tab, queue, sales_f, popular_only_f, scraped_at):
    n_sales = n_prices = 0
    while True:
        async with out_lock:
            if not queue:
                return n_sales, n_prices
            job = queue.popleft()
        try:
            if job["type"] == "sales":
                rec = await fetch_sales_page(tab, job["player_id"], job["slug"], job["platform"])
                rec["scraped_at"] = scraped_at
                async with out_lock:
                    sales_f.write(json.dumps(rec) + "\n")
                    sales_f.flush()
                n_sales += 1
            else:  # popular_only_price
                rec = await fetch_popular_only_price(tab, job["player_id"], job["slug"])
                rec["scraped_at"] = scraped_at
                async with out_lock:
                    popular_only_f.write(json.dumps(rec) + "\n")
                    popular_only_f.flush()
                n_prices += 1
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
    popular_players = await scrape_popular_list(tabs[0])
    popular_only_ids = set(popular_players) - set(market_players)
    print(f"  {len(popular_only_ids)} players are popular-list-only (no market-list price)")

    all_players = dict(market_players)
    all_players.update(popular_players)

    jobs = deque()
    for pid, slug in all_players.items():
        for platform in PLATFORMS:
            jobs.append({"type": "sales", "player_id": pid, "slug": slug, "platform": platform})
    for pid in popular_only_ids:
        jobs.append({"type": "popular_only_price", "player_id": pid, "slug": all_players[pid]})

    print(f"Total jobs: {len(jobs)} ({len(all_players)} players x {len(PLATFORMS)} platforms for sales, "
          f"+ {len(popular_only_ids)} popular-only price lookups)")

    scraped_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    sales_path = SCRAPES_DIR / f"player_sales_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    popular_only_path = SCRAPES_DIR / f"popular_only_prices_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    print(f"Writing sales data to {sales_path}")
    print(f"Writing popular-only prices to {popular_only_path}")

    pbar = tqdm(total=len(jobs), desc="Player jobs", unit="job")
    with open(sales_path, "w", encoding="utf-8") as sales_f, \
         open(popular_only_path, "w", encoding="utf-8") as popular_only_f:
        results = await asyncio.gather(
            *[worker(i, tabs[i], jobs, sales_f, popular_only_f, scraped_at) for i in range(len(tabs))]
        )
    pbar.close()

    total_sales = sum(r[0] for r in results)
    total_prices = sum(r[1] for r in results)
    print(f"\nDone: {total_sales} sales pages, {total_prices} popular-only prices")
    try:
        browser.stop()
    except Exception:
        pass
    kill_chrome()


if __name__ == "__main__":
    asyncio.run(main())
