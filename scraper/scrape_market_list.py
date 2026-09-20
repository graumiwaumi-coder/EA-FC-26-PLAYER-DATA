#!/usr/bin/env python3
"""
Scrape futbin's FC27 market-player-list page for live prices.

Two coverage passes, both writing to the same output file:

  1. SQUAD pass -- current Squads/promo versions (?p_squad=<key>), so we
     always have an explicit, named catch of promo cards.
  2. PRICE-RANGE pass -- full market coverage. The market-player-list page
     caps how many pages it will show for any single query, so to see
     every currently-listed player we bucket by price and sweep each
     bucket separately (?ps_price=<lo>-<hi> / ?pc_price=<lo>-<hi>).

     Confirmed live (2026-09-19) via test_price_param.py: ps_price and
     pc_price are genuinely independent per-platform filters -- e.g.
     ps_price=10000-50000 returns rows where 100% have price_ps in that
     bucket, but only ~62% also have price_pc in that same bucket (the two
     economies diverge). So one platform's buckets alone would already
     catch ~all tradeable players (every row still carries BOTH prices
     regardless of which param filtered it) -- we sweep both anyway to
     also catch anything listed on only one platform's market.

Both passes are flattened into one job queue and run across several
parallel browser tabs (same pattern as rebuild_all.py: one Chrome process,
NUM_TABS tabs, each tab claims and fully paginates one job at a time from
a shared queue) since each job (a squad or a price bucket) is independent
of every other job -- only pages *within* one job must be sequential.

Confirmed page structure (inspected directly in DevTools, 2026-09):
  tr.player-row
    td.market-table-name -> a.table-player-name (name + href like /27/player/21977/slug)
                          -> div.table-player-revision (card version text)
    td.table-price / table-updated / table-trend / table-ea-average /
       table-difference / table-ea-tax, each with BOTH .platform-ps-only AND
       .platform-pc-only variants present in the SAME page load -- no platform
       toggle needed, we grab both in one pass.

SQUADS below is a manually-maintained list (checked directly in the site's
Version -> Squads filter dropdown) since new promo squads drop weekly and
there's no reliable way to auto-discover them without fragile UI-click
automation. Update this list periodically -- as of 2026-09-19 there is only
one (TeamOfTheWeek1), consistent with FC27's season having just started.

Run: xvfb-run python3 scrape_market_list.py [num_tabs]
Requires: pip install nodriver tqdm
"""
import asyncio
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
USER_DATA = SCRIPT_DIR / "chrome_profile"  # reuse the same profile as the existing scraper
OUT_DIR = SCRIPT_DIR / "scrapes"
OUT_DIR.mkdir(exist_ok=True)

BASE_URL = "https://www.futbin.com/27/market-player-list"

NUM_TABS = int(sys.argv[1]) if len(sys.argv) > 1 else 6

# Update this list as new promo squads drop (check Version -> Squads filter on the site)
SQUADS = [
    "TeamOfTheWeek1",
]

# Buckets cover 5k and up -- below that is fodder too cheap to be worth a
# 5% EA sell tax round-trip even on a bulk flip. Keep buckets narrow enough
# that no single bucket is likely to blow past MAX_PAGES_PER_RANGE; a
# warning is printed (and the bucket recorded in the log) if one does, so
# it can be split further later based on real run data.
PRICE_RANGES = [
    (5000, 7500), (7500, 10000),
    (10000, 15000), (15000, 20000),
    (20000, 35000), (35000, 50000), (50000, 75000), (75000, 100000),
    (100000, 150000), (150000, 200000), (200000, 350000), (350000, 500000),
    (500000, 750000), (750000, 1000000), (1000000, 1500000),
    (1500000, 2500000), (2500000, 5000000), (5000000, 15000000),
]
PLATFORMS = ["ps", "pc"]

MAX_PAGES_PER_SQUAD = 50   # safety cap, stop earlier if a page comes back empty
MAX_PAGES_PER_RANGE = 50   # same cap for a single price bucket
PAGE_DELAY = (0.6, 1.2)
MAX_WAIT_SECONDS = 15
POLL_INTERVAL = 0.5

BLOCKED = [
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp", "*.svg",
    "*.woff", "*.woff2", "*.ttf", "*.otf", "*.mp4", "*.mp3", "*.css",
    "*googletagmanager*", "*google-analytics*", "*doubleclick*",
    "*facebook.net*", "*hotjar*", "*segment.io*", "*adsystem*",
    "*adnxs*", "*taboola*", "*outbrain*", "*criteo*", "*amazon-adsystem*",
]

EXTRACT_JS = """
(() => {
    function cell(row, cls, platform) {
        const el = row.querySelector(`.${cls}.platform-${platform}-only`);
        return el ? el.textContent.trim() : null;
    }
    const rows = document.querySelectorAll('tr.player-row');
    return JSON.stringify(Array.from(rows).map(row => {
        const nameEl = row.querySelector('a.table-player-name');
        const revEl = row.querySelector('.table-player-revision');
        return {
            player_name: nameEl ? nameEl.textContent.trim() : null,
            player_url: nameEl ? nameEl.getAttribute('href') : null,
            revision: revEl ? revEl.textContent.trim() : null,
            price_ps: cell(row, 'table-price', 'ps'),
            price_pc: cell(row, 'table-price', 'pc'),
            updated_ps: cell(row, 'table-updated', 'ps'),
            updated_pc: cell(row, 'table-updated', 'pc'),
            trend_ps: cell(row, 'table-trend', 'ps'),
            trend_pc: cell(row, 'table-trend', 'pc'),
            ea_average_ps: cell(row, 'table-ea-average', 'ps'),
            ea_average_pc: cell(row, 'table-ea-average', 'pc'),
            difference_ps: cell(row, 'table-difference', 'ps'),
            difference_pc: cell(row, 'table-difference', 'pc'),
            ea_tax_ps: cell(row, 'table-ea-tax', 'ps'),
            ea_tax_pc: cell(row, 'table-ea-tax', 'pc'),
        };
    }));
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
    kill. Wrapping it lets a single bad page raise instead of freezing the
    whole run (the caller's try/except already handles that)."""
    await asyncio.wait_for(tab.get(url), timeout=GET_PAGE_TIMEOUT)
    await dismiss_cookie_banner(tab)


async def dismiss_cookie_banner(tab):
    """Click through the cookie-consent banner if it's showing. Confirmed
    live via VNC: the market-list page's real content doesn't finish
    rendering underneath this banner, which is why every row extraction
    was silently coming back empty -- not a CAPTCHA or a bot-block."""
    try:
        await tab.evaluate("""
        (() => {
            const btn = document.querySelector('#onetrust-reject-all-handler') ||
                        Array.from(document.querySelectorAll('button')).find(b => b.textContent.trim() === 'Reject All') ||
                        Array.from(document.querySelectorAll('button')).find(b => b.textContent.trim() === 'I Accept');
            if (btn) { btn.click(); return true; }
            return false;
        })()
        """)
    except Exception:
        pass


def parse_player_id(url):
    if not url:
        return None
    m = re.search(r"/27/player/(\d+)/", url)
    return int(m.group(1)) if m else None


async def extract_rows(tab):
    """Poll until the table has actually rendered rows (or times out)."""
    elapsed = 0.0
    while elapsed < MAX_WAIT_SECONDS:
        raw = await tab.evaluate(EXTRACT_JS)
        if isinstance(raw, dict):
            raw = raw.get("value", "[]")
        try:
            rows = json.loads(raw)
        except Exception:
            rows = []
        if rows:
            return rows
        await asyncio.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
    return []


def build_jobs():
    """Flatten the squad pass and price-range pass into one list of independent jobs."""
    jobs = []
    for squad in SQUADS:
        jobs.append({
            "url_params": {"p_squad": squad},
            "label": f"squad={squad}",
            "max_pages": MAX_PAGES_PER_SQUAD,
            "source": "squad",
            "extra_fields": {"squad": squad},
        })
    for platform in PLATFORMS:
        param_name = f"{platform}_price"
        for lo, hi in PRICE_RANGES:
            jobs.append({
                "url_params": {param_name: f"{lo}-{hi}"},
                "label": f"{param_name}={lo}-{hi}",
                "max_pages": MAX_PAGES_PER_RANGE,
                "source": "price_range",
                "extra_fields": {"price_filter_platform": platform, "price_lo": lo, "price_hi": hi},
            })
    return jobs


async def run_job(tab, job, out_f, scraped_at):
    """Paginate a single filtered query (squad or price-bucket) until empty/capped.
    Any error (including a page that never finishes loading) is caught here
    so one bad page skips just this job instead of taking the whole run
    down via asyncio.gather -- gather cancels every other in-flight task the
    moment one worker raises, so an uncaught exception here would silently
    lose whatever every other tab was in the middle of, not just this job."""
    label = job["label"]
    total_rows = 0
    try:
        for page in range(1, job["max_pages"] + 1):
            params = dict(job["url_params"], page=page)
            qs = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"{BASE_URL}?{qs}"
            await get_page(tab, url)
            rows = await extract_rows(tab)
            if not rows and page == 1:
                # First page came back empty -- could be a genuinely empty
                # squad/bucket, or just a slow page load that outran
                # MAX_WAIT_SECONDS. One retry before we accept it as empty.
                await get_page(tab, url)
                rows = await extract_rows(tab)
            if not rows:
                if page == 1:
                    print(f"  NOTE: {label} returned 0 rows on page 1 (after a retry) -- "
                          f"either genuinely empty right now, or worth checking manually.")
                break
            async with out_lock:
                for r in rows:
                    r["player_id"] = parse_player_id(r.get("player_url"))
                    r["source"] = job["source"]
                    r["scraped_at"] = scraped_at
                    r.update(job["extra_fields"])
                    out_f.write(json.dumps(r) + "\n")
                out_f.flush()
            total_rows += len(rows)
            if page == job["max_pages"]:
                print(f"  WARNING: {label} hit MAX_PAGES cap ({job['max_pages']}) without an empty page -- "
                      f"this bucket/squad likely has more results than we saw. Consider narrowing it.")
            await asyncio.sleep(sum(PAGE_DELAY) / 2)
    except Exception as e:
        print(f"  ERROR on {label}: {str(e)[:80]} -- moving on to the next job "
              f"({total_rows} rows already saved from this one)")
    return total_rows


async def worker(name, tab, queue, out_f, scraped_at):
    total = 0
    while True:
        async with out_lock:
            if not queue:
                return total
            job = queue.popleft()
        n = await run_job(tab, job, out_f, scraped_at)
        total += n
        async with out_lock:
            pbar.update(1)
            pbar.set_postfix_str(f"tab{name} last={job['label']} rows={n} tab_total={total}")


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
    jobs = deque(build_jobs())
    scraped_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    out_path = OUT_DIR / f"market_list_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    print(f"Writing to {out_path}")
    print(f"Total jobs: {len(jobs)} (1 squad + {len(PLATFORMS)}x{len(PRICE_RANGES)} price buckets), {NUM_TABS} tabs")

    browser, tabs = await launch_browser()
    print(f"Launched {len(tabs)} tabs")

    pbar = tqdm(total=len(jobs), desc="Jobs", unit="job")
    with open(out_path, "w", encoding="utf-8") as out_f:
        grand_total = sum(await asyncio.gather(
            *[worker(i, tabs[i], jobs, out_f, scraped_at) for i in range(len(tabs))]
        ))
    pbar.close()

    print(f"\nDone: {grand_total} total rows -> {out_path}")
    try:
        browser.stop()
    except Exception:
        pass
    kill_chrome()


if __name__ == "__main__":
    asyncio.run(main())
