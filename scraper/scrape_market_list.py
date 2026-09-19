#!/usr/bin/env python3
"""
Scrape futbin's FC27 market-player-list page for Squads-version (promo) players.

Confirmed page structure (inspected directly in DevTools, 2026-09):
  tr.player-row
    td.market-table-name -> a.table-player-name (name + href like /27/player/21977/slug)
                          -> div.table-player-revision (card version text)
    td.table-price / table-updated / table-trend / table-ea-average /
       table-difference / table-ea-tax, each with BOTH .platform-ps-only AND
       .platform-pc-only variants present in the SAME page load -- no platform
       toggle needed, we grab both in one pass.

Squad filtering is URL-based: ?p_squad=<key>, pagination is ?page=<n>. Both
combine: ?p_squad=<key>&page=<n>.

SQUADS below is a manually-maintained list (checked directly in the site's
Version -> Squads filter dropdown) since new promo squads drop weekly and
there's no reliable way to auto-discover them without fragile UI-click
automation. Update this list periodically -- as of 2026-09-19 there is only
one (TeamOfTheWeek1), consistent with FC27's season having just started.

Run: python3 scrape_market_list.py
Requires: pip install nodriver tqdm
"""
import asyncio
import json
import re
import time
from pathlib import Path

import nodriver as uc
from nodriver import cdp
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent.resolve()
USER_DATA = SCRIPT_DIR / "chrome_profile"  # reuse the same profile as the existing scraper
OUT_DIR = SCRIPT_DIR / "scrapes"
OUT_DIR.mkdir(exist_ok=True)

BASE_URL = "https://www.futbin.com/27/market-player-list"

# Update this list as new promo squads drop (check Version -> Squads filter on the site)
SQUADS = [
    "TeamOfTheWeek1",
]

MAX_PAGES_PER_SQUAD = 50  # safety cap, stop earlier if a page comes back empty
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


def kill_chrome():
    import subprocess as _sp
    _sp.run("pkill -f chrome_profile", shell=True)


async def block_resources(tab):
    try:
        await tab.send(cdp.network.enable())
        await tab.send(cdp.network.set_blocked_urls(BLOCKED))
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


async def scrape_squad(tab, squad, out_f, scraped_at):
    total_rows = 0
    for page in range(1, MAX_PAGES_PER_SQUAD + 1):
        url = f"{BASE_URL}?p_squad={squad}&page={page}"
        await tab.get(url)
        rows = await extract_rows(tab)
        if not rows:
            print(f"  squad={squad} page={page}: no rows, stopping pagination")
            break
        for r in rows:
            r["player_id"] = parse_player_id(r.get("player_url"))
            r["squad"] = squad
            r["scraped_at"] = scraped_at
            out_f.write(json.dumps(r) + "\n")
        out_f.flush()
        total_rows += len(rows)
        print(f"  squad={squad} page={page}: {len(rows)} rows (running total {total_rows})")
        await asyncio.sleep(sum(PAGE_DELAY) / 2)
    return total_rows


async def main():
    kill_chrome()
    await asyncio.sleep(2)
    config = uc.Config(user_data_dir=str(USER_DATA))
    browser = await uc.start(config=config, headless=False)
    tab = browser.main_tab
    await block_resources(tab)

    scraped_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    out_path = OUT_DIR / f"market_list_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    print(f"Writing to {out_path}")

    grand_total = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for squad in tqdm(SQUADS, desc="Squads"):
            n = await scrape_squad(tab, squad, out_f, scraped_at)
            grand_total += n

    print(f"\nDone: {grand_total} total rows across {len(SQUADS)} squad(s) -> {out_path}")
    try:
        browser.stop()
    except Exception:
        pass
    kill_chrome()


if __name__ == "__main__":
    asyncio.run(main())
