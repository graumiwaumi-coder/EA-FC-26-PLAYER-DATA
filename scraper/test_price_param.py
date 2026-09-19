#!/usr/bin/env python3
"""
One-off diagnostic: figure out whether futbin's price-range filter is
platform-specific (ps_price vs a separate pc_price param) or whether the
"ps_price" name is just legacy and it actually filters on whatever platform
is currently toggled server-side (e.g. via a cookie), or something else
entirely.

For each candidate URL we grab the same row data the real scraper grabs
(price_ps, price_pc for every row) and report how many rows actually land
inside the requested bucket for EACH platform's price. Whichever platform's
prices consistently fall inside the bucket tells us what the param controls.

Run: xvfb-run python3 test_price_param.py
"""
import asyncio
import json
from pathlib import Path

import nodriver as uc
from nodriver import cdp

SCRIPT_DIR = Path(__file__).parent.resolve()
USER_DATA = SCRIPT_DIR / "chrome_profile"
BASE_URL = "https://www.futbin.com/27/market-player-list"

LO, HI = 10000, 50000

CANDIDATE_URLS = {
    "ps_price": f"{BASE_URL}?ps_price={LO}-{HI}&page=1",
    "pc_price": f"{BASE_URL}?pc_price={LO}-{HI}&page=1",
}

BLOCKED = [
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp", "*.svg",
    "*.woff", "*.woff2", "*.ttf", "*.otf", "*.mp4", "*.mp3", "*.css",
    "*googletagmanager*", "*google-analytics*", "*doubleclick*",
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
        return {
            player_name: nameEl ? nameEl.textContent.trim() : null,
            price_ps: cell(row, 'table-price', 'ps'),
            price_pc: cell(row, 'table-price', 'pc'),
        };
    }));
})()
"""


def parse_price(s):
    if not s:
        return None
    s = s.strip().upper().replace(",", "")
    mult = 1
    if s.endswith("K"):
        mult, s = 1_000, s[:-1]
    elif s.endswith("M"):
        mult, s = 1_000_000, s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return None


def kill_chrome():
    import subprocess as _sp
    _sp.run("pkill -f chrome_profile", shell=True)


async def fetch(tab, label, url):
    print(f"\n=== {label}: {url} ===")
    await tab.get(url)
    await asyncio.sleep(3)
    raw = await tab.evaluate(EXTRACT_JS)
    if isinstance(raw, dict):
        raw = raw.get("value", "[]")
    try:
        rows = json.loads(raw)
    except Exception:
        rows = []
    print(f"  rows returned: {len(rows)}")
    n_ps_in_range = n_pc_in_range = 0
    for r in rows:
        ps = parse_price(r.get("price_ps"))
        pc = parse_price(r.get("price_pc"))
        if ps is not None and LO <= ps <= HI:
            n_ps_in_range += 1
        if pc is not None and LO <= pc <= HI:
            n_pc_in_range += 1
    print(f"  rows with price_ps in [{LO},{HI}]: {n_ps_in_range}/{len(rows)}")
    print(f"  rows with price_pc in [{LO},{HI}]: {n_pc_in_range}/{len(rows)}")
    for r in rows[:5]:
        print(f"    {r}")
    return rows


async def main():
    kill_chrome()
    await asyncio.sleep(2)
    config = uc.Config(user_data_dir=str(USER_DATA))
    browser = await uc.start(config=config, headless=False)
    tab = browser.main_tab
    try:
        await tab.send(cdp.network.enable())
        await tab.send(cdp.network.set_blocked_urls(BLOCKED))
    except Exception:
        pass

    for label, url in CANDIDATE_URLS.items():
        await fetch(tab, label, url)

    try:
        browser.stop()
    except Exception:
        pass
    kill_chrome()


if __name__ == "__main__":
    asyncio.run(main())
