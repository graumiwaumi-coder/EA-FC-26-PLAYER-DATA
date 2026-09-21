#!/usr/bin/env python3
"""
Scrape futbin's FC27 Market Index data (Daily view) for both Console and PC,
across rating tiers plus Icons -- the FC27 equivalent of the FC26-era
scrape_indices.py. These are futbin's own published rating-tier indices
(e.g. "the 84-rated card index"), used as a market benchmark: a card quietly
beating its own tier's index over time is a stronger signal than one just
drifting up with the whole market.

Unlike the FC26 version (a one-time historical pull that overwrote a single
file), this writes a fresh timestamped snapshot every run, the same pattern
as the other three FC27 scrapers -- futbin's own index history is presumably
also rolling/limited, so our own accumulated snapshots are what preserve
the full picture over time.

Every page-navigation and page-read call is bounded with a timeout and
wrapped in error handling from the start, and the cookie-consent banner is
dismissed automatically -- lessons learned the hard way building the other
three scrapers today, applied here up front instead of discovered live.

Run: xvfb-run python3 scrape_market_indices.py
Requires: pip install nodriver tqdm
"""
import asyncio
import json
import time
from pathlib import Path

import nodriver as uc
from nodriver import cdp

SCRIPT_DIR = Path(__file__).parent.resolve()
USER_DATA = SCRIPT_DIR / "chrome_profile"
SCRAPES_DIR = SCRIPT_DIR / "scrapes"
SCRAPES_DIR.mkdir(exist_ok=True)

BASE = "https://www.futbin.com"

# Same rating tiers futbin published for FC26 -- unconfirmed whether FC27's
# market page exposes exactly the same set until this runs live.
INDICES = {
    "100": "100",
    "86": "86",
    "85": "85",
    "84": "84",
    "83": "83",
    "82": "82",
    "81": "81",
    "icons": "Icons",
}
PLATFORMS = ["console", "pc"]

GET_PAGE_TIMEOUT = 25
EVALUATE_TIMEOUT = 10
MAX_WAIT_SECONDS = 25

BLOCKED = [
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp", "*.svg",
    "*.woff", "*.woff2", "*.ttf", "*.otf", "*.mp4", "*.mp3", "*.css",
    "*googletagmanager*", "*google-analytics*", "*doubleclick*",
    "*facebook.net*", "*hotjar*", "*segment.io*", "*adsystem*",
    "*adnxs*", "*taboola*", "*outbrain*", "*criteo*", "*amazon-adsystem*",
]


def kill_chrome():
    import subprocess as _sp
    _sp.run("pkill -f chrome_profile", shell=True)


async def block_resources(tab):
    try:
        await tab.send(cdp.network.enable())
        await tab.send(cdp.network.set_blocked_urls(BLOCKED))
    except Exception:
        pass


async def evaluate_bounded(tab, js, default=None):
    """tab.evaluate() with no timeout can hang forever if the tab goes
    unresponsive -- confirmed live today on a different scraper's identical
    pattern. Every evaluate call here goes through this."""
    try:
        raw = await asyncio.wait_for(tab.evaluate(js), timeout=EVALUATE_TIMEOUT)
    except asyncio.TimeoutError:
        return default
    if isinstance(raw, dict):
        raw = raw.get("value", default)
    return raw


async def dismiss_cookie_banner(tab):
    await evaluate_bounded(tab, """
    (() => {
        const btn = document.querySelector('#onetrust-reject-all-handler') ||
                    Array.from(document.querySelectorAll('button')).find(b => b.textContent.trim() === 'Reject All') ||
                    Array.from(document.querySelectorAll('button')).find(b => b.textContent.trim() === 'I Accept');
        if (btn) { btn.click(); return true; }
        return false;
    })()
    """, default=False)


async def get_page(tab, url):
    await asyncio.wait_for(tab.get(url), timeout=GET_PAGE_TIMEOUT)
    await dismiss_cookie_banner(tab)


async def click_daily(tab):
    """Robustly clicks the Daily radio input inside its label."""
    return await evaluate_bounded(tab, """
    (() => {
        const labels = Array.from(document.querySelectorAll('label'));
        const dailyLabel = labels.find(l => l.textContent.trim().includes('Daily'));
        if (!dailyLabel) return "label_not_found";

        const radioInput = dailyLabel.querySelector('input[type="radio"]');
        if (radioInput && radioInput.checked) return "already_active";

        if (radioInput) {
            radioInput.click();
            return "clicked_radio";
        } else {
            dailyLabel.click();
            return "clicked_label";
        }
    })()
    """, default="timed_out")


async def click_platform(tab, platform):
    """Robustly clicks the Console/PC toggle."""
    return await evaluate_bounded(tab, f"""
    (() => {{
        const elements = Array.from(document.querySelectorAll('button, div[role="button"], span, a, label'));
        const btn = elements.find(el => el.textContent.trim().toLowerCase() === "{platform}" && el.offsetParent !== null);

        if (!btn) return "not_found";

        const isActive = btn.classList.contains('active') ||
                         (btn.parentElement && btn.parentElement.classList.contains('active')) ||
                         (btn.style.backgroundColor && btn.style.backgroundColor !== 'transparent' && btn.style.backgroundColor !== 'rgba(0, 0, 0, 0)');

        if (isActive) return "already_active";

        btn.click();
        return "clicked";
    }})()
    """, default="timed_out")


async def get_highcharts_data(tab):
    """Extracts the raw data array from the *visible* Highcharts instance."""
    raw = await evaluate_bounded(tab, """
    (() => {
        if (!window.Highcharts || !window.Highcharts.charts) {
            return JSON.stringify({ error: "Highcharts not found" });
        }

        let visibleChart = null;
        window.Highcharts.charts.forEach(chart => {
            if (chart && chart.renderTo && chart.renderTo.offsetParent !== null) {
                visibleChart = chart;
            }
        });

        if (!visibleChart) return JSON.stringify({ error: "No visible chart found" });

        let series = visibleChart.series[0];
        if (!series) return JSON.stringify({ error: "No series found" });

        let data = null;

        if (series.points && series.points.length > 0) {
            data = series.points.map(p => [p.x, p.y]);
        } else if (series.options && series.options.data) {
            data = series.options.data.map(pt => Array.isArray(pt) ? pt : [pt.x, pt.y]);
        }

        if (!data || data.length === 0) return JSON.stringify({ error: "No data points found" });

        return JSON.stringify({ count: data.length, data: data });
    })()
    """, default=None)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


async def extract_index_data(tab, index_name, platform):
    slug = INDICES.get(index_name, index_name)
    url = f"{BASE}/27/market/{slug}" if index_name != "100" else f"{BASE}/27/market"

    print(f"[{platform.upper()}] Loading Index {index_name}...")
    await get_page(tab, url)
    await asyncio.sleep(5)

    platform_status = await click_platform(tab, platform)
    print(f"  [UI] Platform toggle status: {platform_status}")
    if platform_status == "clicked":
        await asyncio.sleep(3)

    daily_status = await click_daily(tab)
    print(f"  [UI] Daily toggle status: {daily_status}")

    if daily_status not in ["clicked_radio", "clicked_label", "already_active"]:
        print("  [ERROR] Could not click Daily toggle. Extraction will fail.")
        return None

    print("  [UI] Waiting for Highcharts to redraw to Daily view...")
    data_res = None
    for attempt in range(10):
        await asyncio.sleep(2)
        data_res = await get_highcharts_data(tab)

        if data_res and data_res.get("count", 0) < 1000:
            print(f"  [Success] Daily data loaded ({data_res['count']} points).")
            break
        elif data_res:
            print(f"  [Wait] Still seeing {data_res['count']} points (Live view). Retrying...")

    if not data_res or data_res.get("count", 0) >= 1000:
        print("  [ERROR] Failed to load Daily data. It might be stuck on Live view.")
        return None

    return {
        "index": index_name,
        "platform": platform,
        "source": "Highcharts",
        "data": data_res["data"],
    }


async def launch_browser():
    kill_chrome()
    await asyncio.sleep(3)
    config = uc.Config(user_data_dir=str(USER_DATA))
    browser = await uc.start(config=config, headless=False)
    tab = browser.main_tab
    await block_resources(tab)
    return browser, tab


async def main():
    out_path = SCRAPES_DIR / f"market_indices_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    out_f = open(out_path, "w", encoding="utf-8")

    browser, tab = await launch_browser()
    print(f"Writing to {out_path}")

    scraped_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    ok, err = 0, 0
    for index_name in INDICES.keys():
        for platform in PLATFORMS:
            try:
                data = await extract_index_data(tab, index_name, platform)
                if data and data.get("data"):
                    data["scraped_at"] = scraped_at
                    out_f.write(json.dumps(data) + "\n")
                    out_f.flush()
                    points = len(data["data"])
                    print(f"OK: Index {index_name} ({platform}) - {points} points\n")
                    ok += 1
                else:
                    print(f"FAILED: Index {index_name} ({platform})\n")
                    err += 1
            except Exception as e:
                print(f"ERROR on Index {index_name} ({platform}): {str(e)[:80]}\n")
                err += 1

            await asyncio.sleep(2)

    out_f.close()
    print(f"\nDone: {ok} ok, {err} failed -> {out_path}")
    try:
        browser.stop()
    except Exception:
        pass
    kill_chrome()


if __name__ == "__main__":
    asyncio.run(main())
