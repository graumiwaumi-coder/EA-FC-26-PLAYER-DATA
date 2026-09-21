#!/usr/bin/env python3
"""
Scrape futbin's FC27 Market Index pages for both Console (ps) and PC, across
rating tiers 81-86 plus Icons and the main "100" index. These are futbin's
own published rating-tier indices (e.g. "the 84-rated card index"), used as
a market benchmark: a card quietly beating its own tier's index over time is
a stronger signal than one just drifting up with the whole market.

Confirmed live via DevTools inspection (2026-09-21) on /27/market/84:

  - The chart's raw time-series is embedded directly in the DOM as JSON,
    no chart-library introspection or UI clicking needed:
      div.market-graph-platform-ps[data-graph-data]  -> "[[ts_ms, value], ...]"
      div.market-graph-platform-pc[data-graph-data]  -> same, for PC
    Both platforms' elements are present in the same page load (one visible,
    one hidden) -- same "both platforms in one page load" pattern already
    confirmed on the market-list page, so no platform-toggle clicking is
    needed here either.
  - Open/Lowest/Highest summary figures sit in similarly platform-suffixed
    blocks: div.market-main-index-summary...platform-ps-only /
    ...platform-pc-only.
  - Top Gainers / Top Losers (site-wide, not tier-specific -- ratings mix
    on every tier page) link to individual players: a[href^="/27/player/"].
  - Each tier page also has a "Top Index {N} Movers" sidebar list, tier-
    specific, via a.xl-row links.
  - A "Market Momentum" card gives a single site-wide score + label.

Player links found in Top Gainers/Losers/Movers are written to their own
output stream so scrape_player_history.py and scrape_player_details.py can
pull them into the main player universe -- these are exactly the kind of
players (recently moving fast) worth having full price/sales history for,
even if a market-list price bucket didn't happen to catch them.

Unlike the FC26 version (a one-time historical pull that overwrote a single
file), this writes a fresh timestamped snapshot every run, the same pattern
as the other three FC27 scrapers -- futbin's own index history is presumably
also rolling/limited, so our own accumulated snapshots are what preserve
the full picture over time.

Every page-navigation and page-read call is bounded with a timeout and
wrapped in error handling, and the cookie-consent banner is dismissed
automatically -- lessons learned the hard way building the other three
scrapers, applied here from the start instead of discovered live.

Run: xvfb-run python3 scrape_market_indices.py
Requires: pip install nodriver tqdm
"""
import asyncio
import json
import re
import time
from pathlib import Path

import nodriver as uc
from nodriver import cdp

SCRIPT_DIR = Path(__file__).parent.resolve()
USER_DATA = SCRIPT_DIR / "chrome_profile"
SCRAPES_DIR = SCRIPT_DIR / "scrapes"
SCRAPES_DIR.mkdir(exist_ok=True)

BASE = "https://www.futbin.com"

# Same rating tiers futbin published for FC26 -- confirmed 84 works live;
# the rest are assumed consistent (same site, same UI) until this runs.
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

GET_PAGE_TIMEOUT = 25
EVALUATE_TIMEOUT = 10
MAX_WAIT_SECONDS = 20
POLL_INTERVAL = 0.5

BLOCKED = [
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp", "*.svg",
    "*.woff", "*.woff2", "*.ttf", "*.otf", "*.mp4", "*.mp3", "*.css",
    "*googletagmanager*", "*google-analytics*", "*doubleclick*",
    "*facebook.net*", "*hotjar*", "*segment.io*", "*adsystem*",
    "*adnxs*", "*taboola*", "*outbrain*", "*criteo*", "*amazon-adsystem*",
]

EXTRACT_JS = r"""
(() => {
    function readGraph(suffix) {
        const el = document.querySelector(`.market-graph-platform-${suffix}[data-graph-data]`);
        if (!el) return null;
        try {
            return JSON.parse(el.getAttribute('data-graph-data'));
        } catch (e) { return null; }
    }

    function readSummary(suffix) {
        const el = document.querySelector(`.market-main-index-summary.platform-${suffix}-only`);
        return el ? el.innerText.replace(/\s+/g, ' ').trim() : null;
    }

    function sectionLinks(headingText) {
        const heading = Array.from(document.querySelectorAll('div,h2,h3,span'))
            .find(el => el.children.length === 0 && el.textContent.trim() === headingText);
        if (!heading) return [];
        let container = heading.parentElement;
        for (let i = 0; i < 5 && container; i++) {
            const links = container.querySelectorAll('a[href^="/27/player/"]');
            if (links.length > 0) return Array.from(links);
            container = container.parentElement;
        }
        return [];
    }

    function cardInfo(link) {
        // Climb one level at a time, keeping the last ancestor that still
        // contains exactly ONE player link -- stop the moment climbing
        // further would pull in a sibling card too. A fixed climb depth
        // (the previous approach) grabbed a shared container covering
        // multiple cards on this page, so every card in a section came
        // back with identical text -- confirmed live (2026-09-21).
        let container = link;
        let el = link;
        for (let i = 0; i < 8 && el.parentElement; i++) {
            el = el.parentElement;
            const n = el.querySelectorAll('a[href^="/27/player/"]').length;
            if (n > 1) break;
            container = el;
        }
        return {
            href: link.getAttribute('href'),
            text: container.innerText.replace(/\s+/g, ' ').trim().slice(0, 300),
        };
    }

    const gainers = sectionLinks('Top Gainers').map(cardInfo);
    const losers = sectionLinks('Top Losers').map(cardInfo);
    const movers = Array.from(document.querySelectorAll('a.xl-row[href^="/27/player/"]')).map(cardInfo);

    const momentumEl = Array.from(document.querySelectorAll('div,section'))
        .find(el => el.textContent.includes('MARKET MOMENTUM') || el.textContent.includes('Market Momentum'));
    const momentum_text = momentumEl ? momentumEl.innerText.replace(/\s+/g, ' ').trim().slice(0, 200) : null;

    return JSON.stringify({
        graph_ps: readGraph('ps'),
        graph_pc: readGraph('pc'),
        summary_ps: readSummary('ps'),
        summary_pc: readSummary('pc'),
        gainers: gainers,
        losers: losers,
        movers: movers,
        momentum_text: momentum_text,
    });
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


def parse_card(card):
    """card = {"href": "/27/player/123/slug", "text": "<raw card blob>"}.
    Pulls out whatever we can reliably get from the blob. Confirmed live:
    price is NOT reliably extractable here -- two adjacent figures (e.g. a
    quick-sell value and the actual listed price, "15M" and "24.75K") come
    through in the raw text with no separating space, so any regex/token
    guess either grabs a meaningless fragment of one of them or some other
    stray digit on the card (weak foot, skill moves). A wrong price is
    worse than no price, so this doesn't attempt one -- the real price
    gets captured reliably anyway once this player goes through the main
    scrape_player_history.py/scrape_player_details.py pipeline, which uses
    dedicated selectors rather than blob-guessing. Rating and % change are
    both reliably anchored (a plausible 75-99 token; a '%'-suffixed
    number) and are kept."""
    m = re.search(r"/27/player/(\d+)/([^/?]+)", card.get("href") or "")
    player_id, slug = (int(m.group(1)), m.group(2)) if m else (None, None)
    text = card.get("text") or ""

    rating = None
    for tok in text.split():
        if re.fullmatch(r"\d{2}", tok) and 40 <= int(tok) <= 99:
            rating = int(tok)
            break

    pct_m = re.search(r"([+-]?\d[\d.]*)%", text)
    return {
        "player_id": player_id,
        "slug": slug,
        "rating": rating,
        "price": None,
        "pct_change": float(pct_m.group(1)) if pct_m else None,
        "raw_text": text,
    }


async def extract_index_page(tab, index_name):
    slug = INDICES.get(index_name, index_name)
    url = f"{BASE}/27/market/{slug}" if index_name != "100" else f"{BASE}/27/market"

    print(f"Loading Index {index_name}...")
    await get_page(tab, url)

    elapsed = 0.0
    result = None
    while elapsed < MAX_WAIT_SECONDS:
        raw = await evaluate_bounded(tab, EXTRACT_JS, default=None)
        try:
            result = json.loads(raw) if raw else None
        except Exception:
            result = None
        if result and (result.get("graph_ps") or result.get("graph_pc")):
            break
        await asyncio.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

    if not result or not (result.get("graph_ps") or result.get("graph_pc")):
        print(f"  ERROR: no graph data found for index {index_name} after {MAX_WAIT_SECONDS}s")
        return None

    n_ps = len(result.get("graph_ps") or [])
    n_pc = len(result.get("graph_pc") or [])
    print(f"  OK: ps={n_ps} points, pc={n_pc} points, "
          f"{len(result.get('gainers') or [])} gainers, {len(result.get('losers') or [])} losers, "
          f"{len(result.get('movers') or [])} tier movers")
    return result


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
    players_path = SCRAPES_DIR / f"index_players_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    out_f = open(out_path, "w", encoding="utf-8")
    players_f = open(players_path, "w", encoding="utf-8")

    browser, tab = await launch_browser()
    print(f"Writing index data to {out_path}")
    print(f"Writing discovered players to {players_path}")

    scraped_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    ok, err = 0, 0
    seen_player_ids = set()

    for index_name in INDICES.keys():
        try:
            result = await extract_index_page(tab, index_name)
            if not result:
                err += 1
                await asyncio.sleep(2)
                continue

            for platform, key in (("ps", "graph_ps"), ("pc", "graph_pc")):
                points = result.get(key)
                if points:
                    out_f.write(json.dumps({
                        "index": index_name, "platform": platform,
                        "data": points, "scraped_at": scraped_at,
                    }) + "\n")

            for platform, key in (("ps", "summary_ps"), ("pc", "summary_pc")):
                summary = result.get(key)
                if summary:
                    out_f.write(json.dumps({
                        "index": index_name, "platform": platform,
                        "summary_text": summary, "scraped_at": scraped_at,
                    }) + "\n")
            out_f.flush()

            for section, cards in (("gainer", result.get("gainers") or []),
                                    ("loser", result.get("losers") or []),
                                    (f"tier_{index_name}_mover", result.get("movers") or [])):
                for card in cards:
                    parsed = parse_card(card)
                    if parsed["player_id"] is None:
                        continue
                    players_f.write(json.dumps({
                        "player_id": parsed["player_id"], "slug": parsed["slug"],
                        "rating": parsed["rating"], "price": parsed["price"],
                        "pct_change": parsed["pct_change"], "section": section,
                        "index": index_name, "scraped_at": scraped_at,
                        "raw_text": parsed["raw_text"],
                    }) + "\n")
                    seen_player_ids.add(parsed["player_id"])
            players_f.flush()

            momentum = result.get("momentum_text")
            if momentum:
                out_f.write(json.dumps({
                    "index": index_name, "platform": None,
                    "momentum_text": momentum, "scraped_at": scraped_at,
                }) + "\n")
                out_f.flush()

            ok += 1
        except Exception as e:
            print(f"ERROR on Index {index_name}: {str(e)[:80]}")
            err += 1

        await asyncio.sleep(2)

    out_f.close()
    players_f.close()
    print(f"\nDone: {ok} index pages ok, {err} failed, "
          f"{len(seen_player_ids)} unique players discovered -> {out_path}, {players_path}")
    try:
        browser.stop()
    except Exception:
        pass
    kill_chrome()


if __name__ == "__main__":
    asyncio.run(main())
