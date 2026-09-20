#!/usr/bin/env python3
"""
Per-player metadata + full price-history scraper for FC27 -- the FC27
equivalent of rebuild_all.py (which built futbin_all_rebuild.jsonl for
FC26). Covers every player in our FC27 universe: everyone found by
scrape_market_list.py plus everyone on /27/popular.

Confirmed live (2026-09) that /27/player/{id}/{slug} exposes a live price
via .price.inline-with-icon.lowest-price-1 (platform-toggled via a POST
/change-platform form). NOT yet confirmed whether the page still embeds
full price-history arrays the way FC26's data-pc-data/data-ps-data
attributes did -- we try those selectors first (and their /27/-scoped
variants, since futbin's CSS/anchor patterns looked year-scoped in FC26,
e.g. a[href*="/26/players?"]) and fall back to /26/-style ones. If neither
exists we still capture whatever meta text is on the page plus the single
current price via the toggle, and log how often the history arrays came
back empty so we know for certain after the first real run.

Output schema matches futbin_all_rebuild.jsonl exactly (id, url, pc, ps,
rating, position, nation, league, club, skills, weak_foot, height, foot,
btype, age, squad, playstyles, roles_text) plus scraped_at, so
analysis/build_live_dataset.py can merge it straight into the existing
players.parquet / prices_long.parquet the same way build_dataset.py
originally parsed futbin_all_rebuild.zip.

Run: xvfb-run python3 scrape_player_history.py [num_tabs]
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
MAX_WAIT_SECONDS = 20
POLL_INTERVAL = 0.5
MIN_ARRAY_LEN = 50
PAGE_DELAY = (0.4, 1.0)

BLOCKED = [
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp", "*.svg",
    "*.woff", "*.woff2", "*.ttf", "*.otf", "*.mp4", "*.mp3", "*.css",
    "*googletagmanager*", "*google-analytics*", "*doubleclick*",
    "*facebook.net*", "*hotjar*", "*segment.io*", "*adsystem*",
    "*adnxs*", "*taboola*", "*outbrain*", "*criteo*", "*amazon-adsystem*",
]

out_lock = asyncio.Lock()
stats = {"ok": 0, "no_history": 0, "err": 0}
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
    fires its load event -- confirmed live on scrape_player_details.py's
    identical pattern (a run got stuck on its last job with no error, no
    progress, nothing short of a hard kill). Wrapping it lets a single bad
    page raise into the caller's existing try/except instead of freezing
    the whole run."""
    await asyncio.wait_for(tab.get(url), timeout=GET_PAGE_TIMEOUT)


def parse_body(body_text):
    m = {}

    def grab(label):
        rx = re.compile(rf"(?:^|\n){re.escape(label)}\n([^\n]+)")
        mm = rx.search(body_text)
        return mm.group(1).strip() if mm else None

    m["skills"] = grab("SKILLS")
    m["weak_foot"] = grab("WEAK FOOT")
    m["height"] = grab("HEIGHT")
    m["foot"] = grab("FOOT")
    m["btype"] = grab("B.TYPE")
    m["age"] = grab("AGE")
    m["squad"] = grab("SQUAD")
    return m


async def extract_everything(tab):
    elapsed = 0.0
    while elapsed < MAX_WAIT_SECONDS:
        raw = await tab.evaluate("""
        (() => {
            const pc = document.querySelector('[data-pc-data]');
            const ps = document.querySelector('[data-ps-data]');
            const pcv = pc ? pc.getAttribute('data-pc-data') : '';
            const psv = ps ? ps.getAttribute('data-ps-data') : '';

            const rt = document.querySelector('.playercard-27-rating, .playercard-26-rating, [class*="playercard"][class*="-rating"]');
            const pos = document.querySelector('.playercard-27-position, .playercard-26-position, [class*="playercard"][class*="-position"]');

            const anchors = [];
            document.querySelectorAll('a[href*="/27/players?"], a[href*="/26/players?"]').forEach(a => {
                const img = a.querySelector('img');
                const sp  = a.querySelector('span');
                if (img && sp) anchors.push({ alt: (img.alt||'').toLowerCase(), text: sp.textContent.trim() });
            });

            const playstyles = Array.from(document.querySelectorAll('a[href*="/27/playstyles/"], a[href*="/26/playstyles/"]'))
                .map(a => a.getAttribute('href').split('/').pop()).filter(Boolean);

            const rolesEl = document.querySelector('.player-info-box-roles');

            const priceEl = document.querySelector('.price.inline-with-icon.lowest-price-1');
            let activePlatform = null;
            document.querySelectorAll('form[action="/change-platform"] button[name="platform"]').forEach(btn => {
                const radio = btn.querySelector('.og-radio');
                if (radio && radio.classList.contains('checked')) activePlatform = btn.getAttribute('value');
            });

            return JSON.stringify({
                pc: pcv,
                ps: psv,
                rating: rt ? ((rt.textContent.trim().match(/^\\d+/) || [])[0] || null) : null,
                position: pos ? pos.textContent.trim() : null,
                anchors: anchors,
                playstyles: playstyles,
                roles_text: rolesEl ? rolesEl.textContent.replace(/\\s+/g,' ').trim() : null,
                title: document.title,
                bodyText: document.body ? document.body.innerText : '',
                current_price: priceEl ? priceEl.textContent.trim() : null,
                current_price_platform: activePlatform,
            });
        })()
        """)
        if isinstance(raw, dict):
            raw = raw.get("value", "{}")
        try:
            d = json.loads(raw)
        except Exception:
            d = {}
        pc = d.get("pc", "") or ""
        ps = d.get("ps", "") or ""
        title = d.get("title", "") or ""
        # ready once either history array looks real, OR (history absent but)
        # the page has clearly finished loading (title/body present)
        if len(pc) >= MIN_ARRAY_LEN or len(ps) >= MIN_ARRAY_LEN:
            d["wait_time"] = elapsed
            return d
        if title and d.get("bodyText") and elapsed >= 3.0:
            d["wait_time"] = elapsed
            return d
        await asyncio.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
    return None


def parse_title_rating(title):
    if not title:
        return None
    m = re.search(r"\s-\s(\d{2})\s-\sRating", title)
    return m.group(1) if m else None


def build_record(pid, url, rec, scraped_at):
    if rec is None:
        return {
            "id": pid, "url": url, "scraped_at": scraped_at,
            "pc": None, "ps": None,
            "rating": None, "position": None,
            "nation": None, "league": None, "club": None,
            "skills": None, "weak_foot": None, "height": None,
            "foot": None, "btype": None, "age": None, "squad": None,
            "playstyles": [], "roles_text": None,
            "current_price": None, "current_price_platform": None,
        }
    anchors = rec.get("anchors", [])
    league = nation = club = None
    for a in anchors:
        if a["alt"] == "league":
            league = a["text"]
        elif a["alt"] == "nation":
            nation = a["text"]
        elif a["alt"] == "club":
            club = a["text"]

    extras = parse_body(rec.get("bodyText", ""))
    rating = rec.get("rating") or parse_title_rating(rec.get("title"))

    return {
        "id": pid, "url": url, "scraped_at": scraped_at,
        "pc": rec.get("pc"),
        "ps": rec.get("ps"),
        "rating": rating,
        "position": rec.get("position"),
        "nation": nation,
        "league": league,
        "club": club,
        "skills": extras["skills"],
        "weak_foot": extras["weak_foot"],
        "height": extras["height"],
        "foot": extras["foot"],
        "btype": extras["btype"],
        "age": extras["age"],
        "squad": extras["squad"] or club,
        "playstyles": rec.get("playstyles") or [],
        "roles_text": rec.get("roles_text"),
        "current_price": rec.get("current_price"),
        "current_price_platform": rec.get("current_price_platform"),
    }


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


def load_player_universe():
    players = {}
    path = latest_market_list_file()
    if path:
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
    else:
        print("WARNING: no market_list_*.jsonl found -- run scrape_market_list.py first")
    return players


async def scrape_popular_list(tab):
    await get_page(tab, POPULAR_URL)
    elapsed = 0.0
    hrefs = []
    while elapsed < MAX_WAIT_SECONDS:
        raw = await tab.evaluate(
            """JSON.stringify(Array.from(document.querySelectorAll('a.playercard-wrapper[href^="/27/player/"]')).map(a => a.getAttribute('href')))"""
        )
        if isinstance(raw, dict):
            raw = raw.get("value", "[]")
        try:
            hrefs = json.loads(raw)
        except Exception:
            hrefs = []
        if hrefs:
            break
        await asyncio.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
    players = {}
    for href in hrefs:
        pid, slug = parse_id_slug(href)
        if pid is not None:
            players[pid] = slug
    print(f"  {len(players)} unique players from /27/popular")
    return players


async def worker(name, tab, queue, out_f, scraped_at):
    while True:
        async with out_lock:
            if not queue:
                return
            pid, slug = queue.popleft()
        url = f"{BASE}/27/player/{pid}/{slug}"
        try:
            await get_page(tab, url)
            rec = await extract_everything(tab)
            if rec is None:
                await get_page(tab, url)
                rec = await extract_everything(tab)
            record = build_record(pid, url, rec, scraped_at)
            async with out_lock:
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                pc_pts = (record["pc"] or "").count("[") - 1
                ps_pts = (record["ps"] or "").count("[") - 1
                if rec is None:
                    stats["err"] += 1
                elif pc_pts <= 0 and ps_pts <= 0:
                    stats["no_history"] += 1
                else:
                    stats["ok"] += 1
                pbar.update(1)
                pbar.set_postfix_str(
                    f"tab{name} id={pid} pc={pc_pts} ps={ps_pts} price={record['current_price']} "
                    f"ok={stats['ok']} no_hist={stats['no_history']} err={stats['err']}"
                )
        except Exception as e:
            async with out_lock:
                stats["err"] += 1
                pbar.update(1)
                pbar.set_postfix_str(f"tab{name} ERR {str(e)[:40]}")
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

    players = load_player_universe()
    try:
        popular = await scrape_popular_list(tabs[0])
        players.update(popular)
    except Exception as e:
        print(f"WARNING: /27/popular fetch failed ({str(e)[:80]}) -- "
              f"continuing with just the market-list players")
    print(f"Total unique players to fetch: {len(players)}")

    if not players:
        print("Nothing to do.")
        return

    queue = deque(players.items())
    scraped_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    out_path = SCRAPES_DIR / f"player_history_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    print(f"Writing to {out_path}")

    pbar = tqdm(total=len(queue), desc="Players", unit="player",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")
    with open(out_path, "w", encoding="utf-8") as out_f:
        await asyncio.gather(*[worker(i, tabs[i], queue, out_f, scraped_at) for i in range(len(tabs))])
    pbar.close()

    print(f"\nDone: {stats['ok']} with history, {stats['no_history']} meta-only (no price-history array found), "
          f"{stats['err']} errors -> {out_path}")
    if stats["no_history"] > stats["ok"]:
        print("NOTE: most pages had no price-history array -- FC27 pages likely don't expose "
              "data-pc-data/data-ps-data the way FC26 did. current_price + current_price_platform "
              "should still be usable; long-run price history will need to be built up from our own "
              "repeated scrapes over time instead.")

    try:
        browser.stop()
    except Exception:
        pass
    kill_chrome()


if __name__ == "__main__":
    asyncio.run(main())
