#!/usr/bin/env python3
"""
Diagnostic: load the market-player-list page and report exactly what the browser
actually saw -- page title, whether common bot-block/captcha phrases appear, how
many rows different possible selectors find, and a screenshot for visual check.

Run: xvfb-run python3 diagnose.py
"""
import asyncio
from pathlib import Path

import nodriver as uc
from nodriver import cdp

SCRIPT_DIR = Path(__file__).parent.resolve()
USER_DATA = SCRIPT_DIR / "chrome_profile"
URL = "https://www.futbin.com/27/market-player-list?p_squad=TeamOfTheWeek1&page=1"

BLOCKED = [
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp", "*.svg",
    "*.woff", "*.woff2", "*.ttf", "*.otf", "*.mp4", "*.mp3",
    "*googletagmanager*", "*google-analytics*", "*doubleclick*",
]


def kill_chrome():
    import subprocess as _sp
    _sp.run("pkill -f chrome_profile", shell=True)


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

    print(f"Loading {URL} ...")
    await tab.get(URL)

    # poll for the Cloudflare "Just a moment..." challenge to clear, the same way the
    # existing working player-page scraper polls for real content up to 20s, instead
    # of guessing a fixed wait is enough
    max_wait = 25
    elapsed = 0.0
    while elapsed < max_wait:
        title = await tab.evaluate("document.title")
        if isinstance(title, dict):
            title = title.get("value", "")
        print(f"  [{elapsed:.1f}s] title={title!r}")
        if title and "just a moment" not in title.lower():
            print(f"  Challenge cleared after {elapsed:.1f}s")
            break
        await asyncio.sleep(1.0)
        elapsed += 1.0
    else:
        print(f"  Still on challenge page after {max_wait}s")

    info = await tab.evaluate("""
    (() => {
        const bodyText = document.body ? document.body.innerText.slice(0, 2000) : '';
        const lower = bodyText.toLowerCase();
        const blockPhrases = ['captcha', 'checking your browser', 'cloudflare', 'access denied',
                               'are you human', 'unusual traffic', 'blocked', 'just a moment',
                               'verify you are human', 'ddos protection'];
        const found = blockPhrases.filter(p => lower.includes(p));
        return JSON.stringify({
            title: document.title,
            url: window.location.href,
            body_text_snippet: bodyText.slice(0, 800),
            block_phrases_found: found,
            n_tr_player_row: document.querySelectorAll('tr.player-row').length,
            n_any_tr: document.querySelectorAll('tr').length,
            n_table: document.querySelectorAll('table').length,
            html_length: document.documentElement.outerHTML.length,
        });
    })()
    """)
    print("\n=== DIAGNOSTIC RESULT ===")
    print(info)

    shot_path = await tab.save_screenshot(filename=str(SCRIPT_DIR / "diagnose_screenshot.jpg"), format="jpeg", full_page=True)
    print(f"\nScreenshot saved to: {shot_path}")

    html_path = SCRIPT_DIR / "diagnose_page.html"
    html = await tab.get_content()
    html_path.write_text(html, encoding="utf-8")
    print(f"Full page HTML saved to: {html_path}")

    try:
        browser.stop()
    except Exception:
        pass
    kill_chrome()


if __name__ == "__main__":
    asyncio.run(main())
