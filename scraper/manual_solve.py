#!/usr/bin/env python3
"""
One-time manual Cloudflare-challenge solve. Opens the browser and keeps it open
so you can view/interact with it via VNC, using the SAME persistent chrome_profile
the real scraper uses -- so whatever cookie Cloudflare issues once you solve the
challenge should carry over to future automated (headless-via-Xvfb) runs.

Run this only while Xvfb + x11vnc are already running and DISPLAY is set to match
(see the instructions you were given alongside this file).

Run: python3 manual_solve.py
"""
import asyncio
from pathlib import Path

import nodriver as uc

SCRIPT_DIR = Path(__file__).parent.resolve()
USER_DATA = SCRIPT_DIR / "chrome_profile"
URL = "https://www.futbin.com/27/market-player-list?p_squad=TeamOfTheWeek1&page=1"


async def main():
    config = uc.Config(user_data_dir=str(USER_DATA))
    browser = await uc.start(config=config, headless=False)
    tab = browser.main_tab
    await tab.get(URL)
    print("Browser is open. Connect via VNC now (see instructions) and solve the")
    print("challenge if one appears. This stays open for 5 minutes -- Ctrl+C to end early.")
    try:
        await asyncio.sleep(300)
    except KeyboardInterrupt:
        pass
    try:
        browser.stop()
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(main())
