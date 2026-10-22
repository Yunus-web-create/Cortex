"""Visual QA harness: drives the real UI in headless Chromium and captures screenshots.

    python tests/ui_screenshots.py            # expects the server on :8000
Set OUT to any writable directory.
"""
import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

OUT = Path("/home/user/shots")
OUT.mkdir(exist_ok=True)
BASE = "http://localhost:8000"
errors = []


async def main():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        ctx = await browser.new_context(viewport={"width": 1440, "height": 950}, device_scale_factor=2)
        page = await ctx.new_page()
        page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))

        await page.goto(BASE, wait_until="networkidle")
        await page.wait_for_timeout(1200)
        if await page.locator("#newChat").count():
            await page.click("#newChat")          # hero only renders on a fresh chat
            await page.wait_for_timeout(900)

        docs = int(await page.locator("#docCount").inner_text() or 0)
        if docs == 0:
            await page.screenshot(path=str(OUT / "01-empty.png"))
            await page.click("#heroDemo")
            await page.wait_for_timeout(7000)
        await page.screenshot(path=str(OUT / "02-loaded.png"))

        # ask via a suggestion card (or type it)
        TOOL_WORDS = ("weather", "time is it", "% of", "wikipedia")
        cards = page.locator(".starter-card:visible")
        chosen = None
        for i in range(await cards.count()):
            text = (await cards.nth(i).inner_text()).lower()
            if not any(w in text for w in TOOL_WORDS):
                chosen = cards.nth(i)
                break
        if chosen is not None:
            print("first suggestion:", (await chosen.inner_text()).split("\n")[0])
            await chosen.click()
        else:
            await page.fill("#input", "How many annual leave days do I get and can I carry any over?")
            await page.press("#input", "Enter")
        await page.wait_for_timeout(6000)
        await page.screenshot(path=str(OUT / "03-answer.png"))
        print("answer:", (await page.locator(".msg.assistant .msg-content").last.inner_text())[:180])

        print("grounding badge:", await page.locator(".grounding").count())
        if await page.locator(".grounding").count():
            print("  text:", await page.locator(".grounding").last.inner_text())
            await page.locator(".grounding").last.click()
            await page.wait_for_timeout(500)
            await page.screenshot(path=str(OUT / "04-grounding.png"))
            await page.click("#popClose")

        # sources + citation
        chips = page.locator(".src-chip")
        print("source chips:", await chips.count())
        if await chips.count():
            await chips.first.click()
            await page.wait_for_timeout(400)
            await page.screenshot(path=str(OUT / "05-source.png"))
            await page.click("#popClose")

        # document scope menu
        await page.click("#scopeChip")
        await page.wait_for_timeout(500)
        await page.screenshot(path=str(OUT / "06-scope.png"))
        await page.locator(".scope-item input").first.check()
        await page.click("#scopeApply")
        await page.wait_for_timeout(400)
        print("scope label:", await page.locator("#scopeLabel").inner_text())

        # answer within the scoped document
        await page.fill("#input", "How many days of annual leave do I get?")
        await page.press("#input", "Enter")
        await page.wait_for_timeout(5000)
        await page.screenshot(path=str(OUT / "07-scoped-answer.png"))

        # reset scope
        await page.click("#scopeChip")
        await page.wait_for_timeout(300)
        await page.click("#scopeAll")
        await page.click("#scopeApply")
        await page.wait_for_timeout(300)

        # retrieval probe
        await page.click("#kbBtn")
        await page.wait_for_timeout(600)
        await page.screenshot(path=str(OUT / "08-drawer.png"))
        await page.click('.tab[data-tab="retrieval"]')
        await page.fill("#probeInput", "What is the approval escalation timeout?")
        await page.click("#probeBtn")
        await page.wait_for_timeout(1500)
        await page.screenshot(path=str(OUT / "09-retrieval.png"))
        await page.click("#drawerClose")

        # settings incl. quality panel
        await page.click("#openSettings")
        await page.wait_for_timeout(900)
        await page.screenshot(path=str(OUT / "10-settings.png"))
        await page.click("#settingsClose")

        # dark mode
        await page.click("#themeBtn")
        await page.wait_for_timeout(700)
        await page.screenshot(path=str(OUT / "11-dark.png"))

        # mobile
        m = await browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=3)
        mp = await m.new_page()
        mp.on("pageerror", lambda e: errors.append(f"mobile pageerror: {e}"))
        await mp.goto(BASE, wait_until="networkidle")
        await mp.wait_for_timeout(1800)
        await mp.screenshot(path=str(OUT / "12-mobile.png"))

        print("\nERRORS:", len(errors))
        for e in errors[:15]:
            print(" -", e)
        await browser.close()


asyncio.run(main())
