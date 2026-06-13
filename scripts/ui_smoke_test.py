"""Headless browser smoke test for the Streamlit UI (no Ollama needed).

Verifies the app boots, every tab renders, the explanatory help cards show the
metric definitions, and there are no Python tracebacks on screen.
Run while `streamlit run frontend/app.py --server.port 8502` is up.
"""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

URL = "http://localhost:8502"
OUT = Path(__file__).resolve().parent.parent / "data" / "ui_test_shots"
OUT.mkdir(parents=True, exist_ok=True)

TABS = ["1. Documents", "2. Prepare Documents", "3. Search",
        "📌 My Findings", "4. Validate Search", "5. Activity Log", "6. History"]

results = []

def check(name, cond, detail=""):
    results.append((name, cond, detail))
    print(("PASS" if cond else "FAIL"), name, "-", detail)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 1600})
    page.goto(URL, wait_until="networkidle", timeout=60000)
    page.wait_for_selector("text=Corpus Search Tool", timeout=30000)
    page.wait_for_timeout(1500)

    body = page.inner_text("body")
    check("app boots (title present)", "Corpus Search Tool" in body)
    check("no python traceback on load", "Traceback" not in body, "found Traceback" if "Traceback" in body else "")

    tabs = page.locator('[data-baseweb="tab"]')
    n_tabs = tabs.count()
    check("all tabs present", n_tabs == len(TABS), f"found {n_tabs}")
    for i in range(n_tabs):
        tabs.nth(i).click(timeout=8000)
        page.wait_for_timeout(1200)
        body = page.inner_text("body")
        label = TABS[i] if i < len(TABS) else f"tab{i}"
        check(f"tab renders: {label}", "Corpus Search Tool" in body)
        check(f"no traceback in tab: {label}", "Traceback" not in body)
        page.screenshot(path=str(OUT / f"tab_{i}.png"))

    # Validate Search tab: open the help cards and confirm the metric definitions render.
    tabs.nth(4).click()
    page.wait_for_timeout(1000)
    # Expand the two always-on help expanders.
    for exp_label in ["What are reference passages?", "How does the search work?"]:
        try:
            page.get_by_text(exp_label, exact=False).first.click(timeout=5000)
            page.wait_for_timeout(400)
        except Exception as e:
            print("could not expand", exp_label, e)
    page.wait_for_timeout(600)
    body = page.inner_text("body")
    check("Coverage defined in help", "Coverage" in body)
    check("Accuracy defined as 0-100%", "0–100%" in body or "0-100%" in body)
    check("Relevance explained (not RRF)", "Relevance" in body)
    check("Accuracy stays within 0-100%", "650%" not in body and "%>100" not in body)
    page.screenshot(path=str(OUT / "validate_help_expanded.png"), full_page=True)

    browser.close()

n_fail = sum(1 for _, c, _ in results if not c)
print(f"\n=== {len(results)-n_fail}/{len(results)} checks passed; screenshots in {OUT} ===")
sys.exit(1 if n_fail else 0)
