import asyncio
import os
import json
import pandas as pd
from datetime import datetime
from playwright.async_api import async_playwright, TimeoutError
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)
username = os.getenv("USERNAME")
password = os.getenv("PASSWORD")

# Define login function
async def signon(page, username, password):
    try:
        await page.goto("https://www.instagram.com/accounts/login/", wait_until="networkidle")
        await page.wait_for_selector('input[name="username"]', timeout=10000)
        await page.fill('input[name="username"]', username)
        await page.fill('input[name="password"]', password)
        await page.wait_for_timeout(500)
        print("Hitting login button")

        # Wait until the element with the aria-label "Log in" is visible.
        await page.wait_for_selector("div[aria-label='Log in']", state="visible")
        # Now attempt a click on it.
        await page.click("div[aria-label='Log in']")

        #wait for networkidle
        await page.wait_for_load_state("networkidle")
        
        #look for this: <span class="x1lliihq x193iq5w x6ikm8r x10wlt62 xlyipyv xuxw1ft">Save info</span>
        # the button is its 5th parent
        await page.wait_for_selector("span:has-text('Save info')", timeout=15000)
        # Click the button
        await page.click("span:has-text('Save info')", timeout=5000)
        #wait for networkidle
        await page.wait_for_load_state("networkidle")
        if "login" in page.url:
            raise ValueError("Login failed: Incorrect username or password.")
        print("✅ Successfully logged in")
    except TimeoutError:
        raise TimeoutError("Login timed out. Check credentials or network connection.")

# Define helper to get total posts
async def get_total_posts(page):
    total_posts = await page.evaluate("""
        () => {
            const element = document.querySelector('header section ul li span');
            return element ? parseInt(element.innerText.replace(',', '')) : null;
        }
    """)
    return total_posts if total_posts is not None else float('inf')

# Modified scrape_instagram_posts to stop when a post is older than Jan 1, 2023
async def scrape_instagram_posts(userhandle: str, max_posts: int, context, page):
    profile_url = f"https://www.instagram.com/{userhandle}/"
    await page.goto(profile_url)
    await page.wait_for_load_state("networkidle")

    total_posts = await get_total_posts(page)
    scrape_limit = min(max_posts, total_posts)
    print(f"🔎 {userhandle}: Total posts {total_posts}, scraping up to {scrape_limit}...")

    unique_posts = {}
    scroll_attempts = 0
    MAX_SCROLL_ATTEMPTS = 20

    while len(unique_posts) < scrape_limit and scroll_attempts < MAX_SCROLL_ATTEMPTS:
        candidate_elements = await page.query_selector_all("a:has(div._aagu)")
        for element in candidate_elements:
            href = await element.get_attribute("href")
            if href and "/p/" in href and href not in unique_posts:
                unique_posts[href] = None

        # Break early if it seems we've hit the end of the user's posts
        new_total = len(candidate_elements)
        if new_total == 0 or len(unique_posts) >= total_posts:
            print("📉 Reached end of available posts.")
            break

        print(f"🔄 Scrolled {scroll_attempts + 1}x — Collected: {len(unique_posts)}")
        if len(unique_posts) >= scrape_limit:
            break
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
        await asyncio.sleep(1.5)
        scroll_attempts += 1


    post_hrefs = list(unique_posts.keys())[:scrape_limit]
    results = {}

    CUTOFF_DATE = datetime(2023, 1, 1)
    print(f"📝 Intercepting posts (cutoff: {CUTOFF_DATE.date()})...")

    for i, href in enumerate(post_hrefs, start=1):
        post_url = f"https://www.instagram.com{href}"
        new_page = await context.new_page()
        try:
            async with new_page.expect_response(
                lambda response: "/api/v1/media/" in response.url and "/info/" in response.url,
                timeout=5000
            ) as response_info:
                await new_page.goto(post_url)

            response = await response_info.value
            data = await response.json()

            # Check the post timestamp
            timestamp = data.get("items", [{}])[0].get("taken_at")
            if timestamp:
                post_date = datetime.fromtimestamp(timestamp)
                if post_date < CUTOFF_DATE:
                    print(f"🛑 Post {href} is from {post_date.date()}, before 2023. Stopping.")
                    await new_page.close()
                    break

            results[href] = data
            print(f"✅ ({i}/{len(post_hrefs)}) {href} — {post_date.date()}")

        except TimeoutError:
            print(f"⏱️ Timeout: {href}")
        finally:
            await new_page.close()

    return results

# Batch scrape users from a CSV and save to JSON
async def scrape_users_from_csv(csv_path: str, max_posts_per_user: int, output_json: str):
    df = pd.read_csv(csv_path, header=None)
    usernames = df[0].dropna().unique().tolist()

    if os.path.exists(output_json):
        with open(output_json, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        device = p.devices["Pixel 5"]
        context = await browser.new_context(**device)
        page = await context.new_page()

        if os.path.exists("cookies.json"):
            print("🔄 Loading cookies...")
            with open("cookies.json", "r") as f:
                cookies = json.load(f)
            await context.add_cookies(cookies)
        else:
            print("🔐 Logging in...")
            await signon(page, username, password)
            cookies = await context.cookies()
            with open("cookies.json", "w") as f:
                json.dump(cookies, f)

        for user in usernames:
            if user in all_results:
                print(f"⏩ Skipping {user} (already scraped)")
                continue
            try:
                result = await scrape_instagram_posts(user, max_posts_per_user, context, page)
                all_results[user] = result
                with open(output_json, 'w') as f:
                    json.dump(all_results, f, indent=2)
            except Exception as e:
                print(f"❌ Error scraping {user}: {e}")

        await browser.close()
        print(f"\n✅ All scraping complete. Results saved to {output_json}")


import asyncio

if __name__ == "__main__":
    asyncio.run(scrape_users_from_csv("influencers_only_reversed.csv", max_posts_per_user=150, output_json="all_instagram_data.json"))
