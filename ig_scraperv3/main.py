import asyncio
import os
import json
import pandas as pd
from datetime import datetime
from playwright.async_api import async_playwright, TimeoutError
from dotenv import load_dotenv
import random

########## CONFIGURATION ##########

#set seed to unix timestamp
random.seed(int(datetime.now().timestamp()))

# Load environment variables
load_dotenv(override=True)
username = os.getenv("USERNAME")
password = os.getenv("PASSWORD")

# Asyncio settings
# Set the maximum number of concurrent pages
MAX_PARALLEL_PAGES = 2
# Set the range for random jitter
# This is used to add a random delay before each request to avoid being blocked
JITTER_RANGE = (1, 2)


MAX_SCROLL_ATTEMPTS = 30
REPEAT_SCROLL_ELEMENTS_LIMIT = 5

GLOBAL_SEM = asyncio.Semaphore(MAX_PARALLEL_PAGES)

############################

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
    if total_posts == float('inf'):
        print(f"Something went horribly wrong.")
        #crash program
        raise ValueError(f"Failed to retrieve total posts for {userhandle}.")
    

    scrape_limit = min(max_posts, total_posts)
    print(f"🔎 {userhandle}: Total posts {total_posts}, scraping up to {scrape_limit}...")

    unique_posts = {}
    scroll_attempts = 0
    
    previous_unique_posts = 0
    repeat_scroll_attempts = 0



    # Scroll and collect post links
    while len(unique_posts) < scrape_limit and scroll_attempts < MAX_SCROLL_ATTEMPTS:
        candidate_elements = await page.query_selector_all("a:has(div._aagu)")
        for element in candidate_elements:
            href = await element.get_attribute("href")
            # Check if href is not None and contains "/p/"
            # and if it is not already in unique_posts
            if href and "/p/" in href and href not in unique_posts:
                unique_posts[href] = None

        # check if our last scroll resulted in new elements, if it has, incrementt repeat_scroll_elements_limit.
        # if it has not for REPEAT_SCROLL_ELEMENTS_LIMIT times, we stop scrolling and just break

        if len(unique_posts) == previous_unique_posts:
            repeat_scroll_attempts += 1
            if repeat_scroll_attempts >= REPEAT_SCROLL_ELEMENTS_LIMIT:
                print(f"🛑 No new posts found after {REPEAT_SCROLL_ELEMENTS_LIMIT} scrolls. Stopping.")
                break
        else:
            repeat_scroll_attempts = 0


        print(f"🔄 Scrolled {scroll_attempts + 1}x / {MAX_SCROLL_ATTEMPTS} — Collected: {len(unique_posts)}, Waivering: {repeat_scroll_attempts} / {REPEAT_SCROLL_ELEMENTS_LIMIT}")
        if len(unique_posts) >= scrape_limit:
            break
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
        await asyncio.sleep(1.5)
        scroll_attempts += 1

        previous_unique_posts = len(unique_posts)


    post_hrefs = list(unique_posts.keys())[:scrape_limit]
    results = {}

    CUTOFF_DATE = datetime(2023, 1, 1)
    print(f"📝 Intercepting posts (cutoff: {CUTOFF_DATE.date()})...")

    ######################     INDIVIDUAL POST SCRAPING     ##########################

    # Iterate over the post links and fetch data (ACTUAL SCRAPING)

    # This is where the parallel scraping happens.         (FAN OUT)
    # We use asyncio.gather to run multiple coroutines concurrently.

    #task list for future use
    tasks   = [asyncio.create_task(_fetch_one(context, h, i, len(post_hrefs)))
           for i, h in enumerate(post_hrefs, start=1)]
    
    #triggered if cutoff date is reached
    early_stop = False

    # Iterate over the tasks as they complete (FAN IN) ######################3
    for fut in asyncio.as_completed(tasks):
        idx, href, data, pdate = await fut

        if data and pdate and pdate < CUTOFF_DATE:
            print(f"🛑  {href} dated {pdate.date()} (<2023). Cancelling remaining.")
            early_stop = True
            break

        if data:
            results[href] = data
            print(f"✅  ({idx}/{len(post_hrefs)}) {href} — "
                f"{pdate.date() if pdate else 'no-date'}")

    if early_stop:
        for t in tasks:
            t.cancel()
        

    return results


async def _fetch_one(context, href, idx, total):
    """
    Open ONE tab (after acquiring semaphore), pull the JSON, close tab.

    args:
        context: Playwright context
        href: URL of the post
        idx: Index of the post in the list
        total: Total number of posts to scrape
    returns:
        idx: Index of the post in the list
        href: URL of the post
        data: JSON data of the post or None if not found
        post_date: Date of the post or None if not found
    """

    # Random jitter to avoid being blocked
    jitter = random.uniform(JITTER_RANGE[0], JITTER_RANGE[1])
    await asyncio.sleep(jitter)

    print(f"{idx=}  looking for semaphore")  

    async with GLOBAL_SEM:                             # ① blocks if >MAX tabs open

        print(f"{idx=}  got permit")  

        page = await context.new_page()         # ② tab is created **inside**
        try:
            async with page.expect_response(
                lambda r: "/api/v1/media/" in r.url and "/info/" in r.url,
                timeout=5_000
                # timeout = 60_000
            ) as resp_info:
                await page.goto(f"https://www.instagram.com{href}?img_index=1")

            print(f"Response received for {idx} ({href})")
            resp  = await resp_info.value
            data  = await resp.json()

            ts        = data.get("items", [{}])[0].get("taken_at")
            post_date = datetime.fromtimestamp(ts) if ts else None
            return idx, href, data, post_date

        except TimeoutError:
            print(f"⏱️ Timeout: {href}")
            return idx, href, None, None
        
        except Exception as e:
            print(f"❌ Error fetching {href}: {e}")
            # wait for a bit before retrying (with jitter)
            await asyncio.sleep(jitter+3)
            return idx, href, None, None

        finally:
            await page.close()                  # ③ tab closed, permit released
            print(f"{idx=}  released permit")

# Batch scrape users from a CSV and save to JSON
async def scrape_users_from_csv(csv_path: str, max_posts_per_user: int, output_json: str):
    df = pd.read_csv(csv_path, header=None)
    usernames = df[0].dropna().unique().tolist()

    USERNAMES_TOTAL = len(usernames)
    print(f"👥 Found {USERNAMES_TOTAL} unique usernames to scrape.")

    if os.path.exists(output_json):
        with open(output_json, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}

    #############  Launch browser #############
    print("Launching browser...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        device = p.devices["Pixel 5"]
        context = await browser.new_context(**device)
        page = await context.new_page()

        # if os.path.exists(".cookies.json"):
        if os.path.exists("/home/asdf/ig-project/ig_scraperv3/.cookies.json"):
            print("🔄 Loading cookies...")
            # with open(".cookies.json", "r") as f:
            with open("/home/asdf/ig-project/ig_scraperv3/.cookies.json", "r") as f:
                cookies = json.load(f)
            await context.add_cookies(cookies)
        else:
            print("🔐 Logging in...")
            await signon(page, username, password)
            
        cookies = await context.cookies()
        # with open(".cookies.json", "w") as f:
        with open("/home/asdf/ig-project/ig_scraperv3/.cookies.json", "w") as f:
            json.dump(cookies, f)


        try:
            # Save cookies to a file in case we're asked again despite being logged in already
            await page.wait_for_selector("span:has-text('Save info')", timeout=5000)
            # Click the button
            await page.click("span:has-text('Save info')", timeout=1000)
            #wait for networkidle
            await page.wait_for_load_state("networkidle")
        except TimeoutError:
            print("🔄 Cookies already saved or not needed.")



        print("Beginning scraping...")
        for idx, user in enumerate(usernames, start=1):
            if user in all_results:
                print(f"⏩ Skipping {user} (already scraped) {idx}/ {USERNAMES_TOTAL}")
                continue
            try:
                print(f"🔍 Scraping {user} ({idx}/ {USERNAMES_TOTAL})...")
                result = await scrape_instagram_posts(user, max_posts_per_user, context, page)
                all_results[user] = result
                with open(output_json, 'w') as f:
                    json.dump(all_results, f, indent=2)
            except Exception as e:
                print(f"❌ Error scraping {user} ({idx}/ {USERNAMES_TOTAL}): {e}")

        await browser.close()
        print(f"\n✅ All scraping complete. Results saved to {output_json}")


import asyncio

if __name__ == "__main__":
    print("Starting scraping...")
    # asyncio.run(scrape_users_from_csv("csv/influencers_only_reversed.csv", max_posts_per_user=150, output_json="json/all_instagram_data.json"))
    asyncio.run(scrape_users_from_csv("/home/asdf/ig-project/ig_scraperv3/csv/influencers_only_reversed.csv", max_posts_per_user=150, output_json="/home/asdf/ig-project/ig_scraperv3/json/all_instagram_data.json"))
