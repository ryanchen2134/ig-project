import os, random, requests
from dotenv import load_dotenv

# Load PAT once (nothing else is read or cached)
load_dotenv()
PAT = os.getenv("PAT")
if not PAT:
    raise RuntimeError("PAT missing in environment or .env")

API     = "https://api.digitalocean.com/v2/regions"
HEADERS = {"Authorization": f"Bearer {PAT}", "Content-Type": "application/json"}

def pick_new_region(prev_slug: str) -> str:
    """
    Return an available DigitalOcean region slug that is ≠ prev_slug.
    Raises RuntimeError if no alternative is found.

    Example:
        new_slug = pick_new_region("nyc3")
    """
    resp = requests.get(API, headers=HEADERS, timeout=15)
    resp.raise_for_status()

    regions = resp.json().get("regions", [])
    candidates = [
        r["slug"] for r in regions 
        if r.get("available") and r["slug"] != prev_slug
    ]

    if not candidates:
        raise RuntimeError("No alternative available regions found.")

    return random.choice(candidates)




print(pick_new_region("nyc2"))