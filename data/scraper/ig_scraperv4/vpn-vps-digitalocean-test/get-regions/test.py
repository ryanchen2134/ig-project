#!/usr/bin/env python3
"""
temp.py – list DigitalOcean regions via the public REST endpoint
----------------------------------------------------------------
Prerequisites:
  • python-dotenv      pip install python-dotenv requests tabulate
  • requests
  • tabulate   (pretty-print table, optional but nice)

The script expects PAT to be defined (env or .env).
"""

import os, sys, requests
from dotenv import load_dotenv
from tabulate import tabulate   # pip install tabulate

API     = "https://api.digitalocean.com/v2/regions"
HEADERS = {
    "Authorization": f"Bearer {os.getenv('PAT', '')}",
    "Content-Type":  "application/json"
}

def main() -> None:
    load_dotenv()                       # loads PAT from .env if present
    token = os.getenv("PAT")
    if not token:
        sys.exit("❌  PAT not found in environment or .env file")

    HEADERS["Authorization"] = f"Bearer {token}"
    try:
        resp = requests.get(API, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        sys.exit(f"❌  API call failed: {e}")

    # Extract and display
    regions = resp.json().get("regions", [])
    table = [(r["slug"], r["name"], r["available"]) for r in regions]
    print(tabulate(table, headers=["Slug", "Name", "Available"], tablefmt="github"))

    #save to file
    with open("regions.json", "w") as f:
        f.write(resp.text)
    print("Regions saved to regions.json")

    f.close()

main()
