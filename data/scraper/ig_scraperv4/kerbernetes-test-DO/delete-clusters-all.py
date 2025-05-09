#!/usr/bin/env python3
"""
destroy_all_clusters.py
-----------------------
Asynchronously queue DELETE for every Kubernetes cluster in the account,
then wait until all HTTP 202 responses are received.

Usage
=====
PAT=<token> in .env
pip install python-dotenv aiohttp
python destroy_all_clusters.py
"""

import asyncio, os, sys, aiohttp, json, signal
from dotenv import load_dotenv

load_dotenv()
PAT = os.getenv("PAT") or sys.exit("PAT missing in .env")

DO   = "https://api.digitalocean.com/v2"
HEAD = {"Authorization": f"Bearer "+PAT,
        "Content-Type":  "application/json"}

async def list_clusters(session):
    async with session.get(f"{DO}/kubernetes/clusters", headers=HEAD) as r:
        r.raise_for_status()
        data = await r.json()
        return [c["id"] for c in data.get("kubernetes_clusters", [])]

async def delete_one(session, cid, results):
    url = f"{DO}/kubernetes/clusters/{cid}"
    try:
        async with session.delete(url, headers=HEAD) as r:
            if r.status == 204: #no response is the expected success
                results["queued"].append(cid)
                print(f"🗑️  {cid} queued")
            else:
                txt = await r.text()
                results["failed"][cid] = f"{r.status} {txt}"
                print(f"❌  {cid} failed → {r.status}")
    except Exception as e:
        results["failed"][cid] = str(e)
        print(f"❌  {cid} exception {e}")

async def main():
    results = {"queued": [], "failed": {}}
    async with aiohttp.ClientSession() as session:
        ids = await list_clusters(session)
        if not ids:
            print("No clusters found."); return
        print(f"Found {len(ids)} clusters → deleting…")

        tasks = [delete_one(session, cid, results) for cid in ids]
        await asyncio.gather(*tasks)

    print("\n=== summary ===")
    print("queued :", len(results["queued"]))
    if results["failed"]:
        print("failed :", json.dumps(results["failed"], indent=2))

if __name__ == "__main__":
    # ensure Ctrl-C still prints partial summary
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted – some deletions may still be running.")
