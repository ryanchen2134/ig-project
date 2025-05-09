#!/usr/bin/env python3
"""
controller_parallel.py
----------------------
Provision a 1-node DOKS cluster in every available region *in parallel*,
run test_runner.py as a Job, stream logs, save cluster metadata, KEEP the
clusters.

Creates/updates clusters.json :

[
  {"id": "...", "region": "nyc1", "version": "1.32.2-do.0",
   "created": "2025-05-07T09:45:03Z", "size": "s-1vcpu-2gb"}
  ...
]
"""
import os, sys, json, time, base64, asyncio, pathlib, aiofiles
from datetime import datetime, timezone

import aiohttp, yaml
from kubernetes import client, config, watch
from dotenv import load_dotenv

# ── CONFIG ───────────────────────────────────────────────────────────────
load_dotenv()
PAT = os.getenv("PAT") or sys.exit("PAT missing in .env")
DO  = "https://api.digitalocean.com/v2"
HEAD = {"Authorization": f"Bearer "+PAT, "Content-Type":"application/json"}

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
LOG_DIR    = SCRIPT_DIR / "logs"; LOG_DIR.mkdir(exist_ok=True)
META_FILE  = SCRIPT_DIR / "clusters.json"
RUNNER_SRC = (SCRIPT_DIR / "test_runner.py").read_text()
RUNNER_B64 = base64.b64encode(RUNNER_SRC.encode()).decode()

# ── HELPER COROUTINES ────────────────────────────────────────────────────
async def do_req(session, method, path, **kw):
    async with session.request(method, f"{DO}{path}", headers=HEAD, **kw) as r:
        if r.status >= 400:
            txt = await r.text()
            print(f"🟥 {method} {path} → {r.status} {txt}")
            r.raise_for_status()
        return await r.json() if r.content_type == "application/json" else await r.text()

async def choose_params(session):
    opts = (await do_req(session,"GET","/kubernetes/options"))["options"]
    version = opts["versions"][0]["slug"]
    size    = next(s["slug"] for s in opts["sizes"] if s["slug"].startswith("s-2vcpu-4gb"))
    regions = [r["slug"] for r in opts["regions"]]
    return version, size, regions

async def append_metadata(meta):
    async with aiofiles.open(META_FILE, "a+") as f:
        await f.write(json.dumps(meta)+"\n")

async def wait_state(session, cid):
    while True:
        st = (await do_req(session,"GET",f"/kubernetes/clusters/{cid}"))["kubernetes_cluster"]["status"]["state"]
        if st == "running": return
        await asyncio.sleep(10)

async def wait_kcfg(session, cid):
    while True:
        r = await session.get(f"{DO}/kubernetes/clusters/{cid}/kubeconfig", headers=HEAD)
        txt = await r.text()
        if r.status == 200 and txt.strip():
            return txt
        await asyncio.sleep(5)

async def run_job(region, cid, kcfg_txt, version):
    # ---- load kubeconfig into a separate client context (threadpool safe)
    cfg_file = pathlib.Path(tempfile.mkstemp(suffix=".kube")[1])
    cfg_file.write_text(kcfg_txt)
    config.load_kube_config(str(cfg_file))
    core, batch = client.CoreV1Api(), client.BatchV1Api()

    # idempotent ConfigMap
    cm = client.V1ConfigMap(metadata=client.V1ObjectMeta(name="runner-src"),
                            data={"test_runner.py": RUNNER_SRC})
    try: core.create_namespaced_config_map("default", cm)
    except client.exceptions.ApiException as e:
        if e.status != 409: raise

    job_name = f"scraper-{region}"
    job = {
      "apiVersion":"batch/v1","kind":"Job","metadata":{"name":job_name},
      "spec":{"ttlSecondsAfterFinished":86400,
              "template":{"spec":{
                "restartPolicy":"Never",
                "containers":[{"name":"runner","image":"python:3.12-slim",
                                "command":["python","/mnt/test_runner.py"],
                                "volumeMounts":[{"name":"src","mountPath":"/mnt"}]}],
                "volumes":[{"name":"src","configMap":{"name":"runner-src"}}]}}}}
    batch.create_namespaced_job("default", job)

    # wait for pod & stream logs
    w = watch.Watch()
    for ev in w.stream(core.list_namespaced_pod, namespace="default",
                       label_selector=f"job-name={job_name}", timeout_seconds=600):
        pod = ev["object"]
        if pod.status.phase in ("Running","Succeeded","Failed"):
            w.stop(); pod_name = pod.metadata.name

    log_file = LOG_DIR / f"{cid}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.txt"
    with open(log_file, "a", buffering=1) as lf:
        stream = core.read_namespaced_pod_log(
            pod_name, "default", follow=True, _preload_content=False,
            insecure_skip_tls_verify_backend=True)
        for chunk in stream.stream():
            line = chunk.decode()
            print(f"[{region}] {line}", end="")
            lf.write(line)
    print(f"📜 [{region}] log saved to {log_file}")

async def provision_one(session, region, version, size):
    body = {
      "name": f"scrape-{region}-{int(time.time())}",
      "region": region,
      "version": version,
      "node_pools":[{"name":"pool","size":size,"count":1}]
    }
    cluster = (await do_req(session,"POST","/kubernetes/clusters", json=body))["kubernetes_cluster"]
    cid = cluster["id"]
    meta = {"id": cid, "region": region, "version": version,
            "size": size, "created": datetime.utcnow().isoformat()+"Z"}
    await append_metadata(meta)
    print(f"🆕 [{region}] cluster {cid} posted")

    await wait_state(session, cid)
    kcfg_txt = await wait_kcfg(session, cid)
    print(f"✅ [{region}] cluster running – submitting job")
    await run_job(region, cid, kcfg_txt, version)
    print(f"🎯 [{region}] done; cluster kept for future use")

async def main():
    async with aiohttp.ClientSession() as session:
        version, size, regions = await choose_params(session)
        print("Regions:", ", ".join(regions))
        tasks = [provision_one(session, r, version, size) for r in regions]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  interrupted – already-running tasks will finish, clusters remain")
