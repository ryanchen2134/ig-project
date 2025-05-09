#!/usr/bin/env python3
"""
controller.py  —  reuse OR create a DO-Kubernetes cluster.
----------------------------------------------------------
.env keys
  PAT          = DigitalOcean personal-access-token  (required)
  CLUSTER_ID   = existing cluster id to reuse        (optional)
"""
import os, sys, time, json, base64, random, textwrap, pathlib, tempfile, threading
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
from kubernetes import client, config, watch

# ── CONFIG ───────────────────────────────────────────────────────────────
load_dotenv()
PAT         = os.getenv("PAT") or sys.exit("PAT missing")
EXISTING_ID = os.getenv("CLUSTER_ID")        # ← set this to reuse
DO   = "https://api.digitalocean.com/v2"
HEAD = {"Authorization": f"Bearer {PAT}", "Content-Type": "application/json"}

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
LOG_DIR    = SCRIPT_DIR / "logs"; LOG_DIR.mkdir(exist_ok=True)

# ── helpers -----------------------------------------------------------------
def api(m, p, **kw):
    r = requests.request(m, f"{DO}{p}", headers=HEAD, timeout=30, **kw)
    if r.status_code >= 400:
        print("🟥 DO-API", r.status_code, r.text)
        r.raise_for_status()
    return r.json() if r.text else {}

def choose_params():
    opts   = api("GET", "/kubernetes/options")["options"]
    ver    = opts["versions"][0]["slug"]
    region = random.choice(opts["regions"])["slug"]
    size   = next(s["slug"] for s in opts["sizes"] if s["slug"].startswith("s-1vcpu-2gb"))
    return ver, region, size

def wait_state(cid, want=("running",), poll=10):
    while True:
        st = api("GET", f"/kubernetes/clusters/{cid}")["kubernetes_cluster"]["status"]["state"]
        if st in want: return st
        time.sleep(poll)

def wait_kubeconfig(cid, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        kc = requests.get(f"{DO}/kubernetes/clusters/{cid}/kubeconfig", headers=HEAD)
        if kc.status_code == 200 and kc.text.strip():
            return kc.text
        time.sleep(5)
    raise TimeoutError("kubeconfig not ready")

def load_kube(kcfg_txt):
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(kcfg_txt.encode()); tmp.close()
    config.load_kube_config(tmp.name)

# ── MAIN --------------------------------------------------------------------
def main():
    cluster_id = EXISTING_ID
    created    = False

    # 0) optionally create
    if not cluster_id:
        ver, region, size = choose_params()
        body = {"name": f"scrape-{int(time.time())}",
                "region": region, "version": ver,
                "node_pools":[{"name":"pool","size":size,"count":1}]}
        print("🆕 creating cluster …", region, ver, size)
        cluster_id = api("POST","/kubernetes/clusters",json=body)["kubernetes_cluster"]["id"]
        created = True

    print("🆔  Using cluster:", cluster_id)
    wait_state(cluster_id, ("running",))
    kcfg = wait_kubeconfig(cluster_id)
    load_kube(kcfg)

    # 1) create ConfigMap + Job
    core  = client.CoreV1Api()
    batch = client.BatchV1Api()

    src = (SCRIPT_DIR/"test_runner.py").read_text()
    cm  = client.V1ConfigMap(metadata=client.V1ObjectMeta(name="runner-src"),
                             data={"test_runner.py":src})
    try: core.create_namespaced_config_map("default", cm)
    except client.exceptions.ApiException as e:
        if e.status != 409: raise  # 409 = already exists

    job = {
      "apiVersion":"batch/v1","kind":"Job","metadata":{"name":"scraper"},
      "spec":{"ttlSecondsAfterFinished":60,
              "template":{"spec":{
                "restartPolicy":"Never",
                "containers":[{
                  "name":"runner","image":"python:3.12-slim",
                  "command":["python","/mnt/test_runner.py"],
                  "volumeMounts":[{"name":"src","mountPath":"/mnt"}]}],
                "volumes":[{"name":"src","configMap":{"name":"runner-src"}}]}}}
          }
    
    batch.create_namespaced_job("default", job)
    print("🚀 Job launched, waiting for pod …")

    # 2) find pod + stream logs
    w = watch.Watch()
    pod = None
    for ev in w.stream(core.list_namespaced_pod,
                       namespace="default",
                       label_selector="job-name=scraper"):
        phase = ev["object"].status.phase
        pod   = ev["object"].metadata.name
        if phase in ("Running","Succeeded","Failed"):
            w.stop()
    assert pod, "pod not found"

    log_f = LOG_DIR / f"{cluster_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.txt"
    with open(log_f,"a",buffering=1) as lf:
        logs = core.read_namespaced_pod_log(
            name=pod, namespace="default", follow=True,
            _preload_content=False, insecure_skip_tls_verify_backend=True)
        for chunk in logs.stream():
            txt = chunk.decode(); print(txt,end=""); lf.write(txt)
    print("\n📜  log saved to", log_f)

    # 3) cleanup if we created the cluster
    if created:
        api("DELETE", f"/kubernetes/clusters/{cluster_id}")
        print("🗑️  cluster deleted")

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: print("\n⏹️  interrupted")
