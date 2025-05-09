import os, sys, time, json, yaml, pathlib, requests, tempfile
from kubernetes import client, config, watch
from dotenv import load_dotenv

load_dotenv(); PAT = os.getenv("PAT") or sys.exit("PAT missing")
HEAD = {"Authorization": f"Bearer "+PAT, "Content-Type":"application/json"}
DO   = "https://api.digitalocean.com/v2"
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
LOG_DIR    = SCRIPT_DIR/"logs"; LOG_DIR.mkdir(exist_ok=True)
def api(m, p, **kw):
    r = requests.request(m, DO+p, headers=HEAD, timeout=30, **kw); r.raise_for_status()
    return r.json()


opts = api("GET", "/kubernetes/options")

#save return json
with open(LOG_DIR/"k8s_options.json", "w") as f:
    json.dump(opts, f, indent=2)
version_slug = opts["options"]["versions"][0]["slug"]      # newest supported
size_slug    = next(s["slug"] for s in opts["options"]["sizes"]
                    if s["available"] and s["slug"].startswith("s-2vcpu"))
