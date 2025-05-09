#!/usr/bin/env python3
"""
test_runner.py  (executed inside the VPS)
----------------------------------------
• Writes log lines into logs/run-<UTC>.txt
• Maintains a symlink logs/current.log → active run file
• On normal exit or SIGINT/SIGTERM:
      1) closes log
      2) DELETEs its own droplet via DigitalOcean API
• Droplet thereby removes itself; controller’s SSH stream ends.
"""

import os, sys, time, signal, datetime, pathlib, requests

BASE     = pathlib.Path(__file__).resolve().parent       # /opt/runner
LOG_DIR  = BASE / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

ts   = datetime.datetime.utcnow().isoformat(timespec="seconds")
run_log_path = LOG_DIR / f"run-{ts}.txt"
log  = open(run_log_path, "a", buffering=1)

# create /logs/current.log symlink to latest run file
symlink = LOG_DIR / "current.log"
try:
    if symlink.exists() or symlink.is_symlink():
        symlink.unlink()
    symlink.symlink_to(run_log_path, target_is_directory=False)
except Exception as e:
    print("symlink error:", e, file=sys.stderr)

def logprint(msg):
    print(msg, file=log); log.flush()

def droplet_id():
    return requests.get("http://169.254.169.254/metadata/v1/id", timeout=10).text

PAT = os.getenv("PAT")      # forwarded in controller’s cloud-init
DID = droplet_id()

def cleanup(sig=None, frm=None):
    logprint(f"⚙️  cleanup: destroying droplet {DID}")
    if PAT:
        try:
            requests.delete(f"https://api.digitalocean.com/v2/droplets/{DID}",
                            headers={"Authorization": f"Bearer {PAT}"}, timeout=20)
        except Exception as e:
            logprint("delete failed: " + str(e))
    log.close(); sys.exit(0)

signal.signal(signal.SIGINT,  cleanup)
signal.signal(signal.SIGTERM, cleanup)

try:
    logprint("🚀 test_runner started in " + str(BASE))
    for i in range(10):
        logprint(f"tick {i}")
        time.sleep(2)
    logprint("✅ finished OK")
finally:
    cleanup()
