#!/usr/bin/env python3
"""
controller.py
-------------
• Spin up a droplet, copy test_runner.py, stream its logs in real time,
  append to ./logs/<dropletId>_<UTC>.txt, and ALWAYS clean up.
• Cleanup is executed:
      – normal exit
      – any exception
      – KeyboardInterrupt
Dependencies: python-dotenv paramiko requests
SSH key: ~/.ssh/dovpn_key (+ .pub) must exist.
"""

import os, sys, time, json, random, textwrap, secrets, base64, pathlib, threading, socket
from datetime import datetime, timezone
import paramiko, requests
from dotenv import load_dotenv
import signal

# ── CONFIG ───────────────────────────────────────────────────────────────
load_dotenv()
PAT = os.getenv("PAT") or sys.exit("❌ PAT missing in .env")

SSH_KEY = pathlib.Path.home() / ".ssh/dovpn_key"
PUB_KEY = SSH_KEY.with_suffix(".pub").read_text().strip()

DO   = "https://api.digitalocean.com/v2"
HEAD = {"Authorization": f"Bearer {PAT}", "Content-Type": "application/json"}
SIZE = "s-1vcpu-1gb"; IMAGE = "ubuntu-22-04-x64"
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

# ── HELPERS ──────────────────────────────────────────────────────────────
def wait_for_ssh(host, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, 22), timeout=5):
                return
        except OSError:
            time.sleep(3)
    raise TimeoutError("SSH port 22 never opened")

def api(method, path, **kw):
    r = requests.request(method, f"{DO}{path}", headers=HEAD, timeout=30, **kw)
    r.raise_for_status(); return r.json() if r.text else {}

def pick_region():
    return random.choice([r["slug"] for r in api("GET", "/regions")["regions"] if r["available"]])

def ensure_ssh_key():
    keys = api("GET", "/account/keys")["ssh_keys"]
    for k in keys:
        if k["public_key"].strip().endswith(PUB_KEY.split()[-1]):
            return k["id"]
    return api("POST", "/account/keys", json={"name": f"autokey-{int(time.time())}",
                                              "public_key": PUB_KEY})["ssh_key"]["id"]

def runner_user_data():
    src = (SCRIPT_DIR / "test_runner.py").read_text()
    b64 = base64.b64encode(src.encode()).decode()
    return textwrap.dedent(f"""\
      #cloud-config
      package_update: true
      packages: [python3, python3-pip]
      write_files:
        - path: /opt/runner/runner.b64
          content: {b64}
          encoding: b64
          permissions: '0644'
      runcmd:
        - mkdir -p /opt/runner
        - base64 -d /opt/runner/runner.b64 > /opt/runner/test_runner.py
        - chmod +x /opt/runner/test_runner.py
        - pip3 install --quiet requests
        - PAT="{PAT}" /usr/bin/env python3 /opt/runner/test_runner.py
    """)

# ── MAIN ─────────────────────────────────────────────────────────────────
def main():
    region  = pick_region();          print("🌎  Region:", region)
    ssh_id  = ensure_ssh_key();       print("🔑  SSH key id:", ssh_id)

    body = {
        "name": f"dsvps-{secrets.token_hex(3)}",
        "region": region,
        "size": SIZE,
        "image": IMAGE,
        "ssh_keys": [ssh_id],
        "user_data": runner_user_data(),
        "tags": ["dsvps"]
    }

    droplet_id = None    # track for cleanup
    ssh_client = None
    log_file   = None
    stream_thread = None

    # graceful Ctrl-C
    def _sigint_handler(sig, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        print("🚀  Creating droplet …")
        droplet_id = api("POST", "/droplets", json=body)["droplet"]["id"]

        # wait for IPv4
        print(f"🆔  Droplet {droplet_id} – waiting for IPv4 …")
        while True:
            d = api("GET", f"/droplets/{droplet_id}")["droplet"]
            if d["status"] == "active" and d["networks"]["v4"]:
                ip = d["networks"]["v4"][0]["ip_address"]; break
            time.sleep(4)

        print("✅  Droplet ready @", ip, " — streaming logs …")

        print("⌛  waiting for SSH port 22 …")
        wait_for_ssh(ip)
        print("🔐 SSH is up, opening stream …")

        # prepare local log
        log_dir = SCRIPT_DIR / "logs"; log_dir.mkdir(exist_ok=True)
        stamp   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        local_fp = log_dir / f"{droplet_id}_{stamp}.txt"
        log_file = open(local_fp, "a", buffering=1)

        # SSH stream setup
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(ip, username="root", key_filename=str(SSH_KEY), timeout=30)
        _, stdout, _ = ssh_client.exec_command("tail -F /opt/runner/logs/current.log")

        # background reader
        def _stream():
            for line in iter(stdout.readline, ""):
                print(line, end=""); log_file.write(line)
        stream_thread = threading.Thread(target=_stream, daemon=True)
        stream_thread.start()
        stream_thread.join()  # wait until log stream ends

        print("\n📜  Log saved to", local_fp)

    finally:
        # ── CLEANUP ────────────────────────────────────────────────
        if ssh_client:      ssh_client.close()
        if log_file:        log_file.close()
        if droplet_id:
            try:
                api("DELETE", f"/droplets/{droplet_id}")
                print("🗑️   Droplet deleted.")
            except Exception as e:
                print("⚠️   Droplet delete failed:", e)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted — cleanup complete.")
    except Exception as e:
        print("❌  Error:", e)
        sys.exit(1)
