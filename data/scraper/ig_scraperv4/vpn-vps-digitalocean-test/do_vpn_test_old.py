#!/usr/bin/env python3
"""
do_vpn_test.py – disposable WireGuard hop on DigitalOcean
--------------------------------------------------------
1. Pick a fresh region.
2. Spawn a droplet with WireGuard via cloud-init.
3. Bring tunnel up and verify external IP.
4. Always tear everything down (Ctrl-C, errors, normal exit).

Reads PAT from .env (key PAT). Writes nothing permanent.
"""

import os, sys, json, time, random, textwrap, secrets, tempfile, signal
import requests, subprocess
from dotenv import load_dotenv

# ─── ENV / CONSTANTS ──────────────────────────────────────────────────────────
load_dotenv()
PAT = os.getenv("PAT")
if not PAT:
    sys.exit("❌  PAT not found in environment or .env")

DO_API   = "https://api.digitalocean.com/v2"
HEADERS  = {"Authorization": f"Bearer {PAT}", "Content-Type": "application/json"}
SIZE     = "s-1vcpu-1gb"
IMAGE    = "ubuntu-22-04-x64"
VPN_CIDR = "10.88.0.0/24"
CLIENT_IP= "10.88.0.2/32"

MAX_BOOT = 120      # seconds
POLL     = 4        # polling interval

# ─── helpers ──────────────────────────────────────────────────────────────────
def api(method: str, path: str, **kw):
    r = requests.request(method, f"{DO_API}{path}", headers=HEADERS,
                         timeout=30, **kw)
    r.raise_for_status()
    return r.json() if r.text else {}

def wg_keypair():
    priv = subprocess.check_output(["wg", "genkey"]).strip().decode()
    pub  = subprocess.check_output(["wg", "pubkey"],
                                   input=priv.encode()).strip().decode()
    return priv, pub

def pick_new_region(prev_slug: str | None = None) -> str:
    print("🔍  Fetching available regions …")
    regions = api("GET", "/regions")["regions"]
    choices = [r["slug"] for r in regions if r["available"] and r["slug"] != prev_slug]
    if not choices:
        raise RuntimeError("No alternative available regions.")
    slug = random.choice(choices)
    print(f"🌎  Selected region: {slug}")
    return slug

def create_and_wait(region: str, user_data: str) -> tuple[str, str]:
    """Return (droplet_id, IPv4) or raise RuntimeError (with cleanup)."""
    body = {
        "name": f"wg-hop-{secrets.token_hex(3)}",
        "region": region,
        "size":   SIZE,
        "image":  IMAGE,
        "user_data": user_data,
        "tags": ["vpn-hop"]
    }
    print("🚀  Creating droplet …")
    r = requests.post(f"{DO_API}/droplets", headers=HEADERS,
                      data=json.dumps(body), timeout=30)
    if r.status_code != 202:
        raise RuntimeError(f"Create failed – HTTP {r.status_code}: {r.text}")
    did = r.json()["droplet"]["id"]
    print(f"🆔  Droplet ID {did}")

    start = time.time()
    try:
        while True:
            d = api("GET", f"/droplets/{did}")["droplet"]
            status = d["status"]
            elapsed = int(time.time() - start)
            print(f"   ↻  status={status:<7}  t={elapsed:>3}s", end="\r")
            if status == "active":
                nets = d["networks"]["v4"]
                if not nets:
                    raise RuntimeError("Droplet active but no IPv4")
                ip = nets[0]["ip_address"]
                print()  # newline after carriage-return status line
                print(f"✅  Droplet active at {ip}")
                return did, ip
            if status in {"archive", "errored"}:
                raise RuntimeError(f"Droplet entered bad state: {status}")
            if elapsed > MAX_BOOT:
                raise RuntimeError("Droplet boot timed out")
            time.sleep(POLL)
    except Exception:
        print("\n🧹  Cleaning up failed droplet …")
        try: api("DELETE", f"/droplets/{did}")
        except Exception as e: print(f"   ⚠️  Cleanup error: {e}")
        raise

# ─── disposable WireGuard hop ────────────────────────────────────────────────
class VPNHop:
    def __init__(self, region: str):
        self.region = region
        self.droplet_id = None
        self.external_ip = None
        self._cfg_path = None

    def __enter__(self):
        print("🔑  Generating WireGuard keys …")
        srv_priv, srv_pub = wg_keypair()
        cli_priv, cli_pub = wg_keypair()

        user_data = textwrap.dedent(f"""\
            #cloud-config
            package_update: true
            packages: [wireguard]
            write_files:
              - path: /etc/wireguard/wg0.conf
                owner: root:root
                permissions: "0600"
                content: |
                  [Interface]
                  Address    = {VPN_CIDR.split('/')[0]}/24
                  ListenPort = 51820
                  PrivateKey = {srv_priv}

                  [Peer]
                  PublicKey  = {cli_pub}
                  AllowedIPs = 0.0.0.0/0
            runcmd:
              # 1) enable IPv4 forwarding now and make it persistent
              - sysctl -w net.ipv4.ip_forward=1
              - sed -i 's/^#*net.ipv4.ip_forward=.*/net.ipv4.ip_forward=1/' /etc/sysctl.conf

              # 2) NAT: rewrite every packet that leaves eth0
              - iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
              - iptables-save > /etc/iptables/rules.v4   # keep rule after reboot

              # 3) start WireGuard
              - systemctl enable --now wg-quick@wg0
        """)

        self.droplet_id, self.external_ip = create_and_wait(self.region, user_data)

        # tmp wg-quick file
        self._cfg_path = tempfile.NamedTemporaryFile(
            delete=False, suffix=".conf").name
        cfg = textwrap.dedent(f"""\
            [Interface]
            Address    = {CLIENT_IP}
            PrivateKey = {cli_priv}

            [Peer]
            PublicKey  = {srv_pub}
            Endpoint   = {self.external_ip}:51820
            AllowedIPs = 0.0.0.0/0
            PersistentKeepalive = 25
        """)
        with open(self._cfg_path, "w") as f:
            f.write(cfg)

        print("↗️  Bringing tunnel up …")
        subprocess.run(["sudo", "-n", "wg-quick", "up", self._cfg_path], check=True)
        print("🟢  Tunnel interface up.")
        return self

    def __exit__(self, exc_type, exc, tb):
        print("⬇️  Tearing tunnel down …")
        try: subprocess.run(["sudo", "-n", "wg-quick", "down", self._cfg_path],
                            check=True)
        except subprocess.CalledProcessError: print("   ⚠️  wg-quick down failed")
        if self.droplet_id:
            print("🗑️   Destroying droplet …")
            try: api("DELETE", f"/droplets/{self.droplet_id}")
            except requests.RequestException as e:
                print(f"   ⚠️  Droplet delete error: {e}")
        if self._cfg_path and os.path.exists(self._cfg_path):
            os.unlink(self._cfg_path)
        print("✅  Cleanup complete.")
        return False  # propagate any original exception

# ─── Ctrl-C becomes KeyboardInterrupt so __exit__ still runs ────────────────
signal.signal(signal.SIGINT, lambda *_: sys.exit("\n⏹️  Interrupted."))

# ─── demo main ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    prev = None
    try:
        region = pick_new_region(prev)
        with VPNHop(region) as hop:
            print("🌐  Verifying external IP …")
            seen_ip = requests.get("https://api.ipify.org")
            print(f"   Internet sees us as {seen_ip}")
            if seen_ip != hop.external_ip:
                raise RuntimeError("IP mismatch – tunnel not routing!")
            print("🎉  Full test passed. Sleeping 10 s …")
            time.sleep(10)
    except Exception as e:
        sys.exit(f"❌  {e}")
