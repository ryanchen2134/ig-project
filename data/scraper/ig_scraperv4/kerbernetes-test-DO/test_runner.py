#!/usr/bin/env python3
"""
Runs *inside* the Kubernetes Job Pod.
Writes logs/ticks to stdout (which controller streams).
"""

import time, sys

print("🚀 test_runner.py – pod started", flush=True)
for i in range(10):
    print(f"tick {i}", flush=True)
    time.sleep(2)
print("✅ finished OK", flush=True)
sys.exit(0)
