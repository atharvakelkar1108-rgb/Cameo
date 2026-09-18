"""
Terminal demo of the Resource-Aware Model Selector.

This does NOT load any model or run inference — it only reads the
already-computed results/*.json files and recommends a model based on
available RAM and a priority mode. The actual selection logic lives in
utils/model_selector.py (shared with the web app, so both give identical
recommendations).

Run:
    python scripts/07_resource_selector.py --ram 2048 --priority fast
    python scripts/07_resource_selector.py --ram 4096 --priority accuracy
    python scripts/07_resource_selector.py --ram 512 --priority auto
"""

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.model_selector import select_model

parser = argparse.ArgumentParser(description="Resource-Aware Model Selector (terminal demo)")
parser.add_argument("--ram", type=float, required=True, help="Available RAM in MB, e.g. 2048")
parser.add_argument("--priority", choices=["fast", "balanced", "accuracy", "memory", "auto"], default="auto")
parser.add_argument("--cores", type=int, default=4, help="CPU core count (informational)")
args = parser.parse_args()

result = select_model(ram_available_mb=args.ram, cpu_cores=args.cores, priority=args.priority)

print("=" * 60)
print("RESOURCE-AWARE MODEL SELECTOR")
print("=" * 60)
print(f"\nAvailable RAM: {args.ram:.0f} MB")
print(f"Priority: {args.priority.upper()}")

if not result.get("selected"):
    print(f"\n{result['explanation']}")
else:
    profile = result["profile"]
    print(f"\nRecommended Model:\n  {profile['display_name']}")
    print(f"\nModel Size:       {profile['size_mb']} MB")
    print(f"Memory Footprint: {profile['footprint_mb']} MB")
    print(f"Mean Latency:     {profile['latency_ms']} ms")
    print(f"BLEU-4:           {profile['bleu4']}")
    if profile.get("sparsity_pct") is not None:
        print(f"Sparsity:         {profile['sparsity_pct']}%")
    print(f"\nReason:\n  {result['explanation']}")
    if result.get("fallback_used"):
        print("\n  NOTE: no model fit comfortably within the RAM safety budget —")
        print("  this is the smallest available variant anyway.")

print("=" * 60)
