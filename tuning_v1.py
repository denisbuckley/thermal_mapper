#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tuning_v1.py — create/update repo-level tuning.json that pipeline_v4.py reads.
You can run non-interactively with flags, or interactively (press Enter to keep defaults).
"""

import json, argparse
from pathlib import Path

ROOT = Path.cwd()
TUNING_FILE = ROOT / "tuning.json"

DEFAULTS = {
    "circle_eps_m":        200.0,  # DBSCAN eps (meters) for circle clustering
    "circle_min_samples":  2,      # DBSCAN min_samples
    "alt_min_gain":        30.0,   # meters
    "alt_min_duration":    20.0,   # seconds
    "match_max_dist_m":    600.0,  # meters
    "match_min_overlap":   0.25,   # 0..1
}

def load_existing():
    if TUNING_FILE.exists():
        try:
            return {**DEFAULTS, **json.loads(TUNING_FILE.read_text(encoding="utf-8"))}
        except Exception:
            return DEFAULTS.copy()
    return DEFAULTS.copy()

def prompt_float(name, cur):
    raw = input(f"{name} [{cur}]: ").strip()
    return cur if raw == "" else float(raw)

def prompt_int(name, cur):
    raw = input(f"{name} [{cur}]: ").strip()
    return cur if raw == "" else int(raw)

def prompt():
    cfg = load_existing()
    print("Set tuning values (Enter = keep current):\n")
    cfg["circle_eps_m"]       = prompt_float("circle_eps_m (m)", cfg["circle_eps_m"])
    cfg["circle_min_samples"] = prompt_int  ("circle_min_samples", cfg["circle_min_samples"])
    cfg["alt_min_gain"]       = prompt_float("alt_min_gain (m)", cfg["alt_min_gain"])
    cfg["alt_min_duration"]   = prompt_float("alt_min_duration (s)", cfg["alt_min_duration"])
    cfg["match_max_dist_m"]   = prompt_float("match_max_dist_m (m)", cfg["match_max_dist_m"])
    cfg["match_min_overlap"]  = prompt_float("match_min_overlap [0..1]", cfg["match_min_overlap"])
    return cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circle-eps-m",        type=float)
    ap.add_argument("--circle-min-samples",  type=int)
    ap.add_argument("--alt-min-gain",        type=float)
    ap.add_argument("--alt-min-duration",    type=float)
    ap.add_argument("--match-max-dist-m",    type=float)
    ap.add_argument("--match-min-overlap",   type=float)
    ap.add_argument("--interactive", action="store_true", help="Prompt for values (overrides flags)")
    args = ap.parse_args()

    cfg = load_existing()
    if args.interactive or all(getattr(args, k.replace("-", "_")) is None for k in [
        "circle-eps-m","circle-min-samples","alt-min-gain","alt-min-duration","match-max-dist-m","match-min-overlap"
    ]):
        cfg = prompt()
    else:
        if args.circle_eps_m        is not None: cfg["circle_eps_m"]       = args.circle_eps_m
        if args.circle_min_samples  is not None: cfg["circle_min_samples"] = args.circle_min_samples
        if args.alt_min_gain        is not None: cfg["alt_min_gain"]       = args.alt_min_gain
        if args.alt_min_duration    is not None: cfg["alt_min_duration"]   = args.alt_min_duration
        if args.match_max_dist_m    is not None: cfg["match_max_dist_m"]   = args.match_max_dist_m
        if args.match_min_overlap   is not None: cfg["match_min_overlap"]  = args.match_min_overlap

    TUNING_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"[OK] wrote {TUNING_FILE}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())