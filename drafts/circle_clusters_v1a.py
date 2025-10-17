#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
circle_clusters_v1a.py — thin wrapper for circle_clusters_v1.py with interactive fallback.
- If no IGC path is provided, prompts for it.
- Passes through optional flags: --clusters-csv, --segments-csv
- Executes circle_clusters_v1.py using the same Python interpreter.
"""

import sys, os, argparse, shlex, subprocess

def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("igc", nargs="?", help="Path to IGC file")
    ap.add_argument("--clusters-csv", default=None, help="Output CSV for clusters (default: circle_clusters.csv)")
    ap.add_argument("--segments-csv", default=None, help="Output CSV for segments (default: circle_segments.csv)")
    args = ap.parse_args()

    # Interactive fallback
    igc_path = args.igc or input("Enter path to IGC file: ").strip()
    if not igc_path:
        print("No IGC path provided. Exiting.", file=sys.stderr)
        sys.exit(2)

    # Resolve target script next to this file
    here = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(here, "circle_clusters_v1.py")
    if not os.path.exists(target):
        print(f"circle_clusters_v1.py not found alongside {__file__}", file=sys.stderr)
        sys.exit(1)

    cmd = [sys.executable, target, igc_path]
    if args.clusters_csv:
        cmd += ["--clusters-csv", args.clusters_csv]
    if args.segments_csv:
        cmd += ["--segments-csv", args.segments_csv]

    # Echo command (helpful for logs/pycharm)
    print("Running:", " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(cmd)
    sys.exit(proc.returncode)

if __name__ == "__main__":
    main()
