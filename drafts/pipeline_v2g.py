
#!/usr/bin/env python3
"""
pipeline_v2g.py — orchestrator (always fresh run for PyCharm)

Changes in this version:
- Always prompts for the IGC path (even if --igc_subset is provided).
- Always re-runs all upstream steps (no "exists → skip" logic).
- Deletes prior CSVs in outputs/ before running, so new results overwrite cleanly.
- Delegates matching to your existing match_clusters_v1.py.
- Uses versioned upstreams:
    * circles_from_brecords_v1d.py
    * circle_clusters_v1s.py
    * overlay_altitude_clusters_v1c.py
    * match_clusters_v1.py

Usage (PyCharm):
- Just click Run. You'll be prompted for the IGC path each time.
"""

import argparse
import sys
import subprocess
from pathlib import Path

# Versioned script names
CIRCLES_SCRIPT = "circles_from_brecords_v1d.py"
CIRCLE_CLUSTERS_SCRIPT = "circle_clusters_v1s.py"
ALT_CLUSTERS_SCRIPT = "overlay_altitude_clusters_v1c.py"
MATCH_SCRIPT = "match_clusters_v1.py"

DEFAULT_OUTDIR = "outputs"
DEFAULT_IGC = "2020-11-08 Lumpy Paterson 108645.igc_subset"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def run(cmd) -> int:
    print(f"[RUN] {' '.join(cmd)}")
    return subprocess.call(cmd)

def prompt_for_igc(default_path: Path) -> Path:
    try:
        inp = input(f"Enter path to IGC file [{default_path}]: ").strip()
        p = Path(inp) if inp else default_path
        return p
    except EOFError:
        return default_path

def main():
    ap = argparse.ArgumentParser(description="Always-fresh orchestrator: re-run upstreams, overwrite outputs, prompt for IGC every time")
    ap.add_argument("--igc_subset", default=DEFAULT_IGC, help=f"Path to the IGC file (default: {DEFAULT_IGC})")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Directory for inputs/outputs (default: %(default)s)")
    # keep args for explicit overrides if ever needed
    ap.add_argument("--circle-out", default=None, help="Override per-circle CSV path (default: {outdir}/circles.csv)")
    ap.add_argument("--circle-clusters", default=None, help="Override circle clusters CSV path (default: {outdir}/circle_clusters_enriched.csv)")
    ap.add_argument("--alt-clusters", default=None, help="Override altitude clusters CSV path (default: {outdir}/overlay_altitude_clusters.csv)")
    ap.add_argument("--out", default=None, help="Override matched output CSV (default: {outdir}/matched_clusters.csv)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    # Always prompt for IGC (PyCharm-friendly)
    igc_path = prompt_for_igc(Path(args.igc))

    # Resolve expected file paths
    circle_out = Path(args.circle_out) if args.circle_out else outdir / "circles.csv"
    circle_clusters = Path(args.circle_clusters) if args.circle_clusters else outdir / "circle_clusters_enriched.csv"
    alt_clusters = Path(args.alt_clusters) if args.alt_clusters else outdir / "overlay_altitude_clusters.csv"
    matched_out = Path(args.out) if args.out else outdir / "matched_clusters.csv"

    # Always start clean: delete prior CSVs if present
    for p in (circle_out, circle_clusters, alt_clusters, matched_out):
        if p.exists():
            try:
                p.unlink()
                print(f"[CLEAN] Removed previous: {p}")
            except Exception as e:
                print(f"[WARN] Could not remove {p}: {e}")

    py = sys.executable

    # Run upstreams (always)
    rc = run([py, CIRCLES_SCRIPT, str(igc_path)]);                  assert rc == 0, "circles step failed"
    rc = run([py, CIRCLE_CLUSTERS_SCRIPT, str(igc_path)]);          assert rc == 0, "circle clusters step failed"
    rc = run([py, ALT_CLUSTERS_SCRIPT, str(igc_path)]);             assert rc == 0, "altitude clusters step failed"

    # Run matcher (delegate)
    rc = run([py, MATCH_SCRIPT, str(circle_clusters), str(alt_clusters), str(matched_out)]);  assert rc == 0, "matching step failed"

    print(f"[OK] Pipeline completed. Fresh matched output at: {matched_out}")

if __name__ == "__main__":
    main()
