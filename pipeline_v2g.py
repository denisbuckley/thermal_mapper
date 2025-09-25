
#!/usr/bin/env python3
"""
pipeline_v2g.py  — simple, non-destructive orchestrator

Purpose:
- Orchestrate the existing upstream scripts and then call your existing
  `match_clusters_v1.py`. This wrapper does NOT implement its own matcher.
- Avoid clobbering upstream outputs. It only triggers scripts; it never
  rewrites intermediate CSVs itself.
- Defaults to using an `--outdir` (outputs/) for all expected files.

Order:
  1) circles_from_brecords_v1d.py         -> {outdir}/circle_output.csv (or whatever your script writes)
  2) circle_clusters_v1s.py               -> {outdir}/circle_clusters_enriched.csv
  3) overlay_altitude_clusters.py         -> {outdir}/overlay_altitude_clusters.csv
  4) match_clusters_v1.py                 -> {outdir}/matched_clusters.csv

This wrapper:
- Calls each script via CLI (no stdin magic). If a step already produced the expected
  file and --force is not set, it will skip re-running that step.
- Allows explicit overrides for each expected file.
- Delegates all matching to match_clusters_v1.py and does not modify its outputs.

Usage:
  python pipeline_v2g.py --igc "data/flight.igc"
  python pipeline_v2g.py --igc "data/flight.igc" --outdir outputs --force --verbose
  python pipeline_v2g.py --circles outputs/circle_clusters_enriched.csv --alt outputs/overlay_altitude_clusters.csv
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

DEFAULT_OUTDIR = "outputs"

def log(msg: str, verbose: bool):
    if verbose:
        print(msg)

def exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def run(cmd, verbose: bool) -> int:
    log(f"[RUN] {' '.join(cmd)}", verbose)
    return subprocess.call(cmd)

def resolve(base: Path, value: Optional[str], default_name: str) -> Path:
    if value:
        p = Path(value)
        return p if p.is_absolute() else (base / p)
    return base / default_name

def main():
    ap = argparse.ArgumentParser(description="Simple orchestrator: run upstream scripts and call match_clusters_v1.py")
    ap.add_argument("--igc", required=False, default="2020-11-08 Lumpy Paterson 108645.igc",
                    help="Path to the IGC file (default: %(default)s)")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Directory for expected inputs/outputs (default: %(default)s)")
    ap.add_argument("--force", action="store_true", help="Re-run child steps even if expected outputs exist")
    ap.add_argument("--dry-run", action="store_true", help="Print what would run, without running anything")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")
    ap.add_argument("--no-clobber", action="store_true", help="Abort if matched output already exists")

    # Explicit overrides for expected files
    ap.add_argument("--circle-out", default=None, help="Per-circle CSV path (default: {outdir}/circle_output.csv)")
    ap.add_argument("--circle-clusters", default=None, help="Circle clusters CSV path (default: {outdir}/circle_clusters_enriched.csv)")
    ap.add_argument("--alt-clusters", default=None, help="Altitude clusters CSV path (default: {outdir}/overlay_altitude_clusters.csv)")
    ap.add_argument("--out", default=None, help="Matched output CSV (default: {outdir}/matched_clusters.csv)")

    # Direct mode to skip running upstream children
    ap.add_argument("--circles", default=None, help="Direct path to circle_clusters_enriched.csv (skip running child if provided)")
    ap.add_argument("--alt", default=None, help="Direct path to overlay_altitude_clusters.csv (skip running child if provided)")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    circle_out = resolve(outdir, args.circle_out, "circle_output.csv")
    circle_clusters = resolve(outdir, args.circle_clusters, "circle_clusters_enriched.csv")
    alt_clusters = resolve(outdir, args.alt_clusters, "overlay_altitude_clusters.csv")
    matched_out = resolve(outdir, args.out, "matched_clusters.csv")

    # If user provided direct CSVs for matching, respect them
    if args.circles:
        circle_clusters = Path(args.circles) if Path(args.circles).is_absolute() else outdir / args.circles
    if args.alt:
        alt_clusters = Path(args.alt) if Path(args.alt).is_absolute() else outdir / args.alt

    cmds = []
    py = sys.executable

    # Upstream step 1
    if not args.circles and (args.force or not exists(circle_out)):
        cmds.append([py, "circles_from_brecords_v1d.py", args.igc])
    else:
        log(f"[SKIP] circles_from_brecords_v1d.py (exists: {exists(circle_out)}, force: {args.force}, circles override: {bool(args.circles)})", args.verbose)

    # Upstream step 2
    if not args.circles and (args.force or not exists(circle_clusters)):
        # Your clusterer may accept the IGC or the per-circle CSV; use IGC by default (keeps behavior consistent with your manual runs)
        cmds.append([py, "circle_clusters_v1s.py", args.igc])
    else:
        log(f"[SKIP] circle_clusters_v1s.py (exists: {exists(circle_clusters)} or circles override supplied)", args.verbose)

    # Upstream step 3
    if not args.alt and (args.force or not exists(alt_clusters)):
        cmds.append([py, "overlay_altitude_clusters.py", args.igc])
    else:
        log(f"[SKIP] overlay_altitude_clusters.py (exists: {exists(alt_clusters)} or alt override supplied)", args.verbose)

    # Matching step (delegated to your matcher); do not clobber unless asked
    if args.no_clobber and exists(matched_out):
        print(f"[ABORT] Matched output already exists: {matched_out}")
        sys.exit(1)

    match_cmd = [py, "match_clusters_v1.py", str(circle_clusters), str(alt_clusters), str(matched_out)]
    cmds.append(match_cmd)

    # Dry-run prints and exits
    if args.dry_run:
        print("\n".join(" ".join(c) for c in cmds))
        sys.exit(0)

    # Execute commands
    for c in cmds:
        rc = run(c, args.verbose)
        if rc != 0:
            print(f"[ERROR] Command failed with code {rc}: {' '.join(c)}", file=sys.stderr)
            sys.exit(rc)

    print(f"[OK] Pipeline completed. Matched output at: {matched_out}")

if __name__ == "__main__":
    main()
