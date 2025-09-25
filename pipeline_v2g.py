
#!/usr/bin/env python3
"""
pipeline_v2g.py — non-destructive orchestrator (versioned upstreams) with IGC prompt

Upstreams (versioned):
  1) circles_from_brecords_v1d.py
  2) circle_clusters_v1s.py
  3) overlay_altitude_clusters_v1c.py
  4) match_clusters_v1.py

Behavior:
- Calls upstreams in order, then delegates matching to match_clusters_v1.py.
- Does NOT implement its own matcher and does NOT alter intermediates.
- Uses --outdir (default: outputs/) for expected files; skips steps unless --force.
- Supports --dry-run, --verbose, and --no-clobber for the final matched output.
- Prompts for an IGC path if:
    * --ask is provided, or
    * the default/given --igc path does not exist.
"""

import argparse
import sys
import subprocess
from pathlib import Path
from typing import Optional

# Versioned script names
CIRCLES_SCRIPT = "circles_from_brecords_v1d.py"
CIRCLE_CLUSTERS_SCRIPT = "circle_clusters_v1s.py"
ALT_CLUSTERS_SCRIPT = "overlay_altitude_clusters_v1c.py"
MATCH_SCRIPT = "match_clusters_v1.py"

DEFAULT_OUTDIR = "outputs"
DEFAULT_IGC = "2020-11-08 Lumpy Paterson 108645.igc"

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

def prompt_for_igc(default_path: Path) -> Path:
    try:
        inp = input(f"Enter path to IGC file [{default_path}]: ").strip()
        p = Path(inp) if inp else default_path
        return p
    except EOFError:
        # Non-interactive: fall back to default
        return default_path

def main():
    ap = argparse.ArgumentParser(description="Simple orchestrator: run versioned upstream scripts and call match_clusters_v1.py")
    ap.add_argument("--igc", default=DEFAULT_IGC,
                    help=f"Path to the IGC file (default: {DEFAULT_IGC})")
    ap.add_argument("--ask", action="store_true",
                    help="Always prompt for the IGC path (even if --igc is set).")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Directory for expected inputs/outputs (default: %(default)s)")
    ap.add_argument("--force", action="store_true", help="Re-run child steps even if expected outputs exist")
    ap.add_argument("--dry-run", action="store_true", help="Print what would run, without running anything")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")
    ap.add_argument("--no-clobber", action="store_true", help="Abort if matched output already exists")

    # Explicit overrides for expected files
    ap.add_argument("--circle-out", default=None, help="Per-circle CSV path (default: {outdir}/circles.csv)")
    ap.add_argument("--circle-clusters", default=None, help="Circle clusters CSV path (default: {outdir}/circle_clusters_enriched.csv)")
    ap.add_argument("--alt-clusters", default=None, help="Altitude clusters CSV path (default: {outdir}/overlay_altitude_clusters.csv)")
    ap.add_argument("--out", default=None, help="Matched output CSV (default: {outdir}/matched_clusters.csv)")

    # Direct mode to skip running upstream children
    ap.add_argument("--circles", default=None, help="Direct path to circle_clusters_enriched.csv (skip running child if provided)")
    ap.add_argument("--alt", default=None, help="Direct path to overlay_altitude_clusters.csv (skip running child if provided)")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    # Decide IGC path (prompt if requested or if the given path doesn't exist)
    igc_path = prompt_for_igc(Path(args.igc))
    # No hard failure here: we pass the path through to children; if it's wrong they will complain.

    circle_out = resolve(outdir, args.circle_out, "circles.csv")
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
        cmds.append([py, CIRCLES_SCRIPT, str(igc_path)])
    else:
        log(f"[SKIP] {CIRCLES_SCRIPT} (exists: {exists(circle_out)}, force: {args.force}, circles override: {bool(args.circles)})", args.verbose)

    # Upstream step 2
    if not args.circles and (args.force or not exists(circle_clusters)):
        cmds.append([py, CIRCLE_CLUSTERS_SCRIPT, str(igc_path)])
    else:
        log(f"[SKIP] {CIRCLE_CLUSTERS_SCRIPT} (exists: {exists(circle_clusters)} or circles override supplied)", args.verbose)

    # Upstream step 3
    if not args.alt and (args.force or not exists(alt_clusters)):
        cmds.append([py, ALT_CLUSTERS_SCRIPT, str(igc_path)])
    else:
        log(f"[SKIP] {ALT_CLUSTERS_SCRIPT} (exists: {exists(alt_clusters)} or alt override supplied)", args.verbose)

    # Matching step (delegated); do not clobber unless asked
    if args.no_clobber and exists(matched_out):
        print(f"[ABORT] Matched output already exists: {matched_out}")
        sys.exit(1)

    match_cmd = [py, MATCH_SCRIPT, str(circle_clusters), str(alt_clusters), str(matched_out)]
    cmds.append(match_cmd)

    # Dry-run
    if args.dry_run:
        print("\n".join(" ".join(c) for c in cmds))
        sys.exit(0)

    # Execute
    for c in cmds:
        rc = run(c, args.verbose)
        if rc != 0:
            print(f"[ERROR] Command failed with code {rc}: {' '.join(c)}", file=sys.stderr)
            sys.exit(rc)

    print(f"[OK] Pipeline completed. Matched output at: {matched_out}")

if __name__ == "__main__":
    main()
