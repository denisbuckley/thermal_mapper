#!/usr/bin/env python3
"""
sweep_eps_for_thermals.py
-------------------------
Iterates over --eps-km values, calls build_thermals_v1a.py for each,
and reports the eps that yields the **maximum number of thermals** (rows
in the output waypoints CSV). Tie-breakers: total encounters (desc), then eps (asc).

Usage (IDE run or CLI):
  python sweep_eps_for_thermals.py \
    --inputs-root outputs/batch_csv \
    --out-dir outputs/waypoints \
    --eps-start 0.2 --eps-end 2.0 --eps-step 0.1 \
    --min-samples 5 \
    --strength-min 2 \
    [--method dbscan] \
    [--center-estimator median] \
    [--apply-best]      # re-run build_thermals_v1a.py with the best eps to write canonical outputs

You can also pass an explicit list:
  --eps-list 0.3 0.5 0.8 1.0 1.5
"""

import argparse, csv, itertools, math, subprocess, sys, tempfile
from pathlib import Path

def frange(start: float, end: float, step: float):
    """Inclusive-ish float range (avoids numpy)."""
    n = max(1, int(math.floor((end - start) / step + 1e-9)) + 1)
    for i in range(n):
        yield round(start + i * step, 10)  # tame FP drift

def read_waypoint_count(csv_path: Path):
    """Return (rows, total_encounters) from a waypoints CSV; 0 if file missing/empty."""
    if not csv_path.exists():
        return 0, 0.0
    rows = 0
    enc_sum = 0.0
    with csv_path.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows += 1
            # 'encounters' might be string-formatted; try float
            try:
                enc_sum += float(row.get("encounters", 0) or 0)
            except Exception:
                pass
    return rows, enc_sum

def call_builder(
    script_path: Path,
    inputs_root: Path,
    out_dir: Path,
    eps_km: float,
    min_samples: int,
    strength_min: float,
    method: str,
    center_estimator: str,
    date_start: str | None,
    date_end: str | None,
    verbose: bool
):
    """
    Run build_thermals_v1a.py once with given eps. Writes to a temp subdir
    (so runs don't clobber each other). Returns (rows, encounters_sum, out_csv_path).
    """
    run_dir = out_dir / f"_sweep_eps_{eps_km:.3f}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_csv = run_dir / "thermal_waypoints_v1.csv"
    out_geo = run_dir / "thermal_waypoints_v1.geojson"
    debug_log = run_dir / "grouping_debug.log"

    cmd = [
        sys.executable, str(script_path),
        "--inputs-root", str(inputs_root),
        "--out-csv", str(out_csv),
        "--out-geojson", str(out_geo),
        "--debug-log", str(debug_log),
        "--method", method,
        "--eps-km", str(eps_km),
        "--min-samples", str(min_samples),
        "--strength-min", str(strength_min),
        "--center-estimator", center_estimator,
    ]
    if date_start: cmd += ["--date-start", date_start]
    if date_end:   cmd += ["--date-end", date_end]

    # Inherit env; capture output to surface errors if any
    res = subprocess.run(cmd, capture_output=True, text=True)
    if verbose:
        print(f"\n=== eps={eps_km:.3f} ===")
        print(res.stdout)
        if res.returncode != 0:
            print(res.stderr, file=sys.stderr)

    rows, enc_sum = read_waypoint_count(out_csv)
    return rows, enc_sum, out_csv, res.returncode, res.stderr.strip()

def parse_args():
    ap = argparse.ArgumentParser(description="Sweep eps for build_thermals_v1a.py and pick the best.")
    ap.add_argument("--script", default="build_thermals_v1a.py", help="Path to build_thermals_v1a.py")
    ap.add_argument("--inputs-root", default="outputs/batch_csv")
    ap.add_argument("--out-dir", default="outputs/waypoints")
    ap.add_argument("--min-samples", type=int, default=5)
    ap.add_argument("--strength-min", type=float, default=2.0)
    ap.add_argument("--method", choices=["dbscan","hdbscan","optics"], default="dbscan")
    ap.add_argument("--center-estimator", choices=["geomedian","median","medoid"], default="median")
    ap.add_argument("--date-start", default=None)
    ap.add_argument("--date-end", default=None)
    ap.add_argument("--verbose", action="store_true")

    # eps options
    grp = ap.add_mutually_exclusive_group(required=False)
    grp.add_argument("--eps-list", nargs="+", type=float, help="Explicit eps values (km).")
    ap.add_argument("--eps-start", type=float, default=0.2)
    ap.add_argument("--eps-end",   type=float, default=2.0)
    ap.add_argument("--eps-step",  type=float, default=0.1)

    # apply best
    ap.add_argument("--apply-best", action="store_true", help="Re-run build_thermals_v1a.py with best eps to write canonical outputs into --out-dir")
    return ap.parse_args()

def main():
    args = parse_args()
    script_path = Path(args.script).resolve()
    inputs_root = Path(args.inputs_root).resolve()
    out_dir     = Path(args.out_dir).resolve()

    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build eps candidates
    if args.eps_list:
        eps_values = [float(x) for x in args.eps_list]
    else:
        eps_values = list(frange(args.eps_start, args.eps_end, args.eps_step))
        # ensure endpoint included if close
        if eps_values[-1] < args.eps_end and (args.eps_end - eps_values[-1]) > 1e-9:
            eps_values.append(args.eps_end)
    eps_values = sorted(set(round(x, 6) for x in eps_values))

    results = []  # (eps, rows, enc_sum, out_csv)
    failures = []

    print(f"[INFO] Sweeping eps values: {', '.join(f'{e:.3f}' for e in eps_values)}")
    for eps in eps_values:
        rows, enc_sum, out_csv, rc, err = call_builder(
            script_path=script_path,
            inputs_root=inputs_root,
            out_dir=out_dir,
            eps_km=eps,
            min_samples=args.min_samples,
            strength_min=args.strength_min,
            method=args.method,
            center_estimator=args.center_estimator,
            date_start=args.date_start,
            date_end=args.date_end,
            verbose=args.verbose,
        )
        if rc != 0:
            failures.append((eps, rc, err))
        results.append((eps, rows, enc_sum, out_csv))
        print(f"[RESULT] eps={eps:.3f} → thermals={rows} (encounters sum={enc_sum:.1f})")

    # Pick best: max rows, then max encounters sum, then min eps
    best = max(results, key=lambda t: (t[1], t[2], -t[0])) if results else None

    print("\n=== SUMMARY ===")
    if failures:
        print("[WARN] Some runs failed:")
        for eps, rc, err in failures:
            print(f"  eps={eps:.3f} rc={rc} err={err}")
    if best:
        best_eps, best_rows, best_enc, best_csv = best
        print(f"[BEST] eps={best_eps:.3f} → thermals={best_rows}, encounters sum={best_enc:.1f}")
        print(f"[BEST CSV SAMPLE] {best_csv}")

        if args.apply_best:
            # Re-run builder to write canonical outputs in out_dir (without sweep subfolder)
            canonical_csv = out_dir / "thermal_waypoints_v1.csv"
            canonical_geo = out_dir / "thermal_waypoints_v1.geojson"
            canonical_log = out_dir / "grouping_debug.log"
            print(f"[APPLY] Re-running with eps={best_eps:.3f} to write canonical outputs...")
            cmd = [
                sys.executable, str(script_path),
                "--inputs-root", str(inputs_root),
                "--out-csv", str(canonical_csv),
                "--out-geojson", str(canonical_geo),
                "--debug-log", str(canonical_log),
                "--method", args.method,
                "--eps-km", str(best_eps),
                "--min-samples", str(args.min_samples),
                "--strength-min", str(args.strength_min),
                "--center-estimator", args.center_estimator,
            ]
            if args.date_start: cmd += ["--date-start", args.date_start]
            if args.date_end:   cmd += ["--date-end", args.date_end]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print("[APPLY ERROR]\n" + res.stderr, file=sys.stderr)
                return res.returncode
            print("[APPLY OK] Wrote canonical outputs.")
    else:
        print("[ERROR] No successful runs.")
        return 1

    return 0

if __name__ == "__main__":
    raise SystemExit(main())