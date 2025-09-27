
#!/usr/bin/env python3
"""
pipeline_v2f.py  (robust wrapper + enriched matcher with --outdir)

This version improves robustness and path handling:
- NEW: --outdir (default: outputs). All default file paths are under this directory.
- Accepts a single IGC path and passes it through child scripts.
- Tries multiple invocation modes for child scripts (IGC vs per-circle CSV).
- Discovers outputs if filenames differ from defaults.
- Allows explicit overrides for intermediate file paths.
- Writes enriched matched_clusters.csv with parameters from both cluster types.

Order:
  1) circles_from_brecords_v1d.py        -> per-circle CSV (default: {outdir}/circle_output.csv)
  2) circle_clusters_v1s.py              -> {outdir}/circle_clusters_enriched.csv
  3) overlay_altitude_clusters.py        -> {outdir}/overlay_altitude_clusters.csv
  4) [IN THIS SCRIPT] match & enrich     -> {outdir}/matched_clusters.csv

Usage examples:
  python pipeline_v2f.py --igc "data/flight.igc"
  python pipeline_v2f.py --igc "data/flight.igc" --outdir outputs --force
  python pipeline_v2f.py --circles outputs/circle_clusters_enriched.csv --alt outputs/overlay_altitude_clusters.csv
"""

import argparse
import os
import sys
import math
import time
import glob
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import numpy as np


# --------------------------- Helpers ---------------------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False


def _recent_csvs(patterns: List[str], within_seconds: int = 3600) -> List[Path]:
    """Return recent CSVs matching patterns, sorted by mtime desc."""
    now = time.time()
    paths = []
    for pat in patterns:
        for fp in glob.glob(pat):
            try:
                mtime = os.path.getmtime(fp)
                if now - mtime <= within_seconds:
                    paths.append((mtime, Path(fp)))
            except Exception:
                pass
    paths.sort(reverse=True)  # newest first
    return [fp for _, fp in paths]


def _run(script: str, args: List[str], mode: str) -> int:
    """Run a child script and stream output."""
    py = sys.executable
    print(f"[INFO] Running {script} ({mode}) -> {script} {' '.join(args)}")
    r = subprocess.run([py, script, *args], check=False, text=True)
    return r.returncode


def _run_try_modes(script: str, arg_sets: List[List[str]]) -> bool:
    """Try multiple argument patterns until success (exit code 0)."""
    for i, args in enumerate(arg_sets, 1):
        rc = _run(script, args, mode=f"mode{i}")
        if rc == 0:
            return True
        else:
            print(f"[WARN] {script} exited with {rc} using args: {args}")
    return False


# --------------------------- Matching utils --------------------------------

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _coerce_time_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    candidates = [
        ("start_time", "end_time"),
        ("start_ts", "end_ts"),
        ("t_start", "t_end"),
        ("start", "end"),
        ("start_s", "end_s"),
        ("start_sec", "end_sec"),
    ]
    cols = set(c.lower() for c in df.columns)
    lower = {c.lower(): c for c in df.columns}
    for s, e in candidates:
        if s in cols and e in cols:
            return lower[s], lower[e]
    # heuristic fallback
    start_guess = next((c for c in df.columns if c.lower().startswith("start")), None)
    end_guess   = next((c for c in df.columns if c.lower().startswith("end")), None)
    if start_guess and end_guess:
        return start_guess, end_guess
    return None, None


def _coerce_lat_lon(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    lat_candidates = ["lat", "latitude", "lat_deg"]
    lon_candidates = ["lon", "lng", "longitude", "lon_deg"]
    lower = {c.lower(): c for c in df.columns}
    lat = next((lower[x] for x in (c.lower() for c in lat_candidates) if x in lower), None)
    lon = next((lower[x] for x in (c.lower() for c in lon_candidates) if x in lower), None)
    return lat, lon


def _to_seconds(series: pd.Series) -> pd.Series:
    if np.issubdtype(series.dtype, np.number):
        return series.astype(float)
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if parsed.notna().any():
        return parsed.view("int64") / 1e9
    return pd.to_numeric(series, errors="coerce")


def _interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    start = max(a0, b0); end = min(a1, b1)
    return max(0.0, end - start)


def _id_col(df: pd.DataFrame) -> str:
    for cand in ["cluster_id", "id", "clusterID", "clusterId"]:
        if cand in df.columns:
            return cand
    raise ValueError("Expected a 'cluster_id' column (or id/clusterId variant).")


def match_clusters_enriched(circles_df: pd.DataFrame,
                            alt_df: pd.DataFrame,
                            dist_threshold_m: float = 500.0,
                            min_time_overlap_s: float = 10.0) -> pd.DataFrame:
    c_start, c_end = _coerce_time_columns(circles_df)
    a_start, a_end = _coerce_time_columns(alt_df)
    c_lat, c_lon = _coerce_lat_lon(circles_df)
    a_lat, a_lon = _coerce_lat_lon(alt_df)
    if not c_lat or not c_lon:
        raise ValueError("Circle clusters missing lat/lon.")
    if not a_lat or not a_lon:
        raise ValueError("Altitude clusters missing lat/lon.")

    c_t0 = _to_seconds(circles_df[c_start]) if (c_start and c_end) else None
    c_t1 = _to_seconds(circles_df[c_end])   if (c_start and c_end) else None
    a_t0 = _to_seconds(alt_df[a_start])     if (a_start and a_end) else None
    a_t1 = _to_seconds(alt_df[a_end])       if (a_start and a_end) else None

    c_id = _id_col(circles_df)
    a_id = _id_col(alt_df)

    matches = []
    a_idx = alt_df[[a_id, a_lat, a_lon]].copy()
    if a_t0 is not None and a_t1 is not None:
        a_idx["__t0"] = a_t0; a_idx["__t1"] = a_t1
    else:
        a_idx["__t0"] = np.nan; a_idx["__t1"] = np.nan

    for i, crow in circles_df.iterrows():
        clat = float(crow[c_lat]); clon = float(crow[c_lon])
        if c_t0 is not None and c_t1 is not None:
            ct0 = float(c_t0.loc[i]); ct1 = float(c_t1.loc[i])
        else:
            ct0 = np.nan; ct1 = np.nan

        for j, arow in a_idx.iterrows():
            d = _haversine_m(clat, clon, float(arow[a_lat]), float(arow[a_lon]))
            if d > dist_threshold_m:
                continue

            if not (np.isnan(ct0) or np.isnan(ct1) or np.isnan(arow["__t0"]) or np.isnan(arow["__t1"])):
                ovl = _interval_overlap(ct0, ct1, float(arow["__t0"]), float(arow["__t1"]))
                if ovl < min_time_overlap_s:
                    continue
                c_dur = max(0.0, ct1 - ct0); a_dur = max(0.0, float(arow["__t1"]) - float(arow["__t0"]))
                denom = max(1e-9, min(c_dur, a_dur))
                frac = ovl / denom
            else:
                ovl = np.nan; frac = np.nan

            matches.append({
                "circle_cluster_id": crow[a_id if 'circle' in a_id.lower() else 'cluster_id'] if False else crow[_id_col(circles_df)],
                "alt_cluster_id": arow[a_id],
                "dist_m": d,
                "time_overlap_s": ovl,
                "overlap_frac": frac,
                # Circle params
                "circle_av_climb_ms": crow.get("av_climb_ms", np.nan),
                "circle_alt_gained_m": crow.get("alt_gained_m", np.nan),
                "circle_lat": clat,
                "circle_lon": clon,
                # Alt params
                "alt_av_climb_ms": alt_df.loc[j].get("av_climb_ms", np.nan),
                "alt_alt_gained_m": alt_df.loc[j].get("alt_gained_m", np.nan),
                "alt_lat": float(arow[a_lat]),
                "alt_lon": float(arow[a_lon]),
            })

    return pd.DataFrame(matches, columns=[
        "circle_cluster_id", "alt_cluster_id", "dist_m",
        "time_overlap_s", "overlap_frac",
        "circle_av_climb_ms", "circle_alt_gained_m", "circle_lat", "circle_lon",
        "alt_av_climb_ms", "alt_alt_gained_m", "alt_lat", "alt_lon",
    ])


# --------------------------- Pipeline orchestration ------------------------

def run_pipeline(igc_path: Path,
                 circle_out: Path,
                 circle_clusters_csv: Path,
                 alt_clusters_csv: Path,
                 force: bool = False) -> Tuple[Path, Path]:
    """
    Run upstream scripts with the same IGC path.
    Tries multiple invocation patterns and discovers outputs if not present.
    Returns (circle_clusters_csv, alt_clusters_csv).
    """

    # 1) circles_from_brecords_v1d.py -> per-circle CSV
    if force or not _exists(circle_out):
        ok = _run_try_modes("circles_from_brecords_v1d.py", [
            [str(igc_path)],                  # mode1: script expects IGC
        ])
        if not ok:
            print("[WARN] circles_from_brecords_v1d.py did not exit cleanly. Will attempt to discover per-circle CSV.")
    if not _exists(circle_out):
        candidates = _recent_csvs([str(circle_out), str(circle_out.parent / "*circle*.csv")], within_seconds=24*3600)
        if candidates:
            print(f"[INFO] Using discovered per-circle CSV: {candidates[0]}")
            circle_out = candidates[0]
        else:
            print(f"[ERROR] Could not find per-circle CSV (expected like '{circle_out}'). You may set --circle-out.", file=sys.stderr)
            raise FileNotFoundError(str(circle_out))

    # 2) circle_clusters_v1s.py -> circle_clusters_enriched.csv
    if force or not _exists(circle_clusters_csv):
        ok = _run_try_modes("circle_clusters_v1s.py", [
            [str(igc_path)],           # mode1: expects IGC
            [str(circle_out)],         # mode2: expects per-circle CSV
        ])
        if not ok:
            print("[WARN] circle_clusters_v1s.py did not exit cleanly.")
    if not _exists(circle_clusters_csv):
        candidates = _recent_csvs([
            str(circle_clusters_csv),
            str(circle_clusters_csv.parent / "*cluster*enriched*.csv"),
            str(circle_clusters_csv.parent / "*circle*cluster*.csv"),
        ], within_seconds=24*3600)
        if candidates:
            print(f"[INFO] Using discovered circle clusters CSV: {candidates[0]}")
            circle_clusters_csv = candidates[0]
        else:
            raise FileNotFoundError(f"Expected '{circle_clusters_csv}' was not created. Check circle_clusters_v1s.py run/paths.")

    # 3) overlay_altitude_clusters.py -> overlay_altitude_clusters.csv
    if force or not _exists(alt_clusters_csv):
        ok = _run_try_modes("overlay_altitude_clusters.py", [
            [str(igc_path)],           # most versions expect IGC
        ])
        if not ok:
            print("[WARN] overlay_altitude_clusters.py did not exit cleanly.")
    if not _exists(alt_clusters_csv):
        candidates = _recent_csvs([
            str(alt_clusters_csv),
            str(alt_clusters_csv.parent / "*overlay*altitude*clusters*.csv"),
            str(alt_clusters_csv.parent / "*altitude*clusters*.csv"),
        ], within_seconds=24*3600)
        if candidates:
            print(f"[INFO] Using discovered altitude clusters CSV: {candidates[0]}")
            alt_clusters_csv = candidates[0]
        else:
            raise FileNotFoundError(f"Expected '{alt_clusters_csv}' was not created. Check overlay_altitude_clusters.py run/paths.")

    return circle_clusters_csv, alt_clusters_csv


# --------------------------- CLI -------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Robust pipeline wrapper: pass the same IGC through, adapt to child scripts, and write enriched matches.")
    ap.add_argument("--igc", default="2020-11-08 Lumpy Paterson 108645.igc",
                    help="Path to the IGC file (default: %(default)s).")
    ap.add_argument("--outdir", default="outputs",
                    help="Directory for default inputs/outputs (default: %(default)s).")
    ap.add_argument("--force", action="store_true",
                    help="Re-run child steps even if outputs already exist.")
    # explicit overrides for intermediate files (resolved relative to outdir if not absolute)
    ap.add_argument("--circle-out", default=None,
                    help="Per-circle CSV from circles_from_brecords_v1d.py (default: {outdir}/circle_output.csv).")
    ap.add_argument("--circle-clusters", default=None,
                    help="Circle clusters CSV (default: {outdir}/circle_clusters_enriched.csv).")
    ap.add_argument("--alt-clusters", default=None,
                    help="Altitude clusters CSV (default: {outdir}/overlay_altitude_clusters.csv).")
    # direct mode (skip children)
    ap.add_argument("--circles", default=None,
                    help="Direct path to circle_clusters_enriched.csv (skip running child if provided).")
    ap.add_argument("--alt", default=None,
                    help="Direct path to overlay_altitude_clusters.csv (skip running child if provided).")
    ap.add_argument("--out", default=None,
                    help="Output CSV for matches (default: {outdir}/matched_clusters.csv).")
    ap.add_argument("--dist-threshold-m", type=float, default=500.0,
                    help="Max distance (m) for spatial match (default: 500).")
    ap.add_argument("--min-time-overlap-s", type=float, default=10.0,
                    help="Minimum time overlap (s) to accept a match (default: 10).")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    _ensure_dir(outdir)

    def _resolve(p_opt: Optional[str], default_name: str) -> Path:
        if p_opt:
            p = Path(p_opt)
            return p if p.is_absolute() else (outdir / p)
        return outdir / default_name

    circle_out = _resolve(args.circle_out, "circle_output.csv")
    circle_clusters_csv = _resolve(args.circle_clusters, "circle_clusters_enriched.csv")
    alt_clusters_csv = _resolve(args.alt_clusters, "overlay_altitude_clusters.csv")
    out_csv = _resolve(args.out, "matched_clusters.csv")

    # Either use provided CSVs directly or run pipeline
    if args.circles and args.alt:
        circle_clusters_csv = Path(args.circles)
        alt_clusters_csv = Path(args.alt)
        if not circle_clusters_csv.is_absolute():
            circle_clusters_csv = outdir / circle_clusters_csv
        if not alt_clusters_csv.is_absolute():
            alt_clusters_csv = outdir / alt_clusters_csv
    else:
        circle_clusters_csv, alt_clusters_csv = run_pipeline(
            igc_path=Path(args.igc),
            circle_out=circle_out,
            circle_clusters_csv=circle_clusters_csv,
            alt_clusters_csv=alt_clusters_csv,
            force=args.force,
        )

    # Load and match
    circles_df = pd.read_csv(circle_clusters_csv)
    alt_df = pd.read_csv(alt_clusters_csv)
    out_df = match_clusters_enriched(circles_df, alt_df,
                                     dist_threshold_m=args.dist_threshold_m,
                                     min_time_overlap_s=args.min_time_overlap_s)
    out_df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote {out_csv} with {len(out_df)} matches.")
    if len(out_df) > 0:
        print(out_df.head(min(5, len(out_df))).to_string(index=False))


if __name__ == "__main__":
    main()
