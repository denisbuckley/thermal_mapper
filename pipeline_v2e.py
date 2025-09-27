
#!/usr/bin/env python3
"""
pipeline_v2e.py  (wrapper + enriched matcher)

This wrapper asks for ONE IGC filename (with a default), runs the upstream
scripts in sequence, and then performs the enriched cluster matching.

Order:
  1) circles_from_brecords_v1d.py        (reads IGC, writes per-circle CSV)
  2) circle_clusters_v1s.py              (clusters circles -> circle_clusters_enriched.csv)
  3) overlay_altitude_clusters.py        (detects altitude clusters -> overlay_altitude_clusters.csv)
  4) [IN THIS SCRIPT] match & enrich -> matched_clusters.csv

Design goals:
  - Use the SAME IGC path for every step.
  - Works whether the child scripts accept CLI args OR prompt for input().
  - Skips re-running a step if its expected output already exists (unless --force).
  - Enriched fields: circle_av_climb_ms, circle_alt_gained_m, circle_lat, circle_lon,
                     alt_av_climb_ms,    alt_alt_gained_m,    alt_lat,   alt_lon

Usage:
  python pipeline_v2e.py
  python pipeline_v2e.py --igc "path/to/file.igc"
  python pipeline_v2e.py --force  # re-run all steps
  python pipeline_v2e.py --circles circle_clusters_enriched.csv --alt overlay_altitude_clusters.csv
"""

import argparse
import os
import sys
import math
import subprocess
from typing import Optional, Tuple

import pandas as pd
import numpy as np


# --------------------------- Utilities -------------------------------------

def _prompt_igc(default_igc: str) -> str:
    try:
        user = input(f"Enter IGC path [{default_igc}]: ").strip()
        return user or default_igc
    except EOFError:
        # Non-interactive: just return default
        return default_igc


def _exists(path: str) -> bool:
    try:
        return os.path.exists(path)
    except Exception:
        return False


def _run_child_try_cli_then_stdin(script: str, igc_path: str, timeout: int = 1800) -> int:
    """
    Try to run a child script by first passing the IGC path as a CLI arg.
    If that fails (non-zero exit), try running without args and feed igc_path via stdin,
    in case the script prompts for input().
    """
    py = sys.executable

    # Attempt 1: Pass IGC as CLI argument
    try:
        print(f"[INFO] Running: {script} (CLI) with IGC: {igc_path}")
        r = subprocess.run([py, script, igc_path], check=False, capture_output=True, text=True, timeout=timeout)
        print(r.stdout)
        if r.returncode == 0:
            return 0
        else:
            print(f"[WARN] {script} exited with code {r.returncode} using CLI arg; stderr:\n{r.stderr}")
    except FileNotFoundError:
        print(f"[ERROR] Could not locate {script} in current directory.", file=sys.stderr)
        return 127
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Timeout running {script} (CLI).", file=sys.stderr)
        return 124

    # Attempt 2: Run without args and feed igc via stdin (for interactive scripts)
    try:
        print(f"[INFO] Retrying: {script} (STDIN) with IGC: {igc_path}")
        p = subprocess.Popen([py, script], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = p.communicate(input=f"{igc_path}\n", timeout=timeout)
        print(out)
        if p.returncode != 0:
            print(f"[ERROR] {script} failed (STDIN). Code {p.returncode}. Stderr:\n{err}", file=sys.stderr)
        return p.returncode
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Timeout running {script} (STDIN).", file=sys.stderr)
        return 124


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
    lower_map = {c.lower(): c for c in df.columns}
    for s, e in candidates:
        if s in cols and e in cols:
            return lower_map[s], lower_map[e]
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
    # find columns
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
                "circle_cluster_id": crow[c_id],
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


# --------------------------- Wrapper Steps ----------------------------------

def run_pipeline(igc_path: str, force: bool = False) -> Tuple[str, str]:
    """
    Run upstream scripts with the same IGC path.
    Returns (circle_clusters_csv, altitude_clusters_csv) paths.
    """
    # Expected outputs (conventional names from your setup)
    circles_out = "circle_output.csv"  # produced by circles_from_brecords_v1d.py (assumed)
    circle_clusters_csv = "circle_clusters_enriched.csv"
    alt_clusters_csv    = "overlay_altitude_clusters.csv"

    # 1) circles_from_brecords_v1d.py
    if force or not _exists(circles_out):
        rc = _run_child_try_cli_then_stdin("circles_from_brecords_v1d.py", igc_path)
        if rc != 0:
            print("[WARN] circles_from_brecords_v1d.py did not complete successfully. Continuing if downstream inputs exist.")

    # 2) circle_clusters_v1s.py
    if force or not _exists(circle_clusters_csv):
        rc = _run_child_try_cli_then_stdin("circle_clusters_v1s.py", igc_path)
        if rc != 0:
            print("[WARN] circle_clusters_v1s.py did not complete successfully.")

    # 3) overlay_altitude_clusters.py
    if force or not _exists(alt_clusters_csv):
        rc = _run_child_try_cli_then_stdin("overlay_altitude_clusters.py", igc_path)
        if rc != 0:
            print("[WARN] overlay_altitude_clusters.py did not complete successfully.")

    # Verify expected outputs
    if not _exists(circle_clusters_csv):
        raise FileNotFoundError(f"Expected '{circle_clusters_csv}' was not created. Check circle_clusters_v1s.py run/paths.")
    if not _exists(alt_clusters_csv):
        raise FileNotFoundError(f"Expected '{alt_clusters_csv}' was not created. Check overlay_altitude_clusters.py run/paths.")

    return circle_clusters_csv, alt_clusters_csv


# --------------------------- CLI -------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Pipeline wrapper: prompt for IGC, run steps, and match clusters (enriched).")
    ap.add_argument("--igc", default="2020-11-08 Lumpy Paterson 108645.igc",
                    help="Path to the IGC file (default: %(default)s). If missing, you'll be prompted.")
    ap.add_argument("--force", action="store_true",
                    help="Re-run child steps even if outputs already exist.")
    ap.add_argument("--dist-threshold-m", type=float, default=500.0,
                    help="Max distance (m) for spatial match (default: 500).")
    ap.add_argument("--min-time-overlap-s", type=float, default=10.0,
                    help="Minimum time overlap (s) to accept a match (default: 10).")
    ap.add_argument("--circles", default=None,
                    help="Optional direct path to circle_clusters_enriched.csv (skip running child if provided).")
    ap.add_argument("--alt", default=None,
                    help="Optional direct path to overlay_altitude_clusters.csv (skip running child if provided).")
    ap.add_argument("--out", default="matched_clusters.csv",
                    help="Output CSV for matches (default: matched_clusters.csv).")
    args = ap.parse_args()

    igc_path = args.igc
    if not _exists(igc_path):
        print(f"[INFO] IGC file not found at '{igc_path}'. You can press Enter to accept the default.")
        igc_path = _prompt_igc(args.igc)

    # Either use provided CSVs or run the upstream steps
    if args.circles and args.alt:
        circle_clusters_csv = args.circles
        alt_clusters_csv = args.alt
    else:
        circle_clusters_csv, alt_clusters_csv = run_pipeline(igc_path, force=args.force)

    # Load and match
    circles_df = pd.read_csv(circle_clusters_csv)
    alt_df = pd.read_csv(alt_clusters_csv)
    out_df = match_clusters_enriched(circles_df, alt_df,
                                     dist_threshold_m=args.dist_threshold_m,
                                     min_time_overlap_s=args.min_time_overlap_s)
    out_df.to_csv(args.out, index=False)
    print(f"[OK] Wrote {args.out} with {len(out_df)} matches.")
    if len(out_df) > 0:
        print(out_df.head(min(5, len(out_df))).to_string(index=False))


if __name__ == "__main__":
    main()
