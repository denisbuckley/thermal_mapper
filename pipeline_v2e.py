
#!/usr/bin/env python3
"""
pipeline_v2e.py

Runs the cluster matching step and writes matched_clusters.csv enriched with key
parameters from both circle and altitude clusters.

Inputs (defaults assume you ran the upstream steps in the same folder):
  - circle_clusters_enriched.csv   (from circle_clusters_v1s.py)
  - overlay_altitude_clusters.csv  (from overlay_altitude_clusters.py)

Output:
  - matched_clusters.csv

Enrichment fields added per match:
  circle_av_climb_ms, circle_alt_gained_m, circle_lat, circle_lon,
  alt_av_climb_ms,    alt_alt_gained_m,    alt_lat,   alt_lon

Usage examples:
  python pipeline_v2e.py
  python pipeline_v2e.py --circles circle_clusters_enriched.csv \
      --alt overlay_altitude_clusters.csv --out matched_clusters.csv \
      --dist-threshold-m 500 --min-time-overlap-s 10
"""

import argparse
import math
import sys
from typing import Tuple, Optional

import pandas as pd
import numpy as np


# --------------------------- Utilities -------------------------------------

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points in meters."""
    R = 6371000.0  # mean Earth radius [m]
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _coerce_time_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to find start and end time columns in df. Accepts a few common variants.
    Returns (start_col, end_col) or (None, None) if not found.
    """
    candidates = [
        ("start_time", "end_time"),
        ("start_ts", "end_ts"),
        ("t_start", "t_end"),
        ("start", "end"),
        ("start_s", "end_s"),
        ("start_sec", "end_sec"),
    ]

    cols = set(df.columns.str.lower())
    lower_map = {c.lower(): c for c in df.columns}

    for s, e in candidates:
        if s in cols and e in cols:
            return lower_map[s], lower_map[e]

    # If only one set not found, try ISO datetime parse fallback for any 'start*'/'end*' pair
    start_guess = None
    end_guess = None
    for c in df.columns:
        lc = c.lower()
        if start_guess is None and lc.startswith("start"):
            start_guess = c
        if end_guess is None and lc.startswith("end"):
            end_guess = c
    if start_guess and end_guess:
        return start_guess, end_guess

    return None, None


def _coerce_lat_lon(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (lat_col, lon_col) if present, else (None, None).
    """
    lat_candidates = ["lat", "latitude", "lat_deg"]
    lon_candidates = ["lon", "lng", "longitude", "lon_deg"]

    lower_cols = {c.lower(): c for c in df.columns}

    lat_col = next((lower_cols[x] for x in (c.lower() for c in lat_candidates) if x in lower_cols), None)
    lon_col = next((lower_cols[x] for x in (c.lower() for c in lon_candidates) if x in lower_cols), None)
    return lat_col, lon_col


def _to_seconds(series: pd.Series) -> pd.Series:
    """
    Convert a time-like column to seconds since start of day; supports numeric seconds,
    pandas datetime, or strings parseable by pandas.to_datetime.
    """
    if np.issubdtype(series.dtype, np.number):
        return series.astype(float)
    # Try datetime parse
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if parsed.notna().any():
        # seconds since epoch; relative overlap is OK as we only need diffs
        return parsed.view("int64") / 1e9
    # Last resort: coerce to float
    return pd.to_numeric(series, errors="coerce")


def _interval_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Return overlap length in seconds between two [start, end] intervals (if negative -> 0)."""
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    return max(0.0, end - start)


# --------------------------- Matching --------------------------------------

def match_clusters(circles_df: pd.DataFrame,
                   alt_df: pd.DataFrame,
                   dist_threshold_m: float = 500.0,
                   min_time_overlap_s: float = 10.0) -> pd.DataFrame:
    """
    Match circle clusters to altitude clusters by proximity and temporal overlap.
    Produces an enriched DataFrame with both cluster IDs and selected parameters.
    """

    # Identify time columns
    c_start, c_end = _coerce_time_columns(circles_df)
    a_start, a_end = _coerce_time_columns(alt_df)

    if not c_start or not c_end:
        print("[WARN] Could not find start/end time columns in circle clusters; time overlap checks disabled.", file=sys.stderr)
    if not a_start or not a_end:
        print("[WARN] Could not find start/end time columns in altitude clusters; time overlap checks disabled.", file=sys.stderr)

    # Identify lat/lon columns
    c_lat_col, c_lon_col = _coerce_lat_lon(circles_df)
    a_lat_col, a_lon_col = _coerce_lat_lon(alt_df)

    if not c_lat_col or not c_lon_col:
        raise ValueError("Circle clusters are missing lat/lon columns (expect columns like 'lat' and 'lon').")
    if not a_lat_col or not a_lon_col:
        raise ValueError("Altitude clusters are missing lat/lon columns (expect columns like 'lat' and 'lon').")

    # Prepare time arrays (seconds)
    c_tstart = _to_seconds(circles_df[c_start]) if (c_start and c_end) else None
    c_tend   = _to_seconds(circles_df[c_end])   if (c_start and c_end) else None
    a_tstart = _to_seconds(alt_df[a_start])     if (a_start and a_end) else None
    a_tend   = _to_seconds(alt_df[a_end])       if (a_start and a_end) else None

    # Ensure we have cluster_id columns
    def _id_col(df: pd.DataFrame) -> str:
        for cand in ["cluster_id", "id", "clusterID", "clusterId"]:
            if cand in df.columns:
                return cand
        raise ValueError("No cluster id column found; expected 'cluster_id' (or id/clusterId variants).")

    c_id = _id_col(circles_df)
    a_id = _id_col(alt_df)

    matches = []
    # Index the altitude df for speed (simple loop; consider spatial index later if needed)
    a_records = alt_df[[a_id, a_lat_col, a_lon_col]].copy()
    if a_tstart is not None and a_tend is not None:
        a_records["__tstart"] = a_tstart
        a_records["__tend"]   = a_tend
    else:
        a_records["__tstart"] = np.nan
        a_records["__tend"]   = np.nan

    # Iterate circle clusters and test all altitude clusters (O(N*M); OK for typical sizes)
    for _, crow in circles_df.iterrows():
        clat = float(crow[c_lat_col])
        clon = float(crow[c_lon_col])

        if c_tstart is not None and c_tend is not None:
            ct0 = float(c_tstart.loc[_])
            ct1 = float(c_tend.loc[_])
        else:
            ct0 = np.nan
            ct1 = np.nan

        for j, arow in a_records.iterrows():
            d = _haversine_m(clat, clon, float(arow[a_lat_col]), float(arow[a_lon_col]))
            if d > dist_threshold_m:
                continue

            # Time overlap
            if not (np.isnan(ct0) or np.isnan(ct1) or np.isnan(arow["__tstart"]) or np.isnan(arow["__tend"])):
                ovl = _interval_overlap(ct0, ct1, float(arow["__tstart"]), float(arow["__tend"]))
                if ovl < min_time_overlap_s:
                    continue
                # overlap fraction relative to the shorter interval
                c_dur = max(0.0, ct1 - ct0)
                a_dur = max(0.0, float(arow["__tend"]) - float(arow["__tstart"]))
                denom = max(1e-9, min(c_dur, a_dur))
                frac = ovl / denom
            else:
                # If we can't compute overlap, record NaNs but still accept spatial match
                ovl = np.nan
                frac = np.nan

            # Append enriched record (per user spec)
            matches.append({
                "circle_cluster_id": crow[c_id],
                "alt_cluster_id": arow[a_id],
                "dist_m": d,
                "time_overlap_s": ovl,
                "overlap_frac": frac,
                # Circle cluster params
                "circle_av_climb_ms": crow.get("av_climb_ms", np.nan),
                "circle_alt_gained_m": crow.get("alt_gained_m", np.nan),
                "circle_lat": clat,
                "circle_lon": clon,
                # Alt cluster params
                "alt_av_climb_ms": alt_df.loc[j].get("av_climb_ms", np.nan),
                "alt_alt_gained_m": alt_df.loc[j].get("alt_gained_m", np.nan),
                "alt_lat": float(arow[a_lat_col]),
                "alt_lon": float(arow[a_lon_col]),
            })

    return pd.DataFrame(matches, columns=[
        "circle_cluster_id", "alt_cluster_id", "dist_m",
        "time_overlap_s", "overlap_frac",
        "circle_av_climb_ms", "circle_alt_gained_m", "circle_lat", "circle_lon",
        "alt_av_climb_ms", "alt_alt_gained_m", "alt_lat", "alt_lon",
    ])


# --------------------------- CLI -------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Match circle vs altitude clusters with enrichment.")
    ap.add_argument("--circles", default="circle_clusters_enriched.csv",
                    help="Path to circle clusters CSV (default: circle_clusters_enriched.csv)")
    ap.add_argument("--alt", default="overlay_altitude_clusters.csv",
                    help="Path to altitude clusters CSV (default: overlay_altitude_clusters.csv)")
    ap.add_argument("--out", default="matched_clusters.csv",
                    help="Output path for matched CSV (default: matched_clusters.csv)")
    ap.add_argument("--dist-threshold-m", type=float, default=500.0,
                    help="Max distance (m) for a spatial match (default: 500)")
    ap.add_argument("--min-time-overlap-s", type=float, default=10.0,
                    help="Minimum time overlap (s) required (default: 10). If times are missing, spatial-only match is allowed.")
    args = ap.parse_args()

    # Load inputs
    circles_df = pd.read_csv(args.circles)
    alt_df = pd.read_csv(args.alt)

    # Match
    out_df = match_clusters(circles_df, alt_df,
                            dist_threshold_m=args.dist_threshold_m,
                            min_time_overlap_s=args.min_time_overlap_s)

    # Write
    out_df.to_csv(args.out, index=False)
    print(f"[OK] Wrote {args.out} with {len(out_df)} matches.")
    if len(out_df) > 0:
        # quick preview
        print(out_df.head(min(5, len(out_df))).to_string(index=False))


if __name__ == "__main__":
    main()
