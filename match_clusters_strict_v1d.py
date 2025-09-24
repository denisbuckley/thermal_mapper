#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import pandas as pd
import numpy as np

# ---- Debug helpers ----
def _debug_df(name, path, df):
    try:
        print(f"DEBUG {name} file: {path}")
        print(f"DEBUG {name} columns: {list(df.columns)} rows: {len(df)}")
        print(df.head())
    except Exception as e:
        print(f"DEBUG {name} print error: {e}")

def _normalize_for_match(df, name):
    """Coerce inputs to required matcher schema & order."""
    # Rename common alternates
    rename_map = {
        'dur_s_sum': 'duration_min',
    }
    df = df.rename(columns=rename_map)

    required = ['cluster_id','n_segments','n_turns_sum','duration_min',
                'alt_gained_m','av_climb_ms','lat','lon','t_start','t_end']

    # Add any missing columns as NaN so matcher can proceed
    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    # Heuristic: if duration_min looks like seconds, convert to minutes
    try:
        mx = pd.to_numeric(df['duration_min'], errors='coerce').max()
        if pd.notna(mx) and mx > 180:  # >3 minutes -> likely seconds
            df['duration_min'] = pd.to_numeric(df['duration_min'], errors='coerce') / 60.0
    except Exception:
        pass

    # Enforce order & types
    df = df[required]
    for col in ['cluster_id','n_segments']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ['n_turns_sum','duration_min','alt_gained_m','av_climb_ms','lat','lon','t_start','t_end']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"NORMALIZED {name} columns: {list(df.columns)} rows: {len(df)}")
    return df

# ---- Matcher params (same as v1c) ----
EPS_M = 2000.0          # max spatial distance for cluster centres
MIN_OVL_FRAC = 0.20     # min overlap fraction in time
MAX_TIME_GAP_S = 900.0  # max allowed time gap

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlmb/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))

def overlap_frac(a_start, a_end, b_start, b_end):
    latest_start = max(a_start, b_start)
    earliest_end = min(a_end, b_end)
    overlap = max(0.0, earliest_end - latest_start)
    durA = max(1.0, a_end - a_start)
    durB = max(1.0, b_end - b_start)
    min_dur = min(durA, durB)
    return overlap / min_dur

def match_clusters(dfA: pd.DataFrame, dfB: pd.DataFrame) -> pd.DataFrame:
    matches = []
    for _, a in dfA.iterrows():
        for _, b in dfB.iterrows():
            # Both need lat/lon and times to match
            if pd.isna(a['lat']) or pd.isna(a['lon']) or pd.isna(b['lat']) or pd.isna(b['lon']):
                continue
            if pd.isna(a['t_start']) or pd.isna(a['t_end']) or pd.isna(b['t_start']) or pd.isna(b['t_end']):
                continue

            d = haversine_m(a['lat'], a['lon'], b['lat'], b['lon'])
            if d > EPS_M:
                continue
            ovl = overlap_frac(a['t_start'], a['t_end'], b['t_start'], b['t_end'])
            if ovl < MIN_OVL_FRAC:
                continue
            t_gap = abs(a['t_start'] - b['t_start'])
            if t_gap > MAX_TIME_GAP_S:
                continue

            matches.append({
                "A_id": a['cluster_id'],
                "B_id": b['cluster_id'],
                "dist_m": d,
                "ovl_frac": ovl,
                "t_gap_s": t_gap,
                "A_turns": a.get("n_turns_sum", np.nan),
                "B_turns": b.get("n_turns_sum", np.nan),
                "A_duration_min": a.get("duration_min", np.nan),
                "B_duration_min": b.get("duration_min", np.nan),
                "A_alt_gained_m": a.get("alt_gained_m", np.nan),
                "B_alt_gained_m": b.get("alt_gained_m", np.nan),
                "A_av_climb_ms": a.get("av_climb_ms", np.nan),
                "B_av_climb_ms": b.get("av_climb_ms", np.nan),
            })
    return pd.DataFrame(matches)

def main():
    ap = argparse.ArgumentParser(description="Strict cluster matcher (v1d, with normalization/debug)")
    ap.add_argument("--circle", default="outputs/circle_clusters_enriched.csv", help="Path to circle clusters enriched CSV")
    ap.add_argument("--alt", default="outputs/overlay_altitude_clusters.csv", help="Path to altitude clusters enriched CSV")
    ap.add_argument("--out", default="outputs/matched_clusters_v1d.csv", help="Path to save matches CSV")
    args = ap.parse_args()

    # Load
    circle_df = pd.read_csv(args.circle)
    _debug_df('circle', args.circle, circle_df)
    alt_df = pd.read_csv(args.alt)
    _debug_df('altitude', args.alt, alt_df)

    # Normalize to required schema
    circle_df = _normalize_for_match(circle_df, "circle")
    alt_df = _normalize_for_match(alt_df, "altitude")

    if circle_df.empty or alt_df.empty:
        print("Missing enriched inputs after normalization (empty data). Ensure detectors produced non-empty CSVs.")
        sys.exit(1)

    print(f"[match_strict v1d] Params: EPS_M={EPS_M} m | MIN_OVL_FRAC={MIN_OVL_FRAC} | MAX_TIME_GAP_S={MAX_TIME_GAP_S}s")

    # Run matching
    matches = match_clusters(circle_df, alt_df)

    # Save
    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    matches.to_csv(out_path, index=False)

    print(f"[match_strict v1d] Candidate pairs: {len(circle_df)*len(alt_df)}")
    print(f"[match_strict v1d] Strict 1:1 matches: {len(matches)}")
    print(f"Output CSV: {out_path}")
    if not matches.empty:
        print(matches.head())

if __name__ == "__main__":
    main()
