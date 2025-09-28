#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import pandas as pd

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    from math import radians, sin, cos, asin, sqrt
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    return 2 * R * asin(sqrt(a))

def time_overlap(a_start, a_end, b_start, b_end):
    latest_start = max(a_start, b_start)
    earliest_end = min(a_end, b_end)
    return max(0.0, earliest_end - latest_start)

def match_clusters(circle_df, alt_df, eps_m=2000.0, min_ovl_frac=0.20, max_time_gap=900.0):
    matches = []
    cand_count = 0
    for _, crow in circle_df.iterrows():
        for _, arow in alt_df.iterrows():
            d = haversine_m(crow["lat"], crow["lon"], arow["lat"], arow["lon"])
            if d > eps_m:
                continue
            dt_gap = abs(crow["t_start"] - arow["t_start"])
            if dt_gap > max_time_gap:
                continue
            ovl = time_overlap(crow["t_start"], crow["t_end"], arow["t_start"], arow["t_end"])
            dur_short = min(crow["t_end"] - crow["t_start"], arow["t_end"] - arow["t_start"])
            frac = (ovl / dur_short) if dur_short > 0 else 0.0
            cand_count += 1
            if frac >= min_ovl_frac:
                matches.append({
                    "circle_cluster_id": crow["cluster_id"],
                    "alt_cluster_id": arow["cluster_id"],
                    "dist_m": d,
                    "time_overlap_s": ovl,
                    "overlap_frac": frac
                })
    return pd.DataFrame(matches), cand_count

def main():
    import argparse, json
    from pathlib import Path
    import pandas as pd

    ap = argparse.ArgumentParser()
    # Positional args optional → convenient manual runs; pipeline passes explicit names
    ap.add_argument("circles", nargs="?", help="Path to circle_clusters_enriched.csv")
    ap.add_argument("alts", nargs="?", help="Path to altitude_clusters.csv")
    ap.add_argument("--out", help="Output CSV (default: matched_clusters.csv)")
    args = ap.parse_args()

    # Resolve paths (fallback to filenames in current dir)
    circles_path = Path(args.circles) if args.circles else Path("circle_clusters_enriched.csv")
    alts_path    = Path(args.alts)    if args.alts    else Path("altitude_clusters.csv")
    out_path     = Path(args.out)     if args.out     else Path("matched_clusters.csv")

    missing = [p for p in (circles_path, alts_path) if not p.exists()]
    if missing:
        print("[ERROR] Missing required inputs:")
        for p in missing:
            print("  -", p)
        return 1

    # Load inputs
    circles = pd.read_csv(circles_path)
    alts    = pd.read_csv(alts_path)

    # Core matching (assumes match_clusters(circles, alts) exists in this module)
    matches, stats = match_clusters(circles, alts)

    # Minimal columns for coord+rate enrichment
    circle_coords = circles[[c for c in ("cluster_id","lat","lon") if c in circles.columns]].copy()
    alt_cols = [c for c in ("cluster_id","lat","lon","alt_gain_m","duration_s","climb_rate_ms") if c in alts.columns]
    alt_coords = alts[alt_cols].copy()

    # Merge circle coords into canonical lat/lon
    enriched = matches.merge(
        circle_coords.rename(columns={"lat":"lat","lon":"lon"}),
        left_on="circle_cluster_id", right_on="cluster_id", how="left", suffixes=("", "")
    ).drop(columns=["cluster_id"], errors="ignore")

    # Merge altitude coords; keep *_alt temp columns
    enriched = enriched.merge(
        alt_coords.rename(columns={"lat":"lat","lon":"lon"}),
        left_on="alt_cluster_id", right_on="cluster_id", how="left", suffixes=("", "_alt")
    ).drop(columns=["cluster_id"], errors="ignore")

    # If both circle+alt coords present, average; else keep whichever exists
    if "lat_alt" in enriched.columns and "lon_alt" in enriched.columns:
        enriched["lat"] = enriched[["lat","lat_alt"]].mean(axis=1, skipna=True)
        enriched["lon"] = enriched[["lon","lon_alt"]].mean(axis=1, skipna=True)
        enriched = enriched.drop(columns=["lat_alt","lon_alt"])

    if "lat" not in enriched.columns or "lon" not in enriched.columns:
        raise ValueError("No usable lat/lon columns to create unified coords")

    # Ensure climb_rate_ms present (compute if needed)
    if "climb_rate_ms" not in enriched.columns:
        if {"alt_gain_m","duration_s"} <= set(enriched.columns):
            enriched["climb_rate_ms"] = enriched["alt_gain_m"] / enriched["duration_s"].replace(0, pd.NA)
        else:
            enriched["climb_rate_ms"] = pd.NA

    # Write outputs
    enriched.to_csv(out_path, index=False)
    print(f"[OK] wrote {len(enriched)} matches → {out_path}")

    # JSON stats alongside CSV
    json_path = out_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[OK] wrote stats → {json_path}")

    return 0
if __name__ == "__main__":
    main()

