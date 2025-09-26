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
    circ_path = "outputs/circle_clusters_enriched.csv"
    alt_path = "outputs/overlay_altitude_clusters.csv"
    if not os.path.exists(circ_path) or not os.path.exists(alt_path):
        print("Missing required inputs. Ensure both circle and altitude cluster CSVs exist in outputs/.")
        return

    circle_df = pd.read_csv(circ_path)
    alt_df = pd.read_csv(alt_path)

    print("[match v1] Params: EPS_M=2000 m | MIN_OVL_FRAC=0.20 | MAX_TIME_GAP_S=900 s")

    matches_df, cand_count = match_clusters(circle_df, alt_df)
    strict_count = len(matches_df)

    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/matched_clusters.csv"
    matches_df.to_csv(out_path, index=False)

    print(f"[match v1] Candidate pairs: {cand_count}")
    print(f"[match v1] Strict matches: {strict_count}")
    print(f"Wrote {strict_count} matches → {out_path}")

if __name__ == "__main__":
    main()
