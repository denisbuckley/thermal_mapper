#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import pandas as pd

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def cluster_circles(df, dist_thresh=200.0, time_gap=300.0):
    """Group circles into clusters based on proximity in space and time."""
    df = df.sort_values("t_start").reset_index(drop=True)
    clusters = []
    cluster_id = 1
    current = {
        "cluster_id": cluster_id,
        "circles": [],
        "t_start": None,
        "t_end": None
    }

    for _, row in df.iterrows():
        if not current["circles"]:
            current["circles"].append(row)
            current["t_start"] = row["t_start"]
            current["t_end"] = row["t_end"]
            continue

        last = current["circles"][-1]
        d = haversine_m(last["lat"], last["lon"], row["lat"], row["lon"])
        dt = row["t_start"] - last["t_end"]
        if d <= dist_thresh and dt <= time_gap:
            current["circles"].append(row)
            current["t_end"] = row["t_end"]
        else:
            clusters.append(current)
            cluster_id += 1
            current = {
                "cluster_id": cluster_id,
                "circles": [row],
                "t_start": row["t_start"],
                "t_end": row["t_end"]
            }

    if current["circles"]:
        clusters.append(current)

    enriched = []
    for c in clusters:
        circles = pd.DataFrame(c["circles"])
        dur_min = circles["duration_s"].sum() / 60.0
        turns_sum = circles["n_turns"].sum() if "n_turns" in circles else circles["duration_s"].sum()/30.0
        alt_gain = circles["alt_gained_m"].sum()
        climb = (alt_gain / (circles["duration_s"].sum())) if circles["duration_s"].sum() > 0 else np.nan
        enriched.append({
            "cluster_id": c["cluster_id"],
            "n_circles": len(circles),
            "n_turns_sum": turns_sum,
            "duration_min": dur_min,
            "alt_gained_m": alt_gain,
            "av_climb_ms": climb,
            "lat": circles["lat"].mean(),
            "lon": circles["lon"].mean(),
            "t_start": c["t_start"],
            "t_end": c["t_end"]
        })
    return pd.DataFrame(enriched)

def main():
    import argparse
    from pathlib import Path
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("circles", nargs="?", help="Path to circles.csv")
    ap.add_argument("--out", help="Path to output enriched clusters CSV")
    args = ap.parse_args()

    # Default file if nothing passed
    default_circles = Path("outputs/circles.csv")
    default_out = Path("outputs/circle_clusters_enriched.csv")

    if args.circles:
        circles_path = Path(args.circles)
    else:
        user_in = input(f"Enter path to circles.csv [default: {default_circles}]: ").strip()
        circles_path = Path(user_in) if user_in else default_circles

    if not circles_path.exists():
        print(f"[ERROR] Missing input: {circles_path}")
        return

    out_path = Path(args.out) if args.out else default_out

    # === existing cluster logic ===
    df = pd.read_csv(circles_path)
    clusters = cluster_circles(df)   # your existing function
    clusters.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path}")
if __name__ == "__main__":
    main()
