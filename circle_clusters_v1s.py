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
    circles_path = "outputs/circles.csv"
    if not os.path.exists(circles_path):
        print(f"Missing input: {circles_path}. Run circles_from_brecords_v1d.py first.")
        return

    circles = pd.read_csv(circles_path)
    if circles.empty:
        print("No circles found in input.")
        return

    clusters_df = cluster_circles(circles)
    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/circle_clusters_enriched.csv"
    clusters_df.to_csv(out_path, index=False)

    print(f"Wrote {len(clusters_df)} clusters → {out_path}")
    print(clusters_df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
