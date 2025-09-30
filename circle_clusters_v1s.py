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
        alt_gain = circles["alt_gain_m"].sum()
        climb = (alt_gain / (circles["duration_s"].sum())) if circles["duration_s"].sum() > 0 else np.nan
        enriched.append({
            "cluster_id": c["cluster_id"],
            "n_circles": len(circles),
            "n_turns_sum": turns_sum,
            "duration_min": dur_min,
            "alt_gain_m": alt_gain,
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

    ap = argparse.ArgumentParser()
    ap.add_argument("circles", nargs="?", help="Path to circles.csv")
    ap.add_argument("--out", help="Output CSV (default: circle_clusters_enriched.csv in same folder)")
    args = ap.parse_args()

    # Prompt if no input path provided
    if args.circles:
        circles_path = Path(args.circles)
    else:
        inp = input("Enter path to circles.csv (required): ").strip()
        if not inp:
            raise SystemExit("[ERROR] circles.csv path is required")
        circles_path = Path(inp)

    if not circles_path.exists():
        raise FileNotFoundError(circles_path)

    # Default out: same folder as input
    out_path = Path(args.out) if args.out else circles_path.parent / "circle_clusters_enriched.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load, process, save
    circles = pd.read_csv(circles_path)

    # --- normalize altitude-gain to canonical: alt_gain_m ---
    gain_variants = ["alt_gain_m", "alt_gain_m", "alt_gain", "altitude_gain_m"]
    for g in gain_variants:
        if g in circles.columns:
            if g != "alt_gain_m":
                circles = circles.rename(columns={g: "alt_gain_m"})
            break
    else:
        cols = set(circles.columns)
        if {"alt_start_m", "alt_end_m"} <= cols:
            circles["alt_gain_m"] = circles["alt_end_m"] - circles["alt_start_m"]
        elif {"climb_rate_ms", "duration_s"} <= cols:
            circles["alt_gain_m"] = circles["climb_rate_ms"] * circles["duration_s"]
        else:
            raise KeyError(
                "No altitude-gain column and cannot derive one. "
                "Expected alt_gain_m (or alt_gain_m/alt_gain/altitude_gain_m), "
                "or derive from alt_start_m+alt_end_m or climb_rate_ms*duration_s."
            )
    clusters = cluster_circles(circles)

    # === PATCH: canonical order for circle_clusters_enriched.csv ===
    need = [
        "cluster_id", "lat", "lon", "t_start", "t_end",
        "climb_rate_ms", "climb_rate_ms_median", "alt_gain_m_mean", "duration_s_mean",
        "n_circles",
    ]
    for c in need:
        if c not in clusters.columns:
            clusters[c] = pd.NA
    clusters = clusters[need]
    # === END PATCH ===

    clusters.to_csv(out_path, index=False)
    print(f"[OK] wrote {len(clusters)} clusters → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

if __name__ == "__main__":
    main()
