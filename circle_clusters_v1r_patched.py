#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
circle_clusters_v1r.py (schema aligned)
- Circle-based clusters with enriched schema aligned to altitude clusters + matcher v1d
- CSV fields:
  cluster_id, n_segments, n_turns_sum, duration_min,
  alt_gained_m, av_climb_ms, lat, lon, t_start, t_end
"""

import os, sys, argparse, math, logging
import numpy as np
import pandas as pd

PROJECT_ROOT = "/Users/denisbuckley/PycharmProjects/chatgpt_igc"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
DEBUG_DIR  = os.path.join(PROJECT_ROOT, "debugs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

LOG_PATH = os.path.join(DEBUG_DIR, "circle_clusters_debug.log")
logging.basicConfig(filename=LOG_PATH, level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_CSV = "/Users/denisbuckley/PycharmProjects/chatgpt_igc/outputs/circle_clusters_enriched.csv"

DEFAULT_IGC = "2020-11-08 Lumpy Paterson 108645.igc"

def parse_igc(path: str) -> pd.DataFrame:
    times, lats, lons, alts = [], [], [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line[0] != "B" or len(line) < 35:
                continue
            hh, mm, ss = int(line[1:3]), int(line[3:5]), int(line[5:7])
            lat_dd, lat_mm, lat_mmm, lat_hem = int(line[7:9]), int(line[9:11]), int(line[11:14]), line[14]
            lon_ddd, lon_mm, lon_mmm, lon_hem = int(line[15:18]), int(line[18:20]), int(line[20:23]), line[23]
            lat = lat_dd + (lat_mm + lat_mmm/1000.0)/60.0
            if lat_hem == "S": lat = -lat
            lon = lon_ddd + (lon_mm + lon_mmm/1000.0)/60.0
            if lon_hem == "W": lon = -lon
            try:
                alt = float(line[25:30])
            except:
                alt = np.nan
            t = hh*3600 + mm*60 + ss
            times.append(t); lats.append(lat); lons.append(lon); alts.append(alt)
    return pd.DataFrame({"time_s": times, "lat": lats, "lon": lons, "alt": alts}).dropna()

# Placeholder for circle detection; assumes you have working logic in your current version
def detect_circles(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(columns=["i_start","i_end","t_start","t_end","dur_s","n_turns","lat","lon"])

def cluster_segments(seg_df: pd.DataFrame, df_fix: pd.DataFrame) -> pd.DataFrame:
    if seg_df.empty:
        return pd.DataFrame(columns=[
            "cluster_id","n_segments","n_turns_sum","duration_min",
            "alt_gained_m","av_climb_ms","lat","lon","t_start","t_end"
        ])

    clusters = []
    for cid, row in seg_df.iterrows():
        dur_s = row["dur_s"]
        alt_start = df_fix.loc[int(row["i_start"]),"alt"]
        alt_end   = df_fix.loc[int(row["i_end"]),"alt"]
        alt_gain = alt_end - alt_start
        av_climb = alt_gain/dur_s if dur_s > 0 else np.nan
        clusters.append({
            "cluster_id": cid+1,
            "n_segments": 1,
            "n_turns_sum": row.get("n_turns", np.nan),
            "duration_min": dur_s/60.0,
            "alt_gained_m": alt_gain,
            "av_climb_ms": av_climb,
            "lat": row["lat"],
            "lon": row["lon"],
            "t_start": row["t_start"],
            "t_end": row["t_end"],
        })
    return pd.DataFrame(clusters)

def main():
    ap = argparse.ArgumentParser(description="Circle-based cluster detection with aligned enriched schema")
    ap.add_argument("igc", nargs="?", help="Path to IGC file")
    ap.add_argument("--clusters-csv", default=os.path.join(OUTPUT_DIR, "outputs/circle_clusters_enriched.csv"))
    args = ap.parse_args()

    igc_path = args.igc or input(f"Enter path to IGC file [default: {DEFAULT_IGC}]: ").strip() or DEFAULT_IGC
    df = parse_igc(igc_path)

    seg_df = detect_circles(df)
    clusters = cluster_segments(seg_df, df)

    clusters.to_csv(args.clusters_csv, index=False)
    print(f"Clusters saved: {args.clusters_csv}")
    if not clusters.empty:
        print(clusters)

if __name__ == "__main__":
    main()
