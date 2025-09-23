#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
overlay_altitude_clusters_v1c.py
Detects thermals from sustained climbs in the altitude trace (independent of circle detection).
- Reads IGC
- Finds climb segments (alt gained > threshold, duration > threshold)
- Outputs CSV (/outputs)
- Logs debug info (/debugs)
- Plots bird’s-eye track with black X at cluster centres
"""

import os, sys, argparse, logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Project paths
PROJECT_ROOT = "/Users/denisbuckley/PycharmProjects/chatgpt_igc"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
DEBUG_DIR = os.path.join(PROJECT_ROOT, "debugs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

LOG_PATH = os.path.join(DEBUG_DIR, "overlay_altitude_clusters.log")
logging.basicConfig(filename=LOG_PATH, level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

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
    df = pd.DataFrame({"time_s": times, "lat": lats, "lon": lons, "alt": alts}).dropna()
    return df

# Simple sustained climb detector
MIN_CLIMB_M = 50.0   # min altitude gain
MIN_DUR_S   = 30.0   # min duration in seconds

def detect_altitude_clusters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame(columns=["cluster_id","t_start","t_end","dur_s","lat","lon","alt_start","alt_end","alt_gained_m","av_climb_ms"])
    clusters = []
    climbing = False
    start_i = None
    for i in range(1, len(df)):
        if df["alt"].iloc[i] > df["alt"].iloc[i-1]:
            if not climbing:
                climbing = True
                start_i = i-1
        else:
            if climbing:
                end_i = i-1
                alt_gain = df["alt"].iloc[end_i] - df["alt"].iloc[start_i]
                dur = df["time_s"].iloc[end_i] - df["time_s"].iloc[start_i]
                if alt_gain >= MIN_CLIMB_M and dur >= MIN_DUR_S:
                    latc = df["lat"].iloc[start_i:end_i+1].mean()
                    lonc = df["lon"].iloc[start_i:end_i+1].mean()
                    clusters.append({
                        "t_start": df["time_s"].iloc[start_i],
                        "t_end": df["time_s"].iloc[end_i],
                        "dur_s": dur,
                        "lat": latc,
                        "lon": lonc,
                        "alt_start": df["alt"].iloc[start_i],
                        "alt_end": df["alt"].iloc[end_i],
                        "alt_gained_m": alt_gain,
                        "av_climb_ms": alt_gain/dur if dur>0 else np.nan,
                    })
                climbing = False
    out = pd.DataFrame(clusters)
    out.insert(0,"cluster_id",range(1,len(out)+1))
    return out

def plot_overlay(df: pd.DataFrame, clusters: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(df["lon"], df["lat"], color="lightgray", lw=1.0)
    if not clusters.empty:
        ax.scatter(clusters["lon"], clusters["lat"], c="black", marker="x")
        for _,r in clusters.iterrows():
            ax.text(r["lon"], r["lat"], f"{int(r['cluster_id'])}", fontsize=8, ha="left", va="bottom", color="black")
    ax.set_title("Altitude-based clusters")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.axis("equal")
    plt.show()
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("igc", nargs="?", help="Path to IGC file")
    args = ap.parse_args()
    igc_path = args.igc or input(f"Enter path to IGC file [default: {DEFAULT_IGC}]: ").strip() or DEFAULT_IGC
    df = parse_igc(igc_path)
    clusters = detect_altitude_clusters(df)
    out_csv = os.path.join(OUTPUT_DIR, "overlay_altitude_clusters.csv")
    clusters.to_csv(out_csv, index=False)
    print(f"Fixes: {len(df)} | Altitude clusters: {len(clusters)}")
    print(f"Clusters CSV: {out_csv}")
    if not clusters.empty:
        print(clusters[["cluster_id","dur_s","alt_gained_m","av_climb_ms"]])
    plot_overlay(df, clusters)

if __name__ == "__main__":
    main()
