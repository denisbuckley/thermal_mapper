#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
overlay_altitude_clusters_v1c.py (patched)
- Detects altitude-based climb segments
- Enriches output to match schema of circle_clusters_enriched.csv
- Saves CSV to /outputs/overlay_altitude_clusters.csv
- Logs debug info to /debugs/overlay_altitude_clusters_debug.log
"""

import os, sys, argparse, logging
import numpy as np
import pandas as pd

OUTPUT_DIR = "outputs"
DEBUG_DIR  = "debugs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

LOG_PATH = os.path.join(DEBUG_DIR, "overlay_altitude_clusters_debug.log")
logging.basicConfig(filename=LOG_PATH, level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_IGC = "2020-11-08 Lumpy Paterson 108645.igc"

def parse_igc(path: str) -> pd.DataFrame:
    times, lats, lons, alts = [], [], [], []
    day_offset = 0
    last_t = None

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
            except Exception:
                continue

            t = hh*3600 + mm*60 + ss
            if last_t is not None and t < last_t:
                day_offset += 86400  # crossed midnight
            last_t = t

            times.append(t + day_offset)
            lats.append(lat)
            lons.append(lon)
            alts.append(alt)

    df = pd.DataFrame({"time_s": times, "lat": lats, "lon": lons, "alt": alts}).dropna()
    return df

# altitude-based climb detection
MIN_CLIMB_M = 50.0
MIN_DUR_S   = 60.0

def detect_altitude_clusters(df: pd.DataFrame) -> pd.DataFrame:
    clusters = []
    n = len(df)
    if n < 2:
        return pd.DataFrame(columns=[
            "cluster_id","n_segments","n_turns_sum","duration_min",
            "alt_gained_m","av_climb_ms","lat","lon","t_start","t_end"
        ])
    cid = 0
    i = 0
    while i < n-1:
        j = i+1
        while j < n and df.loc[j,"alt"] >= df.loc[j-1,"alt"]:
            j += 1
        alt_gain = df.loc[j-1,"alt"] - df.loc[i,"alt"]
        dur_s = df.loc[j-1,"time_s"] - df.loc[i,"time_s"]
        if alt_gain >= MIN_CLIMB_M and dur_s >= MIN_DUR_S:
            cid += 1
            duration_min = dur_s / 60.0
            av_climb_ms = alt_gain / dur_s if dur_s > 0 else np.nan
            clusters.append({
                "cluster_id": cid,
                "n_segments": 1,
                "n_turns_sum": np.nan,
                "duration_min": duration_min,
                "alt_gained_m": alt_gain,
                "av_climb_ms": av_climb_ms,
                "lat": float(np.mean(df.loc[i:j,"lat"])),
                "lon": float(np.mean(df.loc[i:j,"lon"])),
                "t_start": float(df.loc[i,"time_s"]),
                "t_end": float(df.loc[j-1,"time_s"]),
            })
        i = j
    return pd.DataFrame(clusters)

def main():
    ap = argparse.ArgumentParser(description="Altitude-based cluster detection with enriched output schema")
    ap.add_argument("igc", nargs="?", help="Path to IGC file")
    ap.add_argument("--clusters-csv", default=os.path.join(OUTPUT_DIR, "overlay_altitude_clusters.csv"))
    args = ap.parse_args()

    igc_path = args.igc or input(f"Enter path to IGC file [default: {DEFAULT_IGC}]: ").strip() or DEFAULT_IGC
    df = parse_igc(igc_path)
    clusters = detect_altitude_clusters(df)

    clusters.to_csv(args.clusters_csv, index=False)
    print(f"Clusters saved: {args.clusters_csv}")
    if not clusters.empty:
        print(clusters)

if __name__ == "__main__":
    main()
