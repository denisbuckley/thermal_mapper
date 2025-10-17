#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
circle_clusters_v1d.py - self-contained circle detection + clustering with logging and interactive fallback.
"""
import os, sys, argparse, math, logging
import numpy as np
import pandas as pd

# --- logging ---
LOG_DIR = "/debugs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "circle_clusters_debug.log")
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_IGC = "2020-11-08 Lumpy Paterson 108645.igc"

# --- helpers ---
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1 = np.radians(lat1); p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlmb/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))

def bearing_deg(lat1, lon1, lat2, lon2):
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dλ = math.radians(lon2 - lon1)
    y = math.sin(dλ) * math.cos(φ2)
    x = math.cos(φ1)*math.sin(φ2) - math.sin(φ1)*math.cos(φ2)*math.cos(dλ)
    brng = math.degrees(math.atan2(y, x))
    return (brng + 360.0) % 360.0

def unwrap_degrees(deg_series):
    rad = np.deg2rad(deg_series)
    return np.rad2deg(np.unwrap(rad))

# --- parser ---
def parse_igc_minimal(path):
    times, lats, lons, alts = [], [], [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line[0] != "B" or len(line) < 35:
                continue
            hh = int(line[1:3]); mm = int(line[3:5]); ss = int(line[5:7])
            lat_dd = int(line[7:9]); lat_mm = int(line[9:11]); lat_mmm = int(line[11:14]); lat_hem = line[14]
            lon_ddd = int(line[15:18]); lon_mm = int(line[18:20]); lon_mmm = int(line[20:23]); lon_hem = line[23]
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
    df = pd.DataFrame({"t": times, "lat": lats, "lon": lons, "alt": alts}).dropna()
    t = df["t"].to_numpy().astype(float)
    jumps = np.where(np.diff(t) < -43200)[0]
    if len(jumps) > 0:
        t[jumps[0]+1:] += 86400
    df["time_s"] = t
    return df[["time_s","lat","lon","alt"]]

# --- circle detection ---
C_MIN_ARC_DEG  = 30.0
C_MIN_RATE_DPS = 2.0
C_MAX_RATE_DPS = 70.0
C_MIN_RADIUS_M = 15.0
C_MAX_RADIUS_M = 300.0
C_MIN_SAMPLES  = 10
MAX_GAP_S      = 8.0

def detect_circles(df):
    lat = df["lat"].to_numpy(); lon = df["lon"].to_numpy(); t = df["time_s"].to_numpy()
    n = len(df)
    head = np.zeros(n, dtype=float); head[0] = np.nan
    for i in range(1, n):
        head[i] = bearing_deg(lat[i-1], lon[i-1], lat[i], lon[i])
    if n > 1 and np.isnan(head[0]): head[0] = head[1]
    h_unwrap = unwrap_degrees(head)
    dt = np.diff(t, prepend=t[0]); dt[dt <= 0] = np.nan
    dpsi = np.diff(h_unwrap, prepend=h_unwrap[0])
    rate = np.where(np.isfinite(dt), dpsi/np.where(dt==0, np.nan, dt), np.nan)
    dist = np.zeros(n, dtype=float)
    for i in range(1, n):
        dist[i] = haversine_m(lat[i-1], lon[i-1], lat[i], lon[i])
    v = np.where(np.isfinite(dt), dist/np.where(dt==0, np.nan, dt), np.nan)
    omega_rad = np.deg2rad(rate)
    with np.errstate(divide="ignore", invalid="ignore"):
        radius = np.abs(v / np.where(omega_rad==0, np.nan, omega_rad))

    segments = []
    i = 1
    while i < n:
        if not np.isfinite(rate[i]) or dt[i] > MAX_GAP_S:
            i += 1; continue
        sgn = np.sign(rate[i]) if rate[i] != 0 else 0.0
        start = i; cum_arc = 0.0; last_t = t[i]; samples = 1; valid = 0
        while i < n:
            if not np.isfinite(rate[i]) or (t[i]-last_t) > MAX_GAP_S: break
            if abs(rate[i]) < C_MIN_RATE_DPS or abs(rate[i]) > C_MAX_RATE_DPS: break
            if sgn != 0 and np.sign(rate[i]) != sgn: break
            if not (C_MIN_RADIUS_M <= radius[i] <= C_MAX_RADIUS_M or not np.isfinite(radius[i])): break
            if i > start: cum_arc += abs(dpsi[i])
            last_t = t[i]; samples += 1; valid += 1; i += 1
        end = i - 1
        dur = t[end] - t[start] if end > start else 0.0
        if valid >= C_MIN_SAMPLES and cum_arc >= C_MIN_ARC_DEG and dur > 0:
            segments.append({
                "i_start": start, "i_end": end,
                "t_start": t[start], "t_end": t[end], "dur_s": dur,
                "arc_deg": cum_arc, "n_turns": cum_arc/360.0,
                "lat": float(np.nanmean(lat[start:end+1])),
                "lon": float(np.nanmean(lon[start:end+1])),
                "mean_rate_dps": float(np.nanmean(np.abs(rate[start:end+1]))),
                "mean_radius_m": float(np.nanmean(radius[start:end+1])) if np.isfinite(radius[start:end+1]).any() else np.nan
            })
        i += 1
    seg_df = pd.DataFrame(segments)
    logger.info(f"Circling segments: {len(seg_df)}")
    return seg_df

# --- clustering ---
CL_EPS_M     = 400.0
CL_GAP_S     = 240.0
CL_MIN_COUNT = 1

def cluster_segments(seg_df):
    if seg_df.empty:
        return pd.DataFrame(columns=["cluster_id","n_segments","n_turns_sum","dur_s_sum","lat","lon","t_start","t_end"])
    seg_df = seg_df.sort_values("t_start").reset_index(drop=True)
    clusters = []
    current = {"ids": [0], "lats": [seg_df.loc[0,"lat"]], "lons": [seg_df.loc[0,"lon"]],
               "n_turns": seg_df.loc[0,"n_turns"], "dur": seg_df.loc[0,"dur_s"],
               "t_start": seg_df.loc[0,"t_start"], "t_end": seg_df.loc[0,"t_end"]}
    def within(a_lat,a_lon,b_lat,b_lon): return haversine_m(a_lat,a_lon,b_lat,b_lon) <= CL_EPS_M
    for i in range(1, len(seg_df)):
        lat_i, lon_i = seg_df.loc[i,"lat"], seg_df.loc[i,"lon"]
        t_gap = seg_df.loc[i,"t_start"] - current["t_end"]
        cent_lat, cent_lon = float(np.mean(current["lats"])), float(np.mean(current["lons"]))
        if t_gap <= CL_GAP_S and within(lat_i, lon_i, cent_lat, cent_lon):
            current["ids"].append(i); current["lats"].append(lat_i); current["lons"].append(lon_i)
            current["n_turns"] += seg_df.loc[i,"n_turns"]; current["dur"] += seg_df.loc[i,"dur_s"]
            current["t_end"] = max(current["t_end"], seg_df.loc[i,"t_end"])
        else:
            clusters.append(current)
            current = {"ids": [i], "lats": [lat_i], "lons": [lon_i], "n_turns": seg_df.loc[i,"n_turns"],
                       "dur": seg_df.loc[i,"dur_s"], "t_start": seg_df.loc[i,"t_start"], "t_end": seg_df.loc[i,"t_end"]}
    clusters.append(current)
    rows = []
    for cid, c in enumerate(clusters, 1):
        rows.append({
            "cluster_id": cid,
            "n_segments": len(c["ids"]),
            "n_turns_sum": float(c["n_turns"]),
            "dur_s_sum": float(c["dur"]),
            "lat": float(np.mean(c["lats"])),
            "lon": float(np.mean(c["lons"])),
            "t_start": float(c["t_start"]),
            "t_end": float(c["t_end"]),
        })
    out = pd.DataFrame(rows)
    out = out[out["n_segments"] >= CL_MIN_COUNT].reset_index(drop=True)
    logger.info(f"Clusters: {len(out)} kept; Σn_turns={out['n_turns_sum'].sum():.2f}")
    return out

# --- main ---
def main():
    ap = argparse.ArgumentParser(description="Circle cluster detection")
    ap.add_argument("igc", nargs="?", help="Path to IGC file")
    ap.add_argument("--clusters-csv", default="circle_clusters.csv", help="Output CSV for clusters")
    ap.add_argument("--segments-csv", default="circle_segments.csv", help="Output CSV for circling segments")
    args = ap.parse_args()

    igc_path = args.igc
    if not igc_path:
        entered = input(f"Enter path to IGC file [default: {DEFAULT_IGC}]: ").strip()
        igc_path = entered if entered else DEFAULT_IGC

    df = parse_igc_minimal(igc_path)
    seg_df = detect_circles(df)
    clusters = cluster_segments(seg_df)
    seg_df.to_csv(args.segments_csv, index=False)
    clusters.to_csv(args.clusters_csv, index=False)

    print(f"Circling segments: {len(seg_df)}")
    print(f"Clusters: {len(clusters)} (Σn_turns={clusters['n_turns_sum'].sum():.1f})")

if __name__ == "__main__":
    main()
