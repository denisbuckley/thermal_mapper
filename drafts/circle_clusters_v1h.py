#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
circle_clusters_v1h.py — same as v1g but with anti-clutter label stacking.
- Nearby clusters (within LABEL_GROUP_EPS_M) have their labels drawn in a vertical stack.
- Label text stays minimal: "id, n_segments".
- Markers: black X's; track: light blue.

Outputs:
  - circle_clusters_enriched.csv
  - circle_segments.csv
Log:
  - /Users/denisbuckley/PycharmProjects/chatgpt_igc/debugs/circle_clusters_debug.log
"""
import os, sys, argparse, math, logging
import numpy as np
import pandas as pd

# --- logging ---
LOG_DIR = "/debugs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "circle_clusters_debug.log")
logging.basicConfig(filename=LOG_PATH, level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
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
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1)*math.sin(p2) - math.sin(p1)*math.cos(p2)*math.cos(dl)
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
    # time seconds, handle midnight wrap
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
        start = i; cum_arc = 0.0; last_t = t[i]; valid = 0
        while i < n:
            if not np.isfinite(rate[i]) or (t[i]-last_t) > MAX_GAP_S: break
            if abs(rate[i]) < C_MIN_RATE_DPS or abs(rate[i]) > C_MAX_RATE_DPS: break
            if sgn != 0 and np.sign(rate[i]) != sgn: break
            if not (C_MIN_RADIUS_M <= radius[i] <= C_MAX_RADIUS_M or not np.isfinite(radius[i])): break
            if i > start: cum_arc += abs(dpsi[i])
            last_t = t[i]; valid += 1; i += 1
        end = i - 1
        dur = t[end] - t[start] if end > start else 0.0
        if valid >= C_MIN_SAMPLES and cum_arc >= C_MIN_ARC_DEG and dur > 0:
            segments.append({
                "i_start": start, "i_end": end,
                "t_start": t[start], "t_end": t[end], "dur_s": dur,
                "arc_deg": cum_arc, "n_turns": cum_arc/360.0,
                "lat": float(np.nanmean(lat[start:end+1])),
                "lon": float(np.nanmean(lon[start:end+1])),
            })
        i += 1
    return pd.DataFrame(segments)

# --- clustering ---
CL_EPS_M     = 400.0
CL_GAP_S     = 240.0
CL_MIN_COUNT = 1

def cluster_segments(seg_df):
    if seg_df.empty:
        return pd.DataFrame(columns=["cluster_id","n_segments","n_turns_sum","dur_s_sum","lat","lon"])
    seg_df = seg_df.sort_values("t_start").reset_index(drop=True)
    clusters = []
    current = {"ids":[0], "lats":[seg_df.loc[0,"lat"]], "lons":[seg_df.loc[0,"lon"]],
               "n_turns":seg_df.loc[0,"n_turns"], "dur":seg_df.loc[0,"dur_s"],
               "t_start":seg_df.loc[0,"t_start"], "t_end":seg_df.loc[0,"t_end"]}
    def within(a_lat,a_lon,b_lat,b_lon): return haversine_m(a_lat,a_lon,b_lat,b_lon) <= CL_EPS_M
    for i in range(1,len(seg_df)):
        lat_i, lon_i = seg_df.loc[i,"lat"], seg_df.loc[i,"lon"]
        t_gap = seg_df.loc[i,"t_start"] - current["t_end"]
        cent_lat, cent_lon = float(np.mean(current["lats"])), float(np.mean(current["lons"]))
        if t_gap <= CL_GAP_S and within(lat_i, lon_i, cent_lat, cent_lon):
            current["ids"].append(i); current["lats"].append(lat_i); current["lons"].append(lon_i)
            current["n_turns"] += seg_df.loc[i,"n_turns"]; current["dur"] += seg_df.loc[i,"dur_s"]
            current["t_end"] = max(current["t_end"], seg_df.loc[i,"t_end"])
        else:
            clusters.append(current)
            current = {"ids":[i], "lats":[lat_i], "lons":[lon_i],
                       "n_turns":seg_df.loc[i,"n_turns"], "dur":seg_df.loc[i,"dur_s"],
                       "t_start":seg_df.loc[i,"t_start"], "t_end":seg_df.loc[i,"t_end"]}
    clusters.append(current)
    rows = []
    for cid,c in enumerate(clusters,1):
        rows.append({
            "cluster_id": cid,
            "n_segments": len(c["ids"]),
            "n_turns_sum": float(c["n_turns"]),
            "dur_s_sum": float(c["dur"]),
            "lat": float(np.mean(c["lats"])),
            "lon": float(np.mean(c["lons"])),
        })
    out = pd.DataFrame(rows)
    out = out[out["n_segments"] >= CL_MIN_COUNT].reset_index(drop=True)
    return out

# --- plotting with stacking ---
LABEL_GROUP_EPS_M = 250.0   # clusters this close share a stacked label block
LABEL_DY_M        = 80.0    # vertical spacing between stacked labels in meters
LABEL_DX_M        = 50.0    # horizontal nudge so label clears the X marker

def plot_overlay(df, clusters):
    import matplotlib.pyplot as plt
    lat = df["lat"].to_numpy(); lon = df["lon"].to_numpy()
    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(lon, lat, color="skyblue", lw=1.0, alpha=0.7)

    # Plot markers first
    for _, r in clusters.iterrows():
        ax.plot(r["lon"], r["lat"], marker="x", color="black", markersize=6)

    # Build groups of clusters that are nearby (within LABEL_GROUP_EPS_M)
    coords = list(zip(clusters["lat"].to_numpy(), clusters["lon"].to_numpy()))
    remaining = set(range(len(coords)))
    groups = []
    while remaining:
        i = remaining.pop()
        lat_i, lon_i = coords[i]
        group = [i]
        to_remove = set()
        for j in remaining:
            lat_j, lon_j = coords[j]
            if haversine_m(lat_i, lon_i, lat_j, lon_j) <= LABEL_GROUP_EPS_M:
                group.append(j)
                to_remove.add(j)
        remaining -= to_remove
        groups.append(group)

    # Helper: meters to degrees latitude and longitude at given latitude
    def meters_to_deg(dy_m, dx_m, at_lat_deg):
        dlat = dy_m / 111320.0
        dlon = dx_m / (111320.0 * math.cos(math.radians(at_lat_deg)) if math.cos(math.radians(at_lat_deg)) != 0 else 1e-6)
        return dlat, dlon

    # Draw stacked labels per group
    for group in groups:
        # anchor at the group's centroid for stacking reference
        g_lat = float(np.mean([coords[i][0] for i in group]))
        g_lon = float(np.mean([coords[i][1] for i in group]))
        # sort indices by latitude descending so stacks are tidy
        group_sorted = sorted(group, key=lambda k: coords[k][0], reverse=True)
        for k, idx in enumerate(group_sorted):
            r = clusters.iloc[idx]
            dlat, dlon = meters_to_deg(LABEL_DY_M * k, LABEL_DX_M, r["lat"])
            ax.text(r["lon"] + dlon, r["lat"] + dlat,
                    f"{int(r['cluster_id'])}, {int(r['n_segments'])}",
                    fontsize=8, ha="left", va="bottom", color="black")

    ax.set_title("Circle clusters on track")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.axis("equal")
    plt.show()
    plt.close(fig)

# --- main ---
def main():
    ap = argparse.ArgumentParser(description="Circles → clusters with stacked labels to avoid clutter")
    ap.add_argument("igc", nargs="?", help="Path to IGC file")
    ap.add_argument("--clusters-csv", default="circle_clusters_enriched.csv", help="Output CSV for clusters")
    ap.add_argument("--segments-csv", default="circle_segments.csv", help="Output CSV for circling segments")
    args = ap.parse_args()

    igc_path = args.igc or input(f"Enter path to IGC file [default: {DEFAULT_IGC}]: ").strip() or DEFAULT_IGC
    df = parse_igc_minimal(igc_path)
    seg_df = detect_circles(df)
    clusters = cluster_segments(seg_df)

    seg_df.to_csv(args.segments_csv, index=False)
    clusters.to_csv(args.clusters_csv, index=False)
    logger.info(f"Wrote clusters CSV: {args.clusters_csv} rows={len(clusters)}")

    plot_overlay(df, clusters)

    print(f"Segments: {len(seg_df)} | Clusters: {len(clusters)}")

if __name__ == "__main__":
    main()
