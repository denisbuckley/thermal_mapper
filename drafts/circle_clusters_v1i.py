#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
circle_clusters_v1i.py — circles → clusters with outside-of-polygon stacked labels (no overlap within groups).
- Track is treated as a closed polygon; labels are pushed OUTSIDE the polygon boundary.
- Nearby clusters are merged into groups (connected components) using LABEL_GROUP_EPS_M.
- Each group's labels are drawn in a stack along the outward normal of the nearest polygon edge.
- Markers: black X's; Labels: "id, n_segments".
Outputs:
  - circle_clusters_enriched.csv
  - circle_segments.csv
Debug log:
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

def polygon_orientation_area(lats, lons):
    """Return signed 'area' in degree^2; >0 -> counterclockwise, <0 -> clockwise."""
    # Shoelace on lon/lat
    area = 0.0
    n = len(lats)
    for i in range(n):
        j = (i+1) % n
        area += lons[i]*lats[j] - lons[j]*lats[i]
    return area/2.0

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

def meters_per_deg(lat_deg):
    # approximate scale at latitude
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(lat_deg))
    if m_per_deg_lon < 1e-6:
        m_per_deg_lon = 1e-6
    return m_per_deg_lat, m_per_deg_lon

def to_local_xy(lat, lon, lat0, lon0):
    mlat, mlon = meters_per_deg(lat0)
    x = (lon - lon0) * mlon
    y = (lat - lat0) * mlat
    return x, y

def from_local_xy(x, y, lat0, lon0):
    mlat, mlon = meters_per_deg(lat0)
    lon = lon0 + x/mlon
    lat = lat0 + y/mlat
    return lat, lon

def point_segment_distance_xy(px, py, x1, y1, x2, y2):
    # Returns distance and t in [0,1] for projection onto segment
    vx, vy = x2-x1, y2-y1
    wx, wy = px-x1, py-y1
    seg_len2 = vx*vx + vy*vy
    if seg_len2 == 0:
        return math.hypot(px-x1, py-y1), 0.0
    t = max(0.0, min(1.0, (wx*vx + wy*vy)/seg_len2))
    projx, projy = x1 + t*vx, y1 + t*vy
    return math.hypot(px-projx, py-projy), t

# --- parser (minimal) ---
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

# --- label grouping & outside-of-polygon stacking ---
LABEL_GROUP_EPS_M = 250.0
STACK_STEP_M      = 80.0
BASE_OFFSET_M     = 80.0  # distance from boundary along outward normal

def connected_groups(lat_arr, lon_arr, eps_m):
    n = len(lat_arr)
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for i in range(n):
        for j in range(i+1, n):
            if haversine_m(lat_arr[i], lon_arr[i], lat_arr[j], lon_arr[j]) <= eps_m:
                union(i,j)
    groups = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)
    return list(groups.values())

def nearest_edge_outward_normal(lat_arr, lon_arr, plat, plon, poly_ccw):
    # Convert to local XY around point for distance calc
    lat0, lon0 = plat, plon
    xs, ys = [], []
    for la, lo in zip(lat_arr, lon_arr):
        x, y = to_local_xy(la, lo, lat0, lon0)
        xs.append(x); ys.append(y)
    px, py = 0.0, 0.0  # point is origin in local coords
    best = (1e18, 0, 0.0)  # dist, edge_index, t
    n = len(xs)
    for i in range(n):
        j = (i+1) % n
        d, t = point_segment_distance_xy(px, py, xs[i], ys[i], xs[j], ys[j])
        if d < best[0]:
            best = (d, i, t)
    _, ei, t = best
    i, j = ei, (ei+1) % n
    # edge direction
    ex, ey = xs[j]-xs[i], ys[j]-ys[i]
    elen = math.hypot(ex, ey) or 1.0
    tx, ty = ex/elen, ey/elen  # unit tangent
    # outward normal depends on orientation:
    # For CCW polygon, interior is to the left of edge -> outward is to the right (normal = (+ty, -tx)).
    if poly_ccw:
        nx, ny = +ty, -tx
    else:
        nx, ny = -ty, +tx
    return nx, ny  # local XY

def plot_overlay(df, clusters):
    import matplotlib.pyplot as plt
    lat = df["lat"].to_numpy(); lon = df["lon"].to_numpy()
    # ensure closed polygon
    if lat[0] != lat[-1] or lon[0] != lon[-1]:
        lat = np.concatenate([lat, [lat[0]]])
        lon = np.concatenate([lon, [lon[0]]])
    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(lon, lat, color="skyblue", lw=1.0, alpha=0.7)

    # Markers
    for _, r in clusters.iterrows():
        ax.plot(r["lon"], r["lat"], marker="x", color="black", markersize=6)

    # Orientation (CCW positive area)
    ccw = polygon_orientation_area(lat, lon) > 0

    # Group indices by proximity
    latc = clusters["lat"].to_numpy()
    lonc = clusters["lon"].to_numpy()
    groups = connected_groups(latc, lonc, LABEL_GROUP_EPS_M)

    # For each group place stacked labels along outward normal of nearest edge
    for g in groups:
        # group centroid
        g_lat = float(np.mean(latc[g]))
        g_lon = float(np.mean(lonc[g]))
        nx, ny = nearest_edge_outward_normal(lat, lon, g_lat, g_lon, ccw)  # local xy normal
        # convert normal from local to degrees at group latitude
        mlat, mlon = meters_per_deg(g_lat)
        # base offset from each member's own position
        # Sort labels by latitude descending for tidy stacking
        g_sorted = sorted(g, key=lambda idx: latc[idx], reverse=True)
        for k, idx in enumerate(g_sorted):
            r = clusters.iloc[idx]
            # total outward offset = base + k*step
            off_m = BASE_OFFSET_M + k*STACK_STEP_M
            # normal in lon/lat degrees
            dlon = (nx * off_m) / mlon
            dlat = (ny * off_m) / mlat
            ax.text(r["lon"] + dlon, r["lat"] + dlat,
                    f"{int(r['cluster_id'])}, {int(r['n_segments'])}",
                    fontsize=8, ha="left", va="bottom", color="black")

    ax.set_title("Circle clusters on track")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.axis("equal")
    plt.show()
    plt.close(fig)

# --- main ---
def main():
    ap = argparse.ArgumentParser(description="Circles → clusters with outside-of-polygon stacked labels")
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
