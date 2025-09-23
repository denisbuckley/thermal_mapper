#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
circle_clusters_v1r.py - v1o flow restored; adds alt_gained_m & av_climb_ms.
- CSVs -> /outputs ; debug -> /debugs
- Console summary prints: cluster_id, n_segments, turns, duration_min, alt_gained_m, av_climb_ms
- Plot labels unchanged: "cluster_id, turns, minutes"
"""

# --- ensure outputs dir exists (patched) ---
import os as _os_patch
_os_patch.makedirs('/Users/denisbuckley/PycharmProjects/chatgpt_igc/outputs', exist_ok=True)
import os, sys, argparse, math, logging
import numpy as np
import pandas as pd

# --- fixed project paths ---
PROJECT_ROOT = "/Users/denisbuckley/PycharmProjects/chatgpt_igc"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
DEBUG_DIR  = os.path.join(PROJECT_ROOT, "debugs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR,  exist_ok=True)

# logging
LOG_PATH = os.path.join(DEBUG_DIR, "circle_clusters_debug.log")
logging.basicConfig(filename=LOG_PATH, level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)



OUT_CSV = "/Users/denisbuckley/PycharmProjects/chatgpt_igc/outputs/circle_clusters_enriched.csv"

DEFAULT_IGC = "2020-11-08 Lumpy Paterson 108645.igc"

# --- geo helpers ---
def meters_per_deg(lat_deg: float):
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * max(1e-6, math.cos(math.radians(lat_deg)))
    return m_per_deg_lat, m_per_deg_lon

def to_local_xy(lat, lon, lat0, lon0):
    mlat, mlon = meters_per_deg(lat0)
    x = (lon - lon0) * mlon
    y = (lat - lat0) * mlat
    return x, y

def from_local_xy(x, y, lat0, lon0):
    mlat, mlon = meters_per_deg(lat0)
    lon = lon0 + x / mlon
    lat = lat0 + y / mlat
    return lat, lon

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1 = np.radians(lat1); p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlmb/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))

def polygon_orientation_area(lats, lons):
    area = 0.0
    n = len(lats)
    for i in range(n):
        j = (i+1) % n
        area += lons[i]*lats[j] - lons[j]*lats[i]
    return area/2.0  # >0 => CCW

def point_segment_distance_xy(px, py, x1, y1, x2, y2):
    vx, vy = x2-x1, y2-y1
    wx, wy = px-x1, py-y1
    seg_len2 = vx*vx + vy*vy
    if seg_len2 == 0:
        return math.hypot(px-x1, py-y1), 0.0, (x1, y1)
    t = max(0.0, min(1.0, (wx*vx + wy*vy)/seg_len2))
    projx, projy = x1 + t*vx, y1 + t*vy
    return math.hypot(px-projx, py-projy), t, (projx, projy)

# --- IGC minimal ---
def parse_igc_minimal(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"ERROR: IGC file does not exist: {path}")
        sys.exit(2)
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
                alt = float(line[25:30])  # GPS altitude field
            except:
                alt = np.nan
            t = hh*3600 + mm*60 + ss
            times.append(t); lats.append(lat); lons.append(lon); alts.append(alt)
    df = pd.DataFrame({"t": times, "lat": lats, "lon": lons, "alt": alts}).dropna()
    if df.empty:
        return pd.DataFrame(columns=["time_s","lat","lon","alt"])
    t = df["t"].to_numpy().astype(float)
    jumps = np.where(np.diff(t) < -43200)[0]  # midnight rollover
    if len(jumps) > 0:
        t[jumps[0]+1:] += 86400
    df["time_s"] = t
    return df[["time_s","lat","lon","alt"]]

# --- circle detection thresholds ---
C_MIN_ARC_DEG  = 30.0
C_MIN_RATE_DPS = 2.0
C_MAX_RATE_DPS = 70.0
C_MIN_RADIUS_M = 15.0
C_MAX_RADIUS_M = 300.0
C_MIN_SAMPLES  = 10
MAX_GAP_S      = 8.0

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

def detect_circles(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["i_start","i_end","t_start","t_end","dur_s","arc_deg","n_turns","lat","lon"])
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
    # speed & radius
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

# --- clustering with altitude metrics ---
CL_EPS_M     = 1200.0
CL_GAP_S     = 360.0
CL_MIN_COUNT = 1

def cluster_segments(seg_df: pd.DataFrame, df_fix: pd.DataFrame) -> pd.DataFrame:
    if seg_df.empty:
        return pd.DataFrame(columns=[
            "cluster_id","n_segments","n_turns_sum","duration_min","lat","lon","alt_gained_m","av_climb_ms"
        ])
    seg_df = seg_df.sort_values("t_start").reset_index(drop=True)

    def alt_at_index(idx: int) -> float:
        n = len(df_fix)
        if n == 0:
            return float("nan")
        idx = max(0, min(n-1, int(idx)))
        return float(df_fix.iloc[idx]["alt"])

    clusters = []
    i0s = int(seg_df.loc[0, "i_start"]); i0e = int(seg_df.loc[0, "i_end"])
    current = {
        "ids": [0],
        "lats": [seg_df.loc[0, "lat"]],
        "lons": [seg_df.loc[0, "lon"]],
        "n_turns": seg_df.loc[0, "n_turns"],
        "dur": seg_df.loc[0, "dur_s"],
        "t_start": seg_df.loc[0, "t_start"],
        "t_end": seg_df.loc[0, "t_end"],
        "alt_start": alt_at_index(i0s),
        "alt_end": alt_at_index(i0e),
    }

    def within(a_lat,a_lon,b_lat,b_lon):
        return haversine_m(a_lat,a_lon,b_lat,b_lon) <= CL_EPS_M

    for i in range(1, len(seg_df)):
        lat_i, lon_i = seg_df.loc[i,"lat"], seg_df.loc[i,"lon"]
        t_gap = seg_df.loc[i,"t_start"] - current["t_end"]
        cent_lat, cent_lon = float(np.mean(current["lats"])), float(np.mean(current["lons"]))
        if t_gap <= CL_GAP_S and within(lat_i, lon_i, cent_lat, cent_lon):
            current["ids"].append(i)
            current["lats"].append(lat_i); current["lons"].append(lon_i)
            current["n_turns"] += seg_df.loc[i,"n_turns"]; current["dur"] += seg_df.loc[i,"dur_s"]
            current["t_end"] = max(current["t_end"], seg_df.loc[i,"t_end"])
            current["alt_end"] = alt_at_index(int(seg_df.loc[i, "i_end"]))
        else:
            clusters.append(current)
            ist, ien = int(seg_df.loc[i, "i_start"]), int(seg_df.loc[i, "i_end"])
            current = {
                "ids": [i],
                "lats": [lat_i], "lons": [lon_i],
                "n_turns": seg_df.loc[i,"n_turns"],
                "dur": seg_df.loc[i,"dur_s"],
                "t_start": seg_df.loc[i,"t_start"],
                "t_end": seg_df.loc[i,"t_end"],
                "alt_start": alt_at_index(ist),
                "alt_end": alt_at_index(ien),
            }
    clusters.append(current)

    rows = []
    for cid, c in enumerate(clusters, 1):
        alt_gain = float(c["alt_end"] - c["alt_start"]) if (not math.isnan(c["alt_end"]) and not math.isnan(c["alt_start"])) else float("nan")
        av_climb = (alt_gain / c["dur"]) if (c["dur"] > 0 and not math.isnan(alt_gain)) else float("nan")
        rows.append({
            "cluster_id": cid,
            "n_segments": len(c["ids"]),
            "n_turns_sum": float(c["n_turns"]),
            "duration_min": float(c["dur"]),
            "lat": float(np.mean(c["lats"])),
            "lon": float(np.mean(c["lons"])),
            "alt_gained_m": alt_gain,
            "av_climb_ms": av_climb,
        })
    out = pd.DataFrame(rows)
    out = out[out["n_segments"] >= CL_MIN_COUNT].reset_index(drop=True)
    return out

# --- edge-wise label placement (use room) ---
BASE_OFFSET_M = 100.0
MIN_DS_M      = 120.0
MARGIN_FRAC   = 0.08

def assign_to_edges(track_lat, track_lon, c_lat, c_lon):
    n = len(track_lat)
    ccw = polygon_orientation_area(track_lat, track_lon) > 0
    out = []
    for plat, plon in zip(c_lat, c_lon):
        best = (1e18, None, None, None, None, None, None)
        for i in range(n-1):
            j = (i+1) % n
            lat0 = 0.5*(track_lat[i] + track_lat[j])
            lon0 = 0.5*(track_lon[i] + track_lon[j])
            x1, y1 = to_local_xy(track_lat[i], track_lon[i], lat0, lon0)
            x2, y2 = to_local_xy(track_lat[j], track_lon[j], lat0, lon0)
            px, py = to_local_xy(plat, plon, lat0, lon0)
            d, t, (projx, projy) = point_segment_distance_xy(px, py, x1, y1, x2, y2)
            ex, ey = x2-x1, y2-y1
            elen = math.hypot(ex, ey) or 1.0
            tx, ty = ex/elen, ey/elen
            if ccw:
                nx, ny = +ty, -tx
            else:
                nx, ny = -ty, +tx
            if d < best[0]:
                best = (d, i, t, (projx, projy), (lat0, lon0), (nx, ny), (tx, ty), elen)
        out.append(best[1:])
    return out

def plot_overlay(df: pd.DataFrame, clusters: pd.DataFrame):
    import matplotlib.pyplot as plt
    if df.empty:
        print("No B-records parsed from IGC; nothing to plot.")
        return
    lat = df["lat"].to_numpy(); lon = df["lon"].to_numpy()
    if lat[0] != lat[-1] or lon[0] != lon[-1]:
        lat = np.concatenate([lat, [lat[0]]])
        lon = np.concatenate([lon, [lon[0]]])
    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(lon, lat, color="skyblue", lw=1.0, alpha=0.7)
    if not clusters.empty:
        for _, r in clusters.iterrows():
            ax.plot(r["lon"], r["lat"], marker="x", color="black", markersize=6)
        c_lat = clusters["lat"].to_numpy()
        c_lon = clusters["lon"].to_numpy()
        assigns = assign_to_edges(lat, lon, c_lat, c_lon)
        by_edge = {}
        for idx, assign in enumerate(assigns):
            edge_i, t_on, proj_xy, frame, nvec, tvec, elen = assign
            by_edge.setdefault(edge_i, []).append((idx, t_on, proj_xy, frame, nvec, tvec, elen))
        for edge_i, items in by_edge.items():
            items.sort(key=lambda x: x[1])
            lat0 = 0.5*(lat[edge_i] + lat[(edge_i+1)%len(lat)])
            lon0 = 0.5*(lon[edge_i] + lon[(edge_i+1)%len(lon)])
            x1, y1 = to_local_xy(lat[edge_i], lon[edge_i], lat0, lon0)
            x2, y2 = to_local_xy(lat[(edge_i+1)%len(lat)], lon[(edge_i+1)%len(lon)], lat0, lon0)
            ex, ey = x2-x1, y2-y1
            elen = math.hypot(ex, ey) or 1.0
            tx, ty = ex/elen, ey/elen
            ccw = polygon_orientation_area(lat, lon) > 0
            nx, ny = (+ty, -tx) if ccw else (-ty, +tx)
            nlab = len(items)
            usable = max(0.0, elen*(1 - 2*MARGIN_FRAC))
            ds = max(MIN_DS_M, usable / max(1, nlab))
            start = MARGIN_FRAC * elen
            for k, (idx, t_on, (projx, projy), frame, _nvec, _tvec, _elen) in enumerate(items):
                s = min(start + k * ds, elen - MARGIN_FRAC*elen)
                lx = x1 + tx * s + nx * BASE_OFFSET_M
                ly = y1 + ty * s + ny * BASE_OFFSET_M
                llat, llon = from_local_xy(lx, ly, lat0, lon0)
                r = clusters.iloc[idx]
                turns_int = int(round(r["n_turns_sum"])) if pd.notna(r["n_turns_sum"]) else 0
                dur_min = r["duration_min"]/60.0 if pd.notna(r["duration_min"]) else 0.0
                ax.text(llon, llat, f"{int(r['cluster_id'])}, {turns_int}, {dur_min:.1f}m",
                        fontsize=8, ha="center", va="bottom", color="black")
    ax.set_title("Circle clusters on track")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.axis("equal")
    plt.show()
    plt.close(fig)

# --- main ---
def main():
    ap = argparse.ArgumentParser(description="Circles → clusters with edge-wise outside labels; CSVs to /outputs")
    ap.add_argument("igc", nargs="?", help="Path to IGC file")
    ap.add_argument("--clusters-csv", default=os.path.join(OUTPUT_DIR, "circle_clusters_enriched.csv"))
    ap.add_argument("--segments-csv", default=os.path.join(OUTPUT_DIR, "circle_segments.csv"))
    args = ap.parse_args()

    igc_path = args.igc or input(f"Enter path to IGC file [default: {DEFAULT_IGC}]: ").strip() or DEFAULT_IGC
    print(f"IGC: {igc_path}")
    df = parse_igc_minimal(igc_path)

    seg_df = detect_circles(df)
    clusters = cluster_segments(seg_df, df)

    # Write CSVs
    seg_df.to_csv(args.segments_csv, index=False)
    clusters.to_csv(OUT_CSV, index=False)

    # Console summary
    print(f"Fixes: {len(df)} | Segments: {len(seg_df)} | Clusters: {len(clusters)}")
    print(f"Segments CSV: {args.segments_csv}")
    print(f"Clusters CSV: {args.clusters_csv}")
    if not clusters.empty:
        view = clusters[["cluster_id", "n_segments", "n_turns_sum", "duration_min", "alt_gained_m", "av_climb_ms"]].copy()
        view["n_turns_sum"] = view["n_turns_sum"].round(1)
        view["duration_min"] = (view["duration_min"] / 60.0).round(1)
        view["alt_gained_m"] = view["alt_gained_m"].round(1)
        view["av_climb_ms"] = view["av_climb_ms"].round(2)
        view = view[["cluster_id", "n_segments", "n_turns_sum", "duration_min", "alt_gained_m", "av_climb_ms"]].rename(
            columns={"n_turns_sum": "turns"}
        )
        print("Cluster summary:")
        print(view.to_string(index=False))
    else:
        print("No clusters found.")

    # Plot at end
    plot_overlay(df, clusters)

if __name__ == "__main__":
    main()
