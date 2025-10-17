#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
circle_clusters_v1e.py — circles → clusters with enriched metrics + optional plot.

Outputs (in working dir unless overridden):
  - circle_clusters_enriched.csv  (one row per cluster)
  - circle_segments.csv           (per circling segment)
  - Debug log: /Users/denisbuckley/PycharmProjects/chatgpt_igc/debugs/circle_clusters_debug.log
  - Optional: --save-plot cluster_overlay.png (bird's-eye track with cluster labels)

Run:
  python circle_clusters_v1e.py                 # prompts for IGC (default provided)
  python circle_clusters_v1e.py "file.igc_subset" --save-plot overlay.png
"""
import os, sys, argparse, math, logging
import numpy as np
import pandas as pd

# --- logging (absolute path you specified) ---
LOG_DIR = "/debugs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "circle_clusters_debug.log")
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_IGC = "2020-11-08 Lumpy Paterson 108645.igc_subset"

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

# --- parser (minimal B-records) ---
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
    # time seconds, handling wrap
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
    # headings
    head = np.zeros(n, dtype=float); head[0] = np.nan
    for i in range(1, n):
        head[i] = bearing_deg(lat[i-1], lon[i-1], lat[i], lon[i])
    if n > 1 and np.isnan(head[0]): head[0] = head[1]
    h_unwrap = unwrap_degrees(head)
    # kinematics
    dt = np.diff(t, prepend=t[0]); dt[dt <= 0] = np.nan
    dpsi = np.diff(h_unwrap, prepend=h_unwrap[0])
    rate = np.where(np.isfinite(dt), dpsi/np.where(dt==0, np.nan, dt), np.nan)  # deg/s
    # speed & radius
    dist = np.zeros(n, dtype=float)
    for i in range(1, n):
        dist[i] = haversine_m(lat[i-1], lon[i-1], lat[i], lon[i])
    v = np.where(np.isfinite(dt), dist/np.where(dt==0, np.nan, dt), np.nan)  # m/s
    omega_rad = np.deg2rad(rate)
    with np.errstate(divide="ignore", invalid="ignore"):
        radius = np.abs(v / np.where(omega_rad==0, np.nan, omega_rad))  # m

    segments = []
    i = 1
    while i < n:
        if not np.isfinite(rate[i]) or dt[i] > MAX_GAP_S:
            i += 1; continue
        sgn = np.sign(rate[i]) if rate[i] != 0 else 0.0
        start = i; cum_arc = 0.0; last_t = t[i]; valid = 0
        sum_rate = 0.0; sum_v = 0.0; sum_r = 0.0; count = 0
        while i < n:
            if not np.isfinite(rate[i]) or (t[i]-last_t) > MAX_GAP_S: break
            if abs(rate[i]) < C_MIN_RATE_DPS or abs(rate[i]) > C_MAX_RATE_DPS: break
            if sgn != 0 and np.sign(rate[i]) != sgn: break
            if not (C_MIN_RADIUS_M <= radius[i] <= C_MAX_RADIUS_M or not np.isfinite(radius[i])): break
            if i > start: cum_arc += abs(dpsi[i])
            last_t = t[i]; valid += 1
            if np.isfinite(rate[i]): sum_rate += abs(rate[i])
            if np.isfinite(v[i]): sum_v += v[i]
            if np.isfinite(radius[i]): sum_r += radius[i]
            count += 1
            i += 1
        end = i - 1
        dur = t[end] - t[start] if end > start else 0.0
        if valid >= C_MIN_SAMPLES and cum_arc >= C_MIN_ARC_DEG and dur > 0:
            mean_rate = (sum_rate / max(1, count))
            mean_v    = (sum_v / max(1, count))
            mean_r    = (sum_r / max(1, count)) if sum_r > 0 else np.nan
            # bank estimate: tan(bank) = v^2 / (r*g)
            g = 9.80665
            bank = math.degrees(math.atan2(mean_v*mean_v, (mean_r*g))) if (mean_r and mean_r>0) else np.nan
            segments.append({
                "i_start": start, "i_end": end,
                "t_start": t[start], "t_end": t[end], "dur_s": dur,
                "arc_deg": cum_arc, "n_turns": cum_arc/360.0,
                "lat": float(np.nanmean(lat[start:end+1])),
                "lon": float(np.nanmean(lon[start:end+1])),
                "mean_rate_dps": float(mean_rate) if np.isfinite(mean_rate) else np.nan,
                "mean_radius_m": float(mean_r) if np.isfinite(mean_r) else np.nan,
                "mean_speed_mps": float(mean_v) if np.isfinite(mean_v) else np.nan,
                "est_bank_deg": float(bank) if np.isfinite(bank) else np.nan
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
        return pd.DataFrame(columns=[
            "cluster_id","n_segments","n_turns_sum","dur_s_sum","lat","lon",
            "spread_m","mean_radius_m","mean_turn_rate_dps","mean_speed_mps","mean_bank_deg",
            "t_start","t_end"
        ])
    seg_df = seg_df.sort_values("t_start").reset_index(drop=True)
    clusters = []
    current = {"ids": [0], "lats": [seg_df.loc[0,"lat"]], "lons": [seg_df.loc[0,"lon"]],
               "n_turns": seg_df.loc[0,"n_turns"], "dur": seg_df.loc[0,"dur_s"],
               "rates": [seg_df.loc[0,"mean_rate_dps"]], "radii": [seg_df.loc[0,"mean_radius_m"]],
               "speeds": [seg_df.loc[0,"mean_speed_mps"]], "banks": [seg_df.loc[0,"est_bank_deg"]],
               "t_start": seg_df.loc[0,"t_start"], "t_end": seg_df.loc[0,"t_end"]}
    def within(a_lat,a_lon,b_lat,b_lon): return haversine_m(a_lat,a_lon,b_lat,b_lon) <= CL_EPS_M
    for i in range(1, len(seg_df)):
        lat_i, lon_i = seg_df.loc[i,"lat"], seg_df.loc[i,"lon"]
        t_gap = seg_df.loc[i,"t_start"] - current["t_end"]
        cent_lat, cent_lon = float(np.mean(current["lats"])), float(np.mean(current["lons"]))
        if t_gap <= CL_GAP_S and within(lat_i, lon_i, cent_lat, cent_lon):
            current["ids"].append(i); current["lats"].append(lat_i); current["lons"].append(lon_i)
            current["n_turns"] += seg_df.loc[i,"n_turns"]; current["dur"] += seg_df.loc[i,"dur_s"]
            current["rates"].append(seg_df.loc[i,"mean_rate_dps"]); current["radii"].append(seg_df.loc[i,"mean_radius_m"])
            current["speeds"].append(seg_df.loc[i,"mean_speed_mps"]); current["banks"].append(seg_df.loc[i,"est_bank_deg"])
            current["t_end"] = max(current["t_end"], seg_df.loc[i,"t_end"])
        else:
            clusters.append(current)
            current = {"ids": [i], "lats": [lat_i], "lons": [lon_i],
                       "n_turns": seg_df.loc[i,"n_turns"], "dur": seg_df.loc[i,"dur_s"],
                       "rates": [seg_df.loc[i,"mean_rate_dps"]], "radii": [seg_df.loc[i,"mean_radius_m"]],
                       "speeds": [seg_df.loc[i,"mean_speed_mps"]], "banks": [seg_df.loc[i,"est_bank_deg"]],
                       "t_start": seg_df.loc[i,"t_start"], "t_end": seg_df.loc[i,"t_end"]}
    clusters.append(current)
    rows = []
    for cid, c in enumerate(clusters, 1):
        # spatial spread as max distance from centroid
        cent_lat, cent_lon = float(np.mean(c["lats"])), float(np.mean(c["lons"]))
        dists = [haversine_m(cent_lat, cent_lon, la, lo) for la,lo in zip(c["lats"], c["lons"])]
        spread = float(np.nanmax(dists)) if len(dists) else 0.0
        # robust averages (ignore NaN)
        def nanmean(xs): 
            arr = np.array(xs, dtype=float); 
            return float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan
        rows.append({
            "cluster_id": cid,
            "n_segments": len(c["ids"]),
            "n_turns_sum": float(c["n_turns"]),
            "dur_s_sum": float(c["dur"]),
            "lat": cent_lat,
            "lon": cent_lon,
            "spread_m": spread,
            "mean_radius_m": nanmean(c["radii"]),
            "mean_turn_rate_dps": nanmean(c["rates"]),
            "mean_speed_mps": nanmean(c["speeds"]),
            "mean_bank_deg": nanmean(c["banks"]),
            "t_start": float(c["t_start"]),
            "t_end": float(c["t_end"]),
        })
    out = pd.DataFrame(rows)
    out = out[out["n_segments"] >= CL_MIN_COUNT].reset_index(drop=True)
    logger.info(f"Clusters: {len(out)} kept; sum_n_turns={out['n_turns_sum'].sum():.2f}")
    return out

# --- plotting ---
def plot_overlay(df, clusters, save_path=None, show=False):
    import matplotlib.pyplot as plt
    lat = df["lat"].to_numpy(); lon = df["lon"].to_numpy()
    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(lon, lat, lw=1.0, alpha=0.7)
    for _, r in clusters.iterrows():
        ax.scatter(r["lon"], r["lat"], s=40)
        ax.text(r["lon"], r["lat"],
                f"#{int(r['cluster_id'])}\nseg={int(r['n_segments'])} turns~{r['n_turns_sum']:.1f}\nR~{r['mean_radius_m']:.0f}m bank~{r['mean_bank_deg']:.0f}°",
                fontsize=8, ha="left", va="bottom")
    ax.set_title("Circle clusters on track")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.axis("equal")
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

# --- main ---
def main():
    ap = argparse.ArgumentParser(description="Circles → clusters with enriched metrics + optional plot")
    ap.add_argument("igc_subset", nargs="?", help="Path to IGC file")
    ap.add_argument("--clusters-csv", default="circle_clusters_enriched.csv", help="Output CSV for clusters")
    ap.add_argument("--segments-csv", default="circle_segments.csv", help="Output CSV for circling segments")
    ap.add_argument("--save-plot", default=None, help="Save plot PNG to this path")
    ap.add_argument("--show", action="store_true", help="Show interactive plot (off by default)")
    args = ap.parse_args()

    igc_path = args.igc or input(f"Enter path to IGC file [default: {DEFAULT_IGC}]: ").strip() or DEFAULT_IGC
    df = parse_igc_minimal(igc_path)
    logger.info(f"[clusters v1e] Fixes: {len(df)}")

    seg_df = detect_circles(df)
    clusters = cluster_segments(seg_df)

    seg_df.to_csv(args.segments_csv, index=False)
    clusters.to_csv(args.clusters_csv, index=False)
    logger.info(f"Wrote clusters CSV: {args.clusters_csv}  rows={len(clusters)}")
    logger.info(f"Wrote segments CSV: {args.segments_csv}  rows={len(seg_df)}")

    if args.save_plot or args.show:
        plot_overlay(df, clusters, save_path=args.save_plot, show=args.show)

    print(f"Segments: {len(seg_df)} | Clusters: {len(clusters)}  (Σn_turns={clusters['n_turns_sum'].sum():.1f})")
    print(f"CSV: {args.clusters_csv}  | Segments CSV: {args.segments_csv}")
    if args.save_plot:
        print(f"Plot saved: {args.save_plot}")

if __name__ == "__main__":
    main()
