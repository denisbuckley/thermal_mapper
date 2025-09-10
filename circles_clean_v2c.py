
"""
circles_clean_v2c.py
====================
Goal: detect *more* raw circles (green ✕) inside each continuous thermalling
segment by relaxing the "single-direction only" constraint, but still reject
zig‑zag noise.

Approach
--------
We integrate the *absolute* turn magnitude and allow short sign flips,
as long as the majority of the turn in the window is the same direction.

Emit a circle whenever:
    abs_cum_turn >= MIN_ARC_DEG AND (same_sign_ratio >= MIN_DIR_RATIO)

After emission we *subtract* MIN_ARC_DEG from |cum| (keep remainder) and keep
scanning so multiple circles inside the same run are captured.

Plot stays clean: Track=green, Circles=green ✕, Centers=black ✕. No labels.
"""

import sys, math, datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from igc_utils import parse_igc, compute_derived, detect_tow_segment

# ---- Tunables ----
MIN_ARC_DEG = 300.0          # full circle threshold
HEADING_SMOOTH_S = 3.0       # heading smoothing
MAX_LEN_SAMPLES = 2400       # safety ceiling per search window
MIN_DIR_RATIO = 0.65         # fraction of same-direction turn required

EPS_M = 1500.0               # clustering radius (m)
MAX_GAP_S = 900.0            # temporal bridge (s)
MIN_CLUSTER_COUNT = 1

# Markers
CENTER_S = 70; CENTER_LW = 1.4; CENTER_ALPHA = 0.85
CIRCLE_S = 80;  CIRCLE_LW = 2.0;  CIRCLE_Z = 7

R_EARTH_M = 6371000.0

class Tee:
    def __init__(self, *files): self.files = files
    def write(self, obj): 
        for f in self.files: f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()

def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*R_EARTH_M*np.arcsin(np.sqrt(a))

def ensure_time_sec(df):
    df = df.copy()
    if not np.issubdtype(df['time'].dtype, np.datetime64):
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
    if 'sec' not in df.columns:
        df['sec'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds().astype(float)
    return df

def sec_per_fix(df):
    ds = df['sec'].diff().dropna()
    dt = float(ds.median()) if len(ds) else 1.0
    if not np.isfinite(dt) or dt <= 0: dt = 1.0
    return dt

def smooth_series(x: pd.Series, seconds: float, dt_fix: float) -> pd.Series:
    if seconds <= 0: return x.copy()
    w = max(3, int(round(seconds / max(dt_fix, 1e-6))))
    if w % 2 == 0: w += 1
    return x.rolling(window=w, center=True, min_periods=max(1, w//2)).mean()

def bearing_deg(lat1, lon1, lat2, lon2):
    lat1 = math.radians(lat1); lon1 = math.radians(lon1)
    lat2 = math.radians(lat2); lon2 = math.radians(lon2)
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0

def circ_diff_deg(a, b):  # signed smallest diff b-a in (-180, 180]
    return (b - a + 540.0) % 360.0 - 180.0

def detect_circles(df,
                   min_circle_heading_deg=MIN_ARC_DEG,
                   max_len_samples=MAX_LEN_SAMPLES,
                   smooth_s=HEADING_SMOOTH_S,
                   min_dir_ratio=MIN_DIR_RATIO):
    """
    Majority-direction cumulative heading detector.
    Returns: list[(start_idx, end_idx, arc_signed_deg)], cadence_sec_per_fix
    """
    df = ensure_time_sec(df)
    dt = sec_per_fix(df)

    # headings
    headings = [0.0]
    for i in range(1, len(df)):
        headings.append(bearing_deg(df.loc[i-1,'lat'], df.loc[i-1,'lon'],
                                    df.loc[i,'lat'], df.loc[i,'lon']))
    hdg = pd.Series(headings)
    hdg_s = smooth_series(hdg, smooth_s, dt).to_numpy()

    circles = []
    n = len(df)

    i = 0
    while i < n-3:
        j = i + 1
        last_h = hdg_s[i]
        cum_abs = 0.0
        cum_signed = 0.0
        pos_abs = 0.0
        neg_abs = 0.0
        last_emit_j = i

        while j < n and (j - i) < max_len_samples:
            now_h = hdg_s[j]
            d = circ_diff_deg(last_h, now_h)
            last_h = now_h

            if d > 0: pos_abs += abs(d)
            elif d < 0: neg_abs += abs(d)

            cum_abs += abs(d)
            cum_signed += d

            # emit for each threshold passed and direction majority ok
            while cum_abs >= min_circle_heading_deg:
                maj = max(pos_abs, neg_abs)
                tot = pos_abs + neg_abs if (pos_abs+neg_abs) > 0 else 1.0
                ratio = maj / tot
                if ratio >= min_dir_ratio:
                    # circle spans since last emit to current j
                    s_idx = last_emit_j
                    e_idx = j
                    if e_idx - s_idx >= 2:
                        # signed by majority direction
                        sign = 1 if pos_abs >= neg_abs else -1
                        circles.append((s_idx, e_idx, sign * min_circle_heading_deg))
                        last_emit_j = j
                # subtract one threshold worth from the *abs* budget, preserve bias
                if pos_abs >= neg_abs:
                    pos_abs = max(0.0, pos_abs - min_circle_heading_deg)
                else:
                    neg_abs = max(0.0, neg_abs - min_circle_heading_deg)
                cum_abs = max(0.0, cum_abs - min_circle_heading_deg)

            j += 1

        # advance by small amount to continue scanning (avoid infinite loops)
        i = max(i + 5, last_emit_j)  # small overlap

    return circles, dt

def circles_to_dataframe(df, circles):
    rows = []
    for k, (s, e, arc) in enumerate(circles, start=1):
        mid = (s + e) // 2
        dur = float(df['sec'].iloc[e] - df['sec'].iloc[s])
        chord_m = haversine_m(df['lat'].iloc[s], df['lon'].iloc[s],
                              df['lat'].iloc[e], df['lon'].iloc[e])
        radius_est = max(chord_m / max(np.radians(abs(arc)), 1e-6), 1.0)
        rows.append({
            "circle_id": k, "start_idx": int(s), "end_idx": int(e),
            "start_time": df["time"].iloc[s], "end_time": df["time"].iloc[e],
            "duration_s": dur, "arc_deg": float(abs(arc)),
            "turn_rate_deg_s": float(arc / max(dur, 1e-6)),
            "radius_m_est": float(radius_est),
            "lat_mid": float(df["lat"].iloc[mid]), "lon_mid": float(df["lon"].iloc[mid]),
        })
    cols = ["circle_id","start_idx","end_idx","start_time","end_time","duration_s",
            "arc_deg","turn_rate_deg_s","radius_m_est","lat_mid","lon_mid"]
    return pd.DataFrame(rows, columns=cols)

def clusters_from_circles(circles_df, eps_m=EPS_M, max_gap_s=MAX_GAP_S):
    if circles_df.empty:
        return circles_df.assign(cluster_id=[]), pd.DataFrame(columns=[
            "cluster_id","n","start_time","end_time","mean_radius_m","center_lat","center_lon"
        ])

    df = circles_df.reset_index(drop=True)
    n = len(df)
    lats = df["lat_mid"].to_numpy(float); lons = df["lon_mid"].to_numpy(float)
    t0 = pd.to_datetime(df["start_time"]).to_numpy()
    t1 = pd.to_datetime(df["end_time"]).to_numpy()
    radius = df["radius_m_est"].to_numpy(float)

    dist = np.zeros((n, n), dtype=float)
    for a in range(n):
        for b in range(a+1, n):
            d = float(haversine_m(lats[a], lons[a], lats[b], lons[b]))
            dist[a, b] = dist[b, a] = d

    adj = [[] for _ in range(n)]
    for a in range(n):
        for b in range(a+1, n):
            spatial_ok = dist[a, b] <= eps_m
            gap_ab = max(0.0, (pd.to_datetime(t0[b]) - pd.to_datetime(t1[a])).total_seconds())
            gap_ba = max(0.0, (pd.to_datetime(t0[a]) - pd.to_datetime(t1[b])).total_seconds())
            temporal_ok = (gap_ab <= max_gap_s) or (gap_ba <= max_gap_s) or \
                          (pd.to_datetime(t1[a]) >= pd.to_datetime(t0[b])) or \
                          (pd.to_datetime(t1[b]) >= pd.to_datetime(t0[a]))
            if spatial_ok and temporal_ok:
                adj[a].append(b); adj[b].append(a)

    visited = np.zeros(n, dtype=bool)
    comp_id = np.full(n, -1, dtype=int)
    clusters = []
    cid = 0
    for a in range(n):
        if visited[a]: continue
        q = [a]; visited[a] = True; members = []
        while q:
            k = q.pop(0); members.append(k)
            for b in adj[k]:
                if not visited[b]: visited[b] = True; q.append(b)
        idx = np.array(members, int)
        w = np.clip(1.0/np.maximum(radius[idx], 1.0), 0.1, None)
        lat_c = float(np.average(lats[idx], weights=w))
        lon_c = float(np.average(lons[idx], weights=w))
        mean_r = float(np.mean(radius[idx])) if len(idx) else float('nan')
        start_time = pd.to_datetime(t0[idx].min()).to_pydatetime()
        end_time   = pd.to_datetime(t1[idx].max()).to_pydatetime()
        clusters.append({
            "cluster_id": cid+1, "n": int(len(idx)),
            "start_time": start_time, "end_time": end_time,
            "mean_radius_m": mean_r, "center_lat": lat_c, "center_lon": lon_c,
        })
        comp_id[idx] = cid+1; cid += 1

    clusters_df = pd.DataFrame(clusters, columns=[
        "cluster_id","n","start_time","end_time","mean_radius_m","center_lat","center_lon"
    ])
    df_with_cluster = df.copy(); df_with_cluster["cluster_id"] = comp_id
    return df_with_cluster, clusters_df

def params_box_text(dt):
    return (
        f"Cadence: {dt:.2f} s/fix\n"
        f"Circle: arc≥{int(MIN_ARC_DEG)}°, smooth={int(HEADING_SMOOTH_S)} s, max_win≤{MAX_LEN_SAMPLES} samples, dir≥{int(MIN_DIR_RATIO*100)}%\n"
        f"Cluster: eps={int(EPS_M)} m, gap≤{int(MAX_GAP_S/60)} min"
    )

def add_params_box(ax, dt):
    ax.text(0.99, 0.01, params_box_text(dt), transform=ax.transAxes, va='bottom', ha='right',
            fontsize=9, family='monospace',
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, lw=0.5))

def plot_map(df, circles, clusters_df, dt):
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    ax.plot(df["lon"], df["lat"], color="green", lw=1.2, alpha=0.7, label="Track")
    # centers first
    if not clusters_df.empty:
        ax.scatter(clusters_df["center_lon"], clusters_df["center_lat"], marker='x', c='k',
                   s=CENTER_S, linewidths=CENTER_LW, alpha=CENTER_ALPHA, zorder=6)
    # circles on top
    if circles:
        mids_lon = [float(df["lon"].iloc[(s+e)//2]) for (s,e,_) in circles]
        mids_lat = [float(df["lat"].iloc[(s+e)//2]) for (s,e,_) in circles]
        ax.scatter(mids_lon, mids_lat, marker='x', c='green', s=CIRCLE_S,
                   linewidths=CIRCLE_LW, zorder=CIRCLE_Z, label="Circles")
    add_params_box(ax, dt)
    ax.set_title("Circles → Thermals (Track=green, Circles=green ✕, Centers=black ✕) — No labels")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.legend(loc='upper left'); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

def main():
    ts = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    logf = open(f"circles_clean_{ts}.txt", "w")
    sys.stdout = sys.stderr = Tee(sys.stdout, logf)

    igc_file = "2020-11-08 Lumpy Paterson 108645.igc"
    print(f"[circles-clean v2c] Parsing IGC: {igc_file}")
    df = parse_igc(igc_file)
    if df.empty:
        print("[circles-clean v2c] No rows parsed; aborting."); return
    df = compute_derived(df); df = ensure_time_sec(df)

    try:
        tow = detect_tow_segment(df)
    except Exception as e:
        print(f"[circles-clean v2c] Tow detect error: {e}"); tow = None
    if isinstance(tow, tuple) and len(tow) == 2:
        s,e = tow
        print(f"[circles-clean v2c] Tow detected: {s}→{e}. Excluding tow.")
        df = df.iloc[e+1:].reset_index(drop=True); df = ensure_time_sec(df)
    else:
        print("[circles-clean v2c] Tow not clearly detected; using full trace.")
    print(f"[circles-clean v2c] Points after tow trim: {len(df)}")

    circles, dt = detect_circles(df)
    print(f"[circles-clean v2c] Cadence ≈ {dt:.2f} s/fix | raw circles detected: {len(circles)}")

    circ_df = circles_to_dataframe(df, circles)
    circ_with_cluster, clusters_df = clusters_from_circles(circ_df, EPS_M, MAX_GAP_S)

    out = Path("outputs"); out.mkdir(parents=True, exist_ok=True)
    circ_with_cluster.to_csv(out / f"circles_turns_{ts}.csv", index=False)
    clusters_df.to_csv(out / f"circles_clusters_{ts}.csv", index=False)

    plot_map(df, circles, clusters_df, dt)

if __name__ == "__main__":
    main()
