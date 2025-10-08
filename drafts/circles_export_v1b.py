
"""
circles_export_v1b.py
=================================
More tolerant full‑circle detection + same CSV/plot pipeline.
Fixes the “no circles” case by relaxing/robustifying turn accumulation.

Key changes from v1:
- Allow small opposite‑signed jitter without breaking the window
  via STEP_TOL_DEG and MIN_STEP_ABS.
- Slightly wider duration bounds [15, 60] s.
- Lower arc threshold to 300° but still require net ≥360° when
  REQUIRE_NET_360=True.
- Extra DEBUG prints (first few circles) so you can see what's detected.

Outputs (unchanged):
  outputs/circles_turns_<ts>.csv
  outputs/circles_clusters_<ts>.csv
  outputs/circles_cluster_pairs_<ts>.csv
"""

import sys, datetime, math
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from igc_utils import parse_igc, compute_derived, detect_tow_segment

# === Injected by apply_tuning_patches_v1.py ===
from tuning_loader import load_tuning, override_globals
_tuning = load_tuning("config/tuning_params.csv")
override_globals(globals(), _tuning, allowed={
    "C_MIN_ARC_DEG","C_MIN_RATE_DPS","C_MAX_RATE_DPS",
    "C_MIN_RADIUS_M","C_MAX_RADIUS_M","C_MIN_DIR_RATIO",
    "TIME_CAP_S","C_MAX_WIN_SAMPLES","C_EPS_M","C_MIN_SAMPLES"
})



# -------------------- parameters --------------------
CIRC_MIN_ARC_DEG = 300.0     # was 330
REQUIRE_NET_360  = True

CIRC_MIN_DURATION_S = 15.0   # was 18
CIRC_MAX_DURATION_S = 60.0   # was 45

HEADING_SMOOTH_S = 5.0
ALLOW_HEADING_GAP_S = 3.0

# tolerance for per‑step turn jitter
STEP_TOL_DEG = 4.0           # allow up to 4° opposite sign without breaking
MIN_STEP_ABS = 0.5           # ignore sub‑0.5° steps as noise

# clustering
EPS_M = 1500.0
MAX_GAP_S = 900.0

# emphasis
MIN_CLUSTER_COUNT = 2

# -------------------- logging tee --------------------
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# -------------------- helpers --------------------
R_EARTH_M = 6371000.0

def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*R_EARTH_M*np.arcsin(np.sqrt(a))

def ensure_time_sec(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not np.issubdtype(df['time'].dtype, np.datetime64):
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
    if 'sec' not in df.columns:
        df['sec'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds().astype(float)
    return df

def sec_per_fix(df: pd.DataFrame) -> float:
    ds = df['sec'].diff().dropna()
    dt = float(ds.median()) if len(ds) else 1.0
    if dt <= 0 or not (dt == dt):
        dt = 1.0
    return dt

def smooth_series(x: pd.Series, seconds: float, dt_fix: float) -> pd.Series:
    if seconds <= 0:
        return x.copy()
    w = max(3, int(round(seconds / max(dt_fix, 1e-6))))
    if w % 2 == 0:
        w += 1
    return x.rolling(window=w, center=True, min_periods=max(1, w//2)).mean()

def heading_deg(lat, lon):
    lat1 = np.radians(lat[:-1]); lon1 = np.radians(lon[:-1])
    lat2 = np.radians(lat[1:]);  lon2 = np.radians(lon[1:])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    brng = (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0
    return np.append(brng, brng[-1])

def angular_diff_deg(a2, a1):
    d = (a2 - a1 + 540.0) % 360.0 - 180.0
    return d

def params_box_text(dt):
    return (
        f"Cadence: {dt:.2f} s/fix\n"
        f"Circle: arc≥{int(CIRC_MIN_ARC_DEG)}°, net360={REQUIRE_NET_360}, dur∈[{int(CIRC_MIN_DURATION_S)},{int(CIRC_MAX_DURATION_S)}] s, "
        f"smooth={int(HEADING_SMOOTH_S)} s, tol={STEP_TOL_DEG}°\n"
        f"Cluster: eps={int(EPS_M)} m, gap≤{int(MAX_GAP_S/60)} min\n"
        f"Emphasis: count≥{MIN_CLUSTER_COUNT}"
    )

def add_params_box(ax, dt):
    ax.text(0.99, 0.01, params_box_text(dt), transform=ax.transAxes, va='bottom', ha='right',
            fontsize=9, family='monospace',
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.8, lw=0.5))

# -------------------- circle detection --------------------
def detect_full_circles(df: pd.DataFrame):
    """
    Return list of circles: (s_idx, e_idx, arc_deg, dur_s, signed_rate_deg_s, radius_m_est)
    """
    df = ensure_time_sec(df)
    dt = sec_per_fix(df)
    hdg = heading_deg(df["lat"].to_numpy(), df["lon"].to_numpy())
    hdg_s = smooth_series(pd.Series(hdg), HEADING_SMOOTH_S, dt).to_numpy()

    dtheta = angular_diff_deg(hdg_s[1:], hdg_s[:-1])

    circles = []
    n = len(df)
    i = 0
    while i < n - 2:
        s = i
        acc = 0.0
        direction = 0.0
        j = s + 1
        while j < n:
            step = dtheta[j-1]
            if abs(step) < MIN_STEP_ABS:
                j += 1
                continue
            if direction == 0.0 and abs(step) > MIN_STEP_ABS:
                direction = 1.0 if step > 0 else -1.0
            # allow small opposite steps within tolerance
            if direction * step < -STEP_TOL_DEG:
                break
            acc += step
            dur_s = df["sec"].iloc[j] - df["sec"].iloc[s]
            if dur_s > CIRC_MAX_DURATION_S:
                break
            # conditions to accept
            net_ok = (abs(acc) >= 360.0) if REQUIRE_NET_360 else (abs(acc) >= CIRC_MIN_ARC_DEG)
            if net_ok and dur_s >= CIRC_MIN_DURATION_S and abs(acc) >= CIRC_MIN_ARC_DEG:
                dist_m = haversine_m(df["lat"].iloc[s], df["lon"].iloc[s],
                                     df["lat"].iloc[j], df["lon"].iloc[j])
                R = max(dist_m / max(np.radians(abs(acc)), 1e-6), 1.0)
                rate = (acc) / max(dur_s, 1e-6)
                circles.append((s, j, abs(acc), dur_s, rate, R))
                i = j  # jump to end
                break
            j += 1
        i += 1

    return circles, dt

def circles_to_dataframe(df: pd.DataFrame, circles):
    rows = []
    for k, (s, e, arc, dur, rate, R) in enumerate(circles, start=1):
        mid = (s + e) // 2
        rows.append({
            "circle_id": k,
            "start_idx": int(s),
            "end_idx": int(e),
            "start_time": df["time"].iloc[s],
            "end_time": df["time"].iloc[e],
            "duration_s": float(dur),
            "arc_deg": float(arc),
            "turn_rate_deg_s": float(rate),
            "radius_m_est": float(R),
            "lat_mid": float(df["lat"].iloc[mid]),
            "lon_mid": float(df["lon"].iloc[mid]),
        })
    return pd.DataFrame(rows, columns=[
        "circle_id","start_idx","end_idx","start_time","end_time",
        "duration_s","arc_deg","turn_rate_deg_s","radius_m_est","lat_mid","lon_mid"
    ])

# -------------------- clustering --------------------
def clusters_from_circles(circles_df: pd.DataFrame, eps_m: float = EPS_M, max_gap_s: float = MAX_GAP_S):
    if circles_df.empty:
        return circles_df.assign(cluster_id=[]), pd.DataFrame(columns=[
            "cluster_id","n","start_time","end_time","mean_radius_m","center_lat","center_lon"
        ])

    df = circles_df.reset_index(drop=True)
    n = len(df)
    lats = df["lat_mid"].to_numpy(float)
    lons = df["lon_mid"].to_numpy(float)
    t0 = pd.to_datetime(df["start_time"]).to_numpy()
    t1 = pd.to_datetime(df["end_time"]).to_numpy()
    radius = df["radius_m_est"].to_numpy(float)

    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            d = float(haversine_m(lats[i], lons[i], lats[j], lons[j]))
            dist[i, j] = dist[j, i] = d

    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            spatial_ok = dist[i, j] <= eps_m
            gap_ij = max(0.0, (pd.to_datetime(t0[j]) - pd.to_datetime(t1[i])).total_seconds())
            gap_ji = max(0.0, (pd.to_datetime(t0[i]) - pd.to_datetime(t1[j])).total_seconds())
            temporal_ok = (gap_ij <= max_gap_s) or (gap_ji <= max_gap_s) or (pd.to_datetime(t1[i]) >= pd.to_datetime(t0[j])) or (pd.to_datetime(t1[j]) >= pd.to_datetime(t0[i]))
            if spatial_ok and temporal_ok:
                adj[i].append(j); adj[j].append(i)

    visited = np.zeros(n, dtype=bool)
    comp_id = np.full(n, -1, dtype=int)
    clusters = []
    cid = 0
    for i in range(n):
        if visited[i]: continue
        q = [i]; visited[i] = True; members = []
        while q:
            k = q.pop(0); members.append(k)
            for j in adj[k]:
                if not visited[j]:
                    visited[j] = True; q.append(j)
        idx = np.array(members, int)
        w = np.clip(1.0/np.maximum(radius[idx], 1.0), 0.1, None)
        lat_c = float(np.average(lats[idx], weights=w))
        lon_c = float(np.average(lons[idx], weights=w))
        mean_r = float(np.mean(radius[idx])) if len(idx) else float('nan')
        start_time = pd.to_datetime(t0[idx].min()).to_pydatetime()
        end_time   = pd.to_datetime(t1[idx].max()).to_pydatetime()
        clusters.append({
            "cluster_id": cid+1,
            "n": int(len(idx)),
            "start_time": start_time,
            "end_time": end_time,
            "mean_radius_m": mean_r,
            "center_lat": lat_c,
            "center_lon": lon_c,
        })
        comp_id[idx] = cid+1
        cid += 1

    clusters_df = pd.DataFrame(clusters, columns=[
        "cluster_id","n","start_time","end_time","mean_radius_m","center_lat","center_lon"
    ])
    df_with_cluster = df.copy()
    df_with_cluster["cluster_id"] = comp_id
    return df_with_cluster, clusters_df

def successive_cluster_pairs(clusters_df: pd.DataFrame) -> pd.DataFrame:
    if clusters_df.empty or len(clusters_df) == 1:
        return pd.DataFrame(columns=[
            "from_id","to_id","gap_minutes","distance_km","implied_kmh",
            "from_end","to_start","from_lat","from_lon","to_lat","to_lon"
        ])
    df = clusters_df.sort_values("start_time").reset_index(drop=True)
    rows = []
    for i in range(len(df)-1):
        a = df.iloc[i]; b = df.iloc[i+1]
        gap_s = (b["start_time"] - a["end_time"]).total_seconds()
        gap_min = gap_s / 60.0
        d_m = haversine_m(a["center_lat"], a["center_lon"], b["center_lat"], b["center_lon"])
        d_km = d_m / 1000.0
        kmh = d_km / max(gap_s/3600.0, 1e-9) if gap_s > 0 else None
        rows.append({
            "from_id": int(a["cluster_id"]), "to_id": int(b["cluster_id"]),
            "gap_minutes": round(gap_min, 1),
            "distance_km": round(d_km, 2),
            "implied_kmh": round(kmh, 1) if kmh is not None else None,
            "from_end": a["end_time"], "to_start": b["start_time"],
            "from_lat": a["center_lat"], "from_lon": a["center_lon"],
            "to_lat": b["center_lat"], "to_lon": b["center_lon"],
        })
    return pd.DataFrame(rows)

# -------------------- plotting --------------------
def plot_map_with_circles_and_clusters(df: pd.DataFrame, circles, clusters_df: pd.DataFrame, dt: float):
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    ax.plot(df["lon"], df["lat"], color="green", lw=1.2, alpha=0.7, label="Track")
    for (s,e,_,_,_,_) in circles:
        ax.plot(df.loc[s:e, "lon"], df.loc[s:e, "lat"], color="royalblue", lw=2.0, alpha=0.65)
        ax.scatter([df["lon"].iloc[s], df["lon"].iloc[e]], [df["lat"].iloc[s], df["lat"].iloc[e]], s=10, c="red", zorder=5)
    big = clusters_df[clusters_df["n"] >= MIN_CLUSTER_COUNT]
    for _, row in big.iterrows():
        ax.scatter(row["center_lon"], row["center_lat"], marker='x', s=120, c='k', zorder=6)
        ax.annotate(f"T{int(row['cluster_id'])}",
                    (row["center_lon"], row["center_lat"]),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=10, weight='bold')
    add_params_box(ax, dt)
    ax.set_title("Bird’s‑eye — Circles and Thermal clusters (Track=green, Circles=blue, Centers=black X)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.legend(loc='upper left'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# -------------------- main --------------------
def main():
    ts = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    logf = open(f"circles_export_{ts}.txt", "w")
    sys.stdout = sys.stderr = Tee(sys.stdout, logf)

    path = "2020-11-08 Lumpy Paterson 108645.igc_subset"
    print(f"[circles v1b] Parsing IGC: {path}")
    df = parse_igc(path)
    if len(df) == 0:
        print("[circles v1b] No rows parsed; aborting."); return
    df = compute_derived(df)
    df = ensure_time_sec(df)

    try:
        tow = detect_tow_segment(df)
    except Exception as e:
        print(f"[circles v1b] Tow detect error: {e}")
        tow = None
    if tow and isinstance(tow, tuple) and len(tow) == 2:
        s,e = tow
        print(f"[circles v1b] Tow detected: {s}→{e}. Excluding tow.")
        df = df.iloc[e+1:].reset_index(drop=True)
        df = ensure_time_sec(df)
    else:
        print("[circles v1b] Tow not clearly detected; using full trace.")
    print(f"[circles v1b] Points after tow trim: {len(df)}")

    circles, dt = detect_full_circles(df)
    print(f"[circles v1b] Cadence ≈ {dt:.2f} s/fix | circles detected: {len(circles)}")
    for idx, c in enumerate(circles[:5]):
        s,e,arc,dur,rate,R = c
        print(f"  circle[{idx}]: idx {s}->{e}  arc≈{arc:.1f}°  dur={dur:.1f}s  rate={rate:.1f}°/s  R≈{R:.0f} m")

    circ_df = circles_to_dataframe(df, circles)
    circ_with_cluster, clusters_df = clusters_from_circles(circ_df, eps_m=EPS_M, max_gap_s=MAX_GAP_S)
    pairs_df = successive_cluster_pairs(clusters_df)

    out_dir = Path("outputs"); out_dir.mkdir(parents=True, exist_ok=True)
    csv_circles = out_dir / f"circles_turns_{ts}.csv"
    circ_with_cluster.to_csv(csv_circles, index=False)
    print(f"[circles v1b] Wrote circles CSV (with cluster_id): {csv_circles} ({len(circ_with_cluster)} rows)")
    csv_clusters = out_dir / f"circles_clusters_{ts}.csv"
    clusters_df.to_csv(csv_clusters, index=False)
    print(f"[circles v1b] Wrote clusters CSV: {csv_clusters} ({len(clusters_df)} rows)")
    csv_pairs = out_dir / f"circles_cluster_pairs_{ts}.csv"
    pairs_df.to_csv(csv_pairs, index=False)
    print(f"[circles v1b] Wrote cluster separations CSV: {csv_pairs} ({len(pairs_df)} rows)")

    if len(df):
        plot_map_with_circles_and_clusters(df, circles, clusters_df, dt)

    return circ_with_cluster, clusters_df, pairs_df

if __name__ == "__main__":
    main()