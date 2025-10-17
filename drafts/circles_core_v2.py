
"""
circles_core_v2.py
==================
Purpose: keep this tight and focused on *circles → thermalling*.

Pipeline
--------
1) Detect *circles* = cumulative heading change >= MIN_ARC_DEG (default 300°).
   - Light heading smoothing (HEADING_SMOOTH_S).
   - Window cap only (MAX_LEN_SAMPLES) — no hard duration gates.
2) Build *circle clusters* (thermals) allowing drift:
   - Graph connectivity if circles are within EPS_M (meters) AND
     are temporally overlapping or within MAX_GAP_S seconds.
3) Optional *valley merge* between adjacent clusters:
   - If cluster centers are within MERGE_EPS_M and time gap <= MERGE_GAP_S,
     merge them into one “big thermal”. (This mirrors the altitude “valley” idea,
     but it remains circles-only — no altitude signal required.)
4) Export CSVs and draw a clean bird’s-eye plot:
   - Track = green
   - Circles = green ✕ at midpoints
   - Cluster centers = black ✕ with labels T1, T2, …
   - Optional mean-radius ring per cluster

Outputs (./outputs):
  - circles_turns_<ts>.csv
  - circles_clusters_<ts>.csv
  - circles_cluster_pairs_<ts>.csv
  - circles_core_<ts>.txt (console tee)
"""

import sys, math, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from archive.igc_utils import parse_igc, compute_derived, detect_tow_segment

# === Injected by apply_tuning_patches_v1.py ===
from tuning_loader import load_tuning, override_globals
_tuning = load_tuning("config/tuning_params.csv")
override_globals(globals(), _tuning, allowed={
    "C_MIN_ARC_DEG","C_MIN_RATE_DPS","C_MAX_RATE_DPS",
    "C_MIN_RADIUS_M","C_MAX_RADIUS_M","C_MIN_DIR_RATIO",
    "TIME_CAP_S","C_MAX_WIN_SAMPLES","C_EPS_M","C_MIN_SAMPLES"
})



# --------------------- Tunables (keep these simple) ---------------------
MIN_ARC_DEG = 300.0            # accept once cumulative heading reaches this
HEADING_SMOOTH_S = 3.0         # small amount of smoothing (seconds)
MAX_LEN_SAMPLES = 1000         # safety cap on accumulation window

# Spatio-temporal clustering (allow drift)
EPS_M = 1500.0                 # meters, spatial radius for connecting circles
MAX_GAP_S = 900.0              # seconds, temporal bridge between nearby circles
MIN_CLUSTER_COUNT = 1          # keep even singletons (you can raise to 2+ later)

# Second-pass “valley merge” between clusters (optional, mirrors altitude logic idea)
DO_VALLEY_MERGE = True
MERGE_EPS_M = 700.0            # meters, center-to-center distance to merge clusters
MERGE_GAP_S = 600.0            # seconds, max gap between clusters to merge

# Plot options
DRAW_CLUSTER_RINGS = True      # draw a ring at mean radius per cluster

# --------------------- Utilities ---------------------
R_EARTH_M = 6371000.0

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()

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
    if dt <= 0 or not (dt == dt): dt = 1.0
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
    brng = (math.degrees(math.atan2(x, y)) + 360.0) % 360.0
    return brng

def circ_diff_deg(a, b):
    """signed smallest difference b-a in (-180,180]"""
    return (b - a + 540.0) % 360.0 - 180.0

# --------------------- Circle detection ---------------------
def detect_circles(df: pd.DataFrame,
                   min_circle_heading_deg: float = MIN_ARC_DEG,
                   max_len_samples: int = MAX_LEN_SAMPLES,
                   smooth_s: float = HEADING_SMOOTH_S):
    """
    Returns: list[(s_idx, e_idx, arc_deg)], cadence_sec_per_fix
    """
    df = ensure_time_sec(df)
    dt = sec_per_fix(df)

    # Build heading series (light smoothing)
    headings = [0.0]
    for i in range(1, len(df)):
        headings.append(bearing_deg(df.loc[i-1,'lat'], df.loc[i-1,'lon'],
                                    df.loc[i,'lat'], df.loc[i,'lon']))
    hdg = pd.Series(headings)
    hdg_s = smooth_series(hdg, smooth_s, dt).to_numpy()

    circles = []
    n = len(df)
    i = 0
    while i < n-2:
        cum = 0.0
        j = i + 1
        last_h = hdg_s[i]
        while j < n and (j - i) < max_len_samples:
            now_h = hdg_s[j]
            d = circ_diff_deg(last_h, now_h)
            cum += d
            last_h = now_h
            if abs(cum) >= min_circle_heading_deg:
                circles.append((i, j, cum))
                i = j + 1
                break
            j += 1
        else:
            i += 1
    return circles, dt

# --------------------- Circles → clusters (graph connectivity) ---------------------
def circles_to_dataframe(df: pd.DataFrame, circles):
    rows = []
    for k, (s,e,arc) in enumerate(circles, start=1):
        mid = (s + e) // 2
        dur = float(df['sec'].iloc[e] - df['sec'].iloc[s])
        dist_m = haversine_m(df['lat'].iloc[s], df['lon'].iloc[s],
                             df['lat'].iloc[e], df['lon'].iloc[e])
        R = max(dist_m / max(np.radians(abs(arc)), 1e-6), 1.0)  # crude radius
        rows.append({
            "circle_id": k,
            "start_idx": int(s),
            "end_idx": int(e),
            "start_time": df["time"].iloc[s],
            "end_time": df["time"].iloc[e],
            "duration_s": dur,
            "arc_deg": float(abs(arc)),
            "turn_rate_deg_s": float(arc / max(dur, 1e-6)),
            "radius_m_est": float(R),
            "lat_mid": float(df["lat"].iloc[mid]),
            "lon_mid": float(df["lon"].iloc[mid]),
        })
    cols = ["circle_id","start_idx","end_idx","start_time","end_time","duration_s",
            "arc_deg","turn_rate_deg_s","radius_m_est","lat_mid","lon_mid"]
    return pd.DataFrame(rows, columns=cols)

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

    # adjacency under (spatial within eps) AND (temporal overlap or small gap)
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            spatial_ok = dist[i, j] <= eps_m
            gap_ij = max(0.0, (pd.to_datetime(t0[j]) - pd.to_datetime(t1[i])).total_seconds())
            gap_ji = max(0.0, (pd.to_datetime(t0[i]) - pd.to_datetime(t1[j])).total_seconds())
            temporal_ok = (gap_ij <= max_gap_s) or (gap_ji <= max_gap_s) or \
                          (pd.to_datetime(t1[i]) >= pd.to_datetime(t0[j])) or \
                          (pd.to_datetime(t1[j]) >= pd.to_datetime(t0[i]))
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
        # weighted center (tighter radius weighs more)
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

# --------------------- Valley merge (cluster → cluster) ---------------------
def valley_merge(clusters_df: pd.DataFrame,
                 merge_eps_m: float = MERGE_EPS_M,
                 merge_gap_s: float = MERGE_GAP_S) -> pd.DataFrame:
    """
    Merge adjacent clusters whose centers are close and whose time gaps are small.
    This approximates “one big thermal” split into several short circling bursts.
    """
    if clusters_df.empty or len(clusters_df) == 1:
        return clusters_df.copy()

    df = clusters_df.sort_values("start_time").reset_index(drop=True)
    parent = list(range(len(df)))  # disjoint set

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra

    for i in range(len(df)-1):
        a = df.iloc[i]; b = df.iloc[i+1]
        gap_s = (b["start_time"] - a["end_time"]).total_seconds()
        d_m = haversine_m(a["center_lat"], a["center_lon"], b["center_lat"], b["center_lon"])
        if gap_s <= merge_gap_s and d_m <= merge_eps_m:
            union(i, i+1)

    # collect components
    comp_map = {}
    for i in range(len(df)):
        r = find(i)
        comp_map.setdefault(r, []).append(i)

    merged_rows = []
    new_id = 1
    for root, members in comp_map.items():
        idx = np.array(members, int)
        lat_c = float(np.mean(df.loc[idx, "center_lat"]))
        lon_c = float(np.mean(df.loc[idx, "center_lon"]))
        mean_r = float(np.mean(df.loc[idx, "mean_radius_m"]))
        start_time = df.loc[idx, "start_time"].min()
        end_time   = df.loc[idx, "end_time"].max()
        n_total = int(df.loc[idx, "n"].sum())
        merged_rows.append({
            "cluster_id": new_id,
            "n": n_total,
            "start_time": start_time,
            "end_time": end_time,
            "mean_radius_m": mean_r,
            "center_lat": lat_c,
            "center_lon": lon_c,
        })
        new_id += 1

    return pd.DataFrame(merged_rows, columns=list(df.columns))

# --------------------- Pairs export ---------------------
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

# --------------------- Plot ---------------------
def params_box_text(dt):
    return (
        f"Cadence: {dt:.2f} s/fix\\n"
        f"Circle: cum arc≥{int(MIN_ARC_DEG)}°, smooth={int(HEADING_SMOOTH_S)} s, max_win≤{MAX_LEN_SAMPLES} samples\\n"
        f"Cluster: eps={int(EPS_M)} m, gap≤{int(MAX_GAP_S/60)} min | Valley-merge: eps={int(MERGE_EPS_M)} m, gap≤{int(MERGE_GAP_S/60)} min\\n"
        f"Keep clusters with ≥{MIN_CLUSTER_COUNT} circles"
    )

def add_params_box(ax, dt):
    ax.text(0.99, 0.01, params_box_text(dt), transform=ax.transAxes, va='bottom', ha='right',
            fontsize=9, family='monospace',
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, lw=0.5))

def plot_map(df, circles, clusters_df, dt):
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    ax.plot(df["lon"], df["lat"], color="green", lw=1.2, alpha=0.7, label="Track")

    # Circles as green crosses
    mids_lon, mids_lat = [], []
    for (s,e,_) in circles:
        mid = (s + e)//2
        mids_lon.append(float(df["lon"].iloc[mid]))
        mids_lat.append(float(df["lat"].iloc[mid]))
    if mids_lon:
        ax.scatter(mids_lon, mids_lat, marker='x', c='green', s=36, linewidths=1.5, zorder=5, label='Circles')

    # Cluster centers + optional radius rings
    big = clusters_df[clusters_df["n"] >= MIN_CLUSTER_COUNT]
    for _, row in big.iterrows():
        ax.scatter(row["center_lon"], row["center_lat"], marker='x', c='k', s=100, linewidths=1.5, zorder=6)
        ax.annotate(f"T{int(row['cluster_id'])}", (row["center_lon"], row["center_lat"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=10, weight='bold')
        if DRAW_CLUSTER_RINGS and row["mean_radius_m"] > 0 and np.isfinite(row["mean_radius_m"]):
            # draw a small ring approximating mean radius (rough lon/lat meters conversion near center)
            lat_c, lon_c = row["center_lat"], row["center_lon"]
            r = row["mean_radius_m"]
            # approximate radius in degrees (very small circles, so small-angle ok)
            dlat = (r / R_EARTH_M) * (180.0/np.pi)
            dlon = dlat / max(np.cos(np.radians(lat_c)), 1e-6)
            th = np.linspace(0, 2*np.pi, 100)
            xs = lon_c + dlon*np.cos(th)
            ys = lat_c + dlat*np.sin(th)
            ax.plot(xs, ys, color='k', alpha=0.35, lw=0.8)

    add_params_box(ax, dt)
    ax.set_title("Circles-only → Thermals (Track=green, Circles=green ✕, Centers=black ✕)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.legend(loc='upper left'); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

# --------------------- Main ---------------------
def main():
    ts = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    logf = open(f"circles_core_{ts}.txt", "w")
    sys.stdout = sys.stderr = Tee(sys.stdout, logf)

    igc_path = "2020-11-08 Lumpy Paterson 108645.igc_subset"
    print(f"[circles-core] Parsing IGC: {igc_path}")
    df = parse_igc(igc_path)
    if df.empty:
        print("[circles-core] No rows parsed; aborting.")
        return
    df = compute_derived(df)
    df = ensure_time_sec(df)

    try:
        tow = detect_tow_segment(df)
    except Exception as e:
        print(f"[circles-core] Tow detect error: {e}")
        tow = None
    if isinstance(tow, tuple) and len(tow) == 2:
        s,e = tow
        print(f"[circles-core] Tow detected: {s}→{e}. Excluding tow.")
        df = df.iloc[e+1:].reset_index(drop=True)
        df = ensure_time_sec(df)
    else:
        print("[circles-core] Tow not clearly detected; using full trace.")
    print(f"[circles-core] Points after tow trim: {len(df)}")

    circles, dt = detect_circles(df, MIN_ARC_DEG, MAX_LEN_SAMPLES, HEADING_SMOOTH_S)
    print(f"[circles-core] Cadence ≈ {dt:.2f} s/fix | circles detected: {len(circles)}")

    circ_df = circles_to_dataframe(df, circles)
    circ_with_cluster, clusters_df = clusters_from_circles(circ_df, EPS_M, MAX_GAP_S)

    if DO_VALLEY_MERGE:
        merged = valley_merge(clusters_df, MERGE_EPS_M, MERGE_GAP_S)
        print(f"[circles-core] Valley-merge reduced clusters: {len(clusters_df)} → {len(merged)}")
        clusters_df = merged

    pairs_df = successive_cluster_pairs(clusters_df)

    out = Path("outputs"); out.mkdir(parents=True, exist_ok=True)
    f1 = out / f"circles_turns_{ts}.csv"
    circ_with_cluster.to_csv(f1, index=False)
    f2 = out / f"circles_clusters_{ts}.csv"
    clusters_df.to_csv(f2, index=False)
    f3 = out / f"circles_cluster_pairs_{ts}.csv"
    pairs_df.to_csv(f3, index=False)
    print(f"[circles-core] Wrote: {f1.name} ({len(circ_with_cluster)} rows), {f2.name} ({len(clusters_df)} rows), {f3.name} ({len(pairs_df)} rows) in ./outputs")

    plot_map(df, circles, clusters_df, dt)

if __name__ == "__main__":
    main()