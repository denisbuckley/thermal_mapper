
"""
altitude_gain_v3e.py
=================================
Altitude-only detector with **spatio‑temporal + altitude‑continuity clustering**
so groups of climbs inside the same altitude “mountain” merge into one thermal.

Adds to v3d:
- Clustering edge between climbs i and j if ALL are true:
    * Spatial distance between midpoints <= EPS_M (default 2000 m)
    * Time windows overlap OR the gap between them <= MAX_GAP_S (default 1200 s)
    * **Altitude continuity**: the valley drop between i.end and j.start is small:
         drop <= ALT_DROP_M  OR  drop <= ALT_DROP_FRAC * gain_i
      (defaults: 180 m or 0.40×gain of the earlier climb)
- Outputs:
    * outputs/altitude_climbs_<ts>.csv  (includes cluster_id)
    * outputs/altitude_clusters_<ts>.csv
- Plot:
    * Altitude vs time (as before) + cluster labels “T#” at each cluster’s time midpoint.
    * Bird’s‑eye map with cluster centers (black ✕, T#).

Tune at the top:
    EPS_M = 2000.0      # meters
    MAX_GAP_S = 1200.0  # seconds
    ALT_DROP_M = 180.0  # meters
    ALT_DROP_FRAC = 0.40
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
    "MIN_CLIMB_S","MIN_GAIN_M","SMOOTH_RADIUS_S",
    "MAX_GAP_S","ALT_DROP_M","ALT_DROP_FRAC",
    "A_EPS_M","A_MIN_SAMPLES"
})



# -------------------- parameters --------------------
EPS_M = 2000.0
MAX_GAP_S = 1200.0
ALT_DROP_M = 180.0
ALT_DROP_FRAC = 0.40

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

def vario_from_alt(time_s: np.ndarray, alt_m: np.ndarray) -> np.ndarray:
    if len(time_s) < 3:
        return np.zeros_like(alt_m, dtype=float)
    return np.gradient(alt_m.astype(float), time_s.astype(float))

def close_small_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    if max_gap <= 0 or mask.size == 0:
        return mask.copy()
    m = mask.copy()
    n = len(m)
    i = 0
    while i < n:
        if not m[i]:
            j = i
            while j < n and not m[j]:
                j += 1
            left = (i - 1 >= 0) and m[i - 1]
            right = (j < n) and m[j] if j < n else False
            gap_len = j - i
            if left and right and gap_len <= max_gap:
                m[i:j] = True
            i = j
        else:
            i += 1
    return m

# -------------------- detection --------------------
def detect_climb_segments(df: pd.DataFrame,
                          min_avg_mps: float = 0.5,
                          min_duration_s: float = 60.0,
                          min_gain_m: float = 60.0,
                          smooth_alt_seconds: float = 9.0,
                          allow_gap_seconds: float = 6.0):
    """
    Return (segments, vario, alt_s, alt_col, dt). Each segment is (s, e, gain_m, dur_s, mean_mps).
    """
    df = ensure_time_sec(df)
    dt = sec_per_fix(df)
    alt_col = None
    for candidate in ["alt_baro", "alt", "altitude", "gps_alt"]:
        if candidate in df.columns:
            alt_col = candidate
            break
    if alt_col is None:
        raise ValueError("No altitude column found.")

    alt = df[alt_col].astype(float)
    alt_s = smooth_series(alt, smooth_alt_seconds, dt)
    vario = vario_from_alt(df['sec'].to_numpy(), alt_s.to_numpy())

    mask = vario >= float(min_avg_mps)
    max_gap_samples = int(round(allow_gap_seconds / max(dt, 1e-6)))
    if isinstance(mask, pd.Series): mask = mask.to_numpy()
    mask_filled = close_small_gaps(mask, max_gap_samples)

    segments = []
    n = len(df)
    i = 0
    while i < n:
        if not mask_filled[i]:
            i += 1
            continue
        j = i
        while j < n and mask_filled[j]:
            j += 1
        dur_s = df['sec'].iloc[j-1] - df['sec'].iloc[i]
        gain_m = alt_s.iloc[j-1] - alt_s.iloc[i]
        mean_mps = gain_m / max(dur_s, 1e-6) if dur_s > 0 else 0.0
        if dur_s >= min_duration_s and gain_m >= min_gain_m and mean_mps >= min_avg_mps:
            segments.append((i, j-1, float(gain_m), float(dur_s), float(mean_mps)))
        i = j
    return segments, vario, alt_s, alt_col, dt

def segments_to_dataframe(df: pd.DataFrame, segments):
    rows = []
    for k, (s, e, gain, dur, mean_mps) in enumerate(segments, start=1):
        mid = (s + e) // 2
        rows.append({
            "climb_id": k,
            "start_idx": int(s),
            "end_idx": int(e),
            "start_time": df["time"].iloc[s],
            "end_time": df["time"].iloc[e],
            "duration_s": float(dur),
            "gain_m": float(gain),
            "mean_mps": float(mean_mps),
            "lat_mid": float(df["lat"].iloc[mid]),
            "lon_mid": float(df["lon"].iloc[mid]),
        })
    return pd.DataFrame(rows, columns=[
        "climb_id","start_idx","end_idx","start_time","end_time",
        "duration_s","gain_m","mean_mps","lat_mid","lon_mid"
    ])

# -------------------- spatio‑temporal + altitude continuity clustering --------------------
def clusters_from_climbs(df_all: pd.DataFrame, alt_s: pd.Series,
                         eps_m: float = EPS_M, max_gap_s: float = MAX_GAP_S,
                         alt_drop_m: float = ALT_DROP_M, alt_drop_frac: float = ALT_DROP_FRAC):
    """
    Connected-components clustering using three edges tests between climb i and j:
      1) spatially close: haversine(mid_i, mid_j) <= eps_m
      2) temporally close: windows overlap OR gap <= max_gap_s
      3) altitude-continuous: valley drop between i.end_idx and j.start_idx is small
         drop <= alt_drop_m OR drop <= alt_drop_frac * gain_i
    Returns (climbs_with_cluster, clusters_df)
    """
    if df_all.empty:
        return df_all.assign(cluster_id=[]), pd.DataFrame(columns=[
            "cluster_id","n","start_time","end_time","total_gain_m",
            "total_duration_s","mean_mps","center_lat","center_lon"
        ])

    df = df_all.reset_index(drop=True)
    n = len(df)
    lats = df["lat_mid"].to_numpy(float)
    lons = df["lon_mid"].to_numpy(float)
    s_idx = df["start_idx"].to_numpy(int)
    e_idx = df["end_idx"].to_numpy(int)
    t0 = pd.to_datetime(df["start_time"]).to_numpy()
    t1 = pd.to_datetime(df["end_time"]).to_numpy()
    gains = df["gain_m"].to_numpy(float)
    durs = df["duration_s"].to_numpy(float)

    # Precompute distances
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            d = float(haversine_m(lats[i], lons[i], lats[j], lons[j]))
            dist[i, j] = dist[j, i] = d

    # adjacency list with three tests
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            spatial_ok = dist[i, j] <= eps_m
            # time proximity
            gap_ij = max(0.0, (pd.to_datetime(t0[j]) - pd.to_datetime(t1[i])).total_seconds())
            gap_ji = max(0.0, (pd.to_datetime(t0[i]) - pd.to_datetime(t1[j])).total_seconds())
            temporal_ok = (gap_ij <= max_gap_s) or (gap_ji <= max_gap_s) or (pd.to_datetime(t1[i]) >= pd.to_datetime(t0[j])) or (pd.to_datetime(t1[j]) >= pd.to_datetime(t0[i]))
            # altitude continuity
            lo = min(e_idx[i], s_idx[j])
            hi = max(e_idx[i], s_idx[j])
            if hi <= lo+1:
                valley = float(alt_s.iloc[lo])
            else:
                valley = float(np.min(alt_s.iloc[lo:hi+1].to_numpy()))
            end_alt_i = float(alt_s.iloc[e_idx[i]])
            drop = end_alt_i - valley  # positive if we descended after i
            alt_ok = (drop <= alt_drop_m) or (drop <= alt_drop_frac * max(gains[i], 1.0))

            if spatial_ok and temporal_ok and alt_ok:
                adj[i].append(j); adj[j].append(i)

    # connected components
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
        # Gain-weighted center
        w = np.clip(gains[idx], 1.0, None)
        lat_c = float(np.average(lats[idx], weights=w))
        lon_c = float(np.average(lons[idx], weights=w))
        total_gain = float(gains[idx].sum())
        total_dur = float(durs[idx].sum())
        mean_mps = total_gain / total_dur if total_dur > 0 else 0.0
        start_time = pd.to_datetime(t0[idx].min()).to_pydatetime()
        end_time   = pd.to_datetime(t1[idx].max()).to_pydatetime()
        clusters.append({
            "cluster_id": cid+1,
            "n": int(len(idx)),
            "start_time": start_time,
            "end_time": end_time,
            "total_gain_m": total_gain,
            "total_duration_s": total_dur,
            "mean_mps": mean_mps,
            "center_lat": lat_c,
            "center_lon": lon_c,
        })
        comp_id[idx] = cid+1
        cid += 1

    clusters_df = pd.DataFrame(clusters, columns=[
        "cluster_id","n","start_time","end_time","total_gain_m",
        "total_duration_s","mean_mps","center_lat","center_lon"
    ])
    df_with_cluster = df.copy()
    df_with_cluster["cluster_id"] = comp_id
    return df_with_cluster, clusters_df

# -------------------- plotting --------------------
def plot_altitude_with_climbs_and_labels(df: pd.DataFrame, segments, alt_s: pd.Series, alt_col: str, clusters_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    t = df['time']
    ax.plot(t, df[alt_col], lw=0.8, alpha=0.35, label=f"{alt_col} (raw)")
    ax.plot(t, alt_s, lw=1.6, label=f"{alt_col} (smoothed)")
    for (s,e,gain,dur,mean_mps) in segments:
        ax.axvspan(t.iloc[s], t.iloc[e], color='orange', alpha=0.22)
        ax.text(t.iloc[s], alt_s.iloc[s], f"+{int(round(gain))} m\n{int(round(dur))} s",
                fontsize=8, va='bottom', ha='left')

    # Cluster labels at time midpoint
    for _, row in clusters_df.iterrows():
        tm = row["start_time"] + (row["end_time"] - row["start_time"]) / 2
        # find nearest index in df time
        idx = int(np.argmin(np.abs((t - tm).values.astype('timedelta64[s]').astype(np.int64))))
        ax.annotate(f"T{int(row['cluster_id'])}",
                    (t.iloc[idx], alt_s.iloc[idx]),
                    textcoords="offset points", xytext=(6, 8),
                    fontsize=10, weight='bold')

    ax.set_title("Altitude vs Time — Sustained Climb Segments (Thermal labels)")
    ax.set_xlabel("Time"); ax.set_ylabel("Altitude (m)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_map_with_clusters(df: pd.DataFrame, segments, clusters_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(df["lon"], df["lat"], color="green", lw=1.2, alpha=0.7, label="Track")
    for (s,e,_,_,_) in segments:
        ax.plot(df.loc[s:e, "lon"], df.loc[s:e, "lat"], color="orange", lw=2.0, alpha=0.6)
    for _, row in clusters_df.iterrows():
        ax.scatter(row["center_lon"], row["center_lat"], marker='x', s=110, c='k', zorder=6)
        ax.annotate(f"T{int(row['cluster_id'])}",
                    (row["center_lon"], row["center_lat"]),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=10, weight='bold')
    ax.set_title("Bird’s‑eye — Climb clusters (Track=green, Climbs=orange, Centers=black X)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.legend(loc='upper left'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# -------------------- main --------------------
def main():
    ts = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    logf = open(f"altitude_gain_{ts}.txt", "w")
    sys.stdout = sys.stderr = Tee(sys.stdout, logf)

    path = "2020-11-08 Lumpy Paterson 108645.igc_subset"  # hard-coded for PyCharm
    print(f"[alt v3e] Parsing IGC: {path}")
    df = parse_igc(path)
    if len(df) == 0:
        print("[alt v3e] No rows parsed; aborting."); return

    df = compute_derived(df)
    df = ensure_time_sec(df)

    try:
        tow = detect_tow_segment(df)
    except Exception as e:
        print(f"[alt v3e] Tow detect error: {e}")
        tow = None
    if tow and isinstance(tow, tuple) and len(tow) == 2:
        s,e = tow
        print(f"[alt v3e] Tow detected: {s}→{e}. Excluding tow.")
        df = df.iloc[e+1:].reset_index(drop=True)
        df = ensure_time_sec(df)
    else:
        print("[alt v3e] Tow not clearly detected; using full trace.")
    print(f"[alt v3e] Points after tow trim: {len(df)}")

    # Detect climbs
    segments, vario, alt_s, alt_col, dt = detect_climb_segments(df)
    print(f"[alt v3e] Cadence ≈ {dt:.2f} s/fix | climb segments: {len(segments)}")

    # DataFrame of climbs
    climbs_df = segments_to_dataframe(df, segments)
    total_gain = climbs_df["gain_m"].sum() if not climbs_df.empty else 0.0
    print(f"[alt v3e] Total climb across segments: +{total_gain:.0f} m")

    # Spatio-temporal + altitude continuity clustering
    climbs_with_cluster, clusters_df = clusters_from_climbs(
        climbs_df, alt_s,
        eps_m=EPS_M, max_gap_s=MAX_GAP_S,
        alt_drop_m=ALT_DROP_M, alt_drop_frac=ALT_DROP_FRAC
    )

    # CSVs
    out_dir = Path("outputs"); out_dir.mkdir(parents=True, exist_ok=True)
    csv_climbs = out_dir / f"altitude_climbs_{ts}.csv"
    climbs_with_cluster.to_csv(csv_climbs, index=False)
    print(f"[alt v3e] Wrote climbs CSV (with cluster_id): {csv_climbs} ({len(climbs_with_cluster)} rows)")

    csv_clusters = out_dir / f"altitude_clusters_{ts}.csv"
    clusters_df.to_csv(csv_clusters, index=False)
    print(f"[alt v3e] Wrote clusters CSV: {csv_clusters} ({len(clusters_df)} rows)")

    # Logs
    if not clusters_df.empty:
        print("\n[alt v3e] Cluster summary:")
        for _, r in clusters_df.iterrows():
            print(f"  T{int(r['cluster_id'])}: n={int(r['n'])}  total_gain=+{r['total_gain_m']:.0f} m  "
                  f"dur={r['total_duration_s']:.0f} s  mean={r['mean_mps']:.2f} m/s  "
                  f"center=({r['center_lat']:.5f},{r['center_lon']:.5f})")

    # Plots
    if len(df):
        plot_altitude_with_climbs_and_labels(df, segments, alt_s, alt_col, clusters_df)
        plot_map_with_clusters(df, segments, clusters_df)

    return climbs_with_cluster, clusters_df

if __name__ == "__main__":
    main()