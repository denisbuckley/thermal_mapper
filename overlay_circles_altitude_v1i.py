
"""
overlay_circles_altitude_v1i.py
--------------------------------
Purpose
    Overlay *circle-based* thermalling detections and *altitude-based* thermal detections
    on a single map for quick visual QA — now with cluster-size-aware filtering and sizing.

What this script does
    1) Parses IGC and computes derived fields.
    2) Removes tow segment (if detected).
    3) Detects *circles* (multi-emission, full-turn logic) and clusters them (space + time).
    4) Detects *altitude-based* thermals (rising segments merged across shallow valleys).
    5) Computes cluster size `n` and applies a configurable **MIN_CLUSTER_SIZE** filter.
    6) Plots:
        - Track (green line)
        - Raw circle midpoints (green ×)
        - **Circle clusters (black ×) sized by `n`**; sub-threshold clusters shown faint grey ×
        - Altitude clusters (orange ×) with T# labels
    7) Writes a concise debug file to outputs/overlay_debug_<yymmddhhmmss>.txt

Inputs
    - IGC file path set in main() (`igc_file`). Requires igc_utils.py:
      - parse_igc, compute_derived, detect_tow_segment

Key Tuning (edit below)
    # Circles
    C_MIN_ARC_DEG, C_SMOOTH_S, C_MAX_WIN_SAMPLES, C_MIN_DIR_RATIO
    CLUSTER_EPS_M, CLUSTER_GAP_S

    # Cluster filtering (NEW)
    MIN_CLUSTER_SIZE  -> keep clusters with at least this many circles

    # Altitude
    A_SMOOTH_S, A_MIN_RATE_MPS, A_MIN_DUR_S, A_MIN_GAIN_M, A_ALT_DROP_M, A_ALT_DROP_FRAC

Outputs
    - Debug text: outputs/overlay_debug_<timestamp>.txt
    - Plot window with overlays
"""

from pathlib import Path
from typing import Tuple, List
from datetime import datetime
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from igc_utils import parse_igc, compute_derived, detect_tow_segment

# -------------------- Tuning: CIRCLES --------------------
C_MIN_ARC_DEG      = 300.0     # Full-turn emission threshold (deg)
C_SMOOTH_S         = 3.0       # Heading smoothing window (seconds)
C_MAX_WIN_SAMPLES  = 2400      # Per-run cap to avoid runaway accumulation
C_MIN_DIR_RATIO    = 0.65      # Dominance of turning direction (>= this)
CLUSTER_EPS_M      = 1500.0    # Circle-cluster spatial radius (m)
CLUSTER_GAP_S      = 900.0     # Allowable non-overlap time gap (s)

# -------------------- NEW: cluster-size filter ------------
MIN_CLUSTER_SIZE   = 5         # Keep clusters with at least this many circles
# ---------------------------------------------------------

# -------------------- Tuning: ALTITUDE ------------------
A_SMOOTH_S         = 9.0       # Altitude smoothing (seconds)
A_MIN_RATE_MPS     = 0.5       # Start climb when vertical rate >= this (m/s)
A_MIN_DUR_S        = 60.0      # Min duration for a climb (s)
A_MIN_GAIN_M       = 60.0      # Min net altitude gain (m)
A_ALT_DROP_M       = 180.0     # Merge valley if absolute drop below this (m)
A_ALT_DROP_FRAC    = 0.30      # Merge valley if fractional drop below this
# --------------------------------------------------------

R_EARTH_M = 6371000.0

def ts_stamp() -> str:
    return datetime.now().strftime("%y%m%d%H%M%S")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1; dlon = lon2 - lon1
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

def circ_diff_deg(a, b):
    return (b - a + 540.0) % 360.0 - 180.0

def detect_circles_multi(df: pd.DataFrame,
                         min_arc_deg=C_MIN_ARC_DEG,
                         max_len_samples=C_MAX_WIN_SAMPLES,
                         smooth_s=C_SMOOTH_S,
                         min_dir_ratio=C_MIN_DIR_RATIO):
    df = ensure_time_sec(df)
    dt = sec_per_fix(df)

    headings = [0.0]
    for i in range(1, len(df)):
        headings.append(bearing_deg(df.loc[i-1,'lat'], df.loc[i-1,'lon'],
                                    df.loc[i,'lat'], df.loc[i,'lon']))
    hdg_s = smooth_series(pd.Series(headings), smooth_s, dt).to_numpy(dtype=float)

    circles = []
    n = len(df)
    i = 0
    while i < n-3:
        j = i + 1
        last_h = hdg_s[i]
        cum_abs = 0.0
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

            while cum_abs >= min_arc_deg:
                tot = pos_abs + neg_abs if (pos_abs + neg_abs) > 0 else 1.0
                ratio = max(pos_abs, neg_abs) / tot
                if ratio >= min_dir_ratio:
                    s_idx = last_emit_j; e_idx = j
                    if e_idx - s_idx >= 2:
                        sign = 1 if pos_abs >= neg_abs else -1
                        circles.append((s_idx, e_idx, sign * min_arc_deg))
                        last_emit_j = j
                if pos_abs >= neg_abs:
                    pos_abs = max(0.0, pos_abs - min_arc_deg)
                else:
                    neg_abs = max(0.0, neg_abs - min_arc_deg)
                cum_abs = max(0.0, cum_abs - min_arc_deg)
            j += 1

        i = max(i + 5, last_emit_j)

    return circles, dt

def circles_to_df(df: pd.DataFrame, circles: List[tuple]) -> pd.DataFrame:
    rows = []
    for k, (s, e, arc) in enumerate(circles, start=1):
        mid = (s+e)//2
        dur = float(df['sec'].iloc[e] - df['sec'].iloc[s])
        chord = haversine_m(df['lat'].iloc[s], df['lon'].iloc[s],
                            df['lat'].iloc[e], df['lon'].iloc[e])
        radius_est = max(chord / max(np.radians(abs(arc)), 1e-6), 1.0)
        rows.append(dict(circle_id=k, start_idx=int(s), end_idx=int(e),
                         start_time=df['time'].iloc[s], end_time=df['time'].iloc[e],
                         duration_s=dur, arc_deg=float(abs(arc)),
                         turn_rate_deg_s=float(arc/max(dur,1e-6)),
                         radius_m_est=float(radius_est),
                         lat_mid=float(df['lat'].iloc[mid]),
                         lon_mid=float(df['lon'].iloc[mid])))
    cols = ["circle_id","start_idx","end_idx","start_time","end_time","duration_s","arc_deg",
            "turn_rate_deg_s","radius_m_est","lat_mid","lon_mid"]
    return pd.DataFrame(rows, columns=cols)

def circle_clusters(circ_df: pd.DataFrame,
                    eps_m=CLUSTER_EPS_M, gap_s=CLUSTER_GAP_S):
    if circ_df.empty:
        circ_with_cluster = circ_df.assign(cluster_id=[])
        clusters_df = pd.DataFrame(columns=[
            "cluster_id","n","start_time","end_time","mean_radius_m","center_lat","center_lon"
        ])
        return circ_with_cluster, clusters_df, clusters_df  # last is filtered
    df = circ_df.reset_index(drop=True)
    n = len(df)
    lat = df["lat_mid"].to_numpy(dtype=float); lon = df["lon_mid"].to_numpy(dtype=float)
    t0 = pd.to_datetime(df["start_time"]).to_numpy(dtype='datetime64[ns]')
    t1 = pd.to_datetime(df["end_time"]).to_numpy(dtype='datetime64[ns]')
    radius = df["radius_m_est"].to_numpy(dtype=float)

    D = np.zeros((n, n), dtype=float)
    for a in range(n):
        for b in range(a+1, n):
            d = haversine_m(lat[a], lon[a], lat[b], lon[b])
            D[a,b] = D[b,a] = d

    adj = [[] for _ in range(n)]
    for a in range(n):
        for b in range(a+1, n):
            if D[a,b] <= eps_m:
                gap_ab = float(max(0, (t0[b] - t1[a]) / np.timedelta64(1, 's')))
                gap_ba = float(max(0, (t0[a] - t1[b]) / np.timedelta64(1, 's')))
                temporal_ok = (gap_ab <= gap_s) or (gap_ba <= gap_s) or (t1[a] >= t0[b]) or (t1[b] >= t0[a])
                if temporal_ok:
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
                if not visited[b]: visited[b]=True; q.append(b)
        idx = np.array(members, int)
        w = np.clip(1.0/np.maximum(radius[idx],1.0), 0.1, None)
        lat_c = float(np.average(lat[idx], weights=w))
        lon_c = float(np.average(lon[idx], weights=w))
        clusters.append(dict(cluster_id=cid+1,
                             n=int(len(idx)),
                             start_time=pd.to_datetime(t0[idx].min()).to_pydatetime(),
                             end_time=pd.to_datetime(t1[idx].max()).to_pydatetime(),
                             mean_radius_m=float(np.mean(radius[idx])),
                             center_lat=lat_c, center_lon=lon_c))
        comp_id[idx] = cid+1; cid += 1

    clusters_df = pd.DataFrame(clusters, columns=["cluster_id","n","start_time","end_time",
                                                  "mean_radius_m","center_lat","center_lon"])
    circ_with_cluster = df.copy(); circ_with_cluster["cluster_id"] = comp_id

    # filtered view
    clusters_filt = clusters_df[clusters_df["n"] >= MIN_CLUSTER_SIZE].reset_index(drop=True)
    return circ_with_cluster, clusters_df, clusters_filt

def detect_climbs_altitude(df: pd.DataFrame):
    df = ensure_time_sec(df)
    dt = sec_per_fix(df)
    alt = smooth_series(df["gps_alt"], A_SMOOTH_S, dt)
    dalt = alt.diff().fillna(0.0)
    rate = dalt / df["sec"].diff().fillna(dt)

    in_up = False
    start_idx = None
    segs = []
    for i in range(1, len(df)):
        if not in_up and rate.iloc[i] >= A_MIN_RATE_MPS:
            in_up = True; start_idx = i-1
        if in_up and rate.iloc[i] < 0:
            s = start_idx; e = i
            if s is not None and e > s:
                dur = float(df["sec"].iloc[e] - df["sec"].iloc[s])
                gain = float(alt.iloc[e] - alt.iloc[s])
                if dur >= A_MIN_DUR_S and gain >= A_MIN_GAIN_M:
                    segs.append((s,e,gain,dur))
            in_up = False; start_idx=None
    if in_up and start_idx is not None:
        s = start_idx; e = len(df)-1
        dur = float(df["sec"].iloc[e] - df["sec"].iloc[s])
        gain = float(alt.iloc[e] - alt.iloc[s])
        if dur >= A_MIN_DUR_S and gain >= A_MIN_GAIN_M:
            segs.append((s,e,gain,dur))

    # Merge across shallow valleys
    merged = []
    for s,e,gain,dur in segs:
        if not merged:
            merged.append([s,e,gain,dur]); continue
        ps,pe,pg,pd_ = merged[-1]
        valley_drop = float(alt.iloc[pe] - alt.iloc[s]) if s>pe else 0.0
        prev_peak = alt.iloc[pe]
        base = alt.iloc[ps]
        frac_drop = ((prev_peak - alt.iloc[s]) / max(prev_peak - base, 1.0)) if prev_peak>base else 1.0
        if (valley_drop < A_ALT_DROP_M) or (frac_drop < A_ALT_DROP_FRAC):
            merged[-1][1] = e
            merged[-1][2] = float(alt.iloc[merged[-1][1]] - alt.iloc[merged[-1][0]])
            merged[-1][3] = float(df["sec"].iloc[merged[-1][1]] - df["sec"].iloc[merged[-1][0]])
        else:
            merged.append([s,e,gain,dur])

    rows = []
    for k,(s,e,gain,dur) in enumerate(merged, start=1):
        mid = (s+e)//2
        rows.append(dict(thermal_id=k, start_idx=int(s), end_idx=int(e),
                         start_time=df["time"].iloc[s], end_time=df["time"].iloc[e],
                         duration_s=float(dur), gain_m=float(gain),
                         lat_mid=float(df["lat"].iloc[mid]), lon_mid=float(df["lon"].iloc[mid])))
    cols = ["thermal_id","start_idx","end_idx","start_time","end_time","duration_s",
            "gain_m","lat_mid","lon_mid"]
    return pd.DataFrame(rows, columns=cols), dt

def plot_overlay(df: pd.DataFrame,
                 circles, clusters_raw: pd.DataFrame,
                 clusters_filt: pd.DataFrame,
                 alt_clusters_df: pd.DataFrame, dt: float):
    print(f"[overlay] circles={len(circles)}, clusters_raw={len(clusters_raw)}, clusters_filt={len(clusters_filt)}, altitude_clusters={len(alt_clusters_df)}")

    fig, ax = plt.subplots(figsize=(10.8, 7.6))
    ax.plot(df["lon"], df["lat"], color="green", lw=1.2, alpha=0.7, label="Track")

    # Raw circle midpoints
    if circles:
        mids_lon = [float(df["lon"].iloc[(s+e)//2]) for (s,e,_) in circles]
        mids_lat = [float(df["lat"].iloc[(s+e)//2]) for (s,e,_) in circles]
        ax.scatter(mids_lon, mids_lat, marker='x', c='green', s=60, linewidths=1.8, zorder=5, alpha=0.8, label="Circles")

    # Raw clusters (sub-threshold) as faint grey
    if not clusters_raw.empty:
        small = clusters_raw[clusters_raw["n"] < MIN_CLUSTER_SIZE]
        if not small.empty:
            ax.scatter(small["center_lon"], small["center_lat"],
                       marker='x', c='lightgrey', s=40, linewidths=1.5, zorder=6, label=f"Circle clusters (n<{MIN_CLUSTER_SIZE})")

    # Filtered clusters sized by n
    if not clusters_filt.empty:
        sizes = (clusters_filt["n"].clip(lower=MIN_CLUSTER_SIZE) - MIN_CLUSTER_SIZE + 1) * 40.0
        sizes = sizes.clip(80, 400)
        ax.scatter(clusters_filt["center_lon"], clusters_filt["center_lat"],
                   marker='x', c='k', s=sizes, linewidths=2.2, zorder=7, label=f"Circle clusters (n≥{MIN_CLUSTER_SIZE})")

    # Altitude clusters
    if not alt_clusters_df.empty:
        ax.scatter(alt_clusters_df["lon_mid"], alt_clusters_df["lat_mid"],
                   marker='x', c='orange', s=110, linewidths=2.0, zorder=8, label="Altitude clusters (T#)")
        for _, r in alt_clusters_df.iterrows():
            ax.text(r["lon_mid"], r["lat_mid"], f"T{int(r['thermal_id'])}", fontsize=9,
                    ha='center', va='bottom', color='black', zorder=9)

    ax.set_title(f"Overlay: Circles (green ✕), Circle clusters (size∝n), Altitude clusters (orange ✕)\nFilter: n≥{MIN_CLUSTER_SIZE}")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.legend(loc='upper left'); ax.grid(True, alpha=0.3)

    txt = (f"Cadence: {dt:.2f} s/fix | Circles: {len(circles)} | "
           f"Clusters raw: {len(clusters_raw)} | Clusters (n≥{MIN_CLUSTER_SIZE}): {len(clusters_filt)} | Altitude clusters: {len(alt_clusters_df)}")
    ax.text(0.99, 0.01, txt, transform=ax.transAxes, va='bottom', ha='right',
            fontsize=9, family='monospace',
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, lw=0.6))

    plt.tight_layout(); plt.show()

def write_debug_file(clusters_raw: pd.DataFrame, clusters_filt: pd.DataFrame, dt: float, alt_clusters_df: pd.DataFrame, circles_count: int):
    out_dir = Path.cwd() / "outputs"; ensure_dir(out_dir)
    fn = out_dir / f"overlay_debug_{ts_stamp()}.txt"
    lines = [
        f"Cadence: {dt:.2f} s/fix",
        f"Circles count: {circles_count}",
        f"Circle clusters (raw): {len(clusters_raw)}",
        f"Circle clusters (n≥{MIN_CLUSTER_SIZE}): {len(clusters_filt)}",
        f"Altitude clusters: {len(alt_clusters_df)}",
        "",
        "[Top raw clusters by n]",
        clusters_raw.sort_values('n', ascending=False).head(12).to_string(index=False) if not clusters_raw.empty else "(none)",
        "",
        "[Filtered clusters (n≥{thr})]".format(thr=MIN_CLUSTER_SIZE),
        clusters_filt.sort_values('n', ascending=False).head(12).to_string(index=False) if not clusters_filt.empty else "(none)",
        ""
    ]
    fn.write_text("\n".join(lines), encoding="utf-8")
    print(f"[overlay] Wrote debug file: {fn.resolve()}")

def main():
    igc_file = "2020-11-08 Lumpy Paterson 108645.igc"  # <-- EDIT if needed
    print(f"[overlay v1i] Parsing IGC: {igc_file}")
    df = parse_igc(igc_file)
    df = compute_derived(df)
    df = ensure_time_sec(df)

    try:
        tow = detect_tow_segment(df)
    except Exception:
        tow = None
    if isinstance(tow, tuple) and len(tow) == 2:
        s, e = tow
        print(f"[overlay v1i] Tow detected: {s}→{e}. Excluding tow.")
        df = df.iloc[e+1:].reset_index(drop=True)
        df = ensure_time_sec(df)

    circles, dt = detect_circles_multi(df)
    circ_df = circles_to_df(df, circles)
    circ_with_cluster, clusters_raw, clusters_filt = circle_clusters(circ_df)
    alt_clusters_df, _ = detect_climbs_altitude(df)

    write_debug_file(clusters_raw, clusters_filt, dt, alt_clusters_df, len(circles))
    plot_overlay(df, circles, clusters_raw, clusters_filt, alt_clusters_df, dt)

if __name__ == "__main__":
    main()
