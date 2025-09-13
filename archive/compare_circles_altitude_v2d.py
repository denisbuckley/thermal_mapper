
"""
compare_circles_altitude_v2d.py
--------------------------------
Purpose
    Compare altitude-based thermal clusters with circle-based clusters — exposing
    **both raw and size-filtered cluster views** — to quantify agreement.

What this script does
    1) Parses IGC and removes tow.
    2) Detects circles and clusters them (space + time).
    3) Detects altitude-based thermals (rising segments with valley merge).
    4) Computes cluster size `n`; makes a **filtered view** with n≥MIN_CLUSTER_SIZE.
    5) Builds two comparisons:
        a) Altitude vs **raw** circle clusters
        b) Altitude vs **filtered (n≥MIN_CLUSTER_SIZE)** circle clusters
    6) Writes two CSVs:
        - outputs/compare_raw_<timestamp>.csv
        - outputs/compare_filtered_<timestamp>.csv
       Each includes nearest match distance and time overlap/gap.
    7) Prints first 30 rows of the filtered comparison for quick inspection.

Tuning
    - CLUSTER_EPS_M, CLUSTER_GAP_S as in overlay
    - MIN_CLUSTER_SIZE (NEW): threshold for "strong" circle clusters
    - MATCH_EPS_M, MATCH_GAP_S: spatial/temporal thresholds for matching

Usage
    Run directly in PyCharm. Inspect console + CSVs in outputs/.
"""

from pathlib import Path
from typing import Tuple, List
from datetime import datetime
import math
import numpy as np
import pandas as pd

from igc_utils import parse_igc, compute_derived, detect_tow_segment

# === Injected by apply_tuning_patches_v1.py ===
from tuning_loader import load_tuning, override_globals
_tuning = load_tuning("config/tuning_params.csv")
override_globals(globals(), _tuning, allowed={
    "MIN_CLIMB_S","MIN_GAIN_M","SMOOTH_RADIUS_S",
    "MAX_GAP_S","ALT_DROP_M","ALT_DROP_FRAC",
    "A_EPS_M","A_MIN_SAMPLES"
})



R_EARTH_M = 6371000.0

# Matching thresholds
MATCH_EPS_M = 2500.0  # spatial match radius (m)
MATCH_GAP_S = 1200.0  # if no overlap, max allowed time gap (s)

# Circles tuning (same as overlay)
C_MIN_ARC_DEG      = 300.0
C_SMOOTH_S         = 3.0
C_MAX_WIN_SAMPLES  = 2400
C_MIN_DIR_RATIO    = 0.65
CLUSTER_EPS_M      = 1500.0
CLUSTER_GAP_S      = 900.0

# Cluster filter (NEW)
MIN_CLUSTER_SIZE   = 5

# Altitude tuning (same as overlay)
A_SMOOTH_S         = 9.0
A_MIN_RATE_MPS     = 0.5
A_MIN_DUR_S        = 60.0
A_MIN_GAIN_M       = 60.0
A_ALT_DROP_M       = 180.0
A_ALT_DROP_FRAC    = 0.30

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
        return circ_with_cluster, clusters_df, clusters_df
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

def match_alt_to_circles(alt_df: pd.DataFrame, circ_clusters_df: pd.DataFrame):
    rows = []
    if alt_df.empty or circ_clusters_df.empty:
        return pd.DataFrame(rows)
    A = alt_df.copy()
    A['t0'] = pd.to_datetime(A['start_time'])
    A['t1'] = pd.to_datetime(A['end_time'])
    C = circ_clusters_df.copy()
    C['t0'] = pd.to_datetime(C['start_time'])
    C['t1'] = pd.to_datetime(C['end_time'])
    for _, ar in A.iterrows():
        best = None
        for _, cr in C.iterrows():
            dist_m = float(haversine_m(ar['lat_mid'], ar['lon_mid'], cr['center_lat'], cr['center_lon']))
            if dist_m <= MATCH_EPS_M:
                latest_start = max(ar['t0'], cr['t0'])
                earliest_end = min(ar['t1'], cr['t1'])
                overlap_s = max(0.0, (earliest_end - latest_start).total_seconds())
                if overlap_s == 0:
                    gap1 = (cr['t0'] - ar['t1']).total_seconds()
                    gap2 = (ar['t0'] - cr['t1']).total_seconds()
                    min_gap = max(0.0, min(gap1, gap2))
                else:
                    min_gap = 0.0
                score = (dist_m, -overlap_s, min_gap)
                if best is None or score < best[0]:
                    best = (score, cr, dist_m, overlap_s, min_gap)
        rows.append(dict(
            thermal_id=int(ar['thermal_id']),
            thermal_gain_m=float(ar['gain_m']),
            thermal_dur_s=float(ar['duration_s']),
            thermal_time=str(ar['start_time']) + " → " + str(ar['end_time']),
            thermal_lat=float(ar['lat_mid']), thermal_lon=float(ar['lon_mid']),
            match_cluster_id = int(best[1]['cluster_id']) if best else None,
            match_dist_m = float(best[2]) if best else None,
            time_overlap_s = float(best[3]) if best else 0.0,
            time_gap_s = float(best[4]) if best else None,
            cluster_n_circles = int(best[1]['n']) if best else None,
            cluster_mean_radius_m = float(best[1]['mean_radius_m']) if best else None,
        ))
    return pd.DataFrame(rows)

def main():
    igc_file = "../2020-11-08 Lumpy Paterson 108645.igc"  # <-- EDIT if needed
    print(f"[compare v2d] Parsing IGC: {igc_file}")
    df = parse_igc(igc_file)
    df = compute_derived(df)
    df = ensure_time_sec(df)

    try:
        tow = detect_tow_segment(df)
    except Exception:
        tow = None
    if isinstance(tow, tuple) and len(tow) == 2:
        s, e = tow
        print(f"[compare v2d] Tow detected: {s}→{e}. Excluding tow.")
        df = df.iloc[e+1:].reset_index(drop=True)
        df = ensure_time_sec(df)

    # Detect
    circles, dt = detect_circles_multi(df)
    circ_df = circles_to_df(df, circles)
    circ_with_cluster, clusters_raw, clusters_filt = circle_clusters(circ_df)
    alt_clusters_df, _ = detect_climbs_altitude(df)

    out_dir = Path.cwd() / "outputs"
    ensure_dir(out_dir)

    # Build both comparisons
    comp_raw = match_alt_to_circles(alt_clusters_df, clusters_raw)
    comp_filt = match_alt_to_circles(alt_clusters_df, clusters_filt)

    # Save
    csv_raw  = out_dir / f"compare_raw_{ts_stamp()}.csv"
    csv_filt = out_dir / f"compare_filtered_{ts_stamp()}.csv"
    comp_raw.to_csv(csv_raw, index=False)
    comp_filt.to_csv(csv_filt, index=False)

    # Also save cluster summaries for reference
    clusters_raw_csv  = out_dir / f"circle_clusters_raw_{ts_stamp()}.csv"
    clusters_filt_csv = out_dir / f"circle_clusters_filtered_{ts_stamp()}.csv"
    clusters_raw.to_csv(clusters_raw_csv, index=False)
    clusters_filt.to_csv(clusters_filt_csv, index=False)

    print(f"[compare] Circles: {len(circ_df)} | Clusters raw: {len(clusters_raw)} | Clusters (n≥{MIN_CLUSTER_SIZE}): {len(clusters_filt)} | Altitude clusters: {len(alt_clusters_df)}")
    print(f"[compare] Wrote: {csv_raw.name}, {csv_filt.name}")
    with pd.option_context("display.max_rows", 50, "display.width", 160):
        print("\n[Filtered matches preview]\n", comp_filt.fillna("").head(30))

if __name__ == "__main__":
    main()