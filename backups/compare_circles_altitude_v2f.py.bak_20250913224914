
#!/usr/bin/env python3
"""
compare_circles_altitude_v2f.py

Compares altitude-based thermal clusters to *circle clusters* built with the
same guardrails and clustering parameters used by overlay v1k.

Outputs (written to ./outputs):
  - compare_raw_<ts>.csv         : alt cluster -> nearest raw circle cluster (no n filter)
  - compare_filtered_<ts>.csv    : alt cluster -> nearest filtered circle cluster (n >= CL_MIN_COUNT)
  - circle_clusters_raw_<ts>.csv
  - circle_clusters_filtered_<ts>.csv

Also prints a parameter banner so screenshots carry the tuning used.

Author: chatgpt-igc compare v2f
"""

# --------------------------- USER SETTINGS ---------------------------
DEFAULT_IGC = "2020-11-08 Lumpy Paterson 108645.igc"

# Circle detection params (aligned with overlay v1k)
C_SMOOTH_S         = 3.0
C_MIN_ARC_DEG      = 300.0
C_MIN_DIR_RATIO    = 0.65
C_MAX_WIN_SAMPLES  = 1000

# Guardrails (aligned with overlay v1k defaults; widen if needed)
C_TURN_RATE_MIN    = 4.0     # deg/s
C_TURN_RATE_MAX    = 20.0    # deg/s
C_RADIUS_MIN_M     = 30.0    # m
C_RADIUS_MAX_M     = 600.0   # m

# Circle clustering params (aligned with overlay v1k)
CL_EPS_M           = 1500.0  # meters
CL_GAP_S           = 15*60.0 # seconds
CL_MIN_COUNT       = 1       # keep clusters with at least this many circles

# Altitude clustering (lightweight; for matching targets)
ALT_SMOOTH_S       = 9.0
ALT_MIN_RATE_MPS   = 0.5
ALT_MIN_DUR_S      = 60.0
ALT_MIN_GAIN_M     = 60.0

# Matching window
MATCH_EPS_M        = 2500.0  # meters max to consider a match
MATCH_ALLOW_GAP_S  = 20*60.0 # if no overlap, allow up to this time gap

# --------------------------- IMPORTS ---------------------------
import os, math, warnings
from datetime import datetime, timezone
import numpy as np
import pandas as pd

try:
    from igc_utils import parse_igc, compute_derived, detect_tow_segment
    HAVE_IGC_UTILS = True
except Exception:
    HAVE_IGC_UTILS = False
    warnings.warn("igc_utils not found; using minimal IGC parser (lat/lon/time/alt only).")

# --------------------------- HELPERS ---------------------------
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1 = np.radians(lat1); p2 = np.radians(lat2)
    dphi = np.radians(lat2-lat1)
    dlmb = np.radians(lon2-lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlmb/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))

def bearing_deg(lat1, lon1, lat2, lon2):
    phi1 = np.radians(lat1); phi2 = np.radians(lat2)
    dlmb = np.radians(lon2-lon1)
    y = np.sin(dlmb) * np.cos(phi2)
    x = np.cos(phi1)*np.sin(phi2) - np.sin(phi1)*np.cos(phi2)*np.cos(dlmb
    )
    brng = np.degrees(np.arctan2(y, x))
    return (brng + 360.0) % 360.0

def unwrap_signed_deg(d):
    return (d + 180.0) % 360.0 - 180.0

def infer_cadence_s(ts):
    if len(ts) < 3: return 1.0
    dt = np.median(np.diff(ts.values).astype('timedelta64[s]').astype(float))
    return max(0.5, dt)

def minimal_parse_igc(path):
    lats=[]; lons=[]; alts=[]; times=[]
    day_hint = None
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            if line.startswith('HFDTE'):
                d = line.strip()[5:11]
                try:
                    day_hint = datetime.strptime(d, "%d%m%y").date()
                except Exception:
                    day_hint = None
            if len(line) > 35 and line[0] == 'B':
                hh=int(line[1:3]); mm=int(line[3:5]); ss=int(line[5:7])
                lat = int(line[7:9]) + int(line[9:11])/60.0 + int(line[11:14])/60000.0
                if line[14] == 'S': lat = -lat
                lon = int(line[15:18]) + int(line[18:20])/60.0 + int(line[20:23])/60000.0
                if line[23] == 'W': lon = -lon
                try:
                    alt = int(line[30:35])  # GPS alt
                except Exception:
                    alt = np.nan
                if day_hint is None:
                    t = datetime(2000,1,1,hh,mm,ss, tzinfo=timezone.utc)
                else:
                    t = datetime(day_hint.year, day_hint.month, day_hint.day, hh, mm, ss, tzinfo=timezone.utc)
                lats.append(lat); lons.append(lon); alts.append(alt); times.append(t)
    return pd.DataFrame(dict(time=pd.to_datetime(times), lat=lats, lon=lons, gps_alt=alts))

def load_igc(path):
    if HAVE_IGC_UTILS:
        df = parse_igc(path)
        df = compute_derived(df) if 'gs_mps' not in df.columns else df
    else:
        df = minimal_parse_igc(path)
    return df.sort_values('time').reset_index(drop=True)

def maybe_exclude_tow(df):
    if HAVE_IGC_UTILS:
        i0, i1 = detect_tow_segment(df)
        if i1 > i0:
            return df.iloc[i1:].reset_index(drop=True)
    # fallback: heuristic — drop first 8 minutes
    t0 = df['time'].iloc[0]
    return df.loc[df['time'] >= t0 + pd.Timedelta(minutes=8)].reset_index(drop=True)

# --------------------------- Circles ---------------------------
def detect_circles(df):
    ts = pd.to_datetime(df['time'])
    cadence = infer_cadence_s(ts)

    lat = df['lat'].to_numpy()
    lon = df['lon'].to_numpy()

    hdg = np.zeros(len(df))
    hdg[1:] = bearing_deg(lat[:-1], lon[:-1], lat[1:], lon[1:])

    win = max(1, int(round(C_SMOOTH_S / cadence)))
    if win > 1:
        k = np.ones(win)/win
        hdg = np.convolve(hdg, k, mode='same')

    dh = unwrap_signed_deg(np.diff(hdg, prepend=hdg[0]))

    rows = []
    N = len(df)
    max_win = min(C_MAX_WIN_SAMPLES, N)
    i = 0
    while i < N-2:
        pos_sum = 0.0; neg_sum = 0.0
        j = i+1
        while j < min(N, i+max_win):
            d = dh[j]
            if d >= 0: pos_sum += d
            else:      neg_sum += d
            arc = pos_sum + neg_sum  # signed

            if abs(arc) >= C_MIN_ARC_DEG:
                dt = (ts.iloc[j] - ts.iloc[i]).total_seconds()
                if dt <= 1e-6: break
                turn_rate = abs(arc)/dt
                dist_m = haversine_m(lat[i], lon[i], lat[j], lon[j])
                rad_est = dist_m / max(1e-6, math.radians(abs(arc)))
                dir_ratio = max(abs(pos_sum), abs(neg_sum)) / max(1e-6, abs(pos_sum)+abs(neg_sum))

                if (C_TURN_RATE_MIN <= turn_rate <= C_TURN_RATE_MAX and
                    C_RADIUS_MIN_M   <= rad_est   <= C_RADIUS_MAX_M   and
                    dir_ratio >= C_MIN_DIR_RATIO):
                    mid = (i+j)//2
                    rows.append(dict(
                        start_idx=i, end_idx=j,
                        start_time=ts.iloc[i], end_time=ts.iloc[j],
                        lat=df['lat'].iloc[mid], lon=df['lon'].iloc[mid],
                        turn_rate_dps=turn_rate, radius_m=rad_est,
                        arc_deg=arc, dir_ratio=dir_ratio
                    ))
                i = j
                break
            j += 1
        else:
            i += 1
    return pd.DataFrame(rows)

def cluster_circles(circles_df, eps_m=CL_EPS_M, gap_s=CL_GAP_S, min_count=CL_MIN_COUNT):
    if circles_df.empty:
        return circles_df.assign(cluster_id=pd.Series(dtype=int)), pd.DataFrame(columns=['cluster_id','n','lat','lon','start_time','end_time'])
    circles_df = circles_df.sort_values('start_time').reset_index(drop=True)
    cluster_ids = -np.ones(len(circles_df), dtype=int)
    centers_lat = []; centers_lon = []; counts = []; start_times = []; end_times = []
    current_id = -1; last_idx = -1; last_time = None; cur_points = []

    def finalize_cluster(cid, indices):
        if not indices: return
        lats = circles_df.loc[indices,'lat'].to_numpy()
        lons = circles_df.loc[indices,'lon'].to_numpy()
        centers_lat.append(lats.mean()); centers_lon.append(lons.mean())
        counts.append(len(indices))
        start_times.append(circles_df.loc[indices,'start_time'].min())
        end_times.append(circles_df.loc[indices,'end_time'].max())

    for idx, row in circles_df.iterrows():
        t = row['start_time']
        if current_id < 0:
            current_id = 0; cur_points = [idx]; last_idx = idx; last_time = t; cluster_ids[idx] = current_id; continue
        d = haversine_m(circles_df.loc[last_idx,'lat'], circles_df.loc[last_idx,'lon'], row['lat'], row['lon'])
        dt = (t - last_time).total_seconds()
        if d <= eps_m and dt <= gap_s:
            cluster_ids[idx] = current_id; cur_points.append(idx); last_idx = idx; last_time = t
        else:
            finalize_cluster(current_id, cur_points)
            current_id += 1; cur_points = [idx]; last_idx = idx; last_time = t; cluster_ids[idx] = current_id
    finalize_cluster(current_id, cur_points)

    clusters_df = pd.DataFrame(dict(
        cluster_id=np.arange(len(centers_lat)),
        n=counts, lat=centers_lat, lon=centers_lon,
        start_time=start_times, end_time=end_times
    ))
    keep_ids = set(clusters_df.loc[clusters_df['n'] >= min_count, 'cluster_id'].astype(int).tolist())
    return circles_df.assign(cluster_id=cluster_ids), clusters_df, clusters_df[clusters_df['cluster_id'].isin(list(keep_ids))].reset_index(drop=True)

# --------------------------- Altitude (light) ---------------------------
def detect_altitude_clusters(df):
    ts = pd.to_datetime(df['time'])
    cadence = infer_cadence_s(ts)
    w = max(1, int(round(ALT_SMOOTH_S/cadence)))
    alt = df['gps_alt'].astype(float).to_numpy()
    if w > 1:
        k = np.ones(w)/w
        sm = np.convolve(alt, k, mode='same')
    else:
        sm = alt.copy()

    rows = []
    i = 0; N=len(df)
    while i < N-1:
        j = i+1
        gain = 0.0; dur = 0.0
        while j < N:
            dt = (ts.iloc[j]-ts.iloc[j-1]).total_seconds()
            if dt <= 0: dt = 1.0
            rate = (sm[j]-sm[j-1])/dt
            if rate >= ALT_MIN_RATE_MPS:
                gain += (sm[j]-sm[j-1]); dur += dt; j += 1
            else:
                break
        if dur >= ALT_MIN_DUR_S and gain >= ALT_MIN_GAIN_M:
            mid = (i+j)//2
            rows.append(dict(
                thermal_id=len(rows)+1,
                lat=df['lat'].iloc[mid], lon=df['lon'].iloc[mid],
                start_time=ts.iloc[i], end_time=ts.iloc[j-1],
                gain_m=gain, dur_s=dur
            ))
            i = j
        else:
            i += 1
    return pd.DataFrame(rows)

# --------------------------- Matching ---------------------------
def match_alt_to_circles(alt_df, circ_df):
    """Nearest circle cluster (within MATCH_EPS_M) with time-overlap preference."""
    rows = []
    if alt_df.empty or circ_df.empty:
        return pd.DataFrame(rows)
    A = alt_df.copy()
    C = circ_df.copy()
    for _, ar in A.iterrows():
        best = None
        for _, cr in C.iterrows():
            dist_m = float(haversine_m(ar['lat'], ar['lon'], cr['lat'], cr['lon']))
            if dist_m <= MATCH_EPS_M:
                latest_start = max(ar['start_time'], cr['start_time'])
                earliest_end = min(ar['end_time'], cr['end_time'])
                overlap_s = max(0.0, (earliest_end - latest_start).total_seconds())
                if overlap_s == 0:
                    gap1 = (cr['start_time'] - ar['end_time']).total_seconds()
                    gap2 = (ar['start_time'] - cr['end_time']).total_seconds()
                    min_gap = max(0.0, min(gap1, gap2))
                    if min_gap > MATCH_ALLOW_GAP_S:  # too far apart in time
                        continue
                else:
                    min_gap = 0.0
                score = (dist_m, -overlap_s, min_gap)
                if best is None or score < best[0]:
                    best = (score, cr, dist_m, overlap_s, min_gap)
        rows.append(dict(
            thermal_id = int(ar.get('thermal_id', 0)),
            thermal_lat = float(ar['lat']),
            thermal_lon = float(ar['lon']),
            thermal_gain_m = float(ar.get('gain_m', np.nan)),
            thermal_dur_s = float(ar.get('dur_s', np.nan)),
            match_cluster_id = int(best[1]['cluster_id']) if best else None,
            match_dist_m = float(best[2]) if best else None,
            time_overlap_s = float(best[3]) if best else 0.0,
            time_gap_s = float(best[4]) if best else None,
            cluster_n = int(best[1]['n']) if best else None
        ))
    return pd.DataFrame(rows)

# --------------------------- MAIN ---------------------------
def main():
    print("[compare v2f] Parameters:")
    print(f"  Circle: smooth={C_SMOOTH_S}s, arc≥{C_MIN_ARC_DEG}°, dir≥{C_MIN_DIR_RATIO}, win≤{C_MAX_WIN_SAMPLES}")
    print(f"  Guardrails: turn_rate∈[{C_TURN_RATE_MIN},{C_TURN_RATE_MAX}] deg/s, radius∈[{C_RADIUS_MIN_M},{C_RADIUS_MAX_M}] m")
    print(f"  Cluster: eps={CL_EPS_M} m, gap={CL_GAP_S/60:.0f} min, keep n≥{CL_MIN_COUNT}")
    print(f"  Match: eps≤{MATCH_EPS_M} m, gap≤{MATCH_ALLOW_GAP_S/60:.0f} min (if no overlap)")

    from datetime import datetime as _dt
    ts_tag = _dt.utcnow().strftime("%Y%m%d%H%M%S")
    out_dir = "outputs"; os.makedirs(out_dir, exist_ok=True)

    # Load & tow-cut
    df = load_igc(DEFAULT_IGC)
    df = maybe_exclude_tow(df)

    # Circles + clusters
    circles = detect_circles(df)
    circ_with_id, clusters_raw = cluster_circles(circles, eps_m=CL_EPS_M, gap_s=CL_GAP_S, min_count=1)
    _, clusters_fil = cluster_circles(circles, eps_m=CL_EPS_M, gap_s=CL_GAP_S, min_count=CL_MIN_COUNT)

    # Altitude clusters
    alt = detect_altitude_clusters(df)

    # Matches
    comp_raw = match_alt_to_circles(alt, clusters_raw)
    comp_fil = match_alt_to_circles(alt, clusters_fil)

    # Save
    path_raw  = os.path.join(out_dir, f"compare_raw_{ts_tag}.csv")
    path_fil  = os.path.join(out_dir, f"compare_filtered_{ts_tag}.csv")
    path_cr   = os.path.join(out_dir, f"circle_clusters_raw_{ts_tag}.csv")
    path_cf   = os.path.join(out_dir, f"circle_clusters_filtered_{ts_tag}.csv")
    comp_raw.to_csv(path_raw, index=False)
    comp_fil.to_csv(path_fil, index=False)
    clusters_raw.to_csv(path_cr, index=False)
    clusters_fil.to_csv(path_cf, index=False)

    print(f"[compare v2f] Wrote:")
    print(" ", path_raw)
    print(" ", path_fil)
    print(" ", path_cr)
    print(" ", path_cf)

if __name__ == "__main__":
    main()
