
#!/usr/bin/env python3
"""
overlay_circles_altitude_v1k.py

Overlay plot: track, individual circles (green ×), *circle clusters* (black × with count label),
and altitude-based thermal clusters (orange × with T# label).

Also exports a CSV of the *filtered circle clusters* automatically:
  outputs/circle_clusters_filtered_YYYYMMDDHHMMSS.csv

This file is self-contained (no CLI). Edit the DEFAULT_IGC path near the top or
run as a script; it will show the plot and write the CSV.

Key parameters are shown in a banner on the plot.
Guardrails are applied to circle candidates:
  - Turn-rate must be within [C_TURN_RATE_MIN, C_TURN_RATE_MAX] deg/s
  - Estimated radius must be within [C_RADIUS_MIN_M, C_RADIUS_MAX_M] meters
  - One-sidedness (left vs right) must be >= C_MIN_DIR_RATIO
  - Cumulative signed arc over the window must be >= C_MIN_ARC_DEG

Altitude clustering is kept simple here; it’s provided for visual comparison.

Author: chatgpt-igc_subset overlay v1k
"""

# --------------------------- USER SETTINGS ---------------------------
DEFAULT_IGC = "2020-01-11 Norm Bloch 106248.igc_subset"

# Circle detection params
C_SMOOTH_S         = 3.0      # seconds for heading smoothing
C_MIN_ARC_DEG      = 300.0    # cumulative signed arc threshold to call it a circle
C_MIN_DIR_RATIO    = 0.65     # one-sidedness ratio (|sum_pos| or |sum_neg|) / (|sum_pos|+|sum_neg|)
C_MAX_WIN_SAMPLES  = 1000     # cap on window samples (prevents giant windows)

# Guardrails
C_TURN_RATE_MIN    = 4.0      # deg/s (allow wider range if you like)
C_TURN_RATE_MAX    = 20.0     # deg/s
C_RADIUS_MIN_M     = 30.0     # meters
C_RADIUS_MAX_M     = 600.0    # meters

# Circle clustering params (spatial + temporal)
CL_EPS_M           = 1500.0   # meters (DBSCAN-like epsilon)
CL_GAP_S           = 15*60.0  # seconds between subsequent circles to stay in same cluster
CL_MIN_COUNT       = 1        # keep clusters with at least this many circles (plot label shows n)

# Altitude-based (very light) thermal detection, for comparison on overlay
ALT_SMOOTH_S       = 9.0
ALT_MIN_RATE_MPS   = 0.5
ALT_MIN_DUR_S      = 60.0
ALT_MIN_GAIN_M     = 60.0
ALT_CL_EPS_M       = 2000.0
ALT_CL_GAP_S       = 20*60.0

# --------------------------- IMPORTS ---------------------------
import os, math, warnings
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
logging.basicConfig(filename='overlay_debug.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Try to import igc_utils if present in the project; otherwise use simple parser fallback
# Try to import igc_utils if present in the project; otherwise use simple parser fallback
try:
    from archive.igc_utils import parse_igc, compute_derived, detect_tow_segment
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
    # Initial course from point1 to point2
    phi1 = np.radians(lat1); phi2 = np.radians(lat2)
    dlmb = np.radians(lon2-lon1)
    y = np.sin(dlmb) * np.cos(phi2)
    x = np.cos(phi1)*np.sin(phi2) - np.sin(phi1)*np.cos(phi2)*np.cos(dlmb)
    brng = np.degrees(np.arctan2(y, x))
    return (brng + 360.0) % 360.0

def unwrap_signed_deg(d):
    # map delta heading into [-180, +180]
    d = (d + 180.0) % 360.0 - 180.0
    return d

def infer_cadence_s(ts):
    if len(ts) < 3: return 1.0
    dt = np.median(np.diff(ts.values).astype('timedelta64[s]').astype(float))
    return max(0.5, dt)

def minimal_parse_igc(path):
    # Very small fallback: only supports B records with lat/lon/alt/time.
    lats=[]; lons=[]; alts=[]; times=[]
    day_hint = None
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            if line.startswith('HFDTE'):
                # HFDTEddmmyy
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
                alt = None
                try:
                    alt = int(line[30:35])  # GPS altitude
                except Exception:
                    alt = np.nan
                if day_hint is None:
                    t = datetime(2000,1,1,hh,mm,ss, tzinfo=timezone.utc)
                else:
                    t = datetime(day_hint.year, day_hint.month, day_hint.day, hh, mm, ss, tzinfo=timezone.utc)
                lats.append(lat); lons.append(lon); alts.append(alt); times.append(t)
    df = pd.DataFrame(dict(time=pd.to_datetime(times), lat=lats, lon=lons, gps_alt=alts))
    return df

def load_igc(path):
    if HAVE_IGC_UTILS:
        df = parse_igc(path)
        df = compute_derived(df) if 'gs_mps' not in df.columns else df
    else:
        df = minimal_parse_igc(path)
    df = df.sort_values('time').reset_index(drop=True)
    return df

def maybe_exclude_tow(df):
    if HAVE_IGC_UTILS:
        i0, i1 = detect_tow_segment(df)
        if i1 > i0:
            return df.iloc[i1:].reset_index(drop=True), (i0, i1)
    # fallback: heuristic = first 8 minutes
    t0 = df['time'].iloc[0]
    mask = (df['time'] >= t0 + pd.Timedelta(minutes=8))
    cut = df.loc[mask].reset_index(drop=True)
    return cut, None

# -------------------- Circle detection + clustering --------------------
def detect_circles(df):
    """
    Returns a DataFrame of per-circle detections with columns:
    ['start_idx','end_idx','start_time','end_time','lat','lon','turn_rate_dps','radius_m','arc_deg','dir_ratio']
    """
    ts = pd.to_datetime(df['time'])
    cadence = infer_cadence_s(ts)

    lat = df['lat'].to_numpy()
    lon = df['lon'].to_numpy()

    # Headings between successive fixes
    hdg = np.zeros(len(df))
    hdg[1:] = bearing_deg(lat[:-1], lon[:-1], lat[1:], lon[1:])

    # Smooth headings with simple moving average over seconds
    win = max(1, int(round(C_SMOOTH_S / cadence)))
    if win > 1:
        k = np.ones(win)/win
        hdg = np.convolve(hdg, k, mode='same')

    # Signed deltas
    dh = unwrap_signed_deg(np.diff(hdg, prepend=hdg[0]))

    rows = []
    N = len(df)
    max_win = min(C_MAX_WIN_SAMPLES, N)
    i = 0
    while i < N-2:
        # sliding window grow until thresholds
        pos_sum = 0.0
        neg_sum = 0.0
        j = i+1
        while j < min(N, i+max_win):
            d = dh[j]
            if d >= 0:
                pos_sum += d
            else:
                neg_sum += d  # negative
            arc = pos_sum + neg_sum  # signed

            if abs(arc) >= C_MIN_ARC_DEG:
                # window [i..j]
                dt = (ts.iloc[j] - ts.iloc[i]).total_seconds()
                if dt <= 1e-6:
                    break
                turn_rate = abs(arc)/dt  # deg/s

                # estimate radius = distance traveled / (arc in radians)
                dist_m = haversine_m(lat[i], lon[i], lat[j], lon[j])
                rad_est = dist_m / max(1e-6, math.radians(abs(arc)))

                dir_ratio = max(abs(pos_sum), abs(neg_sum)) / max(1e-6, abs(pos_sum)+abs(neg_sum))

                # guardrails
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
                # Advance: jump to j to avoid re-detecting same segment
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
    centers_lat = []
    centers_lon = []
    counts = []
    start_times = []
    end_times = []

    current_id = -1
    last_idx = -1
    last_time = None
    cur_points = []  # (lat,lon,time_index)

    def finalize_cluster(cid, indices):
        if not indices: return
        lats = circles_df.loc[indices, 'lat'].to_numpy()
        lons = circles_df.loc[indices, 'lon'].to_numpy()
        centers_lat.append(lats.mean())
        centers_lon.append(lons.mean())
        counts.append(len(indices))
        start_times.append(circles_df.loc[indices, 'start_time'].min())
        end_times.append(circles_df.loc[indices, 'end_time'].max())

    for idx, row in circles_df.iterrows():
        t = row['start_time']
        if current_id < 0:
            current_id = 0
            cur_points = [idx]
            last_idx = idx
            last_time = t
            cluster_ids[idx] = current_id
            continue

        # distance to last circle in cluster
        d = haversine_m(circles_df.loc[last_idx,'lat'], circles_df.loc[last_idx,'lon'], row['lat'], row['lon'])
        dt = (t - last_time).total_seconds()

        if d <= eps_m and dt <= gap_s:
            cluster_ids[idx] = current_id
            cur_points.append(idx)
            last_idx = idx
            last_time = t
        else:
            # finalize previous
            finalize_cluster(current_id, cur_points)
            # start new
            current_id += 1
            cur_points = [idx]
            last_idx = idx
            last_time = t
            cluster_ids[idx] = current_id

    # finalize last
    finalize_cluster(current_id, cur_points)

    clusters_df = pd.DataFrame(dict(
        cluster_id=np.arange(len(centers_lat)),
        n=counts, lat=centers_lat, lon=centers_lon,
        start_time=start_times, end_time=end_times
    ))
    # filter by min_count
    keep_ids = set(clusters_df.loc[clusters_df['n'] >= min_count, 'cluster_id'].astype(int).tolist())
    mask = [cid in keep_ids for cid in cluster_ids]
    return circles_df.assign(cluster_id=cluster_ids), clusters_df.loc[clusters_df['cluster_id'].isin(list(keep_ids))].reset_index(drop=True)

# -------------------- Altitude (very light) --------------------
def detect_altitude_clusters(df):
    # Smooth altitude
    ts = pd.to_datetime(df['time'])
    cadence = infer_cadence_s(ts)
    w = max(1, int(round(ALT_SMOOTH_S/cadence)))
    if "gps_alt" in df.columns:
        alt = df["gps_alt"].astype(float).to_numpy()
    elif "alt" in df.columns:
        alt = df["alt"].astype(float).to_numpy()
    else:
        raise KeyError("No altitude column found (expected gps_alt or alt)")
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
                gain += (sm[j]-sm[j-1])
                dur  += dt
                j += 1
            else:
                break
        if dur >= ALT_MIN_DUR_S and gain >= ALT_MIN_GAIN_M:
            mid = (i+j)//2
            rows.append(dict(lat=df['lat'].iloc[mid], lon=df['lon'].iloc[mid],
                             start_time=ts.iloc[i], end_time=ts.iloc[j-1],
                             gain_m=gain, dur_s=dur))
            i = j
        else:
            i += 1

    climbs = pd.DataFrame(rows)
    if climbs.empty:
        return climbs

    # cluster climbs similarly (eps+gap)
    climbs = climbs.sort_values('start_time').reset_index(drop=True)
    cid = -1; last_idx=-1; last_t=None; ids=[]; centers=[]; counts=[]
    cl_start=[]; cl_end=[]
    cluster_ids = -np.ones(len(climbs), dtype=int)

    def finalize(indices):
        if not indices: return
        lats = climbs.loc[indices,'lat'].to_numpy()
        lons = climbs.loc[indices,'lon'].to_numpy()
        centers.append((lats.mean(), lons.mean()))
        counts.append(len(indices))
        cl_start.append(climbs.loc[indices,'start_time'].min())
        cl_end.append(climbs.loc[indices,'end_time'].max())

    for idx,row in climbs.iterrows():
        t=row['start_time']
        if cid<0:
            cid=0; ids=[idx]; last_idx=idx; last_t=t; cluster_ids[idx]=cid
            continue
        d=haversine_m(climbs.loc[last_idx,'lat'],climbs.loc[last_idx,'lon'],row['lat'],row['lon'])
        dt=(t-last_t).total_seconds()
        if d<=ALT_CL_EPS_M and dt<=ALT_CL_GAP_S:
            ids.append(idx); cluster_ids[idx]=cid; last_idx=idx; last_t=t
        else:
            finalize(ids); cid+=1; ids=[idx]; cluster_ids[idx]=cid; last_idx=idx; last_t=t
    finalize(ids)
    cl = pd.DataFrame({
        'cluster_id': np.arange(len(centers)),
        'n': counts,
        'lat': [c[0] for c in centers],
        'lon': [c[1] for c in centers],
        'start_time': cl_start,
        'end_time': cl_end
    })
    return cl

# -------------------- Plot & Export --------------------
def plot_overlay(df, circles, circle_clusters, alt_clusters, banner_text, out_csv_path):
    fig, ax = plt.subplots(figsize=(11,7))
    ax.plot(df['lon'], df['lat'], color='g', lw=1.5, label='Track')

    # circles (green ×)
    if not circles.empty:
        ax.scatter(circles['lon'], circles['lat'], marker='x', s=25, color='green', label='Circles')

    # circle clusters (black × with n)
    if not circle_clusters.empty:
        ax.scatter(circle_clusters['lon'], circle_clusters['lat'], marker='x', s=80, color='black',
                   label='Circle clusters')

        for _, r in circle_clusters.iterrows():
            ax.annotate(
                f"{int(r['n'])}",
                xy=(r['lon'], r['lat']),
                xytext=(6, 6),  # offset in points (pixels)
                textcoords='offset points',
                fontsize=9,
                color='black',
                ha='left', va='bottom',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=0.5)
            )

    # altitude clusters (orange × with T#)
    if alt_clusters is not None and not alt_clusters.empty:
        ax.scatter(alt_clusters['lon'], alt_clusters['lat'], marker='x', s=80, color='orange', label='Altitude clusters (T#)')
        for i, r in alt_clusters.reset_index(drop=True).iterrows():
            ax.annotate(
                f"T{i + 1}",
                xy=(r['lon'], r['lat']),
                xytext=(-6, -6),  # SW corner offset in pixels
                textcoords='offset points',
                fontsize=9,
                color='orange',
                ha='right', va='top',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=0.5)
            )
    ax.legend(loc='upper left')
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Overlay: Circles (green ×), Circle clusters (black × w/ n), Altitude clusters (orange ×)")

    # Banner
    ax.text(0.5, -0.08, banner_text, transform=ax.transAxes, ha='center', va='top',
            fontsize=9, bbox=dict(boxstyle='round', fc='white', ec='0.7'))

    plt.tight_layout()
    # Save CSV of circle clusters
    if not circle_clusters.empty:
        os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
        circle_clusters.to_csv(out_csv_path, index=False)
        logger.info(f"[overlay v1k] Wrote circle clusters CSV: {out_csv_path}")
    plt.show()

# -------------------- MAIN --------------------
def main():
    igc_path = DEFAULT_IGC
    logger.info(f"[overlay v1k] Parsing IGC: {os.path.basename(igc_path)}")
    df = load_igc(igc_path)
    logger.info(f"[overlay v1k] Fixes: {len(df)}")

    df2, tow = maybe_exclude_tow(df)
    if tow:
        logger.info(f"[overlay v1k] Tow excluded: {tow[0]}→{tow[1]}")
    else:
        logger.info("[overlay v1k] Tow exclusion: heuristic (first 8 min)")

    # Detect circles
    circles = detect_circles(df2)
    logger.info(f"[overlay v1k] Circles detected: {len(circles)}")

    # Cluster circles
    circ_with_id, circ_clusters = cluster_circles(
        circles, eps_m=CL_EPS_M, gap_s=CL_GAP_S, min_count=CL_MIN_COUNT
    )
    logger.info(f"[overlay v1k] Circle clusters kept (n>={CL_MIN_COUNT}): {len(circ_clusters)}")

    # Altitude clusters (lightweight comparison)
    alt_clusters = detect_altitude_clusters(df2)
    if alt_clusters is None or alt_clusters.empty:
        logger.info("[overlay v1k] Altitude clusters: 0")
    else:
        logger.info(f"[overlay v1k] Altitude clusters: {len(alt_clusters)}")

    # Banner text
    cadence = infer_cadence_s(pd.to_datetime(df2['time']))
    banner = (
        f"Cadence: {cadence:.2f} s/fix | "
        f"Circle: arc≥{C_MIN_ARC_DEG:.0f}°, dir≥{C_MIN_DIR_RATIO:.2f}, smooth={C_SMOOTH_S:.0f} s, win≤{C_MAX_WIN_SAMPLES} | "
        f"Guardrails: turn_rate∈[{C_TURN_RATE_MIN:.0f},{C_TURN_RATE_MAX:.0f}] deg/s, radius∈[{C_RADIUS_MIN_M:.0f},{C_RADIUS_MAX_M:.0f}] m | "
        f"Cluster: eps={CL_EPS_M:.0f} m, gap={CL_GAP_S/60:.0f} min | Keep n≥{CL_MIN_COUNT}"
    )

    # Output CSV path (filtered circle clusters)
    ts_tag = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    out_csv = os.path.join("outputs", f"circle_clusters_filtered_{ts_tag}.csv")

    plot_overlay(df2, circles, circ_clusters, alt_clusters, banner, out_csv)

if __name__ == "__main__":
    main()