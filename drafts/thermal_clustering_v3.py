
"""
thermal_clustering_v3.py
Changes vs v2:
- Visually separates circles that DID NOT make it into any thermal cluster.
  * Clustered circles: blue segments with red endpoints
  * Unclustered circles: grey segments with light endpoints
- Legend shows counts for quick QA.
- Keeps v2 filters (min circles + min duration) and radius rings + labels for clusters.

Requires: igc_utils.py providing parse_igc / compute_derived / detect_tow_segment
"""

import sys, math, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from archive.igc_utils import parse_igc, compute_derived, detect_tow_segment

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

# -------------------- geo helpers --------------------
R_EARTH_M = 6371000.0

def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2*R_EARTH_M*np.arcsin(np.sqrt(a))

def ll_to_xy_m(lat, lon, lat0, lon0):
    lat_r = np.radians(lat); lon_r = np.radians(lon)
    lat0_r = math.radians(lat0); lon0_r = math.radians(lon0)
    x = R_EARTH_M*(lon_r - lon0_r)*math.cos(lat0_r)
    y = R_EARTH_M*(lat_r - lat0_r)
    return x, y

def xy_to_ll_m(x, y, lat0, lon0):
    lat0_r = math.radians(lat0); lon0_r = math.radians(lon0)
    lat = np.degrees(y/R_EARTH_M + lat0_r)
    lon = np.degrees(x/(R_EARTH_M*math.cos(lat0_r)) + lon0_r)
    return lat, lon

def fit_circle_xy(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    A = np.column_stack([2*x, 2*y, np.ones_like(x)])
    b = x**2 + y**2
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        xc, yc, c = sol
        r = math.sqrt(max(c + xc**2 + yc**2, 0.0))
        return xc, yc, r
    except np.linalg.LinAlgError:
        xc, yc = np.mean(x), np.mean(y)
        r = float(np.mean(np.sqrt((x-xc)**2+(y-yc)**2)))
        return xc, yc, r

# -------------------- circle detector --------------------
def _signed_delta_deg(a: float, b: float) -> float:
    d = b - a
    if d > 180.0: d -= 360.0
    elif d <= -180.0: d += 360.0
    return d

def _ensure_sec(df: pd.DataFrame) -> pd.DataFrame:
    if 'sec' not in df.columns:
        df = df.copy()
        df['sec'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds().astype(float)
    return df

def _sec_per_fix(df: pd.DataFrame) -> float:
    s = df['sec']
    ds = s.diff().dropna()
    dt = float(ds.median()) if not ds.empty else 1.0
    if dt <= 0 or not (dt == dt): dt = 1.0
    return dt

def find_circles_360_tuned(df: pd.DataFrame,
                            min_arc_deg: float = 350.0,
                            max_arc_deg: float = 370.0,
                            tiny_step_deg: float = 0.3,
                            target_min_dur_s: float = 18.0,
                            target_max_dur_s: float = 45.0,
                            turn_rate_bounds_dps: tuple = (8.0, 20.0)):
    df = _ensure_sec(df)
    n = len(df)
    if n < 3:
        return [], 1.0

    dt_fix = _sec_per_fix(df)
    min_samp = max(3, int(target_min_dur_s / max(dt_fix, 1e-6)) - 1)
    max_samp = max(min_samp+1, int(target_max_dur_s / max(dt_fix, 1e-6)) + 1)

    headings = df['heading'].to_numpy()
    secs = df['sec'].to_numpy()
    circles = []

    i = 0
    while i < n - 1:
        found = False
        while i < n - 1:
            step = _signed_delta_deg(headings[i], headings[i+1])
            if abs(step) >= tiny_step_deg:
                dirn = 1 if step > 0 else -1
                found = True
                break
            i += 1
        if not found:
            break

        start = i
        cum = 0.0
        sum_abs = 0.0
        j = i + 1
        while j < n and (j - start) <= (max_samp + 2):
            step = _signed_delta_deg(headings[j-1], headings[j])
            if abs(step) < tiny_step_deg:
                j += 1; continue
            if step * dirn < 0:
                i = j; break
            cum += step
            sum_abs += abs(step)
            s_count = (j - start)
            dur_s = secs[j] - secs[start]
            arc = abs(cum)

            if s_count >= min_samp and s_count <= max_samp and min_arc_deg <= arc <= max_arc_deg:
                mean_abs_tr = (sum_abs / max(dur_s, 1e-6))
                lo, hi = turn_rate_bounds_dps
                if lo <= mean_abs_tr <= hi:
                    circles.append((start, j, cum, dur_s, mean_abs_tr))
                    i = j
                    break

            if s_count > (max_samp + 2):
                i = j; break
            j += 1
        else:
            i = j

    return circles, dt_fix

# -------------------- clustering --------------------
def _circle_centers(df: pd.DataFrame, circles):
    centers = []
    for (s,e, arc, dur_s, tr) in circles:
        seg = df.iloc[s:e+1]
        lat0, lon0 = float(seg['lat'].mean()), float(seg['lon'].mean())
        x, y = ll_to_xy_m(seg['lat'].values, seg['lon'].values, lat0, lon0)
        xc, yc, r = fit_circle_xy(x, y)
        clat, clon = xy_to_ll_m(xc, yc, lat0, lon0)
        t0, t1 = seg['time'].iloc[0], seg['time'].iloc[-1]
        centers.append((clat, clon, r, t0, t1))
    return centers

def cluster_circles_spatiotemporal(df: pd.DataFrame,
                                   circles: list,
                                   merge_dist_m: float = 350.0,
                                   max_time_gap_s: float = 20*60.0,
                                   min_circles_per_cluster: int = 3,
                                   min_duration_s: float = 90.0):
    if not circles:
        return [], np.array([], dtype=bool)

    centers = _circle_centers(df, circles)
    n = len(centers)
    used = np.zeros(n, dtype=bool)
    clusters = []
    # Map circle index -> whether it's clustered
    clustered_mask = np.zeros(n, dtype=bool)

    def time_close(a, b):
        latest_start = max(a[0], b[0])
        earliest_end = min(a[1], b[1])
        overlap = (earliest_end - latest_start).total_seconds()
        if overlap >= 0:
            return True
        gap = (latest_start - earliest_end).total_seconds()
        return gap <= max_time_gap_s

    for i in range(n):
        if used[i]: continue
        stack = [i]
        members = []
        while stack:
            k = stack.pop()
            if used[k]: continue
            used[k] = True
            members.append(k)
            for j in range(n):
                if used[j]: continue
                di = float(haversine_m(centers[k][0], centers[k][1], centers[j][0], centers[j][1]))
                if di <= merge_dist_m and time_close((centers[k][3], centers[k][4]), (centers[j][3], centers[j][4])):
                    stack.append(j)

        # Aggregate and filter
        lats = [centers[m][0] for m in members]
        lons = [centers[m][1] for m in members]
        rs   = [centers[m][2] for m in members]
        t0s  = [centers[m][3] for m in members]
        t1s  = [centers[m][4] for m in members]
        span_s = (max(t1s) - min(t0s)).total_seconds() if members else 0.0

        if len(members) >= min_circles_per_cluster and span_s >= min_duration_s:
            for m in members:
                clustered_mask[m] = True
            C_lat = float(np.mean(lats))
            C_lon = float(np.mean(lons))
            Rm    = float(np.mean(rs))
            clusters.append({
                'center_lat': C_lat,
                'center_lon': C_lon,
                'mean_radius_m': Rm,
                'count': len(members),
                'first_time': min(t0s),
                'last_time':  max(t1s),
                'span_s': span_s,
                'member_idxs': members
            })
    return clusters, clustered_mask

# -------------------- plotting --------------------
def _ring_points(lat_c, lon_c, radius_m, npts=90):
    if radius_m <= 0:
        return np.array([lon_c]), np.array([lat_c])
    angles = np.linspace(0, 2*math.pi, npts, endpoint=True)
    x = radius_m * np.cos(angles)
    y = radius_m * np.sin(angles)
    lats, lons = xy_to_ll_m(x, y, lat_c, lon_c)
    return np.asarray(lons), np.asarray(lats)

def plot_circles_and_thermals(df: pd.DataFrame, circles: list, thermals: list, clustered_mask: np.ndarray):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(df["lon"], df["lat"], color="green", lw=1.2, alpha=0.7, label="Track")

    # Split circles by cluster membership
    clustered_indices = [i for i, flag in enumerate(clustered_mask) if flag]
    unclustered_indices = [i for i, flag in enumerate(clustered_mask) if not flag]

    # Clustered: blue + red endpoints
    for idx in clustered_indices:
        s, e, cum, dur_s, mean_tr = circles[idx]
        ax.plot(df.loc[s:e, "lon"], df.loc[s:e, "lat"], color="blue", lw=1.8, alpha=0.95)
        ax.scatter([df.loc[s, "lon"], df.loc[e, "lon"]],
                   [df.loc[s, "lat"], df.loc[e, "lat"]],
                   color="red", s=24, zorder=5)

    # Unclustered: grey + light endpoints
    for idx in unclustered_indices:
        s, e, cum, dur_s, mean_tr = circles[idx]
        ax.plot(df.loc[s:e, "lon"], df.loc[s:e, "lat"], color="0.6", lw=1.5, alpha=0.7)
        ax.scatter([df.loc[s, "lon"], df.loc[e, "lon"]],
                   [df.loc[s, "lat"], df.loc[e, "lat"]],
                   color="0.65", s=18, zorder=4)

    # Thermals: center X + ring + label
    for i, th in enumerate(thermals, start=1):
        ax.scatter(th['center_lon'], th['center_lat'], marker='x', s=80, c='k', zorder=6)
        ring_x, ring_y = _ring_points(th['center_lat'], th['center_lon'], th['mean_radius_m'])
        ax.plot(ring_x, ring_y, lw=1.1, alpha=0.6)
        ax.annotate(f"T{i}", (th['center_lon'], th['center_lat']),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=9, weight='bold')

    # Legend with counts
    ax.set_title("Thermals clustering: clustered (blue/red) vs unclustered (grey)\nCenters=black X, rings=mean radius")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    clustered_n = len(clustered_indices)
    unclustered_n = len(unclustered_indices)
    ax.legend([plt.Line2D([0], [0], color='green', lw=1.2),
               plt.Line2D([0], [0], color='blue', lw=1.8),
               plt.Line2D([0], [0], color='0.6', lw=1.5),
               plt.Line2D([0], [0], marker='x', color='k', lw=0)],
              [f"Track",
               f"Circles in thermals (n={clustered_n})",
               f"Circles NOT clustered (n={unclustered_n})",
               "Thermal centers"],
              loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.show()

# -------------------- main --------------------
def main():
    ts = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    logf = open(f"thermal_cluster_{ts}.txt", "w")
    sys.stdout = sys.stderr = Tee(sys.stdout, logf)

    try:
        path = DEFAULT_IGC_PATH
    except NameError:
        path = "../2020-11-08 Lumpy Paterson 108645.igc"
    print(f"[thermal v3] Parsing IGC: {path}")
    df = parse_igc(path)

    try:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
    except Exception as e:
        print(f"[thermal v3] to_datetime error: {e}")
    if len(df) == 0:
        print("[thermal v3] No rows parsed; aborting.")
        return

    df = compute_derived(df)

    # Tow trim
    try:
        tow = detect_tow_segment(df)
    except Exception as e:
        tow = None
        print(f"[thermal v3] Tow detect error: {e}")
    if tow and isinstance(tow, tuple) and len(tow) == 2:
        s, e = tow
        print(f"[thermal v3] Tow detected: {s}→{e}. Excluding tow.")
        df = df.iloc[e+1:].reset_index(drop=True)
    else:
        print("[thermal v3] Tow not clearly detected; using full trace.")
    print(f"[thermal v3] Points after tow trim: {len(df)}")

    # Circles
    circles, dt_fix = find_circles_360_tuned(df)
    print(f"[thermal v3] Cadence ≈ {dt_fix:.2f} s/fix")
    print(f"[thermal v3] Circles detected: {len(circles)}")

    # Clusters & mask
    thermals, clustered_mask = cluster_circles_spatiotemporal(
        df, circles,
        merge_dist_m=350.0,
        max_time_gap_s=20*60,
        min_circles_per_cluster=3,
        min_duration_s=90.0
    )
    print(f"[thermal v3] Thermals (after filters): {len(thermals)}")
    clustered_n = int(np.sum(clustered_mask)) if len(clustered_mask) else 0
    print(f"[thermal v3] Circles in clusters: {clustered_n} / {len(circles)}")

    for i, th in enumerate(thermals, 1):
        print(f"  Thermal {i}: n={th['count']} span={th['span_s']:.0f}s  center=({th['center_lat']:.6f},{th['center_lon']:.6f}) "
              f"Rmean={th['mean_radius_m']:.0f} m  time={th['first_time'].time()}→{th['last_time'].time()}")

    # Plot
    if len(circles) > 0:
        plot_circles_and_thermals(df, circles, thermals, clustered_mask)
    else:
        print("[thermal v3] No circles; nothing to plot.")

if __name__ == "__main__":
    main()
