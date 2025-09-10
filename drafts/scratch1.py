#!/usr/bin/env python3
"""
IGC Thermal/Tow Analysis — Modular Implementation (per your spec)

Now with robust climb-rate realism:
- Prefer pressure altitude for vertical calculations (smoother than GPS)
- 5-sample moving-average smoothing on altitude before diff
- Sustained-lift segments require >=5 samples and reject unrealistic avg > 5 m/s
- Console adds flight distance, duration, average speed, and average climb rates

Stages:
1) Parser → DataFrame(time, lat, lon, gps_alt, pressure_alt)
2) Derived fields → dt, dist, speed, climb_rate, heading, turn_rate (using smoothed pressure altitude when available)
3) Tow detection → altitude-threshold + pattern-change + fallback
4) Sustained lift → ≥50 m within 20 s OR ≤500 m horizontal (realism filters)
5) Thermals (circling) → circle detection + continuity + thresholds
6) Outputs → console summary + plot (track=light green, sustained=red, thermals=black x)

Run: python igc_thermal_modular.py /path/to/file.igc
Dependencies: numpy, pandas, matplotlib
"""
from __future__ import annotations
import sys
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 0) Config / Defaults
# =========================
DEFAULT_IGC_PATH = "2020-11-08 Lumpy Paterson 108645.igc"

# Tow detection
TOW_MIN_ALT_GAIN_M = 457.0            # 1500 ft
TOW_PATTERN_WINDOW_S = 5 * 60         # 5 minutes
TOW_SUSTAINED_DESCENT_M = 50.0        # m
TOW_MIN_GAIN_REQUIRED = 200.0         # m (fallback min gain)

# Sustained lift thresholds
THERMAL_MIN_ALT_GAIN = 50.0           # m
THERMAL_TIME_WINDOW_S = 20.0          # s (max)
THERMAL_DISTANCE_WINDOW_M = 500.0     # m (max)
SUSTAINED_MIN_SAMPLES = 5             # reject ultra-short runs
MAX_REALISTIC_AVG_CLIMB_MS = 5.0      # reject absurd averages

# Circling / thermals
MIN_TURNS = 3
MAX_DRIFT_M = 300.0
MIN_CLIMB_RATE_MS = 0.5               # m/s (cluster avg)
TURN_SMOOTH_N = 5                     # samples for moving avg
RUNNING_AVG_WINDOW_S = 20.0           # s

G = 9.80665                           # gravity (bank estimate)

# =========================
# 1) Parser
# =========================

def parse_igc(filepath: str) -> pd.DataFrame:
    """Robustly parse IGC B-records into a DataFrame: time, lat, lon, gps_alt, pressure_alt.
    Accepts extended B-lines by slicing the first 35 chars (standard core).
    """
    rows = []
    start_date = None
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            # strip BOM variants + whitespace
            line = line.strip().lstrip("\ufeff").lstrip("\uFEFF").lstrip("\xEF\xBB\xBF")
            if not line:
                continue
            if line.startswith('HFDTE') or line.startswith('HFDTEDATE'):
                digits = ''.join(ch for ch in line if ch.isdigit())
                if len(digits) >= 6:
                    dd, mm, yy = int(digits[:2]), int(digits[2:4]), int(digits[4:6])
                    year = 2000 + yy if yy < 80 else 1900 + yy
                    try:
                        start_date = datetime(year, mm, dd)
                    except Exception:
                        start_date = None
            if line.startswith('B') and len(line) >= 35:
                core = line[:35]
                try:
                    hh, mi, ss = int(core[1:3]), int(core[3:5]), int(core[5:7])
                    if start_date:
                        ts = datetime(start_date.year, start_date.month, start_date.day, hh, mi, ss)
                    else:
                        d = datetime.utcnow().date()
                        ts = datetime(d.year, d.month, d.day, hh, mi, ss)
                    # lat/lon
                    lat_str, lat_hemi = core[7:14], core[14]
                    lon_str, lon_hemi = core[15:23], core[23]
                    lat = int(lat_str[:2]) + float(lat_str[2:]) / 60000.0
                    if lat_hemi == 'S':
                        lat = -lat
                    lon = int(lon_str[:3]) + float(lon_str[3:]) / 60000.0
                    if lon_hemi == 'W':
                        lon = -lon
                    palt = int(core[25:30])
                    galt = int(core[30:35])
                    rows.append((ts, lat, lon, galt, palt))
                except Exception:
                    # ignore malformed B-line
                    continue
    df = pd.DataFrame(rows, columns=['time', 'lat', 'lon', 'gps_alt', 'pressure_alt'])
    df.sort_values('time', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# =========================
# 2) Derived Fields
# =========================

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def bearing_deg(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(phi2)
    y = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dl)
    br = math.degrees(math.atan2(x, y))
    return (br + 360.0) % 360.0

def circ_diff_deg(a, b):
    d = b - a
    if d > 180.0: d -= 360.0
    if d <= -180.0: d += 360.0
    return d

def moving_average(arr, n):
    if n <= 1:
        return np.asarray(arr)
    k = np.ones(n) / n
    return np.convolve(arr, k, mode='same')


def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Select altitude for vertical calcs: prefer pressure_alt if present/positive
    alt = df['pressure_alt'].where(df['pressure_alt'].notna() & (df['pressure_alt'] > 0), df['gps_alt'])
    # Smooth altitude to reduce GPS jitter
    alt_smooth = pd.Series(alt).rolling(window=5, center=True, min_periods=1).mean()
    df['alt_used'] = alt_smooth

    # time delta
    dt = df['time'].diff().dt.total_seconds().fillna(1.0)
    dt[dt <= 0] = 1.0
    df['dt'] = dt

    # distance & heading
    dists, headings = [0.0], [0.0]
    for i in range(1, len(df)):
        dists.append(haversine_m(df.loc[i-1,'lat'], df.loc[i-1,'lon'], df.loc[i,'lat'], df.loc[i,'lon']))
        headings.append(bearing_deg(df.loc[i-1,'lat'], df.loc[i-1,'lon'], df.loc[i,'lat'], df.loc[i,'lon']))
    df['dist'] = dists
    df['cumdist'] = df['dist'].cumsum()
    df['speed'] = df['dist'] / df['dt']

    # climb & turn (use smoothed altitude)
    df['climb_rate'] = df['alt_used'].diff().fillna(0.0) / df['dt']
    df['heading'] = headings
    turn = [0.0]
    for i in range(1, len(df)):
        d = circ_diff_deg(df.loc[i-1,'heading'], df.loc[i,'heading'])
        turn.append(d / df.loc[i,'dt'])
    df['turn_rate_deg_s'] = turn
    df['turn_rate_deg_s_smooth'] = moving_average(df['turn_rate_deg_s'].values, TURN_SMOOTH_N)

    # bank estimate (approx)
    omega = np.deg2rad(df['turn_rate_deg_s_smooth'].values)
    v = df['speed'].values
    with np.errstate(divide='ignore', invalid='ignore'):
        bank = np.arctan(np.clip((omega * v) / G, -10, 10))
    df['bank_deg'] = np.degrees(bank)

    # running 20s avg climb (on smoothed climb)
    run = np.zeros(len(df))
    for i in range(len(df)):
        t0 = df.loc[i,'time'] - timedelta(seconds=RUNNING_AVG_WINDOW_S/2)
        t1 = df.loc[i,'time'] + timedelta(seconds=RUNNING_AVG_WINDOW_S/2)
        mask = (df['time'] >= t0) & (df['time'] <= t1)
        run[i] = df.loc[mask, 'climb_rate'].mean()
    df['climb_rate_run20s'] = run
    return df

# =========================
# 3) Tow Detection
# =========================

def detect_tow_segment(df: pd.DataFrame) -> Optional[Tuple[int,int]]:
    if df.empty:
        return None
    start_alt = df.loc[0,'alt_used']
    threshold = start_alt + TOW_MIN_ALT_GAIN_M
    reached = df.index[df['alt_used'] >= threshold].tolist()
    if not reached:
        return None
    min_idx = reached[0]
    t_start = df.loc[min_idx,'time']
    t_end = t_start + timedelta(seconds=TOW_PATTERN_WINDOW_S)
    win = df[(df['time'] >= t_start) & (df['time'] <= t_end)]
    if len(win) >= 2:
        base_alt = win['alt_used'].iloc[0]
        # descent > 50m within window
        if (base_alt - win['alt_used']).max() >= TOW_SUSTAINED_DESCENT_M:
            drop_idx = win.index[(base_alt - win['alt_used']) >= TOW_SUSTAINED_DESCENT_M][0]
            return (0, drop_idx)
        # transition to tight circling
        avg_tr = win['turn_rate_deg_s'].abs().mean()
        frac_high = (win['turn_rate_deg_s'].abs() > 6.0).sum() / len(win)
        if avg_tr > 6.0 and frac_high >= 0.6:
            first_high = win.index[(win['turn_rate_deg_s'].abs() > 6.0)][0]
            return (0, first_high)
    # fallback: highest point within 20 minutes
    cutoff = df[df['time'] <= (df.loc[0,'time'] + timedelta(minutes=20))]
    if not cutoff.empty:
        max_idx = cutoff['alt_used'].idxmax()
        if df.loc[max_idx,'alt_used'] - start_alt >= TOW_MIN_GAIN_REQUIRED:
            return (0, max_idx)
    return None

# =========================
# 4) Sustained Lift Detection
# =========================

def detect_sustained_climbs(df: pd.DataFrame,
                            min_gain=THERMAL_MIN_ALT_GAIN,
                            time_window_s=THERMAL_TIME_WINDOW_S,
                            dist_window_m=THERMAL_DISTANCE_WINDOW_M) -> List[Dict]:
    """Return non-overlapping sustained-lift segments with realism checks."""
    segments = []
    n = len(df)
    i = 0
    while i < n-1:
        if df.loc[i+1,'alt_used'] > df.loc[i,'alt_used']:
            s = i
            j = i+1
            while j < n and df.loc[j,'alt_used'] >= df.loc[j-1,'alt_used']:
                j += 1
            e = j-1
            if e <= s:
                i = e + 1
                continue
            gain = float(df.loc[e,'alt_used'] - df.loc[s,'alt_used'])
            dur = (df.loc[e,'time'] - df.loc[s,'time']).total_seconds()
            dist = df.loc[s+1:e,'dist'].sum() if e > s else 0.0
            nsamp = (e - s + 1)
            avg_climb = gain / dur if dur > 0 else 0.0
            # Apply core rules + realism guards
            if gain >= min_gain and (dur <= time_window_s or dist <= dist_window_m) \
               and nsamp >= SUSTAINED_MIN_SAMPLES and 0.0 < avg_climb <= MAX_REALISTIC_AVG_CLIMB_MS:
                segments.append({'start': s, 'end': e, 'gain': gain, 'dur_s': dur, 'avg_ms': avg_climb})
            i = e + 1
        else:
            i += 1
    return segments

# =========================
# 5) Circles & Thermals
# =========================
@dataclass
class Circle:
    center_lat: float
    center_lon: float
    radius_m: float
    start_idx: int
    end_idx: int
    heading_change_deg: float
    avg_climb: float


def find_circles(df: pd.DataFrame, min_circle_heading_deg=300.0, max_len_samples=1000) -> List[Tuple[int,int,float]]:
    circles = []
    n = len(df)
    i = 0
    while i < n-2:
        cum = 0.0
        j = i+1
        last_h = df.loc[i,'heading']
        while j < n and (j - i) < max_len_samples:
            now_h = df.loc[j,'heading']
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
    return circles


def fit_circle_center(df: pd.DataFrame, s: int, e: int) -> Tuple[Optional[float],Optional[float],Optional[float]]:
    if e - s < 5:
        return None, None, None
    lat0, lon0 = df.loc[s,'lat'], df.loc[s,'lon']
    xs, ys = [], []
    for i in range(s, e+1):
        la, lo = df.loc[i,'lat'], df.loc[i,'lon']
        dx = haversine_m(lat0, lon0, lat0, lo) * (1 if lo >= lon0 else -1)
        dy = haversine_m(lat0, lon0, la, lon0) * (1 if la >= lat0 else -1)
        xs.append(dx); ys.append(dy)
    x, y = np.array(xs), np.array(ys)
    A = np.column_stack([2*x, 2*y, np.ones_like(x)])
    b = x**2 + y**2
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        xc, yc, c = sol
        r = math.sqrt(max(0.0, xc*xc + yc*yc + c))
        center_lat = lat0 + (yc / 111320.0)
        center_lon = lon0 + (xc / (111320.0 * math.cos(math.radians(lat0))))
        return center_lat, center_lon, r
    except Exception:
        return None, None, None


def analyze_circles_into_thermals(df: pd.DataFrame, raw_circles: List[Tuple[int,int,float]],
                                  min_turns=MIN_TURNS, max_drift=MAX_DRIFT_M,
                                  min_climb_rate=MIN_CLIMB_RATE_MS) -> List[Circle]:
    # Fit each raw circle
    fitted = []
    for s, e, heading_change in raw_circles:
        clat, clon, rad = fit_circle_center(df, s, e)
        avg_climb = df.loc[s:e,'climb_rate'].mean() if e > s else 0.0
        fitted.append({'s': s, 'e': e, 'clat': clat, 'clon': clon, 'rad': rad,
                       'heading': heading_change, 'avg_climb': avg_climb})

    thermals: List[Circle] = []
    i = 0
    while i < len(fitted):
        if fitted[i]['clat'] is None:
            i += 1
            continue
        cluster = [fitted[i]]
        j = i + 1
        while j < len(fitted) and fitted[j]['clat'] is not None:
            drift = haversine_m(cluster[-1]['clat'], cluster[-1]['clon'], fitted[j]['clat'], fitted[j]['clon'])
            if drift <= max_drift:
                cluster.append(fitted[j])
                j += 1
            else:
                break
        # aggregate cluster
        total_heading = sum(abs(c['heading']) for c in cluster)
        approx_turns = int(round(total_heading / 360.0))
        avg_climb_cluster = float(np.mean([c['avg_climb'] for c in cluster]))
        sidx, eidx = cluster[0]['s'], cluster[-1]['e']
        center_lat = float(np.nanmean([c['clat'] for c in cluster]))
        center_lon = float(np.nanmean([c['clon'] for c in cluster]))
        radius = float(np.nanmean([c['rad'] for c in cluster]))
        if approx_turns >= min_turns and avg_climb_cluster >= min_climb_rate:
            thermals.append(Circle(center_lat, center_lon, radius, sidx, eidx, total_heading, avg_climb_cluster))
        i = j if j > i else i + 1
    return thermals

# =========================
# 6) Outputs (summary & plot)
# =========================

def summarize_and_plot(df: pd.DataFrame, sustained: List[Dict], thermals: List[Circle]):
    # Gains & durations
    gain_sust = sum(seg['gain'] for seg in sustained)
    dur_sust = sum(seg['dur_s'] for seg in sustained) if sustained else 0.0

    gain_therm = 0.0
    dur_therm = 0.0
    for t in thermals:
        if t.end_idx > t.start_idx:
            da = df.loc[t.end_idx, 'alt_used'] - df.loc[t.start_idx, 'alt_used']
            if da > 0:
                gain_therm += float(da)
            dur_therm += (df.loc[t.end_idx, 'time'] - df.loc[t.start_idx, 'time']).total_seconds()

    # Flight duration & distance (post-tow analysis segment)
    start_time = df['time'].iloc[0]
    end_time = df['time'].iloc[-1]
    duration_total = (end_time - start_time).total_seconds()
    dist_total = float(df['cumdist'].iloc[-1]) if 'cumdist' in df else float('nan')

    # Average climbs
    avg_climb_sust = (gain_sust / dur_sust) if dur_sust > 0 else 0.0
    avg_climb_therm = (gain_therm / dur_therm) if dur_therm > 0 else 0.0

    # Average speed over analysis segment
    avg_speed_ms = (dist_total / duration_total) if duration_total > 0 else 0.0

    # Console summary
    print("===== Analysis Summary =====")
    print(f"Flight distance: {dist_total/1000:.2f} km")
    print(f"Flight duration: {duration_total/60:.1f} minutes")
    print(f"Average speed: {avg_speed_ms*3.6:.1f} km/h")
    print(f"Number of thermals (circling): {len(thermals)}")
    print(f"Number of sustained lift segments: {len(sustained)}")
    print(f"Total altitude gained in sustained lift: {gain_sust:.1f} m")
    print(f"Total altitude gained while circling: {gain_therm:.1f} m")
    print(f"Average climb rate (sustained lift): {avg_climb_sust:.2f} m/s")
    print(f"Average climb rate (thermals): {avg_climb_therm:.2f} m/s")
    print("============================")

    # Plot
    plt.figure(figsize=(12,9))
    # Base track in light green
    plt.plot(df['lon'], df['lat'], '-', color='lightgreen', linewidth=1.25, alpha=0.9, label='Track')
    # Sustained lift segments in red
    first_label = True
    for seg in sustained:
        s, e = seg['start'], seg['end']
        plt.plot(df['lon'].iloc[s:e+1], df['lat'].iloc[s:e+1], '-', color='red', linewidth=2.5,
                 label='Sustained lift' if first_label else None)
        first_label = False
    # Thermal centers as black crosses
    for i, t in enumerate(thermals):
        plt.plot(t.center_lon, t.center_lat, 'x', color='black', markersize=9, markeredgewidth=2,
                 label='Thermal center' if i == 0 else None)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('IGC Track — Sustained lift (red) and thermals (black x)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        'gain_sustained_m': gain_sust,
        'gain_thermals_m': gain_therm,
        'sustained_duration_s': dur_sust,
        'thermals_duration_s': dur_therm,
        'flight_distance_m': dist_total,
        'flight_duration_s': duration_total,
        'avg_speed_ms': avg_speed_ms,
        'avg_climb_sustained_ms': avg_climb_sust,
        'avg_climb_thermals_ms': avg_climb_therm
    }

# =========================
# 7) Pipeline / Main
# =========================

def analyze_igc(filepath: str):
    print(f"Parsing IGC: {filepath}")
    raw = parse_igc(filepath)
    if raw.empty:
        raise RuntimeError('No B-records parsed from IGC file.')
    df = compute_derived(raw)

    tow = detect_tow_segment(df)
    if tow:
        _, tow_end = tow
        print(f"Tow detected: 0→{tow_end}. Excluding tow from analysis.")
        df_anal = df.loc[tow_end+1:].reset_index(drop=True)
    else:
        print("No tow detected. Using full flight.")
        df_anal = df.copy().reset_index(drop=True)

    sustained = detect_sustained_climbs(df_anal)
    raw_circles = find_circles(df_anal)
    thermals = analyze_circles_into_thermals(df_anal, raw_circles,
                                             min_turns=MIN_TURNS,
                                             max_drift=MAX_DRIFT_M,
                                             min_climb_rate=MIN_CLIMB_RATE_MS)

    metrics = summarize_and_plot(df_anal, sustained, thermals)

    # Return full results for programmatic use
    return {
        'tow_segment': tow,
        'sustained_segments': sustained,
        'thermals': thermals,
        **metrics
    }

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        path = sys.argv[1]
    else:
        path = DEFAULT_IGC_PATH
        print(f"No path provided. Using default: {path}")
    _ = analyze_igc(path)
