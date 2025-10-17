
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
circles_detector_v2_guardrails.py
---------------------------------
Detects *full circles* in an IGC track using a rolling window and adds
three robust guardrails:

1) One‑sidedness (C_MIN_DIR_RATIO)
   - Require that at least this fraction of heading deltas have the same sign
     (all left or all right). This rejects snaking/zig‑zag that sums to 360°
     without being a real circle.

2) Turn‑rate floor (C_MIN_DEG_PER_S)
   - Require the average absolute turn‑rate (deg/s) across the candidate
     circle window to be at least this value. Prevents extremely slow,
     meandering arcs being called circles.

3) Radius bounds (RADIUS_MIN_M, RADIUS_MAX_M)
   - Estimate circle radius two ways and use the safer of the two:
       a) Kinematic: radius ≈ ground_speed(m/s) / turn_rate(rad/s)
       b) Geometric: mean distance of points to their centroid
     The candidate is accepted only if the estimated radius lies within
     these bounds. Prevents very tiny jitter (too small) and huge sweeps
     (too large) from qualifying.

Input:  hardcoded DEFAULT_IGC path (kept for IDE use, no CLI required)
Output: plot + CSV (outputs/circles_250911092836.csv) + debug (outputs/circles_debug_250911092836.txt)

Parameter glossary
------------------
C_MAX_WIN_SAMPLES : maximum samples examined per candidate window. Limits
                    how long a “circle” can be (protects against drifts).
C_MIN_ARC_DEG     : minimum cumulative signed arc across the window.
C_REQUIRE_NET_360 : also require \u2248 360° net (|wrap(heading_end - heading_start)| >= 330).
C_MIN_DIR_RATIO   : one‑sidedness fraction (>= this proportion of deltas share one sign).
C_MIN_DEG_PER_S   : minimum average |turn‑rate| over the window.
RADIUS_MIN_M/MAX_M: accepted radius range.
SMOOTH_S          : heading smoothing window (seconds) before differencing.

Notes
-----
- We assume fixes are uniformly spaced in time; cadence is inferred from the
  timestamp column in the IGC file.
- This script is intentionally self‑contained so you can run it in PyCharm
  without arguments and iterate quickly.

"""

from __future__ import annotations
import math, os, sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Injected by apply_tuning_patches_v1.py ===
from tuning_loader import load_tuning, override_globals
_tuning = load_tuning("config/tuning_params.csv")
override_globals(globals(), _tuning, allowed={
    "C_MIN_ARC_DEG","C_MIN_RATE_DPS","C_MAX_RATE_DPS",
    "C_MIN_RADIUS_M","C_MAX_RADIUS_M","C_MIN_DIR_RATIO",
    "TIME_CAP_S","C_MAX_WIN_SAMPLES","C_EPS_M","C_MIN_SAMPLES"
})



# -------------------- Config --------------------
DEFAULT_IGC = "2020-11-08 Lumpy Paterson 108645.igc_subset"

# circle detection thresholds
C_MAX_WIN_SAMPLES = 2000        # safety cap on window length (samples)
C_MIN_ARC_DEG     = 300.0       # cumulative signed arc across window
C_REQUIRE_NET_360 = True        # also demand ~360 net wrap
C_MIN_DIR_RATIO   = 0.70        # one‑sidedness (same-sign fraction)
C_MIN_DEG_PER_S   = 6.0         # min mean |turn-rate| (deg/s)

# radius guardrails
RADIUS_MIN_M      = 40.0        # too small = noise / wing-rock
RADIUS_MAX_M      = 450.0       # too large = lazy arc / straight flight

# smoothing / clustering
SMOOTH_S          = 3.0         # simple rolling median seconds for headings

# plotting / outputs
OUTDIR = "outputs"

# -------------------- Helpers --------------------
def ensure_outdir():
    os.makedirs(OUTDIR, exist_ok=True)

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlmb/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def bearing_deg(lat1, lon1, lat2, lon2):
    phi1 = np.radians(lat1); phi2 = np.radians(lat2)
    dlmb = np.radians(lon2 - lon1)
    y = np.sin(dlmb)*np.cos(phi2)
    x = np.cos(phi1)*np.sin(phi2) - np.sin(phi1)*np.cos(phi2)*np.cos(dlmb)
    brng = (np.degrees(np.arctan2(y,x)) + 360.0) % 360.0
    return brng

def angdiff_signed(a, b):
    """Return signed smallest difference b-a in degrees in (-180, 180]."""
    d = (b - a + 180.0) % 360.0 - 180.0
    return d

def rolling_median_deg(series, k):
    # work in complex unit circle to avoid 0/360 wrap hurt
    ang = np.radians(series.values)
    z = np.exp(1j*ang)
    # median over real and imag separately
    re = pd.Series(z.real).rolling(k, center=True, min_periods=1).median()
    im = pd.Series(z.imag).rolling(k, center=True, min_periods=1).median()
    sm = np.degrees(np.angle(re.values + 1j*im.values)) % 360.0
    return pd.Series(sm, index=series.index)

def estimate_radius_kinematic(speed_mps, mean_abs_turn_deg_s):
    if mean_abs_turn_deg_s <= 1e-6:
        return np.inf
    turn_rad_s = np.radians(mean_abs_turn_deg_s)
    return float(np.median(speed_mps) / turn_rad_s)

def estimate_radius_geometric(lat, lon):
    # local equirectangular projection for small neighborhood
    lat0 = np.radians(np.median(lat))
    x = np.radians(lon) * 6371000.0 * np.cos(lat0)
    y = np.radians(lat) * 6371000.0
    xc = x.mean(); yc = y.mean()
    r = np.sqrt((x-xc)**2 + (y-yc)**2).mean()
    return float(r)

@dataclass
class Circle:
    i0: int
    i1: int
    arc_deg: float
    duration_s: float
    mean_turn_deg_s: float
    radius_m: float
    lat_c: float
    lon_c: float

# -------------------- IGC parsing (minimal) --------------------
def parse_igc_min(path: str) -> pd.DataFrame:
    """Minimal IGC B‑record parser: returns DataFrame with time(s), lat, lon, gps_alt."""
    times = []; lats = []; lons = []; alts = []
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            if not line.startswith('B'): 
                continue
            hh = int(line[1:3]); mm = int(line[3:5]); ss = int(line[5:7])
            # lat ddmmmmmN
            latd = int(line[7:9]); latm = int(line[9:14]); latH = line[14]
            lond = int(line[15:18]); lonm = int(line[18:23]); lonH = line[23]
            # gps alt at 25-29
            try:
                galt = int(line[25:30])
            except:
                galt = None
            lat = latd + latm/60000.0; 
            if latH == 'S': lat = -lat
            lon = lond + lonm/60000.0
            if lonH == 'W': lon = -lon
            t = hh*3600 + mm*60 + ss
            times.append(t); lats.append(lat); lons.append(lon); alts.append(galt)
    df = pd.DataFrame({'t':times,'lat':lats,'lon':lons,'gps_alt':alts})
    # infer cadence
    dt = np.diff(df['t']).astype(float)
    cadence = float(np.median(dt)) if len(dt)>0 else 3.0
    df['cadence_s'] = cadence
    # headings, speed
    brng = np.zeros(len(df))
    spd = np.zeros(len(df))
    brng[1:] = [bearing_deg(df['lat'].iloc[i-1], df['lon'].iloc[i-1],
                            df['lat'].iloc[i],   df['lon'].iloc[i]) for i in range(1,len(df))]
    dist = np.zeros(len(df)); dist[1:] = [haversine_m(df['lat'].iloc[i-1], df['lon'].iloc[i-1],
                                                     df['lat'].iloc[i],   df['lon'].iloc[i]) for i in range(1,len(df))]
    spd[1:] = dist[1:] / cadence
    df['heading'] = brng
    df['speed_mps'] = spd
    return df

# -------------------- Detection --------------------
def detect_circles(df: pd.DataFrame) -> pd.DataFrame:
    ensure_outdir()
    k = max(1, int(round(SMOOTH_S / max(1.0, df['cadence_s'].iloc[0]))))
    head_smooth = rolling_median_deg(df['heading'], k)
    # signed turn deltas
    dtheta = [angdiff_signed(head_smooth.iloc[i-1], head_smooth.iloc[i]) for i in range(1,len(df))]
    dtheta = np.array(dtheta); dt = df['cadence_s'].iloc[0]
    # cumulative arc scan with variable window
    circles = []
    n = len(df)
    i = 1
    while i < n-1:
        cum = 0.0
        same_sign = 0
        total = 0
        sign_ref = 0
        j = i
        while j < min(n-1, i + C_MAX_WIN_SAMPLES):
            d = dtheta[j-1]
            cum += d
            total += 1
            s = 1 if d >= 0 else -1
            if sign_ref == 0 and abs(d) > 0.2:
                sign_ref = s
            if s == sign_ref:
                same_sign += 1
            # check candidate
            arc_ok = abs(cum) >= C_MIN_ARC_DEG
            if arc_ok:
                dur = total * dt
                mean_abs_turn = np.mean(np.abs(dtheta[i-1:j-1])) / max(dt,1e-9)
                # mean_abs_turn currently deg per sample. Convert to deg/s:
                mean_abs_turn_deg_s = np.mean(np.abs(dtheta[i-1:j-1])) / dt if total>1 else 0.0
                dir_ratio = same_sign / max(total,1)
                net = angdiff_signed(head_smooth.iloc[i-1], head_smooth.iloc[j])
                net_ok = (abs(net) >= 330.0) if C_REQUIRE_NET_360 else True
                turnrate_ok = mean_abs_turn_deg_s >= C_MIN_DEG_PER_S
                # radius estimates
                radius_kine = estimate_radius_kinematic(df['speed_mps'].iloc[i:j], mean_abs_turn_deg_s)
                radius_geom = estimate_radius_geometric(df['lat'].iloc[i:j].values, df['lon'].iloc[i:j].values)
                radius = min(radius_kine, radius_geom)
                radius_ok = (RADIUS_MIN_M <= radius <= RADIUS_MAX_M)
                onesided_ok = (dir_ratio >= C_MIN_DIR_RATIO)
                if net_ok and turnrate_ok and radius_ok and onesided_ok:
                    lat_c = df['lat'].iloc[i:j].mean()
                    lon_c = df['lon'].iloc[i:j].mean()
                    circles.append(Circle(i, j, cum, dur, mean_abs_turn_deg_s, radius, lat_c, lon_c))
                    i = j  # advance past this circle
                    break
            j += 1
        i += 1
    # to DataFrame
    rows = [{'i0':c.i0,'i1':c.i1,'arc_deg':c.arc_deg,'dur_s':c.duration_s,
             'turn_deg_s':c.mean_turn_deg_s,'radius_m':c.radius_m,
             'lat_c':c.lat_c,'lon_c':c.lon_c} for c in circles]
    circ_df = pd.DataFrame(rows)
    # save
    csv_path = os.path.join(OUTDIR, f"circles_250911092836.csv")
    dbg_path = os.path.join(OUTDIR, f"circles_debug_250911092836.txt")
    circ_df.to_csv(csv_path, index=False)
    with open(dbg_path, 'w') as f:
        f.write(f"Cadence ≈ {df['cadence_s'].iloc[0]:.2f} s\n")
        f.write(f"Circles detected: {len(circ_df)}\n")
        f.write(f"Params: C_MIN_ARC_DEG={C_MIN_ARC_DEG}, C_MIN_DIR_RATIO={C_MIN_DIR_RATIO}, "
                f"C_MIN_DEG_PER_S={C_MIN_DEG_PER_S}, RADIUS_MIN_M={RADIUS_MIN_M}, RADIUS_MAX_M={RADIUS_MAX_M}\n")
    print(f"[circles v2] Cadence ≈ {df['cadence_s'].iloc[0]:.2f} s | circles detected: {len(circ_df)}")
    print(f"[circles v2] Wrote CSV: {csv_path}")
    print(f"[circles v2] Wrote debug: {dbg_path}")
    return circ_df

# -------------------- Plot --------------------
def plot_circles(df, circ_df):
    plt.figure(figsize=(9,6))
    plt.plot(df['lon'], df['lat'], color='green', lw=1.8, label='Track')
    if len(circ_df):
        plt.scatter(circ_df['lon_c'], circ_df['lat_c'], marker='x', s=70, color='tab:green', label='Circles')
    plt.legend()
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.title("Circles (green ×) with guardrails")
    plt.tight_layout()
    plt.show()

# -------------------- Main --------------------
def main():
    igc = DEFAULT_IGC
    if not os.path.exists(igc):
        print(f"IGC not found: {igc}")
        return
    df = parse_igc_min(igc)
    circ_df = detect_circles(df)
    plot_circles(df, circ_df)

if __name__ == "__main__":
    main()