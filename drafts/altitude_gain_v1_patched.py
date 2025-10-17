
"""
altitude_gain_v1.py
=================================
Goal: Detect **sustained altitude-gain segments** only (no circle logic).
- Parse IGC via your existing pipeline (igc_utils: parse_igc/compute_derived/detect_tow_segment)
- Smooth altitude
- Compute vario (m/s)
- Identify climb segments with:
    * min average climb rate (m/s)
    * min duration (s)
    * min net gain (m)
    * small gap allowance
- Print a clean table of segments
- Plot altitude vs time with climb segments highlighted
- Plot map with climb spans in orange
- Log console to altitude_gain_<yymmddhhmmss>.txt
"""

import sys, datetime
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from archive.igc_utils import parse_igc, compute_derived, detect_tow_segment

# === Injected by apply_tuning_patches_v1.py ===
from tuning_loader import load_tuning, override_globals
_tuning = load_tuning("config/tuning_params.csv")
override_globals(globals(), _tuning, allowed={
    "MIN_CLIMB_S","MIN_GAIN_M","SMOOTH_RADIUS_S",
    "MAX_GAP_S","ALT_DROP_M","ALT_DROP_FRAC",
    "A_EPS_M","A_MIN_SAMPLES"
})



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
                          allow_gap_seconds: float = 6.0) -> List[Tuple[int,int,float,float,float]]:
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

# -------------------- plotting --------------------
def plot_altitude_with_climbs(df: pd.DataFrame, segments, alt_s: pd.Series, alt_col: str):
    fig, ax = plt.subplots(figsize=(11, 6))
    t = df['time']
    ax.plot(t, df[alt_col], lw=0.8, alpha=0.4, label=f"{alt_col} (raw)")
    ax.plot(t, alt_s, lw=1.5, label=f"{alt_col} (smoothed)")
    for (s,e,gain,dur,mean_mps) in segments:
        ax.axvspan(t.iloc[s], t.iloc[e], color='orange', alpha=0.25)
        ax.text(t.iloc[s], alt_s.iloc[s], f"+{int(round(gain))} m\n{int(round(dur))} s",
                fontsize=8, va='bottom', ha='left')
    ax.set_title("Altitude vs Time — Sustained Climb Segments")
    ax.set_xlabel("Time"); ax.set_ylabel("Altitude (m)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_map_with_climbs(df: pd.DataFrame, segments):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(df["lon"], df["lat"], color="green", lw=1.2, alpha=0.7, label="Track")
    for (s,e,_,_,_) in segments:
        ax.plot(df.loc[s:e, "lon"], df.loc[s:e, "lat"], color="orange", lw=2.0, alpha=0.9)
    ax.set_title("Flight track with sustained climb segments (orange)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# -------------------- main --------------------
def main():
    ts = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    logf = open(f"altitude_gain_{ts}.txt", "w")
    sys.stdout = sys.stderr = Tee(sys.stdout, logf)

    # Hard-coded path for PyCharm run
    path = "2020-11-08 Lumpy Paterson 108645.igc"
    print(f"[alt v1] Parsing IGC: {path}")
    df = parse_igc(path)
    if len(df) == 0:
        print("[alt v1] No rows parsed; aborting."); return

    df = compute_derived(df)
    df = ensure_time_sec(df)

    try:
        tow = detect_tow_segment(df)
    except Exception as e:
        print(f"[alt v1] Tow detect error: {e}")
        tow = None
    if tow and isinstance(tow, tuple) and len(tow) == 2:
        s,e = tow
        print(f"[alt v1] Tow detected: {s}→{e}. Excluding tow.")
        df = df.iloc[e+1:].reset_index(drop=True)
        df = ensure_time_sec(df)
    else:
        print("[alt v1] Tow not clearly detected; using full trace.")
    print(f"[alt v1] Points after tow trim: {len(df)}")

    segments, vario, alt_s, alt_col, dt = detect_climb_segments(df)
    print(f"[alt v1] Cadence ≈ {dt:.2f} s/fix")
    print(f"[alt v1] Climb segments found: {len(segments)}")
    for i,(s,e,gain,dur,mean_mps) in enumerate(segments, start=1):
        print(f"  Climb {i}: +{gain:.0f} m  dur={dur:.0f}s  mean={mean_mps:.2f} m/s  "
              f"{df['time'].iloc[s].time()}→{df['time'].iloc[e].time()}")

    if len(df):
        plot_altitude_with_climbs(df, segments, alt_s, alt_col)
        plot_map_with_climbs(df, segments)

if __name__ == "__main__":
    main()