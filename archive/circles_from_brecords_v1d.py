#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# ------------------- IGC PARSER -------------------
def parse_igc_brecords(path):
    times, lats, lons, p_alts, g_alts = [], [], [], [], []
    day_offset = 0
    last_t = None
    with open(path, "r") as f:
        for line in f:
            if not line.startswith("B"):
                continue
            # Time HHMMSS
            hh, mm, ss = int(line[1:3]), int(line[3:5]), int(line[5:7])
            t = hh * 3600 + mm * 60 + ss
            if last_t is not None and t < last_t:
                day_offset += 86400
            last_t = t
            t += day_offset

            # Latitude DDMMmmm
            lat_deg = int(line[7:9])
            lat_min = int(line[9:11])
            lat_thou = int(line[11:14])   # thousandths of minutes
            lat = lat_deg + (lat_min + lat_thou / 1000.0) / 60.0
            if line[14] == "S":
                lat = -lat

            # Longitude DDDMMmmm
            lon_deg = int(line[15:18])
            lon_min = int(line[18:20])
            lon_thou = int(line[20:23])
            lon = lon_deg + (lon_min + lon_thou / 1000.0) / 60.0
            if line[23] == "W":
                lon = -lon

            # Pressure alt [25:30], GPS alt [30:35] (meters per IGC spec)
            try:
                p_alt = int(line[25:30])
            except ValueError:
                p_alt = np.nan
            try:
                g_alt = int(line[30:35])
            except ValueError:
                g_alt = np.nan

            times.append(t)
            lats.append(lat)
            lons.append(lon)
            p_alts.append(p_alt)
            g_alts.append(g_alt)

    df = pd.DataFrame({
        "time_s": times,
        "lat": lats,
        "lon": lons,
        "alt_pressure": p_alts,
        "alt_gps": g_alts
    })
    # Prefer GPS when present; otherwise pressure
    df["alt_raw"] = df["alt_gps"].where(~pd.isna(df["alt_gps"]), df["alt_pressure"])

    # --- Robust altitude smoothing & spike guard ---
    # 1) rolling median to kill single-sample spikes
    med = df["alt_raw"].rolling(window=5, center=True, min_periods=1).median()
    # 2) MAD-based clipping (limit residuals to 5*MAD)
    resid = df["alt_raw"] - med
    mad = np.nanmedian(np.abs(resid - np.nanmedian(resid)))
    if not np.isnan(mad) and mad > 0:
        clip = 5.0 * 1.4826 * mad  # 1.4826 ~ MAD->sigma
        clipped = np.clip(resid, -clip, clip) + med
    else:
        clipped = med.fillna(df["alt_raw"])
    # 3) light rolling mean to smooth tiny jitter
    df["alt_smooth"] = pd.Series(clipped).rolling(window=5, center=True, min_periods=1).mean()

    return df

# ------------------- GEOMETRY HELPERS -------------------
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def bearing_deg(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(phi2)
    x = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0

def unwrap_angles(angs):
    out = [angs[0]]
    for a in angs[1:]:
        prev = out[-1]
        da = a - (prev % 360.0)
        if da > 180: da -= 360
        if da <= -180: da += 360
        out.append(prev + da)
    return np.array(out, dtype=float)

# ------------------- CIRCLE DETECTOR -------------------
def detect_circles(df,
                   min_duration_s=6.0, max_duration_s=60.0,
                   min_radius_m=8.0, max_radius_m=600.0,
                   min_bank_deg=5.0, vmax_climb_ms=10.0):
    """Detect circles via cumulative heading rotation across B-fix bearings.
    Enforce duration/radius/bank sanity to reject long arcs and shallow meanders.
    """
    lat = df["lat"].to_numpy()
    lon = df["lon"].to_numpy()
    alt = df["alt_smooth"].to_numpy()
    t = df["time_s"].to_numpy()

    n = len(df)
    if n < 3:
        return pd.DataFrame(columns=[
            "circle_id","seg_id","t_start","t_end","duration_s",
            "avg_speed_kmh","alt_gain_m","climb_rate_ms",
            "turn_radius_m","bank_angle_deg","lat","lon"
        ])

    bearings = np.zeros(n)
    for i in range(1, n):
        bearings[i] = bearing_deg(lat[i-1], lon[i-1], lat[i], lon[i])
    uw = unwrap_angles(bearings)

    circles = []
    start_idx, circle_id, g = 0, 0, 9.81
    i = 1
    while i < n:
        rot = abs(uw[i] - uw[start_idx])
        if rot >= 360.0:
            i0, i1 = start_idx, i
            dur = t[i1] - t[i0]

            # reject unrealistic "circles"
            if not (min_duration_s <= dur <= max_duration_s):
                start_idx = i
                i += 1
                continue

            # distance & mean ground speed
            dist = 0.0
            for k in range(i0+1, i1+1):
                dist += haversine_m(lat[k-1], lon[k-1], lat[k], lon[k])
            v_mean = dist/dur if dur > 0 else np.nan  # m/s

            # Angular rate and turn radius
            omega = 2*math.pi/dur  # rad/s ~ one full turn
            radius = (v_mean/omega) if omega > 0 else np.nan
            if (np.isnan(radius)) or (radius < min_radius_m) or (radius > max_radius_m):
                start_idx = i
                i += 1
                continue

            bank = math.degrees(math.atan((v_mean**2)/(g*radius))) if (radius and np.isfinite(radius)) else np.nan
            if (np.isnan(bank)) or (bank < min_bank_deg):
                start_idx = i
                i += 1
                continue

            # Robust altitude gain by least-squares slope (reduces edge effects)
            idx = np.arange(i0, i1+1)
            tt = t[idx] - t[i0]
            aa = alt[idx]
            if len(tt) >= 3 and np.all(np.isfinite(tt)) and np.any(np.isfinite(aa)):
                A = np.vstack([tt, np.ones_like(tt)]).T
                mask = np.isfinite(aa)
                if mask.sum() >= 3:
                    m, c = np.linalg.lstsq(A[mask], aa[mask], rcond=None)[0]
                    alt_gain = float(m * dur)
                else:
                    span = i1 - i0 + 1
                    q = max(1, span // 4)
                    first_med = np.nanmedian(aa[:q])
                    last_med  = np.nanmedian(aa[-q:])
                    alt_gain = (last_med - first_med) if np.isfinite(first_med) and np.isfinite(last_med) else np.nan
            else:
                alt_gain = np.nan

            # Physical plausibility bound
            max_gain = vmax_climb_ms * dur
            if np.isfinite(alt_gain) and abs(alt_gain) > max_gain:
                alt_gain = np.sign(alt_gain) * max_gain

            climb = (alt_gain / dur) if (dur > 0 and np.isfinite(alt_gain)) else np.nan

            circles.append({
                "circle_id": circle_id,
                "seg_id": None,
                "t_start": float(t[i0]),
                "t_end": float(t[i1]),
                "duration_s": float(dur),
                "avg_speed_kmh": float(v_mean*3.6) if np.isfinite(v_mean) else np.nan,
                "alt_gain_m":  float(alt_gain) if np.isfinite(alt_gain) else np.nan,
                "climb_rate_ms": float(climb) if np.isfinite(climb) else np.nan,
                "turn_radius_m": float(radius) if np.isfinite(radius) else np.nan,
                "bank_angle_deg": float(bank) if np.isfinite(bank) else np.nan,
                "lat": float(np.nanmean(lat[i0:i1+1])),
                "lon": float(np.nanmean(lon[i0:i1+1])),
            })
            circle_id += 1

            start_idx = i
        i += 1

    return pd.DataFrame(circles)

def detect_tow_end(track,
                   H_ft=2700.0,      # Height gate (ft)
                   T_s=180.0,        # Time gate (s)
                   D_ft=100.0,       # Descent gate (ft)
                   steady_s=60.0):   # Minimum steady-climb window before enabling descent gate
    """
    Return index of last tow sample (inclusive). You should slice track.iloc[tow_idx+1:] for free flight.

    Logic:
      - H: when altitude gain from launch >= H_ft → end tow.
      - T: when elapsed time >= T_s → end tow.
      - D: once an initial steady climb of >= steady_s is present, end tow at the
           first point where altitude drops by >= D_ft from the running post-release peak.

    Chooses the earliest tow end among satisfied gates.
    """
    if track.empty:
        return -1

    # --- use existing altitude column from this script ---
    alt_col = "alt_smooth" if "alt_smooth" in track.columns else (
        "alt_raw" if "alt_raw" in track.columns else (
            "alt_gps" if "alt_gps" in track.columns else "alt_pressure"
        )
    )

    FT_TO_M = 0.3048
    H_m = H_ft * FT_TO_M
    D_m = D_ft * FT_TO_M

    t = track["time_s"].to_numpy()
    alt = track[alt_col].to_numpy()

    t0 = float(t[0])
    a0 = float(alt[0])

    # --- Gate H: altitude rise from launch
    tow_idx_H = None
    for i in range(len(alt)):
        if alt[i] - a0 >= H_m:
            tow_idx_H = i
            break

    # --- Gate T: elapsed time since launch
    tow_idx_T = None
    for i in range(len(t)):
        if (t[i] - t0) >= T_s:
            tow_idx_T = i
            break

    # --- Gate D: after steady climb, first drop of >= D_m from running peak
    tow_idx_D = None
    climb_start_i = None
    for i in range(1, len(alt)):
        dalt = alt[i] - alt[i-1]
        if dalt > 0:
            if climb_start_i is None:
                climb_start_i = i-1
        else:
            # reset if not climbing
            climb_start_i = None

        # Once we have steady_s of continuous climb, enable descent trigger
        if climb_start_i is not None and (t[i] - t[climb_start_i]) >= steady_s:
            run_peak = alt[i]
            for j in range(i+1, len(alt)):
                if alt[j] > run_peak:
                    run_peak = alt[j]
                if run_peak - alt[j] >= D_m:
                    tow_idx_D = j
                    break
            break  # after enabling, we either found the drop or not

    # Pick the earliest tow end among ones that fired
    candidates = [idx for idx in (tow_idx_H, tow_idx_T, tow_idx_D) if idx is not None]
    if not candidates:
        return -1
    return min(candidates)

# ------------------- MAIN -------------------
def main():
    import argparse
    from pathlib import Path
    import shutil
    import pandas as pd  # harmless reimport if already imported

    ap = argparse.ArgumentParser()
    ap.add_argument("igc", nargs="?", help="Path to IGC file")
    args = ap.parse_args()

    # Defaults
    default_igc = Path("igc/2020-11-08 Lumpy Paterson 108645.igc")
    out_root = Path("../outputs/batch_csv")

    # Resolve IGC path (arg or prompt)
    if args.igc:
        igc_path = Path(args.igc)
    else:
        inp = input(f"Enter IGC file path [default: {default_igc}]: ").strip()
        igc_path = Path(inp) if inp else default_igc

    if not igc_path.exists():
        raise FileNotFoundError(igc_path)

    # Per-flight run directory under outputs/batch_csv/<flight_basename>
    flight = igc_path.stem.rstrip(')').strip()
    run_dir = out_root / flight
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy the source IGC into the run_dir (overwrite if different file)
    igc_local = run_dir / igc_path.name
    try:
        shutil.copy2(igc_path, igc_local)
    except Exception:
        if igc_path.resolve() != igc_local.resolve():
            raise

    # --- Parse track and DETECT CIRCLES ---
    track_df = parse_igc_brecords(igc_local)

    # --- cut off aerotow segment ---
    tow_end_idx = detect_tow_end(track_df,
                                 H_ft=2700.0,
                                 T_s=180.0,
                                 D_ft=100.0,
                                 steady_s=60.0)
    if tow_end_idx >= 0 and tow_end_idx < len(track_df) - 1:
        track_df = track_df.iloc[tow_end_idx + 1:].reset_index(drop=True)
        print(f"[INFO] Tow cut at idx={tow_end_idx}, samples left={len(track_df)}")

    circles_df = detect_circles(track_df)

    # --- Normalize names (tolerate legacy runs) ---
    d = circles_df.reset_index(drop=True).copy()
    if "alt_gained_m" in d.columns and "alt_gain_m" not in d.columns:
        d = d.rename(columns={"alt_gained_m": "alt_gain_m"})

    # Derive duration if needed
    if "duration_s" not in d.columns and {"t_start", "t_end"} <= set(d.columns):
        d["duration_s"] = d["t_end"] - d["t_start"]

    # Derive climb if possible
    if "climb_rate_ms" not in d.columns and {"alt_gain_m", "duration_s"} <= set(d.columns):
        d["climb_rate_ms"] = d["alt_gain_m"] / d["duration_s"].replace(0, pd.NA)

    # Add circle_diameter_m from radius
    if "turn_radius_m" in d.columns and "circle_diameter_m" not in d.columns:
        d["circle_diameter_m"] = d["turn_radius_m"] * 2.0

    # Canonical order you requested
    canonical = [
        "lat", "lon", "t_start", "t_end", "climb_rate_ms", "alt_gain_m", "duration_s",
        "circle_id", "seg_id", "avg_speed_kmh", "turn_radius_m", "circle_diameter_m", "bank_angle_deg"
    ]
    extras = [c for c in d.columns if c not in canonical]
    d = d[[c for c in canonical if c in d.columns] + extras]

    # Write circles.csv in run_dir
    # Write circles.csv in run_dir
    out_csv = run_dir / "circles.csv"
    d.to_csv(out_csv, index=False)
    print(f"[OK] wrote {len(d)} circles → {out_csv}")

    # --- NEW: copy original IGC into run_dir ---
    import shutil
    stem = run_dir.name  # folder name like "123310"
    src_dir = Path("/igc")
    src1 = src_dir / f"{stem}.igc"
    src2 = src_dir / f"{stem}.IGC"
    dest = run_dir / f"{stem}.igc"
    try:
        if src1.exists():
            shutil.copy2(src1, dest)
            print(f"[OK] copied {src1} → {dest}")
        elif src2.exists():
            shutil.copy2(src2, dest)
            print(f"[OK] copied {src2} → {dest}")
        else:
            print(f"[WARN] no IGC found in {src_dir} for {stem}")
    except Exception as e:
        print(f"[ERR] copying IGC failed: {e}")
    # --------------------------------------------


    return 0

if __name__ == "__main__":
    raise SystemExit(main())