#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import pandas as pd

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
def detect_circles(df, min_duration_s=6.0, min_radius_m=8.0, vmax_climb_ms=10.0):
    """Detect circles via cumulative heading rotation across B-fix bearings.
    Adds physically-plausible bounds for altitude gain within a circle.
    """
    lat = df["lat"].to_numpy()
    lon = df["lon"].to_numpy()
    alt = df["alt_smooth"].to_numpy()
    t = df["time_s"].to_numpy()

    n = len(df)
    if n < 3:
        return pd.DataFrame(columns=[
            "circle_id","seg_id","t_start","t_end","duration_s",
            "avg_speed_kmh","alt_gained_m","climb_rate_ms",
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
            if dur >= min_duration_s and dur > 0:
                # d & mean ground speed
                dist = 0.0
                for k in range(i0+1, i1+1):
                    dist += haversine_m(lat[k-1], lon[k-1], lat[k], lon[k])
                v_mean = dist/dur if dur > 0 else np.nan  # m/s

                # Turn radius / bank
                omega = 2*math.pi/dur  # rad/s ~ one full turn
                radius = (v_mean/omega) if omega > 0 else np.nan
                if np.isnan(radius) or radius < min_radius_m:
                    radius = np.nan
                bank = math.degrees(math.atan((v_mean**2)/(g*radius))) if (radius and np.isfinite(radius)) else np.nan

                # Robust altitude gain by linear fit slope (reduces edge effects)
                idx = np.arange(i0, i1+1)
                tt = t[idx] - t[i0]
                aa = alt[idx]
                if len(tt) >= 3 and np.all(np.isfinite(tt)) and np.any(np.isfinite(aa)):
                    # simple least squares slope (m/s)
                    A = np.vstack([tt, np.ones_like(tt)]).T
                    mask = np.isfinite(aa)
                    if mask.sum() >= 3:
                        m, c = np.linalg.lstsq(A[mask], aa[mask], rcond=None)[0]
                        alt_gain = float(m * dur)
                    else:
                        # fallback to quartile-median method
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
                    # clip to bound, preserving sign
                    alt_gain = np.sign(alt_gain) * max_gain

                climb = (alt_gain / dur) if (dur > 0 and np.isfinite(alt_gain)) else np.nan

                circles.append({
                    "circle_id": circle_id,
                    "seg_id": None,
                    "t_start": float(t[i0]),
                    "t_end": float(t[i1]),
                    "duration_s": float(dur),
                    "avg_speed_kmh": float(v_mean*3.6) if np.isfinite(v_mean) else np.nan,
                    "alt_gained_m": float(alt_gain) if np.isfinite(alt_gain) else np.nan,
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

# ------------------- MAIN -------------------
def main():
    default_igc = "2020-11-08 Lumpy Paterson 108645.igc"
    path = input(f"Enter path to IGC file [default: {default_igc}]: ").strip() or default_igc

    df = parse_igc_brecords(path)
    circles_df = detect_circles(df)

    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/circles.csv"
    circles_df.to_csv(out_path, index=False)

    print(f"Wrote {len(circles_df)} circles → {out_path}")
    # Quick stats
    if not circles_df.empty:
        n_climb = int((circles_df["alt_gained_m"] > 0).sum())
        n_sink  = int((circles_df["alt_gained_m"] < 0).sum())
        print(f"Climb circles: {n_climb} | Sink circles: {n_sink}")
        print(circles_df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
