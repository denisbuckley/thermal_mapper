#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import pandas as pd
import numpy as np

# ------------------- IGC PARSER -------------------
def parse_igc_brecords(path):
    times, lats, lons, p_alts, g_alts = [], [], [], [], []
    day_offset = 0
    last_t = None
    with open(path, "r") as f:
        for line in f:
            if not line.startswith("B"):
                continue
            # Time HHMMSS (0-based string indices per IGC spec)
            hh, mm, ss = int(line[1:3]), int(line[3:5]), int(line[5:7])
            t = hh * 3600 + mm * 60 + ss
            if last_t is not None and t < last_t:
                # crossed midnight; advance a day
                day_offset += 86400
            last_t = t
            t += day_offset

            # Latitude DDMMmmm + hemisphere
            lat_deg = int(line[7:9])
            lat_min = int(line[9:11])
            lat_thou = int(line[11:14])  # thousandths of minutes
            lat = lat_deg + (lat_min + lat_thou / 1000.0) / 60.0
            if line[14] == "S":
                lat = -lat

            # Longitude DDDMMmmm + hemisphere
            lon_deg = int(line[15:18])
            lon_min = int(line[18:20])
            lon_thou = int(line[20:23])
            lon = lon_deg + (lon_min + lon_thou / 1000.0) / 60.0
            if line[23] == "W":
                lon = -lon

            # Fix validity at [24], then pressure alt [25:30], GPS alt [30:35]
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

    # Choose best altitude: prefer GPS when present, else pressure
    df["alt"] = df["alt_gps"].where(~df["alt_gps"].isna(), df["alt_pressure"])
    # Simple robust smoothing to reduce spikes (rolling median)
    df["alt_smooth"] = df["alt"].rolling(window=5, center=True, min_periods=1).median()

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
def detect_circles(df, min_duration_s=6.0, min_radius_m=8.0):
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

    # Bearings between consecutive fixes
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
                # distance & mean ground speed
                dist = 0.0
                for k in range(i0+1, i1+1):
                    dist += haversine_m(lat[k-1], lon[k-1], lat[k], lon[k])
                v_mean = dist/dur if dur > 0 else np.nan  # m/s

                # Angular rate and turn radius
                omega = 2*math.pi/dur  # rad/s for ~one full turn
                radius = (v_mean/omega) if omega > 0 else np.nan
                if np.isnan(radius) or radius < min_radius_m:
                    radius = np.nan
                bank = math.degrees(math.atan((v_mean**2)/(g*radius))) if (radius and np.isfinite(radius)) else np.nan

                # Robust altitude gain: median of last quartile minus median of first quartile
                span = i1 - i0 + 1
                q = max(1, span // 4)
                first_med = np.nanmedian(alt[i0:i0+q])
                last_med  = np.nanmedian(alt[i1-q+1:i1+1])
                alt_gain = (last_med - first_med) if np.isfinite(first_med) and np.isfinite(last_med) else np.nan
                climb = (alt_gain / dur) if (dur > 0 and np.isfinite(alt_gain)) else np.nan

                circles.append({
                    "circle_id": circle_id,
                    "seg_id": None,
                    "t_start": t[i0],
                    "t_end": t[i1],
                    "duration_s": dur,
                    "avg_speed_kmh": v_mean*3.6 if np.isfinite(v_mean) else np.nan,
                    "alt_gained_m": alt_gain,
                    "climb_rate_ms": climb,
                    "turn_radius_m": radius,
                    "bank_angle_deg": bank,
                    "lat": float(np.nanmean(lat[i0:i1+1])),
                    "lon": float(np.nanmean(lon[i0:i1+1])),
                })
                circle_id += 1
            start_idx = i
        i += 1

    return pd.DataFrame(circles)

# ------------------- MAIN -------------------
def main():
    default_igc = "2020-11-08 Lumpy Paterson 108645.igc_subset"
    path = input(f"Enter path to IGC file [default: {default_igc}]: ").strip() or default_igc

    df = parse_igc_brecords(path)
    circles_df = detect_circles(df)

    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/circles.csv"
    circles_df.to_csv(out_path, index=False)

    print(f"Wrote {len(circles_df)} circles → {out_path}")
    # Small summary instead of full dump
    if not circles_df.empty:
        print(circles_df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
