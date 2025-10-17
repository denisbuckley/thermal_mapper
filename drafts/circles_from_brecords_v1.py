import os
import math
import pandas as pd
import numpy as np

# ------------------- IGC PARSER -------------------
def parse_igc_brecords(path):
    times, lats, lons, alts = [], [], [], []
    with open(path, "r") as f:
        for line in f:
            if not line.startswith("B"):
                continue
            # Time HHMMSS
            hh, mm, ss = int(line[1:3]), int(line[3:5]), int(line[5:7])
            t = hh * 3600 + mm * 60 + ss
            # Lat
            lat_deg = int(line[7:9])
            lat_min = int(line[9:11])
            lat_sec = int(line[11:14]) / 1000.0 * 60
            lat = lat_deg + (lat_min + lat_sec / 60) / 60.0
            if line[14] == "S":
                lat = -lat
            # Lon
            lon_deg = int(line[15:18])
            lon_min = int(line[18:20])
            lon_sec = int(line[20:23]) / 1000.0 * 60
            lon = lon_deg + (lon_min + lon_sec / 60) / 60.0
            if line[23] == "W":
                lon = -lon
            # Alt (GPS altitude if available, fallback baro)
            try:
                alt = int(line[25:30])
            except ValueError:
                alt = np.nan
            times.append(t)
            lats.append(lat)
            lons.append(lon)
            alts.append(alt)
    return pd.DataFrame({"time_s": times, "lat": lats, "lon": lons, "alt": alts})


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
    lat, lon, alt, t = df["lat"].to_numpy(), df["lon"].to_numpy(), df["alt"].to_numpy(), df["time_s"].to_numpy()
    n = len(df)
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
                dist = sum(haversine_m(lat[k-1], lon[k-1], lat[k], lon[k]) for k in range(i0+1, i1+1))
                v_mean = dist/dur if dur > 0 else np.nan
                omega = 2*math.pi/dur
                radius = (v_mean/omega) if omega > 0 else np.nan
                if np.isnan(radius) or radius < min_radius_m:
                    radius = np.nan
                bank = math.degrees(math.atan((v_mean**2)/(g*radius))) if (radius and np.isfinite(radius)) else np.nan
                alt_gain = alt[i1] - alt[i0] if (np.isfinite(alt[i1]) and np.isfinite(alt[i0])) else np.nan
                climb = alt_gain/dur if (dur > 0 and np.isfinite(alt_gain)) else np.nan
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
    print(circles_df.head())

if __name__ == "__main__":
    main()
