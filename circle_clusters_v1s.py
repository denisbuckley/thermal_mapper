#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import math

def detect_circles(df, lat, lon, alt, t):
    circles = []
    circle_id = 0
    # Placeholder logic: replace with your actual circle detection loop
    # For now, simulate a few circles for demonstration
    for i in range(0, len(df), 200):
        j = min(i+200, len(df)-1)
        dur_s = t[j] - t[i]
        if dur_s <= 0: 
            continue
        alt_gain = alt[j] - alt[i]
        climb = alt_gain / dur_s if dur_s > 0 else np.nan
        dist = np.hypot(lat[j]-lat[i], lon[j]-lon[i]) * 111000
        speed = dist / dur_s * 3.6
        # Assume simple constant radius-turn relation
        radius = (speed/3.6)**2 / (9.81 * math.tan(math.radians(30)))  # assuming 30° bank if no better
        bank = math.degrees(math.atan((speed/3.6)**2/(9.81*radius))) if radius>0 else np.nan
        circles.append({
            "circle_id": circle_id,
            "seg_id": None,
            "t_start": t[i],
            "t_end": t[j],
            "duration_s": dur_s,
            "avg_speed_kmh": speed,
            "alt_gained_m": alt_gain,
            "climb_rate_ms": climb,
            "turn_radius_m": radius,
            "bank_angle_deg": bank,
            "lat": float(np.mean(lat[i:j+1])),
            "lon": float(np.mean(lon[i:j+1])),
        })
        circle_id += 1
    return pd.DataFrame(circles)

def main():
    # Placeholder for reading IGC and detecting circles
    n = 2000
    t = np.arange(n)
    lat = np.linspace(-31.6, -31.8, n)
    lon = np.linspace(117.2, 117.3, n)
    alt = np.linspace(300, 1200, n) + np.sin(np.linspace(0, 20, n))*50
    df = pd.DataFrame({"t": t, "lat": lat, "lon": lon, "alt": alt})
    circles_df = detect_circles(df, df['lat'].to_numpy(), df['lon'].to_numpy(), df['alt'].to_numpy(), df['t'].to_numpy())
    import os
    os.makedirs("outputs", exist_ok=True)
    circles_df.to_csv("outputs/circles.csv", index=False)
    print("Wrote outputs/circles.csv with", len(circles_df), "circles")

if __name__ == "__main__":
    main()
