#!/usr/bin/env python3
"""
filter_thermals_along_leg_strength_labels.py
--------------------------------------------
Filters thermal waypoints to those near a selected flight leg (start→end),
within a widening cone. Labels exported points by numeric strength (e.g., "2.8").

Outputs written to: outputs/waypoints/
"""

import csv
import math
import re
from pathlib import Path

# --- Configuration ---
WAYPOINT_FILE     = "gcwa extended.cup"
INPUT_FILE        = "consolidated_thermal_coords.csv"
FALLBACK_WP_FILE  = "outputs/waypoints/thermal_waypoints_v1.csv"

OUTDIR            = Path("outputs/waypoints")
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV_FILE   = OUTDIR / "filtered_thermals.csv"
OUTPUT_CUP_FILE   = OUTDIR / "filtered_thermals.cup"
OUTPUT_KML_FILE   = OUTDIR / "filtered_thermals.kml"

EARTH_R_KM = 6371.0

# =========================
# Geometry helpers
# =========================
def deg2rad(d): return d * math.pi / 180.0
def rad2deg(r): return r * 180.0 / math.pi

def haversine_km(lat1, lon1, lat2, lon2):
    φ1, λ1, φ2, λ2 = map(deg2rad, [lat1, lon1, lat2, lon2])
    dφ, dλ = φ2 - φ1, λ2 - λ1
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return 2 * EARTH_R_KM * math.asin(min(1.0, math.sqrt(a)))

def initial_bearing_deg(lat1, lon1, lat2, lon2):
    φ1, λ1, φ2, λ2 = map(deg2rad, [lat1, lon1, lat2, lon2])
    dλ = λ2 - λ1
    y = math.sin(dλ) * math.cos(φ2)
    x = math.cos(φ1)*math.sin(φ2) - math.sin(φ1)*math.cos(φ2)*math.cos(dλ)
    return (rad2deg(math.atan2(y, x)) + 360) % 360

def bearing_diff_abs(a, b):
    return abs((a - b + 540) % 360 - 180)

def cross_track_along_track_km(lat1, lon1, lat2, lon2, latp, lonp):
    d_ab = haversine_km(lat1, lon1, lat2, lon2) / EARTH_R_KM
    d_ap = haversine_km(lat1, lon1, latp, lonp) / EARTH_R_KM
    if d_ab == 0:
        return haversine_km(lat1, lon1, latp, lonp), 0.0
    br_ab = deg2rad(initial_bearing_deg(lat1, lon1, lat2, lon2))
    br_ap = deg2rad(initial_bearing_deg(lat1, lon1, latp, lonp))
    xt_ang = math.asin(max(-1, min(1, math.sin(d_ap) * math.sin(br_ap - br_ab))))
    xtrack_km = abs(xt_ang) * EARTH_R_KM
    cos_at = max(-1, min(1, math.cos(d_ap) / max(1e-12, math.cos(xt_ang))))
    at_ang = math.acos(cos_at)
    along_km = at_ang * EARTH_R_KM
    return xtrack_km, along_km

# =========================
# CUP coordinate parsing
# =========================
def parse_cup_coord(coord: str) -> float:
    s = (coord or "").strip().replace("°", ":").replace("º", ":").replace(" ", "")
    # S31:39.123
    m = re.match(r'^([NSEW])(\d{1,3}):(\d{1,2}(?:\.\d+)?)$', s, re.I)
    if m:
        hemi, degs, mins = m.group(1).upper(), int(m.group(2)), float(m.group(3))
        val = degs + mins/60
        return -val if hemi in ('S','W') else val
    # 3222.447S
    m = re.match(r'^([NSEW])?(\d{2,3})(\d{2}(?:\.\d+)?)([NSEW])?$', s, re.I)
    if m:
        hemi1, degs, mins, hemi2 = (m.group(1) or "").upper(), int(m.group(2)), float(m.group(3)), (m.group(4) or "").upper()
        hemi = hemi1 or hemi2
        val = degs + mins/60
        return -abs(val) if hemi in ('S','W') else abs(val)
    # Decimal
    return float(s)

def convert_to_cup_coord(value: float, is_lat: bool) -> str:
    hemi = "N" if is_lat and value >= 0 else "S" if is_lat else "E" if value >= 0 else "W"
    v = abs(value)
    deg = int(v)
    mins = (v - deg) * 60
    if is_lat:
        return f"{deg:02d}{mins:06.3f}{hemi}"
    else:
        return f"{deg:03d}{mins:06.3f}{hemi}"

# =========================
# Filtering
# =========================
def is_within_cone_and_corridor(lat, lon, s_lat, s_lon, e_lat, e_lon, cone_deg, tol_km):
    leg_bearing = initial_bearing_deg(s_lat, s_lon, e_lat, e_lon)
    thr_bearing = initial_bearing_deg(s_lat, s_lon, lat, lon)
    if bearing_diff_abs(leg_bearing, thr_bearing) > (cone_deg/2):
        return False
    xtrack, along = cross_track_along_track_km(s_lat, s_lon, e_lat, e_lon, lat, lon)
    leg_len = haversine_km(s_lat, s_lon, e_lat, e_lon)
    return 0 <= along <= leg_len and xtrack <= tol_km

# =========================
# Writers
# =========================
def write_cup_file(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Title","Code","Country","Latitude","Longitude","Elevation",
                    "Style","RWY Dir","RWY Len","Freq","Desc"])
        for i, r in enumerate(rows, 1):
            label = f"{float(r['strength']):.1f}"  # number only
            code = f"STR{i:03d}"
            w.writerow([
                label, code, "",
                convert_to_cup_coord(r["lat"], True),
                convert_to_cup_coord(r["lon"], False),
                "0ft","1","","","",""
            ])

def write_kml_file(rows, path):
    with open(path, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<kml xmlns="http://www.opengis.net/kml/2.2"><Document>\n')
        for r in rows:
            label = f"{float(r['strength']):.1f}"
            f.write(f'  <Placemark><name>{label}</name>')
            f.write(f'<Point><coordinates>{r["lon"]},{r["lat"]},0</coordinates></Point></Placemark>\n')
        f.write('</Document></kml>\n')

# =========================
# Main
# =========================
def main():
    print(f"Reading waypoints from '{WAYPOINT_FILE}'...")
    waypoints = []
    with open(WAYPOINT_FILE, newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            try:
                lat = parse_cup_coord(r["Latitude"])
                lon = parse_cup_coord(r["Longitude"])
                waypoints.append({"name": r["Title"], "lat": lat, "lon": lon})
            except Exception:
                continue
    if not waypoints:
        print("No usable waypoints found.")
        return 1

    print("\nAvailable Waypoints:")
    for i, wp in enumerate(waypoints, 1):
        print(f"[{i}] {wp['name']}")

    def choose(prompt):
        while True:
            try:
                n = int(input(f"Enter {prompt} waypoint number: "))
                if 1 <= n <= len(waypoints):
                    return waypoints[n-1]
            except Exception:
                pass
            print("Invalid choice.")

    start = choose("start")
    end   = choose("end")

    cone_angle = float(input("Enter cone angle (deg): ") or "30")
    total_dist = haversine_km(start["lat"], start["lon"], end["lat"], end["lon"])
    tolerance  = (total_dist/2) * math.tan(math.radians(cone_angle/2))

    print(f"\nFiltering thermals with cone {cone_angle}°, tolerance {tolerance:.1f} km")

    infile = Path(INPUT_FILE)
    if not infile.exists():
        infile = Path(FALLBACK_WP_FILE)

    filtered = []
    with infile.open(newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                lat = float(row["lat"]); lon = float(row["lon"])
                strength = float(row.get("strength") or row.get("strength_mean_core") or 0)
                if is_within_cone_and_corridor(lat, lon,
                                               start["lat"], start["lon"],
                                               end["lat"], end["lon"],
                                               cone_angle, tolerance):
                    filtered.append({"lat": lat, "lon": lon, "strength": strength})
            except Exception:
                continue

    if not filtered:
        print("No thermals found in range.")
        return 0

    # Write outputs
    with open(OUTPUT_CSV_FILE, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["lat","lon","strength"])
        for r in filtered: w.writerow([r["lat"], r["lon"], r["strength"]])

    write_cup_file(filtered, OUTPUT_CUP_FILE)
    write_kml_file(filtered, OUTPUT_KML_FILE)

    print(f"\nWrote {len(filtered)} thermals →")
    print(f"  {OUTPUT_CSV_FILE}")
    print(f"  {OUTPUT_CUP_FILE}")
    print(f"  {OUTPUT_KML_FILE}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())