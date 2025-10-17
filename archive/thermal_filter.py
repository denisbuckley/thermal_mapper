#!/usr/bin/env python3
"""
thermal_filter.py
-----------------
Filter thermal waypoints that are consistent with a multi-leg flight (return or task)
using a diamond sector per leg.

User picks 2+ waypoints from a .CUP list. Keep pressing ENTER on an empty line to finish.
For each consecutive pair (leg) we apply:

1) Angular gate (cone from START of the leg):
   |bearing(start→thermal) − bearing(start→end)| <= cone_half_angle.

2) Flown-distance gate (diamond bound):
   flown_distance(start→thermal→end) <= L / cos(cone_half_angle),
   where L is great-circle distance start→end.

If a thermal passes ANY leg’s gates, it’s kept.

Inputs:
  - Waypoints   : "gcwa extended.cup"
  - Thermals CSV: "outputs/waypoints/thermal_waypoints_v1.csv"
                  (uses strength_mean_core if present, else strength_mean_all)

Outputs (in outputs/waypoints/):
  - filtered_thermals.csv  (lat,lon,strength)
  - filtered_thermals.cup  (label = strength with 1 decimal, no unit)
  - filtered_thermals.kml  (labels as above)
"""

import csv
import math
from pathlib import Path

# ------------------------- small geo helpers -------------------------

EARTH_R_KM = 6371.0

def haversine_km(lat1, lon1, lat2, lon2):
    r = EARTH_R_KM
    φ1, λ1, φ2, λ2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dφ = φ2 - φ1
    dλ = λ2 - λ1
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return 2*r*math.asin(min(1.0, math.sqrt(a)))

def bearing_deg(lat1, lon1, lat2, lon2):
    φ1, λ1, φ2, λ2 = map(math.radians, [lat1, lon1, lat2, lon2])
    y = math.sin(λ2-λ1)*math.cos(φ2)
    x = math.cos(φ1)*math.sin(φ2) - math.sin(φ1)*math.cos(φ2)*math.cos(λ2-λ1)
    θ = math.degrees(math.atan2(y, x))
    return (θ + 360.0) % 360.0

def bearing_delta(a, b):
    """smallest absolute difference of two bearings in degrees"""
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d

# ------------------------- CUP parsing helpers -------------------------

def parse_cup_coord(coord: str) -> float:
    """
    Parse DDMM.mmmH or DDDMM.mmmH where H is N/S/E/W.
    Example: 3222.447S -> -32.3741167 ; 11716.745E -> 117.2790833
    """
    coord = coord.strip()
    if not coord:
        raise ValueError("empty CUP coord")
    hemi = coord[-1].upper()
    sign = -1 if hemi in ("S", "W") else 1
    body = coord[:-1]
    if "." not in body:
        raise ValueError(f"bad CUP coord: {coord}")
    head, frac = body.split(".", 1)
    # lat has 2 deg digits, lon has 3
    if len(head) in (4, 5):  # 2 deg + 2 min, or 3 deg + 2 min
        if len(head) == 4:  # latitude
            deg = int(head[:2]); mins = int(head[2:])
        else:               # longitude
            deg = int(head[:3]); mins = int(head[3:])
    else:
        raise ValueError(f"bad CUP deg/min: {coord}")
    mins = float(f"{mins}.{frac}")
    val = sign * (deg + mins/60.0)
    return val

def read_waypoints_from_cup(path: Path):
    out = []
    if not path.exists():
        print(f"[WARN] Waypoint file not found: {path}")
        return out
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header
        for row in reader:
            try:
                # Title,Code,Country,Latitude,Longitude,Elevation,Style,...
                name = row[0].strip().strip('"')
                code = row[1].strip().strip('"')
                lat = parse_cup_coord(row[3].strip())
                lon = parse_cup_coord(row[4].strip())
                out.append({"name": name, "code": code, "lat": lat, "lon": lon})
            except Exception:
                continue
    return out

# ------------------------- diamond gate -------------------------

def keep_point_diamond(lat, lon, s_lat, s_lon, e_lat, e_lon, cone_half_deg):
    """
    True if point is inside the angular gate AND satisfies the flown-distance bound.
    """
    L = haversine_km(s_lat, s_lon, e_lat, e_lon)
    if L == 0.0:
        return False

    # Angular gate
    b_leg = bearing_deg(s_lat, s_lon, e_lat, e_lon)
    b_pt  = bearing_deg(s_lat, s_lon, lat, lon)
    if bearing_delta(b_leg, b_pt) > cone_half_deg:
        return False

    # Flown-distance (diamond) gate
    flown = haversine_km(s_lat, s_lon, lat, lon) + haversine_km(lat, lon, e_lat, e_lon)
    max_flown = L / max(1e-9, math.cos(math.radians(cone_half_deg)))
    return flown <= max_flown + 1e-6

# ------------------------- writers -------------------------

OUT_DIR = Path("../outputs/waypoints")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def write_csv(rows, path: Path):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lat","lon","strength"])
        for r in rows:
            w.writerow([f"{r['lat']:.7f}", f"{r['lon']:.7f}", f"{r['strength']:.1f}"])

def convert_to_cup_coord(val: float, is_lat: bool) -> str:
    sign = 'N' if (is_lat and val>=0) else 'S' if is_lat else ('E' if val>=0 else 'W')
    v = abs(val)
    deg = int(v)
    mins = (v - deg) * 60.0
    if is_lat:
        return f"{deg:02d}{mins:06.3f}{sign}"
    else:
        return f"{deg:03d}{mins:06.3f}{sign}"

def write_cup(rows, path: Path):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Title","Code","Country","Latitude","Longitude","Elevation","Style","Direction","Length","Frequency","Description"])
        for i, r in enumerate(rows, 1):
            title = f"{r['strength']:.1f}"
            code  = f"T{i:03d}"
            la = convert_to_cup_coord(r["lat"], True)
            lo = convert_to_cup_coord(r["lon"], False)
            w.writerow([title, code, "AU", la, lo, "0ft", "1", "", "", "", "thermal"])

def write_kml(rows, path: Path):
    from xml.sax.saxutils import escape
    def placemark(r):
        name = f"{r['strength']:.1f}"
        return f"""    <Placemark>
      <name>{escape(name)}</name>
      <Point><coordinates>{r['lon']:.7f},{r['lat']:.7f},0</coordinates></Point>
    </Placemark>"""
    kml = ["<?xml version='1.0' encoding='UTF-8'?>",
           "<kml xmlns='http://www.opengis.net/kml/2.2'>",
           "  <Document>"]
    kml.extend(placemark(r) for r in rows)
    kml.append("  </Document>\n</kml>")
    path.write_text("\n".join(kml), encoding="utf-8")

# ------------------------- input helpers -------------------------

def get_float_input(prompt: str, default: float):
    s = input(f"{prompt} [{default}]: ").strip()
    if not s:
        return float(default)
    try:
        return float(s)
    except ValueError:
        print("Invalid number; using default.")
        return float(default)

# ------------------------- main -------------------------

def main():
    WAYPOINT_FILE = Path("../gcwa extended.cup")
    THERMALS_FILE = Path("../outputs/waypoints/thermal_waypoints_v1.csv")

    print(f"Reading waypoints from '{WAYPOINT_FILE}'...")
    wps = read_waypoints_from_cup(WAYPOINT_FILE)
    if not wps:
        print("[WARN] No usable waypoints found; aborting.")
        return 1

    # Show a compact list
    print("\nAvailable Waypoints:")
    for i, wp in enumerate(wps, 1):
        print(f"[{i:3d}] {wp['name']} ({wp['code']})  {wp['lat']:.5f},{wp['lon']:.5f}")

    # Multi-select loop: user enters numbers; ENTER on blank to finish
    print("\nEnter one waypoint number per line to build your route.")
    print("Press ENTER on an empty line to finish (need at least 2).")
    chosen = []
    while True:
        s = input("Waypoint #: ").strip()
        if s == "":
            break
        try:
            idx = int(s)
            if 1 <= idx <= len(wps):
                chosen.append(wps[idx-1])
                print(f"  added: {wps[idx-1]['name']}")
            else:
                print("  out of range.")
        except ValueError:
            print("  not a number.")

    if len(chosen) < 2:
        print("Need at least TWO waypoints. Nothing to do.")
        return 1

    # Cone half-angle once for all legs
    cone_half = get_float_input("Cone half-angle (degrees)", 25.0)

    # Load thermals (prefer strength_mean_core, fallback to strength_mean_all)
    if not THERMALS_FILE.exists():
        print(f"[ERR] Thermals file not found: {THERMALS_FILE}")
        return 1

    thermals = []
    with THERMALS_FILE.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row["lat"]); lon = float(row["lon"])
                if "strength_mean_core" in row and row["strength_mean_core"]:
                    strength = float(row["strength_mean_core"])
                elif "strength_mean_all" in row and row["strength_mean_all"]:
                    strength = float(row["strength_mean_all"])
                elif "strength_mean" in row and row["strength_mean"]:
                    strength = float(row["strength_mean"])
                elif "climb_rate_ms" in row and row["climb_rate_ms"]:
                    strength = float(row["climb_rate_ms"])
                else:
                    continue
                thermals.append({"lat": lat, "lon": lon, "strength": strength})
            except Exception:
                continue

    if not thermals:
        print("[INFO] No thermals loaded; nothing to filter.")
        return 0

    # Build legs
    legs = list(zip(chosen[:-1], chosen[1:]))
    leg_desc = " → ".join(w["name"] for w in chosen)
    print(f"\nFiltering thermals for route: {leg_desc}")
    print(f"Cone half-angle: {cone_half:.1f}°  (legs: {len(legs)})")

    kept = []
    for th in thermals:
        lat, lon = th["lat"], th["lon"]
        # keep if passes ANY leg
        for a, b in legs:
            if keep_point_diamond(lat, lon, a["lat"], a["lon"], b["lat"], b["lon"], cone_half):
                kept.append(th)
                break

    print(f"Kept {len(kept)} / {len(thermals)} ({(100.0*len(kept)/max(1,len(thermals))):.1f}%).")

    # Write outputs
    out_csv = OUT_DIR / "filtered_thermals.csv"
    out_cup = OUT_DIR / "filtered_thermals.cup"
    out_kml = OUT_DIR / "filtered_thermals.kml"
    write_csv(kept, out_csv)
    write_cup(kept, out_cup)
    write_kml(kept, out_kml)

    print("Outputs:")
    print(f"  {out_csv}")
    print(f"  {out_cup}")
    print(f"  {out_kml}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())