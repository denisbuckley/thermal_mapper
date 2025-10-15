#!/usr/bin/env python3
"""
thermal_filter.py
-----------------
Filter thermal waypoints that are consistent with a flight leg using a
**diamond-shaped sector** (narrow at the ends, widest at the midpoint).

Geometry:
  - Let L = great-circle distance start→end.
  - At fractional position f along the leg (0 at start, 1 at end),
    the maximum lateral deviation allowed is:

        δ_allowed(f) = min(f, 1 - f) * L * tan(cone_half_angle)

    This forms a diamond/lens sector: zero deviation at start and end,
    maximum deviation (L/2 * tan(cone_half_angle)) at the midpoint.

Filter rule:
  - Project each thermal onto the leg SE.
  - Keep it if cross-track distance ≤ δ_allowed(f) and 0 ≤ f ≤ 1.

Outputs:
  - CSV:  outputs/waypoints/filtered_thermals.csv
  - CUP:  outputs/waypoints/filtered_thermals.cup
  - KML:  outputs/waypoints/filtered_thermals.kml

Labels in CUP/KML are numeric strength with 1 decimal (e.g. "2.8").

Waypoints source:
  - "gcwa extended.cup" (supports DDMM.mmmH like 3222.447S,11716.745E).

Thermals source (auto):
  - Primary: "consolidated_thermal_coords.csv" (lat,lon,strength)
  - Fallback: "outputs/waypoints/thermal_waypoints_v1.csv"
              (prefers strength_mean_core; else any strength_*; else strength)
"""

import csv
import math
import re
from pathlib import Path
from typing import List, Dict, Tuple

# ---------- Paths ----------
WAYPOINT_FILE       = "gcwa extended.cup"
THERMALS_FILE       = "outputs/waypoints/thermal_waypoints_v1.csv"  # primary & only

OUTDIR              = Path("outputs/waypoints")
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV_FILE     = OUTDIR / "filtered_thermals.csv"
OUTPUT_CUP_FILE     = OUTDIR / "filtered_thermals.cup"
OUTPUT_KML_FILE     = OUTDIR / "filtered_thermals.kml"

EARTH_R_KM = 6371.0

# =========================
# Geometry helpers
# =========================
def deg2rad(d): return d * math.pi / 180.0
def rad2deg(r): return r * 180.0 / math.pi

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    φ1, λ1, φ2, λ2 = map(deg2rad, [lat1, lon1, lat2, lon2])
    dφ, dλ = φ2 - φ1, λ2 - λ1
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return 2 * EARTH_R_KM * math.asin(min(1.0, math.sqrt(a)))

def initial_bearing_deg(lat1, lon1, lat2, lon2) -> float:
    φ1, λ1, φ2, λ2 = map(deg2rad, [lat1, lon1, lat2, lon2])
    dλ = λ2 - λ1
    y = math.sin(dλ) * math.cos(φ2)
    x = math.cos(φ1)*math.sin(φ2) - math.sin(φ1)*math.cos(φ2)*math.cos(dλ)
    return (rad2deg(math.atan2(y, x)) + 360.0) % 360.0

def cross_track_along_track_km(s_lat, s_lon, e_lat, e_lon, p_lat, p_lon):
    """
    Robust local-plane projection:
      - Convert lon/lat deltas to km in a local ENU frame anchored near the leg.
      - Return (cross_km, along_km) where along is signed:
          along < 0   → point is before START
          along > L   → point is beyond END
    """
    # Local scale factors (km per degree); anchor at mid-lat to minimize distortion
    lat0_rad = math.radians((s_lat + e_lat) * 0.5)
    kx = math.cos(lat0_rad) * 111.320  # km / deg longitude
    ky = 110.574                       # km / deg latitude

    # Vector from START to END in ENU km
    ax = (e_lon - s_lon) * kx
    ay = (e_lat - s_lat) * ky
    L  = math.hypot(ax, ay)
    if L == 0:
        # Degenerate leg: cross is straight-line distance from START
        px = (p_lon - s_lon) * kx
        py = (p_lat - s_lat) * ky
        return math.hypot(px, py), 0.0

    # Unit vector along the leg
    ux, uy = ax / L, ay / L

    # Vector from START to point P in ENU km
    px = (p_lon - s_lon) * kx
    py = (p_lat - s_lat) * ky

    # Signed along-track (km) via dot product, and cross-track magnitude (km)
    along = px * ux + py * uy
    cx = px - along * ux
    cy = py - along * uy
    cross = math.hypot(cx, cy)
    return cross, along
# =========================
# CUP parsing + waypoint reader
# =========================
def parse_cup_coord(coord: str) -> float:
    """
    Parse CUP/coordinate strings to signed decimal degrees.

    Accepts:
      - 'S31:39.123' / 'E116:52.345'           (CUP with colon)
      - '3222.447S' / '11716.745E'             (CUP, no colon, hemi suffix) or 'S3222.447' (prefix)
      - Decimal with optional hemisphere: 'S31.654' / '116.875E' / '-31.654'
    """
    s = (coord or "").strip()
    if not s:
        raise ValueError("empty coord")
    s = s.replace("°", ":").replace("º", ":").replace(" ", "")

    # (1) CUP with colon
    m = re.match(r'^([NSEW])(\d{1,3}):(\d{1,2}(?:\.\d+)?)$', s, re.I)
    if m:
        hemi, degs, mins = m.group(1).upper(), int(m.group(2)), float(m.group(3))
        val = degs + mins/60.0
        return -val if hemi in ('S','W') else val

    # (2) No colon, hemi prefix/suffix, minutes embedded
    m = re.match(r'^([NSEW])?(\d{2,3})(\d{2}(?:\.\d+)?)([NSEW])?$', s, re.I)
    if m:
        hemi1, degs, mins, hemi2 = (m.group(1) or "").upper(), int(m.group(2)), float(m.group(3)), (m.group(4) or "").upper()
        hemi = hemi1 or hemi2
        val = degs + mins/60.0
        return -abs(val) if hemi in ('S','W') else abs(val)

    # (3) Decimal with optional hemi
    m = re.match(r'^([NSEW])?([+-]?\d+(?:\.\d+)?)([NSEW])?$', s, re.I)
    if m:
        hemi1, num, hemi2 = (m.group(1) or "").upper(), float(m.group(2)), (m.group(3) or "").upper()
        hemi = hemi1 or hemi2
        val = num
        if hemi in ('S','W'):
            val = -abs(val)
        elif hemi in ('N','E'):
            val = abs(val)
        return val

    return float(s)

def convert_to_cup_coord(value: float, is_lat: bool) -> str:
    hemi = ("N" if value >= 0 else "S") if is_lat else ("E" if value >= 0 else "W")
    v = abs(value)
    deg = int(v)
    mins = (v - deg) * 60.0
    if is_lat:
        return f"{deg:02d}{mins:06.3f}{hemi}"
    else:
        return f"{deg:03d}{mins:06.3f}{hemi}"

def read_waypoints_from_cup(path: str) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] Waypoint file not found: {p}")
        return []
    rows = []
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        rdr = csv.DictReader((ln for ln in f if not ln.lstrip().startswith("*") and ln.strip()))
        if not rdr.fieldnames:
            print(f"[WARN] No headers in {p.name}")
            return []
        for r in rdr:
            try:
                name = (r.get("Title") or r.get("Name") or "").strip().strip('"')
                code = (r.get("Code")  or name).strip().strip('"')
                lat  = parse_cup_coord(r["Latitude"])
                lon  = parse_cup_coord(r["Longitude"])
                rows.append({"name": name or "WP", "code": code, "lat": lat, "lon": lon})
            except Exception:
                continue
    if not rows:
        print(f"[WARN] No usable waypoints found in {p.name}")
    else:
        print(f"[INFO] Loaded {len(rows)} waypoints from {p.name}")
    return rows

# =========================
# Thermals reader
# =========================
def read_thermals(thermals_path: str) -> List[Dict]:
    """
    Read thermals from thermal_waypoints_v1.csv.
    Returns [{lat, lon, strength}], preferring strength_mean_core,
    else any strength_* column, else 'strength' if present.
    """
    p = Path(thermals_path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {thermals_path}")

    out: List[Dict] = []
    with p.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        header = [h.strip() for h in (rdr.fieldnames or [])]

        # choose best strength column
        if "strength_mean_core" in header:
            prefer = "strength_mean_core"
        else:
            cand = [c for c in header if c.lower().startswith("strength_")]
            prefer = cand[0] if cand else ("strength" if "strength" in header else None)

        for row in rdr:
            try:
                lat = float(str(row["lat"]).strip().strip('"'))
                lon = float(str(row["lon"]).strip().strip('"'))
                if prefer and row.get(prefer, "") != "":
                    strength = float(str(row[prefer]).strip().strip('"'))
                else:
                    # if no strength column, skip
                    continue
                out.append({"lat": lat, "lon": lon, "strength": strength})
            except Exception:
                continue
    return out

# =========================
# Writers (numeric strength labels)
# =========================
def write_cup_points(rows: List[Dict], path: Path):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Title","Code","Country","Latitude","Longitude","Elevation",
                    "Style","RWY Dir","RWY Len","Freq","Desc"])
        for i, r in enumerate(rows, 1):
            label = f"{float(r['strength']):.1f}"  # number only
            code  = f"STR{i:03d}"
            w.writerow([
                label, code, "",
                convert_to_cup_coord(r["lat"], True),
                convert_to_cup_coord(r["lon"], False),
                "0ft","1","","","",""
            ])

def write_kml_points(rows: List[Dict], path: Path):
    with path.open("w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<kml xmlns="http://www.opengis.net/kml/2.2"><Document>\n')
        for r in rows:
            label = f"{float(r['strength']):.1f}"
            f.write(f'  <Placemark><name>{label}</name>')
            f.write(f'<Point><coordinates>{r["lon"]},{r["lat"]},0</coordinates></Point></Placemark>\n')
        f.write('</Document></kml>\n')

# =========================
# Input helpers
# =========================
def get_float_input(prompt: str, default: float) -> float:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except Exception:
            print("Please enter a valid number.")

# =========================
# Diamond keeper
# =========================
def keep_point_diamond(lat, lon, s_lat, s_lon, e_lat, e_lon, cone_deg):
    """
    Diamond/lens sector filter.
      - L = |SE|
      - f = along/L
      - allowed lateral = min(f, 1-f) * L * tan(alpha), alpha = cone_deg/2
    Keep iff 0 <= along <= L and cross_track <= allowed.
    """
    L = haversine_km(s_lat, s_lon, e_lat, e_lon)
    if L <= 0:
        return False

    xtrack_km, along_km = cross_track_along_track_km(s_lat, s_lon, e_lat, e_lon, lat, lon)
    if along_km < 0 or along_km > L:
        return False

    alpha = math.radians(cone_deg / 2.0)
    allowed_km = min(along_km, L - along_km) * math.tan(alpha)
    return xtrack_km <= (allowed_km + 1e-6)

# =========================
# Main
# =========================
def main():
    print(f"Reading waypoints from '{WAYPOINT_FILE}'...")
    waypoints = read_waypoints_from_cup(WAYPOINT_FILE)
    if not waypoints:
        print(f"[WARN] No usable waypoints found in {WAYPOINT_FILE}")
        return 1

    print("\nAvailable Waypoints:")
    for i, wp in enumerate(waypoints, 1):
        code = f" ({wp['code']})" if 'code' in wp and wp['code'] else ""
        print(f"[{i}] {wp['name']}{code}")

    def choose(label):
        while True:
            try:
                n = int(input(f"Enter {label} waypoint number: "))
                if 1 <= n <= len(waypoints):
                    return waypoints[n-1]
            except Exception:
                pass
            print(f"Enter a number between 1 and {len(waypoints)}.")

    start = choose("start")
    end   = choose("end")

    cone_angle = get_float_input("Enter the cone angle (deg)", 30.0)

    print(f"\nFiltering thermals for leg '{start['name']}' → '{end['name']}'")
    print(f"Cone angle: {cone_angle:.1f}°")

    # Load thermals
    try:
        thermals = read_thermals(THERMALS_FILE)
        print(f"[INFO] Using {THERMALS_FILE}")
    except FileNotFoundError as e:
        print(str(e))
        return 1

    print(f"Loaded {len(thermals)} thermals. Filtering…")
    kept: List[Dict] = []
    for r in thermals:
        if keep_point_diamond(r["lat"], r["lon"],
                              start["lat"], start["lon"],
                              end["lat"],   end["lon"],
                              cone_angle):
            kept.append(r)

    print(f"Kept {len(kept)} / {len(thermals)} ({(len(kept)/max(1,len(thermals)))*100:.1f}%).")

    # Always write CSV (empty if none kept)
    with OUTPUT_CSV_FILE.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lat","lon","strength"])
        for r in kept:
            w.writerow([r["lat"], r["lon"], r["strength"]])

    if kept:
        write_cup_points(kept, OUTPUT_CUP_FILE)
        write_kml_points(kept, OUTPUT_KML_FILE)
        print("Outputs:")
        print(f"  {OUTPUT_CSV_FILE}")
        print(f"  {OUTPUT_CUP_FILE}")
        print(f"  {OUTPUT_KML_FILE}")
    else:
        print(f"Wrote empty {OUTPUT_CSV_FILE}. (No CUP/KML created.)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())