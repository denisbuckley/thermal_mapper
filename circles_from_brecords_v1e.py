#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------- FILE UTILS -------------------
def ensure_igc_in_run_dir(igc_src: Path, run_dir: Path, mode: str = "symlink") -> Path:
    """
    Ensure the IGC file is available inside run_dir.
    - mode="symlink": create a symlink (preferred)
    - mode="copy": copy the file
    Returns the path to the file inside run_dir.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    igc_src = igc_src.expanduser().resolve()
    if not igc_src.exists():
        raise FileNotFoundError(f"IGC source not found: {igc_src}")

    target = run_dir / igc_src.name
    # If already present (file or valid symlink), keep it
    if target.exists():
        return target

    if mode == "symlink":
        try:
            if target.is_symlink():
                target.unlink()
            target.symlink_to(igc_src)
            return target
        except (OSError, NotImplementedError):
            pass  # fall back to copy

    shutil.copy2(igc_src, target)
    return target


# ------------------- IGC PARSER -------------------
def parse_igc_brecords(path: Path) -> pd.DataFrame:
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

            # Pressure alt [25:30], GPS alt [30:35]
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
def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(phi2)
    x = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def unwrap_angles(angs: np.ndarray) -> np.ndarray:
    out = [angs[0]]
    for a in angs[1:]:
        prev = out[-1]
        da = a - (prev % 360.0)
        if da > 180:
            da -= 360
        if da <= -180:
            da += 360
        out.append(prev + da)
    return np.array(out, dtype=float)


# ------------------- CIRCLE DETECTOR -------------------
def detect_circles(
    df: pd.DataFrame,
    min_duration_s: float = 6.0,
    max_duration_s: float = 60.0,
    min_radius_m: float = 8.0,
    max_radius_m: float = 600.0,
    min_bank_deg: float = 5.0,
    vmax_climb_ms: float = 10.0,
    min_heading_change_deg: float = 360.0,
) -> pd.DataFrame:
    """
    Detect circles via cumulative heading rotation across B-fix bearings.
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
            "turn_radius_m","bank_angle_deg","bank_std_deg","bank_range_deg","bank_masd_deg",
            "lat","lon"
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
        if rot >= float(min_heading_change_deg):
            i0, i1 = start_idx, i
            dur = t[i1] - t[i0]

            # reject unrealistic "circles"
            if not (min_duration_s <= dur <= max_duration_s):
                start_idx = i
                i += 1
                continue

            # distance & mean ground speed
            dist = 0.0
            for k in range(i0 + 1, i1 + 1):
                dist += haversine_m(lat[k - 1], lon[k - 1], lat[k], lon[k])
            v_mean = dist / dur if dur > 0 else np.nan  # m/s

            # Angular rate and turn radius
            omega = 2 * math.pi / dur  # rad/s ~ one full turn
            radius = (v_mean / omega) if omega > 0 else np.nan
            if (np.isnan(radius)) or (radius < min_radius_m) or (radius > max_radius_m):
                start_idx = i
                i += 1
                continue

            bank = math.degrees(math.atan((v_mean ** 2) / (g * radius))) if (radius and np.isfinite(radius)) else np.nan
            if (np.isnan(bank)) or (bank < min_bank_deg):
                start_idx = i
                i += 1
                continue

            # --- Within-circle bank-angle variability metrics ---
            banks_deg = []
            for k in range(i0 + 1, i1 + 1):
                dt = t[k] - t[k - 1]
                if dt <= 0:
                    continue
                dtheta_rad = math.radians(uw[k] - uw[k - 1])  # unwrapped heading delta
                omega_ins = dtheta_rad / dt                    # rad/s
                seg_m = haversine_m(lat[k - 1], lon[k - 1], lat[k], lon[k])
                v = seg_m / dt                                 # m/s
                b_rad = math.atan((v * omega_ins) / g) if np.isfinite(v) and np.isfinite(omega_ins) else float("nan")
                if np.isfinite(b_rad):
                    banks_deg.append(math.degrees(b_rad))

            if banks_deg:
                b_arr = np.array(banks_deg, dtype=float)
                bank_std_deg_w = float(np.nanstd(b_arr, ddof=1)) if len(b_arr) >= 2 else 0.0
                bank_range_deg_w = float(np.nanmax(b_arr) - np.nanmin(b_arr))
                masd = float(np.nanmean(np.abs(np.diff(b_arr)))) if len(b_arr) >= 2 else 0.0
            else:
                bank_std_deg_w = np.nan
                bank_range_deg_w = np.nan
                masd = np.nan

            # Robust altitude gain by least-squares slope (reduces edge effects)
            idx = np.arange(i0, i1 + 1)
            tt = t[idx] - t[i0]
            aa = alt[idx]
            if len(tt) >= 3 and np.all(np.isfinite(tt)) and np.any(np.isfinite(aa)):
                A = np.vstack([tt, np.ones_like(tt)]).T
                mask = np.isfinite(aa)
                if mask.sum() >= 3:
                    m, _c = np.linalg.lstsq(A[mask], aa[mask], rcond=None)[0]
                    alt_gain = float(m * dur)
                else:
                    span = i1 - i0 + 1
                    q = max(1, span // 4)
                    first_med = np.nanmedian(aa[:q])
                    last_med = np.nanmedian(aa[-q:])
                    alt_gain = (last_med - first_med) if np.isfinite(first_med) and np.isfinite(last_med) else np.nan
            else:
                alt_gain = np.nan

            # Physical plausibility bound
            max_gain = vmax_climb_ms * dur
            if np.isfinite(alt_gain) and abs(alt_gain) > max_gain:
                alt_gain = np.sign(alt_gain) * max_gain

            climb = (alt_gain / dur) if (dur > 0 and np.isfinite(alt_gain)) else np.nan

            circles.append({
                "circle_id": int(circle_id),
                "seg_id": None,
                "t_start": float(t[i0]),
                "t_end": float(t[i1]),
                "duration_s": float(dur),
                "avg_speed_kmh": float(v_mean * 3.6) if np.isfinite(v_mean) else np.nan,
                "alt_gain_m": float(alt_gain) if np.isfinite(alt_gain) else np.nan,
                "climb_rate_ms": float(climb) if np.isfinite(climb) else np.nan,
                "turn_radius_m": float(radius) if np.isfinite(radius) else np.nan,
                "bank_angle_deg": float(bank) if np.isfinite(bank) else np.nan,
                "bank_std_deg": bank_std_deg_w,
                "bank_range_deg": bank_range_deg_w,
                "bank_masd_deg": masd,
                "lat": float(np.nanmean(lat[i0:i1 + 1])),
                "lon": float(np.nanmean(lon[i0:i1 + 1])),
            })
            circle_id += 1

            start_idx = i
        i += 1

    return pd.DataFrame(circles)


# ------------------- TOW-END DETECTOR -------------------
def detect_tow_end(track: pd.DataFrame,
                   H_ft: float = 2700.0,      # Height gate (ft)
                   T_s: float = 180.0,        # Time gate (s)
                   D_ft: float = 100.0,       # Descent gate (ft)
                   steady_s: float = 60.0) -> int:
    """
    Return index of last tow sample (inclusive). You should slice track.iloc[tow_idx+1:] for free flight.
    """
    if track.empty:
        return -1

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
        dalt = alt[i] - alt[i - 1]
        if dalt > 0:
            if climb_start_i is None:
                climb_start_i = i - 1
        else:
            climb_start_i = None

        if climb_start_i is not None and (t[i] - t[climb_start_i]) >= steady_s:
            run_peak = alt[i]
            for j in range(i + 1, len(alt)):
                if alt[j] > run_peak:
                    run_peak = alt[j]
                if run_peak - alt[j] >= D_m:
                    tow_idx_D = j
                    break
            break  # after enabling, we either found the drop or not

    candidates = [idx for idx in (tow_idx_H, tow_idx_T, tow_idx_D) if idx is not None]
    if not candidates:
        return -1
    return min(candidates)


# ------------------- TUNING -------------------
def load_tuning(path: Path = Path("tuning.json")) -> dict:
    try:
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
    return {}


# ------------------- MAIN -------------------
def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("igc", nargs="?", help="Path to IGC file")
    args = ap.parse_args()

    out_root = Path("outputs/batch_csv")
    from glob import glob  # add near the top of the file if not present

    # ... inside main(), replace everything from:
    #   last_file = Path(".last_igc.txt")
    #   # --- Load last used path if available; else hardcoded default ---
    #   ...
    #   # --- Resolve IGC path (arg or prompt) ---
    #   ...
    #   # --- Save as new default for next run ---
    #   ...
    # with the following:

    LAST_IGC_FILE = Path(".last_igc")  # keep this name consistent across tools

    # --- Build a sensible default hint: prefer valid .last_igc, else first file in ./igc ---
    try:
        candidates = sorted(glob("igc/*.igc")) + sorted(glob("igc/*.IGC"))
        fallback = Path(candidates[0]).resolve() if candidates else None
    except Exception:
        fallback = None

    default_hint = None
    if LAST_IGC_FILE.exists():
        try:
            prev = Path(LAST_IGC_FILE.read_text().strip())
            if prev.exists():
                default_hint = prev
        except Exception:
            pass
    if default_hint is None:
        default_hint = fallback

    # --- Resolve IGC path (arg wins; else prompt with hint if any) ---
    if args.igc:
        igc_path = Path(args.igc).expanduser().resolve()
    else:
        prompt = f"Enter IGC file path [{'default: ' + str(default_hint) if default_hint else 'no default'}]: "
        inp = input(prompt).strip()
        igc_path = Path(inp).expanduser().resolve() if inp else default_hint

    if igc_path is None or not igc_path.exists():
        raise FileNotFoundError(igc_path or "<none>")

    # --- Save absolute path for next run ---
    try:
        LAST_IGC_FILE.write_text(str(igc_path), encoding="utf-8")
    except Exception as e:
        print(f"[WARN] Could not save last IGC path: {e}")

    print(f"[INFO] using IGC: {igc_path}")

    # Per-flight run directory under outputs/batch_csv/<flight_basename>
    flight = igc_path.stem.rstrip(')').strip()
    run_dir = out_root / flight
    run_dir.mkdir(parents=True, exist_ok=True)

    # Ensure the IGC is present in the run_dir
    igc_local = ensure_igc_in_run_dir(igc_path, run_dir, mode="symlink")

    # --- Parse & pre-process ---
    track_df = parse_igc_brecords(igc_local)

    # --- Cut off aerotow segment ---
    tow_end_idx = detect_tow_end(track_df, H_ft=2700.0, T_s=180.0, D_ft=100.0, steady_s=60.0)
    if 0 <= tow_end_idx < len(track_df) - 1:
        track_df = track_df.iloc[tow_end_idx + 1:].reset_index(drop=True)
        print(f"[INFO] Tow cut at idx={tow_end_idx}, samples left={len(track_df)}")

    # ---- Load tuning (if present) ----
    params = load_tuning()

    # ---- Detect circles with tuning ----
    circles_df = detect_circles(
        track_df,
        min_duration_s=float(params.get("circle_min_duration_s", 6.0)),
        max_duration_s=float(params.get("circle_max_duration_s", 60.0)),
        min_radius_m=float(params.get("circle_min_radius_m", 8.0)),
        max_radius_m=float(params.get("circle_max_radius_m", 600.0)),
        min_bank_deg=float(params.get("circle_min_bank_deg", 5.0)),
        vmax_climb_ms=float(params.get("circle_vmax_climb_ms", 10.0)),
        # Optional legacy knob; not in your JSON today. Add if desired.
        min_heading_change_deg=float(params.get("min_total_heading_change", 360.0)),
    )

    # --- Console summary: count of circles ---
    print(f"[INFO] Detected {len(circles_df)} circles")

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

    # Canonical order
    canonical = [
        "lat", "lon", "t_start", "t_end", "climb_rate_ms", "alt_gain_m", "duration_s",
        "circle_id", "seg_id", "avg_speed_kmh", "turn_radius_m", "circle_diameter_m",
        "bank_angle_deg", "bank_std_deg", "bank_range_deg", "bank_masd_deg"
    ]
    extras = [c for c in d.columns if c not in canonical]
    d = d[[c for c in canonical if c in d.columns] + extras]

    # Write circles.csv in run_dir
    out_csv = run_dir / "circles.csv"
    d.to_csv(out_csv, index=False)
    print(f"[OK] wrote {len(d)} circles → {out_csv}")

if __name__ == "__main__":
    raise SystemExit(main())

