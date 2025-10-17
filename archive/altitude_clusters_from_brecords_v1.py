#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Altitude-only climb cluster detector (standalone)
-------------------------------------------------
- Parses IGC B-records (time, lat, lon, alt)
- Smooths altitude (median -> MAD clip -> rolling mean)
- Optional tow cut with H/T/D/steady gates
- Detects *positive-altitude* segments that exceed tuning thresholds
- Writes altitude_clusters.csv to outputs/batch_csv/<flight>/
- Remembers last IGC path in .last_igc.txt for convenience
- Reads tuning.json for:
    - alt_min_gain (m)
    - alt_min_duration (s)
a
CSV schema:
    cluster_id, lat, lon, t_start, t_end, climb_rate_ms, alt_gain_m, duration_s
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Optional

import numpy as np
import pandas as pd


# ------------------- Paths -------------------
ROOT = Path.cwd()
OUT_ROOT = ROOT / "outputs" / "batch_csv"
LAST_IGC_FILE = ROOT / ".last_igc.txt"   # shared with circles script for consistency
TUNING_FILE = ROOT / "tuning.json"


# ------------------- Small utils -------------------
def _read_last_igc() -> Optional[Path]:
    try:
        p = Path(LAST_IGC_FILE.read_text(encoding="utf-8").strip())
        return p if p.exists() else None
    except Exception:
        return None


def _write_last_igc(p: Path) -> None:
    try:
        LAST_IGC_FILE.write_text(str(p.resolve()), encoding="utf-8")
    except Exception:
        pass


def load_tuning() -> dict:
    defaults = {
        "alt_min_gain": 30.0,
        "alt_min_duration": 20.0,
    }
    if TUNING_FILE.exists():
        try:
            cfg = json.loads(TUNING_FILE.read_text(encoding="utf-8"))
            return {**defaults, **cfg}
        except Exception:
            return defaults
    return defaults


def ensure_igc_in_run_dir(igc_src: Path, run_dir: Path, mode: str = "symlink") -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    igc_src = igc_src.expanduser().resolve()
    if not igc_src.exists():
        raise FileNotFoundError(f"IGC source not found: {igc_src}")

    target = run_dir / igc_src.name
    if target.exists():
        return target

    if mode == "symlink":
        try:
            if target.is_symlink():
                target.unlink()
            target.symlink_to(igc_src)
            return target
        except (OSError, NotImplementedError):
            pass  # fallback to copy

    shutil.copy2(igc_src, target)
    return target


# ------------------- IGC parsing -------------------
def parse_igc_brecords(path: Path) -> pd.DataFrame:
    times, lats, lons, p_alts, g_alts = [], [], [], [], []
    day_offset = 0
    last_t = None
    with open(path, "r", errors="ignore") as f:
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
    df["alt"] = pd.Series(clipped).rolling(window=5, center=True, min_periods=1).mean()

    return df


# ------------------- Geometry helpers -------------------
def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))


# ------------------- Tow end detection -------------------
def detect_tow_end(track: pd.DataFrame,
                   H_ft: float = 2700.0,      # Height gate (ft)
                   T_s: float = 180.0,        # Time gate (s)
                   D_ft: float = 100.0,       # Descent gate (ft)
                   steady_s: float = 60.0) -> int:
    """
    Return index of last tow sample (inclusive). Slice track.iloc[tow_idx+1:] for free flight.
    """
    if track.empty:
        return -1

    alt_col = "alt" if "alt" in track.columns else (
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

    # Gate H: altitude rise from launch
    tow_idx_H = None
    for i in range(len(alt)):
        if alt[i] - a0 >= H_m:
            tow_idx_H = i
            break

    # Gate T: elapsed time
    tow_idx_T = None
    for i in range(len(t)):
        if (t[i] - t0) >= T_s:
            tow_idx_T = i
            break

    # Gate D: after steady climb, first drop of >= D_m from running peak
    tow_idx_D = None
    climb_start_i = None
    for i in range(1, len(alt)):
        dalt = alt[i] - alt[i-1]
        if dalt > 0:
            if climb_start_i is None:
                climb_start_i = i-1
        else:
            climb_start_i = None

        if climb_start_i is not None and (t[i] - t[climb_start_i]) >= steady_s:
            run_peak = alt[i]
            for j in range(i+1, len(alt)):
                if alt[j] > run_peak:
                    run_peak = alt[j]
                if run_peak - alt[j] >= D_m:
                    tow_idx_D = j
                    break
            break

    candidates = [idx for idx in (tow_idx_H, tow_idx_T, tow_idx_D) if idx is not None]
    if not candidates:
        return -1
    return min(candidates)


# ------------------- Altitude-only cluster detection -------------------
def detect_altitude_clusters(track: pd.DataFrame,
                             min_gain_m: float = 30.0,
                             min_duration_s: float = 20.0) -> pd.DataFrame:
    """
    Identify contiguous *positive* altitude segments meeting gain & duration thresholds.
    Returns DataFrame with columns:
        cluster_id, lat, lon, t_start, t_end, climb_rate_ms, alt_gain_m, duration_s
    """
    cols = ["cluster_id","lat","lon","t_start","t_end","climb_rate_ms","alt_gain_m","duration_s"]
    if track.empty:
        return pd.DataFrame(columns=cols)

    clusters = []
    start_i = None
    for i in range(1, len(track)):
        dalt = float(track.alt.iloc[i] - track.alt.iloc[i-1])
        if dalt > 0:
            if start_i is None:
                start_i = i - 1
        else:
            if start_i is not None:
                seg = track.iloc[start_i:i]
                gain = float(seg.alt.iloc[-1] - seg.alt.iloc[0])
                dur  = float(seg.time_s.iloc[-1] - seg.time_s.iloc[0])
                if gain >= min_gain_m and dur >= min_duration_s:
                    clusters.append((seg, gain, dur))
                start_i = None

    # tail segment
    if start_i is not None:
        seg = track.iloc[start_i:len(track)]
        gain = float(seg.alt.iloc[-1] - seg.alt.iloc[0])
        dur  = float(seg.time_s.iloc[-1] - seg.time_s.iloc[0])
        if gain >= min_gain_m and dur >= min_duration_s:
            clusters.append((seg, gain, dur))

    if not clusters:
        return pd.DataFrame(columns=cols)

    # Build records
    rows = []
    for idx, (seg, gain, dur) in enumerate(clusters):
        rows.append({
            "cluster_id": idx,
            "lat": float(seg.lat.mean()),
            "lon": float(seg.lon.mean()),
            "t_start": float(seg.time_s.iloc[0]),
            "t_end": float(seg.time_s.iloc[-1]),
            "duration_s": float(dur),
            "alt_gain_m": float(gain),
            "climb_rate_ms": float(gain / dur) if dur > 0 else float("nan"),
        })
    df = pd.DataFrame(rows, columns=cols)
    return df


# ------------------- Main -------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Detect altitude-only climb clusters from an IGC")
    ap.add_argument("igc", nargs="?", help="Path to IGC file")
    ap.add_argument("--no-tow-cut", action="store_true", help="Disable tow cut")
    args = ap.parse_args()

    # Default IGC path logic: remember last entry
    last_igc = _read_last_igc()
    if args.igc:
        igc_path = Path(args.igc).expanduser()
    else:
        prompt_default = f" [default: {last_igc}]" if last_igc else ""
        user_in = input(f"Enter IGC file path{prompt_default}: ").strip()
        igc_path = Path(user_in).expanduser() if user_in else (last_igc or Path("igc/sample.igc"))

    if not igc_path.exists():
        print(f"[ERROR] IGC not found: {igc_path}", file=sys.stderr)
        return 2

    # Remember for next time
    _write_last_igc(igc_path)

    # Per-flight run dir
    flight = igc_path.stem.rstrip(')').strip()
    run_dir = OUT_ROOT / flight
    run_dir.mkdir(parents=True, exist_ok=True)

    # Ensure local copy for reproducibility
    igc_local = ensure_igc_in_run_dir(igc_path, run_dir, mode="symlink")

    # Parse and smooth
    track = parse_igc_brecords(igc_local)

    # Optional tow cut
    if not args.no_tow_cut:
        tow_end_idx = detect_tow_end(track, H_ft=2700.0, T_s=180.0, D_ft=100.0, steady_s=60.0)
        if 0 <= tow_end_idx < len(track) - 1:
            track = track.iloc[tow_end_idx + 1:].reset_index(drop=True)
            print(f"[INFO] Tow cut at idx={tow_end_idx}, samples left={len(track)}")

    # Tuning
    cfg = load_tuning()
    min_gain = float(cfg.get("alt_min_gain", 30.0))
    min_dur  = float(cfg.get("alt_min_duration", 20.0))
    print(f"[TUNING] alt_min_gain={min_gain} m, alt_min_duration={min_dur} s")

    # Detect altitude clusters
    alts = detect_altitude_clusters(track, min_gain_m=min_gain, min_duration_s=min_dur)

    # Write CSV
    out_csv = run_dir / "altitude_clusters.csv"
    alts.to_csv(out_csv, index=False)

    print(f"[INFO] Detected {len(alts)} altitude clusters")
    print(f"[OK] wrote altitude clusters → {out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
