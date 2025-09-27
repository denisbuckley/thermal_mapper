#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
igc_utils.py — self-contained helpers used across the pipeline.

Provides:
  - parse_igc(path) -> DataFrame[time, lat, lon, alt]
  - compute_derived(df) -> adds dt(s), dh(m), climb_rate(m/s), heading(deg)
  - detect_tow_segment(df) -> (i_start, i_end) indices of tow to exclude

No imports from drafts/ or scratch files. Safe for batch use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Core parsers
# ------------------------------------------------------------
def parse_igc(path: str | Path) -> pd.DataFrame:
    """
    Minimal B-record parser.
    Returns DataFrame with columns: time (datetime64[ns]), lat (deg), lon (deg), alt (m).
    Notes:
      - Uses GNSS altitude if present, otherwise pressure altitude.
      - Time uses a dummy date (1970-01-01); only time deltas are used downstream.
    """
    path = Path(path)
    times, lats, lons, alts = [], [], [], []
    with path.open("r", errors="ignore") as f:
        for line in f:
            if not line.startswith("B") or len(line) < 35:
                continue
            # B HHMMSS DDMMmmmN DDDMMmmmE A PPPPP GGGGG ...
            hh = int(line[1:3]); mm = int(line[3:5]); ss = int(line[5:7])
            lat_deg = int(line[7:9]); lat_min = int(line[9:11]); lat_thou = int(line[11:14]); lat_hem = line[14]
            lon_deg = int(line[15:18]); lon_min = int(line[18:20]); lon_thou = int(line[20:23]); lon_hem = line[23]

            try:
                p_alt = int(line[25:30])
            except Exception:
                p_alt = 0
            try:
                g_alt = int(line[30:35])
            except Exception:
                g_alt = 0
            alt = float(g_alt if g_alt != 0 else p_alt)

            lat = lat_deg + (lat_min + lat_thou/1000.0)/60.0
            if lat_hem.upper() == "S":
                lat = -lat
            lon = lon_deg + (lon_min + lon_thou/1000.0)/60.0
            if lon_hem.upper() in ("W", "O"):  # 'O' sometimes used for West
                lon = -lon

            times.append(f"{hh:02d}:{mm:02d}:{ss:02d}")
            lats.append(lat); lons.append(lon); alts.append(alt)

    if not times:
        return pd.DataFrame(columns=["time","lat","lon","alt"])

    df = pd.DataFrame({"time_str": times, "lat": lats, "lon": lons, "alt": alts})
    df["time"] = pd.to_datetime("1970-01-01 " + df["time_str"], utc=False)
    return df[["time","lat","lon","alt"]].reset_index(drop=True)


# ------------------------------------------------------------
# Derived kinematics
# ------------------------------------------------------------
def _bearing_deg(lat1, lon1, lat2, lon2) -> float:
    """Forward azimuth (deg in [0,360))."""
    φ1 = np.radians(lat1); φ2 = np.radians(lat2)
    λ1 = np.radians(lon1); λ2 = np.radians(lon2)
    dλ = λ2 - λ1
    y = np.sin(dλ) * np.cos(φ2)
    x = np.cos(φ1)*np.sin(φ2) - np.sin(φ1)*np.cos(φ2)*np.cos(dλ)
    θ = np.degrees(np.arctan2(y, x))
    return (θ + 360.0) % 360.0


def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - dt (s): time delta between fixes (clamped to median for zeros)
      - dh (m): altitude delta
      - climb_rate (m/s): dh/dt
      - heading (deg): forward azimuth per fix
    """
    if df.empty:
        out = df.copy()
        out["dt"] = out["dh"] = out["climb_rate"] = out["heading"] = np.nan
        return out

    out = df.copy()
    dt = out["time"].diff().dt.total_seconds().fillna(0.0)
    pos_dt = dt[dt > 0]
    median_dt = float(pos_dt.median()) if not pos_dt.empty else 1.0
    dt = dt.replace(0.0, median_dt).clip(lower=0.1)
    out["dt"] = dt

    dh = out["alt"].diff().fillna(0.0)
    out["dh"] = dh
    out["climb_rate"] = (dh / dt).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    lat = out["lat"].to_numpy(); lon = out["lon"].to_numpy()
    if len(lat) >= 2:
        heads = np.empty_like(lat, dtype=float)
        heads[0] = np.nan
        heads[1:] = [_bearing_deg(lat[i-1], lon[i-1], lat[i], lon[i]) for i in range(1, len(lat))]
        if np.isnan(heads[0]):
            heads[0] = heads[1] if len(heads) > 1 else 0.0
        out["heading"] = pd.Series(heads, index=out.index).bfill().fillna(0.0)
        out["heading"] = 0.0

    return out


# ------------------------------------------------------------
# Tow detection
# ------------------------------------------------------------
def detect_tow_segment(df: pd.DataFrame) -> Tuple[int, int]:
    """
    Heuristic tow detector:
      - Start at index 0.
      - End when we see >=60s cumulative where climb_rate <= 0.5 m/s (post-release cruise).
    Returns (i_start, i_end). You typically exclude 0..i_end.
    """
    if df.empty:
        return 0, 0
    cr = df["climb_rate"].to_numpy()
    dt = df["dt"].to_numpy() if "dt" in df.columns else np.full_like(cr, 1.0, dtype=float)
    acc = 0.0
    end_idx = len(cr) - 1
    for i, r in enumerate(cr):
        if r <= 0.5:
            acc += dt[i]
            if acc >= 60.0:
                end_idx = i
                break
        else:
            acc = 0.0
    if end_idx < 0:
        end_idx = 0
    return 0, int(end_idx)
