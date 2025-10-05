#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pipeline_v4.1d.py — monolithic pipeline + inline plot (show only)

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import re

def igc_pilot_name(igc_path: Path) -> str:
    """
    Parse pilot name from common IGC headers:
      HFPLTPILOTINCHARGE, HOPLTPILOT, HPILOT
    """
    try:
        with igc_path.open("r", errors="ignore") as f:
            for line in f:
                if not line.startswith(("H", "h")):
                    continue
                L = line.strip()
                for key in ("HFPLTPILOTINCHARGE:", "HOPLTPILOT:", "HPILOT:"):
                    if key in L:
                        name = L.split(key, 1)[1].strip()
                        return name if name else "Unknown"
    except Exception:
        pass
    return "Unknown"


def matched_summary_text(run_dir: Path) -> str:
    """
    Build a compact stats string from matched_clusters.csv.
    Shows counts and simple means/p90.
    """
    p = run_dir / "matched_clusters.csv"
    if not p.exists():
        return "matches: 0"

    try:
        m = pd.read_csv(p)
    except Exception:
        return "matches: (unreadable)"

    if m.empty:
        return "matches: 0"

    cols = {c.lower(): c for c in m.columns}
    def col(name): return cols.get(name)

    n = len(m)

    def stat(name, fn=np.nanmean, fmt=".2f"):
        c = col(name)
        if not c: return "n/a"
        vals = pd.to_numeric(m[c], errors="coerce")
        vals = vals.dropna()
        if not len(vals): return "n/a"
        return format(fn(vals), fmt)

    mean_cr   = stat("climb_rate_ms")
    p90_cr    = stat("climb_rate_ms", fn=lambda s: np.percentile(s, 90))
    mean_gain = stat("alt_gain_m")
    mean_dur  = stat("duration_s")
    mean_dist = stat("dist_m")
    mean_ovl  = stat("overlap_frac", fmt=".2f")

    return (f"matches: {n}\n"
            f"climb_rate_ms mean {mean_cr} (p90 {p90_cr})\n"
            f"alt_gain_m mean {mean_gain}\n"
            f"dur_s mean {mean_dur}\n"
            f"dist_m mean {mean_dist}\n"
            f"overlap {mean_ovl}")

# Optional scikit-learn for DBSCAN
try:
    from sklearn.cluster import DBSCAN
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False

# -------------------- Roots and helpers --------------------
ROOT     = Path.cwd()
IGC_DIR  = ROOT / "igc"
OUT_ROOT = ROOT / "outputs" / "batch_csv"
TUNING_FILE = ROOT / "tuning.json"

def load_tuning() -> dict:
    defaults = {
        "circle_eps_m":       200.0,
        "circle_min_samples": 2,
        "alt_min_gain":       30.0,
        "alt_min_duration":   20.0,
        "match_max_dist_m":   600.0,
        "match_min_overlap":  0.25,
    }
    if TUNING_FILE.exists():
        try:
            cfg = json.loads(TUNING_FILE.read_text(encoding="utf-8"))
            return {**defaults, **cfg}
        except Exception:
            return defaults
    return defaults

def logf_write(logf: Path, msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    logf.parent.mkdir(parents=True, exist_ok=True)
    with logf.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")

def wipe_run_dir(run_dir: Path) -> None:
    if run_dir.exists():
        for child in run_dir.iterdir():
            if child.is_file() or child.is_symlink():
                try:
                    child.unlink(missing_ok=True)
                except Exception:
                    pass
            else:
                import shutil
                shutil.rmtree(child, ignore_errors=True)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371008.8
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dl   = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dl   = math.radians(lon2 - lon1)
    y = math.sin(dl)*math.cos(phi2)
    x = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0

def unwrap_deg(seq: List[float]) -> List[float]:
    if not seq:
        return []
    out = [seq[0]]
    for a in seq[1:]:
        prev = out[-1]
        da = a - prev
        while da > 180: a -= 360; da = a - prev
        while da < -180: a += 360; da = a - prev
        out.append(a)
    return out

# -------------------- IGC parsing (B-records) --------------------
def _decode_lat(dms: str, hemi: str) -> float:
    deg = int(dms[0:2]); mmm = int(dms[2:]) / 1000.0
    val = deg + (mmm/60.0)
    return -val if hemi.upper() == 'S' else val

def _decode_lon(dms: str, hemi: str) -> float:
    deg = int(dms[0:3]); mmm = int(dms[3:]) / 1000.0
    val = deg + (mmm/60.0)
    return -val if hemi.upper() == 'W' else val

def parse_igc_brecords(igc_path: Path) -> pd.DataFrame:
    rows = []
    start_s: Optional[int] = None
    with igc_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line[0] != 'B' or len(line) < 35:
                continue
            try:
                hh = int(line[1:3]); mm = int(line[3:5]); ss = int(line[5:7])
                lat = _decode_lat(line[7:14], line[14])
                lon = _decode_lon(line[15:23], line[23])
                gps_alt = None
                if len(line) >= 35:
                    tail = line[25:35]
                    if tail.isdigit():
                        gps_alt = int(tail[-5:])
                alt = float(gps_alt) if gps_alt is not None else float('nan')
                t_s = hh*3600 + mm*60 + ss
                if start_s is None: start_s = t_s
                rows.append((t_s - start_s, lat, lon, alt))
            except Exception:
                continue
    df = pd.DataFrame(rows, columns=["time_s","lat","lon","alt"])
    if len(df):
        df["alt"] = df["alt"].bfill().ffill()
    return df

# -------------------- Tow end detection --------------------
def detect_tow_end(track: pd.DataFrame,
                   H_ft: float = 2700.0,
                   T_s: float  = 180.0,
                   D_ft: float = 100.0,
                   steady_s: float = 60.0) -> int:
    if track.empty:
        return -1
    alt_col = "alt"
    for c in ("alt", "alt_smooth", "alt_raw", "alt_gps", "alt_pressure"):
        if c in track.columns:
            alt_col = c
            break
    FT_TO_M = 0.3048
    H_m = H_ft * FT_TO_M
    D_m = D_ft * FT_TO_M
    t = track["time_s"].to_numpy()
    alt = track[alt_col].to_numpy()
    t0 = float(t[0]); a0 = float(alt[0])

    tow_idx_H = None
    for i in range(len(alt)):
        if alt[i] - a0 >= H_m:
            tow_idx_H = i
            break
    tow_idx_T = None
    for i in range(len(t)):
        if (t[i] - t0) >= T_s:
            tow_idx_T = i
            break

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

    cand = [x for x in (tow_idx_H, tow_idx_T, tow_idx_D) if x is not None]
    if not cand:
        return -1
    return min(cand)

# -------------------- Circle detection (enriched) --------------------
def detect_circles(df: pd.DataFrame,
                   min_duration_s: float = 6.0,
                   max_duration_s: float = 60.0,
                   min_radius_m: float = 8.0,
                   max_radius_m: float = 600.0,
                   min_bank_deg: float = 5.0,
                   vmax_climb_ms: float = 10.0) -> pd.DataFrame:
    n = len(df)
    base_cols = [
        "circle_id","seg_id","t_start","t_end","duration_s",
        "avg_speed_kmh","alt_gain_m","climb_rate_ms",
        "turn_radius_m","circle_diameter_m","bank_angle_deg","lat","lon",
        "bank_std_deg","bank_range_deg","bank_masd_deg",
    ]
    if n < 3:
        return pd.DataFrame(columns=base_cols)

    lat = df["lat"].to_numpy()
    lon = df["lon"].to_numpy()
    alt = df["alt"].to_numpy() if "alt" in df.columns else np.full(n, np.nan, dtype=float)
    t   = df["time_s"].to_numpy()

    bearings = np.zeros(n)
    for i in range(1, n):
        bearings[i] = bearing_deg(lat[i-1], lon[i-1], lat[i], lon[i])
    uw = np.array(unwrap_deg(list(bearings)), dtype=float)

    circles = []
    start_idx, circle_id, g = 0, 0, 9.81
    i = 1
    while i < n:
        rot = abs(uw[i] - uw[start_idx])
        if rot >= 360.0:
            i0, i1 = start_idx, i
            dur = t[i1] - t[i0]
            if not (min_duration_s <= dur <= max_duration_s):
                start_idx = i; i += 1; continue

            dist = 0.0
            for k in range(i0+1, i1+1):
                dist += haversine_m(lat[k-1], lon[k-1], lat[k], lon[k])
            v_mean = dist/dur if dur > 0 else np.nan  # m/s

            omega = 2*math.pi/dur if dur > 0 else np.nan
            radius = (v_mean/omega) if (omega and omega > 0) else np.nan
            if (np.isnan(radius)) or (radius < min_radius_m) or (radius > max_radius_m):
                start_idx = i; i += 1; continue

            bank = math.degrees(math.atan((v_mean**2)/(g*radius))) if (radius and np.isfinite(radius)) else np.nan
            if (np.isnan(bank)) or (bank < min_bank_deg):
                start_idx = i; i += 1; continue

            #— within-circle bank variability
            banks_deg = []
            for k in range(i0+1, i1+1):
                dt = t[k] - t[k-1]
                if dt <= 0:
                    continue
                dtheta_rad = math.radians(uw[k] - uw[k-1])
                omega_k = dtheta_rad / dt
                seg_m = haversine_m(lat[k-1], lon[k-1], lat[k], lon[k])
                v_inst = seg_m / dt
                b_rad = math.atan((v_inst * omega_k) / g) if np.isfinite(v_inst) and np.isfinite(omega_k) else float("nan")
                if np.isfinite(b_rad):
                    banks_deg.append(math.degrees(b_rad))
            if banks_deg:
                b_arr = np.array(banks_deg, dtype=float)
                bank_std_deg   = float(np.nanstd(b_arr, ddof=1)) if len(b_arr) >= 2 else 0.0
                bank_range_deg = float(np.nanmax(b_arr) - np.nanmin(b_arr))
                bank_masd_deg  = float(np.nanmean(np.abs(np.diff(b_arr)))) if len(b_arr) >= 2 else 0.0
            else:
                bank_std_deg = np.nan; bank_range_deg = np.nan; bank_masd_deg = np.nan

            # alt gain via robust slope
            idx = np.arange(i0, i1+1)
            tt = t[idx] - t[i0]
            aa = alt[idx]
            if len(tt) >= 3 and np.all(np.isfinite(tt)) and np.any(np.isfinite(aa)):
                A = np.vstack([tt, np.ones_like(tt)]).T
                mask = np.isfinite(aa)
                if mask.sum() >= 3:
                    m_slope, _ = np.linalg.lstsq(A[mask], aa[mask], rcond=None)[0]
                    alt_gain = float(m_slope * dur)
                else:
                    span = i1 - i0 + 1
                    q = max(1, span // 4)
                    first_med = np.nanmedian(aa[:q]); last_med = np.nanmedian(aa[-q:])
                    alt_gain = (last_med - first_med) if np.isfinite(first_med) and np.isfinite(last_med) else np.nan
            else:
                alt_gain = np.nan

            max_gain = vmax_climb_ms * dur
            if np.isfinite(alt_gain) and abs(alt_gain) > max_gain:
                alt_gain = np.sign(alt_gain) * max_gain
            climb = (alt_gain / dur) if (dur > 0 and np.isfinite(alt_gain)) else np.nan

            circles.append({
                "circle_id":         circle_id,
                "seg_id":            None,
                "t_start":           float(t[i0]),
                "t_end":             float(t[i1]),
                "duration_s":        float(dur),
                "avg_speed_kmh":     float(v_mean * 3.6) if np.isfinite(v_mean) else np.nan,
                "alt_gain_m":        float(alt_gain) if np.isfinite(alt_gain) else np.nan,
                "climb_rate_ms":     float(climb) if np.isfinite(climb) else np.nan,
                "turn_radius_m":     float(radius) if np.isfinite(radius) else np.nan,
                "circle_diameter_m": float(2 * radius) if np.isfinite(radius) else np.nan,
                "bank_angle_deg":    float(bank) if np.isfinite(bank) else np.nan,
                "lat":               float(np.nanmean(lat[i0:i1+1])),
                "lon":               float(np.nanmean(lon[i0:i1+1])),
                "bank_std_deg":      bank_std_deg,
                "bank_range_deg":    bank_range_deg,
                "bank_masd_deg":     bank_masd_deg,
            })
            circle_id += 1
            start_idx = i
        i += 1

    d = pd.DataFrame(circles)
    return d[base_cols] if not d.empty else pd.DataFrame(columns=base_cols)

# -------------------- Circle clustering --------------------
def cluster_circles(circles: pd.DataFrame,
                    eps_m: float = 200.0,
                    min_samples: int = 2) -> pd.DataFrame:
    base_cols = [
        "cluster_id","lat","lon","t_start","t_end",
        "climb_rate_ms","climb_rate_ms_median","alt_gain_m_mean","duration_s_mean",
        "n_circles"
    ]
    if circles.empty:
        return pd.DataFrame(columns=base_cols)
    d = circles.dropna(subset=["lat","lon"]).copy()
    if d.empty:
        return pd.DataFrame(columns=base_cols)

    if _HAVE_SKLEARN:
        latr = np.radians(d["lat"].to_numpy()); lonr = np.radians(d["lon"].to_numpy())
        X = np.c_[latr, lonr]
        def hav(u, v):
            R = 6371008.8
            dphi = v[0]-u[0]; dl = v[1]-u[1]
            aa = math.sin(dphi/2)**2 + math.cos(u[0])*math.cos(v[0])*math.sin(dl/2)**2
            return 2*R*math.asin(math.sqrt(aa))
        db = DBSCAN(eps=eps_m, min_samples=min_samples, metric=hav).fit(X)
        d = d.reset_index(drop=True)
        d["cluster_id"] = db.labels_
        d = d[d["cluster_id"] != -1]
    else:
        idx = list(d.index)
        parent = {i:i for i in idx}
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        def union(a,b):
            ra, rb = find(a), find(b)
            if ra != rb: parent[rb] = ra
        for i in idx:
            for j in idx:
                if j <= i: continue
                if haversine_m(d.at[i,"lat"], d.at[i,"lon"], d.at[j,"lat"], d.at[j,"lon"]) <= eps_m:
                    union(i,j)
        root_to_id = {}; next_id = 0; labels = []
        for i in idx:
            r = find(i)
            if r not in root_to_id:
                root_to_id[r] = next_id; next_id += 1
            labels.append(root_to_id[r])
        d = d.reset_index(drop=True)
        d["cluster_id"] = labels

    if d.empty:
        return pd.DataFrame(columns=base_cols)

    rows = []
    for gid, sub in d.groupby("cluster_id"):
        rec = {
            "cluster_id": int(gid),
            "lat": float(sub["lat"].mean()),
            "lon": float(sub["lon"].mean()),
            "t_start": float(sub["t_start"].min() if "t_start" in sub.columns else float("nan")),
            "t_end":   float(sub["t_end"].max() if "t_end" in sub.columns else float("nan")),
            "n_circles": int(len(sub)),
            "climb_rate_ms": float("nan"),
            "climb_rate_ms_median": float("nan"),
            "alt_gain_m_mean": float("nan"),
            "duration_s_mean": float("nan"),
        }
        if "climb_rate_ms" in sub and len(sub["climb_rate_ms"].dropna()):
            rec["climb_rate_ms"] = float(sub["climb_rate_ms"].mean())
            rec["climb_rate_ms_median"] = float(sub["climb_rate_ms"].median())
        if "alt_gain_m" in sub and len(sub["alt_gain_m"].dropna()):
            rec["alt_gain_m_mean"] = float(sub["alt_gain_m"].mean())
        if "duration_s" in sub and len(sub["duration_s"].dropna()):
            rec["duration_s_mean"] = float(sub["duration_s"].mean())
        rows.append(rec)

    df = pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)
    for k in base_cols:
        if k not in df.columns:
            df[k] = pd.NA
    return df[base_cols]

# -------------------- Altitude-only climb clusters --------------------
def detect_altitude_clusters(track: pd.DataFrame,
                             min_gain_m: float = 30.0,
                             min_duration_s: float = 20.0) -> pd.DataFrame:
    cols = ["cluster_id","lat","lon","t_start","t_end","climb_rate_ms","alt_gain_m","duration_s"]
    if track.empty:
        return pd.DataFrame(columns=cols)

    clusters = []
    start_i = None
    for i in range(1, len(track)):
        dalt = float(track.alt.iloc[i] - track.alt.iloc[i-1])
        if dalt > 0:
            if start_i is None:
                start_i = i-1
        else:
            if start_i is not None:
                seg = track.iloc[start_i:i]
                gain = float(seg.alt.iloc[-1] - seg.alt.iloc[0])
                dur  = float(seg.time_s.iloc[-1] - seg.time_s.iloc[0])
                if gain >= min_gain_m and dur >= min_duration_s:
                    clusters.append({
                        "lat": float(seg.lat.mean()),
                        "lon": float(seg.lon.mean()),
                        "t_start": float(seg.time_s.iloc[0]),
                        "t_end":   float(seg.time_s.iloc[-1]),
                        "duration_s": dur,
                        "alt_gain_m": gain,
                        "climb_rate_ms": gain/dur if dur>0 else float("nan"),
                    })
                start_i = None
    if start_i is not None:
        seg = track.iloc[start_i:len(track)]
        gain = float(seg.alt.iloc[-1] - seg.alt.iloc[0])
        dur  = float(seg.time_s.iloc[-1] - seg.time_s.iloc[0])
        if gain >= min_gain_m and dur >= min_duration_s:
            clusters.append({
                "lat": float(seg.lat.mean()),
                "lon": float(seg.lon.mean()),
                "t_start": float(seg.time_s.iloc[0]),
                "t_end":   float(seg.time_s.iloc[-1]),
                "duration_s": dur,
                "alt_gain_m": gain,
                "climb_rate_ms": gain/dur if dur>0 else float("nan"),
            })

    df = pd.DataFrame(clusters).reset_index(drop=True)
    if not df.empty:
        df["cluster_id"] = df.index
        for k in cols:
            if k not in df.columns:
                df[k] = pd.NA
        df = df[cols]
    else:
        df = pd.DataFrame(columns=cols)
    return df

# -------------------- Matching --------------------
def _overlap(a0,a1,b0,b1) -> Tuple[float,float]:
    lo = max(a0,b0); hi = min(a1,b1)
    ov = max(0.0, hi-lo)
    shorter = max(1e-9, min(a1-a0, b1-b0))
    return ov, ov/shorter

def match_clusters(circle_clusters: pd.DataFrame,
                   alt_clusters: pd.DataFrame,
                   max_dist_m: float = 600.0,
                   min_overlap_frac: float = 0.25) -> Tuple[pd.DataFrame, dict]:
    cols = [
        "lat","lon","climb_rate_ms","alt_gain_m","duration_s",
        "circle_cluster_id","alt_cluster_id",
        "circle_lat","circle_lon","alt_lat","alt_lon",
        "dist_m","time_overlap_s","overlap_frac"
    ]
    if circle_clusters.empty or alt_clusters.empty:
        return pd.DataFrame(columns=cols), {"pairs": 0}

    out = []
    used = set()
    for c in circle_clusters.itertuples(index=False):
        best = None
        best_key = (-1.0, float("inf"))
        for a in alt_clusters.itertuples(index=False):
            d = haversine_m(c.lat, c.lon, a.lat, a.lon)
            if d > max_dist_m: continue
            ov, of = _overlap(c.t_start, c.t_end, a.t_start, a_tend:=a.t_end)
            if of < min_overlap_frac: continue
            key = (of, d)
            if key > best_key:
                best_key = key
                best = (a, d, ov, of)
        if best:
            a, dist_m, ov, of = best
            out.append({
                "lat": float((c.lat + a.lat)/2),
                "lon": float((c.lon + a.lon)/2),
                "climb_rate_ms": float(getattr(a, "climb_rate_ms", float("nan"))),
                "alt_gain_m":    float(getattr(a, "alt_gain_m", float("nan"))),
                "duration_s":    float(getattr(a, "duration_s", float("nan"))),
                "circle_cluster_id": int(getattr(c, "cluster_id", 0)),
                "alt_cluster_id":    int(getattr(a, "cluster_id", 0)),
                "circle_lat": float(c.lat), "circle_lon": float(c.lon),
                "alt_lat":    float(a.lat), "alt_lon":    float(a.lon),
                "dist_m": float(dist_m),
                "time_overlap_s": float(ov),
                "overlap_frac": float(of),
            })
            used.add(int(getattr(a, "cluster_id", -1)))

    df = pd.DataFrame(out, columns=cols)
    df = df[cols] if not df.empty else pd.DataFrame(columns=cols)
    stats = {"pairs": int(len(df)), "max_dist_m": float(max_dist_m), "min_overlap_frac": float(min_overlap_frac)}
    return df, stats

# -------------------- Ensure IGC copy into run_dir --------------------
import shutil, gzip

def ensure_igc_copy(stem: str, run_dir: Path, src_root: Path = Path("igc")) -> bool:
    run_dir.mkdir(parents=True, exist_ok=True)
    dest = run_dir / f"{stem}.igc"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[IGC] already present → {dest}")
        return True
    for cand in [src_root / f"{stem}.igc", src_root / f"{stem}.IGC", src_root / f"{stem}.igc.gz"]:
        if cand.exists():
            try:
                if cand.suffix.lower() == ".gz":
                    with gzip.open(cand, "rb") as fin, open(dest, "wb") as fout:
                        shutil.copyfileobj(fin, fout)
                    print(f"[IGC] gunzipped {cand} → {dest}")
                else:
                    shutil.copy2(cand, dest)
                    print(f"[IGC] copied {cand} → {dest}")
                return True
            except Exception as e:
                print(f"[IGC] ERROR copying {cand}: {e}")
                return False
    print(f"[IGC] WARN: no IGC found for {stem} in {src_root}")
    return False

# ==================== Inline plotting (show only) ====================
# minimal plotter: no saving, just show; uses artifacts in run_dir

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def load_track_for_plot(run_dir: Path, stem: str) -> pd.DataFrame:
    p = run_dir / "track.csv"
    if p.exists():
        df = pd.read_csv(p)
        cols = {c.lower(): c for c in df.columns}
        need = {"lat","lon","alt"}
        if not need.issubset(cols):
            raise ValueError(f"{p} must include {need}, got {list(df.columns)}")
        out = pd.DataFrame({
            "lat": df[cols["lat"]].astype(float),
            "lon": df[cols["lon"]].astype(float),
            "alt": df[cols["alt"]].astype(float),
        })
        out["time"] = pd.to_datetime(df[cols["time"]], errors="coerce", utc=True) if "time" in cols else pd.NaT
        return out

    # else best effort from IGC in run_dir
    igcs = list(run_dir.glob("*.igc")) + list(run_dir.glob("*.IGC"))
    if not igcs:
        raise FileNotFoundError(f"No track.csv or IGC in {run_dir}")
    # tiny parser: reuse parse_igc_brecords on the copied igc but we only need lat/lon/time_s/alt
    df = parse_igc_brecords(igcs[0])
    return pd.DataFrame({"lat": df["lat"], "lon": df["lon"], "alt": df.get("alt", df.get("alt_gps", df.get("alt_pressure", np.nan))) , "time": pd.NaT})

def climb_segments(track: pd.DataFrame):
    lon = track["lon"].to_numpy()
    lat = track["lat"].to_numpy()
    alt = track["alt"].to_numpy()
    if track["time"].notna().any():
        t_ns = pd.to_datetime(track["time"], utc=True, errors="coerce").astype("int64", copy=False).to_numpy()
        dt = np.diff(t_ns) / 1e9
        dt[dt == 0] = 1e-6
    else:
        dt = np.ones(len(alt) - 1, dtype=float)
    dalt = np.diff(alt)
    climb_mask = (dalt / dt) > 0
    points = np.column_stack([lon, lat])
    segs = np.stack([points[:-1], points[1:]], axis=1)
    return segs, climb_mask

def read_clusters_xy(dir_path: Path, fname: str):
    p = dir_path / fname
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
    except Exception:
        return None
    cols = {c.lower(): c for c in df.columns}
    if "lat" not in cols or "lon" not in cols:
        return None
    out = pd.DataFrame({"lat": pd.to_numeric(df[cols["lat"]], errors="coerce"),
                        "lon": pd.to_numeric(df[cols["lon"]], errors="coerce")})
    if "cluster_id" in cols:
        out["cluster_id"] = df[cols["cluster_id"]]
    return out.dropna(subset=["lat","lon"])

def matched_summary_text(run_dir: Path) -> str:
    p = run_dir / "matched_clusters.csv"
    if not p.exists():
        return "matches: 0"
    try:
        m = pd.read_csv(p)
    except Exception:
        return "matches: 0"
    if m.empty:
        return "matches: 0"
    cols = {c.lower(): c for c in m.columns}
    def col(name): return cols.get(name)
    n = len(m)
    def stat(name, fn=np.nanmean, fmt=".2f"):
        c = col(name);
        if not c: return "n/a"
        v = pd.to_numeric(m[c], errors="coerce").dropna()
        return format(fn(v), fmt) if len(v) else "n/a"
    return (f"matches: {n} | climb_rate_ms mean {stat('climb_rate_ms')} "
            f"(p90 {stat('climb_rate_ms', fn=lambda s: np.percentile(s,90))}) | "
            f"alt_gain_m mean {stat('alt_gain_m')} | dur_s mean {stat('duration_s')} | "
            f"dist_m mean {stat('dist_m')} | overlap {stat('overlap_frac', fmt='.2f')}")

def render_one(stem: str, run_dir: Path, show: bool = True) -> None:
    # --- load track (as your helper already does) ---
    track = load_track_for_plot(run_dir, stem)
    if len(track) < 2:
        print(f"[PLOT] skip {stem}: <2 points")
        return

    # --- compute total distance (km) from lat/lon ---
    lats = track["lat"].to_numpy()
    lons = track["lon"].to_numpy()
    dist_m = 0.0
    for i in range(1, len(track)):
        dist_m += haversine_m(lats[i-1], lons[i-1], lats[i], lons[i])
    dist_km = dist_m / 1000.0

    # --- compute duration (prefer datetime 'time'; else fall back to IGC time_s) ---
    duration_s = None
    if "time" in track.columns and pd.to_datetime(track["time"], errors="coerce").notna().any():
        tmin = pd.to_datetime(track["time"], errors="coerce").min()
        tmax = pd.to_datetime(track["time"], errors="coerce").max()
        if pd.notna(tmin) and pd.notna(tmax):
            duration_s = (tmax - tmin).total_seconds()
    if duration_s is None:
        igc_copy = run_dir / f"{stem}.igc"
        if igc_copy.exists():
            try:
                df_igc = parse_igc_brecords(igc_copy)
                if len(df_igc):
                    duration_s = float(df_igc["time_s"].iloc[-1] - df_igc["time_s"].iloc[0])
            except Exception:
                duration_s = None

    def _fmt_hms(sec: Optional[float]) -> str:
        if sec is None or not np.isfinite(sec) or sec <= 0:
            return "n/a"
        sec = int(round(sec))
        h = sec // 3600
        m = (sec % 3600) // 60
        s = sec % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    # --- figure & axes ---
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_title(f"Glider Track & Clusters — {stem}")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

    # track segments (thick)
    segs, is_climb = climb_segments(track)
    lc_sink  = LineCollection(segs[~is_climb], colors="#1E6EFF", linewidths=4.0, alpha=0.9)
    lc_climb = LineCollection(segs[ is_climb], colors="#FFD400", linewidths=4.0, alpha=0.95)
    ax.add_collection(lc_sink); ax.add_collection(lc_climb)

    # extents
    ax.set_xlim(track["lon"].min() - 0.01, track["lon"].max() + 0.01)
    ax.set_ylim(track["lat"].min() - 0.01, track["lat"].max() + 0.01)

    # overlays
    alt_df = read_clusters_xy(run_dir, "altitude_clusters.csv")
    if alt_df is not None and len(alt_df):
        ax.scatter(alt_df["lon"], alt_df["lat"], s=100, facecolors="none",
                   edgecolors="green", marker="s", linewidths=2.2,
                   label="altitude_clusters (green □)")

    circle_df = read_clusters_xy(run_dir, "circle_clusters_enriched.csv")
    if circle_df is not None and len(circle_df):
        ax.scatter(circle_df["lon"], circle_df["lat"], s=200, facecolors="none",
                   edgecolors="purple", marker="o", linewidths=2.4,
                   label="circle_clusters (purple ○)")

    mfile = run_dir / "matched_clusters.csv"
    if mfile.exists():
        m = pd.read_csv(mfile)
        cols = {c.lower(): c for c in m.columns}
        if "lat" in cols and "lon" in cols and len(m):
            ax.scatter(pd.to_numeric(m[cols["lon"]], errors="coerce"),
                       pd.to_numeric(m[cols["lat"]], errors="coerce"),
                       s=300, color="purple", marker="x", linewidths=2.8,
                       label="matched (purple ×)")

    ax.legend(loc="best", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.4)

    # --- info box (bottom-right *inside* figure) ---
    pilot = igc_pilot_name(run_dir / f"{stem}.igc")
    weglide_url = f"https://www.weglide.org/flight/{stem}"

    # make matched stats multiline (convert " | " to newlines if needed)
    stats_txt = matched_summary_text(run_dir)
    stats_txt = stats_txt.replace(" | ", "\n")

    info_box = (
        f"Pilot: {pilot}\n"
        f"Distance: {dist_km:.1f} km\n"
        f"Duration: {_fmt_hms(duration_s)}\n"
        f"Weglide: {weglide_url}\n"
        f"{stats_txt}"
    )

    # Position slightly in from the bottom-right so it doesn't clip
    fig.text(0.965, 0.035, info_box,
             ha="right", va="bottom",
             fontsize=9, color="black",
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.9))

    plt.tight_layout()
    if show:
        plt.show()
    plt.close(fig)
# ==================== MAIN ====================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("igc", nargs="?", help="Path to IGC file (absolute, or just the filename under ./igc)")
    args = ap.parse_args()

    # Resolve IGC path
    if args.igc:
        cand = Path(args.igc).expanduser()
        igc_path = cand if cand.is_absolute() else (IGC_DIR / cand.name)
    else:
        user_in = input("Enter IGC filename in ./igc (required, no default): ").strip()
        if not user_in:
            print("[ERROR] No filename entered.", file=sys.stderr)
            return 2
        cand = Path(user_in).expanduser()
        igc_path = cand if cand.is_absolute() else (IGC_DIR / cand.name)

    if not igc_path.exists():
        print(f"[ERROR] IGC not found: {igc_path}", file=sys.stderr)
        return 2

    # Load tuning
    cfg = load_tuning()
    circle_eps_m       = float(cfg["circle_eps_m"])
    circle_min_samples = int(cfg["circle_min_samples"])
    alt_min_gain       = float(cfg["alt_min_gain"])
    alt_min_duration   = float(cfg["alt_min_duration"])
    match_max_dist_m   = float(cfg["match_max_dist_m"])
    match_min_overlap  = float(cfg["match_min_overlap"])

    stem = igc_path.stem
    run_dir = OUT_ROOT / stem
    wipe_run_dir(run_dir)

    # ensure igc copied for later plotting convenience
    ensure_igc_copy(stem, run_dir, src_root=IGC_DIR)

    logf = run_dir / "pipeline_debug.log"
    logf_write(logf, "===== pipeline_v4.1d start =====")
    logf_write(logf, f"IGC: {igc_path}")
    logf_write(logf, f"RUN_DIR: {run_dir}")
    logf_write(logf, f"TUNING: circle_eps_m={circle_eps_m}, circle_min_samples={circle_min_samples}, "
                     f"alt_min_gain={alt_min_gain}, alt_min_duration={alt_min_duration}, "
                     f"match_max_dist_m={match_max_dist_m}, match_min_overlap={match_min_overlap}")
    print(
        f"[TUNING] circle_eps_m={circle_eps_m}, circle_min_samples={circle_min_samples}, "
        f"alt_min_gain={alt_min_gain}, alt_min_duration={alt_min_duration}, "
        f"match_max_dist_m={match_max_dist_m}, match_min_overlap={match_min_overlap}"
    )

    # 1) Track
    track = parse_igc_brecords(igc_path)
    if track.empty:
        logf_write(logf, "[WARN] No B-records parsed; exiting.")
        print("[WARN] No B-records parsed; exiting.")
        return 0

    # Tow cut
    tow_end_idx = detect_tow_end(track, H_ft=2700.0, T_s=180.0, D_ft=100.0, steady_s=60.0)
    if tow_end_idx >= 0 and tow_end_idx < len(track) - 1:
        track = track.iloc[tow_end_idx + 1:].reset_index(drop=True)
        logf_write(logf, f"[INFO] Tow cut at idx={tow_end_idx}, samples left={len(track)}")
        print(f"[INFO] Tow cut at idx={tow_end_idx}, samples left={len(track)}")

    # 2) Circles
    circles = detect_circles(track)
    circle_cols = [
        "lat","lon","t_start","t_end","climb_rate_ms","alt_gain_m","duration_s",
        "circle_id","seg_id","avg_speed_kmh","turn_radius_m","circle_diameter_m","bank_angle_deg",
        "bank_std_deg","bank_range_deg","bank_masd_deg"
    ]
    if circles.empty:
        circles = pd.DataFrame(columns=circle_cols)
    else:
        for k in circle_cols:
            if k not in circles.columns:
                circles[k] = pd.NA
        circles = circles[circle_cols]
    (run_dir / "circles.csv").parent.mkdir(parents=True, exist_ok=True)
    circles.to_csv(run_dir / "circles.csv", index=False)
    logf_write(logf, f"[OK] wrote {len(circles)} circles")

    # 3) Circle clusters
    cc = cluster_circles(circles, eps_m=circle_eps_m, min_samples=circle_min_samples)
    cc_cols = [
        "cluster_id","lat","lon","t_start","t_end",
        "climb_rate_ms","climb_rate_ms_median","alt_gain_m_mean","duration_s_mean",
        "n_circles"
    ]
    if cc.empty:
        cc = pd.DataFrame(columns=cc_cols)
    else:
        for k in cc_cols:
            if k not in cc.columns: cc[k] = pd.NA
        cc = cc[cc_cols]
    (run_dir / "circle_clusters_enriched.csv").write_text("")  # ensure file created even if empty reorder fails
    cc.to_csv(run_dir / "circle_clusters_enriched.csv", index=False)
    logf_write(logf, f"[OK] wrote {len(cc)} circle clusters")

    # 4) Altitude clusters
    alts = detect_altitude_clusters(track, min_gain_m=alt_min_gain, min_duration_s=alt_min_duration)
    alt_cols = ["cluster_id","lat","lon","t_start","t_end","climb_rate_ms","alt_gain_m","duration_s"]
    if alts.empty:
        alts = pd.DataFrame(columns=alt_cols)
    else:
        for k in alt_cols:
            if k not in alts.columns: alts[k] = pd.NA
        alts = alts[alt_cols]
    (run_dir / "altitude_clusters.csv").to_csv(run_dir / "altitude_clusters.csv", index=False) if False else None
    alts.to_csv(run_dir / "altitude_clusters.csv", index=False)
    logf_write(logf, f"[OK] wrote {len(alts)} altitude clusters")

    # 5) Matching
    matches, stats = match_clusters(
        cc if not cc.empty else pd.DataFrame(columns=["cluster_id","lat","lon","t_start","t_end"]),
        alts,
        max_dist_m=match_max_dist_m,
        min_overlap_frac=match_min_overlap,
    )
    match_cols = [
        "lat","lon","climb_rate_ms","alt_gain_m","duration_s",
        "circle_cluster_id","alt_cluster_id",
        "circle_lat","circle_lon","alt_lat","alt_lon",
        "dist_m","time_overlap_s","overlap_frac"
    ]
    if matches.empty:
        matches = pd.DataFrame(columns=match_cols)
    else:
        matches = matches.reindex(columns=match_cols)
    match_csv  = run_dir / "matched_clusters.csv"
    matches.to_csv(match_csv, index=False)

    # Summary JSON
    payload = {"stats": stats, "rows": int(len(matches))}
    if len(matches):
        summary = {}
        for c in ["dist_m","time_overlap_s","overlap_frac","climb_rate_ms","alt_gain_m","duration_s"]:
            if c in matches.columns:
                s = matches[c].describe()
                summary[c] = {
                    "min": float(s.get("min", float("nan"))),
                    "max": float(s.get("max", float("nan"))),
                    "mean": float(s.get("mean", float("nan"))),
                }
        payload["summary"] = summary
    (run_dir / "matched_clusters.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Quick console summary
    print(f"[SUMMARY] circles={len(circles)}, circle_clusters={len(cc)}, altitude_clusters={len(alts)}, matched={len(matches)}")
    logf_write(logf, f"[OK] wrote matches={len(matches)}")

    # -------- Render plot (show only; no save) --------
    try:
        render_one(stem, OUT_ROOT / stem, show=True)  # uses artifacts we just wrote
    except Exception as e:
        print(f"[WARN] plot render failed: {e}")

    logf_write(logf, "[DONE]")
    print(f"[DONE] Artifacts in: {run_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())