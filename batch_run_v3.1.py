#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_run_v3.1.py — MONOLITHIC batcher (no imports from pipeline)
Processes all *.igc in a chosen folder, producing per-flight artifacts:
  circles.csv, circle_clusters_enriched.csv, altitude_clusters.csv,
  matched_clusters.{csv,json}, pipeline_debug.log

Global batch progress is written to outputs/batch_debug.log

Canonical orders:
- circles.csv:                 lat,lon,t_start,t_end,climb_rate_ms,alt_gain_m,duration_s
- circle_clusters_enriched.csv:cluster_id,lat,lon,t_start,t_end,climb_rate_ms,climb_rate_ms_median,alt_gain_m_mean,duration_s_mean,n_circles
- altitude_clusters.csv:       cluster_id,lat,lon,t_start,t_end,climb_rate_ms,alt_gain_m,duration_s
- matched_clusters.csv:        lat,lon,climb_rate_ms,alt_gain_m,duration_s,circle_cluster_id,alt_cluster_id,circle_lat,circle_lon,alt_lat,alt_lon,dist_m,time_overlap_s,overlap_frac
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

# Optional scikit-learn for DBSCAN (used for circle clustering)
try:
    from sklearn.cluster import DBSCAN
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False

# ---------------------------------------------------------------------------
# Roots and globals
# ---------------------------------------------------------------------------
ROOT      = Path.cwd()
IGC_DIR   = ROOT / "igc"
OUT_ROOT  = ROOT / "outputs" / "batch_csv"
BATCH_LOG = ROOT / "outputs" / "batch_debug.log"
TUNING    = ROOT / "tuning.json"

# ---------------------------------------------------------------------------
# Logging & helpers
# ---------------------------------------------------------------------------
def logf_write(path: Path, msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")

def wipe_run_dir(run_dir: Path) -> None:
    """Clear a per-flight directory; keep the parent."""
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

# ---------------------------------------------------------------------------
# Load tuning (shared knobs)
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "circle_eps_m":       200.0,
    "circle_min_samples": 2,
    "alt_min_gain":       30.0,
    "alt_min_duration":   20.0,
    "match_max_dist_m":   600.0,
    "match_min_overlap":  0.25,
}

def load_tuning() -> dict:
    if TUNING.exists():
        try:
            cfg = json.loads(TUNING.read_text(encoding="utf-8"))
            return {**_DEFAULTS, **cfg}
        except Exception:
            return _DEFAULTS.copy()
    return _DEFAULTS.copy()

# ---------------------------------------------------------------------------
# 1) IGC parsing (B-records)
# ---------------------------------------------------------------------------
def _decode_lat(dms: str, hemi: str) -> float:
    deg = int(dms[0:2])
    mmm = int(dms[2:]) / 1000.0
    val = deg + (mmm/60.0)
    return -val if hemi.upper() == 'S' else val

def _decode_lon(dms: str, hemi: str) -> float:
    deg = int(dms[0:3])
    mmm = int(dms[3:]) / 1000.0
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

# ---------------------------------------------------------------------------
# Tow end detection (H/T/D gates) — index of last tow sample (inclusive)
# ---------------------------------------------------------------------------
def detect_tow_end(track: pd.DataFrame,
                   H_ft: float = 2700.0,
                   T_s: float  = 180.0,
                   D_ft: float = 100.0,
                   steady_s: float = 60.0) -> int:
    """
    Return index of last tow sample (inclusive). Slice track.iloc[idx+1:] for free flight.

    Gates:
      H: altitude gain from launch >= H_ft
      T: elapsed time since launch >= T_s
      D: once steady climb of >= steady_s, first drop >= D_ft from running peak
    Picks earliest satisfied gate.
    """
    if track.empty:
        return -1

    # prefer 'alt' (this parser), else tolerate alt_smooth/raw/gps/pressure
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

# ---------------------------------------------------------------------------
# 2) Circle detection
# ---------------------------------------------------------------------------
def detect_circles(df: pd.DataFrame,
                   min_rotation_deg: float = 300.0,
                   min_points: int = 12,
                   min_duration_s: float = 15.0,
                   max_duration_s: float = 180.0) -> pd.DataFrame:
    """
    Output columns: t_start, t_end, duration_s, lat, lon, alt_gain_m, climb_rate_ms
    """
    n = len(df)
    if n < 3:
        return pd.DataFrame(columns=["t_start","t_end","duration_s","lat","lon","alt_gain_m","climb_rate_ms"])

    bearings = [bearing_deg(df.lat.iloc[i], df.lon.iloc[i], df.lat.iloc[i+1], df.lon.iloc[i+1]) for i in range(n-1)]
    bu = unwrap_deg(bearings)

    out = []
    i = 0
    while i < n-2:
        j = i + 1
        cum = 0.0
        while j < n-1:
            dth = bu[j] - bu[j-1]
            cum += dth
            dur = float(df.time_s.iloc[j] - df.time_s.iloc[i])
            if abs(cum) >= min_rotation_deg and (j - i + 1) >= min_points and dur >= min_duration_s:
                while dur > max_duration_s and i < j-1:
                    i += 1
                    dur = float(df.time_s.iloc[j] - df.time_s.iloc[i])
                seg = df.iloc[i:j+1]
                alt_gain = float(seg.alt.iloc[-1] - seg.alt.iloc[0]) if "alt" in seg.columns else float("nan")
                climb = alt_gain / dur if dur > 0 else float("nan")
                out.append({
                    "t_start": float(seg.time_s.iloc[0]),
                    "t_end":   float(seg.time_s.iloc[-1]),
                    "duration_s": dur,
                    "lat": float(seg.lat.mean()),
                    "lon": float(seg.lon.mean()),
                    "alt_gain_m": alt_gain,
                    "climb_rate_ms": climb,
                })
                i = j + 1
                break
            j += 1
        else:
            i += 1
    return pd.DataFrame(out)

# ---------------------------------------------------------------------------
# 3) Cluster circles
# ---------------------------------------------------------------------------
def cluster_circles(circles: pd.DataFrame,
                    eps_m: float = 200.0,
                    min_samples: int = 2) -> pd.DataFrame:
    """
    Output columns (canonical order):
      cluster_id, lat, lon, t_start, t_end, climb_rate_ms, climb_rate_ms_median,
      alt_gain_m_mean, duration_s_mean, n_circles
    """
    base_cols = [
        "cluster_id", "lat", "lon", "t_start", "t_end",
        "climb_rate_ms", "climb_rate_ms_median", "alt_gain_m_mean", "duration_s_mean",
        "n_circles"
    ]
    if circles.empty:
        return pd.DataFrame(columns=base_cols)

    d = circles.dropna(subset=["lat","lon"]).copy()
    if d.empty:
        return pd.DataFrame(columns=base_cols)

    if _HAVE_SKLEARN:
        import numpy as np
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

# ---------------------------------------------------------------------------
# 4) Altitude-only climb clusters
# ---------------------------------------------------------------------------
def detect_altitude_clusters(track: pd.DataFrame,
                             min_gain_m: float = 30.0,
                             min_duration_s: float = 20.0) -> pd.DataFrame:
    """
    Output columns (canonical order):
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

# ---------------------------------------------------------------------------
# 5) Matching
# ---------------------------------------------------------------------------
def _overlap(a0,a1,b0,b1) -> Tuple[float,float]:
    lo = max(a0,b0); hi = min(a1,b1)
    ov = max(0.0, hi-lo)
    shorter = max(1e-9, min(a1-a0, b1-b0))
    return ov, ov/shorter

def match_clusters(circle_clusters: pd.DataFrame,
                   alt_clusters: pd.DataFrame,
                   max_dist_m: float = 600.0,
                   min_overlap_frac: float = 0.25) -> Tuple[pd.DataFrame, dict]:
    """
    Output columns (canonical order):
      lat, lon, climb_rate_ms, alt_gain_m, duration_s,
      circle_cluster_id, alt_cluster_id,
      circle_lat, circle_lon, alt_lat, alt_lon,
      dist_m, time_overlap_s, overlap_frac
    """
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
        best_key = (-1.0, float("inf"))  # prefer larger overlap, then nearer
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
                # representative point = midpoint
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

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> int:
    # Choose folder (interactive prompt; default ./igc)
    user_in = input(f"Enter IGC folder [default: {IGC_DIR}]: ").strip()
    igc_dir = Path(user_in) if user_in else IGC_DIR
    if not igc_dir.exists():
        print(f"[ERROR] folder not found: {igc_dir}")
        return 1

    igc_files = sorted(igc_dir.glob("*.igc"))
    if not igc_files:
        print(f"[WARN] no .igc files in {igc_dir}")
        return 0

    # Load tuning once
    cfg = load_tuning()
    circle_eps_m       = float(cfg["circle_eps_m"])
    circle_min_samples = int(cfg["circle_min_samples"])
    alt_min_gain       = float(cfg["alt_min_gain"])
    alt_min_duration   = float(cfg["alt_min_duration"])
    match_max_dist_m   = float(cfg["match_max_dist_m"])
    match_min_overlap  = float(cfg["match_min_overlap"])

    # Echo tunings to console & global log
    tuneline = (f"TUNING: circle_eps_m={circle_eps_m}, circle_min_samples={circle_min_samples}, "
                f"alt_min_gain={alt_min_gain}, alt_min_duration={alt_min_duration}, "
                f"match_max_dist_m={match_max_dist_m}, match_min_overlap={match_min_overlap}")
    print(f"[TUNING] {tuneline}")
    logf_write(BATCH_LOG, f"[BATCH-START] {tuneline}")
    logf_write(BATCH_LOG, f"[BATCH-START] folder={igc_dir} files={len(igc_files)}")

    total = len(igc_files)
    batch_start = time.time()

    for idx, igc_path in enumerate(igc_files, start=1):
        flight = igc_path.stem
        run_dir = OUT_ROOT / flight
        wipe_run_dir(run_dir)
        logf = run_dir / "pipeline_debug.log"

        logf_write(logf, "===== batch_run_v3.1 start =====")
        logf_write(logf, f"IGC: {igc_path}")
        logf_write(logf, f"RUN_DIR: {run_dir}")
        logf_write(logf, tuneline)

        # 1) Track
        track = parse_igc_brecords(igc_path)
        if track.empty:
            logf_write(logf, "[WARN] No B-records; skipping.")
            logf_write(BATCH_LOG, f"[BATCH] flight={flight} status=EMPTY_BRECORDS")
            continue

        # Tow cut BEFORE circles/alt clusters
        tow_end_idx = detect_tow_end(track, H_ft=2700.0, T_s=180.0, D_ft=100.0, steady_s=60.0)
        if tow_end_idx >= 0 and tow_end_idx < len(track) - 1:
            track = track.iloc[tow_end_idx + 1:].reset_index(drop=True)
            logf_write(logf, f"[INFO] Tow cut at idx={tow_end_idx}, samples left={len(track)}")

        # 2) Circles
        circles = detect_circles(track)
        circles_cols = ["lat","lon","t_start","t_end","climb_rate_ms","alt_gain_m","duration_s"]
        if not circles.empty:
            circles = circles.reindex(columns=[c for c in circles_cols if c in circles.columns])
        else:
            circles = pd.DataFrame(columns=circles_cols)
        (run_dir / "circles.csv").parent.mkdir(parents=True, exist_ok=True)
        circles.to_csv(run_dir / "circles.csv", index=False)

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
        cc.to_csv(run_dir / "circle_clusters_enriched.csv", index=False)

        # 4) Altitude clusters
        alts = detect_altitude_clusters(track, min_gain_m=alt_min_gain, min_duration_s=alt_min_duration)
        alt_cols = ["cluster_id","lat","lon","t_start","t_end","climb_rate_ms","alt_gain_m","duration_s"]
        if alts.empty:
            alts = pd.DataFrame(columns=alt_cols)
        else:
            for k in alt_cols:
                if k not in alts.columns: alts[k] = pd.NA
            alts = alts[alt_cols]
        alts.to_csv(run_dir / "altitude_clusters.csv", index=False)

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
        matches.to_csv(run_dir / "matched_clusters.csv", index=False)
        (run_dir / "matched_clusters.json").write_text(
            json.dumps({"stats": stats, "rows": int(len(matches))}, indent=2),
            encoding="utf-8"
        )

        logf_write(logf, f"[OK] wrote circles={len(circles)}, cc={len(cc)}, alts={len(alts)}, matches={len(matches)}")
        logf_write(BATCH_LOG,
                   f"[BATCH] flight={flight} circles={len(circles)} cc={len(cc)} alts={len(alts)} matches={len(matches)}")

        # Progress to console
        elapsed = time.time() - batch_start
        print(f"[PROGRESS] {idx}/{total} files processed ({elapsed:.1f} sec elapsed)")

    # Done
    total_elapsed = time.time() - batch_start
    print(f"[DONE] Processed {total} files in {total_elapsed:.1f} sec")
    logf_write(BATCH_LOG, f"[BATCH-DONE] files={total} elapsed_s={total_elapsed:.1f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())