#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_run_v3.1.py — monolithic batch runner

Processes all IGC files in a chosen folder (default ./igc) with the pipeline flow:

  1) Parse IGC B-records → track DataFrame
  2) Detect circling segments (“circles”) → circles.csv
  3) Cluster circles → circle_clusters_enriched.csv
  4) Detect altitude-only climb segments → altitude_clusters.csv
  5) Match circle clusters ↔ altitude clusters → matched_clusters.csv + .json

Outputs go into ./outputs/batch_csv/<flight_stem>/
"""

import sys, math, time, json
from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd


def load_tuning():
    import json
    from pathlib import Path
    cfg_file = Path("tuning.json")
    if cfg_file.exists():
        return json.loads(cfg_file.read_text(encoding="utf-8"))
    else:
        raise FileNotFoundError("tuning.json not found. Run tuning_v1.py first.")
# ---------------------------------------------------------------------------
# Roots and helpers
# ---------------------------------------------------------------------------
ROOT     = Path.cwd()
IGC_DIR  = ROOT / "igc"
OUT_ROOT = ROOT / "outputs" / "batch_csv"
BATCH_LOG = OUT_ROOT.parent / "batch_debug.log"  # outputs/batch_debug.log

# --- Tuning (read from tuning.json written by tuning_v1.py) ---
import json
TUNING_FILE = ROOT / "tuning.json"

def load_tuning() -> dict:
    if TUNING_FILE.exists():
        try:
            return json.loads(TUNING_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            raise RuntimeError(f"Failed to parse {TUNING_FILE}: {e}")
    raise FileNotFoundError(f"Tuning file not found: {TUNING_FILE}")

def logf_write(logf: Path, msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    logf.parent.mkdir(parents=True, exist_ok=True)
    with logf.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")

def wipe_run_dir(run_dir: Path) -> None:
    """Clear the per-flight run directory; preserve parent."""
    if run_dir.exists():
        for child in run_dir.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink(missing_ok=True)
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
    if not seq: return []
    out = [seq[0]]
    for a in seq[1:]:
        prev = out[-1]
        da = a - prev
        while da > 180: a -= 360; da = a - prev
        while da < -180: a += 360; da = a - prev
        out.append(a)
    return out

# ---------------------------------------------------------------------------
# 1) IGC parsing
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
                alt = float(gps_alt) if gps_alt is not None else float("nan")
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
# 2) Circle detection
# ---------------------------------------------------------------------------
def detect_circles(df: pd.DataFrame,
                   min_rotation_deg=300.0,
                   min_points=12,
                   min_duration_s=15.0,
                   max_duration_s=180.0) -> pd.DataFrame:
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
                    "lat": float(seg.lat.mean()),
                    "lon": float(seg.lon.mean()),
                    "t_start": float(seg.time_s.iloc[0]),
                    "t_end":   float(seg.time_s.iloc[-1]),
                    "climb_rate_ms": climb,
                    "alt_gain_m": alt_gain,
                    "duration_s": dur,
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
try:
    from sklearn.cluster import DBSCAN
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False

def cluster_circles(circles: pd.DataFrame,
                    eps_m: float = 200.0,
                    min_samples: int = 2) -> pd.DataFrame:
    base_cols = ["cluster_id","lat","lon","t_start","t_end",
                 "climb_rate_ms","climb_rate_ms_median","alt_gain_m_mean","duration_s_mean",
                 "n_circles"]
    if circles.empty:
        return pd.DataFrame(columns=base_cols)
    d = circles.dropna(subset=["lat","lon"]).copy()
    if d.empty: return pd.DataFrame(columns=base_cols)
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
        # union-find fallback
        pass
    rows = []
    for gid, sub in d.groupby("cluster_id"):
        rec = {
            "cluster_id": int(gid),
            "lat": float(sub["lat"].mean()),
            "lon": float(sub["lon"].mean()),
            "t_start": float(sub["t_start"].min()),
            "t_end": float(sub["t_end"].max()),
            "n_circles": len(sub),
            "climb_rate_ms": float(sub["climb_rate_ms"].mean()),
            "climb_rate_ms_median": float(sub["climb_rate_ms"].median()),
            "alt_gain_m_mean": float(sub["alt_gain_m"].mean()),
            "duration_s_mean": float(sub["duration_s"].mean()),
        }
        rows.append(rec)
    return pd.DataFrame(rows, columns=base_cols)

# ---------------------------------------------------------------------------
# 4) Altitude-only clusters
# ---------------------------------------------------------------------------
def detect_altitude_clusters(track: pd.DataFrame,
                             min_gain_m=30.0,
                             min_duration_s=20.0) -> pd.DataFrame:
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
                        "cluster_id": len(clusters),
                        "lat": float(seg.lat.mean()),
                        "lon": float(seg.lon.mean()),
                        "t_start": float(seg.time_s.iloc[0]),
                        "t_end": float(seg.time_s.iloc[-1]),
                        "climb_rate_ms": gain/dur if dur>0 else float("nan"),
                        "alt_gain_m": gain,
                        "duration_s": dur,
                    })
                start_i = None
    return pd.DataFrame(clusters)

# ---------------------------------------------------------------------------
# 5) Matching
# ---------------------------------------------------------------------------
def _overlap(a0,a1,b0,b1) -> Tuple[float,float]:
    lo = max(a0,b0); hi = min(a1,b1)
    ov = max(0.0, hi-lo)
    shorter = max(1e-9, min(a1-a0, b1-b0))
    return ov, ov/shorter

def match_clusters(circle_clusters, alt_clusters,
                   max_dist_m=600.0,
                   min_overlap_frac=0.25) -> Tuple[pd.DataFrame, dict]:
    cols = ["lat","lon","climb_rate_ms","alt_gain_m","duration_s",
            "circle_cluster_id","alt_cluster_id",
            "circle_lat","circle_lon","alt_lat","alt_lon",
            "dist_m","time_overlap_s","overlap_frac"]
    if circle_clusters.empty or alt_clusters.empty:
        return pd.DataFrame(columns=cols), {"pairs": 0}
    out = []
    for c in circle_clusters.itertuples(index=False):
        best = None
        best_key = (-1.0,float("inf"))
        for a in alt_clusters.itertuples(index=False):
            d = haversine_m(c.lat,c.lon,a.lat,a.lon)
            if d > max_dist_m: continue
            ov, of = _overlap(c.t_start,c.t_end,a.t_start,a.t_end)
            if of < min_overlap_frac: continue
            key = (of,d)
            if key > best_key:
                best_key = key
                best = (a,d,ov,of)
        if best:
            a, dist_m, ov, of = best
            out.append({
                "circle_cluster_id": int(getattr(c,"cluster_id",0)),
                "alt_cluster_id": int(getattr(a,"cluster_id",0)),
                "dist_m": dist_m,
                "time_overlap_s": ov,
                "overlap_frac": of,
                "circle_lat": c.lat, "circle_lon": c.lon,
                "alt_lat": a.lat, "alt_lon": a.lon,
                "lat": (c.lat+a.lat)/2,
                "lon": (c.lon+a.lon)/2,
                "alt_gain_m": a.alt_gain_m,
                "duration_s": a.duration_s,
                "climb_rate_ms": a.climb_rate_ms,
            })
    df = pd.DataFrame(out, columns=cols)
    return df, {"pairs": len(df)}

# ---------------------------------------------------------------------------
# MAIN batch
# ---------------------------------------------------------------------------
def main():
    import json
    from pathlib import Path

    # Ask for the IGC folder (default: ./igc)
    user_in = input(f"Enter IGC folder [default: {IGC_DIR}]: ").strip()
    igc_dir = Path(user_in) if user_in else IGC_DIR
    if not igc_dir.exists():
        print(f"[ERROR] folder not found: {igc_dir}")
        return 1

    igc_files = list(igc_dir.glob("*.igc"))
    if not igc_files:
        print(f"[WARN] no .igc files in {igc_dir}")
        return 0

    # ---- Global batch log (one file for the whole run) ----
    global_log = OUT_ROOT / "batch_debug.log"
    global_log.parent.mkdir(parents=True, exist_ok=True)
    # fresh log per run
    if global_log.exists():
        global_log.unlink()
    logf_write(global_log, "===== batch_run_v3.1 start =====")
    logf_write(global_log, f"IGC_DIR: {igc_dir}")

    # Load tuning once for the whole batch
    cfg = load_tuning()
    circle_eps_m       = float(cfg["circle_eps_m"])
    circle_min_samples = int(cfg["circle_min_samples"])
    alt_min_gain       = float(cfg["alt_min_gain"])
    alt_min_duration   = float(cfg["alt_min_duration"])
    match_max_dist_m   = float(cfg["match_max_dist_m"])
    match_min_overlap  = float(cfg["match_min_overlap"])

    # Echo & log tuning once
    tuneline = (f"TUNING: circle_eps_m={circle_eps_m}, circle_min_samples={circle_min_samples}, "
                f"alt_min_gain={alt_min_gain}, alt_min_duration={alt_min_duration}, "
                f"match_max_dist_m={match_max_dist_m}, match_min_overlap={match_min_overlap}")
    print(f"[TUNING] {tuneline}")
    logf_write(global_log, tuneline)

    # ---- Iterate flights ----
    for igc_path in sorted(igc_files):
        flight = igc_path.stem
        run_dir = OUT_ROOT / flight
        wipe_run_dir(run_dir)

        # per-flight log remains in the flight folder
        logf = run_dir / "pipeline_debug.log"
        logf_write(logf, f"===== batch_run_v3.1 start =====")
        logf_write(logf, f"IGC: {igc_path}")
        logf_write(logf, f"RUN_DIR: {run_dir}")
        logf_write(logf, tuneline)

        # Parse track
        track = parse_igc_brecords(igc_path)
        if track.empty:
            logf_write(logf, "[WARN] no B-records")
            logf_write(global_log, f"[SKIP] {flight} (no B-records)")
            continue

        # Circles
        circles = detect_circles(track)
        circles_csv = run_dir / "circles.csv"
        circles.to_csv(circles_csv, index=False)

        # Circle clusters
        cc = cluster_circles(circles, eps_m=circle_eps_m, min_samples=circle_min_samples)
        cc_csv = run_dir / "circle_clusters_enriched.csv"
        cc.to_csv(cc_csv, index=False)

        # Altitude clusters
        alts = detect_altitude_clusters(
            track,
            min_gain_m=alt_min_gain,
            min_duration_s=alt_min_duration,
        )
        alt_csv = run_dir / "altitude_clusters.csv"
        alts.to_csv(alt_csv, index=False)

        # Matching
        matches, stats = match_clusters(
            cc if not cc.empty else pd.DataFrame(columns=["cluster_id", "lat", "lon", "t_start", "t_end"]),
            alts,
            max_dist_m=match_max_dist_m,
            min_overlap_frac=match_min_overlap,
        )
        match_csv = run_dir / "matched_clusters.csv"
        matches.to_csv(match_csv, index=False)

        match_json = run_dir / "matched_clusters.json"
        match_json.write_text(json.dumps({"stats": stats, "rows": int(len(matches))}, indent=2), encoding="utf-8")

        # Per-flight log + global summary line
        logf_write(logf, f"[OK] wrote circles={len(circles)} cc={len(cc)} alts={len(alts)} matches={len(matches)}")
        logf_write(global_log, f"[BATCH] flight={flight} circles={len(circles)} cc={len(cc)} alts={len(alts)} matches={len(matches)}")

    logf_write(global_log, "[DONE]")
    return 0

if __name__ == "__main__":
    sys.exit(main())