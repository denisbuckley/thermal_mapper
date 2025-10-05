#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_v4.py — MONOLITHIC (no subprocesses, no imports of your v1 scripts)

Flow for ONE IGC:
  1) Parse IGC B-records → track DataFrame
  2) Detect circling segments (“circles”) → circles.csv
  3) Cluster circles → circle_clusters_enriched.csv
  4) Detect altitude-only climb segments → altitude_clusters.csv
  5) Match circle clusters ↔ altitude clusters → matched_clusters.csv + .json

I/O (run from repo ROOT):
  inputs:  ./igc/<file>.igc  (enter filename when prompted, or pass as CLI arg)
  outputs: ./outputs/batch_csv/<flight_stem>/
             circles.csv
             circle_clusters_enriched.csv
             altitude_clusters.csv
             matched_clusters.csv
             matched_clusters.json
             pipeline_debug.log

Tuning flags:
  --circle-eps-m (m, default 200)
  --circle-min-samples (default 2)
  --alt-min-gain (m, default 30)
  --alt-min-duration (s, default 20)
  --match-max-dist-m (m, default 600)
  --match-min-overlap (0..1, default 0.25)
"""

from __future__ import annotations
import argparse, sys, math, time, json
from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd

# Optional scikit-learn for DBSCAN
try:
    from sklearn.cluster import DBSCAN
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False

# near the top with other imports
try:
    from tuning_v1 import (
        circle_eps_m, circle_min_samples,
        alt_min_gain, alt_min_duration,
        match_max_dist_m, match_min_overlap,
    )
except Exception:
    # fallbacks if tuning_v1.py is missing
    circle_eps_m = 200.0
    circle_min_samples = 2
    alt_min_gain = 30.0
    alt_min_duration = 20.0
    match_max_dist_m = 600.0
    match_min_overlap = 0.25

# ---------------------------------------------------------------------------
# Roots and helpers
# ---------------------------------------------------------------------------
ROOT     = Path.cwd()
IGC_DIR  = ROOT / "igc"
OUT_ROOT = ROOT / "outputs" / "batch_csv"

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
    R = 6371008.8  # meters
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
# 1) IGC parsing (B-records)
# ---------------------------------------------------------------------------
def _decode_lat(dms: str, hemi: str) -> float:
    # dms 'DDMMmmm'
    deg = int(dms[0:2])
    mmm = int(dms[2:]) / 1000.0
    val = deg + (mmm/60.0)
    return -val if hemi.upper() == 'S' else val

def _decode_lon(dms: str, hemi: str) -> float:
    # dms 'DDDMMmmm'
    deg = int(dms[0:3])
    mmm = int(dms[3:]) / 1000.0
    val = deg + (mmm/60.0)
    return -val if hemi.upper() == 'W' else val

def parse_igc_brecords(igc_path: Path) -> pd.DataFrame:
    """
    Minimal B-record parser → DataFrame columns: time_s, lat, lon, alt
    Prefers GPS altitude if present; otherwise ffill/bfill.
    """
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
# 2) Circle detection (circling segments)
# ---------------------------------------------------------------------------
def detect_circles(df: pd.DataFrame,
                   min_rotation_deg: float = 300.0,
                   min_points: int = 12,
                   min_duration_s: float = 15.0,
                   max_duration_s: float = 180.0) -> pd.DataFrame:
    """
    Circling via cumulative heading-change heuristic.
    Output per segment:
      t_start, t_end, duration_s, lat, lon, alt_gain_m, climb_rate_ms
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
                # clamp overlong
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
# 3) Cluster circles (spatial) — includes cluster-level climb_rate_ms
# ---------------------------------------------------------------------------
def cluster_circles(circles: pd.DataFrame,
                    eps_m: float = 200.0,
                    min_samples: int = 2) -> pd.DataFrame:
    """
    Cluster circle centroids into thermal “spots”.

    Returns columns:
      cluster_id, lat, lon, t_start, t_end, n_circles,
      climb_rate_ms, climb_rate_ms_median, alt_gain_m_mean, duration_s_mean
    """
    base_cols = [
        "cluster_id","lat","lon","t_start","t_end","n_circles",
        "climb_rate_ms","climb_rate_ms_median","alt_gain_m_mean","duration_s_mean"
    ]
    if circles.empty:
        return pd.DataFrame(columns=base_cols)

    d = circles.dropna(subset=["lat","lon"]).copy()
    if d.empty:
        return pd.DataFrame(columns=base_cols)

    # --- label points into clusters (DBSCAN if available; else union-find fallback)
    if _HAVE_SKLEARN:
        import numpy as np
        latr = np.radians(d["lat"].to_numpy())
        lonr = np.radians(d["lon"].to_numpy())
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
        root_to_id = {}; next_id = 0
        labels = []
        for i in idx:
            r = find(i)
            if r not in root_to_id:
                root_to_id[r] = next_id; next_id += 1
            labels.append(root_to_id[r])
        d = d.reset_index(drop=True)
        d["cluster_id"] = labels

    if d.empty:
        return pd.DataFrame(columns=base_cols)

    # --- aggregate per cluster, carrying climb stats
    has_climb = "climb_rate_ms" in d.columns
    has_gain  = "alt_gain_m"    in d.columns
    has_dur   = "duration_s"    in d.columns

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
        if has_climb and len(sub["climb_rate_ms"].dropna()):
            rec["climb_rate_ms"] = float(sub["climb_rate_ms"].mean())
            rec["climb_rate_ms_median"] = float(sub["climb_rate_ms"].median())
        if has_gain and len(sub["alt_gain_m"].dropna()):
            rec["alt_gain_m_mean"] = float(sub["alt_gain_m"].mean())
        if has_dur and len(sub["duration_s"].dropna()):
            rec["duration_s_mean"] = float(sub["duration_s"].mean())

        rows.append(rec)

    df = pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)
    # Deterministic column order
    return df[[c for c in base_cols if c in df.columns]]


# ---------------------------------------------------------------------------
# 4) Altitude-only climb clusters
# ---------------------------------------------------------------------------
def detect_altitude_clusters(track: pd.DataFrame,
                             min_gain_m: float = 30.0,
                             min_duration_s: float = 20.0) -> pd.DataFrame:
    """
    Climb segments as contiguous positive-altitude periods meeting thresholds.
    Output:
      t_start, t_end, duration_s, alt_gain_m, climb_rate_ms, lat, lon
    """
    cols = ["t_start","t_end","duration_s","alt_gain_m","climb_rate_ms","lat","lon"]
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
                        "t_start": float(seg.time_s.iloc[0]),
                        "t_end":   float(seg.time_s.iloc[-1]),
                        "duration_s": dur,
                        "alt_gain_m": gain,
                        "climb_rate_ms": gain/dur if dur>0 else float("nan"),
                        "lat": float(seg.lat.mean()),
                        "lon": float(seg.lon.mean()),
                    })
                start_i = None
    # close at EOF
    if start_i is not None:
        seg = track.iloc[start_i:len(track)]
        gain = float(seg.alt.iloc[-1] - seg.alt.iloc[0])
        dur  = float(seg.time_s.iloc[-1] - seg.time_s.iloc[0])
        if gain >= min_gain_m and dur >= min_duration_s:
            clusters.append({
                "t_start": float(seg.time_s.iloc[0]),
                "t_end":   float(seg.time_s.iloc[-1]),
                "duration_s": dur,
                "alt_gain_m": gain,
                "climb_rate_ms": gain/dur if dur>0 else float("nan"),
                "lat": float(seg.lat.mean()),
                "lon": float(seg.lon.mean()),
            })

    df = pd.DataFrame(clusters)
    # Deterministic column order
    return df[cols] if not df.empty else pd.DataFrame(columns=cols)


# ---------------------------------------------------------------------------
# 5) Matching (circle clusters ↔ altitude clusters)
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
    For each circle cluster, pick the best altitude cluster by:
      - centroid distance <= max_dist_m
      - temporal overlap fraction >= min_overlap_frac
      - tie-breaker: higher overlap, then shorter distance
    Output DataFrame columns:
      circle_cluster_id, alt_cluster_id, dist_m, time_overlap_s, overlap_frac,
      circle_lat, circle_lon, alt_lat, alt_lon,
      lat, lon, alt_gain_m, duration_s, climb_rate_ms
    """
    cols = ["circle_cluster_id","alt_cluster_id","dist_m","time_overlap_s","overlap_frac",
            "circle_lat","circle_lon","alt_lat","alt_lon",
            "lat","lon","alt_gain_m","duration_s","climb_rate_ms"]

    if circle_clusters.empty or alt_clusters.empty:
        return pd.DataFrame(columns=cols), {"pairs": 0}

    out = []
    used = set()
    for c in circle_clusters.itertuples(index=False):
        best = None
        best_key = (-1.0, float("inf"))  # overlap first, then distance
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
                "circle_cluster_id": int(getattr(c, "cluster_id", 0)),
                "alt_cluster_id":    int(getattr(a, "cluster_id", 0)) if "cluster_id" in alt_clusters.columns else 0,
                "dist_m": float(dist_m),
                "time_overlap_s": float(ov),
                "overlap_frac": float(of),
                "circle_lat": float(c.lat), "circle_lon": float(c.lon),
                "alt_lat": float(a.lat),   "alt_lon": float(a.lon),
                # unified point = midpoint (simple, deterministic)
                "lat": float((c.lat + a.lat)/2),
                "lon": float((c.lon + a.lon)/2),
                "alt_gain_m": float(getattr(a, "alt_gain_m", float("nan"))),
                "duration_s": float(getattr(a, "duration_s", float("nan"))),
                "climb_rate_ms": float(getattr(a, "climb_rate_ms", float("nan"))),
            })
            used.add(int(getattr(a, "cluster_id", -1)))

    df = pd.DataFrame(out, columns=cols)
    # Deterministic column order
    df = df[cols] if not df.empty else pd.DataFrame(columns=cols)
    stats = {
        "pairs": int(len(df)),
        "max_dist_m": float(max_dist_m),
        "min_overlap_frac": float(min_overlap_frac),
    }
    return df, stats


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "igc",
        nargs="?",
        help="Path to IGC file (absolute, or just the filename under ./igc)"
    )
    args = ap.parse_args()

    # Resolve IGC path: CLI arg or prompt (no default)
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

    flight = igc_path.stem
    run_dir = OUT_ROOT / flight
    wipe_run_dir(run_dir)
    logf = run_dir / "pipeline_debug.log"
    logf_write(logf, "===== pipeline_v4 (monolithic) start =====")
    logf_write(logf, f"IGC: {igc_path}")
    logf_write(logf, f"RUN_DIR: {run_dir}")

    # --- log tuning parameters here ---
    logf_write(logf, f"TUNING: circle_eps_m={circle_eps_m}, circle_min_samples={circle_min_samples}, "
                     f"alt_min_gain={alt_min_gain}, alt_min_duration={alt_min_duration}, "
                     f"match_max_dist_m={match_max_dist_m}, match_min_overlap={match_min_overlap}")

    # --- 1) Track
    track = parse_igc_brecords(igc_path)
    if track.empty:
        logf_write(logf, "[WARN] No B-records parsed; exiting.")
        print("[WARN] No B-records parsed; exiting.")
        return 0

    # --- 2) Circles
    circles = detect_circles(track)
    # enforce canonical order for circles.csv
    circles_cols = ["lat","lon","t_start","t_end","climb_rate_ms", "alt_gain_m","duration_s"]
    if not circles.empty:
        circles = circles.reindex(columns=[c for c in circles_cols if c in circles.columns])
    else:
        circles = pd.DataFrame(columns=circles_cols)

    circles_csv = run_dir / "circles.csv"
    circles.to_csv(circles_csv, index=False)
    logf_write(logf, f"[OK] wrote {len(circles)} circles → {circles_csv}")

    # --- 3) Circle clusters
    cc = cluster_circles(circles, eps_m=circle_eps_m, min_samples=circle_min_samples)    # enforce canonical order for circle_clusters_enriched.csv
    cc_cols = [
        "cluster_id", "lat", "lon", "t_start", "t_end",
        "climb_rate_ms", "climb_rate_ms_median", "alt_gain_m_mean", "duration_s_mean",
        "n_circles"
    ]

    if not cc.empty:
        # ensure any missing canonical columns exist
        for k in cc_cols:
            if k not in cc.columns:
                cc[k] = pd.NA
        cc = cc[cc_cols]
    else:
        cc = pd.DataFrame(columns=cc_cols)

    cc_csv = run_dir / "circle_clusters_enriched.csv"
    cc.to_csv(cc_csv, index=False)
    logf_write(logf, f"[OK] wrote {len(cc)} circle clusters → {cc_csv}")

    # --- 4) Altitude clusters
    alts = detect_altitude_clusters(track, min_gain_m=alt_min_gain, min_duration_s=alt_min_duration)

    # ensure cluster_id exists and enforce canonical order
    ALT_COLS = ["cluster_id", "lat", "lon", "t_start", "t_end",
                "climb_rate_ms", "alt_gain_m", "duration_s"]

    if alts.empty:
        alts = pd.DataFrame(columns=ALT_COLS)
    else:
        # synthesize cluster_id if missing
        if "cluster_id" not in alts.columns:
            alts = alts.reset_index(drop=True)
            alts["cluster_id"] = alts.index

        # guarantee presence + order
        for k in ALT_COLS:
            if k not in alts.columns:
                alts[k] = pd.NA
        alts = alts[ALT_COLS]

    alt_csv = run_dir / "altitude_clusters.csv"
    alts.to_csv(alt_csv, index=False)
    logf_write(logf, f"[OK] wrote {len(alts)} altitude clusters → {alt_csv}")


    # --- 5) Matching
    cc_for_match = cc.rename(columns={"cluster_id":"cluster_id"})  # already good
    matches, stats = match_clusters(
        cc_for_match if not cc_for_match.empty else pd.DataFrame(
            columns=["cluster_id", "lat", "lon", "t_start", "t_end"]),
        alts,
        max_dist_m=match_max_dist_m,
        min_overlap_frac=match_min_overlap
    )
    parse_igc_brecords
    # enforce canonical order for matched_clusters.csv
    match_cols = [
        "lat", "lon", "climb_rate_ms", "alt_gain_m", "duration_s",
        "circle_cluster_id", "alt_cluster_id",
        "circle_lat", "circle_lon", "alt_lat", "alt_lon",
        "dist_m", "time_overlap_s", "overlap_frac"
    ]
    if not matches.empty:
        matches = matches.reindex(columns=match_cols)
    else:
        matches = pd.DataFrame(columns=match_cols)

    match_csv  = run_dir / "matched_clusters.csv"
    matches.to_csv(match_csv, index=False)

    match_json = run_dir / "matched_clusters.json"
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
    match_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    logf_write(logf, f"[OK] wrote {len(matches)} matches → {match_csv}")
    logf_write(logf, "[DONE]")
    print(f"[DONE] Artifacts in: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())