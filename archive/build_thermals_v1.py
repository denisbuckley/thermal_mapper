#!/usr/bin/env python3
"""
build_thermals_v1.py
--------------------
Aggregate per-flight matched cluster points into persistent thermal locations
using spatial density clustering (DBSCAN, haversine).

Inputs:
  - Scans: outputs/batch_csv/**/matched_clusters.csv
  - Uses columns: lat, lon, climb_rate_ms (others are ignored)

Outputs:
  - CSV:     outputs/waypoints/thermal_waypoints_v1.csv
  - GeoJSON: outputs/waypoints/thermal_waypoints_v1.geojson
  - Log:     outputs/waypoints/grouping_debug.log

CLI:
  python build_thermals_v1.py \
    --inputs-root outputs/batch_csv \
    --method dbscan \
    --eps-km 5 \
    --min-samples 2 \
    --strength-min 1.0 \
    --out-csv outputs/waypoints/thermal_waypoints_v1.csv \
    --out-geojson outputs/waypoints/thermal_waypoints_v1.geojson \
    [--date-start YYYY-MM-DD] [--date-end YYYY-MM-DD] \
    [--dry-run]
"""

import argparse, json, math
from pathlib import Path
from datetime import datetime, date
from typing import Optional
import pandas as pd, numpy as np
from sklearn.cluster import DBSCAN

EARTH_R_M = 6_371_000.0

def log(msg: str, fh):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    print(line, end="")
    if fh: fh.write(line); fh.flush()

def parse_args():
    ap = argparse.ArgumentParser(description="Group matched clusters into persistent thermal locations.")
    # tunings: set default=None so we can prompt if not provided
    ap.add_argument("--method", choices=["dbscan","hdbscan","optics"], default=None)
    ap.add_argument("--eps-km", type=float, default=None)
    ap.add_argument("--min-samples", type=int, default=None)
    ap.add_argument("--strength-min", type=float, default=None)

    # keep paths & misc with real defaults (no prompting for these)
    ap.add_argument("--inputs-root", default="outputs/batch_csv")
    ap.add_argument("--out-csv", default="outputs/waypoints/thermal_waypoints_v1.csv")
    ap.add_argument("--out-geojson", default="outputs/waypoints/thermal_waypoints_v1.geojson")
    ap.add_argument("--debug-log", default="outputs/waypoints/grouping_debug.log")
    ap.add_argument("--date-start", default=None)
    ap.add_argument("--date-end", default=None)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()

def to_date(s: Optional[str]) -> Optional[date]:
    return None if not s else datetime.fromisoformat(s).date()

def great_circle_distance_m(lat1, lon1, lat2, lon2):
    dlat, dlon = lat2-lat1, lon2-lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return EARTH_R_M * 2*np.arcsin(np.minimum(1.0, np.sqrt(a)))

def robust_median(series: pd.Series) -> float:
    return float(np.median(series.values))

def load_all_matches(inputs_root: Path, logfh):
    rows = []
    for csv in inputs_root.glob("**/matched_clusters.csv"):
        try:
            df = pd.read_csv(csv)
            df["__source_file"] = str(csv)
            rows.append(df)
            log(f"Loaded {len(df)} rows from {csv}", logfh)
        except Exception as e:
            log(f"[WARN] Failed {csv}: {e}", logfh)

    # filter out empties to silence pandas FutureWarning
    rows = [df for df in rows if not df.empty]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def filter_inputs(df, strength_min, dstart, dend, logfh):
    need=["lat","lon","climb_rate_ms"]
    for c in need:
        if c not in df.columns: raise ValueError(f"Missing col: {c}")
    df2=df[df["climb_rate_ms"]>=strength_min].copy()

    # Drop absurd climb spikes/drops (units: m/s). Adjust if needed.
    pre = len(df2)
    df2 = df2[df2["climb_rate_ms"].between(-5.0, 10.0)]
    dropped = pre - len(df2)
    if dropped:
        log(f"[CLEAN] dropped {dropped} outliers outside [-5, 10] m/s", logfh)

    log(f"Strength filter ≥{strength_min}: {len(df)}→{len(df2)}", logfh)
    df2=df2.dropna(subset=["lat","lon"])
    return df2.reset_index(drop=True)

def run_dbscan_haversine(lat, lon, eps_km, min_samples):
    lat_rad, lon_rad=np.radians(lat), np.radians(lon)
    eps_rad=(eps_km*1000)/EARTH_R_M
    model=DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine")
    return model.fit_predict(np.c_[lat_rad, lon_rad])

def aggregate_clusters(df, labels, method, eps_km, min_samples, strength_min):
    """
    Return:
      - out: DataFrame of waypoints, including 'cluster_label' (DBSCAN label) and 'wp_id'
      - lab_to_wp: dict mapping original DBSCAN label -> wp_id
    """
    df = df.copy()
    df["__label"] = labels
    df = df[df["__label"] >= 0]  # keep only clustered points

    rows = []
    for lab, grp in df.groupby("__label"):
        lat_med, lon_med = robust_median(grp["lat"]), robust_median(grp["lon"])
        lat_rad, lon_rad = np.radians(grp["lat"]), np.radians(grp["lon"])
        r_m = max([
            great_circle_distance_m(np.radians(lat_med), np.radians(lon_med), a, b)
            for a, b in zip(lat_rad, lon_rad)
        ]) if len(grp) else 0.0
        rows.append({
            "cluster_label": lab,            # <-- keep DBSCAN label
            "lat": lat_med,
            "lon": lon_med,
            "radius_m": r_m,
            "strength_mean": float(np.mean(grp["climb_rate_ms"])),
            "strength_p95": float(np.percentile(grp["climb_rate_ms"], 95)),
            "method": method,
            "eps_km": eps_km,
            "min_samples": min_samples,
            "strength_min": strength_min,
            "encounters": int(len(grp)),     # convenience: points in this cluster
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out, {}

    # sort strongest/tightest first (as you had)
    out = out.sort_values(
        ["strength_p95", "strength_mean", "radius_m"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    # assign waypoint ids
    out.insert(0, "wp_id", np.arange(len(out)))

    # build mapping from DBSCAN label -> wp_id
    lab_to_wp = dict(zip(out["cluster_label"].tolist(), out["wp_id"].tolist()))
    return out, lab_to_wp

def write_geojson(df,path:Path):
    feats=[{"type":"Feature","geometry":{"type":"Point","coordinates":[r.lon,r.lat]},
            "properties":r.to_dict()} for _,r in df.iterrows()]
    fc={"type":"FeatureCollection","features":feats}
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w") as f: json.dump(fc,f,indent=2)

def main():
    # ---- parse once ----
    args = parse_args()
    inputs = Path(args.inputs_root)

    # ---- single-pass prompts for tunings (only if flag not given) ----
    def _ask(val, label, default, cast):
        if val is not None:
            return cast(val)
        raw = input(f"{label} [{default}]: ").strip()
        return cast(raw) if raw else cast(default)

    # Defaults for prompts (match your header docs)
    DEF_METHOD      = "dbscan"
    DEF_EPS_KM      = 1
    DEF_MIN_SAMPLES = 5
    DEF_STRENGTH    = 2

    method       = _ask(args.method,        "Clustering method (dbscan|hdbscan|optics)", DEF_METHOD, str).lower()
    eps_km       = _ask(args.eps_km,        "DBSCAN/OPTICS eps (km)",                    DEF_EPS_KM, float)
    min_samples  = _ask(args.min_samples,   "Min samples per cluster",                    DEF_MIN_SAMPLES, int)
    strength_min = _ask(args.strength_min,  "Min strength (weight) per point",           DEF_STRENGTH, float)

    print(f"[TUNING] method={method} eps_km={eps_km} min_samples={min_samples} strength_min={strength_min}")

    # ---- paths & logging ----
    out_csv = Path(args.out_csv)
    out_geo = Path(args.out_geojson)
    logp    = Path(args.debug_log)
    logp.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_geo.parent.mkdir(parents=True, exist_ok=True)

    def write_empty_outputs():
        pd.DataFrame(columns=["lat","lon","radius_m","strength_mean","strength_p95",
                              "method","eps_km","min_samples","strength_min","wp_id"]
                    ).to_csv(out_csv, index=False)
        with open(out_geo, "w") as f:
            json.dump({"type":"FeatureCollection","features":[]}, f, indent=2)

    with open(logp, "a") as lf:
        log("=== build_thermals_v1 start ===", lf)
        log(f"TUNING: method={method} eps_km={eps_km} min_samples={min_samples} strength_min={strength_min}", lf)

        # 1) load all matched_clusters
        df = load_all_matches(inputs, lf)
        if df.empty:
            log("[WARN] No matched_clusters.csv found; writing empty outputs.", lf)
            write_empty_outputs()
            log("=== done ===", lf)
            return 0

        # 2) filter by strength/date using the RESOLVED values (not args.*)
        df = filter_inputs(df, strength_min, to_date(args.date_start), to_date(args.date_end), lf)
        if df.empty:
            log("[INFO] No rows after filter; writing empty outputs.", lf)
            write_empty_outputs()
            log("=== done ===", lf)
            return 0

        # 🔍 Debug: check climb_rate_ms distribution
        print(df["climb_rate_ms"].describe())
        print(df["climb_rate_ms"].head(20))

        # 3) cluster using the RESOLVED values
        labels = run_dbscan_haversine(df.lat, df.lon, eps_km, min_samples)
        out, lab_to_wp = aggregate_clusters(df, labels, method, eps_km, min_samples, strength_min)
        log(f"Clusters: {len(out)}", lf)

        # 3) cluster using the RESOLVED values (run ONCE)
        labels = run_dbscan_haversine(df.lat, df.lon, eps_km, min_samples)
        out, lab_to_wp = aggregate_clusters(df, labels, method, eps_km, min_samples, strength_min)
        log(f"Clusters: {len(out)}", lf)

        # --- diagnostics on clustering ---
        total_pts   = len(labels)
        clustered_n = int((np.array(labels) >= 0).sum())
        noise_n     = total_pts - clustered_n
        uniq_labels = sorted({int(x) for x in labels if x >= 0})
        log(f"[DIAG] points total={total_pts}, clustered={clustered_n}, noise={noise_n}, unique_clusters={len(uniq_labels)}", lf)

        # --- membership mapping (each matched point → wp_id) ---
        members = []
        if not out.empty and lab_to_wp and clustered_n > 0:
            # ensure __source_file exists
            if "__source_file" not in df.columns:
                df["__source_file"] = ""

            for idx, lbl in enumerate(labels):
                if lbl < 0:
                    continue  # skip noise
                wp_id = lab_to_wp.get(int(lbl))
                if wp_id is None:
                    continue
                members.append({
                    "lat": float(df.lat.iloc[idx]),
                    "lon": float(df.lon.iloc[idx]),
                    "climb_rate_ms": float(df.climb_rate_ms.iloc[idx]),
                    "cluster_label": int(lbl),
                    "wp_id": int(wp_id),
                    "__source_file": str(df.__source_file.iloc[idx]),
                })

        # write membership CSV (per-point mapping)
        memb_df = pd.DataFrame(members, columns=[
            "lat","lon","climb_rate_ms","cluster_label","wp_id","__source_file"
        ])
        memb_csv = out_csv.with_name(out_csv.stem + "_membership.csv")
        memb_df.to_csv(memb_csv, index=False)
        log(f"[OK] wrote membership CSV {memb_csv} (rows={len(memb_df)})", lf)

        # --- cluster membership summary (log + separate CSV) ---
        cluster_sizes = pd.Series(labels)[pd.Series(labels) >= 0].value_counts().sort_index()
        for cid, size in cluster_sizes.items():
            log(f"  Cluster {int(cid)}: {int(size)} points", lf)
        if not cluster_sizes.empty:
            log(f"[STATS] Avg pts/cluster: {cluster_sizes.mean():.1f}", lf)

        # persist sizes to a DIFFERENT file (avoid overwriting membership)
        sizes_csv = out_csv.with_name(out_csv.stem + "_cluster_sizes.csv")
        cluster_sizes.rename("points").to_csv(sizes_csv, header=True)
        log(f"[OK] wrote cluster size stats → {sizes_csv}", lf)
        # 4) write artifacts (always)
        if not args.dry_run:
            out.to_csv(out_csv, index=False)
            write_geojson(out, out_geo)
            log(f"[OK] wrote {out_csv} & {out_geo}", lf)

        log("=== done ===", lf)
    return 0

if __name__=="__main__": raise SystemExit(main())