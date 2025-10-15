#!/usr/bin/env python3
"""
build_thermals_v1a.py
---------------------
Aggregate per-flight matched cluster points into persistent thermal locations
using spatial density clustering (DBSCAN, haversine), with centers computed from
DBSCAN **core points** only (center-estimator selectable; default = median).

Key features
- Encounters = **unique flights (IGC files)** contributing points to the cluster
- Center estimator (core points): geomedian | median | medoid  (default: median)
- Info box (CSV/GeoJSON) in strict order (prefixed keys in GeoJSON to lock order)
- Rounding: 1 decimal for all numeric fields EXCEPT lat/lon (full precision)
- strength_* and core_fraction are written as **1-decimal strings** for display
- Membership CSV includes flight_id and is_core for traceability

Inputs:
  - Scans: outputs/batch_csv/**/matched_clusters.csv
  - Uses columns: lat, lon, climb_rate_ms (others ignored unless present for flight_id derivation)

Outputs:
  - CSV:     outputs/waypoints/thermal_waypoints_v1.csv
  - GeoJSON: outputs/waypoints/thermal_waypoints_v1.geojson
  - Log:     outputs/waypoints/grouping_debug.log
  - Membership (points): outputs/waypoints/thermal_waypoints_v1_membership.csv
  - Cluster sizes:       outputs/waypoints/thermal_waypoints_v1_cluster_sizes.csv

CLI/IDE:
  You can run from IDE; if a tuning flag is omitted, the script will prompt once.
  Typical CLI:
    python build_thermals_v1a.py \
      --inputs-root outputs/batch_csv \
      --method dbscan \
      --eps-km 1 \
      --min-samples 5 \
      --strength-min 2 \
      --center-estimator median \
      --out-csv outputs/waypoints/thermal_waypoints_v1.csv \
      --out-geojson outputs/waypoints/thermal_waypoints_v1.geojson
"""

import argparse, json, math
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Tuple, Dict, List
import pandas as pd, numpy as np
from sklearn.cluster import DBSCAN

EARTH_R_M = 6_371_000.0

# --------- Logging ----------
def log(msg: str, fh):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    print(line, end="")
    if fh: fh.write(line); fh.flush()

# --------- Args ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Group matched clusters into persistent thermal locations.")
    # tunings
    ap.add_argument("--method", choices=["dbscan","hdbscan","optics"], default=None)
    ap.add_argument("--eps-km", type=float, default=None)
    ap.add_argument("--min-samples", type=int, default=None)
    ap.add_argument("--strength-min", type=float, default=None)
    ap.add_argument("--center-estimator", choices=["geomedian","median","medoid"], default=None)

    # paths & misc
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

# --------- Geo ----------
def great_circle_distance_m(lat1, lon1, lat2, lon2):
    dlat, dlon = lat2-lat1, lon2-lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return EARTH_R_M * 2*np.arcsin(np.minimum(1.0, np.sqrt(a)))

def robust_median(series: pd.Series) -> float:
    return float(np.median(series.values))

# --------- IO ----------
def load_all_matches(inputs_root: Path, logfh):
    rows = []
    for csvp in inputs_root.glob("**/matched_clusters.csv"):
        try:
            df = pd.read_csv(csvp)
            df["__source_file"] = str(csvp)
            rows.append(df)
            log(f"Loaded {len(df)} rows from {csvp}", logfh)
        except Exception as e:
            log(f"[WARN] Failed {csvp}: {e}", logfh)
    rows = [df for df in rows if not df.empty]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def filter_inputs(df, strength_min, dstart, dend, logfh):
    need=["lat","lon","climb_rate_ms"]
    for c in need:
        if c not in df.columns: raise ValueError(f"Missing col: {c}")
    df2=df[df["climb_rate_ms"]>=strength_min].copy()

    # outlier clamp
    pre = len(df2)
    df2 = df2[df2["climb_rate_ms"].between(-5.0, 10.0)]
    dropped = pre - len(df2)
    if dropped:
        log(f"[CLEAN] dropped {dropped} outliers outside [-5, 10] m/s", logfh)

    log(f"Strength filter ≥{strength_min}: {len(df)}→{len(df2)}", logfh)
    df2=df2.dropna(subset=["lat","lon"])
    return df2.reset_index(drop=True)

# --------- flight_id (encounters) ----------
def _derive_flight_id_rowlike(row) -> str:
    """Best-effort flight identifier."""
    for c in ("igc_file","igc_path","flight_id","track_id","source_igc"):
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            p = Path(str(row[c]).strip())
            return p.stem or p.name
    if "__source_file" in row and pd.notna(row["__source_file"]):
        p = Path(str(row["__source_file"]))
        # common layout: .../<flight_id>/matched_clusters.csv
        return p.parent.name or p.stem
    return "unknown"

def attach_flight_id_column(df: pd.DataFrame) -> pd.DataFrame:
    if "flight_id" in df.columns:
        df = df.copy()
        df["flight_id"] = df["flight_id"].apply(lambda v: Path(str(v)).stem if pd.notna(v) else "unknown")
        return df
    df = df.copy()
    df["flight_id"] = df.apply(_derive_flight_id_rowlike, axis=1)
    return df

# --------- DBSCAN ----------
def run_dbscan_haversine(lat, lon, eps_km, min_samples) -> Tuple[np.ndarray, set]:
    lat_rad, lon_rad=np.radians(lat), np.radians(lon)
    eps_rad=(eps_km*1000)/EARTH_R_M
    model=DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine")
    labels = model.fit_predict(np.c_[lat_rad, lon_rad])
    core_idx = set(model.core_sample_indices_.tolist()) if hasattr(model,"core_sample_indices_") else set()
    return labels, core_idx

# --------- Centers on CORE points ----------
def geomedian_core(grp_core: pd.DataFrame) -> Tuple[float,float]:
    """Weiszfeld-like iterative geometric median on lat/lon in radians."""
    pts = np.radians(grp_core[["lat","lon"]].to_numpy(dtype=float))
    if len(pts)==0:
        return robust_median(grp_core["lat"]), robust_median(grp_core["lon"])
    x = pts.mean(axis=0).copy()
    for _ in range(32):
        d = np.linalg.norm(pts - x, axis=1)
        d = np.where(d<1e-12, 1e-12, d)
        w = 1.0/d
        x_new = (pts * w[:,None]).sum(axis=0) / w.sum()
        if np.linalg.norm(x_new - x) < 1e-12: break
        x = x_new
    return float(np.degrees(x[0])), float(np.degrees(x[1]))

def medoid_core(grp_core: pd.DataFrame) -> Tuple[float,float]:
    """Pick the core point minimizing sum of haversine distances to other core points."""
    pts = grp_core[["lat","lon"]].to_numpy(dtype=float)
    if len(pts)==0:
        return robust_median(grp_core["lat"]), robust_median(grp_core["lon"])
    latr = np.radians(pts[:,0])[:,None]
    lonr = np.radians(pts[:,1])[:,None]
    # pairwise distances
    dists = np.zeros((len(pts), len(pts)), dtype=float)
    for i in range(len(pts)):
        dists[i,:] = great_circle_distance_m(latr[i,0], lonr[i,0], latr[:,0], lonr[:,0])
    s = dists.sum(axis=1)
    i = int(np.argmin(s))
    return float(pts[i,0]), float(pts[i,1])

def choose_center(grp_core: pd.DataFrame, estimator: str) -> Tuple[float,float]:
    estimator = (estimator or "median").lower()
    if estimator == "geomedian":
        return geomedian_core(grp_core)
    if estimator == "medoid":
        return medoid_core(grp_core)
    # default: component-wise median
    return robust_median(grp_core["lat"]), robust_median(grp_core["lon"])

# --------- Aggregation ----------
CSV_ORDER = [
    "strength_mean_core",
    "radius_m",
    "core_count",
    "border_count",
    "encounters",
    "core_fraction",
    "radius_p80_m",
    "strength_p95_core",
    "strength_mean_all",
    "strength_p95_all",
    "eps_km",
    "min_samples",
    "method",
    "center_estimator",
    "strength_min",
    "wp_id",
    "lat",
    "lon",
]

def aggregate_clusters(df, labels, core_idx: set, method, eps_km, min_samples, strength_min, center_estimator):
    """
    Returns:
      - csv_df: DataFrame with columns in CSV_ORDER
      - lab_to_wp: dict DBSCAN label -> wp_id
    """
    df = df.copy()
    df["__label"] = labels
    df = df[df["__label"] >= 0]  # clustered only

    rows = []
    for lab, grp in df.groupby("__label"):
        # split core/border by original indexing
        idxs = grp.index.to_numpy()
        core_mask = np.array([i in core_idx for i in idxs])
        grp_core = grp[core_mask]
        grp_all  = grp

        core_count   = int(len(grp_core))
        border_count = int(len(grp_all) - len(grp_core))
        core_fraction = float(core_count / max(1, len(grp_all)))

        # center from CORE points
        ctr_lat, ctr_lon = choose_center(grp_core if core_count>0 else grp_all, center_estimator)

        # radii from all points relative to center
        lat0, lon0 = np.radians(ctr_lat), np.radians(ctr_lon)
        lat_all, lon_all = np.radians(grp_all["lat"].values), np.radians(grp_all["lon"].values)
        dists_m = np.array([
            great_circle_distance_m(lat0, lon0, a, b) for a,b in zip(lat_all, lon_all)
        ]) if len(grp_all) else np.array([0.0])
        radius_m    = float(np.max(dists_m)) if len(dists_m) else 0.0
        radius_p80  = float(np.percentile(dists_m, 80)) if len(dists_m) else 0.0

        # strength metrics
        def p95(x): return float(np.percentile(x, 95)) if len(x) else 0.0
        strength_mean_core = float(np.mean(grp_core["climb_rate_ms"])) if core_count>0 else float(np.mean(grp_all["climb_rate_ms"]))
        strength_p95_core  = p95(grp_core["climb_rate_ms"].values) if core_count>0 else p95(grp_all["climb_rate_ms"].values)
        strength_mean_all  = float(np.mean(grp_all["climb_rate_ms"]))
        strength_p95_all   = p95(grp_all["climb_rate_ms"].values)

        # encounters = UNIQUE flights
        encounters = int(grp_all["flight_id"].nunique()) if "flight_id" in grp_all.columns else int(len(grp_all))

        rows.append({
            "cluster_label": int(lab),
            "lat": ctr_lat,
            "lon": ctr_lon,
            "radius_m": radius_m,
            "radius_p80_m": radius_p80,
            "core_count": core_count,
            "border_count": border_count,
            "core_fraction": core_fraction,
            "strength_mean_core": strength_mean_core,
            "strength_p95_core": strength_p95_core,
            "strength_mean_all": strength_mean_all,
            "strength_p95_all": strength_p95_all,
            "method": method,
            "eps_km": eps_km,
            "min_samples": min_samples,
            "strength_min": strength_min,
            "encounters": encounters,
            "center_estimator": center_estimator,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=CSV_ORDER), {}

    # sort strongest/tightest first using core metrics
    out = out.sort_values(
        ["strength_p95_core", "strength_mean_core", "radius_m"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    # assign waypoint ids
    out.insert(0, "wp_id", np.arange(len(out)))

    lab_to_wp = dict(zip(out["cluster_label"].tolist(), out["wp_id"].tolist()))

    # Final CSV columns in exact order (skip cluster_label)
    csv_df = out[["strength_mean_core","radius_m","core_count","border_count","encounters",
                  "core_fraction","radius_p80_m","strength_p95_core","strength_mean_all",
                  "strength_p95_all","eps_km","min_samples","method","center_estimator",
                  "strength_min","wp_id","lat","lon"]].copy()

    # Round numeric columns to 1 decimal except lat/lon; keep counts ints
    skip_round = {"lat","lon"}
    for col in CSV_ORDER:
        if col in skip_round or col not in csv_df.columns: continue
        if np.issubdtype(csv_df[col].dtype, np.number):
            # counts stay ints
            if col in {"core_count","border_count","encounters","min_samples","wp_id"}:
                csv_df[col] = csv_df[col].astype(int)
            else:
                csv_df[col] = csv_df[col].round(1)

    # Force 1-decimal STRING formatting for strength_* and core_fraction
    for col in ["strength_mean_core","strength_p95_core","strength_mean_all",
                "strength_p95_all","core_fraction"]:
        if col in csv_df.columns:
            csv_df[col] = csv_df[col].apply(lambda x: (f"{float(x):.1f}" if pd.notna(x) else ""))

    return csv_df, lab_to_wp

# --------- GeoJSON (ordered popup) ----------
def write_geojson_prefixed(df: pd.DataFrame, path: Path):
    feats=[]
    for _,r in df.iterrows():
        props={}
        for i,k in enumerate(CSV_ORDER, start=1):
            key=f"{i:02d}_{k}"
            v=r[k]
            # Keep lat/lon numeric, full precision
            if k in ("lat","lon"):
                props[key]=float(v)
                continue
            # Force 1-decimal STRING for strength_* and core_fraction
            if k.startswith("strength_") or k=="core_fraction":
                try: props[key]=f"{float(v):.1f}"
                except: props[key]=str(v)
                continue
            # Integers preserved
            if k in {"core_count","border_count","encounters","min_samples","wp_id"}:
                try: props[key]=int(v)
                except: props[key]=v
                continue
            # Others rounded to 1 decimal numeric
            try:
                props[key]=round(float(v),1)
            except:
                props[key]=v

        feats.append({
            "type":"Feature",
            "geometry":{"type":"Point","coordinates":[float(r["lon"]), float(r["lat"])]},
            "properties":props
        })
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w") as f: json.dump({"type":"FeatureCollection","features":feats}, f, indent=2)

# --------- Main ----------
def main():
    args = parse_args()
    inputs = Path(args.inputs_root)

    # prompts when run in IDE if flags omitted
    def _ask(val, label, default, cast):
        if val is not None: return cast(val)
        raw = input(f"{label} [{default}]: ").strip()
        return cast(raw) if raw else cast(default)

    DEF_METHOD, DEF_EPS_KM, DEF_MIN_SAMPLES, DEF_STRENGTH, DEF_CENTER = "dbscan", 1.0, 5, 2.0, "median"
    method       = _ask(args.method,        "Clustering method (dbscan|hdbscan|optics)", DEF_METHOD, str).lower()
    eps_km       = _ask(args.eps_km,        "DBSCAN/OPTICS eps (km)",                    DEF_EPS_KM, float)
    min_samples  = _ask(args.min_samples,   "Min samples per cluster",                   DEF_MIN_SAMPLES, int)
    strength_min = _ask(args.strength_min,  "Min strength (weight) per point",          DEF_STRENGTH, float)
    center_estimator = _ask(args.center_estimator, "Center estimator (geomedian|median|medoid)", DEF_CENTER, str).lower()

    out_csv = Path(args.out_csv)
    out_geo = Path(args.out_geojson)
    logp    = Path(args.debug_log)
    logp.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_geo.parent.mkdir(parents=True, exist_ok=True)

    def write_empty_outputs():
        pd.DataFrame(columns=CSV_ORDER).to_csv(out_csv, index=False)
        with open(out_geo, "w") as f:
            json.dump({"type":"FeatureCollection","features":[]}, f, indent=2)

    with open(logp, "a") as lf:
        log("=== build_thermals_v1a start ===", lf)
        log(f"TUNING: method={method} eps_km={eps_km} min_samples={min_samples} "
            f"strength_min={strength_min} center_estimator={center_estimator}", lf)

        df = load_all_matches(inputs, lf)
        if df.empty:
            log("[WARN] No matched_clusters.csv found; writing empty outputs.", lf)
            write_empty_outputs(); log("=== done ===", lf); return 0

        df = filter_inputs(df, strength_min, to_date(args.date_start), to_date(args.date_end), lf)
        if df.empty:
            log("[INFO] No rows after filter; writing empty outputs.", lf)
            write_empty_outputs(); log("=== done ===", lf); return 0

        # attach flight_id for encounters
        df = attach_flight_id_column(df)

        # debug distribution
        print(df["climb_rate_ms"].describe())
        print(df[["climb_rate_ms","lat","lon"]].head(10))

        # clustering
        labels, core_idx = run_dbscan_haversine(df.lat, df.lon, eps_km, min_samples)
        csv_df, lab_to_wp = aggregate_clusters(df, labels, core_idx, method, eps_km, min_samples, strength_min, center_estimator)
        log(f"Clusters (waypoints): {len(csv_df)}", lf)

        # diagnostics
        total_pts   = len(labels)
        clustered_n = int((np.array(labels) >= 0).sum())
        noise_n     = total_pts - clustered_n
        uniq_labels = sorted({int(x) for x in labels if x >= 0})
        log(f"[DIAG] points total={total_pts}, clustered={clustered_n}, noise={noise_n}, unique_clusters={len(uniq_labels)}", lf)

        # membership per point
        members = []
        if not csv_df.empty and lab_to_wp and clustered_n > 0:
            if "__source_file" not in df.columns: df["__source_file"] = ""
            for idx, lbl in enumerate(labels):
                if lbl < 0: continue
                wp_id = lab_to_wp.get(int(lbl))
                if wp_id is None: continue
                members.append({
                    "lat": float(df.lat.iloc[idx]),
                    "lon": float(df.lon.iloc[idx]),
                    "climb_rate_ms": float(df.climb_rate_ms.iloc[idx]),
                    "cluster_label": int(lbl),
                    "wp_id": int(wp_id),
                    "flight_id": str(df.flight_id.iloc[idx]) if "flight_id" in df.columns else "",
                    "is_core": bool(idx in core_idx),
                    "__source_file": str(df.__source_file.iloc[idx]),
                })
        memb_df = pd.DataFrame(members, columns=[
            "lat","lon","climb_rate_ms","cluster_label","wp_id","flight_id","is_core","__source_file"
        ])
        memb_csv = out_csv.with_name(out_csv.stem + "_membership.csv")
        memb_df.to_csv(memb_csv, index=False)
        log(f"[OK] wrote membership CSV {memb_csv} (rows={len(memb_df)})", lf)

        # cluster sizes (points-per-cluster for raw diagnostic)
        cluster_sizes = pd.Series(labels)[pd.Series(labels) >= 0].value_counts().sort_index()
        for cid, size in cluster_sizes.items():
            log(f"  Cluster {int(cid)}: {int(size)} points", lf)
        if not cluster_sizes.empty:
            log(f"[STATS] Avg pts/cluster: {cluster_sizes.mean():.1f}", lf)
        sizes_csv = out_csv.with_name(out_csv.stem + "_cluster_sizes.csv")
        cluster_sizes.rename("points").to_csv(sizes_csv, header=True)
        log(f"[OK] wrote cluster size stats → {sizes_csv}", lf)

        # write artifacts
        if not args.dry_run:
            csv_df.to_csv(out_csv, index=False)
            write_geojson_prefixed(csv_df, out_geo)
            log(f"[OK] wrote {out_csv} & {out_geo}", lf)

        log("=== done ===", lf)
    return 0

if __name__=="__main__": raise SystemExit(main())