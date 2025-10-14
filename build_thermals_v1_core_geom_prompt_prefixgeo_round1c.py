#!/usr/bin/env python3
"""
build_thermals_v1_core_geom_prompt_prefixgeo_round1.py
------------------------------------------------------
- DBSCAN-based thermal cores; centers from core points (prompted center estimator).
- CSV columns in desired order (unprefixed).
- GeoJSON properties **prefixed (01_, 02_, ...)** to force popup order in renderers.
- **Rounding:** Round numeric fields to **1 decimal place** everywhere
  EXCEPT `lat` and `lon` which remain full precision.

Popup/CSV order:
  01_strength_mean_core
  02_radius_m
  03_core_count
  04_border_count
  05_encounters
  06_core_fraction
  07_radius_p80_m
  08_strength_p95_core
  09_strength_mean_all
  10_strength_p95_all
  11_eps_km
  12_min_samples
  13_method
  14_center_estimator
  15_strength_min
  16_wp_id
  17_lat
  18_lon
"""

import argparse, json, math
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Tuple
import pandas as pd, numpy as np
from sklearn.cluster import DBSCAN

EARTH_R_M = 6_371_000.0

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

def log(msg: str, fh):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    print(line, end="")
    if fh: fh.write(line); fh.flush()

def parse_args():
    ap = argparse.ArgumentParser(description="Thermal core waypoints with forced popup order and 1-decimal rounding (except lat/lon).")
    # Tunings (prompted in IDE run if None)
    ap.add_argument("--method", choices=["dbscan","hdbscan","optics"], default=None)
    ap.add_argument("--eps-km", type=float, default=None)
    ap.add_argument("--min-samples", type=int, default=None)
    ap.add_argument("--strength-min", type=float, default=None)
    ap.add_argument("--center-estimator", choices=["geomedian","median","medoid"], default=None,
                    help="Estimator for cluster center (core points). If omitted, you will be prompted [median].")

    # Paths & misc
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

def great_circle_distance_m(lat1_rad, lon1_rad, lat2_rad, lon2_rad):
    dlat, dlon = lat2_rad-lat1_rad, lon2_rad-lon1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad)*np.cos(lat2_rad)*np.sin(dlon/2)**2
    return EARTH_R_M * 2*np.arcsin(np.minimum(1.0, np.sqrt(a)))

def robust_median(series: pd.Series) -> float:
    return float(np.median(series.values)) if len(series) else float("nan")

def geometric_median(lat_arr: np.ndarray, lon_arr: np.ndarray, eps: float = 1e-7, max_iter: int = 1000):
    if len(lat_arr) == 0:
        return float("nan"), float("nan")
    x = np.array([np.mean(lat_arr), np.mean(lon_arr)], dtype=float)
    pts = np.stack([lat_arr, lon_arr], axis=1)
    for _ in range(max_iter):
        diffs = pts - x
        d = np.linalg.norm(diffs, axis=1)
        if np.any(d < eps):
            return float(x[0]), float(x[1])
        w = 1.0 / d
        x_new = np.sum(pts * w[:, None], axis=0) / np.sum(w)
        if np.linalg.norm(x_new - x) < eps:
            x = x_new
            break
        x = x_new
    return float(x[0]), float(x[1])

def medoid_center(lat_arr: np.ndarray, lon_arr: np.ndarray):
    if len(lat_arr) == 0:
        return float("nan"), float("nan")
    lat_rad = np.radians(lat_arr); lon_rad = np.radians(lon_arr)
    n = len(lat_arr)
    tot = np.zeros(n, dtype=float)
    for i in range(n):
        d = great_circle_distance_m(lat_rad[i], lon_rad[i], lat_rad, lon_rad)
        tot[i] = np.sum(d)
    k = int(np.argmin(tot))
    return float(lat_arr[k]), float(lon_arr[k])

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
    rows = [df for df in rows if not df.empty]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def filter_inputs(df, strength_min, dstart, dend, logfh):
    need=["lat","lon","climb_rate_ms"]
    for c in need:
        if c not in df.columns: raise ValueError(f"Missing col: {c}")
    df2=df[df["climb_rate_ms"]>=strength_min].copy()
    pre = len(df2)
    df2 = df2[df2["climb_rate_ms"].between(-5.0, 10.0)]
    dropped = pre - len(df2)
    if dropped:
        log(f"[CLEAN] dropped {dropped} outliers outside [-5, 10] m/s", logfh)
    log(f"Strength filter ≥{strength_min}: {len(df)}→{len(df2)}", logfh)
    df2 = df2.dropna(subset=["lat","lon"])
    return df2.reset_index(drop=True)

def run_dbscan_haversine(lat: pd.Series, lon: pd.Series, eps_km: float, min_samples: int):
    lat_rad, lon_rad=np.radians(lat.values), np.radians(lon.values)
    eps_rad=(eps_km*1000)/EARTH_R_M
    model=DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine")
    labels = model.fit_predict(np.c_[lat_rad, lon_rad])
    core_mask = np.zeros_like(labels, dtype=bool)
    if hasattr(model, "core_sample_indices_") and model.core_sample_indices_ is not None:
        core_mask[model.core_sample_indices_] = True
    return labels, core_mask

def choose_center(use_grp: pd.DataFrame, estimator: str):
    lat_arr = use_grp["lat"].to_numpy(dtype=float)
    lon_arr = use_grp["lon"].to_numpy(dtype=float)
    if estimator == "median":
        return robust_median(use_grp["lat"]), robust_median(use_grp["lon"])
    elif estimator == "medoid":
        return medoid_center(lat_arr, lon_arr)
    else:  # geomedian
        return geometric_median(lat_arr, lon_arr)

def aggregate_clusters(df: pd.DataFrame, labels: np.ndarray, core_mask: np.ndarray,
                       method: str, eps_km: float, min_samples: int, strength_min: float,
                       center_estimator: str):
    df = df.copy()
    df["__label"] = labels
    df["__is_core"] = core_mask
    df = df[df["__label"] >= 0]

    rows = []
    for lab, grp in df.groupby("__label"):
        core_grp = grp[grp["__is_core"]]
        use_grp = core_grp if len(core_grp) > 0 else grp

        lat_c, lon_c = choose_center(use_grp, center_estimator)

        if len(use_grp) > 0 and not (math.isnan(lat_c) or math.isnan(lon_c)):
            lat_c_rad, lon_c_rad = np.radians(lat_c), np.radians(lon_c)
            lat_rad, lon_rad = np.radians(use_grp["lat"].values), np.radians(use_grp["lon"].values)
            dists = great_circle_distance_m(lat_c_rad, lon_c_rad, lat_rad, lon_rad)
            radius_m = float(np.max(dists)) if len(dists) else 0.0
            radius_p80_m = float(np.percentile(dists, 80)) if len(dists) else 0.0
        else:
            radius_m = 0.0
            radius_p80_m = 0.0

        strength_mean_core = float(np.mean(use_grp["climb_rate_ms"])) if len(use_grp) else float("nan")
        strength_p95_core  = float(np.percentile(use_grp["climb_rate_ms"], 95)) if len(use_grp) else float("nan")
        strength_mean_all  = float(np.mean(grp["climb_rate_ms"])) if len(grp) else float("nan")
        strength_p95_all   = float(np.percentile(grp["climb_rate_ms"], 95)) if len(grp) else float("nan")

        rows.append({
            "cluster_label": int(lab),  # internal only; not exported
            "strength_mean_core": strength_mean_core,
            "radius_m": radius_m,
            "core_count": int(core_grp.shape[0]),
            "border_count": int(len(grp) - core_grp.shape[0]),
            "encounters": int(len(grp)),
            "core_fraction": float(core_grp.shape[0] / len(grp)) if len(grp) else 0.0,
            "radius_p80_m": radius_p80_m,
            "strength_p95_core": strength_p95_core,
            "strength_mean_all": strength_mean_all,
            "strength_p95_all": strength_p95_all,
            "eps_km": eps_km,
            "min_samples": min_samples,
            "method": method,
            "center_estimator": center_estimator,
            "strength_min": strength_min,
            "lat": lat_c,
            "lon": lon_c,
        })

    tmp = pd.DataFrame(rows)
    if tmp.empty:
        return pd.DataFrame(columns=CSV_ORDER), {}

    tmp = tmp.sort_values(
        ["strength_p95_core", "strength_mean_core", "radius_m"],
        ascending=[False, False, True]
    ).reset_index(drop=True)
    tmp.insert(CSV_ORDER.index("wp_id"), "wp_id", np.arange(len(tmp)))

    lab_to_wp = dict(zip(tmp["cluster_label"].tolist(), tmp["wp_id"].tolist()))

    # CSV: drop cluster_label and reorder
    csv_df = tmp.drop(columns=["cluster_label"], errors="ignore")
    for col in CSV_ORDER:
        if col not in csv_df.columns:
            csv_df[col] = np.nan
    csv_df = csv_df[CSV_ORDER].copy()

    # ---- ROUND numeric columns to 1 decimal, EXCEPT lat/lon ----
    skip_round = {"lat", "lon"}
    for col in CSV_ORDER:
        if col in skip_round:
            continue
        if np.issubdtype(csv_df[col].dtype, np.number):
            csv_df[col] = csv_df[col].round(1)

    # ---- FORCE 1-decimal string formatting for strength_* and core_fraction in CSV ----
    force_str_cols = [c for c in CSV_ORDER if c.startswith("strength_")] + ["core_fraction"]
    for col in force_str_cols:
        if col in csv_df.columns:
            csv_df[col] = csv_df[col].apply(lambda x: (f"{float(x):.1f}" if pd.notna(x) else ""))

    return csv_df, lab_to_wp

def write_geojson_prefixed(df: pd.DataFrame, path: Path):
    """Write GeoJSON with **prefixed keys** to lock info box order in common renderers.
       Round numeric values to 1 decimal **except lat/lon** (kept full precision)."""
    feats = []
    for _, r in df.iterrows():
        props = {}
        for i, k in enumerate(CSV_ORDER, start=1):
            key = f"{i:02d}_{k}"
            v = r[k]
            # Keep lat/lon unrounded (numeric)
            if k in ("lat", "lon"):
                if isinstance(v, (np.floating, float)): v = float(v)
                if isinstance(v, (np.integer, int)):   v = int(v)
                props[key] = v
                continue
            # Force 1-decimal display for strength_* and core_fraction by string formatting
            if k.startswith("strength_") or k == "core_fraction":
                try:
                    props[key] = f"{float(v):.1f}"
                except Exception:
                    props[key] = v
                continue
            # For other numerics: round to 1 decimal (counts remain ints)
            if isinstance(v, (np.floating, float)):
                props[key] = round(float(v), 1)
            elif isinstance(v, (np.integer, int)):
                props[key] = int(v)
            else:
                props[key] = v
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(r['lon']), float(r['lat'])]},
            "properties": props
        })
    fc = {"type": "FeatureCollection", "features": feats}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(fc, f, indent=2)

def main():
    args = parse_args()
    inputs = Path(args.inputs_root)

    def _ask(val, label, default, cast, validator=None):
        if val is not None:
            return cast(val)
        while True:
            raw = input(f"{label} [{default}]: ").strip()
            raw = raw if raw else str(default)
            try:
                v = cast(raw)
            except Exception:
                print(f"Invalid value: {raw}")
                continue
            if validator and not validator(v):
                print(f"Invalid choice: {v}")
                continue
            return v

    DEF_METHOD      = "dbscan"
    DEF_EPS_KM      = 1
    DEF_MIN_SAMPLES = 5
    DEF_STRENGTH    = 2
    DEF_CENTER      = "median"

    method       = _ask(args.method,        "Clustering method (dbscan|hdbscan|optics)", DEF_METHOD, str).lower()
    eps_km       = _ask(args.eps_km,        "DBSCAN/OPTICS eps (km)",                    DEF_EPS_KM, float)
    min_samples  = _ask(args.min_samples,   "Min samples per cluster",                   DEF_MIN_SAMPLES, int)
    strength_min = _ask(args.strength_min,  "Min strength (weight) per point",          DEF_STRENGTH, float)
    valid_centers = {"geomedian","median","medoid"}
    center_estimator = _ask(args.center_estimator, "Center estimator (geomedian|median|medoid)",
                            DEF_CENTER, str, validator=lambda s: s in valid_centers).lower()

    print(f"[TUNING] method={method} eps_km={eps_km} min_samples={min_samples} strength_min={strength_min} center={center_estimator}")

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
        log("=== build_thermals_v1_core_geom_prompt_prefixgeo_round1 start ===", lf)
        log(f"TUNING: method={method} eps_km={eps_km} min_samples={min_samples} strength_min={strength_min} center={center_estimator}", lf)

        df = load_all_matches(inputs, lf)
        if df.empty:
            log("[WARN] No matched_clusters.csv found; writing empty outputs.", lf)
            write_empty_outputs()
            log("=== done ===", lf)
            return 0

        df = filter_inputs(df, strength_min, to_date(args.date_start), to_date(args.date_end), lf)
        if df.empty:
            log("[INFO] No rows after filter; writing empty outputs.", lf)
            write_empty_outputs()
            log("=== done ===", lf)
            return 0

        print(df["climb_rate_ms"].describe())
        print(df["climb_rate_ms"].head(20))

        labels, core_mask = run_dbscan_haversine(df.lat, df.lon, eps_km, min_samples)
        csv_df, lab_to_wp = aggregate_clusters(df, labels, core_mask, method, eps_km, min_samples, strength_min, center_estimator)
        log(f"Clusters: {len(csv_df)}", lf)

        total_pts   = len(labels)
        clustered_n = int((np.array(labels) >= 0).sum())
        noise_n     = total_pts - clustered_n
        uniq_labels = sorted({int(x) for x in labels if x >= 0})
        log(f"[DIAG] points total={total_pts}, clustered={clustered_n}, noise={noise_n}, unique_clusters={len(uniq_labels)}", lf)

        # Membership mapping CSV
        members = []
        if len(csv_df) > 0 and lab_to_wp and clustered_n > 0:
            if "__source_file" not in df.columns:
                df["__source_file"] = ""
            for idx, (lbl, is_core) in enumerate(zip(labels, core_mask)):
                if lbl < 0:
                    continue
                wp_id = lab_to_wp.get(int(lbl))
                if wp_id is None:
                    continue
                members.append({
                    "lat": float(df.lat.iloc[idx]),
                    "lon": float(df.lon.iloc[idx]),
                    "climb_rate_ms": float(df.climb_rate_ms.iloc[idx]),
                    "cluster_label": int(lbl),
                    "wp_id": int(wp_id),
                    "point_type": "core" if bool(is_core) else "border",
                    "__source_file": str(df.__source_file.iloc[idx]),
                })
        memb_df = pd.DataFrame(members, columns=[
            "lat","lon","climb_rate_ms","cluster_label","wp_id","point_type","__source_file"
        ])
        memb_csv = out_csv.with_name(out_csv.stem + "_membership.csv")
        memb_df.to_csv(memb_csv, index=False)
        log(f"[OK] wrote membership CSV {memb_csv} (rows={len(memb_df)})", lf)

        # Persist outputs
        if not args.dry_run:
            csv_df.to_csv(out_csv, index=False)           # ordered & rounded (except lat/lon)
            write_geojson_prefixed(csv_df, out_geo)       # prefixed & rounded (except lat/lon)
            log(f"[OK] wrote {out_csv} & {out_geo}", lf)

        log("=== done ===", lf)
    return 0

if __name__=="__main__": raise SystemExit(main())
