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
    ap.add_argument("--inputs-root", default="outputs/batch_csv")
    ap.add_argument("--method", default="dbscan", choices=["dbscan","hdbscan","optics"])
    ap.add_argument("--eps-km", type=float, default=5.0)
    ap.add_argument("--min-samples", type=int, default=2)
    ap.add_argument("--strength-min", type=float, default=1.0)
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
    rows=[]
    for csv in inputs_root.glob("**/matched_clusters.csv"):
        try:
            df=pd.read_csv(csv); df["__source_file"]=str(csv); rows.append(df)
            log(f"Loaded {len(df)} rows from {csv}", logfh)
        except Exception as e: log(f"[WARN] Failed {csv}: {e}", logfh)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def filter_inputs(df, strength_min, dstart, dend, logfh):
    need=["lat","lon","climb_rate_ms"]
    for c in need:
        if c not in df.columns: raise ValueError(f"Missing col: {c}")
    df2=df[df["climb_rate_ms"]>=strength_min].copy()
    log(f"Strength filter ≥{strength_min}: {len(df)}→{len(df2)}", logfh)
    df2=df2.dropna(subset=["lat","lon"])
    return df2.reset_index(drop=True)

def run_dbscan_haversine(lat, lon, eps_km, min_samples):
    lat_rad, lon_rad=np.radians(lat), np.radians(lon)
    eps_rad=(eps_km*1000)/EARTH_R_M
    model=DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine")
    return model.fit_predict(np.c_[lat_rad, lon_rad])

def aggregate_clusters(df, labels, method, eps_km, min_samples, strength_min):
    df=df.copy(); df["__label"]=labels; df=df[df["__label"]>=0]
    out=[]
    for lab,grp in df.groupby("__label"):
        lat_med, lon_med=robust_median(grp["lat"]), robust_median(grp["lon"])
        lat_rad, lon_rad=np.radians(grp["lat"]), np.radians(grp["lon"])
        r_m=max([great_circle_distance_m(np.radians(lat_med),np.radians(lon_med),a,b)
                 for a,b in zip(lat_rad,lon_rad)])
        out.append({
            "lat":lat_med,"lon":lon_med,"radius_m":r_m,
            "strength_mean":float(np.mean(grp["climb_rate_ms"])),
            "strength_p95":float(np.percentile(grp["climb_rate_ms"],95)),
            "method":method,"eps_km":eps_km,
            "min_samples":min_samples,"strength_min":strength_min})
    out=pd.DataFrame(out)
    if not out.empty:
        out=out.sort_values(["strength_p95","strength_mean","radius_m"],
                            ascending=[False,False,True]).reset_index(drop=True)
        out.insert(0,"wp_id",np.arange(len(out)))
    return out

def write_geojson(df,path:Path):
    feats=[{"type":"Feature","geometry":{"type":"Point","coordinates":[r.lon,r.lat]},
            "properties":r.to_dict()} for _,r in df.iterrows()]
    fc={"type":"FeatureCollection","features":feats}
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w") as f: json.dump(fc,f,indent=2)

def main():
    a=parse_args(); inputs=Path(a.inputs_root)
    out_csv, out_geo, logp=Path(a.out_csv), Path(a.out_geojson), Path(a.debug_log)
    logp.parent.mkdir(parents=True,exist_ok=True)
    with open(logp,"a") as lf:
        log("=== build_thermals_v1 start ===",lf)
        df=load_all_matches(inputs,lf)
        if df.empty: return 0
        df=filter_inputs(df,a.strength_min,to_date(a.date_start),to_date(a.date_end),lf)
        if df.empty: return 0
        labels=run_dbscan_haversine(df.lat,df.lon,a.eps_km,a.min_samples)
        out=aggregate_clusters(df,labels,a.method,a.eps_km,a.min_samples,a.strength_min)
        log(f"Clusters: {len(out)}",lf)
        if not a.dry_run:
            out_csv.parent.mkdir(parents=True,exist_ok=True)
            out.to_csv(out_csv,index=False); write_geojson(out,out_geo)
            log(f"[OK] wrote {out_csv} & {out_geo}",lf)
        log("=== done ===",lf)
    return 0

if __name__=="__main__": raise SystemExit(main())