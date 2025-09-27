
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dbscan_hotspots_v1.py — cluster matched (per‑flight) thermals across flights using DBSCAN
on latitude/longitude only (haversine metric).

Input: outputs/batch_matches_<TS>.csv (or pass --matches path)
Output: outputs/hotspots_<TS>.csv plus a quick console summary.

Parameters (override via CLI):
  --eps-m       : neighborhood radius in meters (default 3000 m)
  --min-samples : minimum matches per hotspot (default 3)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd, numpy as np
from sklearn.cluster import DBSCAN

EARTH_R = 6371000.0

def load_matches(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Use circle side as the location (or the midpoint of alt/circ if you prefer later)
    lat = df['c_lat'].to_numpy(dtype=float)
    lon = df['c_lon'].to_numpy(dtype=float)
    keep = ~np.isnan(lat) & ~np.isnan(lon)
    return df.loc[keep].reset_index(drop=True)

def to_radians(lat_deg, lon_deg):
    return np.radians(lat_deg), np.radians(lon_deg)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", default=None, help="Path to combined matches CSV; if omitted, pick newest outputs/batch_matches_*.csv")
    ap.add_argument("--eps-m", type=float, default=3000.0, help="DBSCAN radius in meters (haversine metric)")
    ap.add_argument("--min-samples", type=int, default=3, help="DBSCAN min_samples")
    ap.add_argument("--outputs-dir", default="outputs", help="Outputs folder")
    args = ap.parse_args()

    out_dir = Path(args.outputs_dir); out_dir.mkdir(exist_ok=True)

    # pick newest matches if not provided
    if args.matches is None:
        cand = sorted(out_dir.glob("batch_matches_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not cand:
            print("[dbscan] No batch_matches_*.csv found. Run batch_match_igc_v3.py first.")
            return
        matches_path = cand[0]
    else:
        matches_path = Path(args.matches)

    df = load_matches(matches_path)
    if df.empty:
        print(f"[dbscan] No rows in {matches_path}")
        return

    lat_rad, lon_rad = to_radians(df['c_lat'].to_numpy(), df['c_lon'].to_numpy())
    X = np.column_stack([lat_rad, lon_rad])

    eps_rad = args.eps_m / EARTH_R
    model = DBSCAN(eps=eps_rad, min_samples=args.min_samples, metric='haversine')
    labels = model.fit_predict(X)
    df['hotspot_id'] = labels

    # Summarize clusters (ignore noise label = -1)
    valid = df[df['hotspot_id'] >= 0].copy()
    if valid.empty:
        print("[dbscan] No hotspots found (all noise). Try increasing --eps-m or lowering --min-samples.")
        return

    agg = valid.groupby('hotspot_id').agg(
        n=('hotspot_id','size'),
        mean_lat=('c_lat','mean'),
        mean_lon=('c_lon','mean'),
        median_gain_m=('a_gain_m','median'),
        median_rate_mps=('a_rate','median'),
    ).reset_index().sort_values('n', ascending=False)

    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    out_csv = out_dir / f"hotspots_{ts}.csv"
    agg.to_csv(out_csv, index=False)

    print(f"[dbscan] Read: {matches_path.name}  rows={len(df)}")
    print(f"[dbscan] Hotspots: {len(agg)}  (noise dropped)  -> {out_csv}")
    with pd.option_context('display.max_rows', None, 'display.width', 160):
        print(agg.head(20))

if __name__ == "__main__":
    main()
