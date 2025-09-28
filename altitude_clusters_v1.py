#!/usr/bin/env python3
"""
Altitude-based thermal clustering.
Extracts climb segments purely from altitude gain,
clusters them, and saves altitude_clusters.csv.
"""

import argparse
import pandas as pd
import os
from pathlib import Path
from sklearn.cluster import DBSCAN

def detect_altitude_clusters(circles: pd.DataFrame) -> pd.DataFrame:
    # use climb rate threshold
    climbs = circles[circles["climb_rate_ms"] > 0.3].copy()
    if climbs.empty:
        return pd.DataFrame()

    # cluster by lat/lon (EPSG:4326 degrees → rough scaling for ~100 m radius)
    coords = climbs[["lat", "lon"]].to_numpy()
    db = DBSCAN(eps=0.001, min_samples=3).fit(coords)
    climbs["cluster"] = db.labels_
    return climbs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("circles", help="Path to circles.csv")
    ap.add_argument("--out", default="outputs/altitude_clusters.csv",
                    help="Output CSV file (default: outputs/altitude_clusters.csv)")
    args = ap.parse_args()

    circles_path = Path(args.circles)
    if not circles_path.exists():
        print(f"[ERROR] Missing input {circles_path}")
        return

    df = pd.read_csv(circles_path)
    clusters = detect_altitude_clusters(df)
    if clusters.empty:
        print("[WARN] No altitude clusters detected")
    else:
        os.makedirs(Path(args.out).parent, exist_ok=True)
        clusters.to_csv(args.out, index=False)
        print(f"[OK] wrote {args.out}")

if __name__ == "__main__":
    main()