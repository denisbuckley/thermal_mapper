
#!/usr/bin/env python3
# match_clusters_v1.py
# Extended matcher: enriches matched_clusters.csv with lat, lon, and strength

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
OUT_DIR = PROJECT_ROOT / "outputs"

def main():
    # Input files (relative to outputs/)
    cpath = OUT_DIR / "circle_clusters_enriched.csv"
    apath = OUT_DIR / "overlay_altitude_clusters.csv"
    mpath = OUT_DIR / "matched_clusters.csv"  # base match file

    if not (cpath.exists() and apath.exists() and mpath.exists()):
        print("[ERROR] Missing input files for enrichment")
        sys.exit(2)

    c = pd.read_csv(cpath)
    a = pd.read_csv(apath)
    m = pd.read_csv(mpath)

    # Resolve column names
    lat_col = next((c for c in ["lat","latitude","lat_deg","centroid_lat"] if c in c.columns), None)
    lon_col = next((c for c in ["lon","longitude","lon_deg","centroid_lon"] if c in c.columns), None)
    strength_col = next((c for c in ["strength","climb_rate_ms","avg_climb_ms","mean_climb_ms"] if c in a.columns), None)

    if not (lat_col and lon_col and strength_col):
        print("[ERROR] Required columns not found in circle/altitude CSVs")
        sys.exit(2)

    # merge circle coords
    if "circle_cluster_id" in m.columns and "circle_cluster_id" in c.columns:
        m = m.merge(c[["circle_cluster_id", lat_col, lon_col]], on="circle_cluster_id", how="left")
    # merge altitude strength
    if "alt_cluster_id" in m.columns and "alt_cluster_id" in a.columns:
        m = m.merge(a[["alt_cluster_id", strength_col]], on="alt_cluster_id", how="left")

    # rename for consistency
    m = m.rename(columns={lat_col: "lat", lon_col: "lon", strength_col: "strength"})

    # overwrite outputs/matched_clusters.csv with enriched version
    out_csv = OUT_DIR / "matched_clusters.csv"
    m.to_csv(out_csv, index=False)
    print(f"[OK] wrote enriched {out_csv} with lat/lon/strength")

if __name__ == "__main__":
    main()
