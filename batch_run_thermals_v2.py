
#!/usr/bin/env python3
"""
batch_run_thermals_v2.py  — robust batching with per-flight subfolders under outputs/batch/

Why v2?
- Your upstream scripts currently write to shared paths under `outputs/`.
- This runner cleans those shared files per flight, then moves them into
  `outputs/batch/<flight_stem>/` to avoid hundreds of files in one folder.
- It also verifies that each CSV exists and is non-empty before proceeding,
  so you won't hit pandas EmptyDataError in the matcher.

Per-flight layout:
  outputs/batch/<flight_stem>/
    circles.csv
    circle_clusters_enriched.csv
    overlay_altitude_clusters.csv
    matched_clusters.csv

Aggregates:
  outputs/batch/thermals_all_raw.csv
  outputs/batch/thermals_clusters.csv
  outputs/batch/thermals_clusters.geojson

Filtering (proximity-based, no overlap requirement):
  --max-dist-m           default 5000 m
  --min-climb-ms         default 1.0 m/s
  --min-alt-gain-m       default 500 m
  --eps-m                default 300 m (cluster radius)
  --min-samples          default 1

Usage:
  python batch_run_thermals_v2.py \
    --igc-dir igc \
    --base-outdir outputs/batch \
    --max-dist-m 5000 --min-climb-ms 1.0 --min-alt-gain-m 500 \
    --eps-m 300 --min-samples 1 --verbose
"""

import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict

import pandas as pd
import numpy as np

# Upstream script names (as you provided)
CIRCLES_SCRIPT = "circles_from_brecords_v1d.py"
CIRCLE_CLUSTERS_SCRIPT = "circle_clusters_v1s.py"
ALT_CLUSTERS_SCRIPT = "overlay_altitude_clusters_v1c.py"
MATCH_SCRIPT = "match_clusters_v1.py"

# Where upstreams currently write by default
SHARED_OUTDIR = Path("outputs")
SHARED_FILES = [
    SHARED_OUTDIR / "circles.csv",
    SHARED_OUTDIR / "circle_clusters_enriched.csv",
    SHARED_OUTDIR / "overlay_altitude_clusters.csv",
    SHARED_OUTDIR / "matched_clusters.csv",
]

def run(cmd: List[str], verbose: bool) -> None:
    if verbose:
        print("[RUN]", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"Command failed ({rc}): {' '.join(cmd)}")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def clean_shared(verbose: bool) -> None:
    for p in SHARED_FILES:
        if p.exists():
            try:
                p.unlink()
                if verbose:
                    print(f"[CLEAN shared] {p}")
            except Exception as e:
                print(f"[WARN] Could not remove shared file {p}: {e}")

def must_be_nonempty(path: Path, label: str) -> bool:
    if not path.exists():
        print(f"[WARN] Missing {label}: {path}")
        return False
    try:
        if path.stat().st_size == 0:
            print(f"[WARN] Empty {label}: {path}")
            return False
    except Exception as e:
        print(f"[WARN] Could not stat {label} {path}: {e}")
        return False
    return True

def move_shared_to_flight(flight_dir: Path, verbose: bool) -> Dict[str, Path]:
    mapping = {}
    targets = {
        "circles": flight_dir / "circles.csv",
        "circle_clusters": flight_dir / "circle_clusters_enriched.csv",
        "alt_clusters": flight_dir / "overlay_altitude_clusters.csv",
        "matched": flight_dir / "matched_clusters.csv",  # not used until after matcher
    }
    # Move only the first three now (matched will be created later)
    for key, src in [("circles", SHARED_OUTDIR / "circles.csv"),
                     ("circle_clusters", SHARED_OUTDIR / "circle_clusters_enriched.csv"),
                     ("alt_clusters", SHARED_OUTDIR / "overlay_altitude_clusters.csv")]:
        dst = targets[key]
        if src.exists():
            try:
                shutil.move(str(src), str(dst))
                if verbose:
                    print(f"[MOVE] {src} -> {dst}")
            except Exception as e:
                print(f"[WARN] Could not move {src} -> {dst}: {e}")
        mapping[key] = dst
    mapping["matched"] = targets["matched"]
    return mapping

def to_geojson_points(df: pd.DataFrame) -> dict:
    features = []
    for _, row in df.iterrows():
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(row["lon"]), float(row["lat"])]},
            "properties": {
                "n_points": int(row.get("n_points", 1)),
                "max_climb_ms": float(row.get("max_climb_ms", 0)) if pd.notna(row.get("max_climb_ms", np.nan)) else None,
                "median_climb_ms": float(row.get("median_climb_ms", 0)) if pd.notna(row.get("median_climb_ms", np.nan)) else None,
                "median_alt_gain_m": float(row.get("median_alt_gain_m", 0)) if pd.notna(row.get("median_alt_gain_m", np.nan)) else None,
            },
        })
    return {"type": "FeatureCollection", "features": features}

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    phi1 = math.radians(float(lat1)); phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlmb = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0]*n
    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

def cluster_by_radius(latitudes: np.ndarray, longitudes: np.ndarray, eps_m: float) -> np.ndarray:
    n = len(latitudes)
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i+1, n):
            d = haversine_m(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
            if d <= eps_m:
                uf.union(i, j)
    labels = np.array([uf.find(i) for i in range(n)], dtype=int)
    root_map: Dict[int, int] = {}
    next_id = 0
    for r in labels:
        if r not in root_map:
            root_map[r] = next_id
            next_id += 1
    return np.array([root_map[r] for r in labels], dtype=int)

def aggregate_cluster(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    ag = df.groupby(label_col).agg(
        n_points=("cluster_label", "size"),
        lat=("lat", "mean"),
        lon=("lon", "mean"),
        max_climb_ms=("strength_ms", "max"),
        median_climb_ms=("strength_ms", "median"),
        median_alt_gain_m=("alt_gain_m", "median"),
    ).reset_index(drop=True)
    return ag

def main():
    ap = argparse.ArgumentParser(description="Batch IGC runner with robust per-flight isolation under outputs/batch/")
    ap.add_argument("--igc-dir", default="igc", help="Directory containing .igc files (default: %(default)s)")
    ap.add_argument("--base-outdir", default="outputs/batch", help="Base outputs directory (default: %(default)s)")
    ap.add_argument("--max-dist-m", type=float, default=5000.0, help="Max separation (m) between circle and alt clusters in a match (default: %(default)s)")
    ap.add_argument("--min-climb-ms", type=float, default=1.0, help="Min climb strength (m/s) to keep a match (default: %(default)s)")
    ap.add_argument("--min-alt-gain-m", type=float, default=500.0, help="Min altitude gain (m) to keep a match (default: %(default)s)")
    ap.add_argument("--eps-m", type=float, default=300.0, help="Clustering radius in meters for dedupe (default: %(default)s)")
    ap.add_argument("--min-samples", type=int, default=1, help="Min points per clustered location (default: %(default)s)")
    ap.add_argument("--geojson-out", default=None, help="Path to GeoJSON output (default: {base-outdir}/thermals_clusters.geojson)")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = ap.parse_args()

    igc_dir = Path(args.igc_dir)
    base_outdir = Path(args.base_outdir)
    ensure_dir(base_outdir)

    igc_files = sorted(igc_dir.glob("*.igc"))
    if not igc_files:
        print(f"[INFO] No .igc files found in {igc_dir.resolve()}")
        return

    all_rows = []

    for igc in igc_files:
        stem = igc.stem
        flight_dir = base_outdir / stem
        ensure_dir(flight_dir)

        # Clean shared outputs for this flight run
        clean_shared(verbose=args.verbose)

        py = sys.executable

        # Run upstreams (they will write to shared outputs/ paths)
        run([py, CIRCLES_SCRIPT, str(igc)], args.verbose)
        run([py, CIRCLE_CLUSTERS_SCRIPT, str(igc)], args.verbose)
        run([py, ALT_CLUSTERS_SCRIPT, str(igc)], args.verbose)

        # Move shared files into per-flight dir, verifying non-empty
        paths = move_shared_to_flight(flight_dir, verbose=args.verbose)

        # Verify required inputs before matching
        if not (must_be_nonempty(paths["circle_clusters"], "circle clusters") and
                must_be_nonempty(paths["alt_clusters"], "altitude clusters")):
            print(f"[SKIP] Missing/empty inputs for {igc.name}; skipping match step.")
            continue

        matched_out = paths["matched"]
        # Clean any existing per-flight matched file
        if matched_out.exists():
            try:
                matched_out.unlink()
                if args.verbose:
                    print(f"[CLEAN flight] {matched_out}")
            except Exception as e:
                print(f"[WARN] Could not remove {matched_out}: {e}")

        # Invoke matcher with per-flight files
        run([py, MATCH_SCRIPT, str(paths["circle_clusters"]), str(paths["alt_clusters"]), str(matched_out)], args.verbose)

        # Load and filter matched results
        if not must_be_nonempty(matched_out, "matched output"):
            print(f"[SKIP] Empty matched output for {igc.name}")
            continue

        m = pd.read_csv(matched_out)
        if m.empty:
            if args.verbose:
                print(f"[INFO] Matched CSV had headers but no rows for {igc.name}")
            continue

        # Compute strength/gain (max of circle vs alt)
        m["strength_ms"] = m[["circle_av_climb_ms", "alt_av_climb_ms"]].max(axis=1, skipna=True)
        m["alt_gain_m"] = m[["circle_alt_gained_m", "alt_alt_gained_m"]].max(axis=1, skipna=True)

        # Prefer circle lat/lon; fallback to alt
        lat = m["circle_lat"].fillna(m["alt_lat"])
        lon = m["circle_lon"].fillna(m["alt_lon"])

        keep = (m["dist_m"].fillna(np.inf) <= args.max_dist_m) & \
               (m["strength_ms"].fillna(0) >= args.min_climb_ms) & \
               (m["alt_gain_m"].fillna(0) >= args.min_alt_gain_m) & \
               lat.notna() & lon.notna()

        kept = m.loc[keep].copy()
        if kept.empty:
            if args.verbose:
                print(f"[INFO] No matches passed thresholds for {igc.name}")
            continue

        kept["lat"] = lat.loc[keep].values
        kept["lon"] = lon.loc[keep].values
        kept["flight_id"] = stem

        all_rows.append(kept[[
            "flight_id", "lat", "lon", "strength_ms", "alt_gain_m",
            "circle_cluster_id", "alt_cluster_id", "dist_m", "time_overlap_s", "overlap_frac"
        ]])

    # Aggregate + cluster
    agg_raw = base_outdir / "thermals_all_raw.csv"
    agg_csv = base_outdir / "thermals_clusters.csv"
    agg_geo = Path(args.geojson_out) if args.geojson_out else (base_outdir / "thermals_clusters.geojson")

    if not all_rows:
        print("[INFO] No matches across flights after filtering.")
        agg_raw.write_text("", encoding="utf-8")
        agg_csv.write_text("", encoding="utf-8")
        agg_geo.write_text(json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8")
        return

    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_csv(agg_raw, index=False)
    print(f"[OK] Wrote {agg_raw} ({len(all_df)} rows)")

    labels = cluster_by_radius(all_df["lat"].to_numpy(), all_df["lon"].to_numpy(), eps_m=args.eps_m)
    all_df["cluster_label"] = labels

    counts = all_df["cluster_label"].value_counts()
    keep_labels = set(counts[counts >= args.min_samples].index.tolist())
    clustered = all_df[all_df["cluster_label"].isin(keep_labels)].copy()

    agg = clustered.groupby("cluster_label").agg(
        n_points=("cluster_label", "size"),
        lat=("lat", "mean"),
        lon=("lon", "mean"),
        max_climb_ms=("strength_ms", "max"),
        median_climb_ms=("strength_ms", "median"),
        median_alt_gain_m=("alt_gain_m", "median"),
    ).reset_index(drop=True)

    agg.to_csv(agg_csv, index=False)
    print(f"[OK] Wrote {agg_csv} ({len(agg)} clusters)")

    fc = to_geojson_points(agg)
    agg_geo.write_text(json.dumps(fc), encoding="utf-8")
    print(f"[OK] Wrote {agg_geo}")

if __name__ == "__main__":
    main()
