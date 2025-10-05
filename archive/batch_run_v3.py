#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_run_v4.py — MONOLITHIC BATCH

Flow for ALL IGCs in a folder:
  For each *.igc file:
    1) Parse IGC B-records → track DataFrame
    2) Detect circling segments → circles.csv
    3) Cluster circles → circle_clusters_enriched.csv
    4) Detect altitude-only climb segments → altitude_clusters.csv
    5) Match circle clusters ↔ altitude clusters → matched_clusters.csv + .json

Inputs:
  ./igc/<*.igc>  (or another folder you specify when prompted)

Outputs:
  ./outputs/batch_csv/<flight_stem>/
    circles.csv
    circle_clusters_enriched.csv
    altitude_clusters.csv
    matched_clusters.csv
    matched_clusters.json
    pipeline_debug.log
"""

import argparse, sys
from pathlib import Path

# import everything from your pipeline (reuse helpers + funcs)
from pipeline_v4 import (
    IGC_DIR, OUT_ROOT, logf_write, wipe_run_dir,
    parse_igc_brecords, detect_circles, cluster_circles, detect_altitude_clusters, match_clusters
)

import pandas as pd
import json

def process_igc(igc_path: Path, args) -> None:
    flight = igc_path.stem
    run_dir = OUT_ROOT / flight
    wipe_run_dir(run_dir)
    logf = run_dir / "pipeline_debug.log"
    logf_write(logf, f"===== pipeline_v4 batch start =====")
    logf_write(logf, f"IGC: {igc_path}")
    logf_write(logf, f"RUN_DIR: {run_dir}")

    # --- 1) Track
    track = parse_igc_brecords(igc_path)
    if track.empty:
        logf_write(logf, "[WARN] No B-records parsed; skipping.")
        return

    # --- 2) Circles
    circles = detect_circles(track)
    circles_cols = ["lat","lon","t_start","t_end","climb_rate_ms","alt_gain_m","duration_s"]
    if not circles.empty:
        for c in circles_cols:
            if c not in circles.columns:
                circles[c] = pd.NA
        extras = [c for c in circles.columns if c not in circles_cols]
        circles = circles[circles_cols + extras]
    else:
        circles = pd.DataFrame(columns=circles_cols)
    circles_csv = run_dir / "circles.csv"
    circles.to_csv(circles_csv, index=False)
    logf_write(logf, f"[OK] wrote {len(circles)} circles → {circles_csv}")

    # --- 3) Circle clusters
    cc = cluster_circles(circles, eps_m=args.circle_eps_m, min_samples=args.circle_min_samples)
    cc_cols = [
        "cluster_id","lat","lon","t_start","t_end",
        "climb_rate_ms","climb_rate_ms_median","alt_gain_m_mean","duration_s_mean","n_circles"
    ]
    if not cc.empty:
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
    alts = detect_altitude_clusters(track, min_gain_m=args.alt_min_gain, min_duration_s=args.alt_min_duration)
    ALT_COLS = ["cluster_id","lat","lon","t_start","t_end","climb_rate_ms","alt_gain_m","duration_s"]
    if alts.empty:
        alts = pd.DataFrame(columns=ALT_COLS)
    else:
        if "cluster_id" not in alts.columns:
            alts = alts.reset_index(drop=True)
            alts["cluster_id"] = alts.index
        for k in ALT_COLS:
            if k not in alts.columns:
                alts[k] = pd.NA
        alts = alts[ALT_COLS]
    alt_csv = run_dir / "altitude_clusters.csv"
    alts.to_csv(alt_csv, index=False)
    logf_write(logf, f"[OK] wrote {len(alts)} altitude clusters → {alt_csv}")

    # --- 5) Matching
    matches, stats = match_clusters(cc, alts,
        max_dist_m=args.match_max_dist_m,
        min_overlap_frac=args.match_min_overlap
    )
    match_cols = [
        "lat","lon","climb_rate_ms","alt_gain_m","duration_s",
        "circle_cluster_id","alt_cluster_id",
        "circle_lat","circle_lon","alt_lat","alt_lon",
        "dist_m","time_overlap_s","overlap_frac"
    ]
    if not matches.empty:
        matches = matches.reindex(columns=match_cols)
    else:
        matches = pd.DataFrame(columns=match_cols)
    match_csv = run_dir / "matched_clusters.csv"
    matches.to_csv(match_csv, index=False)

    match_json = run_dir / "matched_clusters.json"
    payload = {"stats": stats, "rows": int(len(matches))}
    match_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logf_write(logf, f"[OK] wrote {len(matches)} matches → {match_csv}")
    logf_write(logf, "[DONE]")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--igc-dir", type=str, default="igc", help="Folder containing .igc files")
    ap.add_argument("--circle-eps-m", type=float, default=200.0)
    ap.add_argument("--circle-min-samples", type=int, default=2)
    ap.add_argument("--alt-min-gain", type=float, default=30.0)
    ap.add_argument("--alt-min-duration", type=float, default=20.0)
    ap.add_argument("--match-max-dist-m", type=float, default=600.0)
    ap.add_argument("--match-min-overlap", type=float, default=0.25)
    args = ap.parse_args()

    igc_dir = Path(args.igc_dir)
    if not igc_dir.exists():
        user_in = input(f"Enter folder path with IGC files [default {igc_dir}]: ").strip()
        igc_dir = Path(user_in) if user_in else igc_dir

    igc_files = sorted(igc_dir.glob("*.igc"))
    if not igc_files:
        print(f"[ERROR] No .igc files found in {igc_dir}")
        return 1

    for igc in igc_files:
        print(f"[INFO] Processing {igc.name}")
        process_igc(igc, args)

    print(f"[DONE] Processed {len(igc_files)} flights from {igc_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())