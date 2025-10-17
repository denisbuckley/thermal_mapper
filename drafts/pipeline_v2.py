#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pipeline_v2.py
Clean wrapper pipeline: runs IGC → circles → circle clusters → altitude clusters → matches
with a single IGC input prompt (default provided).
"""

import os

from archive.circles_from_brecords_v1d import parse_igc_brecords, detect_circles
from circle_clusters_v1s import cluster_circles
from overlay_altitude_clusters import detect_altitude_clusters
from match_clusters_v1 import match_clusters

def main():
    default_igc = "2020-11-08 Lumpy Paterson 108645.igc_subset"
    igc_path = input(f"Enter path to IGC file [default: {default_igc}]: ").strip() or default_igc

    os.makedirs("outputs", exist_ok=True)

    # --- Circles ---
    df = parse_igc_brecords(igc_path)
    circles_df = detect_circles(df)
    circles_path = "outputs/circles.csv"
    circles_df.to_csv(circles_path, index=False)
    print(f"✓ Wrote {len(circles_df)} circles → {circles_path}")

    # --- Circle clusters ---
    clusters_df = cluster_circles(circles_df)
    circ_clusters_path = "outputs/circle_clusters_enriched.csv"
    clusters_df.to_csv(circ_clusters_path, index=False)
    print(f"✓ Wrote {len(clusters_df)} circle clusters → {circ_clusters_path}")

    # --- Altitude clusters ---
    alt_clusters_df = detect_altitude_clusters(igc_path)
    alt_clusters_path = "outputs/overlay_altitude_clusters.csv"
    alt_clusters_df.to_csv(alt_clusters_path, index=False)
    print(f"✓ Wrote {len(alt_clusters_df)} altitude clusters → {alt_clusters_path}")

    # --- Matching ---
    matches_df, cand_count = match_clusters(clusters_df, alt_clusters_df)
    matches_path = "outputs/matched_clusters.csv"
    matches_df.to_csv(matches_path, index=False)
    print(f"✓ Wrote {len(matches_df)} matches (candidates: {cand_count}) → {matches_path}")

    print("\nPipeline finished. All outputs in outputs/.")

if __name__ == "__main__":
    main()
