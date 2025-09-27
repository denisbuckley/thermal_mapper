#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess

def main():
    default_igc = "2020-11-08 Lumpy Paterson 108645.igc"
    igc_path = input(f"Enter path to IGC file [default: {default_igc}]: ").strip() or default_igc

    os.makedirs("outputs", exist_ok=True)

    steps = [
        ("circles_from_brecords_v1d.py", "outputs/circles.csv"),
        ("circle_clusters_v1s.py", "outputs/circle_clusters_enriched.csv"),
        ("overlay_altitude_clusters.py", "outputs/overlay_altitude_clusters.csv"),
        ("match_clusters_v1.py", "outputs/matched_clusters.csv"),
    ]

    for script, expected in steps:
        print(f"\n--- Running {script} ---")
        if not os.path.exists(script):
            print(f"Missing {script}, skipping.")
            continue
        subprocess.run(["python3", script], check=False)

        if os.path.exists(expected):
            print(f"✓ {expected} written.")
        else:
            print(f"✗ {expected} not found.")

    print("\nPipeline finished. Check outputs/ directory.")

if __name__ == "__main__":
    main()
