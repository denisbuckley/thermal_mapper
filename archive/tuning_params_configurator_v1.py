
#!/usr/bin/env python3
"""
tuning_params_configurator_v1.py

Interactive script to collect tuning parameters for circle detection,
altitude gain detection, and cluster matching. Prompts with defaults
and saves results to config/tuning_params.csv.
"""

import os
import pandas as pd
"""
tuning_params_configurator_v1.py

Interactive configurator for tuning parameters used in the thermal analysis pipeline.
Prompts user for values (with defaults) and saves results to config/tuning_params.csv.

----------------------------------------------------------------------
Circle Detection
----------------------------------------------------------------------
- C_MIN_ARC_DEG (360): Minimum cumulative heading change (deg) to count as a circling segment.
- C_MIN_RATE_DPS (4): Minimum average turn rate (deg/s). Filters out flat arcs.
- C_MAX_RATE_DPS (20): Maximum average turn rate (deg/s). Filters out jittery/sharp turns.
- C_MIN_RADIUS_M (30): Minimum radius of a circle (m).
- C_MAX_RADIUS_M (600): Maximum radius of a circle (m).
- C_MIN_DIR_RATIO (0.6): Ratio of samples in consistent turning direction (CW vs CCW).
- TIME_CAP_S (120): Maximum segment duration (s) before splitting long circles.
- C_MAX_WIN_SAMPLES (100): Max number of points per circle window.
- C_EPS_M (500): DBSCAN epsilon (m) for clustering circle centroids.
- C_MIN_SAMPLES (2): Minimum samples per cluster for circle clustering.

----------------------------------------------------------------------
Altitude Gain Detection
----------------------------------------------------------------------
- MIN_CLIMB_S (20): Minimum climb duration (s).
- MIN_GAIN_M (30): Minimum climb altitude gain (m).
- SMOOTH_RADIUS_S (15): Rolling window length (s) for smoothing altitude.
- MAX_GAP_S (1200): Maximum allowed time gap (s) for stitching climb segments.
- ALT_DROP_M (180): Maximum altitude drop (m) allowed before splitting a climb.
- ALT_DROP_FRAC (0.40): Maximum fractional drop (e.g. 40% of total gain).
- A_EPS_M (600): DBSCAN epsilon (m) for clustering climb segments.
- A_MIN_SAMPLES (2): Minimum samples per altitude cluster.

----------------------------------------------------------------------
Circle ↔ Altitude Cluster Matching
----------------------------------------------------------------------
- EPS_M (500): Maximum allowed centroid distance (m) between circle and altitude cluster.
- MIN_OVL_FRAC (0.3): Minimum required temporal overlap fraction between clusters.
- MAX_TIME_GAP_S (900): Maximum allowed time gap (s) when evaluating matches.
"""
defaults = {
    # Circle detection
    "C_MIN_ARC_DEG": 360,
    "C_MIN_RATE_DPS": 4,
    "C_MAX_RATE_DPS": 20,
    "C_MIN_RADIUS_M": 30,
    "C_MAX_RADIUS_M": 600,
    "C_MIN_DIR_RATIO": 0.6,
    "TIME_CAP_S": 120,
    "C_MAX_WIN_SAMPLES": 100,
    "C_EPS_M": 500,
    "C_MIN_SAMPLES": 2,

    # Altitude gain
    "MIN_CLIMB_S": 20,
    "MIN_GAIN_M": 30,
    "SMOOTH_RADIUS_S": 15,
    "MAX_GAP_S": 1200,
    "ALT_DROP_M": 180,
    "ALT_DROP_FRAC": 0.40,
    "A_EPS_M": 600,
    "A_MIN_SAMPLES": 2,

    # Matching
    "EPS_M": 500,
    "MIN_OVL_FRAC": 0.3,
    "MAX_TIME_GAP_S": 900,
}

def main():
    vals = {}
    print("=== Tuning Parameter Configurator ===")
    for name, default in defaults.items():
        try:
            raw = input(f"Enter {name} [{default}]: ").strip()
        except EOFError:
            raw = ""
        if raw == "":
            vals[name] = default
        else:
            try:
                vals[name] = float(raw) if "." in raw or "e" in raw.lower() else int(raw)
            except ValueError:
                vals[name] = raw

    df = pd.DataFrame(list(vals.items()), columns=["name","value"])

    os.makedirs("config", exist_ok=True)
    out = "config/tuning_params.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} parameters to {out}")

if __name__ == "__main__":
    main()
