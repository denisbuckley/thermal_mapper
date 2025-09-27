
#!/usr/bin/env python3
"""
check_total_climb_v1.py

Reads the latest altitude_clusters_*.csv from outputs/,
sums the gain_m column, and compares against a theoretical
climb requirement given distance (km) and L/D ratio.

Defaults: distance=1000 km, L/D=40:1.
"""

import glob, os
import pandas as pd

OUTPUTS_DIR = "outputs"

DIST_KM = 1000.0
LD = 40.0

def latest_csv(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files[0]

def main():
    f = latest_csv(f"{OUTPUTS_DIR}/altitude_clusters_*.csv")
    if not f:
        print("[check_total_climb] No altitude_clusters_*.csv found in outputs/")
        return
    df = pd.read_csv(f)
    required = DIST_KM * 1000.0 / LD
    detected = df['gain_m'].sum() if 'gain_m' in df.columns else 0.0
    pct = (detected/required*100) if required > 0 else float('nan')

    print(f"[check_total_climb] Using: {f}")
    print(f"Required climb (L/D={LD:.0f}, {DIST_KM:.0f} km): {required:,.0f} m")
    print(f"Detected climb (sum gain_m):                  {detected:,.0f} m")
    print(f"Coverage:                                     {pct:.1f} %")

if __name__ == "__main__":
    main()
