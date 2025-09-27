
#!/usr/bin/env python3
"""
tuning_params_configurator_v1.py

Interactive script to collect tuning parameters for circle detection,
altitude gain detection, and cluster matching. Prompts with defaults
and saves results to config/tuning_params.csv.
"""

import os
import pandas as pd

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
