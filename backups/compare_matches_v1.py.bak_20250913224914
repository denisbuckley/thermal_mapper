
#!/usr/bin/env python3
"""
compare_matches_v1.py

Stacks match outputs from different modes into ONE CSV with a `mode` column:
  - strict:        outputs/matched_clusters_*.csv
  - many→one:      outputs/matched_clusters_many_to_one_*.csv
  - alt→many:      outputs/matched_alt_to_many_*.csv

Writes:
  - outputs/compare_matches_<ts>.csv

Behavior:
  - Loads the most recent file for each mode (if present)
  - Normalizes columns to a common schema
  - Concatenates rows and prints a small summary + head preview

Common schema (columns):
  c_id, a_id, d_km, ovl_s, ovl_f, gap_s, c_gain_m, c_rate, a_gain_m, a_rate, mode
"""

import os, glob
import pandas as pd
import numpy as np
from datetime import datetime

OUTPUTS_DIR = "outputs"

def latest(patts):
    files=[]
    for p in patts:
        files.extend(glob.glob(p))
    if not files: return None
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files[0]

def load_mode(path, mode):
    if not path: return None
    df = pd.read_csv(path)
    # Normalize columns
    rename = {}
    if 'circle_id' in df.columns: rename['circle_id'] = 'c_id'
    if 'alt_id' in df.columns:    rename['alt_id']    = 'a_id'
    if 'dist_km' in df.columns:   rename['dist_km']   = 'd_km'
    if 'overlap_s' in df.columns: rename['overlap_s'] = 'ovl_s'
    if 'overlap_frac' in df.columns: rename['overlap_frac'] = 'ovl_f'
    if 'time_gap_s' in df.columns:   rename['time_gap_s']   = 'gap_s'
    df = df.rename(columns=rename)

    # Ensure required columns exist
    required = ['c_id','a_id','d_km','ovl_s','ovl_f','gap_s','c_gain_m','c_rate','a_gain_m','a_rate']
    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    df = df[required].copy()
    df['mode'] = mode
    return df

def main():
    strict = latest([f"{OUTPUTS_DIR}/matched_clusters_*.csv"])
    many1  = latest([f"{OUTPUTS_DIR}/matched_clusters_many_to_one_*.csv"])
    alt2   = latest([f"{OUTPUTS_DIR}/matched_alt_to_many_*.csv"])

    parts = []
    if strict:
        print(f"[compare] strict:   {strict}")
        parts.append(load_mode(strict, "strict"))
    else:
        print("[compare] strict:   (none)")
    if many1:
        print(f"[compare] many2one: {many1}")
        parts.append(load_mode(many1, "many2one"))
    else:
        print("[compare] many2one: (none)")
    if alt2:
        print(f"[compare] alt2many: {alt2}")
        parts.append(load_mode(alt2, "alt2many"))
    else:
        print("[compare] alt2many: (none)")

    if not parts:
        print("[compare] No match files found. Run a matcher first.")
        return

    X = pd.concat(parts, ignore_index=True)
    # Round for neatness
    for col, ndp in [('d_km',3), ('ovl_s',1), ('ovl_f',3), ('gap_s',1),
                     ('c_gain_m',1), ('c_rate',2), ('a_gain_m',1), ('a_rate',2)]:
        if col in X.columns:
            X[col] = X[col].astype(float).round(ndp)

    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    out = f"{OUTPUTS_DIR}/compare_matches_{ts}.csv"
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    X.to_csv(out, index=False)
    print(f"[compare] Wrote {out}")
    print(f"[compare] Counts by mode:\n{X['mode'].value_counts()}")

    # Preview
    with pd.option_context('display.max_columns', None, 'display.width', 140):
        print("\n[compare] Preview (head):")
        print(X.head(12))

if __name__ == "__main__":
    main()
