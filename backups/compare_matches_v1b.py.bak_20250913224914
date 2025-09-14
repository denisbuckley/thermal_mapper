
#!/usr/bin/env python3
"""
compare_matches_v1b.py

Stacks match outputs from different modes into ONE CSV with a `mode` column,
and adds cluster sizes from the enriched sources:

  - strict:        outputs/matched_clusters_*.csv
  - many→one:      outputs/matched_clusters_many_to_one_*.csv
  - alt→many:      outputs/matched_alt_to_many_*.csv

Also loads:
  - outputs/circle_clusters_enriched_*.csv  (for c_n map)
  - outputs/altitude_clusters_enriched_*.csv (for a_n map)

Writes:
  - outputs/compare_matches_<ts>.csv

Columns:
  c_id, a_id, d_km, ovl_s, ovl_f, gap_s, c_gain_m, c_rate, a_gain_m, a_rate, c_n, a_n, mode
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

def load_enriched_maps():
    """Return dicts: c_id->n and a_id->n from latest enriched CSVs."""
    circ = latest([f"{OUTPUTS_DIR}/circle_clusters_enriched_*.csv"])
    alti = latest([f"{OUTPUTS_DIR}/altitude_clusters_enriched_*.csv"])
    c_map = {}; a_map = {}
    if circ:
        df = pd.read_csv(circ)
        if 'cluster_id' in df.columns and 'n' in df.columns:
            c_map = dict(zip(df['cluster_id'], df['n']))
    if alti:
        df = pd.read_csv(alti)
        if 'cluster_id' in df.columns and 'n' in df.columns:
            a_map = dict(zip(df['cluster_id'], df['n']))
    return c_map, a_map, circ, alti

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
    # Build size maps from enriched sources
    c_map, a_map, c_enriched_path, a_enriched_path = load_enriched_maps()
    if not c_map and not a_map:
        print("[compare v1b] Warning: no enriched maps found; c_n/a_n will be NaN.")
    else:
        print(f"[compare v1b] Using enriched sources:")
        if c_enriched_path: print("  circles:", c_enriched_path)
        if a_enriched_path: print("  altitude:", a_enriched_path)

    strict = latest([f"{OUTPUTS_DIR}/matched_clusters_*.csv"])
    many1  = latest([f"{OUTPUTS_DIR}/matched_clusters_many_to_one_*.csv"])
    alt2   = latest([f"{OUTPUTS_DIR}/matched_alt_to_many_*.csv"])

    parts = []
    if strict:   parts.append(load_mode(strict, "strict"));   print(f"[compare v1b] strict:   {strict}")
    else:        print("[compare v1b] strict:   (none)")
    if many1:    parts.append(load_mode(many1,  "many2one")); print(f"[compare v1b] many2one: {many1}")
    else:        print("[compare v1b] many2one: (none)")
    if alt2:     parts.append(load_mode(alt2,   "alt2many")); print(f"[compare v1b] alt2many: {alt2}")
    else:        print("[compare v1b] alt2many: (none)")

    if not parts:
        print("[compare v1b] No match files found. Run a matcher first.")
        return

    X = pd.concat(parts, ignore_index=True)

    # Attach c_n and a_n via maps
    X['c_n'] = X['c_id'].map(c_map) if c_map else np.nan
    X['a_n'] = X['a_id'].map(a_map) if a_map else np.nan

    # Round for neatness
    for col, ndp in [('d_km',3), ('ovl_s',1), ('ovl_f',3), ('gap_s',1),
                     ('c_gain_m',1), ('c_rate',2), ('a_gain_m',1), ('a_rate',2)]:
        if col in X.columns:
            X[col] = X[col].astype(float).round(ndp)

    # Reorder columns
    cols = ['c_id','a_id','d_km','ovl_s','ovl_f','gap_s',
            'c_gain_m','c_rate','a_gain_m','a_rate','c_n','a_n','mode']
    X = X[cols]

    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    out = f"{OUTPUTS_DIR}/compare_matches_{ts}.csv"
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    X.to_csv(out, index=False)
    print(f"[compare v1b] Wrote {out}")
    print(f"[compare v1b] Counts by mode:\n{X['mode'].value_counts()}")
    print("[compare v1b] Head:")
    with pd.option_context('display.max_columns', None, 'display.width', 160):
        print(X.head(12))

if __name__ == "__main__":
    main()
