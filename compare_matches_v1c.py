
#!/usr/bin/env python3
"""
compare_matches_v1c.py

Stacks match outputs from different modes into ONE CSV with a `mode` column,
and augments with cluster sizes from enriched inputs:

Inputs scanned (most recent of each if present):
  - matches:
      * outputs/matched_clusters_*.csv               -> mode=strict
      * outputs/matched_clusters_many_to_one_*.csv   -> mode=many2one
      * outputs/matched_alt_to_many_*.csv            -> mode=alt2many
  - enrichment maps (for sizes):
      * outputs/circle_clusters_enriched_*.csv  -> provides c_n (n per circle cluster_id)
      * outputs/altitude_clusters_enriched_*.csv -> provides a_n (n per altitude cluster_id)

Writes:
  - outputs/compare_matches_<ts>.csv

Console:
  - Prints a compact, aligned table INCLUDING c_n and a_n.
"""

import os, glob
import pandas as pd
import numpy as np
from datetime import datetime

# === Injected by apply_tuning_patches_v1.py ===
from tuning_loader import load_tuning, override_globals
_tuning = load_tuning("config/tuning_params.csv")
override_globals(globals(), _tuning, allowed={
    "EPS_M","MIN_OVL_FRAC","MAX_TIME_GAP_S"
})



OUTPUTS_DIR = "outputs"

def latest(patts):
    files = []
    for p in patts:
        files.extend(glob.glob(p))
    if not files:
        return None
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files[0]

def load_enriched_maps():
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
    # Normalize column names from various matchers
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
    # size maps
    c_map, a_map, c_enriched_path, a_enriched_path = load_enriched_maps()
    if not c_map and not a_map:
        print("[compare v1c] Warning: no enriched maps found; c_n/a_n will be NaN.")
    else:
        print("[compare v1c] Using enriched sources:")
        if c_enriched_path: print("  circles:", c_enriched_path)
        if a_enriched_path: print("  altitude:", a_enriched_path)

    # match files
    strict = latest([f"{OUTPUTS_DIR}/matched_clusters_*.csv"])
    many1  = latest([f"{OUTPUTS_DIR}/matched_clusters_many_to_one_*.csv"])
    alt2   = latest([f"{OUTPUTS_DIR}/matched_alt_to_many_*.csv"])

    parts = []
    if strict:   parts.append(load_mode(strict, "strict"));   print(f"[compare v1c] strict:   {strict}")
    else:        print("[compare v1c] strict:   (none)")
    if many1:    parts.append(load_mode(many1,  "many2one")); print(f"[compare v1c] many2one: {many1}")
    else:        print("[compare v1c] many2one: (none)")
    if alt2:     parts.append(load_mode(alt2,   "alt2many")); print(f"[compare v1c] alt2many: {alt2}")
    else:        print("[compare v1c] alt2many: (none)")

    if not parts:
        print("[compare v1c] No match files found. Run a matcher first.")
        return

    X = pd.concat(parts, ignore_index=True)

    # attach sizes
    X['c_n'] = X['c_id'].map(c_map) if c_map else np.nan
    X['a_n'] = X['a_id'].map(a_map) if a_map else np.nan

    # rounding
    for col, ndp in [('d_km',3), ('ovl_s',1), ('ovl_f',3), ('gap_s',1),
                     ('c_gain_m',1), ('c_rate',2), ('a_gain_m',1), ('a_rate',2)]:
        if col in X.columns:
            X[col] = X[col].astype(float).round(ndp)

    # reorder for readability
    cols = ['c_id','a_id','d_km','ovl_s','ovl_f','gap_s',
            'c_gain_m','c_rate','c_n','a_gain_m','a_rate','a_n','mode']
    X = X[cols]

    # write
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    out = f"{OUTPUTS_DIR}/compare_matches_{ts}.csv"
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    X.to_csv(out, index=False)
    print(f"[compare v1c] Wrote {out}")
    print(f"[compare v1c] Counts by mode:\n{X['mode'].value_counts()}")

    # console table with sizes
    if not X.empty:
        print("\n c_id  a_id   d_km   ovl_s   ovl_f   gap_s   c_gain  c_rate   c_n   a_gain  a_rate   a_n    mode")
        print(" ----  ----  ------  ------  ------  ------  ------  ------  ----   ------  ------  ----   ------")
        for _, r in X.iterrows():
            # use blanks for NaN sizes to keep narrow
            cg = "" if pd.isna(r['c_gain_m']) else f"{r['c_gain_m']:.1f}"
            cr = "" if pd.isna(r['c_rate'])   else f"{r['c_rate']:.2f}"
            ag = "" if pd.isna(r['a_gain_m']) else f"{r['a_gain_m']:.1f}"
            ar = "" if pd.isna(r['a_rate'])   else f"{r['a_rate']:.2f}"
            cn = "" if pd.isna(r['c_n']) else f"{int(r['c_n'])}"
            an = "" if pd.isna(r['a_n']) else f"{int(r['a_n'])}"
            print(f"{int(r['c_id']):5d} {int(r['a_id']):5d}  {r['d_km']:6.3f}  {r['ovl_s']:6.1f}  {r['ovl_f']:6.3f}"
                  f"  {r['gap_s']:6.1f}  {cg:>6}  {cr:>6}  {cn:>4}   {ag:>6}  {ar:>6}  {an:>4}   {r['mode']}")
    else:
        print("[compare v1c] No rows to display.")

if __name__ == "__main__":
    main()