
#!/usr/bin/env python3
"""
batch_match_igc_v1.py

Batch-process IGCs by matching circle vs altitude clusters per IGC and aggregating
results into one master CSV.

Default IGC folder:
  - Uses a lowercase `igc_subset/` folder **next to this script** by default.
  - You can override with --igc_subset-dir.

Assumptions
- For each IGC basename (e.g., "2020-11-08 Lumpy Paterson 108645"), you have already
  produced per-IGC enriched cluster files with names containing that basename:
    outputs/<base>_circle_clusters_enriched_*.csv
    outputs/<base>_altitude_clusters_enriched_*.csv

Outputs
  - outputs/batch_matched_<YYYYmmddHHMMSS>.csv

Columns
  igc_base, c_id, a_id, d_km, ovl_s, ovl_f, gap_s,
  c_gain_m, c_rate, c_n, c_lat, c_lon, c_start, c_end,
  a_gain_m, a_rate, a_n, a_lat, a_lon, a_start, a_end
"""

import os, glob, math, argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# --- Matching thresholds (overridable via tuning_loader, if present) ---
EPS_M = 2000.0
MIN_OVL_FRAC = 0.20
MAX_TIME_GAP_S = 15*60

try:
    # optional: honor repo-wide tuning file if present
    from tuning_loader import load_tuning, override_globals
    _t = load_tuning("config/tuning_params.csv")
    override_globals(globals(), _t, allowed={"EPS_M","MIN_OVL_FRAC","MAX_TIME_GAP_S"})
except Exception:
    pass

def to_ts(s):
    return pd.to_datetime(s, utc=True, errors='coerce')

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1 = np.radians(lat1); p2 = np.radians(lat2)
    dphi = np.radians(lat2-lat1); dlmb = np.radians(lon2-lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlmb/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))

def latest_with_base(glob_patt: str, base_token: str):
    files = [p for p in glob.glob(glob_patt) if base_token in os.path.basename(p)]
    if not files: return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]

def time_overlap_and_gap(a0, a1, b0, b1):
    latest_start = max(a0, b0)
    earliest_end = min(a1, b1)
    ovl = (earliest_end - latest_start).total_seconds()
    if ovl > 0:
        return ovl, 0.0
    # non-overlap → positive gap
    gap = (a0 - b1).total_seconds() if a0 > b1 else (b0 - a1).total_seconds()
    return 0.0, gap

def normalize_enriched(df: pd.DataFrame) -> pd.DataFrame:
    # expected: cluster_id, lat, lon, start_time, end_time, n, gain_m, avg_rate_mps
    out = df.copy()
    rename = {}
    if 'thermal_id' in out.columns and 'cluster_id' not in out.columns:
        rename['thermal_id'] = 'cluster_id'
    if 'avg_rate' in out.columns and 'avg_rate_mps' not in out.columns:
        rename['avg_rate'] = 'avg_rate_mps'
    out = out.rename(columns=rename)
    # coerce time
    for col in ('start_time','end_time'):
        if col in out.columns:
            out[col] = to_ts(out[col])
    # fill missing columns
    for col in ('n','gain_m','avg_rate_mps'):
        if col not in out.columns:
            out[col] = np.nan
    return out

def match_one_igc(circ_df: pd.DataFrame, alt_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, cr in circ_df.iterrows():
        best = None
        for _, ar in alt_df.iterrows():
            d_m = float(haversine_m(cr['lat'], cr['lon'], ar['lat'], ar['lon']))
            if d_m > EPS_M:
                continue
            ovl_s, gap_s = time_overlap_and_gap(cr['start_time'], cr['end_time'],
                                                ar['start_time'], ar['end_time'])
            c_dur = max(1.0, float((cr['end_time'] - cr['start_time']).total_seconds()))
            a_dur = max(1.0, float((ar['end_time'] - ar['start_time']).total_seconds()))
            shorter = min(c_dur, a_dur)
            ovl_frac = max(0.0, ovl_s / shorter)

            if ovl_s <= 0 and gap_s > MAX_TIME_GAP_S:
                continue
            if ovl_s > 0 and ovl_frac < MIN_OVL_FRAC:
                continue

            score = (d_m/1000.0, -ovl_frac)
            cand = (score, dict(
                c_id = int(cr['cluster_id']),
                a_id = int(ar['cluster_id']),
                d_km = round(d_m/1000.0, 3),
                ovl_s = round(ovl_s, 1),
                ovl_f = round(ovl_frac, 3),
                gap_s = round(gap_s, 1),
                # circle side
                c_gain_m = None if pd.isna(cr.get('gain_m')) else float(cr['gain_m']),
                c_rate   = None if pd.isna(cr.get('avg_rate_mps')) else float(cr['avg_rate_mps']),
                c_n      = None if pd.isna(cr.get('n')) else int(cr['n']),
                c_lat    = float(cr['lat']), c_lon = float(cr['lon']),
                c_start  = cr['start_time'], c_end = cr['end_time'],
                # altitude side
                a_gain_m = None if pd.isna(ar.get('gain_m')) else float(ar['gain_m']),
                a_rate   = None if pd.isna(ar.get('avg_rate_mps')) else float(ar['avg_rate_mps']),
                a_n      = None if pd.isna(ar.get('n')) else int(ar['n']),
                a_lat    = float(ar['lat']), a_lon = float(ar['lon']),
                a_start  = ar['start_time'], a_end = ar['end_time'],
            ))
            if best is None or cand[0] < best[0]:
                best = cand
        if best is not None:
            rows.append(best[1])
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--igc_subset-dir", default=None, help="Folder with *.igc_subset files. Default: ./igc_subset next to this script")
    ap.add_argument("--circle-glob", default="outputs/*_circle_clusters_enriched_*.csv",
                    help="Glob for per-IGC circle enriched CSVs")
    ap.add_argument("--alt-glob", default="outputs/*_altitude_clusters_enriched_*.csv",
                    help="Glob for per-IGC altitude enriched CSVs")
    args = ap.parse_args()

    # Resolve default igc_subset dir to ./igc_subset next to this script
    if args.igc-dir is None:
        script_dir = Path(__file__).resolve().parent
        igc_dir = script_dir / "igc_subset"
    else:
        igc_dir = Path(args.igc_dir)

    if not igc_dir.exists() or not igc_dir.is_dir():
        print(f"[batch] IGC folder not found: {igc_dir}. Create ./igc_subset next to this script or pass --igc_subset-dir.")
        return

    igc_files = sorted(igc_dir.glob("*.igc_subset"))
    if not igc_files:
        print(f"[batch] No IGC files found in {igc_dir}."); return

    print(f"[batch] Scanning IGCs in: {igc_dir} (found {len(igc_files)})")

    all_rows = []
    for igc in igc_files:
        base = igc.stem  # basename without extension
        c_csv = latest_with_base(args.circle_glob, base)
        a_csv = latest_with_base(args.alt_glob, base)
        if not c_csv or not a_csv:
            print(f"[batch] Skip {base}: missing enriched CSVs (C={bool(c_csv)} A={bool(a_csv)})")
            continue
        circ = normalize_enriched(pd.read_csv(c_csv))
        alti = normalize_enriched(pd.read_csv(a_csv))
        if circ.empty or alti.empty:
            print(f"[batch] Skip {base}: empty enriched CSV(s)."); continue
        M = match_one_igc(circ, alti)
        if M.empty:
            print(f"[batch] No matches for {base} under current thresholds."); continue
        M.insert(0, "igc_base", base)
        all_rows.append(M)
        print(f"[batch] {base}: matched {len(M)} pairs.")

    if not all_rows:
        print("[batch] No matches across all IGCs."); return

    OUT = "outputs"
    os.makedirs(OUT, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    out_csv = os.path.join(OUT, f"batch_matched_{ts}.csv")
    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(out_csv, index=False)
    print(f"[batch] Wrote {out_csv}")
    with pd.option_context('display.max_columns', None, 'display.width', 160):
        print(df.head(12))

if __name__ == "__main__":
    main()
