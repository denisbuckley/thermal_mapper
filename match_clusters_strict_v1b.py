
#!/usr/bin/env python3
"""
match_clusters_strict_v1b.py

Strict one-to-one matching between *circle* and *altitude* enriched clusters,
with safe console printing (no nested f-string quotes).

See v1 docstring for details.
"""

import os, glob, math
import numpy as np
import pandas as pd
from datetime import datetime

# ---------------- Parameters ----------------
EPS_M = 2000.0          # spatial threshold in meters (2 km)
MIN_OVL_FRAC = 0.20     # min required overlap fraction of the SHORTER interval
MAX_TIME_GAP_S = 15*60  # if no overlap, allow up to 15 minutes gap

OUTPUTS_DIR = "outputs"

# -------------- Helpers ---------------------
def latest(glob_patts):
    files = []
    for patt in glob_patts:
        files.extend(glob.glob(patt))
    if not files: return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]

def to_ts(s):
    return pd.to_datetime(s, utc=True, errors='coerce')

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1 = np.radians(lat1); p2 = np.radians(lat2)
    dphi = np.radians(lat2-lat1)
    dlmb = np.radians(lon2-lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlmb/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))

def time_overlap_and_gap(a0, a1, b0, b1):
    """Return (overlap_s, gap_s) with gap_s=0 when overlapping."""
    latest_start = max(a0, b0)
    earliest_end = min(a1, b1)
    ovl = (earliest_end - latest_start).total_seconds()
    if ovl > 0:
        return ovl, 0.0
    # no overlap → positive gap
    if a0 > b1:
        gap = (a0 - b1).total_seconds()
    else:
        gap = (b0 - a1).total_seconds()
    return 0.0, gap

def load_enriched():
    circ = latest([os.path.join(OUTPUTS_DIR,"circle_clusters_enriched_*.csv")])
    alti = latest([os.path.join(OUTPUTS_DIR,"altitude_clusters_enriched_*.csv")])
    if not circ or not alti:
        print("[match_strict v1b] Missing enriched inputs. Run enrichment first.")
        return None, None, None, None
    C = pd.read_csv(circ)
    A = pd.read_csv(alti)
    # normalize columns just in case
    for df in (C, A):
        if 'cluster_id' not in df.columns and 'thermal_id' in df.columns:
            df.rename(columns={'thermal_id':'cluster_id'}, inplace=True)
        # coerce times
        for col in ('start_time','end_time'):
            df[col] = to_ts(df[col])
    return C, A, circ, alti

# -------------- Matching --------------------
def build_candidates(C, A):
    cands = []
    for _, cr in C.iterrows():
        for _, ar in A.iterrows():
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

            gain_diff = abs(float(cr.get('gain_m', np.nan)) - float(ar.get('gain_m', np.nan)))
            rate_diff = abs(float(cr.get('avg_rate_mps', np.nan)) - float(ar.get('avg_rate_mps', np.nan)))
            gain_pen = 0.0 if (math.isnan(gain_diff)) else min(1.0, gain_diff / 200.0)
            rate_pen = 0.0 if (math.isnan(rate_diff)) else min(1.0, rate_diff / 5.0)
            score = (d_m/1000.0, -ovl_frac, gain_pen + 0.5*rate_pen)

            cands.append(dict(
                circle_id = int(cr['cluster_id']),
                alt_id    = int(ar['cluster_id']),
                dist_m    = d_m,
                overlap_s = ovl_s,
                overlap_frac = ovl_frac,
                time_gap_s = gap_s,
                gain_diff_m = None if math.isnan(gain_diff) else gain_diff,
                rate_diff_mps = None if math.isnan(rate_diff) else rate_diff,
                score = score
            ))
    cands.sort(key=lambda x: x['score'])
    return cands

def greedy_assign(cands):
    matched_c = set(); matched_a = set(); pairs = []
    for cand in cands:
        if cand['circle_id'] in matched_c or cand['alt_id'] in matched_a:
            continue
        matched_c.add(cand['circle_id']); matched_a.add(cand['alt_id']); pairs.append(cand)
    return pairs

def main():
    C, A, circ_path, alti_path = load_enriched()
    if C is None: return

    print("[match_strict v1b] Inputs:")
    print("  Circle enriched:", circ_path)
    print("  Altitude enriched:", alti_path)
    print(f"  Params: EPS_M={EPS_M:.0f} m | MIN_OVL_FRAC={MIN_OVL_FRAC:.2f} | MAX_TIME_GAP_S={MAX_TIME_GAP_S:.0f}s")

    cands = build_candidates(C, A)
    print(f"[match_strict v1b] Candidate pairs: {len(cands)}")

    pairs = greedy_assign(cands)
    print(f"[match_strict v1b] Strict 1:1 matches: {len(pairs)}")

    rows = []
    for p in pairs:
        rows.append(dict(
            circle_id=p['circle_id'],
            alt_id=p['alt_id'],
            dist_km=round(p['dist_m']/1000.0, 3),
            overlap_s=round(p['overlap_s'], 1),
            overlap_frac=round(p['overlap_frac'], 3),
            time_gap_s=round(p['time_gap_s'], 1),
            gain_diff_m=(None if p['gain_diff_m'] is None else round(p['gain_diff_m'],1)),
            rate_diff_mps=(None if p['rate_diff_mps'] is None else round(p['rate_diff_mps'],2)),
        ))
    M = pd.DataFrame(rows)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    out_csv = os.path.join(OUTPUTS_DIR, f"matched_clusters_{ts}.csv")
    M.to_csv(out_csv, index=False)
    print(f"[match_strict v1b] Wrote {out_csv}")

    if not M.empty:
        print("\ncircle_id  alt_id  dist_km  overlap_s  overlap_frac  time_gap_s  gain_diff_m  rate_diff_mps")
        print("---------  ------  -------  ---------  ------------  ----------  -----------  --------------")
        for _, r in M.iterrows():
            gain_str = "" if pd.isna(r['gain_diff_m']) else f"{r['gain_diff_m']:.1f}"
            rate_str = "" if pd.isna(r['rate_diff_mps']) else f"{r['rate_diff_mps']:.2f}"
            print(f"{int(r['circle_id']):9d}  {int(r['alt_id']):6d}  {r['dist_km']:7.3f}  {r['overlap_s']:9.1f}"
                  f"  {r['overlap_frac']:12.3f}  {r['time_gap_s']:10.1f}  {gain_str:>11}  {rate_str:>14}")
    else:
        print("[match_strict v1b] No matches under current thresholds. Consider widening EPS_M or lowering MIN_OVL_FRAC.")

if __name__ == "__main__":
    main()
