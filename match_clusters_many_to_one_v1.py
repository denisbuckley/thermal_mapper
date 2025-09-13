
#!/usr/bin/env python3
"""
match_clusters_many_to_one_v1.py

Many-to-one matching: multiple *circle* clusters may map to the same *altitude*
cluster (but each circle matches at most one altitude). Uses the same columns
as strict v1c.

Inputs (auto-detected from ./outputs):
  - circle_clusters_enriched_*.csv
  - altitude_clusters_enriched_*.csv

Outputs:
  - outputs/matched_clusters_many_to_one_<ts>.csv
      Columns:
        c_id, a_id, d_km, ovl_s, ovl_f, gap_s, c_gain_m, c_rate, a_gain_m, a_rate

Params:
  EPS_M=2000.0       # max distance (meters)
  MIN_OVL_FRAC=0.20  # min overlap fraction of the shorter interval
  MAX_TIME_GAP_S=900 # if no overlap, allow up to 15 minutes gap

Strategy:
  - Build all candidate pairs subject to thresholds.
  - For each *circle* cluster, pick the best altitude cluster (smallest distance,
    then higher overlap, then smaller gain/rate penalty). Different circles may
    pick the same altitude cluster (many-to-one allowed).
"""

import os, glob, math
import numpy as np
import pandas as pd
from datetime import datetime

# === Injected by apply_tuning_patches_v1.py ===
from tuning_loader import load_tuning, override_globals
_tuning = load_tuning("config/tuning_params.csv")
override_globals(globals(), _tuning, allowed={
    "EPS_M","MIN_OVL_FRAC","MAX_TIME_GAP_S"
})



# ---------------- Parameters ----------------
EPS_M = 2000.0
MIN_OVL_FRAC = 0.20
MAX_TIME_GAP_S = 15*60

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
    latest_start = max(a0, b0)
    earliest_end = min(a1, b1)
    ovl = (earliest_end - latest_start).total_seconds()
    if ovl > 0:
        return ovl, 0.0
    if a0 > b1:
        gap = (a0 - b1).total_seconds()
    else:
        gap = (b0 - a1).total_seconds()
    return 0.0, gap

def load_enriched():
    circ = latest([os.path.join(OUTPUTS_DIR,"circle_clusters_enriched_*.csv")])
    alti = latest([os.path.join(OUTPUTS_DIR,"altitude_clusters_enriched_*.csv")])
    if not circ or not alti:
        print("[many2one] Missing enriched inputs. Run enrichment first.")
        return None, None, None, None
    C = pd.read_csv(circ)
    A = pd.read_csv(alti)
    # normalize
    for df in (C, A):
        if 'cluster_id' not in df.columns and 'thermal_id' in df.columns:
            df.rename(columns={'thermal_id':'cluster_id'}, inplace=True)
        for col in ('start_time','end_time'):
            df[col] = to_ts(df[col])
        if 'gain_m' not in df.columns: df['gain_m'] = np.nan
        if 'avg_rate_mps' not in df.columns: df['avg_rate_mps'] = np.nan
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
                c_gain_m  = float(cr.get('gain_m', np.nan)),
                c_rate    = float(cr.get('avg_rate_mps', np.nan)),
                a_gain_m  = float(ar.get('gain_m', np.nan)),
                a_rate    = float(ar.get('avg_rate_mps', np.nan)),
                score = score
            ))
    # sort by score (best first)
    cands.sort(key=lambda x: x['score'])
    return cands

def pick_best_per_circle(cands):
    """Allow many circles → one altitude. Keep only the best altitude for each circle."""
    best = {}
    for cand in cands:
        cid = cand['circle_id']
        if cid not in best:
            best[cid] = cand
        else:
            # compare by score tuple
            if cand['score'] < best[cid]['score']:
                best[cid] = cand
    return list(best.values())

# -------------- Runner ----------------------
def main():
    C, A, circ_path, alti_path = load_enriched()
    if C is None: return

    print("[many2one] Inputs:")
    print("  Circle enriched:", circ_path)
    print("  Altitude enriched:", alti_path)
    print(f"  Params: EPS_M={EPS_M:.0f} m | MIN_OVL_FRAC={MIN_OVL_FRAC:.2f} | MAX_TIME_GAP_S={MAX_TIME_GAP_S:.0f}s")

    cands = build_candidates(C, A)
    print(f"[many2one] Candidate pairs: {len(cands)}")

    picks = pick_best_per_circle(cands)
    print(f"[many2one] Circles matched (many→one): {len(picks)}")

    rows = []
    for p in picks:
        rows.append(dict(
            c_id = p['circle_id'],
            a_id = p['alt_id'],
            d_km = round(p['dist_m']/1000.0, 3),
            ovl_s = round(p['overlap_s'], 1),
            ovl_f = round(p['overlap_frac'], 3),
            gap_s = round(p['time_gap_s'], 1),
            c_gain_m = None if math.isnan(p['c_gain_m']) else round(p['c_gain_m'], 1),
            c_rate   = None if math.isnan(p['c_rate'])   else round(p['c_rate'], 2),
            a_gain_m = None if math.isnan(p['a_gain_m']) else round(p['a_gain_m'], 1),
            a_rate   = None if math.isnan(p['a_rate'])   else round(p['a_rate'], 2),
        ))
    M = pd.DataFrame(rows)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    out_csv = os.path.join(OUTPUTS_DIR, f"matched_clusters_many_to_one_{ts}.csv")
    M.to_csv(out_csv, index=False)
    print(f"[many2one] Wrote {out_csv}")

    # Console view
    if not M.empty:
        print("\nc_id  a_id   d_km   ovl_s   ovl_f   gap_s   c_gain  c_rate   a_gain  a_rate")
        print("----  ----  ------  ------  ------  ------  ------  ------  ------  ------")
        for _, r in M.iterrows():
            cg = "" if pd.isna(r['c_gain_m']) else f"{r['c_gain_m']:.1f}"
            cr = "" if pd.isna(r['c_rate'])   else f"{r['c_rate']:.2f}"
            ag = "" if pd.isna(r['a_gain_m']) else f"{r['a_gain_m']:.1f}"
            ar = "" if pd.isna(r['a_rate'])   else f"{r['a_rate']:.2f}"
            print(f"{int(r['c_id']):4d}  {int(r['a_id']):4d}  {r['d_km']:6.3f}  {r['ovl_s']:6.1f}  {r['ovl_f']:6.3f}"
                  f"  {r['gap_s']:6.1f}  {cg:>6}  {cr:>6}  {ag:>6}  {ar:>6}")
    else:
        print("[many2one] No matches under current thresholds. Consider widening EPS_M or lowering MIN_OVL_FRAC.")

if __name__ == "__main__":
    main()