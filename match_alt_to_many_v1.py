
#!/usr/bin/env python3
"""
match_alt_to_many_v1.py

Inverse matcher: for each *altitude* cluster, pick the single best *circle* cluster.
Allows many altitude clusters to map to the same circle (many→one in the *other* direction).

Inputs (auto from ./outputs):
  - circle_clusters_enriched_*.csv
  - altitude_clusters_enriched_*.csv

Outputs:
  - outputs/matched_alt_to_many_<ts>.csv
      Columns: c_id, a_id, d_km, ovl_s, ovl_f, gap_s, c_gain_m, c_rate, a_gain_m, a_rate

Params identical to other matchers: EPS_M, MIN_OVL_FRAC, MAX_TIME_GAP_S.
"""

import os, glob, math
import numpy as np
import pandas as pd
from datetime import datetime

EPS_M = 2000.0
MIN_OVL_FRAC = 0.20
MAX_TIME_GAP_S = 15*60
OUTPUTS_DIR = "outputs"

def latest(patts):
    files=[]; [files.extend(glob.glob(p)) for p in patts]
    if not files: return None
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files[0]

def to_ts(s):
    return pd.to_datetime(s, utc=True, errors='coerce')

def haversine_m(lat1, lon1, lat2, lon2):
    R=6371000.0
    p1=np.radians(lat1); p2=np.radians(lat2)
    dphi=np.radians(lat2-lat1); dlmb=np.radians(lon2-lon1)
    a=np.sin(dphi/2.0)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlmb/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))

def time_overlap_and_gap(a0,a1,b0,b1):
    latest_start=max(a0,b0); earliest_end=min(a1,b1)
    ovl=(earliest_end-latest_start).total_seconds()
    if ovl>0: return ovl,0.0
    gap=(a0-b1).total_seconds() if a0>b1 else (b0-a1).total_seconds()
    return 0.0,gap

def load_enriched():
    circ=latest([f"{OUTPUTS_DIR}/circle_clusters_enriched_*.csv"])
    alti=latest([f"{OUTPUTS_DIR}/altitude_clusters_enriched_*.csv"])
    if not circ or not alti:
        print("[alt2many] Missing enriched inputs."); return None,None,None,None
    C=pd.read_csv(circ); A=pd.read_csv(alti)
    for df in (C,A):
        if 'cluster_id' not in df.columns and 'thermal_id' in df.columns:
            df.rename(columns={'thermal_id':'cluster_id'}, inplace=True)
        for col in ('start_time','end_time'): df[col]=to_ts(df[col])
        if 'gain_m' not in df.columns: df['gain_m']=np.nan
        if 'avg_rate_mps' not in df.columns: df['avg_rate_mps']=np.nan
    return C,A,circ,alti

def build_candidates(C,A):
    cands=[]
    for _, ar in A.iterrows():
        for _, cr in C.iterrows():
            d_m=float(haversine_m(cr['lat'],cr['lon'],ar['lat'],ar['lon']))
            if d_m>EPS_M: continue
            ovl_s,gap_s=time_overlap_and_gap(cr['start_time'],cr['end_time'],ar['start_time'],ar['end_time'])
            c_dur=max(1.0,float((cr['end_time']-cr['start_time']).total_seconds()))
            a_dur=max(1.0,float((ar['end_time']-ar['start_time']).total_seconds()))
            shorter=min(c_dur,a_dur); ovl_frac=max(0.0, ovl_s/shorter)
            if ovl_s<=0 and gap_s>MAX_TIME_GAP_S: continue
            if ovl_s>0 and ovl_frac<MIN_OVL_FRAC: continue

            gain_diff=abs(float(cr.get('gain_m',np.nan))-float(ar.get('gain_m',np.nan)))
            rate_diff=abs(float(cr.get('avg_rate_mps',np.nan))-float(ar.get('avg_rate_mps',np.nan)))
            gain_pen=0.0 if np.isnan(gain_diff) else min(1.0, gain_diff/200.0)
            rate_pen=0.0 if np.isnan(rate_diff) else min(1.0, rate_diff/5.0)
            score=(d_m/1000.0, -ovl_frac, gain_pen+0.5*rate_pen)

            cands.append(dict(
                c_id=int(cr['cluster_id']), a_id=int(ar['cluster_id']),
                dist_m=d_m, ovl_s=ovl_s, ovl_f=ovl_frac, gap_s=gap_s,
                c_gain_m=float(cr.get('gain_m',np.nan)), c_rate=float(cr.get('avg_rate_mps',np.nan)),
                a_gain_m=float(ar.get('gain_m',np.nan)), a_rate=float(ar.get('avg_rate_mps',np.nan)),
                score=score
            ))
    cands.sort(key=lambda x: x['score'])
    return cands

def pick_best_per_alt(cands):
    best={}
    for cand in cands:
        aid=cand['a_id']
        if aid not in best or cand['score']<best[aid]['score']:
            best[aid]=cand
    return list(best.values())

def main():
    C,A,circ_path,alti_path=load_enriched()
    if C is None: return
    print("[alt2many] Inputs:"); print("  Circle enriched:", circ_path); print("  Altitude enriched:", alti_path)
    print(f"  Params: EPS_M={EPS_M:.0f} m | MIN_OVL_FRAC={MIN_OVL_FRAC:.2f} | MAX_TIME_GAP_S={MAX_TIME_GAP_S:.0f}s")

    cands=build_candidates(C,A)
    print(f"[alt2many] Candidate pairs: {len(cands)}")
    picks=pick_best_per_alt(cands)
    print(f"[alt2many] Altitudes matched (alt→best circle): {len(picks)}")

    import pandas as pd, os
    from datetime import datetime
    rows=[dict(
        c_id=p['c_id'], a_id=p['a_id'], d_km=round(p['dist_m']/1000.0,3),
        ovl_s=round(p['ovl_s'],1), ovl_f=round(p['ovl_f'],3), gap_s=round(p['gap_s'],1),
        c_gain_m=None if pd.isna(p['c_gain_m']) else round(p['c_gain_m'],1),
        c_rate=None if pd.isna(p['c_rate']) else round(p['c_rate'],2),
        a_gain_m=None if pd.isna(p['a_gain_m']) else round(p['a_gain_m'],1),
        a_rate=None if pd.isna(p['a_rate']) else round(p['a_rate'],2),
    ) for p in picks]
    M=pd.DataFrame(rows)
    os.makedirs(OUTPUTS_DIR,exist_ok=True)
    ts=datetime.utcnow().strftime("%Y%m%d%H%M%S")
    out=f"{OUTPUTS_DIR}/matched_alt_to_many_{ts}.csv"
    M.to_csv(out,index=False); print(f"[alt2many] Wrote {out}")

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
        print("[alt2many] No matches under current thresholds.")

if __name__ == "__main__":
    main()
