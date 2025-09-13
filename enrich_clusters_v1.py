
#!/usr/bin/env python3
"""
enrich_clusters_v1.py

Adds altitude gain, duration, and avg climb rate to:
  - circle_clusters_filtered_*.csv  -> circle_clusters_enriched_<ts>.csv
  - altitude_clusters_*.csv         -> altitude_clusters_enriched_<ts>.csv

Usage: Run directly in PyCharm (no args). Edit DEFAULT_IGC if needed.
"""
import os, glob, warnings
from datetime import datetime, timezone
import numpy as np
import pandas as pd

DEFAULT_IGC = "2020-11-08 Lumpy Paterson 108645.igc"
OUTPUTS_DIR = "outputs"

try:
    from igc_utils import parse_igc, compute_derived
    HAVE_IGC_UTILS = True
except Exception:
    HAVE_IGC_UTILS = False
    warnings.warn("igc_utils not found; using minimal IGC parser.")

def minimal_parse_igc(path):
    lats=[]; lons=[]; alts=[]; times=[]
    day_hint=None
    with open(path,'r',errors='ignore') as f:
        for line in f:
            if line.startswith('HFDTE'):
                d=line.strip()[5:11]
                try: day_hint=datetime.strptime(d,"%d%m%y").date()
                except: day_hint=None
            if len(line)>35 and line[0]=='B':
                hh=int(line[1:3]); mm=int(line[3:5]); ss=int(line[5:7])
                lat = int(line[7:9]) + int(line[9:11])/60 + int(line[11:14])/60000
                if line[14]=='S': lat=-lat
                lon = int(line[15:18]) + int(line[18:20])/60 + int(line[20:23])/60000
                if line[23]=='W': lon=-lon
                try: alt=int(line[30:35])
                except: alt=np.nan
                if day_hint is None:
                    t=datetime(2000,1,1,hh,mm,ss,tzinfo=timezone.utc)
                else:
                    t=datetime(day_hint.year,day_hint.month,day_hint.day,hh,mm,ss,tzinfo=timezone.utc)
                lats.append(lat); lons.append(lon); alts.append(alt); times.append(t)
    return pd.DataFrame(dict(time=pd.to_datetime(times),lat=lats,lon=lons,gps_alt=alts))

def load_igc(path):
    if HAVE_IGC_UTILS:
        df=parse_igc(path)
        df=compute_derived(df) if 'gps_alt' in df.columns else df
    else:
        df=minimal_parse_igc(path)
    return df.sort_values('time').reset_index(drop=True)

def latest(glob_patt):
    files=glob.glob(glob_patt)
    return max(files,key=os.path.getmtime) if files else None

def compute_gain_duration(df,t0,t1):
    df=df.copy()
    df['time']=pd.to_datetime(df['time'])
    m=(df['time']>=t0)&(df['time']<=t1)
    sub=df.loc[m]
    if sub.empty:
        # nearest indices fallback
        diffs=(df['time']-t0).values.astype('timedelta64[s]').astype(float)
        i0=int(np.argmin(np.abs(diffs)))
        diffs=(df['time']-t1).values.astype('timedelta64[s]').astype(float)
        i1=int(np.argmin(np.abs(diffs)))
        if i1<i0: i0,i1=i1,i0
        sub=df.iloc[i0:i1+1]
    if sub.empty:
        return 0.0,0.0,0.0
    alt=sub['gps_alt'].astype(float).to_numpy()
    gain=float(np.nanmax(alt)-np.nanmin(alt))
    dur=float((sub['time'].iloc[-1]-sub['time'].iloc[0]).total_seconds())
    rate=gain/dur if dur>0 else 0.0
    return gain,dur,rate

def main():
    os.makedirs(OUTPUTS_DIR,exist_ok=True)
    circ_path=latest(os.path.join(OUTPUTS_DIR,"circle_clusters_filtered_*.csv"))
    alt_path =latest(os.path.join(OUTPUTS_DIR,"altitude_clusters_*.csv"))
    if circ_path is None or alt_path is None:
        print("[enrich] Missing required inputs in outputs/."); return

    print(f"[enrich] circles:  {os.path.basename(circ_path)}")
    print(f"[enrich] altitude: {os.path.basename(alt_path)}")

    igc=load_igc(DEFAULT_IGC)

    # Circles
    cdf=pd.read_csv(circ_path,parse_dates=['start_time','end_time'])
    rows=[]
    for _,r in cdf.iterrows():
        gain,dur,rate=compute_gain_duration(igc,r['start_time'],r['end_time'])
        rows.append(dict(cluster_id=int(r['cluster_id']), n=int(r.get('n',0)),
                         lat=float(r['lat']), lon=float(r['lon']),
                         start_time=r['start_time'], end_time=r['end_time'],
                         duration_s=round(dur,1), gain_m=round(gain,1), avg_rate_mps=round(rate,2)))
    c_enriched=pd.DataFrame(rows)

    # Altitude
    a=pd.read_csv(alt_path,parse_dates=['start_time','end_time'])
    a['duration_s']=(a['end_time']-a['start_time']).dt.total_seconds()
    if 'gain_m' not in a.columns: a['gain_m']=np.nan
    a['avg_rate_mps']=a['gain_m']/a['duration_s'].replace(0,np.nan)
    a_enriched=a[['cluster_id','n','lat','lon','start_time','end_time','duration_s','gain_m','avg_rate_mps']].copy()
    a_enriched[['duration_s','gain_m']]=a_enriched[['duration_s','gain_m']].round(1)
    a_enriched['avg_rate_mps']=a_enriched['avg_rate_mps'].round(2)

    ts=datetime.utcnow().strftime("%Y%m%d%H%M%S")
    out_c=os.path.join(OUTPUTS_DIR,f"circle_clusters_enriched_{ts}.csv")
    out_a=os.path.join(OUTPUTS_DIR,f"altitude_clusters_enriched_{ts}.csv")
    c_enriched.to_csv(out_c,index=False); a_enriched.to_csv(out_a,index=False)
    print(f"[enrich] Wrote {out_c}"); print(f"[enrich] Wrote {out_a}")

if __name__=="__main__":
    main()
