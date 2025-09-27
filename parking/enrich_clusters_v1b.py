
#!/usr/bin/env python3
"""
enrich_clusters_v1b.py

Robust enrichment of circle & altitude cluster CSVs:
- Accepts varied column names for lat/lon (lat, latitude, center_lat, avg_lat, y) and lon (lon, longitude, center_lon, avg_lon, x)
- Accepts varied time columns (start_time/first_time, end_time/last_time)
- Computes duration, gain, avg climb rate from IGC between cluster start/end

Outputs:
  outputs/circle_clusters_enriched_<ts>.csv
  outputs/altitude_clusters_enriched_<ts>.csv
"""

import os, glob, warnings
from datetime import datetime, timezone
import numpy as np
import pandas as pd

DEFAULT_IGC = "2020-11-08 Lumpy Paterson 108645.igc"
OUTPUTS_DIR = "outputs"

# Optional igc_utils
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

def latest(patt):
    files=glob.glob(patt)
    return max(files, key=os.path.getmtime) if files else None

def normalize_columns(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Rename possible variants to expected names: lat, lon, start_time, end_time, cluster_id, n"""
    mapping = {}
    # lat
    for cand in ['lat','latitude','center_lat','avg_lat','y']:
        if cand in df.columns: mapping[cand]='lat'; break
    # lon
    for cand in ['lon','longitude','center_lon','avg_lon','x']:
        if cand in df.columns: mapping[cand]='lon'; break
    # times
    if 'start_time' not in df.columns:
        for cand in ['first_time','start','t_start']:
            if cand in df.columns: mapping[cand]='start_time'; break
    if 'end_time' not in df.columns:
        for cand in ['last_time','end','t_end']:
            if cand in df.columns: mapping[cand]='end_time'; break
    # ids and counts
    if 'cluster_id' not in df.columns:
        for cand in ['cluster','id']:
            if cand in df.columns: mapping[cand]='cluster_id'; break
    if 'n' not in df.columns:
        for cand in ['count','size','num','k']:
            if cand in df.columns: mapping[cand]='n'; break

    if mapping:
        df = df.rename(columns=mapping)
    return df

def compute_gain_duration(igc: pd.DataFrame, t0, t1):
    igc = igc.copy()
    igc['time'] = pd.to_datetime(igc['time'])
    m=(igc['time']>=t0)&(igc['time']<=t1)
    sub=igc.loc[m]
    if sub.empty:
        # nearest fallback
        diffs=(igc['time']-t0).values.astype('timedelta64[s]').astype(float)
        i0=int(np.argmin(np.abs(diffs)))
        diffs=(igc['time']-t1).values.astype('timedelta64[s]').astype(float)
        i1=int(np.argmin(np.abs(diffs)))
        if i1<i0: i0,i1=i1,i0
        sub=igc.iloc[i0:i1+1]
    if sub.empty:
        return 0.0,0.0,0.0
    alt=sub['gps_alt'].astype(float).to_numpy()
    gain=float(np.nanmax(alt)-np.nanmin(alt))
    dur=float((sub['time'].iloc[-1]-sub['time'].iloc[0]).total_seconds())
    rate=gain/dur if dur>0 else 0.0
    return gain,dur,rate

def main():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    circ_path = latest(os.path.join(OUTPUTS_DIR,"circle_clusters_filtered_*.csv"))
    alt_path  = latest(os.path.join(OUTPUTS_DIR,"altitude_clusters_*.csv"))
    if circ_path is None and alt_path is None:
        print("[enrich v1b] No inputs found in outputs/."); return

    igc = load_igc(DEFAULT_IGC)

    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    if circ_path:
        cdf = pd.read_csv(circ_path, parse_dates=['start_time','end_time'], infer_datetime_format=True)
        cdf = normalize_columns(cdf, 'circle')
        missing = [c for c in ['lat','lon','start_time','end_time'] if c not in cdf.columns]
        if missing:
            print(f"[enrich v1b] circle CSV missing columns {missing}; cannot enrich.")
        else:
            rows=[]
            for _,r in cdf.iterrows():
                gain,dur,rate = compute_gain_duration(igc, r['start_time'], r['end_time'])
                rows.append(dict(
                    cluster_id = int(r.get('cluster_id', len(rows)+1)),
                    n          = int(r.get('n', 0)),
                    lat        = float(r['lat']), lon=float(r['lon']),
                    start_time = r['start_time'], end_time=r['end_time'],
                    duration_s = round(dur,1), gain_m=round(gain,1), avg_rate_mps=round(rate,2)
                ))
            out_c = os.path.join(OUTPUTS_DIR, f"circle_clusters_enriched_{ts}.csv")
            pd.DataFrame(rows).to_csv(out_c, index=False)
            print(f"[enrich v1b] Wrote {out_c}")

    if alt_path:
        adf = pd.read_csv(alt_path, parse_dates=['start_time','end_time'], infer_datetime_format=True)
        adf = normalize_columns(adf, 'alt')
        missing = [c for c in ['lat','lon','start_time','end_time'] if c not in adf.columns]
        if missing:
            print(f"[enrich v1b] altitude CSV missing columns {missing}; cannot enrich.")
        else:
            # duration & gain from file if present; else compute from IGC window
            if 'gain_m' not in adf.columns:
                # compute from IGC bounds as fallback
                gains=[]; durs=[]; rates=[]
                for _,r in adf.iterrows():
                    g,d,rt = compute_gain_duration(igc, r['start_time'], r['end_time'])
                    gains.append(g); durs.append(d); rates.append(rt if d>0 else 0.0)
                adf['gain_m']=gains; adf['duration_s']=durs; adf['avg_rate_mps']=adf['gain_m']/adf['duration_s'].replace(0,np.nan)
            else:
                if 'duration_s' not in adf.columns:
                    adf['duration_s']=(adf['end_time']-adf['start_time']).dt.total_seconds()
                adf['avg_rate_mps']=adf['gain_m']/adf['duration_s'].replace(0,np.nan)

            keep=['cluster_id','n','lat','lon','start_time','end_time','duration_s','gain_m','avg_rate_mps']
            aout = adf[[c for c in keep if c in adf.columns]].copy()
            aout[['duration_s','gain_m']] = aout[['duration_s','gain_m']].astype(float).round(1)
            aout['avg_rate_mps'] = aout['avg_rate_mps'].astype(float).round(2)
            out_a = os.path.join(OUTPUTS_DIR, f"altitude_clusters_enriched_{ts}.csv")
            aout.to_csv(out_a, index=False)
            print(f"[enrich v1b] Wrote {out_a}")

if __name__ == "__main__":
    main()
