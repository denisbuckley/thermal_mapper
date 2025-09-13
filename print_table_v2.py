
#!/usr/bin/env python3
"""
print_table_v2.py

Auto-detects the latest enriched cluster CSV in ./outputs and prints it
as an aligned console table with one decimal place for numeric fields.
"""
import os, glob
import pandas as pd

CSV_PATH = ""  # leave empty to auto-detect

HEADERS=[("cluster_id",11),("n",5),("lat",10),("lon",11),
         ("start_time",20),("end_time",20),("duration_s",12),
         ("gain_m",10),("avg_rate_mps",14)]

def latest(patts):
    files=[]; [files.extend(glob.glob(p)) for p in patts]
    if not files: return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]

def fmt(col,val):
    if col in ("lat","lon","duration_s","gain_m","avg_rate_mps"):
        try: return f"{float(val):.1f}"
        except: return str(val)
    return str(val)

def main():
    global CSV_PATH
    if not CSV_PATH:
        CSV_PATH = latest(["outputs/circle_clusters_enriched_*.csv",
                           "outputs/altitude_clusters_enriched_*.csv"]) or \
                   latest(["outputs/*_enriched_*.csv"])
    if not CSV_PATH or not os.path.exists(CSV_PATH):
        print("[print_table_v2] No enriched CSV found in outputs/."); return

    df=pd.read_csv(CSV_PATH)
    cols=[c for c,_ in HEADERS if c in df.columns]
    if not cols:
        print(f"[print_table_v2] Unexpected columns in {CSV_PATH}: {list(df.columns)}"); return

    print(f"[print_table_v2] Showing: {CSV_PATH}")
    line=""; 
    for c,w in HEADERS:
        if c in cols: line+=f"{c:<{w}}"
    print(line); print("-"*len(line))
    for _,r in df.iterrows():
        line=""
        for c,w in HEADERS:
            if c in cols: line+=f"{fmt(c,r[c]):<{w}}"
        print(line)

if __name__=="__main__":
    main()
