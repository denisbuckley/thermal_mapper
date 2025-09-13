
#!/usr/bin/env python3
"""
print_table_v3.py

Print BOTH latest enriched tables (if present) in one run:
  - outputs/circle_clusters_enriched_*.csv
  - outputs/altitude_clusters_enriched_*.csv

Tables are printed with aligned columns and 1-decimal numeric formatting.
If one is missing, the other is
still printed. No arguments required.
"""

import os, glob
import pandas as pd

HEADERS = [
    ("cluster_id",    11),
    ("n",              5),
    ("lat",           10),
    ("lon",           11),
    ("start_time",    20),
    ("end_time",      20),
    ("duration_s",    12),
    ("gain_m",        10),
    ("avg_rate_mps",  14),
]

def latest(glob_patts):
    files = []
    for patt in glob_patts:
        files.extend(glob.glob(patt))
    if not files: return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]

def fmt_cell(col, val):
    if col in ("lat","lon","duration_s","gain_m","avg_rate_mps"):
        try: return f"{float(val):.1f}"
        except Exception: return str(val)
    return str(val)

def print_table(path, title):
    if not path or not os.path.exists(path):
        print(f"[print_table_v3] No file for {title}."); return
    df = pd.read_csv(path)
    cols_present = [c for c,_ in HEADERS if c in df.columns]
    if not cols_present:
        print(f"[print_table_v3] {title}: unexpected columns in {path}")
        print("[print_table_v3] Columns:", list(df.columns))
        return

    print("="*len(title))
    print(title)
    print("="*len(title))
    print(f"File: {path}")
    # header
    line = ""
    for col, w in HEADERS:
        if col in cols_present:
            line += f"{col:<{w}}"
    print(line)
    print("-" * len(line))
    # rows
    for _, r in df.iterrows():
        line = ""
        for col, w in HEADERS:
            if col in cols_present:
                line += f"{fmt_cell(col, r[col]):<{w}}"
        print(line)
    print("")

def main():
    circ = latest([os.path.join("outputs","circle_clusters_enriched_*.csv")])
    alti = latest([os.path.join("outputs","altitude_clusters_enriched_*.csv")])

    if not circ and not alti:
        print("[print_table_v3] No enriched CSVs found. Run your enrichment step first.")
        return

    if circ: print_table(circ, "Circle Clusters (enriched)")
    if alti: print_table(alti, "Altitude Clusters (enriched)")

if __name__ == "__main__":
    main()
