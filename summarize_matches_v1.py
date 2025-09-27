
#!/usr/bin/env python3
"""
summarize_matches_v1.py

Reads the latest matched cluster CSVs from ./outputs and produces BOTH:
  - per-altitude summary (how many circles matched each altitude thermal, totals/means)
  - per-circle summary (how many altitude clusters matched each circle, totals/means)

Inputs (auto-detected, priority order):
  1) outputs/matched_clusters_many_to_one_*.csv
  2) outputs/matched_clusters_*.csv (strict)

Outputs:
  - outputs/matched_summary_per_alt_<ts>.csv
      a_id, n_circles, mean_d_km, sum_c_gain, mean_c_rate, sum_a_gain, mean_a_rate
  - outputs/matched_summary_per_circle_<ts>.csv
      c_id, n_alts, mean_d_km, sum_a_gain, mean_a_rate, sum_c_gain, mean_c_rate

Also prints compact console tables for both summaries.
"""

import os, glob
import pandas as pd
from datetime import datetime

# === Injected by apply_tuning_patches_v1.py ===
from tuning_loader import load_tuning, override_globals
_tuning = load_tuning("config/tuning_params.csv")
override_globals(globals(), _tuning, allowed={
    "EPS_M","MIN_OVL_FRAC","MAX_TIME_GAP_S"
})



OUTPUTS_DIR = "outputs"

def latest(patts):
    files=[]
    for p in patts:
        files.extend(glob.glob(p))
    if not files: return None
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files[0]

def main():
    path = latest([
        f"{OUTPUTS_DIR}/matched_clusters_many_to_one_*.csv",
        f"{OUTPUTS_DIR}/matched_clusters_*.csv",
    ])
    if not path:
        print("[summ] No matched CSV found. Run a matcher first."); return

    M = pd.read_csv(path)
    print(f"[summ] Using: {path}")

    # Ensure expected columns exist
    required = {'c_id','a_id','d_km','c_gain_m','c_rate','a_gain_m','a_rate'}
    missing = required - set(M.columns)
    if missing:
        print("[summ] Missing columns:", missing)
        print("[summ] Aborting.")
        return

    # Per-altitude aggregation
    per_alt = M.groupby('a_id').agg(
        n_circles=('c_id','count'),
        mean_d_km=('d_km','mean'),
        sum_c_gain=('c_gain_m','sum'),
        mean_c_rate=('c_rate','mean'),
        sum_a_gain=('a_gain_m','sum'),
        mean_a_rate=('a_rate','mean'),
    ).reset_index()

    # Per-circle aggregation
    per_circ = M.groupby('c_id').agg(
        n_alts=('a_id','count'),
        mean_d_km=('d_km','mean'),
        sum_a_gain=('a_gain_m','sum'),
        mean_a_rate=('a_rate','mean'),
        sum_c_gain=('c_gain_m','sum'),
        mean_c_rate=('c_rate','mean'),
    ).reset_index()

    # Rounding
    for df in (per_alt, per_circ):
        for col in df.columns:
            if df[col].dtype.kind in "fc":
                df[col] = df[col].round(3)

    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    out_alt = f"{OUTPUTS_DIR}/matched_summary_per_alt_{ts}.csv"
    out_cir = f"{OUTPUTS_DIR}/matched_summary_per_circle_{ts}.csv"
    per_alt.to_csv(out_alt, index=False)
    per_circ.to_csv(out_cir, index=False)
    print(f"[summ] Wrote {out_alt}")
    print(f"[summ] Wrote {out_cir}")

    # Console tables
    if not per_alt.empty:
        print("\nPer-altitude summary")
        print("a_id  n_cir  mean_d  sum_c_g  mean_c_r  sum_a_g  mean_a_r")
        print("----  -----  ------  -------  --------  -------  --------")
        for _, r in per_alt.iterrows():
            print(f"{int(r['a_id']):4d}  {int(r['n_circles']):5d}  {r['mean_d_km']:6.3f}  {r['sum_c_gain']:7.1f}"
                  f"  {r['mean_c_rate']:8.2f}  {r['sum_a_gain']:7.1f}  {r['mean_a_rate']:8.2f}")
    if not per_circ.empty:
        print("\nPer-circle summary")
        print("c_id  n_alts  mean_d  sum_a_g  mean_a_r  sum_c_g  mean_c_r")
        print("----  ------  ------  -------  --------  -------  --------")
        for _, r in per_circ.iterrows():
            print(f"{int(r['c_id']):4d}  {int(r['n_alts']):6d}  {r['mean_d_km']:6.3f}  {r['sum_a_gain']:7.1f}"
                  f"  {r['mean_a_rate']:8.2f}  {r['sum_c_gain']:7.1f}  {r['mean_c_rate']:8.2f}")

if __name__ == "__main__":
    main()