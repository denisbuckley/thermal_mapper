
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_match_igc_v3.py — per‑flight matching (altitude↔circles), single combined CSV

For each *.igc in --igc-dir:
  1) Run altitude_gain_v3g.py and circles_clean_v2c.py via runpy (inject DEFAULT_IGC).
  2) Collect the newest cluster CSVs those detectors emitted (content-detected).
  3) Perform per‑flight matching (alt vs circles) with spatial/time overlap gates.
  4) Append matches to one combined CSV: outputs/batch_matches_<TS>.csv
  5) Write a single run log:   outputs/batch_run_<TS>.log

Notes
- Uses tuning_loader.py if present to override EPS_M, MIN_OVL_FRAC, MAX_TIME_GAP_S.
- Still saves per‑IGC cluster CSV copies into outputs/batch_clusters/ for debugging.

Columns in combined CSV (per match):
  igc_base, d_km, ovl_s, ovl_f, gap_s,
  c_id, c_lat, c_lon, c_n, c_gain_m, c_rate,
  a_id, a_lat, a_lon, a_n, a_gain_m, a_rate,
  c_start, c_end, a_start, a_end
"""

from __future__ import annotations

import argparse, runpy, sys
from pathlib import Path
from datetime import datetime
import pandas as pd, numpy as np

# ---- Matching thresholds (overridable) ----
EPS_M = 2000.0           # spatial tolerance (meters)
MIN_OVL_FRAC = 0.20      # minimum overlap fraction (on the shorter interval)
MAX_TIME_GAP_S = 15*60   # allowed gap if no overlap (seconds)

# Optional tuning via CSV
try:
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

def latest_after(folder: Path, mtime: float):
    # return files in folder modified after mtime (newest first)
    cand = [p for p in folder.glob("*.csv") if p.stat().st_mtime >= mtime]
    cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cand

def looks_like_alt(df: pd.DataFrame) -> bool:
    cols = {c.lower() for c in df.columns}
    return ('lat' in cols and 'lon' in cols) and (('gain_m' in cols) or ('avg_rate_mps' in cols))

def looks_like_circ(df: pd.DataFrame) -> bool:
    cols = {c.lower() for c in df.columns}
    return ('lat' in cols and 'lon' in cols) and ('n' in cols)

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame(columns=[
        'cluster_id','lat','lon','start_time','end_time','n','gain_m','avg_rate_mps'
    ])
    out = df.copy()
    ren = {}
    if 'thermal_id' in out.columns and 'cluster_id' not in out.columns:
        ren['thermal_id'] = 'cluster_id'
    if 'avg_rate' in out.columns and 'avg_rate_mps' not in out.columns:
        ren['avg_rate'] = 'avg_rate_mps'
    out = out.rename(columns=ren)
    # ensure cols
    for col in ('cluster_id','lat','lon','start_time','end_time','n','gain_m','avg_rate_mps'):
        if col not in out.columns: out[col] = np.nan
    # coerce times
    for col in ('start_time','end_time'):
        out[col] = to_ts(out[col])
    return out

def time_overlap_and_gap(a0, a1, b0, b1):
    latest_start = max(a0, b0)
    earliest_end = min(a1, b1)
    ovl = (earliest_end - latest_start).total_seconds() if (not pd.isna(latest_start) and not pd.isna(earliest_end)) else 0.0
    if ovl > 0:
        return ovl, 0.0
    # if either is NaT we can't reason; treat as big gap
    if pd.isna(a0) or pd.isna(a1) or pd.isna(b0) or pd.isna(b1):
        return 0.0, 1e12
    gap = (a0 - b1).total_seconds() if a0 > b1 else (b0 - a1).total_seconds()
    return 0.0, gap

def match_per_flight(circ_df: pd.DataFrame, alt_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, cr in circ_df.iterrows():
        best = None
        for _, ar in alt_df.iterrows():
            if any(pd.isna(v) for v in (cr['lat'], cr['lon'], ar['lat'], ar['lon'])):
                continue
            d_m = float(haversine_m(cr['lat'], cr['lon'], ar['lat'], ar['lon']))
            if d_m > EPS_M:  # spatial gate
                continue
            ovl_s, gap_s = time_overlap_and_gap(cr['start_time'], cr['end_time'], ar['start_time'], ar['end_time'])
            c_dur = max(1.0, float((cr['end_time'] - cr['start_time']).total_seconds())) if (pd.notna(cr['start_time']) and pd.notna(cr['end_time'])) else 1.0
            a_dur = max(1.0, float((ar['end_time'] - ar['start_time']).total_seconds())) if (pd.notna(ar['start_time']) and pd.notna(ar['end_time'])) else 1.0
            shorter = min(c_dur, a_dur)
            ovl_frac = max(0.0, ovl_s / shorter) if shorter > 0 else 0.0
            # temporal gate
            if ovl_s <= 0 and gap_s > MAX_TIME_GAP_S:
                continue
            if ovl_s > 0 and ovl_frac < MIN_OVL_FRAC:
                continue
            score = (d_m/1000.0, -ovl_frac)  # prefer closer & more overlap
            cand = (score, dict(
                d_km=round(d_m/1000.0, 3), ovl_s=round(ovl_s, 1), ovl_f=round(ovl_frac, 3), gap_s=round(gap_s, 1),
                c_id=int(cr.get('cluster_id', -1)), c_lat=float(cr['lat']), c_lon=float(cr['lon']),
                c_n=None if pd.isna(cr.get('n')) else int(cr['n']),
                c_gain_m=None if pd.isna(cr.get('gain_m')) else float(cr['gain_m']),
                c_rate=None if pd.isna(cr.get('avg_rate_mps')) else float(cr['avg_rate_mps']),
                c_start=cr.get('start_time'), c_end=cr.get('end_time'),
                a_id=int(ar.get('cluster_id', -1)), a_lat=float(ar['lat']), a_lon=float(ar['lon']),
                a_n=None if pd.isna(ar.get('n')) else int(ar['n']),
                a_gain_m=None if pd.isna(ar.get('gain_m')) else float(ar['gain_m']),
                a_rate=None if pd.isna(ar.get('avg_rate_mps')) else float(ar['avg_rate_mps']),
                a_start=ar.get('start_time'), a_end=ar.get('end_time')
            ))
            if best is None or cand[0] < best[0]:
                best = cand
        if best is not None:
            rows.append(best[1])
    return pd.DataFrame(rows)

def safe_run_detector(script: Path, igc_path: Path) -> str:
    try:
        runpy.run_path(str(script), init_globals={"DEFAULT_IGC": str(igc_path), "__name__": "__main__"})
        return "ok"
    except SystemExit as se:
        return f"SystemExit({se.code})"
    except Exception as e:
        return f"error: {e}"

def collect_latest_clusters(out_dir: Path, since_mtime: float):
    newest = [p for p in out_dir.glob("*.csv") if p.stat().st_mtime >= since_mtime]
    newest.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    alt_csv = circ_csv = None
    alt_df = circ_df = None
    for p in newest:
        try:
            df = pd.read_csv(p, nrows=100)
        except Exception:
            continue
        if alt_csv is None and looks_like_alt(df):
            alt_csv = p; alt_df = pd.read_csv(p)
        elif circ_csv is None and looks_like_circ(df):
            circ_csv = p; circ_df = pd.read_csv(p)
        if alt_csv and circ_csv:
            break
    return alt_df, circ_df, alt_csv, circ_csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--igc-dir", default=None, help="Folder with IGCs (default ./igc next to this script)")
    ap.add_argument("--alt-script", default="altitude_gain_v3g.py")
    ap.add_argument("--circ-script", default="circles_clean_v2c.py")
    ap.add_argument("--outputs-dir", default="outputs")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    igc_dir = Path(args.igc_dir) if args.igc_dir else (script_dir / "igc")
    out_dir = Path(args.outputs_dir); out_dir.mkdir(exist_ok=True)
    clusters_dir = out_dir / "batch_clusters"; clusters_dir.mkdir(parents=True, exist_ok=True)

    alt_script = Path(args.alt_script).resolve()
    circ_script = Path(args.circ_script).resolve()

    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    log_path = out_dir / f"batch_run_{ts}.log"
    combined_path = out_dir / f"batch_matches_{ts}.csv"

    igc_files = sorted(igc_dir.glob("*.igc"))
    if not igc_files:
        print(f"[batch] No IGC files in {igc_dir}")
        return

    all_matches = []
    with open(log_path, "w", encoding="utf-8") as log:
        print(f"[batch] Start — {len(igc_files)} IGC in {igc_dir}", file=log)
        print(f"[batch] Thresholds: EPS_M={EPS_M}  MIN_OVL_FRAC={MIN_OVL_FRAC}  MAX_TIME_GAP_S={MAX_TIME_GAP_S}", file=log)
        for igc in igc_files:
            base = igc.stem
            print(f"[IGC] {base}", file=log)
            frontier = out_dir.stat().st_mtime

            alt_res = safe_run_detector(alt_script, igc)
            circ_res = safe_run_detector(circ_script, igc)

            alt_df_raw, circ_df_raw, alt_csv, circ_csv = collect_latest_clusters(out_dir, frontier)
            alt_df = normalize(alt_df_raw); circ_df = normalize(circ_df_raw)

            # Save copies for debugging
            if not alt_df.empty:
                (clusters_dir / f"altitude_clusters_{base}.csv").write_text(alt_df.to_csv(index=False))
            if not circ_df.empty:
                (clusters_dir / f"circle_clusters_{base}.csv").write_text(circ_df.to_csv(index=False))

            n_alt = len(alt_df) if not alt_df.empty else 0
            n_circ = len(circ_df) if not circ_df.empty else 0
            if n_alt and n_circ:
                M = match_per_flight(circ_df, alt_df)
                m = 0 if M.empty else len(M)
                if m:
                    M.insert(0, "igc_base", base)
                    all_matches.append(M)
                print(f"  -> {n_alt} altitude clusters | {n_circ} circle clusters | {m} matches | alt:{alt_res} circ:{circ_res}", file=log)
            else:
                print(f"  -> {n_alt} altitude clusters | {n_circ} circle clusters | 0 matches | alt:{alt_res} circ:{circ_res}", file=log)

    if not all_matches:
        print(f"[batch] Completed. No per‑flight matches. See log: {log_path}")
        return

    big = pd.concat(all_matches, ignore_index=True)
    big.to_csv(combined_path, index=False)
    print(f"[batch] Wrote combined matches: {combined_path}")
    print(f"[batch] Log: {log_path}")
    with pd.option_context('display.max_columns', None, 'display.width', 160):
        print(big.head(12))

if __name__ == "__main__":
    main()
