
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_match_igc_v1c.py — force-run detectors via runpy (DEFAULT_IGC only),
then collect whatever cluster CSVs they emitted, rename into a tidy folder,
and finally match circle↔altitude clusters per IGC. Single log file.

Changes vs v1:
- **No CLI execution**. We always use runpy with DEFAULT_IGC to avoid detector-CLI drift.
- **Filename-agnostic collection**. We detect *newly written* CSVs after each detector run,
  look for altitude/circle cluster tables by content/columns, and then rename them to:
    outputs/batch_clusters/altitude_clusters_<base>.csv
    outputs/batch_clusters/circle_clusters_<base>.csv

Outputs:
  - outputs/batch_clusters/altitude_clusters_<base>.csv
  - outputs/batch_clusters/circle_clusters_<base>.csv
  - outputs/batch_matched_<ts>.csv
  - outputs/batch_run_<ts>.log
"""

from __future__ import annotations

import argparse, runpy, sys, os, re, glob
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# ---- Matching thresholds (overridable) ----
EPS_M = 2000.0           # spatial tolerance (meters) for cluster centers
MIN_OVL_FRAC = 0.20      # min overlap fraction (on shorter duration) when intervals overlap
MAX_TIME_GAP_S = 15*60   # max allowed gap (s) when no overlap

# Optional tuning override
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

def time_overlap_and_gap(a0, a1, b0, b1):
    latest_start = max(a0, b0)
    earliest_end = min(a1, b1)
    ovl = (earliest_end - latest_start).total_seconds()
    if ovl > 0:
        return ovl, 0.0
    gap = (a0 - b1).total_seconds() if a0 > b1 else (b0 - a1).total_seconds()
    return 0.0, gap

def normalize_enriched(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ren = {}
    if 'thermal_id' in out.columns and 'cluster_id' not in out.columns:
        ren['thermal_id'] = 'cluster_id'
    if 'avg_rate' in out.columns and 'avg_rate_mps' not in out.columns:
        ren['avg_rate'] = 'avg_rate_mps'
    out = out.rename(columns=ren)
    # Ensure essential columns exist
    for col in ('lat','lon','start_time','end_time','n','gain_m','avg_rate_mps','cluster_id'):
        if col not in out.columns: out[col] = np.nan
    # Coerce times
    for col in ('start_time','end_time'):
        out[col] = to_ts(out[col])
    return out

def looks_like_altitude_clusters(df: pd.DataFrame) -> bool:
    cols = set(c.lower() for c in df.columns)
    return ('gain_m' in cols or 'avg_rate_mps' in cols) and ('lat' in cols and 'lon' in cols)

def looks_like_circle_clusters(df: pd.DataFrame) -> bool:
    cols = set(c.lower() for c in df.columns)
    # circle clusters usually have 'n' (number of circles) even if gain is missing
    return ('n' in cols) and ('lat' in cols and 'lon' in cols)

def newest_csvs_since(folder: Path, since_mtime: float) -> list[Path]:
    return sorted([p for p in folder.glob("*.csv") if p.stat().st_mtime >= since_mtime],
                  key=lambda p: p.stat().st_mtime, reverse=True)

def collect_cluster_csv(folder: Path, since: float, want_alt: bool) -> Path|None:
    # scan newest first, read small sample to determine type
    for p in newest_csvs_since(folder, since):
        try:
            df = pd.read_csv(p, nrows=100)
        except Exception:
            continue
        if want_alt and looks_like_altitude_clusters(df):
            return p
        if not want_alt and looks_like_circle_clusters(df):
            return p
    return None

def match_one_igc(circ_df: pd.DataFrame, alt_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, cr in circ_df.iterrows():
        best = None
        for _, ar in alt_df.iterrows():
            if pd.isna(cr['lat']) or pd.isna(cr['lon']) or pd.isna(ar['lat']) or pd.isna(ar['lon']):
                continue
            d_m = float(haversine_m(cr['lat'], cr['lon'], ar['lat'], ar['lon']))
            if d_m > EPS_M:
                continue
            ovl_s, gap_s = time_overlap_and_gap(cr['start_time'], cr['end_time'],
                                                ar['start_time'], ar['end_time'])
            c_dur = max(1.0, float((cr['end_time'] - cr['start_time']).total_seconds()))
            a_dur = max(1.0, float((ar['end_time'] - ar['start_time']).total_seconds()))
            shorter = min(c_dur, a_dur)
            ovl_frac = max(0.0, ovl_s / shorter) if shorter > 0 else 0.0
            if ovl_s <= 0 and gap_s > MAX_TIME_GAP_S:
                continue
            if ovl_s > 0 and ovl_frac < MIN_OVL_FRAC:
                continue
            score = (d_m/1000.0, -ovl_frac)
            cand = (score, dict(
                c_id=int(cr.get('cluster_id', -1)),
                a_id=int(ar.get('cluster_id', -1)),
                d_km=round(d_m/1000.0, 3),
                ovl_s=round(ovl_s, 1),
                ovl_f=round(ovl_frac, 3),
                gap_s=round(gap_s, 1),
                c_gain_m=None if pd.isna(cr.get('gain_m')) else float(cr['gain_m']),
                c_rate=None if pd.isna(cr.get('avg_rate_mps')) else float(cr['avg_rate_mps']),
                c_n=None if pd.isna(cr.get('n')) else int(cr['n']),
                c_lat=float(cr['lat']), c_lon=float(cr['lon']),
                c_start=cr.get('start_time'), c_end=cr.get('end_time'),
                a_gain_m=None if pd.isna(ar.get('gain_m')) else float(ar['gain_m']),
                a_rate=None if pd.isna(ar.get('avg_rate_mps')) else float(ar['avg_rate_mps']),
                a_n=None if pd.isna(ar.get('n')) else int(ar['n']),
                a_lat=float(ar['lat']), a_lon=float(ar['lon']),
                a_start=ar.get('start_time'), a_end=ar.get('end_time'),
            ))
            if best is None or cand[0] < best[0]:
                best = cand
        if best is not None:
            rows.append(best[1])
    return pd.DataFrame(rows)

def safe_runpy(script: Path, igc_file: Path) -> str:
    # Run detector by injecting DEFAULT_IGC and executing as __main__
    try:
        runpy.run_path(str(script), init_globals={"DEFAULT_IGC": str(igc_file), "__name__": "__main__"})
        return "ok"
    except SystemExit as se:
        return f"SystemExit({se.code})"
    except Exception as e:
        return f"error: {e}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--igc_subset-dir", default=None, help="Folder with IGCs (default: ./igc_subset next to this script)")
    ap.add_argument("--alt-script", default="altitude_gain_v3g.py", help="Altitude detector script path")
    ap.add_argument("--circ-script", default="circles_clean_v2c.py", help="Circles detector script path")
    ap.add_argument("--outputs-dir", default="outputs", help="Outputs folder")
    ap.add_argument("--force", action="store_true", help="Force re-run detectors even if prior outputs exist")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    igc_dir = Path(args.igc_dir) if args.igc_dir else (script_dir / "igc_subset")
    out_dir = Path(args.outputs_dir); out_dir.mkdir(exist_ok=True)
    clusters_dir = out_dir / "batch_clusters"; clusters_dir.mkdir(parents=True, exist_ok=True)

    alt_script = Path(args.alt_script).resolve()
    circ_script = Path(args.circ_script).resolve()

    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    log_path = out_dir / f"batch_run_{ts}.log"
    csv_path = out_dir / f"batch_matched_{ts}.csv"

    igc_files = sorted(igc_dir.glob("*.igc_subset"))
    if not igc_files:
        print(f"[batch] No IGC files in {igc_dir}")
        return

    with open(log_path, "w", encoding="utf-8") as log:
        print(f"[batch] Start — IGC dir: {igc_dir} (found {len(igc_files)})", file=log)
        print(f"[batch] Detectors (runpy only): ALT={alt_script}  CIRC={circ_script}", file=log)
        print(f"[batch] Thresholds: EPS_M={EPS_M}m  MIN_OVL_FRAC={MIN_OVL_FRAC}  MAX_TIME_GAP_S={MAX_TIME_GAP_S}", file=log)

        all_rows = []
        for igc in igc_files:
            base = igc.stem
            print(f"[IGC] {base}", file=log)

            # Record mtime frontier BEFORE running detectors
            frontier = out_dir.stat().st_mtime

            # Always run both detectors in runpy mode
            alt_res = safe_runpy(alt_script, igc)
            circ_res = safe_runpy(circ_script, igc)

            # Collect the newest CSVs written since frontier
            alt_csv = collect_cluster_csv(out_dir, frontier, want_alt=True)
            circ_csv = collect_cluster_csv(out_dir, frontier, want_alt=False)

            if alt_csv:
                alt_df = normalize_enriched(pd.read_csv(alt_csv))
                alt_out = clusters_dir / f"altitude_clusters_{base}.csv"
                alt_df.to_csv(alt_out, index=False)
                alt_n = len(alt_df)
            else:
                alt_df = pd.DataFrame()
                alt_n = 0

            if circ_csv:
                circ_df = normalize_enriched(pd.read_csv(circ_csv))
                circ_out = clusters_dir / f"circle_clusters_{base}.csv"
                circ_df.to_csv(circ_out, index=False)
                circ_n = len(circ_df)
            else:
                circ_df = pd.DataFrame()
                circ_n = 0

            # Match if possible
            matches = 0
            if not alt_df.empty and not circ_df.empty:
                M = match_one_igc(circ_df, alt_df)
                matches = 0 if M.empty else len(M)
                if matches:
                    M.insert(0, "igc_base", base)
                    all_rows.append(M)

            print(f"  -> {alt_n} altitude clusters | {circ_n} circle clusters | {matches} matches | alt:{alt_res} circ:{circ_res}", file=log)

    if not all_rows:
        print(f"[batch] Completed. No matches across flights. See log: {log_path}")
        return

    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"[batch] Wrote {csv_path}")
    print(f"[batch] Log:   {log_path}")
    with pd.option_context('display.max_columns', None, 'display.width', 160):
        print(df.head(12))

if __name__ == "__main__":
    main()
