
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_match_igc_v1.py  — single-log, always-export clusters, then match

For each *.igc in --igc-dir (default: ./igc next to this script):
  1) Run altitude & circles detector scripts (CLI `--igc` if available; else runpy with DEFAULT_IGC).
  2) Locate the latest enriched CSVs for that IGC.
  3) Export per-IGC cluster CSVs into outputs/batch_clusters/:
        altitude_clusters_<basename>.csv
        circle_clusters_<basename>.csv
  4) Match circle↔altitude clusters (strict) and aggregate to one CSV.
  5) Append ONE line per IGC to a SINGLE run log file (no per-IGC logs).

Outputs:
  - outputs/batch_clusters/altitude_clusters_<base>.csv
  - outputs/batch_clusters/circle_clusters_<base>.csv
  - outputs/batch_matched_<ts>.csv
  - outputs/batch_run_<ts>.log

Honors thresholds from config/tuning_params.csv if tuning_loader.py exists:
  - EPS_M, MIN_OVL_FRAC, MAX_TIME_GAP_S
"""

from __future__ import annotations

import os, re, glob, argparse, subprocess, sys, runpy
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# ---- Matching thresholds (overridable) ----
EPS_M = 2000.0           # spatial tolerance (meters) for cluster centers
MIN_OVL_FRAC = 0.20      # min overlap fraction (on shorter duration) when intervals overlap
MAX_TIME_GAP_S = 15*60   # max allowed gap (s) when no overlap

# Optional tuning override
try:
    from tuning_loader import load_tuning, override_globals
    _t = load_tuning("../config/tuning_params.csv")
    override_globals(globals(), _t, allowed={"EPS_M","MIN_OVL_FRAC","MAX_TIME_GAP_S"})
except Exception:
    pass

def norm(s:str)->str:
    return re.sub(r'[^a-z0-9]', '', s.lower())

def to_ts(s):
    return pd.to_datetime(s, utc=True, errors='coerce')

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1 = np.radians(lat1); p2 = np.radians(lat2)
    dphi = np.radians(lat2-lat1); dlmb = np.radians(lon2-lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlmb/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))

def latest_with_base(glob_patt: str, base_token: str):
    files = [p for p in glob.glob(glob_patt) if norm(base_token) in norm(os.path.basename(p))]
    if not files: return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]

def normalize_enriched(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ren = {}
    if 'thermal_id' in out.columns and 'cluster_id' not in out.columns:
        ren['thermal_id'] = 'cluster_id'
    if 'avg_rate' in out.columns and 'avg_rate_mps' not in out.columns:
        ren['avg_rate'] = 'avg_rate_mps'
    out = out.rename(columns=ren)
    # Ensure essential columns exist
    for col in ('lat','lon','start_time','end_time','n','gain_m','avg_rate_mps'):
        if col not in out.columns: out[col] = np.nan
    # Coerce times
    for col in ('start_time','end_time'):
        out[col] = to_ts(out[col])
    return out

def time_overlap_and_gap(a0, a1, b0, b1):
    latest_start = max(a0, b0)
    earliest_end = min(a1, b1)
    ovl = (earliest_end - latest_start).total_seconds()
    if ovl > 0:
        return ovl, 0.0
    gap = (a0 - b1).total_seconds() if a0 > b1 else (b0 - a1).total_seconds()
    return 0.0, gap

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
                # circle side
                c_gain_m=None if pd.isna(cr.get('gain_m')) else float(cr['gain_m']),
                c_rate=None if pd.isna(cr.get('avg_rate_mps')) else float(cr['avg_rate_mps']),
                c_n=None if pd.isna(cr.get('n')) else int(cr['n']),
                c_lat=float(cr['lat']), c_lon=float(cr['lon']),
                c_start=cr.get('start_time'), c_end=cr.get('end_time'),
                # altitude side
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

def run_detector_script(script_path: Path, igc_path: Path, label:str) -> tuple[bool,str]:
    """
    Execute a detector script so it writes enriched CSVs for this IGC.
    Tries CLI first:   python script.py --igc <file>
    Fallback runpy:    inject DEFAULT_IGC and run as __main__
    Returns (ok, combined_stdout_stderr)
    """
    script_path = script_path.resolve()
    igc_path = igc_path.resolve()
    out = ""
    if not script_path.exists():
        return False, f"[{label}] script not found: {script_path}"
    # Try CLI first
    try:
        r = subprocess.run([sys.executable, str(script_path), "--igc", str(igc_path)],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        out += f"[{label}] CLI rc={r.returncode}\n{r.stdout}{r.stderr}"
        if r.returncode == 0:
            return True, out
    except Exception as e:
        out += f"[{label}] CLI exec failed: {e}\n"
    # Fallback: runpy
    try:
        out += f"[{label}] Fallback runpy DEFAULT_IGC={igc_path}\n"
        runpy.run_path(str(script_path), init_globals={"DEFAULT_IGC": str(igc_path), "__name__": "__main__"})
        return True, out
    except SystemExit as se:
        out += f"[{label}] runpy SystemExit: {se}\n";  return (int(se.code)==0), out
    except Exception as e:
        out += f"[{label}] runpy failed: {e}\n";       return False, out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--igc-dir", default=None, help="Folder with IGCs (default: ./igc next to this script)")
    ap.add_argument("--alt-script", default="altitude_gain_v3g.py", help="Altitude detector script path")
    ap.add_argument("--circ-script", default="circles_clean_v2c.py", help="Circles detector script path")
    ap.add_argument("--outputs-dir", default="outputs", help="Outputs folder")
    ap.add_argument("--force", action="store_true", help="Force re-run detectors even if enriched CSVs exist")
    ap.add_argument("--verbose", action="store_true", help="Verbose: include detector stdout/stderr in the single log")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    igc_dir = Path(args.igc_dir) if args.igc_dir else (script_dir / "igc")
    out_dir = Path(args.outputs_dir); out_dir.mkdir(exist_ok=True)
    clusters_dir = out_dir / "batch_clusters"; clusters_dir.mkdir(parents=True, exist_ok=True)

    alt_script = Path(args.alt_script)
    circ_script = Path(args.circ_script)

    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    log_path = out_dir / f"batch_run_{ts}.log"
    csv_path = out_dir / f"batch_matched_{ts}.csv"

    igc_files = sorted(igc_dir.glob("*.igc"))
    if not igc_files:
        print(f"[batch] No IGC files in {igc_dir}")
        return

    with open(log_path, "w", encoding="utf-8") as log:
        print(f"[batch] Start — IGC dir: {igc_dir} (found {len(igc_files)})", file=log)
        print(f"[batch] Detectors: ALT={alt_script}  CIRC={circ_script}", file=log)
        print(f"[batch] Thresholds: EPS_M={EPS_M}m  MIN_OVL_FRAC={MIN_OVL_FRAC}  MAX_TIME_GAP_S={MAX_TIME_GAP_S}", file=log)

        all_rows = []
        for igc in igc_files:
            base = igc.stem
            print(f"[IGC] {base}", file=log)

            # Check existing enriched
            c_csv = latest_with_base(str(out_dir / "*_circle_clusters_enriched_*.csv"), base)
            a_csv = latest_with_base(str(out_dir / "*_altitude_clusters_enriched_*.csv"), base)

            ran_alt_out = ran_circ_out = ""
            if args.force or not (c_csv and a_csv):
                ok_alt, ran_alt_out = run_detector_script(alt_script, igc, "ALT")
                ok_cir, ran_circ_out = run_detector_script(circ_script, igc, "CIRC")
                if args.verbose:
                    print(ran_alt_out, file=log)
                    print(ran_circ_out, file=log)

                a_csv = latest_with_base(str(out_dir / "*_altitude_clusters_enriched_*.csv"), base)
                c_csv = latest_with_base(str(out_dir / "*_circle_clusters_enriched_*.csv"), base)

            if not a_csv or not c_csv:
                print(f"  -> enriched missing (ALT={'ok' if a_csv else 'missing'}, CIRC={'ok' if c_csv else 'missing'}) — skip match", file=log)
                continue

            alt_df = normalize_enriched(pd.read_csv(a_csv))
            circ_df = normalize_enriched(pd.read_csv(c_csv))

            # Always export per-IGC clusters to a tidy subfolder
            alt_out = clusters_dir / f"altitude_clusters_{base}.csv"
            circ_out = clusters_dir / f"circle_clusters_{base}.csv"
            alt_df.to_csv(alt_out, index=False)
            circ_df.to_csv(circ_out, index=False)

            # Match and summarize
            M = match_one_igc(circ_df, alt_df)
            matches = 0 if M is None or M.empty else len(M)
            if matches:
                M.insert(0, "igc_base", base)
                all_rows.append(M)

            # One-line summary per IGC in the single log
            n_alt = 0 if alt_df is None or alt_df.empty else len(alt_df)
            n_circ = 0 if circ_df is None or circ_df.empty else len(circ_df)
            print(f"  -> {n_alt} altitude clusters | {n_circ} circle clusters | {matches} matches", file=log)

    # Final aggregation
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
