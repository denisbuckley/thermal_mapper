
#!/usr/bin/env python3
"""
batch_pipeline_igc_v1.py

End-to-end batch runner:
  For every *.igc_subset in ./igc_subset (by default), it will:
    1) Run the altitude detector script to produce altitude_clusters_enriched CSVs
    2) Run the circles detector script to produce circle_clusters_enriched CSVs
    3) Match circle↔altitude clusters (strict) and append to a master CSV
    4) Write a verbose run log

It does **not** re-implement the detectors; instead it executes your existing
scripts and relies on them to write enriched CSVs. This avoids code drift.

ASSUMPTIONS
- Your detector scripts, when executed, write per-IGC enriched CSVs whose
  filenames contain the IGC basename (e.g., "2020-11-08 Lumpy Paterson 108645").
- If your detector scripts expect a DEFAULT_IGC variable rather than a CLI arg,
  we set that DEFAULT_IGC at runtime via runpy (init_globals) and invoke as __main__.

DEFAULTS
- igc_dir: ./igc_subset   (next to this script)
- alt_script: ./altitude_gain_v3g.py
- circ_script: ./circles_clean_v2c.py
- outputs_dir: ./outputs

TUNING
- Matching respects tuning_loader (if present): EPS_M, MIN_OVL_FRAC, MAX_TIME_GAP_S

USAGE
  python batch_pipeline_igc_v1.py
  python batch_pipeline_igc_v1.py --igc_subset-dir data/igc_subset --alt-script my_alt.py --circ-script my_circ.py --force

RESULTS
  - outputs/batch_pipeline_log_<ts>.txt
  - outputs/batch_matched_<ts>.csv
"""

import os, re, glob, argparse, subprocess, sys, runpy
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# ---- Matching thresholds (overridable) ----
EPS_M = 2000.0
MIN_OVL_FRAC = 0.20
MAX_TIME_GAP_S = 15*60

try:
    from tuning_loader import load_tuning, override_globals
    _t = load_tuning("config/tuning_params.csv")
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
    for col in ('start_time','end_time'):
        if col in out.columns: out[col] = to_ts(out[col])
    for col in ('n','gain_m','avg_rate_mps','lat','lon'):
        if col not in out.columns: out[col] = np.nan
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

def run_detector_script(script_path: Path, igc_path: Path, log, label:str):
    """
    Execute a detector script so it writes enriched CSVs for this IGC.
    Supports two modes:
      - If the script accepts CLI arg --igc_subset, we pass it.
      - Otherwise, we set DEFAULT_IGC via runpy and run as __main__.
    """
    script_path = script_path.resolve()
    igc_path = igc_path.resolve()
    if not script_path.exists():
        print(f"[{label}] script not found: {script_path}", file=log); return False
    # Try CLI first
    try:
        r = subprocess.run(
            [sys.executable, str(script_path), "--igc_subset", str(igc_path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
        )
        print(f"[{label}] CLI run rc={r.returncode}", file=log)
        if r.stdout: print(r.stdout.strip(), file=log)
        if r.stderr: print(r.stderr.strip(), file=log)
        if r.returncode == 0:
            return True
    except Exception as e:
        print(f"[{label}] CLI exec failed: {e}", file=log)
    # Fallback: runpy with DEFAULT_IGC injected
    try:
        print(f"[{label}] Fallback to runpy with DEFAULT_IGC", file=log)
        runpy.run_path(str(script_path), init_globals={
            "DEFAULT_IGC": str(igc_path),
            "__name__": "__main__"
        })
        return True
    except SystemExit as se:
        print(f"[{label}] runpy SystemExit: {se}", file=log); return int(se.code)==0
    except Exception as e:
        print(f"[{label}] runpy failed: {e}", file=log); return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--igc_subset-dir", default=None, help="Folder with IGCs (default: ./igc_subset next to this script)")
    ap.add_argument("--alt-script", default="altitude_gain_v3g.py", help="Altitude detector script path")
    ap.add_argument("--circ-script", default="circles_clean_v2c.py", help="Circles detector script path")
    ap.add_argument("--outputs-dir", default="outputs", help="Outputs folder")
    ap.add_argument("--force", action="store_true", help="Force re-run detectors even if enriched CSVs exist")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    igc_dir = Path(args.igc_dir) if args.igc_dir else (script_dir / "igc_subset")
    out_dir = Path(args.outputs_dir); out_dir.mkdir(exist_ok=True)

    alt_script = Path(args.alt_script)
    circ_script = Path(args.circ_script)

    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    log_path = out_dir / f"batch_pipeline_log_{ts}.txt"
    csv_path = out_dir / f"batch_matched_{ts}.csv"

    igc_files = sorted(igc_dir.glob("*.igc_subset"))
    if not igc_files:
        print(f"[batch] No IGC files in {igc_dir}"); return

    with open(log_path, "w", encoding="utf-8") as log:
        print(f"[batch] Start. IGC dir: {igc_dir} (found {len(igc_files)})", file=log)
        print(f"[batch] Alt script:  {alt_script}", file=log)
        print(f"[batch] Circ script: {circ_script}", file=log)
        print(f"[batch] EPS_M={EPS_M}m, MIN_OVL_FRAC={MIN_OVL_FRAC}, MAX_TIME_GAP_S={MAX_TIME_GAP_S}", file=log)

        all_rows = []
        for igc in igc_files:
            base = igc.stem
            print(f"\n[IGC] {base}", file=log)

            # Pre-check existing enriched files if not forced
            c_csv = latest_with_base(str(out_dir / "*_circle_clusters_enriched_*.csv"), base)
            a_csv = latest_with_base(str(out_dir / "*_altitude_clusters_enriched_*.csv"), base)

            if not args.force and c_csv and a_csv:
                print(f"  -> reuse enriched (circle & altitude present)", file=log)
            else:
                # Run altitude detector
                ok_alt = run_detector_script(alt_script, igc, log, "ALT")
                # Run circles detector
                ok_cir = run_detector_script(circ_script, igc, log, "CIRC")

                # Find enriched outputs (newest for this base)
                a_csv = latest_with_base(str(out_dir / "*_altitude_clusters_enriched_*.csv"), base)
                c_csv = latest_with_base(str(out_dir / "*_circle_clusters_enriched_*.csv"), base)

                print(f"  -> enriched: ALT={'ok' if a_csv else 'missing'} | CIRC={'ok' if c_csv else 'missing'}", file=log)
                if not a_csv or not c_csv:
                    print(f"  -> skip matching (missing enriched)", file=log)
                    continue

            circ = normalize_enriched(pd.read_csv(c_csv))
            alti = normalize_enriched(pd.read_csv(a_csv))
            if circ.empty or alti.empty:
                print("  -> skip: empty enriched CSV(s)", file=log); continue

            M = match_one_igc(circ, alti)
            if M.empty:
                print(f"  -> no matches under thresholds", file=log); continue
            M.insert(0, "igc_base", base)
            all_rows.append(M)
            print(f"  -> matched {len(M)} pairs", file=log)

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
