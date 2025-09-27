
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_match_igc_v3d.py — per‑flight matching using **subprocess** (mirrors PyCharm CLI runs)

Why this version:
- Calls your detectors exactly like you do manually:
    python altitude_gain_v3g.py "<igc>"
    python circles_clean_v2c.py  "<igc>"
  (no runpy; no DEFAULT_IGC injection) — this avoids any differences in argparse / globals.
- Picks per‑IGC outputs by prefix + mtime since each subprocess start.
- Normalizes schema (center_lat/lon→lat/lon; total_gain_m/mean_mps→gain_m/avg_rate_mps).
- Logs picked filenames + row counts per IGC.
- Optional --loosen to widen gates fast.
"""

from __future__ import annotations

import argparse, subprocess, time, sys
from pathlib import Path
from datetime import datetime
import pandas as pd, numpy as np

# ---- Matching thresholds (overridable) ----
EPS_M = 2000.0
MIN_OVL_FRAC = 0.20
MAX_TIME_GAP_S = 15*60

# Optional tuning
try:
    from tuning_loader import load_tuning, override_globals
    _t = load_tuning("config/tuning_params.csv")
    override_globals(globals(), _t, allowed={"EPS_M","MIN_OVL_FRAC","MAX_TIME_GAP_S"})
except Exception:
    pass

def to_ts(s): return pd.to_datetime(s, utc=True, errors='coerce')

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1 = np.radians(lat1); p2 = np.radians(lon1)
    dphi = np.radians(lat2-lat1); dlmb = np.radians(lon2-lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlmb/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))

def looks_like_alt(df: pd.DataFrame) -> bool:
    cols = {c.lower() for c in df.columns}
    latlon = ('lat' in cols and 'lon' in cols) or ('center_lat' in cols and 'center_lon' in cols)
    climbish = any(k in cols for k in ('gain_m','total_gain_m','avg_rate_mps','mean_mps'))
    return latlon and climbish

def looks_like_circ(df: pd.DataFrame) -> bool:
    cols = {c.lower() for c in df.columns}
    latlon = ('lat' in cols and 'lon' in cols) or ('center_lat' in cols and 'center_lon' in cols)
    return latlon and ('n' in cols or 'mean_radius_m' in cols)

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=['cluster_id','lat','lon','start_time','end_time','n','gain_m','avg_rate_mps','mean_radius_m'])
    out = df.copy()
    ren = {}
    if 'center_lat' in out.columns: ren['center_lat'] = 'lat'
    if 'center_lon' in out.columns: ren['center_lon'] = 'lon'
    if 'total_gain_m' in out.columns: ren['total_gain_m'] = 'gain_m'
    if 'mean_mps' in out.columns: ren['mean_mps'] = 'avg_rate_mps'
    if 'thermal_id' in out.columns and 'cluster_id' not in out.columns: ren['thermal_id'] = 'cluster_id'
    out = out.rename(columns=ren)
    for col in ('cluster_id','lat','lon','start_time','end_time','n','gain_m','avg_rate_mps','mean_radius_m'):
        if col not in out.columns: out[col] = np.nan
    for col in ('start_time','end_time'):
        out[col] = to_ts(out[col])
    return out

def time_overlap_and_gap(a0, a1, b0, b1):
    latest_start = max(a0, b0)
    earliest_end = min(a1, b1)
    if pd.isna(latest_start) or pd.isna(earliest_end):
        return 0.0, 1e12
    ovl = (earliest_end - latest_start).total_seconds()
    if ovl > 0: return ovl, 0.0
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
            if d_m > EPS_M: continue
            ovl_s, gap_s = time_overlap_and_gap(cr['start_time'], cr['end_time'], ar['start_time'], ar['end_time'])
            c_dur = max(1.0, float((cr['end_time'] - cr['start_time']).total_seconds())) if (pd.notna(cr['start_time']) and pd.notna(cr['end_time'])) else 1.0
            a_dur = max(1.0, float((ar['end_time'] - ar['start_time']).total_seconds())) if (pd.notna(ar['start_time']) and pd.notna(ar['end_time'])) else 1.0
            shorter = min(c_dur, a_dur)
            ovl_f = max(0.0, ovl_s / shorter) if shorter > 0 else 0.0
            if ovl_s <= 0 and gap_s > MAX_TIME_GAP_S: continue
            if ovl_s > 0 and ovl_f < MIN_OVL_FRAC: continue
            score = (d_m/1000.0, -ovl_f)
            cand = (score, dict(
                d_km=round(d_m/1000.0, 3), ovl_s=round(ovl_s, 1), ovl_f=round(ovl_f, 3), gap_s=round(gap_s, 1),
                c_id=int(cr.get('cluster_id', -1)), c_lat=float(cr['lat']), c_lon=float(cr['lon']),
                c_n=None if pd.isna(cr.get('n')) else int(cr['n']),
                c_gain_m=None if pd.isna(cr.get('gain_m')) else float(cr['gain_m']),
                c_rate=None if pd.isna(cr.get('avg_rate_mps')) else float(cr['avg_rate_mps']),
                c_start=cr.get('start_time'), c_end=cr.get('end_time'),
                a_id=int(ar.get('cluster_id', -1)), a_lat=float(ar['lat']), a_lon=float(ar['lon']),
                a_n=None if pd.isna(ar.get('n')) else int(ar['n']),
                a_gain_m=None if pd.isna(ar.get('gain_m')) else float(ar['gain_m']),
                a_rate=None if pd.isna(ar.get('avg_rate_mps')) else float(ar['avg_rate_mps']),
                a_start=ar.get('start_time'), a_end=ar.get('end_time'),
            ))
            if best is None or cand[0] < best[0]:
                best = cand
        if best is not None:
            rows.append(best[1])
    return pd.DataFrame(rows)

def run_cli(script: Path, igc: Path, cwd: Path) -> str:
    try:
        r = subprocess.run([sys.executable, str(script), str(igc)], cwd=str(cwd), capture_output=True, text=True)
        if r.returncode != 0:
            return f"rc={r.returncode} stderr={r.stderr.strip()[:200]}"
        return "ok"
    except Exception as e:
        return f"error: {e}"

def pick_latest_since(folder: Path, since_ts: float, patterns: list[str]):
    cands = []
    for pat in patterns:
        cands.extend(folder.glob(pat))
    cands = [p for p in cands if p.suffix.lower()=='.csv' and p.stat().st_mtime >= since_ts]
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--igc-dir", default=None, help="Folder with IGCs (default ./igc next to this script)")
    ap.add_argument("--alt-script", default="altitude_gain_v3g.py")
    ap.add_argument("--circ-script", default="circles_clean_v2c.py")
    ap.add_argument("--outputs-dir", default="outputs")
    ap.add_argument("--loosen", action="store_true")
    args = ap.parse_args()

    global EPS_M, MIN_OVL_FRAC, MAX_TIME_GAP_S
    if args.loosen:
        EPS_M = max(EPS_M, 5000.0)
        MIN_OVL_FRAC = min(MIN_OVL_FRAC, 0.05)
        MAX_TIME_GAP_S = max(MAX_TIME_GAP_S, 1800.0)

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
            start_ts = time.time()

            alt_res = run_cli(alt_script, igc, cwd=script_dir)
            circ_res = run_cli(circ_script, igc, cwd=script_dir)

            alt_path = pick_latest_since(out_dir, start_ts, ["altitude_clusters_*.csv"])
            circ_path = pick_latest_since(out_dir, start_ts, ["circle_clusters_*.csv", "circles_clusters_*.csv"])

            alt_df = pd.read_csv(alt_path) if alt_path else pd.DataFrame()
            circ_df = pd.read_csv(circ_path) if circ_path else pd.DataFrame()

            if not alt_df.empty and not looks_like_alt(alt_df): alt_df = pd.DataFrame()
            if not circ_df.empty and not looks_like_circ(circ_df): circ_df = pd.DataFrame()

            alt_norm = normalize(alt_df) if not alt_df.empty else pd.DataFrame()
            circ_norm = normalize(circ_df) if not circ_df.empty else pd.DataFrame()

            if not alt_norm.empty:
                (clusters_dir / f"altitude_clusters_{base}.csv").write_text(alt_norm.to_csv(index=False))
            if not circ_norm.empty:
                (clusters_dir / f"circle_clusters_{base}.csv").write_text(circ_norm.to_csv(index=False))

            n_alt = len(alt_norm) if not alt_norm.empty else 0
            n_circ = len(circ_norm) if not circ_norm.empty else 0

            if n_alt and n_circ:
                M = match_per_flight(circ_norm, alt_norm)
                m = len(M) if not M.empty else 0
                if m:
                    M.insert(0, "igc_base", base)
                    all_matches.append(M)
                print(f"  -> alt_rows={n_alt} circ_rows={n_circ} matches={m} | alt:{alt_res} circ:{circ_res} | picked alt={alt_path.name if alt_path else 'NA'} circ={circ_path.name if circ_path else 'NA'}", file=log)
            else:
                print(f"  -> alt_rows={n_alt} circ_rows={n_circ} matches=0 | alt:{alt_res} circ:{circ_res} | picked alt={alt_path.name if alt_path else 'NA'} circ={circ_path.name if circ_path else 'NA'}", file=log)

    if not all_matches:
        print(f"[batch] Completed. No per‑flight matches. See log: {log_path}")
        return

    big = pd.concat(all_matches, ignore_index=True)
    big.to_csv(combined_path, index=False)
    print(f("[batch] Wrote combined matches: {combined_path}"))
    print(f"[batch] Log: {log_path}")
    with pd.option_context('display.max_columns', None, 'display.width', 160):
        print(big.head(12))

if __name__ == "__main__":
    main()
