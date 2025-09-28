#!/usr/bin/env python3
"""
Batch runner for pipeline_v3.py

- Scans an IGC directory (default: ./igc) for *.igc
- For each flight, runs pipeline_v3.py into outputs/batch_csv/<flight_basename>/
- Skips flights that already have matched_clusters.csv unless --force
- Writes a master log at outputs/batch_csv/_batch_debug.log
"""

import sys, shlex, subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
PIPE = ROOT / "pipeline_v3.py"
OUT_ROOT = ROOT / "outputs" / "batch_csv"

def safe_basename(p: Path) -> str:
    name = p.stem.strip()
    for ch in '\\/:*?"<>|':
        name = name.replace(ch, " ")
    return " ".join(name.split())

def run_cmd(cmd, cwd: Path, logf):
    if isinstance(cmd, str):
        cmd_list = shlex.split(cmd)
    else:
        cmd_list = cmd
    logf.write(f"[RUN] (cwd={cwd}) {' '.join(shlex.quote(x) for x in cmd_list)}\n"); logf.flush()
    proc = subprocess.run(cmd_list, cwd=str(cwd), text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    logf.write(proc.stdout)
    logf.write(f"[RC] {proc.returncode}\n\n"); logf.flush()
    return proc.returncode

def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--igc-dir", default="igc", help="Folder containing .igc files (default: igc)")
    ap.add_argument("--out-root", default=str(OUT_ROOT), help="Batch outputs root (default: outputs/batch_csv)")
    ap.add_argument("--resume-from", default=None, help="Substring; skip until a filename contains this")
    ap.add_argument("--limit", type=int, default=0, help="Stop after N flights (0 = no limit)")
    ap.add_argument("--force", action="store_true", help="Re-run even if matched_clusters.csv exists")
    ap.add_argument("--dry-run", action="store_true", help="List what would run, don’t execute")
    args = ap.parse_args()

    igc_dir = Path(args.igc_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if not igc_dir.exists():
        print(f"[ERROR] IGC dir not found: {igc_dir}")
        return 2

    flights = sorted(igc_dir.glob("*.igc"))
    if not flights:
        print(f"[INFO] No .igc files in {igc_dir}")
        return 0

    batch_log = out_root / "_batch_debug.log"
    with open(batch_log, "a", encoding="utf-8") as logf:
        logf.write(f"===== batch_run_v3 start {datetime.now().isoformat()} =====\n")
        logf.write(f"IGC_DIR: {igc_dir}\nOUT_ROOT: {out_root}\n\n")

        started = args.resume_from is None
        ran = 0
        errs = 0

        for igc in flights:
            fn = igc.name
            if not started:
                if args.resume_from and args.resume_from in fn:
                    started = True
                else:
                    logf.write(f"[SKIP-resume] {fn}\n")
                    continue

            flight = safe_basename(igc)
            run_dir = out_root / flight
            done_csv = run_dir / "matched_clusters.csv"

            if done_csv.exists() and not args.force:
                logf.write(f"[SKIP] already has matched_clusters.csv → {run_dir}\n")
                continue

            cmd = [sys.executable, str(PIPE), str(igc)]
            if args.dry_run:
                print(f"[DRY] would run: {' '.join(shlex.quote(x) for x in cmd)}")
                continue

            rc = run_cmd(cmd, cwd=ROOT, logf=logf)
            if rc != 0:
                errs += 1
                logf.write(f"[WARN] pipeline failed for {fn} (rc={rc})\n")
            else:
                ran += 1
                logf.write(f"[OK] pipeline completed for {fn}\n")

            if args.limit and ran >= args.limit:
                logf.write(f"[STOP] reached limit={args.limit}\n")
                break

        logf.write(f"\n===== batch_run_v3 done {datetime.now().isoformat()} =====\n")
        logf.write(f"Summary: total={len(flights)} ran={ran} errs={errs} skipped={len(flights)-ran-errs}\n\n")

    print("[DONE] batch complete")
    print(" Log:", batch_log)
    return 0

if __name__ == "__main__":
    sys.exit(main())