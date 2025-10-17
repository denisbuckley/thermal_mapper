#!/usr/bin/env python3
import sys, shutil, subprocess, shlex
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "outputs" / "batch_csv"

def safe_basename(p: Path) -> str:
    name = p.stem.strip()
    # keep spaces; just strip bad path chars
    for ch in '\\/:*?"<>|':
        name = name.replace(ch, " ")
    return " ".join(name.split())  # collapse whitespace

def run(cmd, cwd: Path, logf):
    logf.write(f"[RUN] (cwd={cwd}) {cmd}\n"); logf.flush()
    # Prefer list form for subprocess; accept strings for readability here
    if isinstance(cmd, str):
        cmd_list = shlex.split(cmd)
    else:
        cmd_list = cmd
    proc = subprocess.run(cmd_list, cwd=str(cwd), text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    logf.write(proc.stdout)
    logf.write(f"[RC] {proc.returncode}\n\n"); logf.flush()
    if proc.returncode != 0:
        raise RuntimeError(f"Step failed (rc={proc.returncode}) running: {cmd}")

def main():
    # -------- get IGC path (argv first, fallback to prompt) --------
    if len(sys.argv) > 1:
        igc_path = Path(sys.argv[1])
    else:
        default_igc = Path("igc_subset/2020-11-08 Lumpy Paterson 108645.igc")
        try:
            user_in = input(f"Enter path to IGC file [default: {default_igc}]: ").strip()
        except EOFError:
            user_in = ""
        igc_path = Path(user_in) if user_in else default_igc

    if not igc_path.exists():
        print(f"[ERROR] IGC not found: {igc_path}")
        sys.exit(2)

    # -------- prepare run directory --------
    flight = safe_basename(igc_path)
    run_dir = OUT_ROOT / flight
    run_dir.mkdir(parents=True, exist_ok=True)

    # copy source IGC next to outputs for provenance
    igc_local = run_dir / igc_path.name
    if not igc_local.exists():
        shutil.copy2(igc_path, igc_local)

    log_path = run_dir / "pipeline_debug.log"
    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write(f"===== pipeline_v3 start {datetime.now().isoformat()} =====\n")
        logf.write(f"IGC: {igc_path}\nRUN_DIR: {run_dir}\n\n")

        py = sys.executable

        # 1) circles_from_brecords_v1d.py -> circles.csv (in CWD)
        run([py, str(ROOT / "circles_from_brecords_v1d.py"),
             str(igc_local), "--out", "circles.csv"], cwd=run_dir, logf=logf)

        # 2) circle_clusters_v1s.py -> circle_clusters_enriched.csv
        run([py, str(ROOT / "circle_clusters_v1s.py"),
             "circles.csv", "--out", "circle_clusters_enriched.csv"], cwd=run_dir, logf=logf)

        # 3) altitude_clusters_v1.py -> altitude_clusters.csv
        run([py, str(ROOT / "altitude_clusters_v1.py"),
             str(igc_local), "--out", "altitude_clusters.csv"], cwd=run_dir, logf=logf)

        # 4) match_clusters_v1.py -> matched_clusters.csv (+ .json)
        run([py, str(ROOT / "match_clusters_v1.py"),
             "circle_clusters_enriched.csv", "altitude_clusters.csv",
             "--out", "matched_clusters.csv"], cwd=run_dir, logf=logf)

        logf.write(f"===== pipeline_v3 done {datetime.now().isoformat()} =====\n")

    # -------- summary to console --------
    print("[DONE] Artifacts in:", run_dir)
    for fname in [
        "circles.csv",
        "circle_clusters_enriched.csv",
        "altitude_clusters.csv",
        "matched_clusters.csv",
        "matched_clusters.json",
        "pipeline_debug.log",
    ]:
        p = run_dir / fname
        print(("[OK] " if p.exists() else "[MISS] "), p)

if __name__ == "__main__":
    main()