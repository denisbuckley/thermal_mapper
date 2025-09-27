
#!/usr/bin/env python3
# batch_run_v2c.py
# Run upstream pipeline across igc/ and ALSO run the matcher so each flight folder
# contains circles.csv, circle_clusters_enriched.csv, overlay_altitude_clusters.csv, matched_clusters.csv
#
# Final layout per flight:
#   outputs/batch_csv/<flight>/
#       circles.csv
#       circle_clusters_enriched.csv
#       overlay_altitude_clusters.csv
#       matched_clusters.csv
#
# Note: Upstream scripts write to ./outputs/*.csv relative to CWD.
# We run all four steps with CWD = outputs/batch_csv/<flight>, then move the files up one level.

import sys
import subprocess
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent
IGC_DIR = PROJECT_ROOT / "igc"
OUT_ROOT = PROJECT_ROOT / "outputs" / "batch_csv"

CIRCLES_SCRIPT = PROJECT_ROOT / "circles_from_brecords_v1d.py"
CLUSTERS_SCRIPT = PROJECT_ROOT / "circle_clusters_v1s.py"
ALT_SCRIPT = PROJECT_ROOT / "overlay_altitude_clusters_v1c.py"
MATCH_SCRIPT = PROJECT_ROOT / "match_clusters_v1.py"

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def run_in_cwd(py: str, workdir: Path, cmd: list) -> int:
    print(f"[RUN] (cwd={workdir}) {' '.join(map(str, cmd))}")
    return subprocess.run(cmd, cwd=str(workdir)).returncode

def move_up(workdir: Path, rel_src: Path, dst_name: str) -> bool:
    src = workdir / rel_src
    dst = workdir / dst_name
    if not src.exists():
        print(f"[WARN] expected file not found: {src}")
        return False
    try:
        if dst.exists():
            dst.unlink()
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        # remove empty parent dir (./outputs) if empty
        parent = src.parent
        try:
            if parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
        except Exception:
            pass
        print(f"[OK] wrote {dst}")
        return True
    except Exception as e:
        print(f"[WARN] move failed {src} -> {dst}: {e}")
        return False

def main():
    py = sys.executable

    # sanity
    for s in (CIRCLES_SCRIPT, CLUSTERS_SCRIPT, ALT_SCRIPT, MATCH_SCRIPT):
        if not s.exists():
            print(f"[ERROR] Missing script: {s}")
            sys.exit(2)

    if not IGC_DIR.exists():
        print(f"[ERROR] IGC folder not found: {IGC_DIR}")
        sys.exit(2)

    igc_files = sorted(IGC_DIR.glob("*.igc"))
    if not igc_files:
        print(f"[ERROR] No .igc files in: {IGC_DIR}")
        sys.exit(2)

    print(f"[OK] Found {len(igc_files)} IGC files in {IGC_DIR}")

    for igc_path in igc_files:
        flight = igc_path.stem
        workdir = OUT_ROOT / flight
        outputs_dir = workdir / "outputs"
        ensure_dir(outputs_dir)

        print(f"\n===== {flight} =====")

        # 1) circles → outputs/circles.csv
        rc = run_in_cwd(py, workdir, [py, str(CIRCLES_SCRIPT), str(igc_path)])
        if rc != 0:
            print(f"[WARN] circles_from_brecords failed (rc={rc}); skipping flight")
            continue
        if not (outputs_dir / "circles.csv").exists():
            print("[WARN] circles.csv not produced; skipping flight"); continue

        # 2) circle clustering (reads outputs/circles.csv) → outputs/circle_clusters_enriched.csv
        rc = run_in_cwd(py, workdir, [py, str(CLUSTERS_SCRIPT)])
        if rc != 0:
            print(f"[WARN] circle_clusters step failed (rc={rc}); skipping flight")
            continue
        if not (outputs_dir / "circle_clusters_enriched.csv").exists():
            print("[WARN] circle_clusters_enriched.csv not produced; skipping flight"); continue

        # 3) altitude overlay → outputs/overlay_altitude_clusters.csv
        #    (script expects positional 'igc' after your recent fix)
        rc = run_in_cwd(py, workdir, [py, str(ALT_SCRIPT), str(igc_path)])
        if rc != 0:
            print(f"[WARN] altitude_clusters step failed (rc={rc}); skipping flight")
            continue
        if not (outputs_dir / "overlay_altitude_clusters.csv").exists():
            print("[WARN] overlay_altitude_clusters.csv not produced; skipping flight"); continue

        # 4) match clusters (reads ./outputs/circle_clusters_enriched.csv + ./outputs/overlay_altitude_clusters.csv)
        rc = run_in_cwd(py, workdir, [py, str(MATCH_SCRIPT)])
        if rc != 0:
            print(f"[WARN] match_clusters step failed (rc={rc}); skipping flight")
            continue
        if not (outputs_dir / "matched_clusters.csv").exists():
            print("[WARN] matched_clusters.csv not produced; skipping flight"); continue

        # Move all four up one level per your desired layout
        move_up(workdir, Path("outputs")/"circles.csv", "circles.csv")
        move_up(workdir, Path("outputs")/"circle_clusters_enriched.csv", "circle_clusters_enriched.csv")
        move_up(workdir, Path("outputs")/"overlay_altitude_clusters.csv", "overlay_altitude_clusters.csv")
        move_up(workdir, Path("outputs")/"matched_clusters.csv", "matched_clusters.csv")

    print("\n[DONE] Batch CSVs under:", OUT_ROOT)

if __name__ == "__main__":
    main()
