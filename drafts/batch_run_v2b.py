
#!/usr/bin/env python3
# batch_run_v2b.py
# Fix: keep circles at outputs/circles.csv until clustering is done,
# then move both circles.csv and circle_clusters_enriched.csv up one level.
# Altitude step is run last; its output is then moved up as well.
#
# Layout produced per IGC:
#   outputs/batch_csv/<flight>/
#       circles.csv
#       circle_clusters_enriched.csv
#       overlay_altitude_clusters.csv

import sys
import subprocess
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent
IGC_DIR = PROJECT_ROOT / "igc_subset"
OUT_ROOT = PROJECT_ROOT / "outputs" / "batch_csv"

CIRCLES_SCRIPT = PROJECT_ROOT / "circles_from_brecords_v1d.py"
CLUSTERS_SCRIPT = PROJECT_ROOT / "circle_clusters_v1s.py"
ALT_SCRIPT = PROJECT_ROOT / "overlay_altitude_clusters_v1c.py"

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def run_in_cwd(py: str, workdir: Path, cmd: list) -> int:
    print(f"[RUN] (cwd={workdir}) {' '.join(map(str, cmd))}")
    proc = subprocess.run(cmd, cwd=str(workdir))
    return proc.returncode

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
        # clean up now-empty parent directory if possible
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

    if not IGC_DIR.exists():
        print(f"[ERROR] IGC folder not found: {IGC_DIR}")
        sys.exit(2)

    igc_files = sorted(IGC_DIR.glob("*.igc_subset"))
    if not igc_files:
        print(f"[ERROR] No .igc_subset files in: {IGC_DIR}")
        sys.exit(2)

    for s in (CIRCLES_SCRIPT, CLUSTERS_SCRIPT, ALT_SCRIPT):
        if not s.exists():
            print(f"[ERROR] Missing upstream script: {s}")
            sys.exit(2)

    print(f"[OK] Found {len(igc_files)} IGC files in {IGC_DIR}")

    for igc_path in igc_files:
        flight = igc_path.stem
        workdir = OUT_ROOT / flight
        ensure_dir(workdir)
        outputs_dir = workdir / "outputs"
        ensure_dir(outputs_dir)

        print(f"\n===== {flight} =====")

        # 1) Circles: writes outputs/circles.csv
        rc = run_in_cwd(py, workdir, [py, str(CIRCLES_SCRIPT), str(igc_path)])
        if rc != 0:
            print(f"[WARN] circles_from_brecords failed (rc={rc}); skipping flight")
            continue
        circles_rel = Path("outputs") / "circles.csv"
        if not (workdir / circles_rel).exists():
            print("[WARN] circles.csv not produced; skipping flight")
            continue

        # 2) Clusters: reads outputs/circles.csv, writes outputs/circle_clusters_enriched.csv
        rc = run_in_cwd(py, workdir, [py, str(CLUSTERS_SCRIPT)])
        if rc != 0:
            print(f"[WARN] circle_clusters step failed (rc={rc}); skipping flight")
            continue
        clusters_rel = Path("outputs") / "circle_clusters_enriched.csv"
        if not (workdir / clusters_rel).exists():
            print("[WARN] circle_clusters_enriched.csv not produced; skipping flight")
            continue

        # Now move the two files up into the flight folder
        move_up(workdir, circles_rel, "circles.csv")
        move_up(workdir, clusters_rel, "circle_clusters_enriched.csv")

        # 3) Altitude: writes outputs/overlay_altitude_clusters.csv
        rc = run_in_cwd(py, workdir, [py, str(ALT_SCRIPT), str(igc_path)])
        if rc != 0:
            print(f"[WARN] altitude_clusters step failed (rc={rc}); skipping flight")
            continue
        alt_rel = Path("outputs") / "overlay_altitude_clusters.csv"
        if not (workdir / alt_rel).exists():
            print("[WARN] overlay_altitude_clusters.csv not produced; skipping flight")
            continue
        move_up(workdir, alt_rel, "overlay_altitude_clusters.csv")

    print("\n[DONE] Batch CSVs under:", OUT_ROOT)

if __name__ == "__main__":
    main()
