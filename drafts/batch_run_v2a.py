
#!/usr/bin/env python3
# batch_run_v2a.py
# Run the 3 upstream scripts across all IGC files in ./igc_subset
# and place their CSV outputs directly under:
#   outputs/batch_csv/<flight_basename>/
# Files expected (per flight):
#   circles.csv
#   circle_clusters_enriched.csv
#   overlay_altitude_clusters.csv
#
# Implementation detail:
# - Each upstream script writes to ./outputs/<file>.csv relative to its CWD.
# - We set CWD = outputs/batch_csv/<flight_basename>, run the script,
#   then move outputs/<file>.csv up one level to the CWD (removing the extra outputs/ layer).
#
# No CLI args required; just Run in PyCharm from the project root.

import sys
import subprocess
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent
IGC_DIR = PROJECT_ROOT / "igc_subset"
OUT_ROOT = PROJECT_ROOT / "outputs" / "batch_csv"

# Upstream scripts expected in the project root (adjust paths if needed)
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
        shutil.move(str(src), str(dst))
        # clean up now-empty outputs/ directory if present
        outdir = src.parent
        try:
            if outdir.is_dir() and not any(outdir.iterdir()):
                outdir.rmdir()
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

    # Sanity: scripts exist
    for s in (CIRCLES_SCRIPT, CLUSTERS_SCRIPT, ALT_SCRIPT):
        if not s.exists():
            print(f"[ERROR] Missing upstream script: {s}")
            sys.exit(2)

    print(f"[OK] Found {len(igc_files)} IGC files in {IGC_DIR}")

    for igc_path in igc_files:
        flight = igc_path.stem
        workdir = OUT_ROOT / flight
        ensure_dir(workdir)
        print(f"\n===== {flight} =====")

        # 1) circles_from_brecords_v1d.py <IGC>  → outputs/circles.csv
        rc = run_in_cwd(py, workdir, [py, str(CIRCLES_SCRIPT), str(igc_path)])
        if rc != 0:
            print(f"[WARN] circles_from_brecords failed (rc={rc}); skipping flight")
            continue
        if not move_up(workdir, Path("../outputs") / "circles.csv", "circles.csv"):
            print("[WARN] circles.csv missing; skipping flight")
            continue

        # 2) circle_clusters_v1s.py  (reads outputs/circles.csv) → outputs/circle_clusters_enriched.csv
        rc = run_in_cwd(py, workdir, [py, str(CLUSTERS_SCRIPT)])
        if rc != 0:
            print(f"[WARN] circle_clusters step failed (rc={rc}); skipping flight")
            continue
        if not move_up(workdir, Path("../outputs") / "circle_clusters_enriched.csv", "circle_clusters_enriched.csv"):
            print("[WARN] circle_clusters_enriched.csv missing; skipping flight")
            continue

        # 3) overlay_altitude_clusters_v1c.py <IGC> → outputs/overlay_altitude_clusters.csv
        rc = run_in_cwd(py, workdir, [py, str(ALT_SCRIPT), str(igc_path)])
        if rc != 0:
            print(f"[WARN] altitude_clusters step failed (rc={rc}); skipping flight")
            continue
        if not move_up(workdir, Path("../outputs") / "overlay_altitude_clusters.csv", "overlay_altitude_clusters.csv"):
            print("[WARN] overlay_altitude_clusters.csv missing; skipping flight")
            continue

    print("\n[DONE] Batch CSVs written under:", OUT_ROOT)

if __name__ == "__main__":
    main()
