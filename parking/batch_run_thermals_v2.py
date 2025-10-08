#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch runner for IGC thermal analysis"
    )
    parser.add_argument(
        "--igc_subset-dir",
        type=str,
        default="igc_subset",
        help="Path to directory containing IGC files (default: ./igc_subset)"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/batch",
        help="Where outputs are written (default: ./outputs/batch)"
    )
    # Add any other args here (matcher, tuning, etc.)
    return parser.parse_args()

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def main():
    args = parse_args()

    igc_dir = Path(args.igc_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    ensure_dir(outdir)

    # --- Preflight checks ---
    if not igc_dir.exists():
        print(f"[ERROR] IGC folder not found: {igc_dir}")
        sys.exit(2)

    igc_files = sorted(igc_dir.glob("*.igc_subset"))
    if not igc_files:
        print(f"[ERROR] No .igc_subset files found in: {igc_dir}")
        sys.exit(2)

    print(f"[OK] Found {len(igc_files)} IGC files in {igc_dir}")
    print(f"[INFO] Outputs will be written to {outdir}")

    # --- your existing batch logic goes here ---
    # for f in igc_files:
    #     process_file(f, outdir)

if __name__ == "__main__":
    main()
