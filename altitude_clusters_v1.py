#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
import argparse, os

LAST_IGC_FILE = Path(".last_igc")

def _read_last_igc() -> Path | None:
    try:
        p = Path(LAST_IGC_FILE.read_text(encoding="utf-8").strip())
        return p if p.exists() else None
    except Exception:
        return None

def _write_last_igc(p: Path) -> None:
    try:
        LAST_IGC_FILE.write_text(str(p.resolve()), encoding="utf-8")
    except Exception:
        pass
def parse_igc_brecords(igc_path: Path) -> pd.DataFrame:
    """
    Parse IGC B-records into a DataFrame with time, lat, lon, alt.
    Very simplified parser: assumes IGC lines starting with 'B'.
    """
    rows = []
    with open(igc_path, "r") as f:
        for line in f:
            if not line.startswith("B"):
                continue
            # Example B record: BHHMMSSDDMMmmmNDDDMMmmmEXXXXX
            t = int(line[1:3]) * 3600 + int(line[3:5]) * 60 + int(line[5:7])
            lat = int(line[7:9]) + int(line[9:14]) / 60000.0
            if line[14] == "S": lat = -lat
            lon = int(line[15:18]) + int(line[18:23]) / 60000.0
            if line[23] == "W": lon = -lon
            alt = int(line[25:30])
            rows.append((t, lat, lon, alt))
    return pd.DataFrame(rows, columns=["time_s", "lat", "lon", "alt"])

def detect_altitude_clusters(df: pd.DataFrame,
                             min_gain=30,   # meters
                             min_duration=20):  # seconds
    """
    Identify climb segments from altitude gains over time.
    Returns DataFrame of clusters with mean lat/lon and climb rate.
    """
    clusters = []
    start_i = None

    for i in range(1, len(df)):
        dalt = df.alt.iloc[i] - df.alt.iloc[i-1]
        dt = df.time_s.iloc[i] - df.time_s.iloc[i-1]
        if dalt > 0:  # climbing
            if start_i is None:
                start_i = i-1
        else:
            if start_i is not None:
                # close a climb segment
                seg = df.iloc[start_i:i]
                gain = seg.alt.iloc[-1] - seg.alt.iloc[0]
                dur = seg.time_s.iloc[-1] - seg.time_s.iloc[0]
                if gain >= min_gain and dur >= min_duration:
                    clusters.append({
                        "t_start": seg.time_s.iloc[0],
                        "t_end": seg.time_s.iloc[-1],
                        "duration_s": dur,
                        "alt_gain_m": gain,
                        "climb_rate_ms": gain / dur,
                        "lat": seg.lat.mean(),
                        "lon": seg.lon.mean(),
                    })
                start_i = None
    out = pd.DataFrame(clusters).reset_index(drop=True)
    out["cluster_id"] = out.index  # 0..N-1
    return out

def outdir_for(igc_path: Path) -> Path:
    base = igc_path.stem
    outdir = Path("outputs") / base
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

def main():
    import argparse
    from pathlib import Path
    import glob

    ap = argparse.ArgumentParser()
    ap.add_argument("igc", nargs="?", help="Path to IGC file")
    ap.add_argument("--out", help="Override output CSV path")
    args = ap.parse_args()

    # Resolve IGC path: prefer arg, else valid .last_igc, else first file in ./igc
    last_igc = _read_last_igc()
    hint = None

    # Build a fallback list from ./igc
    try:
        from glob import glob
        candidates = (
            sorted(glob("igc/*.igc")) +
            sorted(glob("igc/*.IGC")) +
            []  # add "igc/*.igc.gz" if you later support gunzip here
        )
        fallback = Path(candidates[0]) if candidates else None
    except Exception:
        fallback = None

    # Prefer last_igc if it still exists; else fallback
    if last_igc and last_igc.exists():
        hint = last_igc
    elif fallback and fallback.exists():
        hint = fallback

    if args.igc:
        igc_path = Path(args.igc).expanduser()
    else:
        prompt = f"Enter IGC file path [{'default: ' + str(hint) if hint else 'no default'}]: "
        user_in = input(prompt).strip()
        igc_path = Path(user_in).expanduser() if user_in else hint

    if igc_path is None or not igc_path.exists():
        raise FileNotFoundError(igc_path or "<none>")

    # Remember the chosen file for next run
    _write_last_igc(igc_path)
    print(f"[INFO] using IGC: {igc_path}")

    # Per-flight run directory: outputs/batch_csv/<flight>
    flight = igc_path.stem.rstrip(')').strip()
    run_dir = Path("outputs") / "batch_csv" / flight
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build altitude clusters
    track = parse_igc_brecords(igc_path)
    clusters = detect_altitude_clusters(track)

    # Give each cluster an ID and standardize column order
    clusters = clusters.reset_index(drop=True)
    clusters["cluster_id"] = clusters.index
    canonical = ["cluster_id", "t_start", "t_end", "lat", "lon",
                 "climb_rate_ms", "alt_gain_m", "duration_s"]
    extras = [c for c in clusters.columns if c not in canonical]
    clusters = clusters[[c for c in canonical if c in clusters.columns] + extras]

    # Output path: per-flight folder unless overridden
    out_path = Path(args.out) if args.out else (run_dir / "altitude_clusters.csv")
    clusters.to_csv(out_path, index=False)
    print(f"[OK] wrote {len(clusters)} altitude clusters → {out_path}")
    return 0
if __name__ == "__main__":
    main()