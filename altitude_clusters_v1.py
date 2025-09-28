#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
import argparse, os

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
    return pd.DataFrame(clusters)

def outdir_for(igc_path: Path) -> Path:
    base = igc_path.stem
    outdir = Path("outputs") / base
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("igc", nargs="?", help="Path to IGC file")
    ap.add_argument("--out", help="Output CSV", default=None)
    args = ap.parse_args()

    igc_path = Path(args.igc) if args.igc else Path(input("Enter IGC file path: ").strip())
    if not igc_path.exists():
        raise FileNotFoundError(igc_path)

    df = parse_igc_brecords(igc_path)
    clusters = detect_altitude_clusters(df)

    outdir = outdir_for(igc_path)
    out_csv = Path(args.out) if args.out else outdir / "altitude_clusters.csv"
    clusters.to_csv(out_csv, index=False)
    print(f"[OK] wrote {len(clusters)} clusters → {out_csv}")

if __name__ == "__main__":
    main()