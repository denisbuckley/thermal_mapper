#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from datetime import datetime, timezone
import re
import warnings

# Base directory for your outputs
BASE_DIR = Path("/Users/denisbuckley/PycharmProjects/chatgpt_igc/outputs/batch_csv")

# Optional default stem you can change if you want
DEFAULT_STEM = "123310"

# ---------------- IGC parsing helpers ----------------

B_REC = re.compile(
    r"^B"
    r"(?P<h>\d{2})(?P<m>\d{2})(?P<s>\d{2})"              # HHMMSS
    r"(?P<latd>\d{2})(?P<latm>\d{2})(?P<latmm>\d{3})(?P<latns>[NS])"  # DDMMmmmN/S
    r"(?P<lond>\d{3})(?P<lonm>\d{2})(?P<lonmm>\d{3})(?P<lonew>[EW])"  # DDDMMmmmE/W
    r"(?P<fix>[AV])"                                       # A/V
    r"(?P<palt>-?\d{5})(?P<galt>-?\d{5})"                  # pressure, gnss
)

DATE_REC = re.compile(r"^HFDTE(?P<dd>\d{2})(?P<mm>\d{2})(?P<yy>\d{2})")

def _dm_to_deg(d_int, m_int, mm_int):
    # degrees + minutes.mmm/1000
    return d_int + (m_int + mm_int / 1000.0) / 60.0

def parse_igc(igc_path: Path) -> pd.DataFrame:
    """
    Parse core fields from a standard IGC:
      - lat (deg), lon (deg), alt (GNSS if available else pressure), time (UTC datetime)
    """
    date_utc = None
    rows = []
    with igc_path.open("r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            mdate = DATE_REC.match(line)
            if mdate:
                dd, mm, yy = int(mdate["dd"]), int(mdate["mm"]), int(mdate["yy"])
                # IGC YY is 00..99; assume 2000..2099
                year = 2000 + yy
                try:
                    date_utc = datetime(year, mm, dd, tzinfo=timezone.utc).date()
                except ValueError:
                    date_utc = None
                continue

            m = B_REC.match(line)
            if not m:
                continue

            h = int(m["h"]); mi = int(m["m"]); s = int(m["s"])
            latd = int(m["latd"]); latm = int(m["latm"]); latmm = int(m["latmm"])
            lond = int(m["lond"]); lonm = int(m["lonm"]); lonmm = int(m["lonmm"])
            lat = _dm_to_deg(latd, latm, latmm)
            lon = _dm_to_deg(lond, lonm, lonmm)
            if m["latns"] == "S":
                lat = -lat
            if m["lonew"] == "W":
                lon = -lon

            try:
                palt = int(m["palt"])
            except Exception:
                palt = None
            try:
                galt = int(m["galt"])
            except Exception:
                galt = None
            alt = galt if galt not in (None,) else palt

            # timestamp (UTC) if we know the date, else NaT + we’ll still color by dAlt sign
            if date_utc is not None:
                t = datetime(
                    date_utc.year, date_utc.month, date_utc.day,
                    h, mi, s, tzinfo=timezone.utc
                )
            else:
                t = pd.NaT

            rows.append((lat, lon, alt, t))

    if not rows:
        raise ValueError(f"No B-records found in {igc_path.name}")

    df = pd.DataFrame(rows, columns=["lat", "lon", "alt", "time"])
    # sanitize
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["lat", "lon", "alt"])
    return df.reset_index(drop=True)

# parse igc from stream

def parse_igc_from_stream(fobj, igc_name="<stream>") -> pd.DataFrame:
    """
    Parse IGC from an open text stream (used when reading from gzip).
    fobj: file-like object with lines of IGC text
    """
    date_utc = None
    rows = []
    for raw in fobj:
        line = raw.strip()
        if not line:
            continue
        mdate = DATE_REC.match(line)
        if mdate:
            dd, mm, yy = int(mdate["dd"]), int(mdate["mm"]), int(mdate["yy"])
            year = 2000 + yy
            try:
                date_utc = datetime(year, mm, dd, tzinfo=timezone.utc).date()
            except ValueError:
                date_utc = None
            continue
        m = B_REC.match(line)
        if not m:
            continue
        h = int(m["h"]); mi = int(m["m"]); s = int(m["s"])
        latd = int(m["latd"]); latm = int(m["latm"]); latmm = int(m["latmm"])
        lond = int(m["lond"]); lonm = int(m["lonm"]); lonmm = int(m["lonmm"])
        lat = _dm_to_deg(latd, latm, latmm)
        lon = _dm_to_deg(lond, lonm, lonmm)
        if m["latns"] == "S": lat = -lat
        if m["lonew"] == "W": lon = -lon
        try: palt = int(m["palt"])
        except: palt = None
        try: galt = int(m["galt"])
        except: galt = None
        alt = galt if galt is not None else palt
        if date_utc is not None:
            t = datetime(date_utc.year, date_utc.month, date_utc.day, h, mi, s, tzinfo=timezone.utc)
        else:
            t = pd.NaT
        rows.append((lat, lon, alt, t))

    if not rows:
        raise ValueError(f"No B-records found in {igc_name}")

    df = pd.DataFrame(rows, columns=["lat", "lon", "alt", "time"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["lat", "lon", "alt"])
    return df.reset_index(drop=True)

# ---------------- plotting helpers ----------------

def climb_segments(track: pd.DataFrame):
    """Return segments and boolean mask for climbing (yellow) vs sinking (blue)."""
    lon = track["lon"].to_numpy()
    lat = track["lat"].to_numpy()
    alt = track["alt"].to_numpy()

    # time diffs (s). If time missing, assume 1s steps so sign(dAlt) still works.
    if track["time"].notna().any():
        t_ns = pd.to_datetime(track["time"], utc=True, errors="coerce").view("int64").to_numpy()
        dt = np.diff(t_ns) / 1e9
        dt[dt == 0] = 1e-6
    else:
        dt = np.ones(len(alt) - 1, dtype=float)

    dalt = np.diff(alt)
    climb_mask = (dalt / dt) > 0

    points = np.column_stack([lon, lat])
    segs = np.stack([points[:-1], points[1:]], axis=1)
    return segs, climb_mask

def read_clusters(dir_path: Path, fname: str):
    p = dir_path / fname
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
    except Exception as e:
        warnings.warn(f"Failed reading {fname}: {e}")
        return None
    cols = {c.lower(): c for c in df.columns}
    if "lat" not in cols or "lon" not in cols:
        warnings.warn(f"{fname} missing lat/lon columns")
        return None
    out = pd.DataFrame({"lat": df[cols["lat"]].astype(float),
                        "lon": df[cols["lon"]].astype(float)})
    # grab cluster_id if present
    if "cluster_id" in cols:
        out["cluster_id"] = df[cols["cluster_id"]]
    return out

def matched_positions(circle_df, alt_df, dir_path: Path):
    """Use matched_clusters.csv if present; else intersect by cluster_id if both have it."""
    mfile = dir_path / "matched_clusters.csv"
    if mfile.exists():
        m = pd.read_csv(mfile)
        cols = {c.lower(): c for c in m.columns}
        if "lat" in cols and "lon" in cols:
            return pd.DataFrame({"lat": m[cols["lat"]].astype(float),
                                 "lon": m[cols["lon"]].astype(float)})
    if circle_df is not None and alt_df is not None and \
       "cluster_id" in circle_df and "cluster_id" in alt_df:
        common = pd.Index(circle_df["cluster_id"]).intersection(alt_df["cluster_id"])
        if len(common):
            sub = circle_df[circle_df["cluster_id"].isin(common)][["lat","lon"]].copy()
            return sub.reset_index(drop=True)
    return None

# ---------------------------------------------------------------------------
# UTIL: ensure_igc_copy  (monolithic; no external imports)
# ---------------------------------------------------------------------------
from pathlib import Path
import shutil, gzip

def ensure_igc_copy(stem: str, run_dir: Path, src_root: Path = Path("igc")) -> bool:
    """
    Ensure run_dir/<stem>.igc exists by copying (or gunzipping) from ./igc/.
    Accepts .igc, .IGC, or .igc.gz in src_root.
    Returns True if <stem>.igc exists in run_dir after this call.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    dest = run_dir / f"{stem}.igc"

    # already there
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[IGC] already present → {dest}")
        return True

    for cand in [src_root / f"{stem}.igc",
                 src_root / f"{stem}.IGC",
                 src_root / f"{stem}.igc.gz"]:
        if cand.exists():
            try:
                if cand.suffix == ".gz":
                    with gzip.open(cand, "rb") as fin, open(dest, "wb") as fout:
                        shutil.copyfileobj(fin, fout)
                    print(f"[IGC] gunzipped {cand.name} → {dest.name}")
                else:
                    shutil.copy2(cand, dest)
                    print(f"[IGC] copied {cand.name} → {dest.name}")
                return True
            except Exception as e:
                print(f"[IGC] ERROR copying {cand} → {dest}: {e}")
                return False

    print(f"[IGC] WARN: no IGC found for {stem} in {src_root}")
    return False

# ---------------- main ----------------

def main():
    stem = input(f"Enter flight stem under outputs/batch_csv (default: {DEFAULT_STEM}): ").strip() or DEFAULT_STEM
    dir_path = BASE_DIR / stem
    if not dir_path.exists():
        sys.exit(f"Directory not found: {dir_path}")

    # Prefer track.csv; else parse IGC
    track_csv = dir_path / "track.csv"
    if track_csv.exists():
        df = pd.read_csv(track_csv)
        # normalize column names
        cols = {c.lower(): c for c in df.columns}
        need = {"lat","lon","alt"}
        if not need.issubset(cols):
            raise ValueError(f"track.csv must include {need}, got {list(df.columns)}")
        track = pd.DataFrame({
            "lat": df[cols["lat"]].astype(float),
            "lon": df[cols["lon"]].astype(float),
            "alt": df[cols["alt"]].astype(float),
        })
        if "time" in cols:
            track["time"] = pd.to_datetime(df[cols["time"]], errors="coerce", utc=True)
        else:
            track["time"] = pd.NaT
    else:
        # try specific stems in common extensions, else any IGC-like file
        candidates = [
            dir_path / f"{stem}.igc",
            dir_path / f"{stem}.IGC",
            dir_path / f"{stem}.igc.gz",
            dir_path / f"{stem}.IGC.gz",
        ]
        igc_path = next((p for p in candidates if p.exists()), None)
        if igc_path is None:
            igcs = (list(dir_path.glob("*.igc")) + list(dir_path.glob("*.IGC")) +
                    list(dir_path.glob("*.igc.gz")) + list(dir_path.glob("*.IGC.gz")))
            if not igcs:
                sys.exit(f"No track.csv and no .igc/.IGC/.igc.gz files found in {dir_path}")
            igc_path = igcs[0]

        # handle gzip transparently
        if igc_path.suffix.lower() == ".gz":
            import gzip, io
            with gzip.open(igc_path, "rt", errors="ignore") as f:
                text = f.read()
            # feed the text to a small parse wrapper
            from io import StringIO
            track = parse_igc_from_stream(StringIO(text), igc_name=igc_path.name)
        else:
            track = parse_igc(igc_path)

    if len(track) < 2:
        sys.exit("Track has fewer than 2 points; cannot plot segments.")

    segs, is_climb = climb_segments(track)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_title(f"Glider Track & Clusters — {stem}")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

    lc_sink = LineCollection(segs[~is_climb], colors="#1E6EFF", linewidths=3.2, alpha=0.9)  # blue much thicker
    lc_climb = LineCollection(segs[is_climb], colors="#FFD400", linewidths=4.0, alpha=0.95)  # yellow much thicker
    ax.add_collection(lc_sink); ax.add_collection(lc_climb)

    # sensible extents
    ax.set_xlim(track["lon"].min() - 0.01, track["lon"].max() + 0.01)
    ax.set_ylim(track["lat"].min() - 0.01, track["lat"].max() + 0.01)

    # Cluster overlays
    circle_df = read_clusters(dir_path, "circle_clusters_enriched.csv")
    if circle_df is not None and len(circle_df):
        ax.scatter(circle_df["lon"], circle_df["lat"], s=200, facecolors="none",
                   edgecolors="red", marker="o", linewidths=2.2,
                   label="circle_clusters (purple ○)")

    alt_df = read_clusters(dir_path, "altitude_clusters.csv")
    ax.scatter(alt_df["lon"], alt_df["lat"], s=80, facecolors="none",
               edgecolors="green", marker="s", linewidths=2.0,
               label="altitude_clusters (green □)")

    '''m = matched_positions(circle_df, alt_df, dir_path)
    if m is not None and len(m):
        ax.scatter(m["lon"], m["lat"], s=70, color="red", marker="x",
                   linewidths=2.0, label="matched (red ×)")'''

    ax.legend(loc="best", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()