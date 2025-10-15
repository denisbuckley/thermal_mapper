#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
track_plotter_batch_v1a.py
Batch-render track + cluster overlays for all flight stems under outputs/batch_csv.

Outputs PNGs to outputs/batch_png/<stem>.png by default.

Usage examples:
  python track_plotter_batch_v1a.py
  python track_plotter_batch_v1a.py --glob "105*" --limit 20
  python track_plotter_batch_v1a.py --stems 105351 105354 --show
  python track_plotter_batch_v1a.py --save-dir outputs/custom_png --dpi 200 --overwrite
"""

from __future__ import annotations
import sys, re, math, gzip, argparse
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import warnings

# ---- roots ----
ROOT = Path.cwd()
BATCH_CSV = ROOT / "outputs" / "batch_csv"
BATCH_PNG = ROOT / "outputs" / "batch_png"

# ---------------- IGC parsing helpers ----------------

B_REC = re.compile(
    r"^B"
    r"(?P<h>\d{2})(?P<m>\d{2})(?P<s>\d{2})"
    r"(?P<latd>\d{2})(?P<latm>\d{2})(?P<latmm>\d{3})(?P<latns>[NS])"
    r"(?P<lond>\d{3})(?P<lonm>\d{2})(?P<lonmm>\d{3})(?P<lonew>[EW])"
    r"(?P<fix>[AV])"
    r"(?P<palt>-?\d{5})(?P<galt>-?\d{5})"
)
DATE_REC = re.compile(r"^HFDTE(?P<dd>\d{2})(?P<mm>\d{2})(?P<yy>\d{2})")
PILOT_REC = re.compile(r"^HFPLTPILOTINCHARGE:(?P<name>.+)$")

def _dm_to_deg(d_int, m_int, mm_int):
    return d_int + (m_int + mm_int / 1000.0) / 60.0

def _parse_lines(iter_lines, igc_name="<stream>") -> pd.DataFrame:
    date_utc = None
    rows = []
    for raw in iter_lines:
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

def parse_igc(path: Path) -> pd.DataFrame:
    return _parse_lines(path.open("r", errors="ignore"), igc_name=path.name)

def parse_igc_gz(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", errors="ignore") as f:
        return _parse_lines(f, igc_name=path.name)

def extract_pilot_from_igc(path: Path) -> str | None:
    """Return pilot name from HFPLTPILOTINCHARGE header, if present."""
    opener = (lambda p: gzip.open(p, "rt", errors="ignore")) if path.suffix.lower() == ".gz" else (lambda p: open(p, "r", errors="ignore"))
    try:
        with opener(path) as f:
            for _ in range(200):  # header lines only
                line = f.readline()
                if not line:
                    break
                m = PILOT_REC.match(line.strip())
                if m:
                    return m["name"].strip()
    except Exception:
        pass
    return None

# ---------------- small math helpers ----------------

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371008.8
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dl   = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

# ---------------- plotting helpers ----------------

def climb_segments(track: pd.DataFrame):
    lon = track["lon"].to_numpy()
    lat = track["lat"].to_numpy()
    alt = track["alt"].to_numpy()
    if track["time"].notna().any():
        t_ns = pd.to_datetime(track["time"], utc=True, errors="coerce").astype("int64", copy=False).to_numpy()
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
    if "cluster_id" in cols:
        out["cluster_id"] = df[cols["cluster_id"]]
    return out

def matched_positions(circle_df, alt_df, dir_path: Path):
    mfile = dir_path / "matched_clusters.csv"
    if mfile.exists():
        m = pd.read_csv(mfile)
        cols = {c.lower(): c for c in m.columns}
        if "lat" in cols and "lon" in cols:
            return pd.DataFrame({"lat": m[cols["lat"]].astype(float),
                                 "lon": m[cols["lon"]].astype(float)})
    if (circle_df is not None and alt_df is not None and
        "cluster_id" in getattr(circle_df, "columns", []) and
        "cluster_id" in getattr(alt_df, "columns", [])):
        common = pd.Index(circle_df["cluster_id"]).intersection(alt_df["cluster_id"])
        if len(common):
            sub = circle_df[circle_df["cluster_id"].isin(common)][["lat","lon"]].copy()
            return sub.reset_index(drop=True)
    return None

def load_track(run_dir: Path, stem: str) -> pd.DataFrame:
    # track.csv preferred
    p = run_dir / "track.csv"
    if p.exists():
        df = pd.read_csv(p)
        cols = {c.lower(): c for c in df.columns}
        need = {"lat","lon","alt"}
        if not need.issubset(cols):
            raise ValueError(f"{p} must include {need}, got {list(df.columns)}")
        out = pd.DataFrame({
            "lat": df[cols["lat"]].astype(float),
            "lon": df[cols["lon"]].astype(float),
            "alt": df[cols["alt"]].astype(float),
        })
        out["time"] = pd.to_datetime(df[cols["time"]], errors="coerce", utc=True) if "time" in cols else pd.NaT
        return out

    # else try IGC variants inside run_dir
    for cand in [
        run_dir / f"{stem}.igc",
        run_dir / f"{stem}.IGC",
        run_dir / f"{stem}.igc.gz",
        run_dir / f"{stem}.IGC.gz",
    ] + list(run_dir.glob("*.igc")) + list(run_dir.glob("*.IGC")) + list(run_dir.glob("*.igc.gz")) + list(run_dir.glob("*.IGC.gz")):
        if cand.exists():
            return parse_igc_gz(cand) if cand.suffix.lower() == ".gz" else parse_igc(cand)

    raise FileNotFoundError(f"No track.csv or IGC found in {run_dir}")

def find_pilot(run_dir: Path, stem: str) -> str:
    """Look for pilot in any IGC sitting in the run folder (best-effort)."""
    for cand in [
        run_dir / f"{stem}.igc",
        run_dir / f"{stem}.IGC",
        run_dir / f"{stem}.igc.gz",
        run_dir / f"{stem}.IGC.gz",
    ] + list(run_dir.glob("*.igc")) + list(run_dir.glob("*.IGC")) + list(run_dir.glob("*.igc.gz")) + list(run_dir.glob("*.IGC.gz")):
        if cand.exists():
            name = extract_pilot_from_igc(cand)
            if name:
                return name
    return "Unknown"

def km(x):  # meters -> kilometers nicely
    return f"{x/1000.0:.1f} km"

def hhmmss(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds <= 0:
        return "00:00"
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def matched_stats_lines(run_dir: Path) -> list[str]:
    """Multiline summary derived from matched_clusters.csv (if present)."""
    p = run_dir / "matched_clusters.csv"
    if not p.exists():
        return ["matches: 0"]
    try:
        m = pd.read_csv(p)
    except Exception:
        return ["matches: (read error)"]
    if m.empty:
        return ["matches: 0"]

    cols = {c.lower(): c for c in m.columns}
    def get(col): return cols.get(col)
    lines = [f"matches: {len(m)}"]

    def add_stat(label, colname, fn=np.nanmean, fmt=".2f", prefix="", suffix=""):
        c = get(colname)
        if not c:
            lines.append(f"{label}: n/a");
            return
        vals = pd.to_numeric(m[c], errors="coerce").dropna()
        if not len(vals):
            lines.append(f"{label}: n/a")
            return
        val = format(fn(vals), fmt)
        lines.append(f"{label}: {prefix}{val}{suffix}")

    add_stat("climb_rate_ms mean", "climb_rate_ms")
    # p90 of climb rate
    c = get("climb_rate_ms")
    if c and m[c].notna().any():
        p90 = np.percentile(pd.to_numeric(m[c], errors="coerce").dropna(), 90)
        lines[-1] += f" (p90 {p90:.2f})"

    add_stat("alt_gain_m mean", "alt_gain_m")
    add_stat("dur_s mean", "duration_s")
    add_stat("dist_m mean", "dist_m")
    if get("overlap_frac"):
        vals = pd.to_numeric(m[get("overlap_frac")], errors="coerce").dropna()
        if len(vals):
            lines.append(f"overlap {vals.mean():.2f}")
    return lines

# ---------------- rendering ----------------

def render_one(stem: str, run_dir: Path, out_png: Path, show: bool = False, dpi: int = 160):
    track = load_track(run_dir, stem)
    if len(track) < 2:
        print(f"[SKIP] {stem}: <2 points")
        return

    segs, is_climb = climb_segments(track)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_title(f"Glider Track & Clusters — {stem}")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

    # thick track lines
    lc_sink  = LineCollection(segs[~is_climb], colors="#1E6EFF", linewidths=4.0, alpha=0.9)
    lc_climb = LineCollection(segs[ is_climb], colors="#FFD400", linewidths=4.0, alpha=0.95)
    ax.add_collection(lc_sink); ax.add_collection(lc_climb)

    # sensible extents
    ax.set_xlim(track["lon"].min() - 0.01, track["lon"].max() + 0.01)
    ax.set_ylim(track["lat"].min() - 0.01, track["lat"].max() + 0.01)

    # overlays
    alt_df = read_clusters(run_dir, "altitude_clusters.csv")
    if alt_df is not None and len(alt_df):
        ax.scatter(alt_df["lon"], alt_df["lat"], s=100, facecolors="none",
                   edgecolors="green", marker="s", linewidths=2.2,
                   label="altitude_clusters (green □)")

    circle_df = read_clusters(run_dir, "circle_clusters_enriched.csv")
    if circle_df is not None and len(circle_df):
        ax.scatter(circle_df["lon"], circle_df["lat"], s=200, facecolors="none",
                   edgecolors="red", marker="o", linewidths=2.4,
                   label="circle_clusters (red ○)")

    m = matched_positions(circle_df, alt_df, run_dir)
    if m is not None and len(m):
        ax.scatter(m["lon"], m["lat"], s=300, color="purple", marker="x",
                   linewidths=2.8, label="matched (purple ×)")

    ax.legend(loc="best", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()

    # ---------- Stats box (outside axes, bottom-right; vertical) ----------
    # Pilot
    pilot = find_pilot(run_dir, stem)

    # Distance & duration
    # distance over 2D track (lon/lat); meters
    lon = track["lon"].to_numpy(); lat = track["lat"].to_numpy()
    dist_m = float(np.sum([haversine_m(lat[i-1], lon[i-1], lat[i], lon[i]) for i in range(1, len(track))]))
    # duration from first/last valid time
    if track["time"].notna().any():
        t = pd.to_datetime(track["time"], utc=True, errors="coerce")
        t0 = t.dropna().iloc[0] if t.notna().any() else None
        t1 = t.dropna().iloc[-1] if t.notna().any() else None
        dur_s = float((t1 - t0).total_seconds()) if (t0 is not None and t1 is not None) else float("nan")
    else:
        dur_s = float("nan")

    # Weglide URL (PNG can’t be clickable; included as text)
    wg_url = f"https://www.weglide.org/flight/{stem}"

    lines = [
        f"Pilot: {pilot}",
        f"Distance: {km(dist_m)}",
        f"Duration: {hhmmss(dur_s)}",
        f"Weglide: {wg_url}",
        *matched_stats_lines(run_dir),
    ]
    box_text = "\n".join(lines)

    # Place at figure bottom-right, outside axes
    fig.text(
        0.99, 0.01, box_text,
        ha="right", va="bottom",
        fontsize=9, color="black",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.95)
    )

    # ---------- Save ----------
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    print(f"[OK] saved {out_png}")

    if show:
        plt.show()
    plt.close(fig)

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description="Batch track+cluster plotter")
    ap.add_argument("--glob", default="*", help="glob for stems under outputs/batch_csv (default: *)")
    ap.add_argument("--stems", nargs="*", help="explicit list of stems to plot (overrides --glob)")
    ap.add_argument("--limit", type=int, default=0, help="limit number of plots (0 = no limit)")
    ap.add_argument("--save-dir", default=str(BATCH_PNG), help="output PNG folder")
    ap.add_argument("--dpi", type=int, default=160, help="PNG DPI (default 160)")
    ap.add_argument("--show", action="store_true", help="also show on screen")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing PNGs")
    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg", force=True)

    save_dir = Path(args.save_dir)

    if args.stems:
        stems = args.stems
    else:
        stems = sorted([p.name for p in BATCH_CSV.glob(args.glob) if p.is_dir()])

    if not stems:
        print(f"[INFO] no stems under {BATCH_CSV} matching '{args.glob}'")
        return 0

    count = 0
    for stem in stems:
        run_dir = BATCH_CSV / stem
        if not run_dir.exists():
            continue
        out_png = save_dir / f"{stem}.png"
        if out_png.exists() and not args.overwrite:
            print(f"[SKIP] {out_png.name} exists (use --overwrite to replace)")
            count += 1
            if args.limit and count >= args.limit:
                break
            continue
        try:
            render_one(stem, run_dir, out_png, show=args.show, dpi=args.dpi)
        except Exception as e:
            print(f"[ERR] {stem}: {e}")
        count += 1
        if args.limit and count >= args.limit:
            break

    print(f"[DONE] rendered {count} plot(s)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())