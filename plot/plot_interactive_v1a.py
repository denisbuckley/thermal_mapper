#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# plot_interactive_v1.py — standalone interactive (Plotly) explorer with hover sync

from __future__ import annotations

import json
import re
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ---------- roots ----------
ROOT = Path.cwd()
OUT_ROOT = ROOT / "outputs" / "batch_csv"
LAST_IGC_PATH_FILE = ROOT / ".last_igc_path"

# ---------- minimal IGC parser (B-records) ----------
B_REC = re.compile(
    r"^B"
    r"(?P<h>\d{2})(?P<m>\d{2})(?P<s>\d{2})"
    r"(?P<latd>\d{2})(?P<latm>\d{2})(?P<latmm>\d{3})(?P<latns>[NS])"
    r"(?P<lond>\d{3})(?P<lonm>\d{2})(?P<lonmm>\d{3})(?P<lonew>[EW])"
    r"(?P<fix>[AV])"
    r"(?P<palt>-?\d{5})(?P<galt>-?\d{5})"
)
DATE_REC = re.compile(r"^HFDTE(?P<dd>\d{2})(?P<mm>\d{2})(?P<yy>\d{2})")


def _dm_to_deg(d_int: int, m_int: int, mm_int: int) -> float:
    return d_int + (m_int + mm_int / 1000.0) / 60.0


def parse_igc_brecords(igc_path: Path) -> pd.DataFrame:
    """
    Returns columns: time_s (float), lat (deg), lon (deg), alt (m)
    time_s is seconds from first B-fix of the file date (if present in headers, else from 00:00).
    """
    date_utc = None
    rows = []
    with igc_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
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

            # prefer GNSS alt; fall back to pressure alt
            try:
                galt = int(m["galt"])
            except Exception:
                galt = None
            try:
                palt = int(m["palt"])
            except Exception:
                palt = None
            alt = float(galt if galt is not None else palt) if (galt is not None or palt is not None) else float("nan")

            # construct seconds of day
            t_s = h * 3600 + mi * 60 + s
            rows.append((t_s, lat, lon, alt))

    if not rows:
        return pd.DataFrame(columns=["time_s", "lat", "lon", "alt"])

    df = pd.DataFrame(rows, columns=["time_s", "lat", "lon", "alt"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["lat", "lon", "alt"])
    # normalize time_s to start at zero
    if not df.empty:
        df["time_s"] = df["time_s"] - float(df["time_s"].iloc[0])
    return df.reset_index(drop=True)


# ---------- helpers to read batch artifacts for overlays ----------
def read_clusters(dir_path: Path, fname: str) -> pd.DataFrame | None:
    p = dir_path / fname
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
    except Exception:
        return None
    cols = {c.lower(): c for c in df.columns}
    if "lat" not in cols or "lon" not in cols:
        return None
    out = pd.DataFrame({
        "lat": pd.to_numeric(df[cols["lat"]], errors="coerce"),
        "lon": pd.to_numeric(df[cols["lon"]], errors="coerce"),
    }).dropna()
    return out


# ---------- interactive HTML builder with hover sync ----------
def save_interactive_html(igc_path: Path, out_html: Path) -> None:
    from plotly.offline import plot as plotly_plot
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = parse_igc_brecords(igc_path)
    if df.empty:
        print(f"[WARN] No valid B-records in {igc_path.name}")
        return

    # time axis in minutes
    tx = df["time_s"].to_numpy(dtype=float) / 60.0
    lon = df["lon"].to_numpy(dtype=float)
    lat = df["lat"].to_numpy(dtype=float)
    alt = df["alt"].to_numpy(dtype=float)

    # Optional overlays from batch outputs
    stem = igc_path.stem
    run_dir = OUT_ROOT / stem
    circles = read_clusters(run_dir, "circle_clusters_enriched.csv")
    alts = read_clusters(run_dir, "altitude_clusters.csv")
    matched = read_clusters(run_dir, "matched_clusters.csv")  # NEW: matched overlay

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=False,
        row_heights=[0.60, 0.40], vertical_spacing=0.08,
        subplot_titles=(f"Track (lon/lat) — {stem}", "Altitude vs time (min)")
    )

    # Base traces (colors unchanged)
    track_trace = go.Scattergl(
        x=lon, y=lat, mode="lines",
        line=dict(width=2, color="blue"), name="Track"
    )
    alt_trace = go.Scattergl(
        x=tx, y=alt, mode="lines",
        line=dict(width=2, color="black"), name="Altitude"
    )

    fig.add_trace(track_trace, row=1, col=1)
    fig.add_trace(alt_trace,   row=2, col=1)

    # Overlays (keep colors/shapes)
    if alts is not None and len(alts):
        fig.add_trace(
            go.Scattergl(
                x=alts["lon"], y=alts["lat"], mode="markers",
                marker=dict(symbol="square-open", size=22, color="green"),
                name="altitude_clusters"
            ),
            row=1, col=1
        )
    if circles is not None and len(circles):
        fig.add_trace(
            go.Scattergl(
                x=circles["lon"], y=circles["lat"], mode="markers",
                marker=dict(symbol="circle-open", size=20, color='black'),
                name="circle_clusters"
            ),
            row=1, col=1
        )
    # NEW: matched clusters (red ×)
    if matched is not None and len(matched):
        fig.add_trace(
            go.Scattergl(
                x=matched["lon"], y=matched["lat"], mode="markers",
                marker=dict(symbol="x", size=15, color="red"),
                name="matched (red ×)"
            ),
            row=1, col=1
        )

    # Cursor markers (updated via JS)
    fig.add_trace(
        go.Scattergl(x=[], y=[], mode="markers",
                     marker=dict(color="blue", size=10),
                     name="cursor_map"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scattergl(x=[], y=[], mode="markers",
                     marker=dict(color="blue", size=10),
                     name="cursor_alt"),
        row=2, col=1
    )

    fig.update_xaxes(title_text="Longitude", row=1, col=1)
    fig.update_yaxes(title_text="Latitude",  row=1, col=1)
    fig.update_xaxes(title_text="Time (min)", row=2, col=1)
    fig.update_yaxes(title_text="Altitude (m)", row=2, col=1)
    fig.update_layout(title=f"Interactive Flight Explorer — {stem}", hovermode="closest")

    # Render to a single <div> with plotlyjs hosted on CDN
    html_div = plotly_plot(fig, output_type="div", include_plotlyjs="cdn", auto_open=False)

    # Arrays for JS (use json.dumps to serialize safely)
    lon_js = json.dumps(lon.tolist())
    lat_js = json.dumps(lat.tolist())
    tx_js  = json.dumps(tx.tolist())
    alt_js = json.dumps(alt.tolist())

    # Inject small JS that syncs the hover by pointIndex
    # IMPORTANT: Ignore hovers coming from overlay markers so we only sync from main lines.
    post_js = f"""
<script>
(function() {{
  var gd = document.querySelector('div.plotly-graph-div');
  if(!gd) return;

  var LON = {lon_js};
  var LAT = {lat_js};
  var TX  = {tx_js};
  var ALT = {alt_js};

  function clamp(i, n) {{ return Math.max(0, Math.min(n-1, i)); }}

  gd.on('plotly_hover', function(evt) {{
    if(!evt || !evt.points || !evt.points.length) return;
    var pt = evt.points[0];

    // Only respond when hovering the main lines (Track or Altitude)
    var traceName = (gd.data[pt.curveNumber] && gd.data[pt.curveNumber].name) || "";
    if (traceName !== "Track" && traceName !== "Altitude") return;

    var idx = clamp(pt.pointIndex, LON.length);

    // cursor traces are always the last two
    var totalTraces = gd.data.length;
    var cursorMapIdx = totalTraces - 2;
    var cursorAltIdx = totalTraces - 1;

    Plotly.restyle(gd, {{x: [[LON[idx]]], y: [[LAT[idx]]]}}, [cursorMapIdx]);
    Plotly.restyle(gd, {{x: [[TX[idx]]],  y: [[ALT[idx]]]}}, [cursorAltIdx]);
  }});
}})();
</script>
"""

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html_div + post_js, encoding="utf-8")
    print(f"[OK] saved interactive HTML with hover sync → {out_html}")


# ---------- main (prompts for IGC path, remembers last) ----------
def main() -> int:
    # load prior default if present
    default_path = ""
    if LAST_IGC_PATH_FILE.exists():
        try:
            default_path = LAST_IGC_PATH_FILE.read_text(encoding="utf-8").strip()
        except Exception:
            default_path = ""

    prompt = f"Enter path to IGC file{f' [default: {default_path}]' if default_path else ''}: "
    user_in = input(prompt).strip()
    if not user_in:
        if not default_path:
            print("[ERROR] No path provided and no previous default.")
            return 2
        igc_path = Path(default_path).expanduser()
    else:
        igc_path = Path(user_in).expanduser()
        # persist as new default
        try:
            LAST_IGC_PATH_FILE.write_text(str(igc_path), encoding="utf-8")
        except Exception:
            pass

    if not igc_path.exists():
        print(f"[ERROR] IGC not found: {igc_path}")
        return 2

    stem = igc_path.stem
    out_html = OUT_ROOT / stem / "interactive" / f"{stem}.html"
    try:
        save_interactive_html(igc_path, out_html)
    except Exception as e:
        print(f"[WARN] interactive HTML export failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())