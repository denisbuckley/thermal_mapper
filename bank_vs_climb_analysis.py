
#!/usr/bin/env python3
"""
bank_vs_climb_analysis.py

Analyze correlation between climb rate (m/s) and bank angle (deg)
from per-circle outputs produced by your glider pipeline.

It will:
  1) Discover and concatenate all circles.csv across your batch outputs
     (or read them from a manifest CSV).
  2) Clean the data with configurable filters.
  3) Compute Pearson & Spearman correlations (r, p).
  4) Fit a simple linear regression: climb_rate_ms ~ bank_angle_deg
     (using scipy.stats.linregress).
  5) Produce binned summaries by bank angle (mean/median/P95).
  6) Save a summary CSV/JSON and two plots (scatter+fit, hexbin+binned means).

USAGE (examples):
  # auto-discover circles.csv under ~/PycharmProjects/chatgpt_igc/outputs/batch
  python bank_vs_climb_analysis.py

  # specify project root explicitly
  python bank_vs_climb_analysis.py --root ~/PycharmProjects/chatgpt_igc

  # use a manifest (batch.csv) listing flight subfolders (one per line or a CSV
  # with a 'flight_dir' column); circles expected at <root>/<flight_dir>/circles.csv
  python bank_vs_climb_analysis.py --batch-manifest ~/PycharmProjects/chatgpt_igc/outputs/batch.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# scipy is commonly available; use it for correlation & regression
from scipy import stats
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# ----------------------------- CLI --------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Climb rate vs bank angle correlation on circles.csv batch outputs.")
    parser.add_argument("--root", type=str, default="~/PycharmProjects/chatgpt_igc",
                        help="Project root where outputs/batch/*/circles.csv live. Default: %(default)s")
    parser.add_argument("--batch-manifest", type=str, default=None,
                        help="Optional path to batch.csv (manifest). If provided, use it to locate circles.csv files.")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Output directory. Default: <root>/outputs/stats/bank_vs_climb")

    # Filtering
    parser.add_argument("--min-bank", type=float, default=0.0, help="Minimum bank angle to include (deg).")
    parser.add_argument("--max-bank", type=float, default=70.0, help="Maximum bank angle to include (deg).")
    parser.add_argument("--min-climb", type=float, default=-2.0, help="Minimum climb rate to include (m/s).")
    parser.add_argument("--max-climb", type=float, default=10.0, help="Maximum climb rate to include (m/s).")
    parser.add_argument("--drop-na", action="store_true", help="Drop rows with NA in required columns (on by default).")
    parser.add_argument("--no-drop-na", dest="drop_na", action="store_false", help="Disable dropping NA.")
    parser.set_defaults(drop_na=True)

    # Outlier control
    parser.add_argument("--iqr-filter", action="store_true",
                        help="Apply IQR filter on climb_rate_ms and bank_angle_deg to remove extreme outliers.")
    parser.add_argument("--iqr-multiplier", type=float, default=3.0, help="IQR multiplier (default 3.0).")

    # Binning
    parser.add_argument("--bin-width", type=float, default=5.0, help="Bank-angle bin width in degrees.")

    # Hexbin density plot options
    parser.add_argument("--hexbin-gridsize", type=int, default=50, help="Hexbin grid size for density plot.")

    # Save merged data
    parser.add_argument("--save-merged", action="store_true", help="Save merged circles dataframe as CSV for auditing.")

    args = parser.parse_args()
    return args


# ----------------------------- Discovery --------------------------------
def find_circles_via_glob(root: Path) -> List[Path]:
    # your layout: <root>/outputs/waypoints/batch_csv/**/circles.csv
    patt = root.expanduser() / "outputs" /"batch_csv"
    paths = list(patt.glob("**/circles.csv"))
    return sorted(paths)


def find_circles_via_manifest(root: Path, manifest: Path) -> List[Path]:
    manifest = manifest.expanduser()
    if not manifest.exists():
        print(f"[WARN] Manifest not found: {manifest}. Falling back to glob discovery.", file=sys.stderr)
        return find_circles_via_glob(root)

    # Accept either:
    #   - CSV with 'flight_dir' column
    #   - Plain text with one path per line (relative to root or absolute)
    circles_paths: List[Path] = []
    try:
        # Try CSV first
        dfm = pd.read_csv(manifest)
        if "flight_dir" in dfm.columns:
            for s in dfm["flight_dir"].dropna().astype(str):
                p = (root.expanduser() / s / "circles.csv").resolve()
                circles_paths.append(p)
        else:
            # If it's CSV but no expected column, try first column as path
            first_col = dfm.columns[0]
            for s in dfm[first_col].dropna().astype(str):
                p = Path(s)
                if not p.is_absolute():
                    p = (root.expanduser() / s)
                if p.is_dir():
                    p = p / "circles.csv"
                circles_paths.append(p.resolve())
    except Exception:
        # Treat as plain text
        with open(manifest, "r", encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                p = Path(s)
                if not p.is_absolute():
                    p = (root.expanduser() / s)
                if p.is_dir():
                    p = p / "circles.csv"
                circles_paths.append(p.resolve())

    # keep only existing
    circles_paths = [p for p in circles_paths if p.exists()]
    return sorted(circles_paths)


# ----------------------------- Data Loading --------------------------------
REQUIRED_COLS = ["climb_rate_ms", "turn_radius_m"]

def load_circles(paths: List[Path]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            missing = [c for c in REQUIRED_COLS if c not in df.columns]
            if missing:
                print(f"[WARN] Missing columns {missing} in {p}; skipping.", file=sys.stderr)
                continue
            # add source flight folder id for auditing
            df["source_file"] = str(p)
            dfs.append(df[REQUIRED_COLS + ["source_file"]].copy())
        except Exception as e:
            print(f"[WARN] Failed to read {p}: {e}", file=sys.stderr)
    if not dfs:
        raise SystemExit("No valid circles.csv files found.")
    big = pd.concat(dfs, ignore_index=True)
    return big


# ----------------------------- Cleaning --------------------------------
def iqr_bounds(s: pd.Series, k: float) -> Tuple[float, float]:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def clean_df(df: pd.DataFrame, args) -> pd.DataFrame:
    out = df.copy()
    if args.drop_na:
        out = out.dropna(subset=REQUIRED_COLS)

    # hard bounds first (domain sanity)
    out = out[(out["turn_radius_m"] >= args.min_bank) & (out["turn_radius_m"] <= args.max_bank)]
    out = out[(out["climb_rate_ms"] >= args.min_climb) & (out["climb_rate_ms"] <= args.max_climb)]

    if args.iqr_filter and not out.empty:
        lo_ba, hi_ba = iqr_bounds(out["turn_radius_m"], args.iqr_multiplier)
        lo_cr, hi_cr = iqr_bounds(out["climb_rate_ms"], args.iqr_multiplier)
        out = out[(out["turn_radius_m"] >= lo_ba) & (out["turn_radius_m"] <= hi_ba)]
        out = out[(out["climb_rate_ms"] >= lo_cr)   & (out["climb_rate_ms"] <= hi_cr)]

    return out.reset_index(drop=True)


# ----------------------------- Stats --------------------------------
@dataclass
class CorrelationResult:
    method: str
    r: float
    p: float
    n: int


@dataclass
class LinRegResult:
    slope: float
    intercept: float
    r: float
    p: float
    stderr: float
    r2: float
    slope_ci95: Tuple[float, float]
    intercept_ci95: Tuple[float, float]


def compute_correlations(df: pd.DataFrame) -> List[CorrelationResult]:
    x = df["turn_radius_m"].to_numpy()
    y = df["climb_rate_ms"].to_numpy()

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_rho, spearman_p = stats.spearmanr(x, y)

    return [
        CorrelationResult("pearson", float(pearson_r), float(pearson_p), len(df)),
        CorrelationResult("spearman", float(spearman_rho), float(spearman_p), len(df)),
    ]


def linreg_with_ci(df: pd.DataFrame) -> LinRegResult:
    x = df["turn_radius_m"].to_numpy()
    y = df["climb_rate_ms"].to_numpy()

    lr = stats.linregress(x, y)
    r2 = lr.rvalue ** 2

    # 95% CI for slope & intercept
    # Following standard formulas:
    n = len(x)
    x_mean = x.mean()
    s_xx = np.sum((x - x_mean) ** 2)
    y_hat = lr.intercept + lr.slope * x
    residuals = y - y_hat
    dof = max(n - 2, 1)
    s2 = np.sum(residuals ** 2) / dof
    se_slope = math.sqrt(s2 / s_xx) if s_xx > 0 else float("nan")
    se_intercept = math.sqrt(s2 * (1.0 / n + (x_mean ** 2) / s_xx)) if s_xx > 0 else float("nan")
    tcrit = stats.t.ppf(0.975, dof) if dof > 0 else float("nan")
    slope_ci = (lr.slope - tcrit * se_slope, lr.slope + tcrit * se_slope) if dof > 0 else (float("nan"), float("nan"))
    intercept_ci = (lr.intercept - tcrit * se_intercept, lr.intercept + tcrit * se_intercept) if dof > 0 else (float("nan"), float("nan"))

    return LinRegResult(
        slope=float(lr.slope),
        intercept=float(lr.intercept),
        r=float(lr.rvalue),
        p=float(lr.pvalue),
        stderr=float(lr.stderr),
        r2=float(r2),
        slope_ci95=(float(slope_ci[0]), float(slope_ci[1])),
        intercept_ci95=(float(intercept_ci[0]), float(intercept_ci[1])),
    )


def binned_stats(df: pd.DataFrame, bin_width: float = 5.0) -> pd.DataFrame:
    # bins like [0-5), [5-10), ...
    min_ba = np.floor(df["turn_radius_m"].min() / bin_width) * bin_width
    max_ba = np.ceil(df["turn_radius_m"].max() / bin_width) * bin_width
    bins = np.arange(min_ba, max_ba + bin_width, bin_width)
    labels = [f"[{bins[i]:.0f},{bins[i+1]:.0f})" for i in range(len(bins) - 1)]
    cat = pd.cut(df["turn_radius_m"], bins=bins, labels=labels, include_lowest=True, right=False)

    g = df.groupby(cat)["climb_rate_ms"]
    out = pd.DataFrame({
        "bank_bin": labels
    })
    # align by index
    agg = g.agg(["count", "mean", "median", lambda s: np.percentile(s.dropna(), 95)])
    agg = agg.rename(columns={"<lambda_0>": "p95"}).reindex(labels)
    out = out.set_index("bank_bin").join(agg)
    out = out.reset_index()
    return out


# ----------------------------- Plotting --------------------------------
def plot_scatter_with_fit(df: pd.DataFrame, lin: LinRegResult, out_png: Path):
    x = df["turn_radius_m"].to_numpy()
    y = df["climb_rate_ms"].to_numpy()

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=8, alpha=0.35)
    # regression line
    xgrid = np.linspace(x.min(), x.max(), 200)
    yhat = lin.intercept + lin.slope * xgrid
    plt.plot(xgrid, yhat, linewidth=2)

    plt.title("Climb rate vs Bank angle (scatter + linear fit)")
    plt.xlabel("Bank angle (deg)")
    plt.ylabel("Climb rate (m/s)")
    subtitle = f"slope={lin.slope:.4f} m/s/deg, R²={lin.r2:.3f}, n={len(df)}"
    plt.suptitle(subtitle, y=0.94, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_hexbin_with_bins(df: pd.DataFrame, binned: pd.DataFrame, out_png: Path, gridsize: int = 50):
    x = df["turn_radius_m"].to_numpy()
    y = df["climb_rate_ms"].to_numpy()

    plt.figure(figsize=(8, 6))
    hb = plt.hexbin(x, y, gridsize=gridsize, mincnt=1)
    plt.colorbar(hb, label="count")
    plt.title("Climb rate vs Bank angle (hexbin density + binned P95)")
    plt.xlabel("Bank angle (deg)")
    plt.ylabel("Climb rate (m/s)")

    # overlay binned P95 as line at bin midpoints
    def bin_mid(b):
        # "[a,b)" -> midpoint
        a = float(b.split(",")[0][1:])
        b2 = float(b.split(",")[1][:-1])
        return (a + b2) / 2.0

    mids = binned["bank_bin"].dropna().map(bin_mid).to_numpy()
    p95s = binned["p95"].to_numpy()
    plt.plot(mids, p95s, linewidth=2)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# ----------------------------- Main --------------------------------
def main():
    args = parse_args()
    root = Path(args.root).expanduser()
    outdir = Path(args.outdir) if args.outdir else (root / "outputs" / "stats" / "bank_vs_climb")
    outdir.mkdir(parents=True, exist_ok=True)

    # Discover circles.csv files
    if args.batch_manifest:
        circles_paths = find_circles_via_manifest(root, Path(args.batch_manifest))
    else:
        circles_paths = find_circles_via_glob(root)

    if not circles_paths:
        raise SystemExit(f"No circles.csv files found under {root} (or via manifest).")

    print(f"[INFO] Found {len(circles_paths)} circles.csv files.")
    big = load_circles(circles_paths)
    print(f"[INFO] Loaded {len(big)} circle rows.")

    cleaned = clean_df(big, args)
    print(f"[INFO] After filtering: {len(cleaned)} rows remain.")

    if args.save_merged:
        merged_csv = outdir / "merged_circles_cleaned.csv"
        cleaned.to_csv(merged_csv, index=False)
        print(f"[OK] Saved merged cleaned data -> {merged_csv}")

    if cleaned.empty:
        raise SystemExit("No data after filtering; adjust filters and try again.")

    # Stats
    cors = compute_correlations(cleaned)
    lin = linreg_with_ci(cleaned)
    bins = binned_stats(cleaned, bin_width=args.bin_width)

    # Save summaries
    summary_rows = []
    for c in cors:
        summary_rows.append({
            "metric": f"corr_{c.method}",
            "r": c.r,
            "p": c.p,
            "n": c.n
        })
    summary_rows.append({
        "metric": "linreg",
        "slope": lin.slope,
        "intercept": lin.intercept,
        "r": lin.r,
        "p": lin.p,
        "stderr": lin.stderr,
        "r2": lin.r2,
        "slope_ci95_lo": lin.slope_ci95[0],
        "slope_ci95_hi": lin.slope_ci95[1],
        "intercept_ci95_lo": lin.intercept_ci95[0],
        "intercept_ci95_hi": lin.intercept_ci95[1],
        "n": len(cleaned),
    })
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = outdir / "bank_vs_climb_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    # JSON too
    summary_json = {
        "correlations": [{ "method": c.method, "r": c.r, "p": c.p, "n": c.n } for c in cors],
        "linreg": {
            "slope": lin.slope,
            "intercept": lin.intercept,
            "r": lin.r,
            "p": lin.p,
            "stderr": lin.stderr,
            "r2": lin.r2,
            "slope_ci95": list(lin.slope_ci95),
            "intercept_ci95": list(lin.intercept_ci95),
            "n": len(cleaned)
        }
    }
    with open(outdir / "bank_vs_climb_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary_json, fh, indent=2)

    # Save bins
    bins_csv = outdir / "bank_binned_stats.csv"
    bins.to_csv(bins_csv, index=False)

    # Plots
    scatter_png = outdir / "scatter_fit.png"
    plot_scatter_with_fit(cleaned, lin, scatter_png)

    hexbin_png = outdir / "hexbin_binned_p95.png"
    plot_hexbin_with_bins(cleaned, bins, hexbin_png, gridsize=args.hexbin_gridsize)

    # Console summary
    print("\n=== RESULTS ===")
    for c in cors:
        print(f"{c.method.title()} r={c.r:.4f}, p={c.p:.3e}, n={c.n}")
    print(f"LinReg: slope={lin.slope:.4f} m/s/deg (95% CI {lin.slope_ci95[0]:.4f}..{lin.slope_ci95[1]:.4f}), "
          f"intercept={lin.intercept:.3f}, R^2={lin.r2:.4f}, p={lin.p:.3e}, n={len(cleaned)}")
    print(f"\n[OK] Wrote summary -> {summary_csv}")
    print(f"[OK] Wrote JSON    -> {outdir / 'bank_vs_climb_summary.json'}")
    print(f"[OK] Wrote bins    -> {bins_csv}")
    print(f"[OK] Plots         -> {scatter_png} and {hexbin_png}")
    print("\nDone.")


if __name__ == "__main__":
    main()
