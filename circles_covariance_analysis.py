
#!/usr/bin/env python3
"""
circles_covariance_analysis.py

Flexible analysis over per-circle outputs from your glider pipeline.
Lets you choose (interactively or via CLI) which variables to analyze:
    - Correlations (Pearson & Spearman)
    - Simple linear regression (y ~ x) with 95% CI on slope/intercept
    - Binned summaries of y by x (count/mean/median/P95)
    - Scatter + best-fit line
    - Hexbin density + binned P95 overlay

Defaults are tailored to your layout:
    <root>/outputs/waypoints/batch_csv/**/circles.csv

USAGE (examples):
  # Interactive variable selection
  python circles_covariance_analysis.py

  # Non-interactive: pick variables explicitly
  python circles_covariance_analysis.py --x turn_radius_m --y climb_rate_ms

  # Specify project root and save merged cleaned data
  python circles_covariance_analysis.py --root ~/PycharmProjects/chatgpt_igc --save-merged

  # Use a manifest (batch.csv) listing flight dirs or paths (1st column 'flight_dir' or paths)
  python circles_covariance_analysis.py --batch-manifest ~/PycharmProjects/chatgpt_igc/outputs/batch.csv
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------- CLI --------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Flexible covariance/regression on circles.csv data.")
    parser.add_argument("--root", type=str, default="~/PycharmProjects/chatgpt_igc",
                        help="Project root. Will search <root>/outputs/waypoints/batch_csv/**/circles.csv by default.")
    parser.add_argument("--batch-manifest", type=str, default=None,
                        help="Optional path to batch.csv listing flight dirs/paths. If present, used to locate circles.csv.")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Output directory. Default: <root>/outputs/stats/circles_covariance")
    parser.add_argument("--x", type=str, default=None, help="X variable (predictor). If omitted, prompt interactively.")
    parser.add_argument("--y", type=str, default=None, help="Y variable (response). If omitted, prompt interactively.")
    parser.add_argument("--bin-width", type=float, default=None,
                        help="Bin width for X (if omitted, auto-choose via Freedman–Diaconis).")
    parser.add_argument("--bins", type=int, default=None,
                        help="Alternatively specify number of bins for X (overrides bin-width if set).")
    parser.add_argument("--hexbin-gridsize", type=int, default=50, help="Hexbin grid size.")
    parser.add_argument("--save-merged", action="store_true", help="Save merged cleaned dataset CSV.")
    parser.add_argument("--drop-na", action="store_true", help="Drop NA in chosen columns (default True).")
    parser.add_argument("--no-drop-na", dest="drop_na", action="store_false")
    parser.set_defaults(drop_na=True)

    # Generic optional hard filters that apply IF those columns exist
    parser.add_argument("--min-climb", type=float, default=-2.0, help="If 'climb_rate_ms' exists, enforce minimal value.")
    parser.add_argument("--max-climb", type=float, default=10.0, help="If 'climb_rate_ms' exists, enforce maximal value.")
    parser.add_argument("--min-bank", type=float, default=0.0, help="If 'bank_angle_deg' exists, enforce minimal value.")
    parser.add_argument("--max-bank", type=float, default=70.0, help="If 'bank_angle_deg' exists, enforce maximal value.")
    parser.add_argument("--min-radius", type=float, default=5.0, help="If 'turn_radius_m' exists, enforce minimal value.")
    parser.add_argument("--max-radius", type=float, default=200.0, help="If 'turn_radius_m' exists, enforce maximal value.")

    # IQR outlier filter (optional), applied to selected x & y only
    parser.add_argument("--iqr-filter", action="store_true", help="Remove outliers on X & Y using IQR*k.")
    parser.add_argument("--iqr-multiplier", type=float, default=3.0, help="k for IQR fence (default 3.0).")

    return parser.parse_args()


# ----------------------------- Discovery --------------------------------
def find_circles_via_glob(root: Path) -> List[Path]:
    patt = root.expanduser() / "outputs" / "batch_csv"
    paths = list(patt.glob("**/circles.csv"))
    return sorted(paths)


def find_circles_via_manifest(root: Path, manifest: Path) -> List[Path]:
    manifest = manifest.expanduser()
    if not manifest.exists():
        print(f"[WARN] Manifest not found: {manifest}. Falling back to glob discovery.", file=sys.stderr)
        return find_circles_via_glob(root)

    circles_paths: List[Path] = []
    try:
        dfm = pd.read_csv(manifest)
        if "flight_dir" in dfm.columns:
            for s in dfm["flight_dir"].dropna().astype(str):
                p = (root.expanduser() / s / "circles.csv").resolve()
                circles_paths.append(p)
        else:
            first_col = dfm.columns[0]
            for s in dfm[first_col].dropna().astype(str):
                p = Path(s)
                if not p.is_absolute():
                    p = (root.expanduser() / s)
                if p.is_dir():
                    p = p / "circles.csv"
                circles_paths.append(p.resolve())
    except Exception:
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

    circles_paths = [p for p in circles_paths if p.exists()]
    return sorted(circles_paths)


# ----------------------------- Load & Clean --------------------------------
def load_all(paths: List[Path]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df["source_file"] = str(p)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Failed to read {p}: {e}", file=sys.stderr)
    if not dfs:
        raise SystemExit("No valid circles.csv files found.")
    big = pd.concat(dfs, ignore_index=True)
    return big


def apply_domain_filters(df: pd.DataFrame, args) -> pd.DataFrame:
    out = df.copy()
    # Apply column-specific hard filters only if present
    if "climb_rate_ms" in out.columns:
        out = out[(out["climb_rate_ms"] >= args.min_climb) & (out["climb_rate_ms"] <= args.max_climb)]
    if "bank_angle_deg" in out.columns:
        out = out[(out["bank_angle_deg"] >= args.min_bank) & (out["bank_angle_deg"] <= args.max_bank)]
    if "turn_radius_m" in out.columns:
        out = out[(out["turn_radius_m"] >= args.min_radius) & (out["turn_radius_m"] <= args.max_radius)]
    return out


def iqr_bounds(s: pd.Series, k: float):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def iqr_filter_xy(df: pd.DataFrame, x: str, y: str, k: float) -> pd.DataFrame:
    out = df.copy()
    lo_x, hi_x = iqr_bounds(out[x], k)
    lo_y, hi_y = iqr_bounds(out[y], k)
    return out[(out[x] >= lo_x) & (out[x] <= hi_x) & (out[y] >= lo_y) & (out[y] <= hi_y)].reset_index(drop=True)


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


def compute_correlations(df: pd.DataFrame, x: str, y: str) -> List[CorrelationResult]:
    xx = df[x].to_numpy()
    yy = df[y].to_numpy()
    pearson_r, pearson_p = stats.pearsonr(xx, yy)
    spearman_rho, spearman_p = stats.spearmanr(xx, yy)
    return [
        CorrelationResult("pearson", float(pearson_r), float(pearson_p), len(df)),
        CorrelationResult("spearman", float(spearman_rho), float(spearman_p), len(df)),
    ]


def linreg_with_ci(df: pd.DataFrame, x: str, y: str) -> LinRegResult:
    xx = df[x].to_numpy()
    yy = df[y].to_numpy()
    lr = stats.linregress(xx, yy)
    r2 = lr.rvalue ** 2

    n = len(xx)
    x_mean = xx.mean()
    s_xx = np.sum((xx - x_mean) ** 2)
    y_hat = lr.intercept + lr.slope * xx
    residuals = yy - y_hat
    dof = max(n - 2, 1)
    s2 = np.sum(residuals ** 2) / dof
    se_slope = math.sqrt(s2 / s_xx) if s_xx > 0 else float("nan")
    se_intercept = math.sqrt(s2 * (1.0 / n + (x_mean ** 2) / s_xx)) if s_xx > 0 else float("nan")
    tcrit = stats.t.ppf(0.975, dof) if dof > 0 else float("nan")
    slope_ci = (lr.slope - tcrit * se_slope, lr.slope + tcrit * se_slope) if dof > 0 else (float("nan"), float("nan"))
    intercept_ci = (lr.intercept - tcrit * se_intercept) if dof > 0 else float("nan"), (lr.intercept + tcrit * se_intercept) if dof > 0 else float("nan")

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


# ----------------------------- Binning --------------------------------
def freedman_diaconis_bin_width(x: np.ndarray) -> float:
    if len(x) < 2:
        return 1.0
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        return max((x.max() - x.min()) / 10.0, 1.0)
    h = 2 * iqr * (len(x) ** (-1/3))
    if h <= 0:
        return max((x.max() - x.min()) / 10.0, 1.0)
    return h


def make_bins(x: np.ndarray, bin_width: Optional[float], bins: Optional[int]):
    if bins is not None and bins > 0:
        edges = np.linspace(x.min(), x.max(), bins + 1)
        return edges
    bw = bin_width if (bin_width is not None and bin_width > 0) else freedman_diaconis_bin_width(x)
    start = np.floor(x.min() / bw) * bw
    stop = np.ceil(x.max() / bw) * bw
    edges = np.arange(start, stop + bw, bw)
    if len(edges) < 6:
        edges = np.linspace(start, stop, 6)
    return edges


def binned_stats(df: pd.DataFrame, x: str, y: str, bin_width: Optional[float], bins_ct: Optional[int]) -> pd.DataFrame:
    xx = df[x].to_numpy()
    edges = make_bins(xx, bin_width, bins_ct)
    labels = [f"[{edges[i]:.2f},{edges[i+1]:.2f})" for i in range(len(edges) - 1)]
    cat = pd.cut(df[x], bins=edges, labels=labels, include_lowest=True, right=False)

    g = df.groupby(cat, observed=True)[y]
    out = pd.DataFrame({"x_bin": labels})
    agg = g.agg([
        "count",
        "mean",
        "median",
        "std",
        lambda s: np.percentile(s.dropna(), 95) if s.dropna().size else np.nan
    ])
    agg = agg.rename(columns={"<lambda_0>": "p95"}).reindex(labels)
    out = out.set_index("x_bin").join(agg)
    out = out.reset_index().rename(columns={"index": "x_bin"})
    mids = 0.5 * (edges[:-1] + edges[1:])
    out["x_mid"] = mids
    return out


# ----------------------------- Plotting --------------------------------
def plot_scatter_with_fit(df: pd.DataFrame, x: str, y: str, lin: LinRegResult, out_png: Path):
    xx = df[x].to_numpy()
    yy = df[y].to_numpy()
    plt.figure(figsize=(8, 6))
    plt.scatter(xx, yy, s=8, alpha=0.35)
    xgrid = np.linspace(xx.min(), xx.max(), 200)
    yhat = lin.intercept + lin.slope * xgrid
    plt.plot(xgrid, yhat, linewidth=2)
    plt.title(f"{y} vs {x} (scatter + linear fit)")
    plt.xlabel(x)
    plt.ylabel(y)
    subtitle = f"slope={lin.slope:.4f} per {x}, R²={lin.r2:.3f}, n={len(df)}"
    plt.suptitle(subtitle, y=0.94, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_hexbin_with_bins(df: pd.DataFrame, x: str, y: str, binned: pd.DataFrame, out_png: Path, gridsize: int = 50):
    xx = df[x].to_numpy()
    yy = df[y].to_numpy()
    plt.figure(figsize=(8, 6))
    hb = plt.hexbin(xx, yy, gridsize=gridsize, mincnt=1)
    plt.colorbar(hb, label="count")
    plt.title(f"{y} vs {x} (hexbin density + binned P95)")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.plot(binned["x_mid"].to_numpy(), binned["p95"].to_numpy(), linewidth=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# ----------------------------- Main --------------------------------
def choose_var_interactively(df: pd.DataFrame) -> Tuple[str, str]:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    priority_order = [
        "climb_rate_ms", "bank_angle_deg", "turn_radius_m",
        "alt_gain_m", "duration_s", "avg_speed_kmh"
    ]
    numeric = sorted(numeric, key=lambda c: (priority_order.index(c) if c in priority_order else 999, c))

    print("\nAvailable numeric columns:")
    for i, c in enumerate(numeric):
        print(f"  [{i}] {c}")

    def ask_idx(prompt, default_idx=None):
        while True:
            raw = input(f"{prompt} (index or name){' ['+str(default_idx)+']' if default_idx is not None else ''}: ").strip()
            if not raw and default_idx is not None:
                return numeric[default_idx]
            if raw.isdigit():
                idx = int(raw)
                if 0 <= idx < len(numeric):
                    return numeric[idx]
            elif raw in df.columns:
                return raw
            print("  Invalid choice. Try again.")

    x = ask_idx("Choose X (predictor)", default_idx=numeric.index("bank_angle_deg") if "bank_angle_deg" in numeric else 0)
    y = ask_idx("Choose Y (response)", default_idx=numeric.index("climb_rate_ms") if "climb_rate_ms" in numeric else 0)
    print(f"Selected: X={x}, Y={y}")
    return x, y


def main():
    args = parse_args()
    root = Path(args.root).expanduser()
    outdir = Path(args.outdir) if args.outdir else (root / "outputs" / "stats" / "circles_covariance")
    outdir.mkdir(parents=True, exist_ok=True)

    if args.batch_manifest:
        circles_paths = find_circles_via_manifest(root, Path(args.batch_manifest))
    else:
        circles_paths = find_circles_via_glob(root)

    if not circles_paths:
        raise SystemExit(f"No circles.csv files found under {root}/outputs/waypoints/batch_csv (or via manifest).")

    print(f"[INFO] Found {len(circles_paths)} circles.csv files.")
    big = load_all(circles_paths)
    print(f"[INFO] Loaded {len(big)} rows (all circles).")

    big = apply_domain_filters(big, args)

    x = args.x
    y = args.y
    if not x or not y:
        x, y = choose_var_interactively(big)

    for col in (x, y):
        if col not in big.columns:
            raise SystemExit(f"Column '{col}' not found in data. Available columns: {list(big.columns)}")

    cols = ["source_file"] if "source_file" in big.columns else []
    df = big[[x, y] + cols].copy()

    if args.drop_na:
        df = df.dropna(subset=[x, y]).reset_index(drop=True)

    if args.iqr_filter and not df.empty:
        before = len(df)
        df = iqr_filter_xy(df, x, y, args.iqr_multiplier)
        print(f"[INFO] IQR filtered {before-len(df)} rows; remaining {len(df)}.")

    if df.empty:
        raise SystemExit("No data left after filtering; relax filters and retry.")

    # --- Global standard deviations ---
    print(f"Global std dev of {x}: {df[x].std():.3f}")
    print(f"Global std dev of {y}: {df[y].std():.3f}")

    # Stats
    cors = compute_correlations(df, x, y)
    lin = linreg_with_ci(df, x, y)
    bins = binned_stats(df, x, y, args.bin_width, args.bins)

    summary = {
        "x": x,
        "y": y,
        "n": int(len(df)),
        "correlations": [{"method": c.method, "r": c.r, "p": c.p, "n": c.n} for c in cors],
        "linreg": {
            "slope": lin.slope,
            "intercept": lin.intercept,
            "r": lin.r,
            "p": lin.p,
            "stderr": lin.stderr,
            "r2": lin.r2,
            "slope_ci95": list(lin.slope_ci95),
            "intercept_ci95": list(lin.intercept_ci95),
            "n": len(df),
        },
    }
    summary_json = outdir / f"summary_{y}_vs_{x}.json"
    with open(summary_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    summary_csv = outdir / f"summary_{y}_vs_{x}.csv"
    pd.DataFrame([summary["linreg"]]).to_csv(summary_csv, index=False)

    bins_csv = outdir / f"binned_{y}_vs_{x}.csv"
    bins.to_csv(bins_csv, index=False)

    scatter_png = outdir / f"scatter_{y}_vs_{x}.png"
    plot_scatter_with_fit(df, x, y, lin, scatter_png)

    hexbin_png = outdir / f"hexbin_{y}_vs_{x}.png"
    plot_hexbin_with_bins(df, x, y, bins, hexbin_png, gridsize=args.hexbin_gridsize)

    if args.save_merged:
        merged_csv = outdir / f"merged_{y}_vs_{x}.csv"
        df.to_csv(merged_csv, index=False)

    print("\n=== RESULTS ===")
    for c in cors:
        print(f"{c.method.title()} r={c.r:.4f}, p={c.p:.3e}, n={c.n}")
    print(f"LinReg: slope={lin.slope:.4f} per {x} (95% CI {lin.slope_ci95[0]:.4f}..{lin.slope_ci95[1]:.4f}), "
          f"intercept={lin.intercept:.3f}, R^2={lin.r2:.4f}, p={lin.p:.3e}, n={len(df)}")
    print(f"[OK] Wrote {summary_json.name}, {summary_csv.name}, {bins_csv.name}, plots:")
    print(f"     {scatter_png.name}, {hexbin_png.name}")
    print("Done.")


if __name__ == "__main__":
    main()
