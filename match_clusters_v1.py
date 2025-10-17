#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import pandas as pd

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    from math import radians, sin, cos, asin, sqrt
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    return 2 * R * asin(sqrt(a))

def time_overlap(a_start, a_end, b_start, b_end):
    latest_start = max(a_start, b_start)
    earliest_end = min(a_end, b_end)
    return max(0.0, earliest_end - latest_start)

def match_clusters(circle_df, alt_df, eps_m=2000.0, min_ovl_frac=0.20, max_time_gap=900.0):
    """
    Return (matches_df, candidate_pairs_count)
    """
    # Work on local copies to avoid SettingWithCopyWarning
    c = circle_df.copy()
    a = alt_df.copy()

    # Normalize dtypes safely
    for d in (c, a):
        for col in ("lat","lon","t_start","t_end","alt_gain_m","duration_s","climb_rate_ms","cluster_id"):
            if col in d.columns:
                d.loc[:, col] = pd.to_numeric(d[col], errors="coerce")

    matches = []
    cand_count = 0

    for _, crow in c.iterrows():
        clat = float(crow.get("lat", float("nan")))
        clon = float(crow.get("lon", float("nan")))
        ct0  = float(crow.get("t_start", float("nan")))
        ct1  = float(crow.get("t_end", float("nan")))
        cid  = int(crow.get("cluster_id", 0))

        for _, arow in a.iterrows():
            alat = float(arow.get("lat", float("nan")))
            alon = float(arow.get("lon", float("nan")))
            at0  = float(arow.get("t_start", float("nan")))
            at1  = float(arow.get("t_end", float("nan")))
            aid  = int(arow.get("cluster_id", 0))

            d = haversine_m(clat, clon, alat, alon)
            if d > eps_m:
                continue

            dt_gap = abs(ct0 - at0)
            if dt_gap > max_time_gap:
                continue

            ovl = time_overlap(ct0, ct1, at0, at1)
            dur_short = min(max(ct1 - ct0, 0.0), max(at1 - at0, 0.0))
            frac = (ovl / dur_short) if dur_short > 0 else 0.0
            cand_count += 1
            if frac < min_ovl_frac:
                continue

            matches.append({
                "circle_cluster_id": cid,
                "alt_cluster_id":    aid,
                "dist_m":            float(d),
                "time_overlap_s":    float(ovl),
                "overlap_frac":      float(frac),
                "circle_lat": clat, "circle_lon": clon,
                "alt_lat":    alat, "alt_lon":    alon,
                "lat": (clat + alat)/2.0,
                "lon": (clon + alon)/2.0,
                "alt_gain_m":    float(arow.get("alt_gain_m", float("nan"))),
                "duration_s":    float(arow.get("duration_s", float("nan"))),
                "climb_rate_ms": float(arow.get("climb_rate_ms", float("nan"))),
            })

    return pd.DataFrame(matches), cand_count
def main():
    import argparse, json
    from pathlib import Path

    ap = argparse.ArgumentParser(
        description="Match circle clusters to altitude clusters and write matched_clusters.csv (+ .json)."
    )
    ap.add_argument("circles", nargs="?", help="Path to circle_clusters_enriched.csv")
    ap.add_argument("alts", nargs="?", help="Path to altitude_clusters.csv")
    ap.add_argument("--out", help="Output CSV path (default: <circles_dir>/matched_clusters.csv)")
    # Optional tuning flags (keep defaults same as code)
    ap.add_argument("--max-dist-m", type=float, default=600.0, help="Max centroid distance (m)")
    ap.add_argument("--min-overlap", type=float, default=0.25, help="Min temporal overlap fraction [0..1]")
    args = ap.parse_args()

    # ---- Resolve inputs (prompt if missing) ----
    if args.circles:
        circles_path = Path(args.circles)
    else:
        cin = input("Enter path to circle_clusters_enriched.csv (required): ").strip()
        circles_path = Path(cin)

    if args.alts:
        alts_path = Path(args.alts)
    else:
        ain = input("Enter path to altitude_clusters.csv (required): ").strip()
        alts_path = Path(ain)

    if not circles_path.exists():
        print(f"[ERROR] circles not found: {circles_path}")
        return 2
    if not alts_path.exists():
        print(f"[ERROR] altitude clusters not found: {alts_path}")
        return 2

    # ---- Output path (defaults beside circles) ----
    out_csv = Path(args.out) if args.out else (circles_path.parent / "matched_clusters.csv")
    out_json = out_csv.with_suffix(".json")

    # ---- Load inputs ----
    circles = pd.read_csv(circles_path)
    alts    = pd.read_csv(alts_path)
    print(f"[INFO] circles: {len(circles)} rows | alts: {len(alts)} rows")

    # ---- Minimal normalization for downstream expectations ----
    # circles need: cluster_id, lat, lon, t_start, t_end
    if "cluster_id" not in circles.columns:
        circles = circles.reset_index(drop=True)
        circles["cluster_id"] = circles.index
    missing_c = [c for c in ["lat","lon","t_start","t_end"] if c not in circles.columns]
    if missing_c:
        print(f"[ERROR] circles missing required columns: {missing_c}")
        return 2

    # alts should have: (optional cluster_id), t_start,t_end,lat,lon,alt_gain_m,duration_s,climb_rate_ms
    if "cluster_id" not in alts.columns:
        alts = alts.reset_index(drop=True)
        alts["cluster_id"] = alts.index
    # derive climb_rate if possible
    if "climb_rate_ms" not in alts.columns and {"alt_gain_m","duration_s"} <= set(alts.columns):
        alts["climb_rate_ms"] = alts["alt_gain_m"] / alts["duration_s"].replace(0, pd.NA)
    missing_a = [c for c in ["t_start","t_end","lat","lon","alt_gain_m","duration_s","climb_rate_ms"] if c not in alts.columns]
    if missing_a:
        print(f"[ERROR] altitude clusters missing required columns: {missing_a}")
        return 2

    # ---- Match ----

    matched, stats = match_clusters(
        circles[["cluster_id","lat","lon","t_start","t_end"]],
        alts[["cluster_id","t_start","t_end","lat","lon","alt_gain_m","duration_s","climb_rate_ms"]],
        args.max_dist_m,
        args.min_overlap,
    )

    # Canonical column order
    match_cols = [
        "lat", "lon", "climb_rate_ms", "alt_gain_m", "duration_s",
        "circle_cluster_id", "alt_cluster_id",
        "circle_lat", "circle_lon", "alt_lat", "alt_lon",
        "dist_m", "time_overlap_s", "overlap_frac"
    ]
    if not matched.empty:
        matched = matched.reindex(columns=match_cols)
    else:
        matched = pd.DataFrame(columns=match_cols)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(out_csv, index=False)

    payload = {"stats": stats, "rows": int(len(matched))}
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[OK] wrote {len(matched)} matches → {out_csv}")
    print(f"[OK] wrote stats → {out_json}")
    return 0
if __name__ == "__main__":
    main()

