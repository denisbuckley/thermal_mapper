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
    matches = []
    cand_count = 0
    for _, crow in circle_df.iterrows():
        for _, arow in alt_df.iterrows():
            d = haversine_m(crow["lat"], crow["lon"], arow["lat"], arow["lon"])
            if d > eps_m:
                continue
            dt_gap = abs(crow["t_start"] - arow["t_start"])
            if dt_gap > max_time_gap:
                continue
            ovl = time_overlap(crow["t_start"], crow["t_end"], arow["t_start"], arow["t_end"])
            dur_short = min(crow["t_end"] - crow["t_start"], arow["t_end"] - arow["t_start"])
            frac = (ovl / dur_short) if dur_short > 0 else 0.0
            cand_count += 1
            if frac >= min_ovl_frac:
                matches.append({
                    "circle_cluster_id": crow["cluster_id"],
                    "alt_cluster_id": arow["cluster_id"],
                    "dist_m": d,
                    "time_overlap_s": ovl,
                    "overlap_frac": frac
                })
    return pd.DataFrame(matches), cand_count

def main():
    import argparse, json
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("circles", nargs="?", help="Path to circle_clusters_enriched.csv")
    ap.add_argument("alts", nargs="?", help="Path to altitude_clusters.csv")
    ap.add_argument("--out", help="Output CSV (default: outputs/matched_clusters.csv)")
    args = ap.parse_args()

    # Defaults
    default_circles = Path("outputs/circle_clusters_enriched.csv")
    default_alts = Path("outputs/altitude_clusters.csv")
    default_out = Path("outputs/matched_clusters.csv")

    circles_path = Path(args.circles) if args.circles else Path("outputs/circle_clusters_enriched.csv")
    alts_path = Path(args.alts) if args.alts else Path("outputs/altitude_clusters.csv")
    out_path = Path(args.out) if args.out else Path("outputs/matched_clusters.csv")
    if not circles_path.exists() or not alts_path.exists():
        print("[ERROR] Missing required inputs. Ensure both circle and altitude cluster CSVs exist in outputs/.")
        return 1

    circles = pd.read_csv(circles_path)
    alts = pd.read_csv(alts_path)

    # 🔑 returns (matches_df, stats_dict)
    matches, stats = match_clusters(circles, alts)

    # --- Enrich with lat/lon from both inputs ---

    def _pick_cols(df, id_col_candidates=("cluster_id", "circle_cluster_id", "alt_cluster_id", "id"),
                   lat_candidates=("lat", "centroid_lat", "lat_c", "latitude"),
                   lon_candidates=("lon", "centroid_lon", "lon_c", "longitude")):
        """Return a minimal (id, lat, lon) frame, renaming to cluster_id/lat/lon as needed."""
        df2 = df.copy()
        # id
        for c in id_col_candidates:
            if c in df2.columns:
                if c != "cluster_id":
                    df2 = df2.rename(columns={c: "cluster_id"})
                break
        else:
            # synthesize if missing
            df2 = df2.reset_index(drop=True)
            df2["cluster_id"] = df2.index

        # lat/lon
        lat_col = next((c for c in lat_candidates if c in df2.columns), None)
        lon_col = next((c for c in lon_candidates if c in df2.columns), None)
        if lat_col is None or lon_col is None:
            # If coords absent, keep empty lat/lon to avoid KeyErrors
            df2["lat"] = pd.NA
            df2["lon"] = pd.NA
        else:
            if lat_col != "lat": df2 = df2.rename(columns={lat_col: "lat"})
            if lon_col != "lon": df2 = df2.rename(columns={lon_col: "lon"})
        return df2[["cluster_id", "lat", "lon"]].copy()

    # Normalize ID columns in matches to have canonical names for joining
    def _ensure_match_ids(m):
        if "circle_cluster_id" not in m.columns:
            # common aliases
            for c in ("circle_id", "circle_cluster", "circle"):
                if c in m.columns:
                    m = m.rename(columns={c: "circle_cluster_id"})
                    break
        if "alt_cluster_id" not in m.columns:
            for c in ("alt_id", "alt_cluster", "alt"):
                if c in m.columns:
                    m = m.rename(columns={c: "alt_cluster_id"})
                    break
        return m

    matches = _ensure_match_ids(matches)

    circle_coords = _pick_cols(circles)
    alt_coords = _pick_cols(alts)

    # Join circle coords
    enriched = matches.merge(circle_coords, left_on="circle_cluster_id", right_on="cluster_id", how="left",
                             suffixes=("", ""))
    enriched = enriched.rename(columns={"lat": "circle_lat", "lon": "circle_lon"}).drop(columns=["cluster_id"])

    # Join altitude coords
    enriched = enriched.merge(alt_coords, left_on="alt_cluster_id", right_on="cluster_id", how="left",
                              suffixes=("", ""))
    enriched = enriched.rename(columns={"lat": "alt_lat", "lon": "alt_lon"}).drop(columns=["cluster_id"])

    # Provide unified lat/lon (prefer circle centroid, fallback to altitude centroid)
    enriched["lat"] = enriched["circle_lat"].where(enriched["circle_lat"].notna(), enriched["alt_lat"])
    enriched["lon"] = enriched["circle_lon"].where(enriched["circle_lon"].notna(), enriched["alt_lon"])

    # Optional: also a simple mean of both, when both present
    enriched["lat_mean"] = pd.concat([enriched["circle_lat"], enriched["alt_lat"]], axis=1).mean(axis=1, skipna=True)
    enriched["lon_mean"] = pd.concat([enriched["circle_lon"], enriched["alt_lon"]], axis=1).mean(axis=1, skipna=True)

    # Save matches CSV (now enriched)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(out_path, index=False)
    print(f"[OK] wrote {len(enriched)} matches → {out_path}")

    # Save stats JSON alongside
    stats_path = out_path.with_suffix(".json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[OK] wrote stats → {stats_path}")

    return 0

if __name__ == "__main__":
    main()

