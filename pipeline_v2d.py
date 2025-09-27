#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pipeline_v2d.py
Self-contained pipeline: IGC → circles → circle clusters → altitude clusters → matches
Single prompt for IGC path. Thresholds for matching are module-level constants.
Outputs are saved under outputs/.
"""

import os
import math
import numpy as np
import pandas as pd

# ------------------- MATCHER PARAMETERS -------------------
EPS_M = 2000.0          # max spatial separation (m)
MIN_OVL_FRAC = 0.20     # min temporal overlap (fraction of shorter duration)
MAX_TIME_GAP_S = 900.0  # max start-time difference (s)

# ------------------- IGC PARSER -------------------
def parse_igc_brecords(path):
    """Parse B-records from an IGC file into a DataFrame with time, lat, lon, alt (GPS preferred)."""
    times, lats, lons, p_alts, g_alts = [], [], [], [], []
    day_offset = 0
    last_t = None
    with open(path, "r") as f:
        for line in f:
            if not line.startswith("B"):
                continue
            # Time HHMMSS, handle midnight rollover
            hh, mm, ss = int(line[1:3]), int(line[3:5]), int(line[5:7])
            t = hh * 3600 + mm * 60 + ss
            if last_t is not None and t < last_t:
                day_offset += 86400
            last_t = t
            t += day_offset

            # Latitude DDMMmmm + hemisphere
            lat_deg = int(line[7:9])
            lat_min = int(line[9:11])
            lat_thou = int(line[11:14])  # thousandths of minutes
            lat = lat_deg + (lat_min + lat_thou / 1000.0) / 60.0
            if line[14] == "S":
                lat = -lat

            # Longitude DDDMMmmm + hemisphere
            lon_deg = int(line[15:18])
            lon_min = int(line[18:20])
            lon_thou = int(line[20:23])
            lon = lon_deg + (lon_min + lon_thou / 1000.0) / 60.0
            if line[23] == "W":
                lon = -lon

            # Pressure alt [25:30], GPS alt [30:35]
            def _safe_int(s):
                try:
                    return int(s)
                except Exception:
                    return np.nan
            p_alt = _safe_int(line[25:30])
            g_alt = _safe_int(line[30:35])

            times.append(t)
            lats.append(lat)
            lons.append(lon)
            p_alts.append(p_alt)
            g_alts.append(g_alt)

    df = pd.DataFrame({
        "time_s": times,
        "lat": lats,
        "lon": lons,
        "alt_pressure": p_alts,
        "alt_gps": g_alts
    })
    # Prefer GPS when present; otherwise pressure
    df["alt_raw"] = df["alt_gps"].where(~pd.isna(df["alt_gps"]), df["alt_pressure"])

    # Robust smoothing: rolling median, MAD clip, rolling mean
    med = df["alt_raw"].rolling(window=5, center=True, min_periods=1).median()
    resid = df["alt_raw"] - med
    mad = np.nanmedian(np.abs(resid - np.nanmedian(resid)))
    if not np.isnan(mad) and mad > 0:
        clip = 5.0 * 1.4826 * mad
        clipped = np.clip(resid, -clip, clip) + med
    else:
        clipped = med.fillna(df["alt_raw"])
    df["alt_smooth"] = pd.Series(clipped).rolling(window=5, center=True, min_periods=1).mean()

    return df

# ------------------- GEOMETRY -------------------
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def bearing_deg(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(phi2)
    x = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0

def unwrap_angles(angs):
    out = [angs[0]]
    for a in angs[1:]:
        prev = out[-1]
        da = a - (prev % 360.0)
        if da > 180: da -= 360
        if da <= -180: da += 360
        out.append(prev + da)
    return np.array(out, dtype=float)

# ------------------- CIRCLES (from B) -------------------
def detect_circles(df, min_duration_s=6.0, max_duration_s=60.0, min_radius_m=8.0, max_radius_m=600.0, min_bank_deg=5.0, vmax_climb_ms=10.0):
    """Detect circles via cumulative heading rotation across B-fix bearings, with realism filters."""
    lat = df["lat"].to_numpy()
    lon = df["lon"].to_numpy()
    alt = df["alt_smooth"].to_numpy()
    t = df["time_s"].to_numpy()
    n = len(df)
    if n < 3:
        return pd.DataFrame(columns=[
            "circle_id","seg_id","t_start","t_end","duration_s",
            "avg_speed_kmh","alt_gained_m","climb_rate_ms",
            "turn_radius_m","bank_angle_deg","lat","lon"
        ])

    bearings = np.zeros(n)
    for i in range(1, n):
        bearings[i] = bearing_deg(lat[i-1], lon[i-1], lat[i], lon[i])
    uw = unwrap_angles(bearings)

    circles = []
    start_idx, circle_id, g = 0, 0, 9.81
    i = 1
    while i < n:
        rot = abs(uw[i] - uw[start_idx])
        if rot >= 360.0:
            i0, i1 = start_idx, i
            dur = t[i1] - t[i0]

            # Reject unrealistic circles
            if not (min_duration_s <= dur <= max_duration_s):
                start_idx = i
                i += 1
                continue

            # Distance & mean ground speed
            dist = 0.0
            for k in range(i0+1, i1+1):
                dist += haversine_m(lat[k-1], lon[k-1], lat[k], lon[k])
            v_mean = dist/dur if dur > 0 else np.nan

            # Turn radius / bank
            omega = 2*math.pi/dur
            radius = (v_mean/omega) if omega > 0 else np.nan
            if (np.isnan(radius)) or (radius < min_radius_m) or (radius > max_radius_m):
                start_idx = i
                i += 1
                continue

            bank = math.degrees(math.atan((v_mean**2)/(g*radius))) if (radius and np.isfinite(radius)) else np.nan
            if (np.isnan(bank)) or (bank < min_bank_deg):
                start_idx = i
                i += 1
                continue

            # Robust altitude gain by least-squares slope (reduces edge effects)
            idx = np.arange(i0, i1+1)
            tt = t[idx] - t[i0]
            aa = alt[idx]
            if len(tt) >= 3 and np.all(np.isfinite(tt)) and np.any(np.isfinite(aa)):
                A = np.vstack([tt, np.ones_like(tt)]).T
                mask = np.isfinite(aa)
                if mask.sum() >= 3:
                    m, _ = np.linalg.lstsq(A[mask], aa[mask], rcond=None)[0]
                    alt_gain = float(m * dur)
                else:
                    span = i1 - i0 + 1
                    q = max(1, span // 4)
                    first_med = np.nanmedian(aa[:q]); last_med  = np.nanmedian(aa[-q:])
                    alt_gain = (last_med - first_med) if np.isfinite(first_med) and np.isfinite(last_med) else np.nan
            else:
                alt_gain = np.nan

            # Physical plausibility bound
            max_gain = vmax_climb_ms * dur
            if np.isfinite(alt_gain) and abs(alt_gain) > max_gain:
                alt_gain = np.sign(alt_gain) * max_gain

            climb = (alt_gain / dur) if (dur > 0 and np.isfinite(alt_gain)) else np.nan

            circles.append({
                "circle_id": circle_id,
                "seg_id": None,
                "t_start": float(t[i0]),
                "t_end": float(t[i1]),
                "duration_s": float(dur),
                "avg_speed_kmh": float(v_mean*3.6) if np.isfinite(v_mean) else np.nan,
                "alt_gained_m": float(alt_gain) if np.isfinite(alt_gain) else np.nan,
                "climb_rate_ms": float(climb) if np.isfinite(climb) else np.nan,
                "turn_radius_m": float(radius) if np.isfinite(radius) else np.nan,
                "bank_angle_deg": float(bank) if np.isfinite(bank) else np.nan,
                "lat": float(np.nanmean(lat[i0:i1+1])),
                "lon": float(np.nanmean(lon[i0:i1+1])),
            })
            circle_id += 1

            start_idx = i
        i += 1

    return pd.DataFrame(circles)

# ------------------- CIRCLE CLUSTERS -------------------
def cluster_circles(circles_df, dist_thresh=200.0, time_gap=300.0):
    """Group per-circle rows into thermal clusters based on proximity in space and time."""
    if circles_df.empty:
        return pd.DataFrame(columns=[
            "cluster_id","n_segments","n_turns_sum","duration_min","alt_gained_m",
            "av_climb_ms","lat","lon","t_start","t_end"
        ])
    df = circles_df.sort_values("t_start").reset_index(drop=True).copy()
    # derive n_turns per circle (approx from duration; ~30s per 360° typical)
    df["n_turns"] = df["duration_s"] / 30.0

    clusters = []
    cluster_id = 1
    current = {"cluster_id": cluster_id, "rows": [], "t_start": None, "t_end": None}

    for _, row in df.iterrows():
        if not current["rows"]:
            current["rows"].append(row)
            current["t_start"] = row["t_start"]
            current["t_end"] = row["t_end"]
            continue

        last = current["rows"][-1]
        d = haversine_m(last["lat"], last["lon"], row["lat"], row["lon"])
        dt = row["t_start"] - last["t_end"]
        if d <= dist_thresh and dt <= time_gap:
            current["rows"].append(row)
            current["t_end"] = row["t_end"]
        else:
            clusters.append(current)
            cluster_id += 1
            current = {"cluster_id": cluster_id, "rows": [row], "t_start": row["t_start"], "t_end": row["t_end"]}

    if current["rows"]:
        clusters.append(current)

    enriched = []
    for c in clusters:
        cdf = pd.DataFrame(c["rows"])
        dur_min = cdf["duration_s"].sum() / 60.0
        turns_sum = cdf["n_turns"].sum()
        alt_gain = cdf["alt_gained_m"].sum()
        climb = (alt_gain / cdf["duration_s"].sum()) if cdf["duration_s"].sum() > 0 else np.nan
        enriched.append({
            "cluster_id": c["cluster_id"],
            "n_segments": len(cdf),  # count of circles (header aligned to altitude CSV)
            "n_turns_sum": turns_sum,
            "duration_min": dur_min,
            "alt_gained_m": alt_gain,
            "av_climb_ms": climb,
            "lat": cdf["lat"].mean(),
            "lon": cdf["lon"].mean(),
            "t_start": c["t_start"],
            "t_end": c["t_end"]
        })
    return pd.DataFrame(enriched)

# ------------------- ALTITUDE CLUSTERS -------------------
def detect_altitude_clusters(igc_path,
                             min_seg_dur_s=60.0,
                             min_gain_m=80.0,
                             vs_window=11,
                             vs_thresh_ms=0.0):
    """Detect climb segments purely from altitude time series; output cluster-like rows."""
    df = parse_igc_brecords(igc_path)
    t = df["time_s"].to_numpy()
    alt = df["alt_smooth"].to_numpy()

    if len(t) < 3:
        return pd.DataFrame(columns=[
            "cluster_id","n_segments","n_turns_sum","duration_min","alt_gained_m",
            "av_climb_ms","lat","lon","t_start","t_end"
        ])

    # simple slope via rolling linear fit
    vs = np.full(len(t), np.nan, dtype=float)
    half = vs_window // 2
    for i in range(len(t)):
        a = max(0, i - half); b = min(len(t)-1, i + half)
        tt = t[a:b+1] - t[a]
        aa = alt[a:b+1]
        if len(tt) >= 3 and np.all(np.isfinite(tt)) and np.any(np.isfinite(aa)):
            A = np.vstack([tt, np.ones_like(tt)]).T
            mask = np.isfinite(aa)
            if mask.sum() >= 3:
                m, _ = np.linalg.lstsq(A[mask], aa[mask], rcond=None)[0]
                vs[i] = m  # m/s

    # find contiguous regions with positive (or >= thresh) climb
    is_climb = vs >= vs_thresh_ms
    segments = []
    i = 0
    while i < len(t):
        if not is_climb[i]:
            i += 1
            continue
        j = i
        while j+1 < len(t) and is_climb[j+1]:
            j += 1
        dur = t[j] - t[i]
        if dur >= min_seg_dur_s:
            idx = slice(i, j+1)
            alt_gain = float(alt[j] - alt[i]) if (np.isfinite(alt[j]) and np.isfinite(alt[i])) else np.nan
            if np.isfinite(alt_gain) and alt_gain >= min_gain_m:
                lat_c = float(np.nanmean(df["lat"].to_numpy()[idx]))
                lon_c = float(np.nanmean(df["lon"].to_numpy()[idx]))
                av_climb = alt_gain / dur if dur > 0 else np.nan
                segments.append({
                    "t_start": float(t[i]),
                    "t_end": float(t[j]),
                    "duration_min": float(dur/60.0),
                    "alt_gained_m": float(alt_gain),
                    "av_climb_ms": float(av_climb),
                    "lat": lat_c,
                    "lon": lon_c
                })
        i = j + 1

    rows = []
    for cid, s in enumerate(segments, start=1):
        rows.append({
            "cluster_id": cid,
            "n_segments": 1,
            "n_turns_sum": np.nan,
            "duration_min": s["duration_min"],
            "alt_gained_m": s["alt_gained_m"],
            "av_climb_ms": s["av_climb_ms"],
            "lat": s["lat"],
            "lon": s["lon"],
            "t_start": s["t_start"],
            "t_end": s["t_end"]
        })
    return pd.DataFrame(rows)

# ------------------- MATCHER -------------------
def time_overlap(a_start, a_end, b_start, b_end):
    latest_start = max(a_start, b_start)
    earliest_end = min(a_end, b_end)
    return max(0.0, earliest_end - latest_start)

def match_clusters(circle_df, alt_df):
    """Return (matches_df, candidate_count)."""
    matches = []
    cand_count = 0
    for _, crow in circle_df.iterrows():
        for _, arow in alt_df.iterrows():
            d = haversine_m(crow["lat"], crow["lon"], arow["lat"], arow["lon"])
            if d > EPS_M:
                continue
            dt_gap = abs(crow["t_start"] - arow["t_start"])
            if dt_gap > MAX_TIME_GAP_S:
                continue
            ovl = time_overlap(crow["t_start"], crow["t_end"],
                               arow["t_start"], arow["t_end"])
            dur_short = min(crow["t_end"] - crow["t_start"],
                            arow["t_end"] - arow["t_start"])
            frac = (ovl / dur_short) if dur_short > 0 else 0.0
            cand_count += 1
            if frac >= MIN_OVL_FRAC:
                matches.append({
                    "circle_cluster_id": crow["cluster_id"],
                    "alt_cluster_id": arow["cluster_id"],
                    "dist_m": d,
                    "time_overlap_s": ovl,
                    "overlap_frac": frac
                })
    return pd.DataFrame(matches), cand_count

# ------------------- MAIN -------------------
def main():
    default_igc = "2020-11-08 Lumpy Paterson 108645.igc"
    igc_path = input(f"Enter path to IGC file [default: {default_igc}]: ").strip() or default_igc

    os.makedirs("outputs", exist_ok=True)

    # Circles
    df = parse_igc_brecords(igc_path)
    circles_df = detect_circles(df)
    circles_path = "outputs/circles.csv"
    circles_df.to_csv(circles_path, index=False)
    print(f"✓ Wrote {len(circles_df)} circles → {circles_path}")

    # Circle clusters
    circ_clusters_df = cluster_circles(circles_df)
    circ_clusters_path = "outputs/circle_clusters_enriched.csv"
    circ_clusters_df.to_csv(circ_clusters_path, index=False)
    print(f"✓ Wrote {len(circ_clusters_df)} circle clusters → {circ_clusters_path}")

    # Altitude clusters
    alt_clusters_df = detect_altitude_clusters(igc_path)
    alt_clusters_path = "outputs/overlay_altitude_clusters.csv"
    alt_clusters_df.to_csv(alt_clusters_path, index=False)
    print(f"✓ Wrote {len(alt_clusters_df)} altitude clusters → {alt_clusters_path}")

    # Matching
    matches_df, cand_count = match_clusters(circ_clusters_df, alt_clusters_df)
    matches_path = "outputs/matched_clusters.csv"
    matches_df.to_csv(matches_path, index=False)
    print(f"✓ Wrote {len(matches_df)} matches (candidates: {cand_count}) → {matches_path}")

    print("\nPipeline finished. All outputs in outputs/.")
    print(f"Matcher thresholds: EPS_M={EPS_M} m, MIN_OVL_FRAC={MIN_OVL_FRAC}, MAX_TIME_GAP_S={MAX_TIME_GAP_S} s")

if __name__ == "__main__":
    main()
