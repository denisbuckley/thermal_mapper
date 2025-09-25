# Changelog
All notable changes are documented here. Uses scoped Conventional Commits.

## [Unreleased]

### circles_from_brecords
- 

### circle_clusters
- 

### overlay_altitude_clusters
- 

### match_clusters
- 

### pipeline
- 

# Changelog
All notable changes are documented here.
This project tracks multiple components; entries are grouped by script (scope).

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) with scoped Conventional Commits.

---

## [Unreleased]
- Placeholder for ongoing development.

---

## [v2e] - 2025-09-25
### pipeline_v2e
- feat: Enrich `matched_clusters.csv` with key parameters from both cluster sources:
  - **Circle clusters:** `circle_av_climb_ms`, `circle_alt_gained_m`, `circle_lat`, `circle_lon`
  - **Altitude clusters:** `alt_av_climb_ms`, `alt_alt_gained_m`, `alt_lat`, `alt_lon`
- Maintains backward compatibility with previous fields (`circle_cluster_id`, `alt_cluster_id`, `dist_m`, `time_overlap_s`, `overlap_frac`).

---

## [v2d] - 2025-09-??  
### pipeline_v2d
- feat: Introduced full pipeline wrapper that chains:
  - `circles_from_brecords_v1d.py`
  - `circle_clusters_v1s.py`
  - `overlay_altitude_clusters.py`
  - `match_clusters_v1.py`
- Output: `matched_clusters.csv` containing:
  - `circle_cluster_id`, `alt_cluster_id`, `dist_m`, `time_overlap_s`, `overlap_frac`

---

## [v1s] - 2025-09-??  
### circle_clusters_v1s
- feat: Added clustering of individual circles into thermal events.
- Output: `circle_clusters_enriched.csv` with cluster-level stats:
  - `av_climb_ms`, `alt_gained_m`, `duration_min`, `lat`, `lon`

---

## [v1d] - 2025-09-??  
### circles_from_brecords_v1d
- feat: Detects individual circles directly from IGC B-records.
- Output: per-circle CSV containing:
  - `speed`, `climb`, `radius`, `bank_angle`, etc.

---

## [Initial] - 2025-09-??  
### overlay_altitude_clusters
- feat: First version detecting thermals from the altitude time series.
- Output: `overlay_altitude_clusters.csv` with cluster parameters (`av_climb_ms`, `alt_gained_m`, `duration_min`, `lat`, `lon`).

### match_clusters_v1
- feat: Matches circle clusters with altitude clusters.
- Output: `matched_clusters.csv` with IDs, distance, time overlap, and overlap fraction.