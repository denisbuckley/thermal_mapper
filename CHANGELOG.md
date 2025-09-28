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
git add CHANGELOG.md
git commit -m "docs(changelog): update for v3.2 stable batch pipeline"

# Changelog
All notable changes are documented here.
This project tracks multiple components; entries are grouped by script (scope).

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) with scoped Conventional Commits.

v3.2 – Stable Batch Pipeline Integration

Date: 2025-09-28

Added
	•	batch_run_v3.py: new batch runner that orchestrates pipeline_v3 across all IGC files in igc/.
	•	Each flight now runs under its own folder inside outputs/batch_csv/<flight_basename>/.
	•	Collects and logs per-flight artifacts:
	•	circles.csv
	•	circle_clusters_enriched.csv
	•	altitude_clusters.csv
	•	matched_clusters.csv
	•	matched_clusters.json
	•	pipeline_debug.log

Changed
	•	pipeline_v3.py: confirmed stable with upstreams; runs consistently from repo root.
	•	Upstream scripts (circles_from_brecords_v1d.py, circle_clusters_v1s.py, altitude_clusters_v1.py, match_clusters_v1.py) aligned to work seamlessly with batch runner.

Notes
	•	Tag v3.2 marks the first stable release with batch execution integrated.
	•	Future work: filtering/grouping (build_thermals_v1.py) to generate thermal waypoints.


## [batch_run_thermals_v1] - 2025-09-25
### Fixed
- Removed invalid inline `if` syntax (`; if ...:`) after `pd.read_csv`, which caused `SyntaxError` in PyCharm.
- Replaced with proper multi-line `if` statements for compatibility across Python interpreters.

### Added
- GeoJSON export (`thermals_clusters.geojson`) alongside CSV outputs, for easy map visualization.
- Confirmed stable overwrite of per-flight outputs (`circles.csv`, `circle_clusters_enriched.csv`, `overlay_altitude_clusters.csv`, `matched_clusters.csv`).

### Notes
- Default thresholds: `--max-dist-m 5000`, `--min-alt-gain-m 500`, `--min-climb-ms 1.0`.
- Produces both `thermals_all_raw.csv` and `thermals_clusters.csv` plus GeoJSON for clusters.
## [v2e] - 2025-09-25
### pipeline_v2e
- feat: Wrapper now prompts for a single IGC filename (with default).
- feat: Passes the same IGC path through the entire pipeline:
  1. `circles_from_brecords_v1d.py`
  2. `circle_clusters_v1s.py`
  3. `overlay_altitude_clusters.py`
  4. Enriched matcher inside `pipeline_v2e.py`
- Added dual-mode execution of child scripts:
  - Try CLI (`python <script> <igc>`).
  - Fallback to interactive stdin if child script prompts with `input()`.
- Skips re-running steps if outputs already exist (use `--force` to override).
- Output: `matched_clusters.csv` enriched with climb rate, altitude gain, and lat/lon from both cluster types.
---
## [v2g] - 2025-09-25
### pipeline_v2g
- feat: Replace prior wrapper with a **non-destructive orchestrator**.
- Calls existing upstream scripts and delegates matching to `match_clusters_v1.py`.
- Adds `--outdir` (default `outputs/`), `--force`, `--no-clobber`, `--dry-run`, `--verbose`.
- Avoids reading or rewriting intermediate CSVs to prevent accidental clobbering.
---
## [v2f] - 2025-09-25
### pipeline_v2f
- feat: Add `--outdir` (default `outputs/`) so all default inputs/outputs are colocated.
- feat: Robust child execution (tries IGC and per-circle CSV modes) and auto-discovers recent outputs if filenames differ.
- feat: Path overrides via `--circle-out`, `--circle-clusters`, `--alt-clusters`; supports direct `--circles/--alt`.
- Output remains `matched_clusters.csv` enriched with climb rate, altitude gain, and lat/lon from both cluster types.
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