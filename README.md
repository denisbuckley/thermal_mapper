# Thermal Mapper

A Python-based pipeline for analyzing glider IGC flight logs.  
This project parses B-records, detects thermals (via circle detection and altitude-based clustering), matches clusters,  
and produces per-flight artifacts (CSV/JSON) for further analysis of climb rates, thermal locations, and flight performance.  

The repository also includes scripts for batch processing, statistical analysis, and optional scraping of IGC files from WeGlide.

---

## Features

- **Circle detection**: Identify circles with duration, radius, bank angle, speed and climb rate.
- **Circle clusters**: Cluster circles associated in time and space to identify thermalling flight.
- **Altitude clusters**: Detect climb phases from smoothed altitude traces.
- **Cluster matching**: Combine circle and altitude clusters into unified thermal events.
- **Batch processing**: Run the full pipeline across multiple IGC files.
- **WeGlide scraping**: Optional Bash scripts for collecting `.igc` files automatically.
- **Outputs**: Per-flight CSVs and JSON stats for downstream analysis.
- **Visuals**: Plot interactive track and altitude with clusters and thermals.
- **Thermal waypoints** Geojson, kml, cup and csv output for Google Earth and flight computer

---

## Installation

Clone the repository and set up a Python environment:

```bash
git clone https://github.com/<your-org>/thermal_mapper.git
cd thermal_mapper
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Python 3.11+ recommended.
Tested on macOS, should work on Linux.

---

## Usage

### Batch Mode

Run batch *.igc files in default directory <project root>/igc.

```bash
python batch_run_v3.1.py
python build_thermals_v1a.py
python sweep_eps_for_thermals.py
python track_plotter_batch_v1b.py
python match_clusters_v1.py
```
---

### Single *.igc Mode

Run a single *.igc file and return a plot showing track, circle/altitude and matched clusters with thermals.

``` bash
python pipeline_v4.1j.py

```
---
### Outputs

Each flight generates a subfolder under `outputs/batch_csv/<flight>` containing:
- **circles.csv** — individual circling events with climb, radius, bank, etc.
- **circle_clusters_enriched.csv** — grouped circle clusters with summary stats.
- **altitude_clusters.csv** — altitude-based climb segments.
- **matched_clusters.csv / matched_clusters.json** — combined circle/altitude matches and stats.
- **pipeline_debug.log** — per-flight debug log.

A global log is written to `outputs/batch_debug.log`.

--- 
### Tuning

All scripts use shared settings from `tuning.json`.  
To explore tuning, run the standalone scripts in the following order:  

```
python circles_from_brecords_v1e.py
python circle_clusters_v1s.py
python altitude_clusters_v1.py
python match_clusters_v1.py
```
---

## Task Filter 

Select task waypoints from .cup file to filter thermal waypoints within optional deviation from flight path, accepts multiple turnpoints.
```bash
python thermal_filter_v1a.py
```
---

### WeGlide IGC Scraping (Optional)

A Bash script can download flights from WeGlide (by region and date range):

```bash
./weglide_igc_scrape.sh
```
---

## License

This project is licensed under the [MIT License](LICENSE).  
You are free to use, modify, and distribute this software, subject to the conditions of the license.
