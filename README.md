# Thermal Mapper

A Python-based pipeline for analyzing glider IGC flight logs.  
This project parses B-records, detects thermals (via circle detection and altitude-based clustering), matches clusters,  
and produces per-flight artifacts (CSV/JSON) for further analysis of climb rates, thermal locations, and flight performance.  

The repository also includes scripts for batch processing, statistical analysis, and optional scraping of IGC files from WeGlide.

---

## Features

- **Circle detection**: Identify circling segments with radius, bank angle, and climb rate.
- **Altitude clusters**: Detect climb phases from smoothed altitude traces.
- **Cluster matching**: Combine circle and altitude clusters into unified thermal events.
- **Batch processing**: Run the full pipeline across multiple IGC files.
- **WeGlide scraping**: Optional Bash scripts for collecting `.igc` files automatically.
- **Outputs**: Per-flight CSVs and JSON stats for downstream analysis and visualization.
