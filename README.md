# Thermal Mapper

A Python-based pipeline for analyzing glider IGC flight logs.  
This project parses B-records, detects thermals (via circle detection and altitude-based clustering), matches clusters,  
and produces per-flight artifacts (CSV/JSON) for further analysis of climb rates, thermal locations, and flight performance.  

The repository also includes scripts for batch processing, statistical analysis, and optional scraping of IGC files from WeGlide.
