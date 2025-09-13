# IGC Analysis Flow

This diagram shows the current pipeline of scripts and data flow.

```mermaid
flowchart TD
  IGC[IGC file] --> U[igc_utils: parse + tow exclude]

  U --> A[altitude_gain_v3g: detect climbs]
  U --> C[circles_detector_v2: detect circles]

  A --> AE[altitude_clusters_enriched_*.csv]
  C --> CE[circle_clusters_enriched_*.csv]

  AE --> O[overlay_circles_altitude_v1k: map]
  CE --> O

  AE -->|strict / many→one / alt→many| M[match_*: CSVs]
  CE -->|strict / many→one / alt→many| M --> X[compare_matches_v1c: stacked table]

  AE --> B[batch_match_igc_v1: multi-IGC]
  CE --> B --> BM[batch_matched_*.csv]

  subgraph Config
    TP[tuning_params_configurator_v1.py] --> CFG[config/tuning_params.csv]
    CFG --> TL[tuning_loader.py]
    TL --> PV[ *_patched.py via patcher ]
  end