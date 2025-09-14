
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tuning_loader.py
Lightweight loader for tuning parameters from CSV and a helper to override globals.

CSV format (no header required):
    key,value
    EPS_M,2000
    MIN_OVL_FRAC,0.2
    MAX_TIME_GAP_S,900
Blank lines and lines starting with '#' are ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

def _coerce(val: str) -> Any:
    s = val.strip()
    if s.lower() in ("true","false"):
        return s.lower() == "true"
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        return s

def load_tuning(csv_path: str | Path = "config/tuning_params.csv") -> Dict[str, Any]:
    p = Path(csv_path)
    if not p.exists():
        return {}
    params: Dict[str, Any] = {}
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # allow key,value or key = value
        if "," in line:
            key, val = line.split(",", 1)
        elif "=" in line:
            key, val = line.split("=", 1)
        else:
            # skip malformed
            continue
        key = key.strip()
        val = _coerce(val)
        if key:
            params[key] = val
    return params

def override_globals(g: dict, params: Dict[str, Any], allowed: set[str] | None = None) -> None:
    if not params:
        return
    for k, v in params.items():
        if allowed is not None and k not in allowed:
            continue
        g[k] = v
