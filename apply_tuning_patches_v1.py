
#!/usr/bin/env python3
"""
apply_tuning_patches_v1.py

Safely injects tuning parameter loading into your scripts so they read
values from config/tuning_params.csv via tuning_loader.py.

- Idempotent: skips files already patched (detects "from tuning_loader import").
- Makes a timestamped .bak backup for each modified file.
- Chooses the correct "allowed" parameter set based on filename heuristics.

USAGE
  python apply_tuning_patches_v1.py --root .
  python apply_tuning_patches_v1.py --files overlay_circles_altitude_v1k.py altitude_gain_v3.py

Targets (auto-detected by name if using --root):
  - altitude*: altitude detectors (A_* & altitude-drop params)
  - circles*:  circle detectors (C_* params)
  - overlay*:  overlay plotters (both C_* and A_* params)
  - match*:    matchers (EPS_M, MIN_OVL_FRAC, MAX_TIME_GAP_S)
  - compare*:  comparison utilities (usually no tunables, but we patch match params just in case)

You must have: tuning_loader.py present in PYTHONPATH, and config/tuning_params.csv (optional).

Author: chatgpt
"""

import argparse, re, sys, os
from datetime import datetime
from pathlib import Path

HEADER_BLOCK = """# === Injected by apply_tuning_patches_v1.py ===
from tuning_loader import load_tuning, override_globals
_tuning = load_tuning("config/tuning_params.csv")
"""

ALLOWED_CIRCLES = """override_globals(globals(), _tuning, allowed={
    "C_MIN_ARC_DEG","C_MIN_RATE_DPS","C_MAX_RATE_DPS",
    "C_MIN_RADIUS_M","C_MAX_RADIUS_M","C_MIN_DIR_RATIO",
    "TIME_CAP_S","C_MAX_WIN_SAMPLES","C_EPS_M","C_MIN_SAMPLES"
})"""

ALLOWED_ALT = """override_globals(globals(), _tuning, allowed={
    "MIN_CLIMB_S","MIN_GAIN_M","SMOOTH_RADIUS_S",
    "MAX_GAP_S","ALT_DROP_M","ALT_DROP_FRAC",
    "A_EPS_M","A_MIN_SAMPLES"
})"""

ALLOWED_MATCH = """override_globals(globals(), _tuning, allowed={
    "EPS_M","MIN_OVL_FRAC","MAX_TIME_GAP_S"
})"""

def choose_blocks(fname: str):
    lower = fname.lower()
    blocks = []
    if "overlay" in lower:
        blocks = [ALLOWED_CIRCLES, ALLOWED_ALT, ALLOWED_MATCH]
    elif "altitude" in lower or "alt_" in lower:
        blocks = [ALLOWED_ALT]
    elif "circle" in lower:
        blocks = [ALLOWED_CIRCLES]
    elif "match" in lower:
        blocks = [ALLOWED_MATCH]
    elif "compare" in lower:
        # usually read-only; still allow match tunables
        blocks = [ALLOWED_MATCH]
    else:
        # default: be conservative (no-op)
        blocks = []
    return blocks

def already_patched(text: str) -> bool:
    return "from tuning_loader import" in text

def splice_after_imports(text: str, inject: str) -> str:
    """
    Insert inject block after the last top-level import block.
    If none found, insert near the top after shebang/encoding/comments.
    """
    lines = text.splitlines()
    insert_idx = 0
    # skip shebang and encoding
    while insert_idx < len(lines) and (lines[insert_idx].startswith("#!") or lines[insert_idx].startswith("# -*-")):
        insert_idx += 1
    # scan for imports
    last_import = -1
    for i, line in enumerate(lines):
        # stop if we hit def/class at top
        if re.match(r'^\s*(def|class)\s+\w+', line):
            break
        if re.match(r'^\s*(import|from)\s+\w', line):
            last_import = i
    insert_idx = last_import + 1 if last_import >= 0 else insert_idx
    new_lines = lines[:insert_idx] + ["", inject, ""] + lines[insert_idx:]
    return "\n".join(new_lines)

def patch_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if already_patched(text):
        print(f"[skip] {path} already patched.")
        return False
    blocks = choose_blocks(path.name)
    if not blocks:
        print(f"[warn] {path} not recognized; no tuning blocks applied.")
        return False
    inject = HEADER_BLOCK + "\n".join(blocks) + "\n"
    patched = splice_after_imports(text, inject)
    if patched == text:
        print(f"[noop] {path} unchanged.")
        return False
    # backup
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    bak = path.with_suffix(path.suffix + f".bak_{ts}")
    bak.write_text(text, encoding="utf-8")
    path.write_text(patched, encoding="utf-8")
    print(f"[ok] Patched {path} (backup -> {bak.name})")
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, help="root folder to scan (recursively)")
    ap.add_argument("--files", nargs="*", help="explicit files to patch")
    args = ap.parse_args()

    targets = []
    if args.files:
        targets = [Path(f) for f in args.files]
    elif args.root:
        root = Path(args.root)
        for ext in ("*.py",):
            targets.extend(root.rglob(ext))
        # filter to likely project scripts
        targets = [p for p in targets if any(k in p.name.lower() for k in
                    ("overlay","altitude","circle","match","compare"))]
    else:
        print("Use --root <dir> or --files <list>"); return

    changed = 0
    for p in targets:
        try:
            changed += 1 if patch_file(p) else 0
        except Exception as e:
            print(f"[err] {p}: {e}")

    print(f"\nDone. Modified {changed} file(s).")

if __name__ == "__main__":
    main()
