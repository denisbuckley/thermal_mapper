
#!/usr/bin/env python3
"""
apply_tuning_patches_v2_emit_new.py

Emit **new** patched files (suffix: *_patched.py) that load tuning params from
config/tuning_params.csv via tuning_loader.py. The original files are **not**
modified. This is idempotent: if a *_patched.py already contains the tuning
import, it will be skipped.

USAGE
  python apply_tuning_patches_v2_emit_new.py --root .
  python apply_tuning_patches_v2_emit_new.py --files overlay_circles_altitude_v1k.py altitude_gain_v3.py

Targets (filename heuristics):
  - altitude*: inject altitude params
  - circles*:  inject circle params
  - overlay*:  inject circle + altitude + match params
  - match*:    inject match params
  - compare*:  inject match params (harmless if unused)

Author: chatgpt
"""

import argparse, re, sys
from pathlib import Path

HEADER_BLOCK = """# === Injected by apply_tuning_patches_v2_emit_new.py ===
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
        blocks = [ALLOWED_MATCH]
    else:
        blocks = []
    return blocks

def already_patched(text: str) -> bool:
    return "from tuning_loader import" in text

def splice_after_imports(text: str, inject: str) -> str:
    lines = text.splitlines()
    insert_idx = 0
    # skip shebang/encoding
    while insert_idx < len(lines) and (lines[insert_idx].startswith("#!") or lines[insert_idx].startswith("# -*-")):
        insert_idx += 1
    last_import = -1
    for i, line in enumerate(lines):
        if re.match(r'^\s*(def|class)\s+\w+', line):
            break
        if re.match(r'^\s*(import|from)\s+\w', line):
            last_import = i
    insert_idx = last_import + 1 if last_import >= 0 else insert_idx
    new_lines = lines[:insert_idx] + ["", inject, ""] + lines[insert_idx:]
    return "\n".join(new_lines)

def patch_text(src_text: str, fname: str) -> str:
    blocks = choose_blocks(fname)
    if not blocks:
        return ""  # signal "no-op"
    if already_patched(src_text):
        return src_text  # it's already patched; emit as-is
    inject = HEADER_BLOCK + "\n".join(blocks) + "\n"
    return splice_after_imports(src_text, inject)

def process_file(path: Path) -> str:
    src = path.read_text(encoding="utf-8")
    out_text = patch_text(src, path.name)
    if out_text == "":
        return "[warn] Not a target file (heuristics): " + path.name
    out_path = path.with_name(path.stem + "_patched.py")
    if out_path.exists():
        dst = out_path.read_text(encoding="utf-8")
        if already_patched(dst):
            return f"[skip] {out_path.name} already contains tuning import."
    out_path.write_text(out_text, encoding="utf-8")
    return f"[ok] Wrote {out_path.name}"

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
        for p in root.rglob("*.py"):
            if any(k in p.name.lower() for k in ("overlay","altitude","circle","match","compare")):
                targets.append(p)
    else:
        print("Use --root <dir> or --files <list>"); return

    for p in targets:
        try:
            print(process_file(p))
        except Exception as e:
            print(f"[err] {p}: {e}")

if __name__ == "__main__":
    main()
