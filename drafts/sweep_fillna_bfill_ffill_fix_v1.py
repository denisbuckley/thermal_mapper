#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sweep_fillna_bfill_ffill_fix_v1.py
Scan your repo for deprecated pandas calls like .fillna(method="bfill"/"ffill")
and an accidental trailing "else:" on the same line as .bfill()/.ffill().

Default: DRY RUN (no changes). Use --apply to write changes with timestamped .bak files.
Skips common folders: .git, venv, outputs, __pycache__, .idea
"""

from __future__ import annotations
import argparse, re, ast
from pathlib import Path
from datetime import datetime

SKIP_DIRS = {'.git', 'venv', 'outputs', '__pycache__', '.idea', '.pytest_cache'}

# Patterns
RE_DEPR_BFILL = re.compile(r"\.fillna\s*\(\s*method\s*=\s*(['\"])bfill\1\s*\)")
RE_DEPR_FFILL = re.compile(r"\.fillna\s*\(\s*method\s*=\s*(['\"])ffill\1\s*\)")
RE_TRAILING_ELSE = re.compile(r"(\.bfill\(\)\.fillna\([^)]*\))\s*else\s*:\s*$")
RE_TRAILING_ELSE_FF = re.compile(r"(\.ffill\(\)\.fillna\([^)]*\))\s*else\s*:\s*$")

def iter_py_files(root: Path):
    for p in root.rglob("*.py"):
        parts = set(p.parts)
        if parts & SKIP_DIRS:
            continue
        yield p

def fix_content(text: str):
    stats = {'depr_bfill': 0, 'depr_ffill': 0, 'trailing_else': 0}
    b_bef = len(RE_DEPR_BFILL.findall(text))
    f_bef = len(RE_DEPR_FFILL.findall(text))
    text = RE_DEPR_BFILL.sub(".bfill()", text)
    text = RE_DEPR_FFILL.sub(".ffill()", text)
    stats['depr_bfill'] = b_bef
    stats['depr_ffill'] = f_bef

    out_lines = []
    tr_els = 0
    for ln in text.splitlines():
        new_ln = RE_TRAILING_ELSE.sub(r"\1", ln)
        new_ln = RE_TRAILING_ELSE_FF.sub(r"\1", new_ln)
        if new_ln != ln:
            tr_els += 1
        out_lines.append(new_ln)
    text2 = "\n".join(out_lines)
    stats['trailing_else'] = tr_els
    return text2, stats

def syntax_ok(text: str, path: Path) -> bool:
    try:
        ast.parse(text, filename=str(path))
        return True
    except SyntaxError as e:
        print(f"[syntax ERR] {path}:{e.lineno}:{e.offset} {e.msg}")
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Project root (default: .)")
    ap.add_argument("--apply", action="store_true", help="Write changes with .bak_<timestamp> backups")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    ts = datetime.now().strftime("%Y%m%d%H%M%S")

    total = {'files': 0, 'depr_bfill':0, 'depr_ffill':0, 'trailing_else':0, 'changed':0, 'syntax_fail':0}
    changed = []

    for p in iter_py_files(root):
        text = p.read_text(encoding="utf-8", errors="ignore")
        new_text, stats = fix_content(text)
        if any(stats.values()):
            total['files'] += 1
            for k in ('depr_bfill','depr_ffill','trailing_else'):
                total[k] += stats[k]
            if args.apply and new_text != text:
                if not syntax_ok(new_text, p):
                    total['syntax_fail'] += 1
                    print(f"[skip write] {p} (would break syntax)")
                    continue
                bak = p.with_suffix(p.suffix + f".bak_{ts}")
                bak.write_text(text, encoding="utf-8")
                p.write_text(new_text, encoding="utf-8")
                total['changed'] += 1
                changed.append(str(p))
            else:
                print(f"[would fix] {p}  bfill:{stats['depr_bfill']} ffill:{stats['depr_ffill']} trailing_else:{stats['trailing_else']}")

    print(f"\nSummary: files:{total['files']} depr_bfill:{total['depr_bfill']} depr_ffill:{total['depr_ffill']} trailing_else:{total['trailing_else']} changed:{total['changed']} syntax_fail:{total['syntax_fail']}")
    if args.apply and changed:
        print("Changed files:")
        for c in changed:
            print("  -", c)

if __name__ == "__main__":
    main()
