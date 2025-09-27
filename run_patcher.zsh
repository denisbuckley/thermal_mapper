#!/bin/zsh
# run_patcher.zsh — robust zsh runner for the patcher
# - Collects likely target scripts in the current repo
# - Skips already-patched files (*_patched.py)
# - Falls back to --root . if no explicit matches are found

set -euo pipefail

# Prefer the new patcher that emits *_patched.py
PATCHER="apply_tuning_patches_v2_emit_new.py"

if [[ ! -f "$PATCHER" ]]; then
  echo "ERROR: $PATCHER not found in current directory. Place it in project root."
  exit 1
fi

typeset -a files
files=()

# Explicit common entry points (add if present)
[[ -f overlay_circles_altitude_v1k.py ]] && files+=("overlay_circles_altitude_v1k.py")
[[ -f compare_circles_altitude_v2f.py ]] && files+=("compare_circles_altitude_v2f.py")
[[ -f match_clusters_strict_v1c.py    ]] && files+=("match_clusters_strict_v1c.py")

# Add any altitude / circles / match / overlay / compare scripts (excluding already-patched)
for f in altitude_gain_*.py circles_*.py match_*clusters*.py overlay_*.py compare_*.py; do
  [[ -f "$f" ]] || continue
  [[ "$f" == *_patched.py ]] && continue
  files+=("$f")
done

# Print what we'll patch
if (( ${#files[@]} )); then
  echo "[run_patcher] Patching ${#files[@]} file(s):"
  for f in "${files[@]}"; do
    echo "  - $f"
  done
  python "$PATCHER" --files "${files[@]}"
else
  echo "[run_patcher] No specific files matched; scanning repo recursively with --root ."
  python "$PATCHER" --root .
fi

echo "[run_patcher] Done."
