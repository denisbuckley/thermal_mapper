#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# CONFIG — copy your Request URL query (from DevTools) but
# REMOVE any &skip=... The script paginates with skip/limit.
BASE_QS="scoring_date_start=2000-01-01&scoring_date_end=2025-09-30&region_id_in=AU-WA&limit=100&include_story=false&include_stats=false"
OUTDIR="./weglide_igc"
SLEEP_SECS="0.25"
# ------------------------------------------------------------

API_LIST="https://api.weglide.org/v1/flight"
API_FLIGHT="https://api.weglide.org/v1/flight"
SITE_BASE="https://weglide.org/flight"

TMP_JSON=".wg_list.json"
TMP_HTML=".wg_page.html"
TMP_FLIGHT_JSON=".wg_flight.json"
IDS_TMP=".ids.tmp"

LOG_FAIL="failed_weglide_ids.log"
LOG_NOIGC="no_igc_in_page.log"
LOG_OK="downloaded_ok.log"

mkdir -p "$OUTDIR"
: > "$IDS_TMP"
: > "$LOG_FAIL"
: > "$LOG_NOIGC"
: > "$LOG_OK"

need () { command -v "$1" >/dev/null || { echo "Please install $1 (e.g. brew install $1)"; exit 1; }; }
need jq
need curl

echo "== Step 1: Collecting flight IDs =="
skip=0
while : ; do
  URL="${API_LIST}?${BASE_QS}&skip=${skip}"
  echo "GET $URL"
  if ! curl -fsSL -H "Accept: application/json" "$URL" -o "$TMP_JSON"; then
    echo "⚠️  Failed list page $URL"; break
  fi
  count=$(jq 'if type=="array" then length else 0 end' "$TMP_JSON")
  [[ "$count" -eq 0 ]] && break
  jq -r '.[].id' "$TMP_JSON" >> "$IDS_TMP"
  skip=$((skip + 100))
  sleep "$SLEEP_SECS"
done

sort -u "$IDS_TMP" -o "$IDS_TMP"
TOTAL=$(wc -l < "$IDS_TMP" | tr -d '[:space:]')
echo "Collected $TOTAL unique IDs."
[[ "$TOTAL" -gt 0 ]] || { rm -f "$TMP_JSON" "$TMP_HTML" "$TMP_FLIGHT_JSON" "$IDS_TMP"; exit 0; }

echo
echo "== Step 2: Downloading IGCs to: $OUTDIR =="

while read -r id; do
  [[ -z "$id" ]] && continue
  out="${OUTDIR}/${id}.igc"
  [[ -f "$out" ]] && { echo "✓ ${id} (exists)"; echo "$id" >> "$LOG_OK"; continue; }

  igc_url=""

  # --- 2a) Try API for this flight (requires WG_TOKEN) ---
  if [[ -n "${WG_TOKEN:-}" ]]; then
    if curl -fsSL -H "Authorization: Bearer $WG_TOKEN" -H "Accept: application/json" \
         "${API_FLIGHT}/${id}" -o "$TMP_FLIGHT_JSON"; then
      igc_url=$(jq -r '.. | strings | select(test("\\.igc($|\\?)"))' "$TMP_FLIGHT_JSON" | head -n1 || true)
    fi
  fi

  # --- 2b) Fallback: scrape HTML for a .igc URL (public) ---
  if [[ -z "$igc_url" ]]; then
    if curl -fsSL -L "${SITE_BASE}/${id}" -o "$TMP_HTML"; then
      raw=$(grep -oE 'https://[^"]+\.igc' "$TMP_HTML" | head -n1 || true)
      if [[ -n "$raw" ]]; then
        # If it’s in a SkySight wrapper, trim to the real .igc after igc=
        if [[ "$raw" == *"skysight.io"* ]]; then
          igc_url=$(echo "$raw" | sed -E 's/.*igc=(https[^&]+\.igc).*/\1/')
        else
          igc_url="$raw"
        fi
      fi
    fi
  fi

  # --- 2c) If still nothing, log and continue ---
  if [[ -z "$igc_url" ]]; then
    echo "— ${id}: no .igc link (private/not uploaded)" | tee -a "$LOG_NOIGC"
    sleep "$SLEEP_SECS"
    continue
  fi

  # --- 3) Download the IGC ---
  if curl -fsSL -L "$igc_url" -o "$out"; then
    if head -n1 "$out" | grep -qE '^(A|HFDTE)'; then
      echo "✓ ${id}  → $(basename "$out")"
      echo "$id" >> "$LOG_OK"
    else
      echo "✗ ${id} (non-IGC content)" | tee -a "$LOG_FAIL"
      rm -f "$out"
    fi
  else
    echo "✗ ${id} (download failed)" | tee -a "$LOG_FAIL"
  fi

  sleep "$SLEEP_SECS"
done < "$IDS_TMP"

rm -f "$TMP_JSON" "$TMP_HTML" "$TMP_FLIGHT_JSON" "$IDS_TMP"

echo
OKC=$(wc -l < "$LOG_OK" 2>/dev/null || echo 0)
NOC=$(wc -l < "$LOG_NOIGC" 2>/dev/null || echo 0)
FLC=$(wc -l < "$LOG_FAIL" 2>/dev/null || echo 0)
echo "Done. Saved: $OKC  |  No IGC link: $NOC  |  Failed: $FLC"
echo "Folder: $OUTDIR"
[[ -s "$LOG_NOIGC" ]] && echo "See: $LOG_NOIGC"
[[ -s "$LOG_FAIL"  ]] && echo "See: $LOG_FAIL"
