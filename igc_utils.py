"""igc_utils.py
Thin wrapper to import your existing project functions so other scripts
(thermal clustering, detectors, etc.) can use them cleanly.

It tries in this order:
  1) circle_detector_full_360.py  (your promoted baseline file)
  2) scratch2.py                  (your working scratch file)

If neither is present, it raises a clear ImportError with guidance.
"""

# Try the promoted baseline first
try:
    from drafts.circle_detector_full_360 import parse_igc, compute_derived, detect_tow_segment
except Exception as e1:
    try:
        from scratch2 import parse_igc, compute_derived, detect_tow_segment
    except Exception as e2:
        raise ImportError(
            "igc_utils.py could not import required functions. "
            "Please ensure either circle_detector_full_360.py or scratch2.py "
            "is in the same project and defines parse_igc, compute_derived, detect_tow_segment.\n"
            f"Import errors were:\n- circle_detector_full_360: {e1}\n- scratch2: {e2}"
        )

__all__ = ['parse_igc', 'compute_derived', 'detect_tow_segment']
