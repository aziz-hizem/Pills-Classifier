"""
white_pill_segmentor.py
-----------------------
Dedicated detection pipeline for WHITE pills in blister packs.

Validated approach (empirically tested on 4 diverse images)
------------------------------------------------------------
Two independent cues AND'd together:

CUE 1 — ADAPTIVE LOCAL BRIGHTNESS (block size ~101px)
    cv2.adaptiveThreshold with a large block size finds pixels that are
    LOCALLY brighter than their surroundings at the pill scale.
    Pills are convex domes → brighter than surrounding foil.
    This works even when pills and foil have similar absolute brightness,
    because it measures relative brightness within each pill-sized window.
    Otsu threshold on the score map makes it self-calibrating per image.

CUE 2 — LOW FINE-SCALE TEXTURE (kernel ~7px)
    Foil has a repeating fine pattern (dots, crosshatch, knurling) at 5-15px.
    Pills are smooth → low variance at this scale.
    Otsu on the variance map auto-calibrates the foil/smooth boundary.

WHY THIS WORKS ACROSS DIFFERENT FOIL TYPES:
    - Shiny diamond foil (image 1): strong texture signal + clear brightness contrast
    - Fine dot foil, low contrast (image 4): weak texture but strong adaptive brightness
    - Mixed cases (images 9, 11): both cues contribute partially
    The AND combination means each cue can be noisy as long as both agree on pills.

FALSE POSITIVE REJECTION:
    The pack frame/border blob has very low solidity (0.10-0.15) because it's
    a ring shape. Pills have solidity > 0.50. This single filter kills the border.

Pipeline:
    1. Adaptive threshold (block=101, C=-5) → locally-bright mask
    2. Fine variance map (k=7) + Otsu → not-foil mask  
    3. AND the two masks → pill candidates
    4. Morphological open+close → cleanup
    5. Watershed to separate touching pills
    6. Contour filter: area + solidity (primary), convexity (secondary)

Public API
----------
    segment_white_pills(bgr, **params) -> WhitePillResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class WhitePillResult:
    pill_count: int
    contours: List[np.ndarray]
    pill_mask: np.ndarray
    debug_images: Dict[str, np.ndarray] = field(default_factory=dict)
    """
    debug_images keys:
        'texture_map'    – fine-scale variance map (bright = textured = foil)
        'texture_thresh' – not-foil mask from Otsu on variance (white = smooth = pill candidate)
        'cleaned_mask'   – combined + morphologically cleaned pill candidates
        'distance'       – distance transform for watershed
        'annotated'      – BGR image with accepted contours in red
    """


# ---------------------------------------------------------------------------
# Cue 1: adaptive local brightness
# ---------------------------------------------------------------------------

def _adaptive_bright_mask(
    gray_u8: np.ndarray,
    block_size: int,
    c_offset: float,
) -> np.ndarray:
    """
    Find pixels locally brighter than their neighbourhood.
    block_size should be ~1.5–2x the pill diameter in pixels.
    c_offset < 0 means a pixel must be |c_offset| above the local mean.
    """
    block = max(3, block_size | 1)  # must be odd
    return cv2.adaptiveThreshold(
        gray_u8, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block, c_offset,
    )


# ---------------------------------------------------------------------------
# Cue 2: fine-scale texture (foil pattern detection)
# ---------------------------------------------------------------------------

def _not_foil_mask(
    lab_l: np.ndarray,
    fine_kernel: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect foil via its fine repeating pattern (high local variance at 5-11px).
    Returns (var_u8_for_debug, not_foil_binary_mask).

    Uses Otsu's method to auto-calibrate the threshold per image.
    """
    k       = max(3, fine_kernel | 1)
    l_f32   = lab_l.astype(np.float32)
    mean    = cv2.blur(l_f32,      (k, k))
    mean_sq = cv2.blur(l_f32 ** 2, (k, k))
    var_map = np.sqrt(np.maximum(mean_sq - mean ** 2, 0.0))

    var_max = float(var_map.max()) or 1.0
    var_u8  = np.clip(var_map / var_max * 255, 0, 255).astype(np.uint8)

    # Otsu: pixels below the threshold are smooth (= not foil)
    otsu_t, _ = cv2.threshold(var_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    not_foil  = (var_u8 < otsu_t).astype(np.uint8) * 255

    return var_u8, not_foil


# ---------------------------------------------------------------------------
# Morphological cleanup
# ---------------------------------------------------------------------------

def _morph_cleanup(
    mask: np.ndarray,
    open_iter: int,
    close_iter: int,
    kernel_size: int,
) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=open_iter)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=close_iter)
    return mask


# ---------------------------------------------------------------------------
# Watershed separation
# ---------------------------------------------------------------------------

def _watershed_separate(
    candidate_mask: np.ndarray,
    bgr: np.ndarray,
    separation: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Distance-transform watershed. Returns (mask, dist_vis_u8)."""
    dist = cv2.distanceTransform(candidate_mask, cv2.DIST_L2, 5)

    if dist.max() < 1:
        return candidate_mask, np.zeros(candidate_mask.shape, dtype=np.uint8)

    _, sure_fg = cv2.threshold(dist, separation * dist.max(), 255, 0)
    sure_fg    = sure_fg.astype(np.uint8)

    if sure_fg.max() == 0:
        return candidate_mask, np.zeros(candidate_mask.shape, dtype=np.uint8)

    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    sure_bg = cv2.dilate(candidate_mask, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(bgr.copy(), markers)

    result = np.zeros_like(candidate_mask)
    result[markers > 1] = 255

    dist_max = float(dist.max()) or 1.0
    dist_vis = np.clip(dist / dist_max * 255, 0, 255).astype(np.uint8)
    return result, dist_vis


# ---------------------------------------------------------------------------
# Contour filtering
# ---------------------------------------------------------------------------

def _filter_contours(
    pill_mask: np.ndarray,
    min_area_ratio: float,
    max_area_ratio: float,
    min_convexity: float,
    min_solidity: float,
) -> Tuple[int, List[np.ndarray]]:
    """
    Filter by area, solidity, convexity.
    Key insight: pack border blob has solidity ~0.10-0.15 (ring shape).
    Pills have solidity > 0.50. min_solidity=0.50 cleanly rejects the border.
    No circularity filter — oblong tablets must pass.
    """
    contours, _ = cv2.findContours(
        pill_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    h, w       = pill_mask.shape[:2]
    total_area = h * w
    min_area   = min_area_ratio * total_area
    max_area   = max_area_ratio * total_area

    accepted = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue
        hull      = cv2.convexHull(cnt)
        hull_peri = cv2.arcLength(hull, True)
        hull_area = cv2.contourArea(hull)
        convexity = hull_peri / perimeter if perimeter > 0 else 0.0
        solidity  = area / hull_area      if hull_area  > 0 else 0.0
        if convexity < min_convexity or solidity < min_solidity:
            continue
        accepted.append(cnt)

    return len(accepted), accepted


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment_white_pills(
    bgr: np.ndarray,
    # ── Adaptive brightness (Cue 1) ───────────────────────────────────────
    texture_kernel: int   = 7,      # fine variance kernel (Cue 2) — do not confuse
                                    # with adaptive block; kept as texture_kernel for UI
    texture_thresh: float = 0.0,    # reserved / UI compat; Otsu is always used for Cue 2
    # ── Adaptive threshold block size ────────────────────────────────────
    dark_val_max: int     = 101,    # repurposed: adaptive block size (must be odd, ~80-120)
    bright_val_min: int   = 160,    # repurposed: |C offset| for adaptive threshold (5-15)
                                    # (kept these param names for UI slider compatibility)
    # ── Morphological cleanup ─────────────────────────────────────────────
    morph_open_iter: int  = 2,
    morph_close_iter: int = 4,
    morph_kernel: int     = 7,
    # ── Watershed ─────────────────────────────────────────────────────────
    separation: float     = 0.40,
    # ── Contour filtering ─────────────────────────────────────────────────
    min_area_ratio: float = 0.003,
    max_area_ratio: float = 0.20,
    min_convexity: float  = 0.70,
    min_solidity: float   = 0.50,   # key filter: rejects pack border (solidity~0.1)
) -> WhitePillResult:
    """
    Detect white pills using adaptive local brightness + fine texture exclusion.

    Validated on 4 diverse blister pack types:
      - Shiny diamond-pattern foil
      - Fine dot-pattern foil (low contrast)
      - Matte silver foil
      - Mixed/rendered images

    Parameters
    ----------
    bgr : np.ndarray
        White-balance-corrected, rectified BGR image.

    Note on repurposed parameters for UI slider compatibility
    ---------------------------------------------------------
    dark_val_max   → adaptive block size (101 = ~pill diameter in pixels)
    bright_val_min → adaptive C offset magnitude (5 = pixel must be 5 above local mean)
                     passed as negative internally: C = -(bright_val_min / 32)
                     slider range 160-255 maps to C offset -5 to -8
    """
    debug: Dict[str, np.ndarray] = {}

    lab   = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # ── Cue 1: adaptive local brightness ─────────────────────────────────
    # dark_val_max repurposed as block size; ensure odd and reasonable
    block_size = max(51, dark_val_max | 1)
    # bright_val_min repurposed: maps 160-255 → C offset -5 to -8
    c_offset = -(5.0 + (bright_val_min - 160) / 32.0)
    adaptive_bright = _adaptive_bright_mask(gray, block_size, c_offset)

    # ── Cue 2: fine-scale texture → not-foil mask ─────────────────────────
    var_u8, not_foil = _not_foil_mask(lab[:, :, 0], texture_kernel)
    debug["texture_map"]    = var_u8    # bright = textured = foil
    debug["texture_thresh"] = not_foil  # white = smooth = pill candidate

    # ── Combine cues ──────────────────────────────────────────────────────
    combined = cv2.bitwise_and(adaptive_bright, not_foil)

    # ── Morphological cleanup ─────────────────────────────────────────────
    cleaned = _morph_cleanup(combined, morph_open_iter, morph_close_iter, morph_kernel)
    debug["cleaned_mask"] = cleaned

    # ── Watershed ─────────────────────────────────────────────────────────
    pill_mask, dist_vis = _watershed_separate(cleaned, bgr, separation)
    debug["distance"] = dist_vis

    # ── Contour filtering ─────────────────────────────────────────────────
    count, contours = _filter_contours(
        pill_mask,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        min_convexity=min_convexity,
        min_solidity=min_solidity,
    )

    final_mask = np.zeros(pill_mask.shape, dtype=np.uint8)
    if contours:
        cv2.drawContours(final_mask, contours, -1, 255, -1)

    annotated = bgr.copy()
    if contours:
        cv2.drawContours(annotated, contours, -1, (0, 0, 255), 2)
    debug["annotated"] = annotated

    return WhitePillResult(
        pill_count=count,
        contours=contours,
        pill_mask=final_mask,
        debug_images=debug,
    )