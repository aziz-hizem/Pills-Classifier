"""
white_pill_segmentor.py
-----------------------
White pill detection inside a blister contour.

Approach
--------
White pills (gray ~220-255) are plainly brighter than silver aluminum foil
(gray ~120-180).  The detection is restricted to the blister contour so the
background (white tables, labels) cannot interfere.

Pipeline
--------
1. Otsu threshold on brightness within the contour → white candidate mask
2. Morphological open  → remove small noise
3. Morphological close → bridge score-line gaps
4. Hole fill           → handle pill logos / markings
5. Watershed           → separate touching pills
6. Contour filter      → area + solidity
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
        'white_mask'   – brightness-threshold result (BGR, threshold value shown)
        'cleaned_mask' – after morphological cleanup + hole fill
        'distance'     – distance transform used for watershed
        'annotated'    – BGR image with accepted contours drawn in red
    """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _otsu_in_mask(gray: np.ndarray, mask: np.ndarray, floor: int) -> int:
    """Otsu threshold computed only on pixels inside mask.
    Floored at `floor` to avoid degenerate results when one class dominates."""
    pixels = gray[mask == 255]
    if len(pixels) < 200:
        return floor
    tmp = pixels.reshape(-1, 1)
    otsu_t, _ = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return max(int(otsu_t), floor)


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


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed holes (score lines, logos) via flood-fill from corners."""
    inv   = cv2.bitwise_not(mask)
    h, w  = mask.shape[:2]
    flood = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(inv, flood, (0, 0), 255)
    return cv2.bitwise_or(mask, cv2.bitwise_not(inv))


def _watershed_separate(
    candidate_mask: np.ndarray,
    bgr: np.ndarray,
    separation: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Distance-transform watershed to split touching pills."""
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


def _filter_contours(
    pill_mask: np.ndarray,
    min_area_ratio: float,
    max_area_ratio: float,
    min_solidity: float,
) -> Tuple[int, List[np.ndarray]]:
    """Keep contours within area bounds and above solidity threshold."""
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
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity  = area / hull_area if hull_area > 0 else 0.0
        if solidity < min_solidity:
            continue
        accepted.append(cnt)

    return len(accepted), accepted


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment_white_pills(
    bgr: np.ndarray,
    contour_mask: Optional[np.ndarray] = None,  # restrict search to this region
    brightness_floor: int  = 170,   # Otsu result is never allowed below this
    morph_open_iter: int   = 1,
    morph_close_iter: int  = 3,
    morph_kernel: int      = 9,
    separation: float      = 0.40,
    min_area_ratio: float  = 0.003,
    max_area_ratio: float  = 0.20,
    min_solidity: float    = 0.50,
) -> WhitePillResult:
    """
    Detect white pills by brightness thresholding inside the blister contour.

    Parameters
    ----------
    bgr : np.ndarray
        White-balance-corrected, rectified BGR image.
    contour_mask : np.ndarray or None
        Binary mask (255 = inside blister). Search is restricted to this region.
        If None, the entire image is searched.
    brightness_floor : int
        Minimum threshold value. Otsu auto-calibrates upward from this floor.
        Lower = more lenient (may include darker foil). Higher = more selective.
    morph_close_iter : int
        Iterations of morphological closing. Increase to bridge wider score lines.
    """
    debug: Dict[str, np.ndarray] = {}

    gray   = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    search = contour_mask if contour_mask is not None else np.ones_like(gray) * 255

    # ── Brightness threshold (Otsu within contour) ────────────────────────
    thresh = _otsu_in_mask(gray, search, brightness_floor)

    white_mask = (gray >= thresh).astype(np.uint8) * 255
    white_mask = cv2.bitwise_and(white_mask, search)

    # Debug: show binary mask with threshold value overlaid
    wm_bgr = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(wm_bgr, f"thresh={thresh}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2, cv2.LINE_AA)
    debug["white_mask"] = wm_bgr

    # ── Morphological cleanup + hole fill ────────────────────────────────
    cleaned = _morph_cleanup(white_mask, morph_open_iter, morph_close_iter, morph_kernel)
    cleaned = _fill_holes(cleaned)
    debug["cleaned_mask"] = cleaned

    # ── Watershed ─────────────────────────────────────────────────────────
    pill_mask, dist_vis = _watershed_separate(cleaned, bgr, separation)
    debug["distance"] = dist_vis

    # ── Contour filtering ─────────────────────────────────────────────────
    count, contours = _filter_contours(
        pill_mask,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
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
