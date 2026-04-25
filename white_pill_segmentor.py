"""
white_pill_segmentor.py
-----------------------
Dedicated detection pipeline for WHITE pills in blister packs.

Core insight: white pills and silver foil cannot be separated by color or
brightness alone — they occupy the same HSV/LAB neighborhood.  What CAN
separate them is LOCAL TEXTURE:

    • Foil surface  → knurled / embossed → HIGH local brightness variance
    • Pill surface  → smooth / matte     → LOW  local brightness variance

Pipeline:
    1. Build a local-standard-deviation map (texture map)
    2. Threshold low-texture regions  → pill candidates
    3. Morphological cleanup
    4. Contour filtering  (area + convexity — NOT circularity,
                           because oblong tablets are valid)
    5. Watershed separation for touching pills

Public API
----------
    segment_white_pills(bgr, **params) -> (pill_mask, contours, debug_images)

The debug_images dict is consumed by the main UI to fill the display panels.
All tunable parameters are exposed so the main file can wire them to sliders.
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
    pill_mask: np.ndarray                        # final binary mask, pills = 255
    debug_images: Dict[str, np.ndarray] = field(default_factory=dict)
    """
    debug_images keys (all same spatial size as input):
        'texture_map'   – float std-dev map visualised as 8-bit gray
        'texture_thresh'– binary: white = low-texture (pill candidate) regions
        'cleaned_mask'  – after morphological open+close
        'distance'      – distance-transform used for watershed seeds
        'annotated'     – colour image with contours drawn in red
    """


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _local_stddev_map(gray_f32: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Compute per-pixel local standard deviation using the identity:
        std²(x) = E[x²] - E[x]²

    This is O(n) via box-filter and far faster than cv2.StdDev on patches.

    Parameters
    ----------
    gray_f32 : float32 single-channel image (0–255 range)
    kernel_size : side length of the local neighbourhood (should be odd)

    Returns
    -------
    std_map : float32, same shape as input, values ≥ 0
    """
    k = max(3, kernel_size | 1)          # force odd
    mean    = cv2.blur(gray_f32,        (k, k))
    mean_sq = cv2.blur(gray_f32 ** 2,   (k, k))
    var     = mean_sq - mean ** 2
    return np.sqrt(np.maximum(var, 0.0))


def _build_texture_mask(
    bgr: np.ndarray,
    texture_kernel: int,
    texture_thresh: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (std_map_u8, texture_binary_mask).

    texture_binary_mask: 255 where local std < texture_thresh
                         (i.e. smooth regions = pill candidates)
    """
    # Work on the luminance channel – captures brightness variation cleanly
    lab   = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_f32 = lab[:, :, 0].astype(np.float32)

    std_map = _local_stddev_map(l_f32, texture_kernel)

    # Visualise for debug (stretch to 0–255)
    std_max = float(std_map.max()) or 1.0
    std_u8  = np.clip(std_map / std_max * 255, 0, 255).astype(np.uint8)

    # Low-texture mask: pill candidates
    low_tex = (std_map < texture_thresh).astype(np.uint8) * 255

    return std_u8, low_tex


def _remove_background(
    low_tex_mask: np.ndarray,
    bgr: np.ndarray,
    dark_val_max: int,
    bright_val_min: int,
) -> np.ndarray:
    """
    Subtract obviously-background pixels from the low-texture mask:
      - Very dark pixels  → shadows / pack edges (not pills)
      - Very bright pixels → specular highlights on foil (also smooth → false positives)

    White pills live in the mid-to-high brightness band.
    """
    hsv   = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v_ch  = hsv[:, :, 2]

    not_dark    = (v_ch > dark_val_max).astype(np.uint8) * 255
    not_specular= (v_ch < bright_val_min).astype(np.uint8) * 255

    return cv2.bitwise_and(low_tex_mask, cv2.bitwise_and(not_dark, not_specular))


def _morphological_cleanup(
    mask: np.ndarray,
    open_iter: int,
    close_iter: int,
    kernel_size: int,
) -> np.ndarray:
    """Open to kill foil-texture noise, close to fill pill-interior holes."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=open_iter)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=close_iter)
    return mask


def _watershed_separate(
    cleaned: np.ndarray,
    bgr: np.ndarray,
    separation: float,
) -> np.ndarray:
    """
    Apply watershed to split touching pills.
    Uses the same logic as the coloured-pill pipeline for consistency.
    """
    dist = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)

    if dist.max() < 1:
        return cleaned

    _, sure_fg = cv2.threshold(dist, separation * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    sure_bg = cv2.dilate(cleaned, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(bgr.copy(), markers)

    result = np.zeros_like(cleaned)
    result[markers > 1] = 255
    return result, dist


def _filter_contours(
    pill_mask: np.ndarray,
    bgr: np.ndarray,
    min_area_ratio: float,
    max_area_ratio: float,
    min_convexity: float,
    min_solidity: float,
) -> Tuple[int, List[np.ndarray]]:
    """
    Filter contours by:
        • area       – relative to total image area
        • convexity  – convex_hull_perimeter / actual_perimeter
                       (pills are convex; foil fragments are not)
        • solidity   – area / convex_hull_area
                       (another convexity proxy, robust to perimeter noise)

    Deliberately NOT filtering by circularity — oblong / oval tablets are fine.
    """
    contours, _ = cv2.findContours(
        pill_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    h, w = pill_mask.shape[:2]
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

        hull         = cv2.convexHull(cnt)
        hull_perim   = cv2.arcLength(hull, True)
        hull_area    = cv2.contourArea(hull)

        convexity = hull_perim / perimeter if perimeter > 0 else 0.0
        solidity  = area / hull_area       if hull_area  > 0 else 0.0

        if convexity < min_convexity:
            continue
        if solidity < min_solidity:
            continue

        accepted.append(cnt)

    return len(accepted), accepted


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment_white_pills(
    bgr: np.ndarray,
    # ── Texture analysis ──────────────────────────────────────────────────
    texture_kernel: int   = 15,     # local neighbourhood for std-dev (pixels)
    texture_thresh: float = 10.0,   # std-dev below this → smooth → pill candidate
    # ── Background removal ────────────────────────────────────────────────
    dark_val_max: int     = 60,     # exclude very dark pixels (shadow/edge)
    bright_val_min: int   = 240,    # exclude blown-out specular highlights
    # ── Morphological cleanup ─────────────────────────────────────────────
    morph_open_iter: int  = 2,      # iterations of opening  (noise removal)
    morph_close_iter: int = 3,      # iterations of closing  (hole filling)
    morph_kernel: int     = 7,      # element size for morph ops
    # ── Watershed separation ──────────────────────────────────────────────
    separation: float     = 0.50,   # fraction of dist-max for sure-foreground
    # ── Contour filtering ─────────────────────────────────────────────────
    min_area_ratio: float = 0.003,  # min pill area as fraction of image
    max_area_ratio: float = 0.25,   # max pill area (reject giant blobs)
    min_convexity: float  = 0.80,   # convex-hull-perimeter / perimeter ratio
    min_solidity: float   = 0.75,   # area / convex-hull-area ratio
) -> WhitePillResult:
    """
    Detect white pills in a white-balance-corrected, rectified blister-pack image.

    Parameters
    ----------
    bgr : np.ndarray
        White-balance-corrected BGR image (output of correct_white_balance()).
        Must already be rectified (output of _rectify_pack()).

    All other parameters are tunable and exposed to the UI as sliders.

    Returns
    -------
    WhitePillResult
        .pill_count     – number of detected pills
        .contours       – list of contours (for external drawing / further use)
        .pill_mask      – binary mask, 255 = pill, 0 = background
        .debug_images   – dict of intermediate images for UI display
    """
    debug: Dict[str, np.ndarray] = {}

    # ── Step 1: texture map ───────────────────────────────────────────────
    std_u8, low_tex = _build_texture_mask(bgr, texture_kernel, texture_thresh)
    debug["texture_map"]    = std_u8
    debug["texture_thresh"] = low_tex

    # ── Step 2: remove definite non-pill regions ──────────────────────────
    candidates = _remove_background(low_tex, bgr, dark_val_max, bright_val_min)

    # ── Step 3: morphological cleanup ────────────────────────────────────
    cleaned = _morphological_cleanup(
        candidates, morph_open_iter, morph_close_iter, morph_kernel
    )
    debug["cleaned_mask"] = cleaned

    # ── Step 4: watershed ─────────────────────────────────────────────────
    watershed_result = _watershed_separate(cleaned, bgr, separation)
    if isinstance(watershed_result, tuple):
        pill_mask, dist = watershed_result
        dist_max   = float(dist.max()) or 1.0
        dist_vis   = np.clip(dist / dist_max * 255, 0, 255).astype(np.uint8)
        debug["distance"] = dist_vis
    else:
        # dist.max() < 1 fallback — no watershed applied
        pill_mask = watershed_result
        debug["distance"] = np.zeros(bgr.shape[:2], dtype=np.uint8)

    # ── Step 5: contour filtering ─────────────────────────────────────────
    count, contours = _filter_contours(
        pill_mask, bgr,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        min_convexity=min_convexity,
        min_solidity=min_solidity,
    )

    # Build final mask from accepted contours only
    final_mask = np.zeros(pill_mask.shape, dtype=np.uint8)
    if contours:
        cv2.drawContours(final_mask, contours, -1, 255, -1)

    # Annotated colour image
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
