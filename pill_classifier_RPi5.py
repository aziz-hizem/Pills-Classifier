"""
pill_classifier_RPi5.py — Pill blister pack classifier for Raspberry Pi 5.

Demo command : python3 pill_classifier_RPi5.py --use-gpio --wait-key

GPIO outputs (BCM numbering):
  Red  LED : GPIO 17  — empty cells detected
  Blue LED : GPIO 27  — colour anomaly detected
  Green LED: GPIO 22  — all clear
  Servo    : GPIO 12  — angle encodes inspection result

Servo duty-cycle map (50 Hz, lgpio):
  All OK           →  7.5 % (≈  90°, centre)
  Empty cells only → 12.0 % (≈ 180°)
  Anomaly only     →  9.75% (≈ 135°)
  Both faults      →  3.0 % (≈   0°)

Usage:
  python pill_classifier_RPi5.py                       # dry-run, no GPIO
  python pill_classifier_RPi5.py --use-gpio            # real GPIO
  python pill_classifier_RPi5.py --use-gpio --wait-key # Pi demo (press Enter between images)
  python pill_classifier_RPi5.py --use-gpio --loop-delay 3  # auto-advance every 3 s
"""

import argparse
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import lgpio
    _HAS_LGPIO = True
except ImportError:
    _HAS_LGPIO = False


# ── Hardware configuration ────────────────────────────────────────────────────

PIN_LED_RED   = 17
PIN_LED_BLUE  = 27
PIN_LED_GREEN = 22
PIN_SERVO     = 12
PWM_FREQ      = 50  # Hz

# Duty cycles for a standard SG90 servo (adjust on your unit if needed)
_DUTY_0   = 3.0    # ≈   0°
_DUTY_90  = 7.5    # ≈  90°  (neutral / all-clear)
_DUTY_135 = 9.75   # ≈ 135°
_DUTY_180 = 12.0   # ≈ 180°

# Servo position per inspection outcome
_SERVO_DUTY: Dict[str, float] = {
    "OK":           _DUTY_90,
    "EMPTY_ONLY":   _DUTY_180,
    "ANOMALY_ONLY": _DUTY_135,
    "BOTH":         _DUTY_0,
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ProcessedImages:
    original_bgr: np.ndarray
    white_balanced_bgr: np.ndarray
    color_debug_mask: np.ndarray
    mask: np.ndarray
    binary: np.ndarray
    annotated_bgr: np.ndarray
    pill_count: int
    color_class: "PillColorClass"
    median_saturation: float
    sample_pixel_ratio: float
    color_name: str
    white_debug_images: Optional[Dict[str, np.ndarray]] = field(default=None)
    empty_cell_count: int = 0
    empty_cell_contours: list = field(default_factory=list)
    color_anomaly: bool = False


class PillColorClass(Enum):
    WHITE   = auto()
    COLORED = auto()
    UNKNOWN = auto()


@dataclass
class ColorClassificationResult:
    color_class: PillColorClass
    median_saturation: float
    dominant_hue: float
    sample_pixel_ratio: float
    debug_mask: np.ndarray


@dataclass
class WhitePillResult:
    pill_count: int
    contours: List[np.ndarray]
    pill_mask: np.ndarray
    debug_images: Dict[str, np.ndarray] = field(default_factory=dict)


class InspectionStatus(Enum):
    OK           = "OK"
    EMPTY_ONLY   = "EMPTY_ONLY"
    ANOMALY_ONLY = "ANOMALY_ONLY"
    BOTH         = "BOTH"


# ── GPIO controller ───────────────────────────────────────────────────────────

class GPIOController:
    """
    Wraps lgpio for LED + servo control.
    Becomes a no-op (dry-run) when lgpio is missing or --no-gpio is passed.
    """

    def __init__(self, dry_run: bool = False) -> None:
        self._dry_run = dry_run or not _HAS_LGPIO
        self._h: Optional[int] = None
        if not self._dry_run:
            self._h = lgpio.gpiochip_open(0)
            for pin in (PIN_LED_RED, PIN_LED_BLUE, PIN_LED_GREEN):
                lgpio.gpio_claim_output(self._h, pin, 0)
            lgpio.gpio_claim_output(self._h, PIN_SERVO, 0)
            self.apply(InspectionStatus.OK)

    def set_leds(self, red: bool, blue: bool, green: bool) -> None:
        if self._dry_run:
            print(f"  [GPIO dry-run] LEDs  red={red}  blue={blue}  green={green}")
            return
        lgpio.gpio_write(self._h, PIN_LED_RED,   int(red))
        lgpio.gpio_write(self._h, PIN_LED_BLUE,  int(blue))
        lgpio.gpio_write(self._h, PIN_LED_GREEN, int(green))

    def set_servo(self, duty: float) -> None:
        if self._dry_run:
            print(f"  [GPIO dry-run] Servo duty={duty}%")
            return
        lgpio.tx_pwm(self._h, PIN_SERVO, PWM_FREQ, duty)

    def apply(self, status: InspectionStatus) -> None:
        s     = status.value
        duty  = _SERVO_DUTY[s]
        red   = s in ("EMPTY_ONLY", "BOTH")
        blue  = s in ("ANOMALY_ONLY", "BOTH")
        green = s == "OK"
        self.set_leds(red, blue, green)
        self.set_servo(duty)
        print(f"  → {s}  LEDs(R={red}, B={blue}, G={green})  servo={duty}%")

    def close(self) -> None:
        if self._dry_run or self._h is None:
            return
        for pin in (PIN_LED_RED, PIN_LED_BLUE, PIN_LED_GREEN):
            lgpio.gpio_write(self._h, pin, 0)
        lgpio.tx_pwm(self._h, PIN_SERVO, 0, 0)
        lgpio.gpiochip_close(self._h)
        self._h = None


# ── Status resolution ─────────────────────────────────────────────────────────

def resolve_status(processed: ProcessedImages) -> InspectionStatus:
    empty   = processed.empty_cell_count > 0
    anomaly = processed.color_anomaly
    if empty and anomaly:
        return InspectionStatus.BOTH
    if empty:
        return InspectionStatus.EMPTY_ONLY
    if anomaly:
        return InspectionStatus.ANOMALY_ONLY
    return InspectionStatus.OK


# ── White pill segmentor ──────────────────────────────────────────────────────

def _otsu_in_mask(gray: np.ndarray, mask: np.ndarray, floor: int) -> int:
    pixels = gray[mask == 255]
    if len(pixels) < 200:
        return floor
    tmp = pixels.reshape(-1, 1)
    otsu_t, _ = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return max(int(otsu_t), floor)


def _morph_cleanup(
    mask: np.ndarray, open_iter: int, close_iter: int, kernel_size: int
) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=open_iter)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=close_iter)
    return mask


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    inv   = cv2.bitwise_not(mask)
    h, w  = mask.shape[:2]
    flood = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(inv, flood, (0, 0), 255)
    return cv2.bitwise_or(mask, cv2.bitwise_not(inv))


def _watershed_separate(
    candidate_mask: np.ndarray, bgr: np.ndarray, separation: float
) -> Tuple[np.ndarray, np.ndarray]:
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
    contours, _ = cv2.findContours(pill_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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


def segment_white_pills(
    bgr: np.ndarray,
    contour_mask: Optional[np.ndarray] = None,
    brightness_floor: int  = 170,
    morph_open_iter: int   = 1,
    morph_close_iter: int  = 3,
    morph_kernel: int      = 9,
    separation: float      = 0.40,
    min_area_ratio: float  = 0.003,
    max_area_ratio: float  = 0.20,
    min_solidity: float    = 0.50,
) -> WhitePillResult:
    debug: Dict[str, np.ndarray] = {}
    gray   = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    search = contour_mask if contour_mask is not None else np.ones_like(gray) * 255
    thresh = _otsu_in_mask(gray, search, brightness_floor)
    white_mask = (gray >= thresh).astype(np.uint8) * 255
    white_mask = cv2.bitwise_and(white_mask, search)
    wm_bgr = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(wm_bgr, f"thresh={thresh}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2, cv2.LINE_AA)
    debug["white_mask"] = wm_bgr
    cleaned = _morph_cleanup(white_mask, morph_open_iter, morph_close_iter, morph_kernel)
    cleaned = _fill_holes(cleaned)
    debug["cleaned_mask"] = cleaned
    pill_mask, dist_vis = _watershed_separate(cleaned, bgr, separation)
    debug["distance"] = dist_vis
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
        pill_count=count, contours=contours, pill_mask=final_mask, debug_images=debug
    )


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _order_points(points: np.ndarray) -> np.ndarray:
    ordered = np.zeros((4, 2), dtype=np.float32)
    sum_values = points.sum(axis=1)
    ordered[0] = points[np.argmin(sum_values)]
    ordered[2] = points[np.argmax(sum_values)]
    diff_values = np.diff(points, axis=1)
    ordered[1] = points[np.argmin(diff_values)]
    ordered[3] = points[np.argmax(diff_values)]
    return ordered


def _four_point_warp(
    bgr: np.ndarray, points: np.ndarray, output_size: Tuple[int, int]
) -> np.ndarray:
    ordered = _order_points(points)
    width, height = output_size
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(ordered, destination)
    return cv2.warpPerspective(bgr, transform, (width, height))


def _resize_keep_aspect(bgr: np.ndarray, max_size: Tuple[int, int]) -> np.ndarray:
    h, w = bgr.shape[:2]
    max_w, max_h = max_size
    if h <= 0 or w <= 0:
        return bgr.copy()
    scale = min(max_w / w, max_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(bgr, (new_w, new_h), interpolation=interp)


# ── Pack rectification ────────────────────────────────────────────────────────

def _rectify_pack(bgr: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    resize_width = 700
    scale = resize_width / bgr.shape[1]
    resized = cv2.resize(bgr, (resize_width, int(bgr.shape[0] * scale)))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    blur = cv2.GaussianBlur(v_channel, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _resize_keep_aspect(bgr, output_size)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    pack_contour = contours[0]
    image_area = resized.shape[0] * resized.shape[1]
    if cv2.contourArea(pack_contour) < 0.2 * image_area:
        return _resize_keep_aspect(bgr, output_size)
    rect = cv2.minAreaRect(pack_contour)
    box = cv2.boxPoints(rect)
    points = box.reshape(-1, 2) / scale
    ordered = _order_points(points.astype(np.float32))
    (tl, tr, br, bl) = ordered
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    warp_width  = int(round(max(width_a, width_b)))
    warp_height = int(round(max(height_a, height_b)))
    if warp_width <= 1 or warp_height <= 1:
        return _resize_keep_aspect(bgr, output_size)
    warped = _four_point_warp(bgr, points.astype(np.float32), (warp_width, warp_height))
    target_landscape = output_size[0] >= output_size[1]
    warp_landscape   = warped.shape[1] >= warped.shape[0]
    if target_landscape != warp_landscape:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return _resize_keep_aspect(warped, output_size)


# ── Blister contour detection ─────────────────────────────────────────────────

def detect_blister_contour(bgr: np.ndarray) -> Optional[np.ndarray]:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_blur = cv2.GaussianBlur(lab[:, :, 0], (7, 7), 0)
    adaptive = cv2.adaptiveThreshold(
        l_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 2
    )
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, not_white = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.bitwise_and(adaptive, not_white)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN,  kernel, iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    hull    = cv2.convexHull(largest)
    rect    = cv2.minAreaRect(hull)
    box     = cv2.boxPoints(rect)
    return box.astype("int32").reshape(-1, 1, 2)


# ── Empty cell detection ──────────────────────────────────────────────────────

def detect_empty_cells(
    bgr: np.ndarray,
    blister_contour: Optional[np.ndarray],
    pill_contours: list,
) -> Tuple[int, list]:
    if not pill_contours:
        return 0, []

    h, w = bgr.shape[:2]
    areas      = [cv2.contourArea(c) for c in pill_contours]
    avg_area   = float(np.mean(areas))
    if avg_area <= 0:
        return 0, []
    avg_radius = max(5, int(np.sqrt(avg_area / np.pi)))

    centers: list = []
    for cnt in pill_contours:
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            centers.append((float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"])))
    if len(centers) < 2:
        return 0, []

    if blister_contour is not None:
        raw_dists = [
            cv2.pointPolygonTest(blister_contour, (px, py), measureDist=True)
            for px, py in centers
        ]
        valid_dists    = [d for d in raw_dists if d > 0]
        min_border_dist = float(min(valid_dists)) if valid_dists else 0.0
    else:
        min_border_dist = 0.0

    def _inside(x: float, y: float) -> bool:
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < w and 0 <= yi < h):
            return False
        if blister_contour is None:
            return True
        return cv2.pointPolygonTest(blister_contour, (x, y), measureDist=True) >= min_border_dist

    row_tol = avg_radius * 0.9
    by_y    = sorted(centers, key=lambda c: c[1])
    rows: list = [[by_y[0]]]
    for cx, cy in by_y[1:]:
        row_mean_y = float(np.mean([c[1] for c in rows[-1]]))
        if abs(cy - row_mean_y) <= row_tol:
            rows[-1].append((cx, cy))
        else:
            rows.append([(cx, cy)])
    rows = [sorted(r, key=lambda c: c[0]) for r in rows]

    per_row_min_gap: list = []
    for row in rows:
        if len(row) >= 2:
            gaps = [row[i][0] - row[i - 1][0] for i in range(1, len(row))]
            per_row_min_gap.append(min(gaps))
    if not per_row_min_gap:
        return 0, []
    col_spacing = float(np.median(per_row_min_gap))
    if col_spacing < avg_radius * 1.2:
        return 0, []

    row_ys      = [float(np.mean([c[1] for c in r])) for r in rows]
    row_spacing = float(np.median(np.diff(row_ys))) if len(row_ys) >= 2 else col_spacing
    if row_spacing < avg_radius * 1.2:
        row_spacing = col_spacing

    all_x = sorted(c[0] for c in centers)
    col_groups: list = [[all_x[0]]]
    for x in all_x[1:]:
        if x - float(np.mean(col_groups[-1])) <= col_spacing * 0.45:
            col_groups[-1].append(x)
        else:
            col_groups.append([x])
    col_pos = [float(np.mean(g)) for g in col_groups]

    full_row_ys: list = [row_ys[0]]
    for y in row_ys[1:]:
        n_fill = round((y - full_row_ys[-1]) / row_spacing) - 1
        for _ in range(n_fill):
            full_row_ys.append(full_row_ys[-1] + row_spacing)
        full_row_ys.append(y)

    mean_col_x = float(np.mean(col_pos))
    for direction in (-1, 1):
        y = (full_row_ys[0] if direction == -1 else full_row_ys[-1]) + direction * row_spacing
        while True:
            if not _inside(mean_col_x, y):
                break
            if direction == -1:
                full_row_ys.insert(0, y)
            else:
                full_row_ys.append(y)
            y += direction * row_spacing

    mean_row_y = float(np.mean(full_row_ys))
    for direction in (-1, 1):
        x = (col_pos[0] if direction == -1 else col_pos[-1]) + direction * col_spacing
        while True:
            if not _inside(x, mean_row_y):
                break
            if direction == -1:
                col_pos.insert(0, x)
            else:
                col_pos.append(x)
            x += direction * col_spacing

    hsv     = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s_ch    = hsv[:, :, 1]
    v_ch    = hsv[:, :, 2]
    match_r  = col_spacing * 0.45
    r_sample = max(3, avg_radius // 4)

    empty_contours: list = []
    for grid_y in full_row_ys:
        for grid_x in col_pos:
            if any(
                abs(px - grid_x) <= match_r and abs(py - grid_y) <= row_tol
                for px, py in centers
            ):
                continue
            if any(
                cv2.pointPolygonTest(cnt, (float(grid_x), float(grid_y)), measureDist=True) >= -avg_radius
                for cnt in pill_contours
            ):
                continue
            if not _inside(grid_x, grid_y):
                continue
            cx_i = int(round(grid_x))
            cy_i = int(round(grid_y))
            y0 = max(0, cy_i - r_sample)
            y1 = min(h, cy_i + r_sample + 1)
            x0 = max(0, cx_i - r_sample)
            x1 = min(w, cx_i + r_sample + 1)
            roi_s = s_ch[y0:y1, x0:x1]
            roi_v = v_ch[y0:y1, x0:x1]
            if roi_s.size == 0:
                continue
            med_s = float(np.median(roi_s))
            med_v = float(np.median(roi_v))
            if med_s > 70 or med_v < 50 or med_v > 220:
                continue
            tmp = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(tmp, (cx_i, cy_i), avg_radius, 255, -1)
            cnts, _ = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                empty_contours.append(cnts[0])

    return len(empty_contours), empty_contours


# ── Color anomaly detection ───────────────────────────────────────────────────

def detect_color_anomaly(
    bgr: np.ndarray,
    pill_contours: list,
    color_class: PillColorClass,
    hue_threshold: float   = 25.0,
    min_sat_for_hue: int   = 40,
    min_valid_pixels: int  = 20,
    white_sat_spike: float   = 40.0,
    colored_sat_spike: float = 45.0,
) -> bool:
    if color_class == PillColorClass.UNKNOWN or len(pill_contours) < 2:
        return False
    hsv      = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_chan   = hsv[:, :, 0]
    s_chan   = hsv[:, :, 1]
    mask_buf = np.zeros(bgr.shape[:2], dtype=np.uint8)
    per_pill_hues: list = []
    per_pill_sats: list = []
    for cnt in pill_contours:
        mask_buf[:] = 0
        cv2.drawContours(mask_buf, [cnt], -1, 255, -1)
        pixels = mask_buf == 255
        if int(pixels.sum()) < min_valid_pixels:
            continue
        per_pill_sats.append(float(np.median(s_chan[pixels])))
        if color_class == PillColorClass.COLORED:
            valid = pixels & (s_chan >= min_sat_for_hue)
            if int(valid.sum()) < min_valid_pixels:
                continue
            hues   = h_chan[valid]
            angles = hues * (2.0 * np.pi / 180.0)
            circ   = float(np.angle(np.mean(np.exp(1j * angles))) * 180.0 / (2.0 * np.pi))
            if circ < 0:
                circ += 180.0
            per_pill_hues.append(circ)
    if color_class == PillColorClass.COLORED:
        if len(per_pill_sats) >= 2:
            median_sat = float(np.median(per_pill_sats))
            if any(abs(s - median_sat) > colored_sat_spike for s in per_pill_sats):
                return True
        if len(per_pill_hues) < 2:
            return False
        pill_names = [hue_to_color_name(h) for h in per_pill_hues]
        if len(set(pill_names)) > 1:
            return True
        angles = np.array(per_pill_hues) * (2.0 * np.pi / 180.0)
        consensus_angle = float(np.angle(np.mean(np.exp(1j * angles))))
        consensus_hue   = consensus_angle * 180.0 / (2.0 * np.pi)
        if consensus_hue < 0:
            consensus_hue += 180.0
        for h in per_pill_hues:
            d = abs(h - consensus_hue)
            if min(d, 180.0 - d) > hue_threshold:
                return True
        return False
    else:
        if len(per_pill_sats) < 2:
            return False
        median_sat = float(np.median(per_pill_sats))
        return any(s > median_sat + white_sat_spike for s in per_pill_sats)


# ── Color helpers ─────────────────────────────────────────────────────────────

def hue_to_color_name(hue_degrees: float) -> str:
    h = hue_degrees * 2.0
    if h < 15 or h >= 345:   return "red"
    if h < 38:                return "orange"
    if h < 70:                return "yellow"
    if h < 155:               return "green"
    if h < 195:               return "cyan"
    if h < 255:               return "blue"
    if h < 285:               return "purple"
    if h < 345:               return "pink/magenta"
    return "unknown"


def _pill_color_letters(
    bgr: np.ndarray,
    pill_contours: list,
    color_class: PillColorClass,
    min_sat_for_hue: int  = 40,
    min_valid_pixels: int = 20,
) -> list:
    if not pill_contours:
        return []
    hsv      = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_chan   = hsv[:, :, 0]
    s_chan   = hsv[:, :, 1]
    mask_buf = np.zeros(bgr.shape[:2], dtype=np.uint8)
    letters  = []
    for cnt in pill_contours:
        mask_buf[:] = 0
        cv2.drawContours(mask_buf, [cnt], -1, 255, -1)
        pixels = mask_buf == 255
        if int(pixels.sum()) < min_valid_pixels:
            letters.append("?")
            continue
        if color_class == PillColorClass.WHITE:
            if float(np.median(s_chan[pixels])) < min_sat_for_hue:
                letters.append("W")
                continue
        valid = pixels & (s_chan >= min_sat_for_hue)
        if int(valid.sum()) < min_valid_pixels:
            letters.append("W" if color_class == PillColorClass.WHITE else "?")
            continue
        hues   = h_chan[valid]
        angles = hues * (2.0 * np.pi / 180.0)
        circ   = float(np.angle(np.mean(np.exp(1j * angles))) * 180.0 / (2.0 * np.pi))
        if circ < 0:
            circ += 180.0
        letters.append(hue_to_color_name(circ)[0].upper())
    return letters


def _draw_pill_letters(image: np.ndarray, contours: list, letters: list) -> None:
    for cnt, letter in zip(contours, letters):
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        _, _, w, h = cv2.boundingRect(cnt)
        scale = max(0.3, min(0.9, min(w, h) / 50.0))
        (tw, th), _ = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        cv2.putText(image, letter, (cx - tw // 2, cy + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)


def _contour_mask(shape: Tuple[int, int, int], contour: Optional[np.ndarray]) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if contour is None:
        return mask
    cv2.drawContours(mask, [contour], -1, 255, -1)
    return mask


def _filter_white_contours_by_size(
    contours: list,
    image_shape: Tuple[int, int, int],
    min_short_ratio: float   = 0.015,
    max_long_ratio: float    = 0.40,
    max_aspect_ratio: float  = 4.0,
    median_low: float        = 0.5,
    median_high: float       = 1.8,
    min_for_median_filter: int = 4,
) -> list:
    height, width = image_shape[:2]
    max_dim       = float(max(height, width))
    candidates    = []
    for contour in contours:
        (_, _), (w, h), _ = cv2.minAreaRect(contour)
        long_side  = float(max(w, h))
        short_side = float(min(w, h))
        if short_side <= 1e-6:
            continue
        if short_side < min_short_ratio * float(height):
            continue
        if long_side > max_long_ratio * max_dim:
            continue
        if long_side / short_side > max_aspect_ratio:
            continue
        candidates.append((contour, short_side, long_side))
    if len(candidates) < min_for_median_filter:
        return [c for c, _, _ in candidates]
    short_sides   = np.array([s for _, s, _ in candidates], dtype=np.float32)
    long_sides    = np.array([l for _, _, l in candidates], dtype=np.float32)
    median_short  = float(np.median(short_sides))
    median_long   = float(np.median(long_sides))
    return [
        c for c, s, l in candidates
        if (median_low * median_short <= s <= median_high * median_short
            and median_low * median_long  <= l <= median_high * median_long)
    ]


# ── White balance correction ──────────────────────────────────────────────────

def correct_white_balance(
    rectified_bgr: np.ndarray,
    foil_val_min: int   = 80,
    foil_val_max: int   = 230,
    foil_sat_max: int   = 60,
    target_gray: int    = 180,
    min_foil_ratio: float = 0.10,
) -> Tuple[np.ndarray, bool]:
    bgr_f = rectified_bgr.astype(np.float32)
    hsv   = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2HSV)
    v_ch  = hsv[:, :, 2].astype(np.float32)
    s_ch  = hsv[:, :, 1].astype(np.float32)
    foil_mask  = (v_ch >= foil_val_min) & (v_ch <= foil_val_max) & (s_ch <= foil_sat_max)
    foil_count = int(np.sum(foil_mask))
    if foil_count / (rectified_bgr.shape[0] * rectified_bgr.shape[1]) < min_foil_ratio:
        return rectified_bgr.copy(), False
    mean_b = float(np.mean(bgr_f[:, :, 0][foil_mask]))
    mean_g = float(np.mean(bgr_f[:, :, 1][foil_mask]))
    mean_r = float(np.mean(bgr_f[:, :, 2][foil_mask]))
    if mean_b < 1 or mean_g < 1 or mean_r < 1:
        return rectified_bgr.copy(), False
    if mean_r - mean_b > 40 and mean_r - mean_g > 25:
        return rectified_bgr.copy(), False
    gain_b = target_gray / mean_b
    gain_g = target_gray / mean_g
    gain_r = target_gray / mean_r
    corrected_f = bgr_f.copy()
    corrected_f[:, :, 0] = np.clip(bgr_f[:, :, 0] * gain_b, 0, 255)
    corrected_f[:, :, 1] = np.clip(bgr_f[:, :, 1] * gain_g, 0, 255)
    corrected_f[:, :, 2] = np.clip(bgr_f[:, :, 2] * gain_r, 0, 255)
    return corrected_f.astype(np.uint8), True


# ── Color classification ──────────────────────────────────────────────────────

def classify_pill_color(
    rectified_bgr: np.ndarray,
    dark_val_max: int           = 40,
    colored_sat_threshold: int  = 45,
    colored_area_threshold: float = 0.04,
) -> ColorClassificationResult:
    hsv   = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2HSV)
    h_ch  = hsv[:, :, 0].astype(np.float32)
    s_ch  = hsv[:, :, 1].astype(np.float32)
    v_ch  = hsv[:, :, 2].astype(np.float32)
    not_dark_mask  = v_ch > dark_val_max
    not_dark_count = int(np.sum(not_dark_mask))
    sample_ratio   = not_dark_count / (rectified_bgr.shape[0] * rectified_bgr.shape[1])
    debug_mask = np.zeros(rectified_bgr.shape[:2], dtype=np.uint8)
    debug_mask[not_dark_mask] = 100
    if not_dark_count == 0:
        return ColorClassificationResult(
            color_class=PillColorClass.UNKNOWN, median_saturation=-1.0,
            dominant_hue=-1.0, sample_pixel_ratio=sample_ratio, debug_mask=debug_mask,
        )
    colored_pixels_mask = not_dark_mask & (s_ch > colored_sat_threshold)
    colored_ratio = int(np.sum(colored_pixels_mask)) / not_dark_count
    debug_mask[colored_pixels_mask] = 255
    median_sat = float(np.median(s_ch[not_dark_mask]))
    if colored_ratio > colored_area_threshold:
        pill_hues = h_ch[colored_pixels_mask]
        pill_sats = s_ch[colored_pixels_mask]
        hue_hist  = np.zeros(180, dtype=np.float32)
        np.add.at(hue_hist, np.clip(pill_hues.astype(np.int32), 0, 179), pill_sats)
        return ColorClassificationResult(
            color_class=PillColorClass.COLORED, median_saturation=median_sat,
            dominant_hue=float(np.argmax(hue_hist)),
            sample_pixel_ratio=sample_ratio, debug_mask=debug_mask,
        )
    return ColorClassificationResult(
        color_class=PillColorClass.WHITE, median_saturation=median_sat,
        dominant_hue=-1.0, sample_pixel_ratio=sample_ratio, debug_mask=debug_mask,
    )


# ── Colored-pill segmentation & counting ─────────────────────────────────────

def _segment_pills(
    bgr: np.ndarray,
    clahe_clip: float, clahe_tile: int,
    sat_thresh: float, val_dark_thresh: float,
    adaptive_block: int, threshold_bias: float,
    separation: float,
) -> np.ndarray:
    lab     = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe   = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
    l_eq    = clahe.apply(lab[:, :, 0])
    blur    = cv2.GaussianBlur(l_eq, (5, 5), 0)
    block   = max(3, adaptive_block | 1)
    adaptive = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        block, threshold_bias,
    )
    hsv       = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]
    thresh = cv2.bitwise_or(adaptive, (s_channel >= sat_thresh).astype(np.uint8) * 255)
    thresh = cv2.bitwise_or(thresh,   (v_channel <= val_dark_thresh).astype(np.uint8) * 255)
    h, w   = thresh.shape
    border = np.zeros_like(thresh, dtype=np.uint8)
    mh, mw = max(1, int(h * 0.05)), max(1, int(w * 0.05))
    border[:mh, :] = border[-mh:, :] = border[:, :mw] = border[:, -mw:] = 1
    pill_mask = cv2.bitwise_not(thresh) if float(np.mean(thresh[border == 1] == 255)) > 0.5 else thresh.copy()
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    pill_mask = cv2.morphologyEx(pill_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    pill_mask = cv2.morphologyEx(pill_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    dist = cv2.distanceTransform(pill_mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, separation * dist.max(), 255, 0)
    sure_fg  = sure_fg.astype(np.uint8)
    sure_bg  = cv2.dilate(pill_mask, kernel, iterations=2)
    unknown  = cv2.subtract(sure_bg, sure_fg)
    markers  = cv2.connectedComponents(sure_fg)[1] + 1
    markers[unknown == 255] = 0
    markers  = cv2.watershed(bgr.copy(), markers)
    refined  = np.zeros_like(pill_mask)
    refined[markers > 1] = 255
    return refined


def _count_pills(
    pill_mask: np.ndarray, ref_bgr: np.ndarray,
    min_area_ratio: float, min_circularity: float, stddev_max: float,
) -> Tuple[int, list]:
    contours, _ = cv2.findContours(pill_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = pill_mask.shape[:2]
    min_area      = min_area_ratio * (height * width)
    pill_contours = []
    hsv = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2HSV)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        if min_circularity > 0 and 4.0 * np.pi * area / (perimeter * perimeter) < min_circularity:
            continue
        if stddev_max > 0:
            mask = np.zeros(pill_mask.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            if cv2.meanStdDev(hsv[:, :, 2], mask=mask)[1][0][0] > stddev_max:
                continue
        pill_contours.append(contour)
    return len(pill_contours), pill_contours


# ── Main processing pipeline ──────────────────────────────────────────────────

_DEFAULTS = dict(
    clahe_clip=2.0,     clahe_tile=8,
    sat_thresh=35.0,    val_dark_thresh=80.0,
    adaptive_block=35,  threshold_bias=15.0,
    separation=0.45,
    min_area_ratio=0.002,  min_circularity=0.15,  stddev_max=0.0,
    morph_open=1,       morph_close=3,
    wp_separation=0.40, wp_min_area=0.30,   wp_solidity=0.50,
    wp_brightness_floor=205,
)


def process_image(image_path: Path, **kwargs) -> ProcessedImages:
    p   = {**_DEFAULTS, **kwargs}
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    rectified        = _rectify_pack(bgr, (800, 500))
    white_balanced, _ = correct_white_balance(rectified)
    color_result     = classify_pill_color(white_balanced)
    blister_contour  = detect_blister_contour(rectified)
    contour_mask     = _contour_mask(rectified.shape, blister_contour)

    pill_mask      = np.zeros(rectified.shape[:2], dtype=np.uint8)
    count          = 0
    contours: list = []
    annotated      = white_balanced.copy()
    white_debug: Dict[str, np.ndarray] = {}
    empty_count    = 0
    empty_contours: list = []

    if color_result.color_class == PillColorClass.COLORED:
        pill_mask = _segment_pills(
            white_balanced,
            p["clahe_clip"], p["clahe_tile"],
            p["sat_thresh"], p["val_dark_thresh"],
            p["adaptive_block"], p["threshold_bias"],
            p["separation"],
        )
        count, contours = _count_pills(
            pill_mask, white_balanced,
            min_area_ratio=p["min_area_ratio"],
            min_circularity=p["min_circularity"],
            stddev_max=p["stddev_max"],
        )
        if blister_contour is not None:
            cv2.drawContours(annotated, [blister_contour], -1, (255, 0, 0), 3)
        if contours:
            cv2.drawContours(annotated, contours, -1, (0, 0, 255), 2)
            _draw_pill_letters(annotated, contours,
                               _pill_color_letters(white_balanced, contours, color_result.color_class))
        empty_count, empty_contours = detect_empty_cells(white_balanced, blister_contour, contours)
        if empty_contours:
            cv2.drawContours(annotated, empty_contours, -1, (0, 165, 255), 3)

    elif color_result.color_class == PillColorClass.WHITE:
        search_mask = (
            contour_mask if blister_contour is not None
            else np.ones(white_balanced.shape[:2], dtype=np.uint8) * 255
        )
        wp_result = segment_white_pills(
            white_balanced,
            contour_mask     = search_mask,
            brightness_floor = int(p["wp_brightness_floor"]),
            morph_open_iter  = p["morph_open"],
            morph_close_iter = p["morph_close"],
            separation       = p["wp_separation"],
            min_area_ratio   = p["wp_min_area"] / 100.0,
            min_solidity     = p["wp_solidity"],
        )
        pill_mask   = wp_result.pill_mask
        contours, _ = cv2.findContours(pill_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours    = _filter_white_contours_by_size(contours, white_balanced.shape)
        count       = len(contours)
        pill_mask   = np.zeros_like(pill_mask)
        if contours:
            cv2.drawContours(pill_mask, contours, -1, 255, -1)
        annotated = white_balanced.copy()
        if blister_contour is not None:
            cv2.drawContours(annotated, [blister_contour], -1, (255, 0, 0), 3)
        if contours:
            cv2.drawContours(annotated, contours, -1, (0, 0, 255), 2)
            _draw_pill_letters(annotated, contours,
                               _pill_color_letters(white_balanced, contours, color_result.color_class))
        empty_count, empty_contours = detect_empty_cells(white_balanced, blister_contour, contours)
        if empty_contours:
            cv2.drawContours(annotated, empty_contours, -1, (0, 165, 255), 3)
        white_debug = wp_result.debug_images

    binary     = cv2.bitwise_not(pill_mask)
    color_name = (
        hue_to_color_name(color_result.dominant_hue)
        if color_result.color_class == PillColorClass.COLORED else ""
    )
    color_anomaly = detect_color_anomaly(white_balanced, contours, color_result.color_class)

    return ProcessedImages(
        original_bgr        = bgr,
        white_balanced_bgr  = white_balanced,
        color_debug_mask    = color_result.debug_mask,
        mask                = pill_mask,
        binary              = binary,
        annotated_bgr       = annotated,
        pill_count          = count,
        color_class         = color_result.color_class,
        median_saturation   = color_result.median_saturation,
        sample_pixel_ratio  = color_result.sample_pixel_ratio,
        color_name          = color_name,
        white_debug_images  = white_debug or None,
        empty_cell_count    = empty_count,
        empty_cell_contours = empty_contours,
        color_anomaly       = color_anomaly,
    )


# ── Headless runner ───────────────────────────────────────────────────────────

def run_headless(
    image_dir: Path,
    gpio: GPIOController,
    loop_delay: float,
    wait_key: bool,
) -> None:
    paths: list = []
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
        paths.extend(sorted(image_dir.glob(pattern)))
    if not paths:
        raise SystemExit(f"No images found in {image_dir}")

    print(f"Headless mode — {len(paths)} image(s) in {image_dir}")
    print("Ctrl-C to quit.\n")

    try:
        for idx, path in enumerate(paths):
            print(f"[{idx + 1}/{len(paths)}] {path.name}")
            try:
                result = process_image(path)
            except Exception as exc:
                print(f"  ERROR processing image: {exc}")
                continue

            status = resolve_status(result)
            print(
                f"  Pills={result.pill_count}  "
                f"Empty={result.empty_cell_count}  "
                f"Anomaly={result.color_anomaly}  "
                f"PillColor={result.color_class.name}"
            )
            gpio.apply(status)

            if wait_key:
                try:
                    input("  [Enter] next image…")
                except EOFError:
                    break
            elif loop_delay > 0:
                time.sleep(loop_delay)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        gpio.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pill classifier for Raspberry Pi 5 (headless)")
    p.add_argument("--use-gpio",   action="store_true",
                   help="Enable real GPIO output (LEDs + servo); requires lgpio")
    p.add_argument("--no-gpio",    action="store_true",
                   help="Force GPIO dry-run even if --use-gpio is set")
    p.add_argument("--loop-delay", type=float, default=2.0, metavar="SEC",
                   help="Seconds to hold each result before moving to next image (default 2)")
    p.add_argument("--wait-key",   action="store_true",
                   help="Wait for Enter between images instead of --loop-delay")
    p.add_argument("--images-dir", default="", metavar="DIR",
                   help="Folder with input images (default: images/ next to this script)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    images_dir = (
        Path(args.images_dir) if args.images_dir
        else Path(__file__).resolve().parent / "images"
    )

    dry_run = args.no_gpio or (not args.use_gpio)
    gpio    = GPIOController(dry_run=dry_run)

    run_headless(images_dir, gpio, loop_delay=args.loop_delay, wait_key=args.wait_key)


if __name__ == "__main__":
    main()
