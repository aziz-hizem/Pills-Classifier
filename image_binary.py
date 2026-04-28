import argparse
import base64
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk

from white_pill_segmentor import segment_white_pills


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


class PillColorClass(Enum):
	WHITE = auto()
	COLORED = auto()
	UNKNOWN = auto()


@dataclass
class ColorClassificationResult:
	color_class: PillColorClass
	median_saturation: float
	dominant_hue: float
	sample_pixel_ratio: float
	debug_mask: np.ndarray


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
	bgr: np.ndarray,
	points: np.ndarray,
	output_size: Tuple[int, int],
) -> np.ndarray:
	ordered = _order_points(points)
	width, height = output_size
	destination = np.array(
		[[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
		dtype=np.float32,
	)
	transform = cv2.getPerspectiveTransform(ordered, destination)
	return cv2.warpPerspective(bgr, transform, (width, height))


def _resize_keep_aspect(
	bgr: np.ndarray,
	max_size: Tuple[int, int],
) -> np.ndarray:
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
	warp_width = int(round(max(width_a, width_b)))
	warp_height = int(round(max(height_a, height_b)))
	if warp_width <= 1 or warp_height <= 1:
		return _resize_keep_aspect(bgr, output_size)

	warped = _four_point_warp(bgr, points.astype(np.float32), (warp_width, warp_height))

	# Rotation is allowed; use it to match target orientation without stretching.
	target_landscape = output_size[0] >= output_size[1]
	warp_landscape = warped.shape[1] >= warped.shape[0]
	if target_landscape != warp_landscape:
		warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

	return _resize_keep_aspect(warped, output_size)


# ── Blister contour detection ───────────────────────────────────────────────

def detect_blister_contour(bgr: np.ndarray) -> Optional[np.ndarray]:
	lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
	l_channel = lab[:, :, 0]
	l_blur = cv2.GaussianBlur(l_channel, (7, 7), 0)

	adaptive = cv2.adaptiveThreshold(
		l_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 2
	)

	gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
	_, not_white = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)
	mask = cv2.bitwise_and(adaptive, not_white)

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
	closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
	closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

	contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if not contours:
		return None

	largest = max(contours, key=cv2.contourArea)
	hull = cv2.convexHull(largest)
	rect = cv2.minAreaRect(hull)
	box = cv2.boxPoints(rect)
	return box.astype("int32").reshape(-1, 1, 2)


def detect_empty_cells(
	bgr: np.ndarray,
	blister_contour: Optional[np.ndarray],
	pill_contours: list,
) -> Tuple[int, list]:
	"""
	Detect empty blister cells by inferring the pack's grid structure from
	detected pill positions, then verifying unoccupied grid positions with a
	foil-colour check.

	Strategy: we already know where pills ARE. A missing pill at an expected
	grid position — confirmed by the foil's achromatic, medium-brightness
	signature — is an empty cell.

	Border constraint: every inferred empty cell must sit at least as far from
	the blister contour as the closest real pill does, enforced via
	cv2.pointPolygonTest (negative = outside; positive = inside at that depth).
	"""
	if not pill_contours:
		return 0, []

	h, w = bgr.shape[:2]

	# ── Pill statistics ──────────────────────────────────────────────────────
	areas = [cv2.contourArea(c) for c in pill_contours]
	avg_area = float(np.mean(areas))
	if avg_area <= 0:
		return 0, []
	avg_radius = max(5, int(np.sqrt(avg_area / np.pi)))

	# ── Pill centres ─────────────────────────────────────────────────────────
	centers: list = []
	for cnt in pill_contours:
		M = cv2.moments(cnt)
		if M["m00"] > 0:
			centers.append((float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"])))
	if len(centers) < 2:
		return 0, []

	# ── Minimum pill-to-contour distance ─────────────────────────────────────
	# Computed first so extrapolation can use the same constraint.
	# pointPolygonTest returns negative for points outside the contour, so any
	# candidate outside the blister is automatically rejected by >= min_dist.
	if blister_contour is not None:
		raw_dists = [
			cv2.pointPolygonTest(blister_contour, (px, py), measureDist=True)
			for px, py in centers
		]
		valid_dists = [d for d in raw_dists if d > 0]
		min_border_dist = float(min(valid_dists)) if valid_dists else 0.0
	else:
		min_border_dist = 0.0

	def _inside(x: float, y: float) -> bool:
		"""True iff (x, y) is inside the blister with the required standoff."""
		xi, yi = int(round(x)), int(round(y))
		if not (0 <= xi < w and 0 <= yi < h):
			return False
		if blister_contour is None:
			return True
		d = cv2.pointPolygonTest(blister_contour, (x, y), measureDist=True)
		return d >= min_border_dist

	# ── Group centres into rows by y-coordinate clustering ───────────────────
	row_tol = avg_radius * 0.9
	by_y = sorted(centers, key=lambda c: c[1])
	rows: list = [[by_y[0]]]
	for cx, cy in by_y[1:]:
		row_mean_y = float(np.mean([c[1] for c in rows[-1]]))
		if abs(cy - row_mean_y) <= row_tol:
			rows[-1].append((cx, cy))
		else:
			rows.append([(cx, cy)])
	rows = [sorted(r, key=lambda c: c[0]) for r in rows]

	# ── Column spacing: minimum x-gap per row (robust to missing pills) ──────
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

	# ── Row spacing ──────────────────────────────────────────────────────────
	row_ys = [float(np.mean([c[1] for c in r])) for r in rows]
	row_spacing = (
		float(np.median(np.diff(row_ys))) if len(row_ys) >= 2 else col_spacing
	)
	if row_spacing < avg_radius * 1.2:
		row_spacing = col_spacing

	# ── Global column positions: cluster all pill x-coords ───────────────────
	all_x = sorted(c[0] for c in centers)
	col_groups: list = [[all_x[0]]]
	for x in all_x[1:]:
		if x - float(np.mean(col_groups[-1])) <= col_spacing * 0.45:
			col_groups[-1].append(x)
		else:
			col_groups.append([x])
	col_pos = [float(np.mean(g)) for g in col_groups]

	# ── Full row y-grid: interpolate missing interior rows ───────────────────
	full_row_ys: list = [row_ys[0]]
	for y in row_ys[1:]:
		n_fill = round((y - full_row_ys[-1]) / row_spacing) - 1
		for _ in range(n_fill):
			full_row_ys.append(full_row_ys[-1] + row_spacing)
		full_row_ys.append(y)

	# ── Extrapolate rows beyond detected range ────────────────────────────────
	# _inside() uses pointPolygonTest so it rejects positions outside the
	# blister contour or within its border margin — no mask rounding issues.
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

	# ── Extrapolate columns beyond detected range ─────────────────────────────
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

	# ── HSV for foil-colour verification ─────────────────────────────────────
	hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
	s_ch = hsv[:, :, 1]
	v_ch = hsv[:, :, 2]

	match_r  = col_spacing * 0.45
	r_sample = max(3, avg_radius // 4)

	# ── Scan all grid positions; emit confirmed empty cells ───────────────────
	empty_contours: list = []
	for grid_y in full_row_ys:
		for grid_x in col_pos:
			# Skip positions that already have a detected pill
			if any(
				abs(px - grid_x) <= match_r and abs(py - grid_y) <= row_tol
				for px, py in centers
			):
				continue

			# Reject if outside contour or too close to its edge
			if not _inside(grid_x, grid_y):
				continue

			cx_i = int(round(grid_x))
			cy_i = int(round(grid_y))

			# Foil-colour check on a small central patch
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
			# Foil: achromatic (S ≤ 70), medium brightness (50 ≤ V ≤ 220)
			# Rejects: coloured pills (high S), shadows (low V), white pills (V > 220)
			if med_s > 70 or med_v < 50 or med_v > 220:
				continue

			# Emit a circle contour at the inferred empty position
			tmp = np.zeros((h, w), dtype=np.uint8)
			cv2.circle(tmp, (cx_i, cy_i), avg_radius, 255, -1)
			cnts, _ = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			if cnts:
				empty_contours.append(cnts[0])

	return len(empty_contours), empty_contours


def _contour_mask(shape: Tuple[int, int, int], contour: Optional[np.ndarray]) -> np.ndarray:
	mask = np.zeros(shape[:2], dtype=np.uint8)
	if contour is None:
		return mask
	cv2.drawContours(mask, [contour], -1, 255, -1)
	return mask


def _filter_white_contours_by_size(
	contours: list,
	image_shape: Tuple[int, int, int],
	min_short_ratio: float = 0.015,
	max_long_ratio: float = 0.40,
	max_aspect_ratio: float = 4.0,
	median_low: float = 0.5,
	median_high: float = 1.8,
	min_for_median_filter: int = 4,
) -> list:
	"""Reject implausible white-pill contours such as long/flat glare streaks."""
	height, width = image_shape[:2]
	max_dim = float(max(height, width))

	candidates = []
	for contour in contours:
		(_, _), (w, h), _ = cv2.minAreaRect(contour)
		long_side = float(max(w, h))
		short_side = float(min(w, h))
		if short_side <= 1e-6:
			continue
		aspect_ratio = long_side / short_side

		# Hard bounds remove obvious non-pill artifacts before robust statistics.
		if short_side < min_short_ratio * float(height):
			continue
		if long_side > max_long_ratio * max_dim:
			continue
		if aspect_ratio > max_aspect_ratio:
			continue

		candidates.append((contour, short_side, long_side))

	if len(candidates) < min_for_median_filter:
		return [contour for contour, _, _ in candidates]

	short_sides = np.array([short_side for _, short_side, _ in candidates], dtype=np.float32)
	long_sides = np.array([long_side for _, _, long_side in candidates], dtype=np.float32)
	median_short = float(np.median(short_sides))
	median_long = float(np.median(long_sides))

	filtered = []
	for contour, short_side, long_side in candidates:
		if (
			median_low * median_short <= short_side <= median_high * median_short
			and median_low * median_long <= long_side <= median_high * median_long
		):
			filtered.append(contour)

	return filtered


# ── White balance correction ──────────────────────────────────────────────────

def correct_white_balance(
	rectified_bgr: np.ndarray,
	foil_val_min: int = 80,
	foil_val_max: int = 230,
	foil_sat_max: int = 60,
	target_gray: int = 180,
	min_foil_ratio: float = 0.10,
) -> Tuple[np.ndarray, bool]:
	"""
	Correct white balance using the blister foil as a neutral gray reference.
	The foil is physically achromatic; any per-channel imbalance reveals the
	camera's color cast, which we invert and apply globally.
	Returns (corrected_bgr, correction_was_applied).
	"""
	bgr_f = rectified_bgr.astype(np.float32)

	hsv = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2HSV)
	v_ch = hsv[:, :, 2].astype(np.float32)
	s_ch = hsv[:, :, 1].astype(np.float32)

	foil_mask = (v_ch >= foil_val_min) & (v_ch <= foil_val_max) & (s_ch <= foil_sat_max)
	foil_count = int(np.sum(foil_mask))
	total_pixels = rectified_bgr.shape[0] * rectified_bgr.shape[1]

	if foil_count / total_pixels < min_foil_ratio:
		return rectified_bgr.copy(), False

	mean_b = float(np.mean(bgr_f[:, :, 0][foil_mask]))
	mean_g = float(np.mean(bgr_f[:, :, 1][foil_mask]))
	mean_r = float(np.mean(bgr_f[:, :, 2][foil_mask]))

	if mean_b < 1 or mean_g < 1 or mean_r < 1:
		return rectified_bgr.copy(), False

	# Skip correction for gold/copper foil — it is not a neutral reference
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
	dark_val_max: int = 40,
	colored_sat_threshold: int = 45,
	colored_area_threshold: float = 0.04,
) -> ColorClassificationResult:
	"""
	Classify pills as WHITE or COLORED without prior segmentation.

	Strategy: after WB correction, white pills + foil are both achromatic
	(S < 40).  Colored pills have S > 45 over a significant image area.
	We simply ask: does any non-dark region contain meaningful saturation?
	"""
	hsv = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2HSV)
	h_ch = hsv[:, :, 0].astype(np.float32)
	s_ch = hsv[:, :, 1].astype(np.float32)
	v_ch = hsv[:, :, 2].astype(np.float32)

	not_dark_mask = v_ch > dark_val_max
	not_dark_count = int(np.sum(not_dark_mask))
	total_pixels = rectified_bgr.shape[0] * rectified_bgr.shape[1]
	sample_ratio = not_dark_count / total_pixels

	debug_mask = np.zeros(rectified_bgr.shape[:2], dtype=np.uint8)
	debug_mask[not_dark_mask] = 100  # gray = sampled region

	if not_dark_count == 0:
		return ColorClassificationResult(
			color_class=PillColorClass.UNKNOWN,
			median_saturation=-1.0,
			dominant_hue=-1.0,
			sample_pixel_ratio=sample_ratio,
			debug_mask=debug_mask,
		)

	colored_pixels_mask = not_dark_mask & (s_ch > colored_sat_threshold)
	colored_ratio = int(np.sum(colored_pixels_mask)) / not_dark_count
	debug_mask[colored_pixels_mask] = 255  # white = detected saturated pixels

	median_sat = float(np.median(s_ch[not_dark_mask]))

	if colored_ratio > colored_area_threshold:
		# Dominant hue weighted by saturation so strongly-colored pixels vote more
		pill_hues = h_ch[colored_pixels_mask]
		pill_sats = s_ch[colored_pixels_mask]
		hue_hist = np.zeros(180, dtype=np.float32)
		hue_indices = np.clip(pill_hues.astype(np.int32), 0, 179)
		np.add.at(hue_hist, hue_indices, pill_sats)
		dominant_hue = float(np.argmax(hue_hist))

		return ColorClassificationResult(
			color_class=PillColorClass.COLORED,
			median_saturation=median_sat,
			dominant_hue=dominant_hue,
			sample_pixel_ratio=sample_ratio,
			debug_mask=debug_mask,
		)

	return ColorClassificationResult(
		color_class=PillColorClass.WHITE,
		median_saturation=median_sat,
		dominant_hue=-1.0,
		sample_pixel_ratio=sample_ratio,
		debug_mask=debug_mask,
	)


def hue_to_color_name(hue_degrees: float) -> str:
	"""Map OpenCV hue (0–179 = half of 0–360°) to a human-readable color name."""
	h = hue_degrees * 2.0  # convert to full 0–360° range
	if h < 15 or h >= 345:
		return "red"
	if h < 45:
		return "orange"
	if h < 75:
		return "yellow"
	if h < 150:
		return "green"
	if h < 195:
		return "cyan"
	if h < 255:
		return "blue"
	if h < 285:
		return "purple"
	if h < 345:
		return "pink/magenta"
	return "unknown"


# ── Colored-pill segmentation & counting ─────────────────────────────────────

def _segment_pills(
	bgr: np.ndarray,
	clahe_clip: float,
	clahe_tile: int,
	sat_thresh: float,
	val_dark_thresh: float,
	adaptive_block: int,
	threshold_bias: float,
	separation: float,
) -> np.ndarray:
	lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
	l_channel = lab[:, :, 0]
	clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
	l_eq = clahe.apply(l_channel)
	blur = cv2.GaussianBlur(l_eq, (5, 5), 0)
	block = max(3, adaptive_block | 1)
	adaptive = cv2.adaptiveThreshold(
		blur, 255,
		cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
		block, threshold_bias,
	)

	hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
	s_channel = hsv[:, :, 1]
	v_channel = hsv[:, :, 2]
	s_mask = (s_channel >= sat_thresh).astype(np.uint8) * 255
	dark_mask = (v_channel <= val_dark_thresh).astype(np.uint8) * 255
	thresh = cv2.bitwise_or(adaptive, s_mask)
	thresh = cv2.bitwise_or(thresh, dark_mask)

	border = np.zeros_like(thresh, dtype=np.uint8)
	h, w = thresh.shape
	margin_h = max(1, int(h * 0.05))
	margin_w = max(1, int(w * 0.05))
	border[:margin_h, :] = 1
	border[-margin_h:, :] = 1
	border[:, :margin_w] = 1
	border[:, -margin_w:] = 1
	border_white_ratio = float(np.mean(thresh[border == 1] == 255))
	pill_mask = cv2.bitwise_not(thresh) if border_white_ratio > 0.5 else thresh.copy()

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
	pill_mask = cv2.morphologyEx(pill_mask, cv2.MORPH_OPEN, kernel, iterations=1)
	pill_mask = cv2.morphologyEx(pill_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

	dist = cv2.distanceTransform(pill_mask, cv2.DIST_L2, 5)
	_, sure_fg = cv2.threshold(dist, separation * dist.max(), 255, 0)
	sure_fg = sure_fg.astype(np.uint8)
	sure_bg = cv2.dilate(pill_mask, kernel, iterations=2)
	unknown = cv2.subtract(sure_bg, sure_fg)

	markers = cv2.connectedComponents(sure_fg)[1]
	markers = markers + 1
	markers[unknown == 255] = 0
	markers = cv2.watershed(bgr.copy(), markers)

	refined = np.zeros_like(pill_mask)
	refined[markers > 1] = 255
	return refined


def _count_pills(
	pill_mask: np.ndarray,
	ref_bgr: np.ndarray,
	min_area_ratio: float,
	min_circularity: float,
	stddev_max: float,
) -> Tuple[int, list]:
	contours, _ = cv2.findContours(pill_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	height, width = pill_mask.shape[:2]
	min_area = min_area_ratio * (height * width)
	pill_contours = []
	hsv = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2HSV)
	for contour in contours:
		area = cv2.contourArea(contour)
		if area < min_area:
			continue
		perimeter = cv2.arcLength(contour, True)
		if perimeter <= 0:
			continue
		circularity = 4.0 * np.pi * area / (perimeter * perimeter)
		if min_circularity > 0 and circularity < min_circularity:
			continue
		if stddev_max > 0:
			mask = np.zeros(pill_mask.shape, dtype=np.uint8)
			cv2.drawContours(mask, [contour], -1, 255, -1)
			stddev = cv2.meanStdDev(hsv[:, :, 2], mask=mask)[1][0][0]
			if stddev > stddev_max:
				continue
		pill_contours.append(contour)
	return len(pill_contours), pill_contours


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_image(
	image_path: Path,
	# colored-pill params
	clahe_clip: float,
	clahe_tile: int,
	sat_thresh: float,
	val_dark_thresh: float,
	adaptive_block: int,
	threshold_bias: float,
	separation: float,
	min_area_ratio: float,
	min_circularity: float,
	stddev_max: float,
	# white-pill params
	morph_open: int,
	morph_close: int,
	wp_separation: float,
	wp_min_area: float,
	wp_solidity: float,
	wp_brightness_floor: int,
) -> ProcessedImages:
	bgr = cv2.imread(str(image_path))
	if bgr is None:
		raise ValueError(f"Could not read image: {image_path}")

	rectified        = _rectify_pack(bgr, (800, 500))
	white_balanced, _= correct_white_balance(rectified)
	color_result     = classify_pill_color(white_balanced)
	blister_contour = detect_blister_contour(rectified)
	contour_mask = _contour_mask(rectified.shape, blister_contour)

	# Initialise outputs — overwritten by whichever path runs
	pill_mask      = np.zeros(rectified.shape[:2], dtype=np.uint8)
	count          = 0
	contours       = []
	annotated      = white_balanced.copy()
	white_debug: Dict[str, np.ndarray] = {}
	empty_count    = 0
	empty_contours: list = []

	if color_result.color_class == PillColorClass.COLORED:
		pill_mask = _segment_pills(
			white_balanced, clahe_clip, clahe_tile, sat_thresh,
			val_dark_thresh, adaptive_block, threshold_bias, separation,
		)
		count, contours = _count_pills(
			pill_mask, white_balanced,
			min_area_ratio=min_area_ratio,
			min_circularity=min_circularity,
			stddev_max=stddev_max,
		)
		if blister_contour is not None:
			cv2.drawContours(annotated, [blister_contour], -1, (255, 0, 0), 3)
		if contours:
			cv2.drawContours(annotated, contours, -1, (0, 0, 255), 2)
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
			brightness_floor = int(wp_brightness_floor),
			morph_open_iter  = morph_open,
			morph_close_iter = morph_close,
			separation       = wp_separation,
			min_area_ratio   = wp_min_area / 100.0,
			min_solidity     = wp_solidity,
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
		empty_count, empty_contours = detect_empty_cells(white_balanced, blister_contour, contours)
		if empty_contours:
			cv2.drawContours(annotated, empty_contours, -1, (0, 165, 255), 3)
		white_debug = wp_result.debug_images

	# UNKNOWN: all outputs stay at safe empty defaults set above

	binary     = cv2.bitwise_not(pill_mask)
	color_name = (
		hue_to_color_name(color_result.dominant_hue)
		if color_result.color_class == PillColorClass.COLORED else ""
	)

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
	)


# ── Display helpers ───────────────────────────────────────────────────────────

def _resize_for_display(image: np.ndarray, max_size: Tuple[int, int]) -> np.ndarray:
	height, width = image.shape[:2]
	max_width, max_height = max_size
	scale = min(max_width / width, max_height / height, 1.0)
	if scale == 1.0:
		return image
	return cv2.resize(image, (int(width * scale), int(height * scale)),
					  interpolation=cv2.INTER_AREA)


def _to_photo_image(image: np.ndarray) -> tk.PhotoImage:
	success, buffer = cv2.imencode(".png", image)
	if not success:
		raise ValueError("Failed to encode image for display")
	return tk.PhotoImage(data=base64.b64encode(buffer).decode("ascii"), format="png")


# ── UI ────────────────────────────────────────────────────────────────────────

class ImagePipelineUI(tk.Tk):
	def __init__(self, image_path: Path, image_paths: Optional[list] = None):
		super().__init__()
		self.title("Pill Counter")
		self.geometry("1200x700")
		self.minsize(900, 600)

		self._images      = []
		self._image_path  = image_path
		self._image_paths = image_paths if image_paths is not None else self._load_images_from_folder()
		self._image_index = self._resolve_image_index(image_path)
		self._update_job  = None
		self._colored_visible = False
		self._white_visible = False

		# ── Colored-pill sliders ──────────────────────────────────────
		self._clahe_clip      = tk.DoubleVar(value=2.0)
		self._sat_thresh      = tk.DoubleVar(value=35.0)
		self._val_dark        = tk.DoubleVar(value=80.0)
		self._adaptive_block  = tk.IntVar(value=35)
		self._adaptive_c      = tk.DoubleVar(value=15.0)
		self._separation      = tk.DoubleVar(value=0.45)
		self._min_area        = tk.DoubleVar(value=0.20)
		self._min_circularity = tk.DoubleVar(value=0.15)
		self._stddev_max      = tk.DoubleVar(value=0.0)
		# ── White-pill sliders ────────────────────────────────────────
		self._wp_brightness_floor = tk.DoubleVar(value=205.0)
		self._morph_open          = tk.IntVar(value=1)
		self._morph_close         = tk.IntVar(value=3)
		self._wp_separation       = tk.DoubleVar(value=0.40)
		self._wp_min_area         = tk.DoubleVar(value=0.30)
		self._wp_solidity         = tk.DoubleVar(value=0.50)
		self._build_layout()
		self._run_pipeline()

	# ── Layout ───────────────────────────────────────────────────────────────

	def _build_layout(self) -> None:
		header = ttk.Frame(self)
		header.pack(fill="x", padx=12, pady=10)

		self.count_label = ttk.Label(header, text="Pills: 0", font=("Segoe UI", 14, "bold"))
		self.count_label.pack(side="left")
		self.color_label = ttk.Label(header, text="Color: --", font=("Segoe UI", 11))
		self.color_label.pack(side="left", padx=(16, 0))

		btn_group = ttk.Frame(header)
		btn_group.pack(side="right")

		# Status lights (packed right-to-left, so declare right first)
		lights_frame = ttk.Frame(header)
		lights_frame.pack(side="right", padx=(0, 16))

		self._color_canvas = tk.Canvas(lights_frame, width=18, height=18, highlightthickness=0)
		self._color_oval   = self._color_canvas.create_oval(1, 1, 17, 17, fill="green", outline="#555")
		self._color_canvas.pack(side="right", padx=(4, 0))
		ttk.Label(lights_frame, text="Color anomaly").pack(side="right")

		ttk.Label(lights_frame, text="   ").pack(side="right")

		self._empty_canvas = tk.Canvas(lights_frame, width=18, height=18, highlightthickness=0)
		self._empty_oval   = self._empty_canvas.create_oval(1, 1, 17, 17, fill="green", outline="#555")
		self._empty_canvas.pack(side="right", padx=(4, 0))
		ttk.Label(lights_frame, text="Empty cells").pack(side="right")

		ttk.Label(lights_frame, text="   ").pack(side="right")
		ttk.Button(btn_group, text="Previous",   command=self._prev_image).pack(side="left", padx=(0, 6))
		ttk.Button(btn_group, text="Next",        command=self._next_image).pack(side="left", padx=(0, 6))
		ttk.Button(btn_group, text="Open Image",  command=self._open_image).pack(side="left")

		controls = ttk.Frame(self)
		controls.pack(fill="x", padx=12, pady=(0, 8))

		def slider_row(parent, row, label_var, text, from_, to, var) -> None:
			label_var.configure(text=text)
			label_var.grid(row=row, column=0, sticky="w")
			ttk.Scale(
				parent,
				from_=from_,
				to=to,
				variable=var,
				command=self._schedule_update,
			).grid(row=row, column=1, sticky="ew", padx=8)

		colored_header = ttk.Frame(controls)
		colored_header.grid(row=0, column=0, columnspan=2, sticky="ew")
		ttk.Label(colored_header, text="Colored pill settings", font=("Segoe UI", 9, "bold")).pack(
			side="left"
		)
		self._colored_toggle = ttk.Button(colored_header, text="Show", command=self._toggle_colored)
		self._colored_toggle.pack(side="right")

		self._colored_frame = ttk.Frame(controls)
		self._colored_frame.grid(row=1, column=0, columnspan=2, sticky="ew")
		self._colored_frame.grid_remove()

		self.sat_label = ttk.Label(self._colored_frame)
		slider_row(self._colored_frame, 0, self.sat_label, "Saturation min: 35", 0.0, 120.0, self._sat_thresh)
		self.val_label = ttk.Label(self._colored_frame)
		slider_row(self._colored_frame, 1, self.val_label, "Dark value max: 80", 0.0, 160.0, self._val_dark)
		self.clahe_label = ttk.Label(self._colored_frame)
		slider_row(self._colored_frame, 2, self.clahe_label, "CLAHE clip: 2.00", 1.0, 4.0, self._clahe_clip)
		self.block_label = ttk.Label(self._colored_frame)
		slider_row(self._colored_frame, 3, self.block_label, "Adaptive block: 35", 11, 71, self._adaptive_block)
		self.bias_label = ttk.Label(self._colored_frame)
		slider_row(self._colored_frame, 4, self.bias_label, "Adaptive C: 15.0", -10.0, 15.0, self._adaptive_c)
		self.separation_label = ttk.Label(self._colored_frame)
		slider_row(self._colored_frame, 5, self.separation_label, "Separation: 0.45", 0.25, 0.65, self._separation)
		self.area_label = ttk.Label(self._colored_frame)
		slider_row(self._colored_frame, 6, self.area_label, "Min area %: 0.20", 0.05, 1.00, self._min_area)
		self.circ_label = ttk.Label(self._colored_frame)
		slider_row(self._colored_frame, 7, self.circ_label, "Min circularity: 0.15", 0.0, 0.9, self._min_circularity)
		self.stddev_label = ttk.Label(self._colored_frame)
		slider_row(self._colored_frame, 8, self.stddev_label, "Max V stddev (0=off): 0", 0.0, 60.0, self._stddev_max)

		ttk.Separator(controls, orient="horizontal").grid(
			row=2, column=0, columnspan=2, sticky="ew", pady=6
		)

		white_header = ttk.Frame(controls)
		white_header.grid(row=3, column=0, columnspan=2, sticky="ew")
		ttk.Label(white_header, text="White pill settings", font=("Segoe UI", 9, "bold")).pack(
			side="left"
		)
		self._white_toggle = ttk.Button(white_header, text="Show", command=self._toggle_white)
		self._white_toggle.pack(side="right")

		self._white_frame = ttk.Frame(controls)
		self._white_frame.grid(row=4, column=0, columnspan=2, sticky="ew")
		self._white_frame.grid_remove()

		self.wp_bright_floor_label = ttk.Label(self._white_frame)
		slider_row(self._white_frame, 0, self.wp_bright_floor_label, "Brightness floor: 205", 140.0, 240.0, self._wp_brightness_floor)
		self.morph_open_label = ttk.Label(self._white_frame)
		slider_row(self._white_frame, 1, self.morph_open_label, "Morph open iter: 1", 1, 6, self._morph_open)
		self.morph_close_label = ttk.Label(self._white_frame)
		slider_row(self._white_frame, 2, self.morph_close_label, "Morph close iter: 3", 1, 8, self._morph_close)
		self.wp_sep_label = ttk.Label(self._white_frame)
		slider_row(self._white_frame, 3, self.wp_sep_label, "WP separation: 0.40", 0.25, 0.70, self._wp_separation)
		self.wp_area_label = ttk.Label(self._white_frame)
		slider_row(self._white_frame, 4, self.wp_area_label, "WP min area %: 0.30", 0.05, 2.0, self._wp_min_area)
		self.wp_solid_label = ttk.Label(self._white_frame)
		slider_row(self._white_frame, 5, self.wp_solid_label, "WP solidity: 0.50", 0.10, 1.0, self._wp_solidity)

		controls.columnconfigure(1, weight=1)

		# Scrollable image canvas
		self.canvas = tk.Canvas(self, highlightthickness=0)
		self.canvas.pack(fill="both", expand=True)
		scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
		scrollbar.pack(side="right", fill="y")
		self.canvas.configure(yscrollcommand=scrollbar.set)

		self.content = ttk.Frame(self.canvas)
		self._content_window = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
		self.content.bind("<Configure>", self._on_content_configure)
		self.canvas.bind("<Configure>",  self._on_canvas_configure)
		self._bind_mousewheel(self.canvas)

	# ── Image loading / navigation ────────────────────────────────────────────

	def _open_image(self) -> None:
		filename = filedialog.askopenfilename(
			title="Select Image",
			filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")],
		)
		if filename:
			self._image_path  = Path(filename)
			self._image_paths = self._load_images_from_folder()
			self._image_index = self._resolve_image_index(self._image_path)
			self._run_pipeline()

	def _load_images_from_folder(self) -> list:
		images_dir = Path(__file__).resolve().parent / "images"
		if not images_dir.exists():
			return []
		paths = []
		for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
			paths.extend(sorted(images_dir.glob(pattern)))
		return paths

	def _resolve_image_index(self, image_path: Path) -> int:
		try:
			return self._image_paths.index(image_path)
		except (ValueError, IndexError):
			return 0

	def _next_image(self) -> None:
		if not self._image_paths:
			return
		self._image_index = (self._image_index + 1) % len(self._image_paths)
		self._image_path  = self._image_paths[self._image_index]
		self._run_pipeline()

	def _prev_image(self) -> None:
		if not self._image_paths:
			return
		self._image_index = (self._image_index - 1) % len(self._image_paths)
		self._image_path  = self._image_paths[self._image_index]
		self._run_pipeline()

	# ── Canvas helpers ────────────────────────────────────────────────────────

	def _on_content_configure(self, _event: tk.Event) -> None:
		self.canvas.configure(scrollregion=self.canvas.bbox("all"))

	def _on_canvas_configure(self, event: tk.Event) -> None:
		self.canvas.itemconfigure(self._content_window, width=event.width)

	def _bind_mousewheel(self, widget: tk.Widget) -> None:
		widget.bind_all("<MouseWheel>", self._on_mousewheel)
		widget.bind_all("<Button-4>",   self._on_mousewheel)
		widget.bind_all("<Button-5>",   self._on_mousewheel)

	def _on_mousewheel(self, event: tk.Event) -> None:
		self.canvas.yview_scroll(-1 if (event.num == 4 or event.delta > 0) else 1, "units")

	# ── Pipeline update ───────────────────────────────────────────────────────

	def _schedule_update(self, _value: str = "") -> None:
		# Refresh all slider labels immediately for snappy UI feedback
		self.sat_label.configure(text=f"Saturation min: {self._sat_thresh.get():.0f}")
		self.val_label.configure(text=f"Dark value max: {self._val_dark.get():.0f}")
		self.clahe_label.configure(text=f"CLAHE clip: {self._clahe_clip.get():.2f}")
		self.block_label.configure(text=f"Adaptive block: {int(self._adaptive_block.get())}")
		self.bias_label.configure(text=f"Adaptive C: {self._adaptive_c.get():.1f}")
		self.separation_label.configure(text=f"Separation: {self._separation.get():.2f}")
		self.area_label.configure(text=f"Min area %: {self._min_area.get():.2f}")
		self.circ_label.configure(text=f"Min circularity: {self._min_circularity.get():.2f}")
		self.stddev_label.configure(text=f"Max V stddev (0=off): {self._stddev_max.get():.1f}")
		self.wp_bright_floor_label.configure(text=f"Brightness floor: {self._wp_brightness_floor.get():.0f}")
		self.morph_open_label.configure(text=f"Morph open iter: {int(self._morph_open.get())}")
		self.morph_close_label.configure(text=f"Morph close iter: {int(self._morph_close.get())}")
		self.wp_sep_label.configure(text=f"WP separation: {self._wp_separation.get():.2f}")
		self.wp_area_label.configure(text=f"WP min area %: {self._wp_min_area.get():.2f}")
		self.wp_solid_label.configure(text=f"WP solidity: {self._wp_solidity.get():.2f}")
		# Debounce: wait 120 ms after the last slider move before re-running
		if self._update_job is not None:
			self.after_cancel(self._update_job)
		self._update_job = self.after(120, self._run_pipeline)

	def _toggle_colored(self) -> None:
		self._colored_visible = not self._colored_visible
		if self._colored_visible:
			self._colored_frame.grid()
			self._colored_toggle.configure(text="Hide")
		else:
			self._colored_frame.grid_remove()
			self._colored_toggle.configure(text="Show")

	def _toggle_white(self) -> None:
		self._white_visible = not self._white_visible
		if self._white_visible:
			self._white_frame.grid()
			self._white_toggle.configure(text="Hide")
		else:
			self._white_frame.grid_remove()
			self._white_toggle.configure(text="Show")

	def _run_pipeline(self) -> None:
		for widget in self.content.winfo_children():
			widget.destroy()
		self._images.clear()
		self._update_job = None

		processed = process_image(
			self._image_path,
			clahe_clip       = float(self._clahe_clip.get()),
			clahe_tile       = 8,
			sat_thresh       = float(self._sat_thresh.get()),
			val_dark_thresh  = float(self._val_dark.get()),
			adaptive_block   = int(self._adaptive_block.get()),
			threshold_bias   = float(self._adaptive_c.get()),
			separation       = float(self._separation.get()),
			min_area_ratio   = float(self._min_area.get()) / 100.0,
			min_circularity  = float(self._min_circularity.get()),
			stddev_max       = float(self._stddev_max.get()),
			morph_open           = int(self._morph_open.get()),
			morph_close          = int(self._morph_close.get()),
			wp_separation        = float(self._wp_separation.get()),
			wp_min_area          = float(self._wp_min_area.get()),
			wp_solidity          = float(self._wp_solidity.get()),
			wp_brightness_floor  = int(self._wp_brightness_floor.get()),
		)

		self.count_label.configure(text=f"Pills: {processed.pill_count}")
		self._update_color_label(processed)

		empty_light = "red" if processed.empty_cell_count > 0 else "green"
		self._empty_canvas.itemconfigure(self._empty_oval, fill=empty_light)
		self._color_canvas.itemconfigure(self._color_oval, fill="green")  # placeholder

		# Adaptive panel set: white path shows texture debug; colored path shows standard
		if processed.color_class == PillColorClass.WHITE and processed.white_debug_images:
			wd = processed.white_debug_images
			steps = [
				("Original",           processed.original_bgr),
				("White balanced",     processed.white_balanced_bgr),
				("White mask",         wd.get("white_mask",    processed.white_balanced_bgr)),
				("Cleaned mask",       wd.get("cleaned_mask",  processed.white_balanced_bgr)),
				("Distance transform", wd.get("distance",       processed.white_balanced_bgr)),
				("Mask (pills white)", processed.mask),
				("Detected pills",     processed.annotated_bgr),
			]
		else:
			steps = [
				("Original",           processed.original_bgr),
				("White balanced",     processed.white_balanced_bgr),
				("Pill sample mask",   processed.color_debug_mask),
				("Mask (pills white)", processed.mask),
				("Binary",             processed.binary),
				("Detected pills",     processed.annotated_bgr),
			]

		max_display = (480, 360)
		for idx, (label, image) in enumerate(steps):
			frame = ttk.Frame(self.content)
			frame.grid(row=idx // 2, column=idx % 2, padx=12, pady=12, sticky="n")
			ttk.Label(frame, text=label, font=("Segoe UI", 11, "bold")).pack(pady=(0, 6))
			photo = _to_photo_image(_resize_for_display(image, max_display))
			self._images.append(photo)
			ttk.Label(frame, image=photo).pack()

	def _update_color_label(self, processed: ProcessedImages) -> None:
		cls = processed.color_class
		if cls == PillColorClass.WHITE:
			text = (f"Color: WHITE  |  "
					f"S_med={processed.median_saturation:.1f}  |  "
					f"sample={processed.sample_pixel_ratio:.2f}")
		elif cls == PillColorClass.COLORED:
			text = (f"Color: COLORED ({processed.color_name})  |  "
					f"S_med={processed.median_saturation:.1f}  |  "
					f"sample={processed.sample_pixel_ratio:.2f}")
		else:
			text = f"Color: UNKNOWN  |  sample={processed.sample_pixel_ratio:.2f}"
		self.color_label.configure(text=text)


# ── Entry point ───────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Pill counting pipeline with Tkinter UI")
	parser.add_argument("image", nargs="?", default="", help="Path to input image")
	return parser.parse_args()


def main() -> None:
	args = _parse_args()
	image_path  = Path(args.image) if args.image else None
	image_paths = None

	if image_path is None or not image_path.exists():
		images_dir = Path(__file__).resolve().parent / "images"
		paths = []
		for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
			paths.extend(sorted(images_dir.glob(pattern)))
		image_paths = paths
		image_path  = paths[0] if paths else Path()

	if not image_path or not image_path.exists():
		raise SystemExit("No images found. Put images in an 'images/' folder next to this script.")

	ImagePipelineUI(image_path, image_paths=image_paths).mainloop()


if __name__ == "__main__":
	main()