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
	rectified_bgr: np.ndarray
	white_balanced_bgr: np.ndarray
	color_debug_mask: np.ndarray
	mask: np.ndarray
	binary: np.ndarray
	annotated_bgr: np.ndarray
	pill_count: int
	color_class: "PillColorClass"
	median_saturation: float
	dominant_hue: float
	sample_pixel_ratio: float
	color_name: str
	white_debug_images: Optional[Dict[str, np.ndarray]] = field(default=None)


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
		return cv2.resize(bgr, output_size)

	contours = sorted(contours, key=cv2.contourArea, reverse=True)
	pack_contour = contours[0]
	image_area = resized.shape[0] * resized.shape[1]
	if cv2.contourArea(pack_contour) < 0.2 * image_area:
		return cv2.resize(bgr, output_size)

	rect = cv2.minAreaRect(pack_contour)
	box = cv2.boxPoints(rect)
	points = box.reshape(-1, 2) / scale

	rect_w, rect_h = rect[1]
	if rect_w <= 1 or rect_h <= 1:
		return cv2.resize(bgr, output_size)

	max_w, max_h = output_size
	if rect_w < rect_h:
		rect_w, rect_h = rect_h, rect_w
	aspect = rect_w / rect_h
	if max_w / max_h > aspect:
		width = int(max_h * aspect)
		height = max_h
	else:
		width = max_w
		height = int(max_w / aspect)

	return _four_point_warp(bgr, points.astype(np.float32), (width, height))


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
	tex_kernel: int,
	tex_thresh: float,
	morph_open: int,
	morph_close: int,
	wp_separation: float,
	wp_min_area: float,
	wp_convexity: float,
	wp_solidity: float,
	wp_dark_vmax: float,
	wp_bright_vmin: float,
) -> ProcessedImages:
	bgr = cv2.imread(str(image_path))
	if bgr is None:
		raise ValueError(f"Could not read image: {image_path}")

	rectified        = _rectify_pack(bgr, (800, 500))
	white_balanced, _= correct_white_balance(rectified)
	color_result     = classify_pill_color(white_balanced)

	# Initialise outputs — overwritten by whichever path runs
	pill_mask    = np.zeros(rectified.shape[:2], dtype=np.uint8)
	count        = 0
	contours     = []
	annotated    = white_balanced.copy()
	white_debug: Dict[str, np.ndarray] = {}

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
		if contours:
			cv2.drawContours(annotated, contours, -1, (0, 0, 255), 2)

	elif color_result.color_class == PillColorClass.WHITE:
		wp_result    = segment_white_pills(
			white_balanced,
			texture_kernel   = tex_kernel,
			texture_thresh   = tex_thresh,
			dark_val_max     = int(wp_dark_vmax),
			bright_val_min   = int(wp_bright_vmin),
			morph_open_iter  = morph_open,
			morph_close_iter = morph_close,
			separation       = wp_separation,
			min_area_ratio   = wp_min_area / 100.0,
			min_convexity    = wp_convexity,
			min_solidity     = wp_solidity,
		)
		pill_mask    = wp_result.pill_mask
		count        = wp_result.pill_count
		contours     = wp_result.contours
		annotated    = wp_result.debug_images.get("annotated", annotated)
		white_debug  = wp_result.debug_images

	# UNKNOWN: all outputs stay at safe empty defaults set above

	binary     = cv2.bitwise_not(pill_mask)
	color_name = (
		hue_to_color_name(color_result.dominant_hue)
		if color_result.color_class == PillColorClass.COLORED else ""
	)

	return ProcessedImages(
		original_bgr        = bgr,
		rectified_bgr       = rectified,
		white_balanced_bgr  = white_balanced,
		color_debug_mask    = color_result.debug_mask,
		mask                = pill_mask,
		binary              = binary,
		annotated_bgr       = annotated,
		pill_count          = count,
		color_class         = color_result.color_class,
		median_saturation   = color_result.median_saturation,
		dominant_hue        = color_result.dominant_hue,
		sample_pixel_ratio  = color_result.sample_pixel_ratio,
		color_name          = color_name,
		white_debug_images  = white_debug or None,
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
		self._colored_visible = True
		self._white_visible = True

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
		self._tex_kernel      = tk.IntVar(value=7)
		self._tex_thresh      = tk.DoubleVar(value=0.0)
		self._morph_open      = tk.IntVar(value=2)
		self._morph_close     = tk.IntVar(value=4)
		self._wp_separation   = tk.DoubleVar(value=0.40)
		self._wp_min_area     = tk.DoubleVar(value=0.30)
		self._wp_convexity    = tk.DoubleVar(value=0.70)
		self._wp_solidity     = tk.DoubleVar(value=0.50)
		self._wp_dark_vmax    = tk.DoubleVar(value=101.0)
		self._wp_bright_vmin  = tk.DoubleVar(value=160.0)

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
		self._colored_toggle = ttk.Button(colored_header, text="Hide", command=self._toggle_colored)
		self._colored_toggle.pack(side="right")

		self._colored_frame = ttk.Frame(controls)
		self._colored_frame.grid(row=1, column=0, columnspan=2, sticky="ew")

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
		self._white_toggle = ttk.Button(white_header, text="Hide", command=self._toggle_white)
		self._white_toggle.pack(side="right")

		self._white_frame = ttk.Frame(controls)
		self._white_frame.grid(row=4, column=0, columnspan=2, sticky="ew")

		self.tex_kernel_label = ttk.Label(self._white_frame)
		slider_row(self._white_frame, 0, self.tex_kernel_label, "Texture kernel: 7", 3, 15, self._tex_kernel)
		self.tex_thresh_label = ttk.Label(self._white_frame)
		slider_row(self._white_frame, 1, self.tex_thresh_label, "Foil var thresh (0=auto): 0.0", 0.0, 30.0, self._tex_thresh)
		self.morph_open_label = ttk.Label(self._white_frame)
		slider_row(self._white_frame, 2, self.morph_open_label, "Morph open iter: 2", 1, 6, self._morph_open)
		self.morph_close_label = ttk.Label(self._white_frame)
		slider_row(self._white_frame, 3, self.morph_close_label, "Morph close iter: 4", 1, 8, self._morph_close)
		self.wp_sep_label = ttk.Label(self._white_frame)
		slider_row(self._white_frame, 4, self.wp_sep_label, "WP separation: 0.40", 0.25, 0.70, self._wp_separation)
		self.wp_area_label = ttk.Label(self._white_frame)
		slider_row(self._white_frame, 5, self.wp_area_label, "WP min area %: 0.30", 0.05, 2.0, self._wp_min_area)
		self.wp_conv_label = ttk.Label(self._white_frame)
		slider_row(self._white_frame, 6, self.wp_conv_label, "WP convexity: 0.70", 0.50, 1.0, self._wp_convexity)
		self.wp_solid_label = ttk.Label(self._white_frame)
		slider_row(self._white_frame, 7, self.wp_solid_label, "WP solidity: 0.50", 0.10, 1.0, self._wp_solidity)
		self.wp_dark_label = ttk.Label(self._white_frame)
		slider_row(self._white_frame, 8, self.wp_dark_label, "Adaptive block size: 101", 51.0, 151.0, self._wp_dark_vmax)
		self.wp_bright_label = ttk.Label(self._white_frame)
		slider_row(self._white_frame, 9, self.wp_bright_label, "Adaptive C strength: 160", 160.0, 255.0, self._wp_bright_vmin)

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
		self.tex_kernel_label.configure(text=f"Texture kernel: {int(self._tex_kernel.get())}")
		self.tex_thresh_label.configure(text=f"Foil var thresh (0=auto): {self._tex_thresh.get():.1f}")
		self.morph_open_label.configure(text=f"Morph open iter: {int(self._morph_open.get())}")
		self.morph_close_label.configure(text=f"Morph close iter: {int(self._morph_close.get())}")
		self.wp_sep_label.configure(text=f"WP separation: {self._wp_separation.get():.2f}")
		self.wp_area_label.configure(text=f"WP min area %: {self._wp_min_area.get():.2f}")
		self.wp_conv_label.configure(text=f"WP convexity: {self._wp_convexity.get():.2f}")
		self.wp_solid_label.configure(text=f"WP solidity: {self._wp_solidity.get():.2f}")
		self.wp_dark_label.configure(text=f"Adaptive block size: {self._wp_dark_vmax.get():.0f}")
		self.wp_bright_label.configure(text=f"Adaptive C strength: {self._wp_bright_vmin.get():.0f}")
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
			tex_kernel       = int(self._tex_kernel.get()),
			tex_thresh       = float(self._tex_thresh.get()),
			morph_open       = int(self._morph_open.get()),
			morph_close      = int(self._morph_close.get()),
			wp_separation    = float(self._wp_separation.get()),
			wp_min_area      = float(self._wp_min_area.get()),
			wp_convexity     = float(self._wp_convexity.get()),
			wp_solidity      = float(self._wp_solidity.get()),
			wp_dark_vmax     = float(self._wp_dark_vmax.get()),
			wp_bright_vmin   = float(self._wp_bright_vmin.get()),
		)

		self.count_label.configure(text=f"Pills: {processed.pill_count}")
		self._update_color_label(processed)

		# Adaptive panel set: white path shows texture debug; colored path shows standard
		if processed.color_class == PillColorClass.WHITE and processed.white_debug_images:
			wd = processed.white_debug_images
			steps = [
				("Original",           processed.original_bgr),
				("White balanced",     processed.white_balanced_bgr),
				("Texture map",        wd.get("texture_map",    processed.white_balanced_bgr)),
				("Texture threshold",  wd.get("texture_thresh", processed.white_balanced_bgr)),
				("Cleaned mask",       wd.get("cleaned_mask",   processed.white_balanced_bgr)),
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