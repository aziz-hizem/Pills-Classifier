import argparse
import base64
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk


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


def _order_points(points: np.ndarray) -> np.ndarray:
	ordered = np.zeros((4, 2), dtype=np.float32)
	sum_values = points.sum(axis=1)
	ordered[0] = points[np.argmin(sum_values)]
	ordered[2] = points[np.argmax(sum_values)]
	diff_values = np.diff(points, axis=1)
	ordered[1] = points[np.argmin(diff_values)]
	ordered[3] = points[np.argmax(diff_values)]
	return ordered


def _four_point_warp(bgr: np.ndarray, points: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
	ordered = _order_points(points)
	(width, height) = output_size
	destination = np.array(
		[[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
		dtype=np.float32,
	)
	transform = cv2.getPerspectiveTransform(ordered, destination)
	return cv2.warpPerspective(bgr, transform, (width, height))


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

	(rect_w, rect_h) = rect[1]
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
		blur,
		255,
		cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
		cv2.THRESH_BINARY_INV,
		block,
		threshold_bias,
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
	if border_white_ratio > 0.5:
		pill_mask = cv2.bitwise_not(thresh)
	else:
		pill_mask = thresh.copy()

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


def correct_white_balance(
	rectified_bgr: np.ndarray,
	foil_val_min: int = 80,
	foil_val_max: int = 230,
	foil_sat_max: int = 60,
	target_gray: int = 180,
	min_foil_ratio: float = 0.10,
) -> Tuple[np.ndarray, bool]:
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

	# Skip correction if foil is likely gold/copper rather than neutral gray.
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


def classify_pill_color(
	rectified_bgr: np.ndarray,
	dark_val_max: int = 40,
	colored_sat_threshold: int = 45,
	colored_area_threshold: float = 0.04,
) -> ColorClassificationResult:
	hsv = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2HSV)
	h_ch = hsv[:, :, 0].astype(np.float32)
	s_ch = hsv[:, :, 1].astype(np.float32)
	v_ch = hsv[:, :, 2].astype(np.float32)

	not_dark_mask = v_ch > dark_val_max
	not_dark_count = int(np.sum(not_dark_mask))

	total_pixels = rectified_bgr.shape[0] * rectified_bgr.shape[1]
	sample_ratio = not_dark_count / total_pixels

	debug_mask = np.zeros(rectified_bgr.shape[:2], dtype=np.uint8)
	debug_mask[not_dark_mask] = 100

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
	debug_mask[colored_pixels_mask] = 255

	median_sat = float(np.median(s_ch[not_dark_mask]))

	if colored_ratio > colored_area_threshold:
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
	h = hue_degrees * 2.0
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
		mask = np.zeros(pill_mask.shape, dtype=np.uint8)
		cv2.drawContours(mask, [contour], -1, 255, -1)
		v_channel = hsv[:, :, 2]
		if stddev_max > 0:
			stddev = cv2.meanStdDev(v_channel, mask=mask)[1][0][0]
			if stddev > stddev_max:
				continue
		pill_contours.append(contour)
	return len(pill_contours), pill_contours


def process_image(
	image_path: Path,
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
) -> ProcessedImages:
	bgr = cv2.imread(str(image_path))
	if bgr is None:
		raise ValueError(f"Could not read image: {image_path}")

	rectified = _rectify_pack(bgr, (800, 500))
	white_balanced, _ = correct_white_balance(rectified)
	color_result = classify_pill_color(white_balanced)

	if color_result.color_class == PillColorClass.COLORED:
		pill_mask = _segment_pills(
			white_balanced,
			clahe_clip,
			clahe_tile,
			sat_thresh,
			val_dark_thresh,
			adaptive_block,
			threshold_bias,
			separation,
		)
		count, contours = _count_pills(
			pill_mask,
			white_balanced,
			min_area_ratio=min_area_ratio,
			min_circularity=min_circularity,
			stddev_max=stddev_max,
		)
	else:
		pill_mask = np.zeros(rectified.shape[:2], dtype=np.uint8)
		count, contours = 0, []
	binary = cv2.bitwise_not(pill_mask)
	annotated = white_balanced.copy()
	if contours:
		cv2.drawContours(annotated, contours, -1, (0, 0, 255), 2)
	color_name = (
		hue_to_color_name(color_result.dominant_hue)
		if color_result.color_class == PillColorClass.COLORED
		else ""
	)
	return ProcessedImages(
		original_bgr=bgr,
		rectified_bgr=rectified,
		white_balanced_bgr=white_balanced,
		color_debug_mask=color_result.debug_mask,
		mask=pill_mask,
		binary=binary,
		annotated_bgr=annotated,
		pill_count=count,
		color_class=color_result.color_class,
		median_saturation=color_result.median_saturation,
		dominant_hue=color_result.dominant_hue,
		sample_pixel_ratio=color_result.sample_pixel_ratio,
		color_name=color_name,
	)


def _resize_for_display(image: np.ndarray, max_size: Tuple[int, int]) -> np.ndarray:
	height, width = image.shape[:2]
	max_width, max_height = max_size
	scale = min(max_width / width, max_height / height, 1.0)
	if scale == 1.0:
		return image
	new_size = (int(width * scale), int(height * scale))
	return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def _to_photo_image(image: np.ndarray) -> tk.PhotoImage:
	success, buffer = cv2.imencode(".png", image)
	if not success:
		raise ValueError("Failed to encode image for display")
	data = base64.b64encode(buffer).decode("ascii")
	return tk.PhotoImage(data=data, format="png")


class ImagePipelineUI(tk.Tk):
	def __init__(self, image_path: Path, image_paths: list | None = None):
		super().__init__()
		self.title("Pill Counter")
		self.geometry("1200x700")
		self.minsize(900, 600)

		self._images = []
		self._image_path = image_path
		self._image_paths = image_paths if image_paths is not None else self._load_images_from_folder()
		self._image_index = self._resolve_image_index(image_path)
		self._update_job = None
		self._clahe_clip = tk.DoubleVar(value=2.0)
		self._sat_thresh = tk.DoubleVar(value=35.0)
		self._val_dark = tk.DoubleVar(value=80.0)
		self._adaptive_block = tk.IntVar(value=35)
		self._adaptive_c = tk.DoubleVar(value=15.0)
		self._separation = tk.DoubleVar(value=0.45)
		self._min_area = tk.DoubleVar(value=0.20)
		self._min_circularity = tk.DoubleVar(value=0.15)
		self._stddev_max = tk.DoubleVar(value=0.0)

		self._build_layout()
		self._run_pipeline()

	def _build_layout(self) -> None:
		header = ttk.Frame(self)
		header.pack(fill="x", padx=12, pady=10)

		self.count_label = ttk.Label(header, text="Pills: 0", font=("Segoe UI", 14, "bold"))
		self.count_label.pack(side="left")
		self.color_label = ttk.Label(header, text="Color: --", font=("Segoe UI", 11))
		self.color_label.pack(side="left", padx=(16, 0))

		button_group = ttk.Frame(header)
		button_group.pack(side="right")
		ttk.Button(button_group, text="Previous", command=self._prev_image).pack(side="left", padx=(0, 6))
		ttk.Button(button_group, text="Next", command=self._next_image).pack(side="left", padx=(0, 6))
		ttk.Button(button_group, text="Open Image", command=self._open_image).pack(side="left")

		controls = ttk.Frame(self)
		controls.pack(fill="x", padx=12, pady=(0, 8))

		self.sat_label = ttk.Label(controls, text="Saturation min: 35")
		self.sat_label.grid(row=0, column=0, sticky="w")
		sat_slider = ttk.Scale(
			controls,
			from_=0.0,
			to=120.0,
			variable=self._sat_thresh,
			command=self._schedule_update,
		)
		sat_slider.grid(row=0, column=1, sticky="ew", padx=8)

		self.val_label = ttk.Label(controls, text="Dark value max: 80")
		self.val_label.grid(row=1, column=0, sticky="w")
		val_slider = ttk.Scale(
			controls,
			from_=0.0,
			to=160.0,
			variable=self._val_dark,
			command=self._schedule_update,
		)
		val_slider.grid(row=1, column=1, sticky="ew", padx=8)

		self.clahe_label = ttk.Label(controls, text="CLAHE clip: 2.00")
		self.clahe_label.grid(row=2, column=0, sticky="w")
		clahe_slider = ttk.Scale(
			controls,
			from_=1.0,
			to=4.0,
			variable=self._clahe_clip,
			command=self._schedule_update,
		)
		clahe_slider.grid(row=2, column=1, sticky="ew", padx=8)

		self.block_label = ttk.Label(controls, text="Adaptive block: 35")
		self.block_label.grid(row=3, column=0, sticky="w")
		block_slider = ttk.Scale(
			controls,
			from_=11,
			to=71,
			variable=self._adaptive_block,
			command=self._schedule_update,
		)
		block_slider.grid(row=3, column=1, sticky="ew", padx=8)

		self.bias_label = ttk.Label(controls, text="Adaptive C: 3.0")
		self.bias_label.grid(row=4, column=0, sticky="w")
		bias_slider = ttk.Scale(
			controls,
			from_=-10.0,
			to=15.0,
			variable=self._adaptive_c,
			command=self._schedule_update,
		)
		bias_slider.grid(row=4, column=1, sticky="ew", padx=8)

		self.separation_label = ttk.Label(controls, text="Separation: 0.52")
		self.separation_label.grid(row=5, column=0, sticky="w")
		separation_slider = ttk.Scale(
			controls,
			from_=0.25,
			to=0.65,
			variable=self._separation,
			command=self._schedule_update,
		)
		separation_slider.grid(row=5, column=1, sticky="ew", padx=8)

		self.area_label = ttk.Label(controls, text="Min area %: 0.20")
		self.area_label.grid(row=6, column=0, sticky="w")
		area_slider = ttk.Scale(
			controls,
			from_=0.05,
			to=1.00,
			variable=self._min_area,
			command=self._schedule_update,
		)
		area_slider.grid(row=6, column=1, sticky="ew", padx=8)

		self.circularity_label = ttk.Label(controls, text="Min circularity: 0.15")
		self.circularity_label.grid(row=7, column=0, sticky="w")
		circularity_slider = ttk.Scale(
			controls,
			from_=0.0,
			to=0.9,
			variable=self._min_circularity,
			command=self._schedule_update,
		)
		circularity_slider.grid(row=7, column=1, sticky="ew", padx=8)

		self.stddev_label = ttk.Label(controls, text="Max V stddev (0=off): 0.0")
		self.stddev_label.grid(row=8, column=0, sticky="w")
		stddev_slider = ttk.Scale(
			controls,
			from_=0.0,
			to=60.0,
			variable=self._stddev_max,
			command=self._schedule_update,
		)
		stddev_slider.grid(row=8, column=1, sticky="ew", padx=8)

		controls.columnconfigure(1, weight=1)

		self.canvas = tk.Canvas(self, highlightthickness=0)
		self.canvas.pack(fill="both", expand=True)
		scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
		scrollbar.pack(side="right", fill="y")
		self.canvas.configure(yscrollcommand=scrollbar.set)

		self.content = ttk.Frame(self.canvas)
		self._content_window = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
		self.content.bind("<Configure>", self._on_content_configure)
		self.canvas.bind("<Configure>", self._on_canvas_configure)
		self._bind_mousewheel(self.canvas)

	def _open_image(self) -> None:
		filename = filedialog.askopenfilename(
			title="Select Image",
			filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")],
		)
		if filename:
			self._image_path = Path(filename)
			self._image_paths = self._load_images_from_folder()
			self._image_index = self._resolve_image_index(self._image_path)
			self._run_pipeline()

	def _load_images_from_folder(self) -> list:
		images_dir = Path(__file__).resolve().parent / "images"
		if not images_dir.exists():
			return []
		patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
		paths = []
		for pattern in patterns:
			paths.extend(sorted(images_dir.glob(pattern)))
		return paths

	def _resolve_image_index(self, image_path: Path) -> int:
		if not self._image_paths:
			return 0
		try:
			return self._image_paths.index(image_path)
		except ValueError:
			return 0

	def _next_image(self) -> None:
		if not self._image_paths:
			return
		self._image_index = (self._image_index + 1) % len(self._image_paths)
		self._image_path = self._image_paths[self._image_index]
		self._run_pipeline()

	def _prev_image(self) -> None:
		if not self._image_paths:
			return
		self._image_index = (self._image_index - 1) % len(self._image_paths)
		self._image_path = self._image_paths[self._image_index]
		self._run_pipeline()

	def _on_content_configure(self, _event: tk.Event) -> None:
		self.canvas.configure(scrollregion=self.canvas.bbox("all"))

	def _on_canvas_configure(self, event: tk.Event) -> None:
		self.canvas.itemconfigure(self._content_window, width=event.width)

	def _bind_mousewheel(self, widget: tk.Widget) -> None:
		widget.bind_all("<MouseWheel>", self._on_mousewheel)
		widget.bind_all("<Button-4>", self._on_mousewheel)
		widget.bind_all("<Button-5>", self._on_mousewheel)

	def _on_mousewheel(self, event: tk.Event) -> None:
		if event.num == 5 or event.delta < 0:
			self.canvas.yview_scroll(1, "units")
		else:
			self.canvas.yview_scroll(-1, "units")

	def _schedule_update(self, _value: str) -> None:
		self.sat_label.configure(text=f"Saturation min: {self._sat_thresh.get():.0f}")
		self.val_label.configure(text=f"Dark value max: {self._val_dark.get():.0f}")
		self.clahe_label.configure(text=f"CLAHE clip: {self._clahe_clip.get():.2f}")
		self.block_label.configure(text=f"Adaptive block: {int(self._adaptive_block.get())}")
		self.bias_label.configure(text=f"Adaptive C: {self._adaptive_c.get():.1f}")
		self.separation_label.configure(text=f"Separation: {self._separation.get():.2f}")
		self.area_label.configure(text=f"Min area %: {self._min_area.get():.2f}")
		self.circularity_label.configure(text=f"Min circularity: {self._min_circularity.get():.2f}")
		self.stddev_label.configure(text=f"Max V stddev (0=off): {self._stddev_max.get():.1f}")
		if self._update_job is not None:
			self.after_cancel(self._update_job)
		self._update_job = self.after(120, self._run_pipeline)

	def _run_pipeline(self) -> None:
		for widget in self.content.winfo_children():
			widget.destroy()
		self._images.clear()
		self._update_job = None

		clahe_clip = float(self._clahe_clip.get())
		sat_thresh = float(self._sat_thresh.get())
		val_dark = float(self._val_dark.get())
		adaptive_block = int(self._adaptive_block.get())
		adaptive_c = float(self._adaptive_c.get())
		separation = float(self._separation.get())
		min_area_ratio = float(self._min_area.get()) / 100.0
		min_circularity = float(self._min_circularity.get())
		stddev_max = float(self._stddev_max.get())
		processed = process_image(
			self._image_path,
			clahe_clip=clahe_clip,
			clahe_tile=8,
			sat_thresh=sat_thresh,
			val_dark_thresh=val_dark,
			adaptive_block=adaptive_block,
			threshold_bias=adaptive_c,
			separation=separation,
			min_area_ratio=min_area_ratio,
			min_circularity=min_circularity,
			stddev_max=stddev_max,
		)
		self.count_label.configure(text=f"Pills: {processed.pill_count}")
		self._update_color_label(processed)

		max_display = (480, 360)
		steps = [
			("Original", processed.original_bgr),
			("Rectified", processed.rectified_bgr),
			("White-balanced", processed.white_balanced_bgr),
			("Pill sample mask", processed.color_debug_mask),
			("Mask (pills in white)", processed.mask),
			("Binary (pills in black)", processed.binary),
			("Detected pills", processed.annotated_bgr),
		]

		for idx, (label, image) in enumerate(steps):
			frame = ttk.Frame(self.content)
			frame.grid(row=idx // 2, column=idx % 2, padx=12, pady=12, sticky="n")

			ttk.Label(frame, text=label, font=("Segoe UI", 11, "bold")).pack(pady=(0, 6))
			display_image = _resize_for_display(image, max_display)
			photo = _to_photo_image(display_image)
			self._images.append(photo)
			ttk.Label(frame, image=photo).pack()

	def _update_color_label(self, processed: ProcessedImages) -> None:
		if processed.color_class == PillColorClass.WHITE:
			text = f"Color: WHITE | Smed={processed.median_saturation:.1f} | sample={processed.sample_pixel_ratio:.2f}"
		elif processed.color_class == PillColorClass.COLORED:
			text = (
				"Color: COLORED"
				f" ({processed.color_name}) | Smed={processed.median_saturation:.1f}"
				f" | sample={processed.sample_pixel_ratio:.2f}"
			)
		else:
			text = f"Color: UNKNOWN | sample={processed.sample_pixel_ratio:.2f}"
		self.color_label.configure(text=text)


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Pill counting pipeline with Tkinter UI")
	parser.add_argument("image", nargs="?", default="", help="Path to input image")
	return parser.parse_args()


def main() -> None:
	args = _parse_args()
	image_path = Path(args.image) if args.image else None
	image_paths = None
	if image_path is None or not image_path.exists():
		images_dir = Path(__file__).resolve().parent / "images"
		patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
		paths = []
		for pattern in patterns:
			paths.extend(sorted(images_dir.glob(pattern)))
		image_paths = paths
		image_path = paths[0] if paths else Path()
	if not image_path or not image_path.exists():
		raise SystemExit("No images found in the images folder.")
	app = ImagePipelineUI(image_path, image_paths=image_paths)
	app.mainloop()


if __name__ == "__main__":
	main()
