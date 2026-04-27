import os
import cv2
import tkinter as tk
from tkinter import ttk

SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(folder):
    return [
        os.path.join(folder, name)
        for name in sorted(os.listdir(folder))
        if name.lower().endswith(SUPPORTED_EXTS)
    ]


def detect_blister_contour(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    l_blur = cv2.GaussianBlur(l_channel, (7, 7), 0)

    # Adaptive threshold helps when lighting changes across the blister.
    adaptive = cv2.adaptiveThreshold(
        l_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 2
    )

    # Suppress near-white background explicitly to avoid leaking into table.
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

    # Fit a stable rectangle so small lighting artifacts do not create jagged edges.
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    return box.astype("int32").reshape(-1, 1, 2)


def render_with_contour(bgr, contour):
    annotated = bgr.copy()
    if contour is not None:
        cv2.drawContours(annotated, [contour], -1, (255, 0, 0), 3)
    return annotated


def bgr_to_tk_photo(bgr, max_size):
    h, w = bgr.shape[:2]
    max_w = max(int(max_size[0]), 1)
    max_h = max(int(max_size[1]), 1)
    scale = min(max_w / w, max_h / h, 1.0)
    if scale != 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    header = f"P6 {w} {h} 255\n".encode("ascii")
    ppm = header + rgb.tobytes()
    return tk.PhotoImage(data=ppm, format="PPM")


class BlisterViewer(tk.Tk):
    def __init__(self, images, cycle_ms=2000):
        super().__init__()
        self.title("Blister Contour Viewer")
        self.geometry("1000x800")

        self.images = images
        self.index = 0
        self.cycle_ms = cycle_ms
        self.is_playing = True
        self._after_id = None

        self.canvas = tk.Label(self, background="#111")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(self)
        controls.pack(fill=tk.X, padx=10, pady=10)

        self.prev_btn = ttk.Button(controls, text="Prev", command=self.show_prev)
        self.prev_btn.pack(side=tk.LEFT)

        self.play_btn = ttk.Button(controls, text="Pause", command=self.toggle_play)
        self.play_btn.pack(side=tk.LEFT, padx=6)

        self.next_btn = ttk.Button(controls, text="Next", command=self.show_next)
        self.next_btn.pack(side=tk.LEFT)

        self.status = ttk.Label(controls, text="")
        self.status.pack(side=tk.RIGHT)

        self.bind("<Left>", lambda _e: self.show_prev())
        self.bind("<Right>", lambda _e: self.show_next())
        self.bind("<space>", lambda _e: self.toggle_play())

        self.show_image()
        self.schedule_next()

    def show_image(self):
        if not self.images:
            self.status.config(text="No images found.")
            self.canvas.config(image="", text="No images", fg="white")
            return

        path = self.images[self.index]
        bgr = cv2.imread(path)
        if bgr is None:
            self.status.config(text=f"Failed to read: {os.path.basename(path)}")
            return

        contour = detect_blister_contour(bgr)
        annotated = render_with_contour(bgr, contour)

        max_size = (self.winfo_width() - 40, self.winfo_height() - 140)
        photo = bgr_to_tk_photo(annotated, max_size)

        self.canvas.image = photo
        self.canvas.config(image=photo)
        self.status.config(text=f"{self.index + 1}/{len(self.images)}  {os.path.basename(path)}")

    def show_next(self):
        if not self.images:
            return
        self.index = (self.index + 1) % len(self.images)
        self.show_image()
        self.reschedule()

    def show_prev(self):
        if not self.images:
            return
        self.index = (self.index - 1) % len(self.images)
        self.show_image()
        self.reschedule()

    def toggle_play(self):
        self.is_playing = not self.is_playing
        self.play_btn.config(text="Pause" if self.is_playing else "Play")
        self.reschedule()

    def schedule_next(self):
        if self.is_playing and self.images:
            self._after_id = self.after(self.cycle_ms, self.show_next)

    def reschedule(self):
        if self._after_id is not None:
            self.after_cancel(self._after_id)
            self._after_id = None
        self.schedule_next()


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(base_dir, "images")
    images = list_images(images_dir) if os.path.isdir(images_dir) else []

    app = BlisterViewer(images)
    app.mainloop()


if __name__ == "__main__":
    main()
