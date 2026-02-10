"""
easy_cartoonify.py  –  OpenCV Cartoonify  (GUI edition)

Layout
──────
┌─────────────────────────────────────────────────────────┐
│  [Open Image]          [Save Result]                    │  ← toolbar
├──────────────┬──────────────────────────────────────────┤
│              │                                          │
│   Controls   │           Image Canvas                   │
│   panel      │  (original on left / result on right)   │
│              │                                          │
└──────────────┴──────────────────────────────────────────┘
│  Status bar                                             │
└─────────────────────────────────────────────────────────┘

Controls panel
──────────────
  • Style radio buttons  (4 styles)
  • Sigma-S slider        (spatial smoothing,  1–200)
  • Sigma-R slider        (range smoothing,    0.01–1.0)
  • Shade-factor slider   (pencil styles only, 0.01–0.10)
  • [Apply] button
"""

import os
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

STYLES = {
    "Style 1 – Heavy":          "style1",
    "Style 2 – Light":          "style2",
    "Pencil Sketch (Grey)":     "pencil_grey",
    "Pencil Sketch (Colour)":   "pencil_colour",
    "Detail Enhance":           "detail",
}

DEFAULTS = {
    "style1":          dict(sigma_s=150, sigma_r=0.25, shade_factor=0.05),
    "style2":          dict(sigma_s=60,  sigma_r=0.50, shade_factor=0.05),
    "pencil_grey":     dict(sigma_s=60,  sigma_r=0.07, shade_factor=0.05),
    "pencil_colour":   dict(sigma_s=60,  sigma_r=0.07, shade_factor=0.05),
    "detail":          dict(sigma_s=10,  sigma_r=0.15, shade_factor=0.05),
}

# Shade-factor is only relevant for the pencil styles
SHADE_ENABLED_STYLES = {"pencil_grey", "pencil_colour"}

CANVAS_MAX_W = 900   # maximum total canvas width  (both panels combined)
CANVAS_MAX_H = 520   # maximum canvas height

PANEL_BG   = "#2b2b2b"
ACCENT     = "#4a9eff"
TEXT_FG    = "#e0e0e0"
SLIDER_BG  = "#3c3f41"
BTN_BG     = "#4a9eff"
BTN_FG     = "#ffffff"
BTN_ACTIVE = "#5aaeFF"
DIVIDER    = "#555555"


# ──────────────────────────────────────────────────────────────────────────────
# Helper: OpenCV image  →  PIL Image  →  ImageTk.PhotoImage
# ──────────────────────────────────────────────────────────────────────────────

def cv_to_pil(cv_img: np.ndarray) -> Image.Image:
    """Convert a BGR or greyscale OpenCV array to a PIL RGB/L image."""
    if len(cv_img.shape) == 2:                          # greyscale
        return Image.fromarray(cv_img)
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


def fit_image(pil_img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """Down-scale *pil_img* so it fits inside (max_w × max_h), preserving ratio."""
    w, h = pil_img.size
    scale = min(max_w / w, max_h / h, 1.0)             # never up-scale
    if scale < 1.0:
        pil_img = pil_img.resize(
            (int(w * scale), int(h * scale)), Image.LANCZOS
        )
    return pil_img


# ──────────────────────────────────────────────────────────────────────────────
# Processing logic (pure, no GUI dependency)
# ──────────────────────────────────────────────────────────────────────────────

def apply_style(
    bgr_image: np.ndarray,
    style_key: str,
    sigma_s: float,
    sigma_r: float,
    shade_factor: float,
) -> np.ndarray:
    """
    Apply the requested cartoon style and return the result as a BGR ndarray.
    Raises ValueError for unknown style_key.
    """
    if style_key == "style1":
        return cv2.stylization(bgr_image, sigma_s=sigma_s, sigma_r=sigma_r)

    elif style_key == "style2":
        return cv2.stylization(bgr_image, sigma_s=sigma_s, sigma_r=sigma_r)

    elif style_key == "pencil_grey":
        grey, _ = cv2.pencilSketch(
            bgr_image,
            sigma_s=sigma_s,
            sigma_r=sigma_r,
            shade_factor=shade_factor,
        )
        return grey                                     # single-channel

    elif style_key == "pencil_colour":
        _, colour = cv2.pencilSketch(
            bgr_image,
            sigma_s=sigma_s,
            sigma_r=sigma_r,
            shade_factor=shade_factor,
        )
        return colour

    elif style_key == "detail":
        return cv2.detailEnhance(bgr_image, sigma_s=sigma_s, sigma_r=sigma_r)

    else:
        raise ValueError(f"Unknown style key: {style_key!r}")


# ──────────────────────────────────────────────────────────────────────────────
# Main application window
# ──────────────────────────────────────────────────────────────────────────────

class CartoonifyApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("OpenCV Cartoonify")
        self.resizable(True, True)
        self.configure(bg=PANEL_BG)

        # ── state ──────────────────────────────────────────────────────────
        self._source_path: Path | None = None          # path of the opened file
        self._bgr_original: np.ndarray | None = None   # original BGR image
        self._bgr_result:   np.ndarray | None = None   # latest processed image

        # Tkinter variables (kept as instance attrs so they are never GC'd)
        self._style_var      = tk.StringVar(value="style1")
        self._sigma_s_var    = tk.DoubleVar(value=150.0)
        self._sigma_r_var    = tk.DoubleVar(value=0.25)
        self._shade_var      = tk.DoubleVar(value=0.05)
        self._status_var     = tk.StringVar(value="Open an image to get started.")

        # PhotoImage references – must stay alive while displayed
        self._tk_original: ImageTk.PhotoImage | None = None
        self._tk_result:   ImageTk.PhotoImage | None = None

        self._build_ui()
        self._refresh_controls()          # set initial enable/disable states

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_toolbar()
        self._build_main_area()
        self._build_status_bar()

    def _build_toolbar(self):
        bar = tk.Frame(self, bg=PANEL_BG, pady=6)
        bar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(
            bar, text="⊕  Open Image", command=self._open_image,
            bg=BTN_BG, fg=BTN_FG, activebackground=BTN_ACTIVE,
            activeforeground=BTN_FG, relief=tk.FLAT, padx=12, pady=4,
            font=("Segoe UI", 10, "bold"), cursor="hand2",
        ).pack(side=tk.LEFT, padx=(10, 6))

        tk.Button(
            bar, text="⬇  Save Result", command=self._save_result,
            bg="#4caf50", fg=BTN_FG, activebackground="#66bb6a",
            activeforeground=BTN_FG, relief=tk.FLAT, padx=12, pady=4,
            font=("Segoe UI", 10, "bold"), cursor="hand2",
        ).pack(side=tk.LEFT, padx=6)

    def _build_main_area(self):
        container = tk.Frame(self, bg=PANEL_BG)
        container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ── left controls panel ────────────────────────────────────────────
        ctrl = tk.Frame(container, bg=PANEL_BG, width=210, padx=12, pady=10)
        ctrl.pack(side=tk.LEFT, fill=tk.Y)
        ctrl.pack_propagate(False)

        self._build_controls(ctrl)

        # vertical divider
        ttk.Separator(container, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=2
        )

        # ── right canvas area ──────────────────────────────────────────────
        canvas_frame = tk.Frame(container, bg=PANEL_BG)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self._build_canvas_area(canvas_frame)

    def _build_controls(self, parent: tk.Frame):
        """Populate the left-hand controls panel."""

        def section_label(text):
            tk.Label(
                parent, text=text, bg=PANEL_BG, fg=ACCENT,
                font=("Segoe UI", 9, "bold"), anchor="w",
            ).pack(fill=tk.X, pady=(10, 2))

        # ── style selection ────────────────────────────────────────────────
        section_label("STYLE")
        self._style_radios: list[tk.Radiobutton] = []
        for label, key in STYLES.items():
            rb = tk.Radiobutton(
                parent, text=label,
                variable=self._style_var, value=key,
                bg=PANEL_BG, fg=TEXT_FG, selectcolor=SLIDER_BG,
                activebackground=PANEL_BG, activeforeground=TEXT_FG,
                font=("Segoe UI", 9), anchor="w",
                command=self._on_style_changed,
            )
            rb.pack(fill=tk.X, ipady=1)
            self._style_radios.append(rb)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        # ── sliders ────────────────────────────────────────────────────────
        section_label("PARAMETERS")
        self._sigma_s_label = self._make_slider(
            parent,
            label_prefix="Sigma-S (spatial)",
            variable=self._sigma_s_var,
            from_=1, to=200, resolution=1,
        )
        self._sigma_r_label = self._make_slider(
            parent,
            label_prefix="Sigma-R (range)",
            variable=self._sigma_r_var,
            from_=0.01, to=1.0, resolution=0.01,
        )
        self._shade_label, self._shade_slider = self._make_slider(
            parent,
            label_prefix="Shade Factor",
            variable=self._shade_var,
            from_=0.01, to=0.10, resolution=0.005,
            return_slider=True,
        )

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        # ── apply button ───────────────────────────────────────────────────
        self._apply_btn = tk.Button(
            parent, text="▶  Apply",
            command=self._apply,
            bg=BTN_BG, fg=BTN_FG,
            activebackground=BTN_ACTIVE, activeforeground=BTN_FG,
            relief=tk.FLAT, padx=12, pady=6,
            font=("Segoe UI", 10, "bold"), cursor="hand2",
        )
        self._apply_btn.pack(fill=tk.X, pady=(4, 0))

    def _make_slider(
        self,
        parent,
        label_prefix: str,
        variable: tk.DoubleVar,
        from_: float,
        to: float,
        resolution: float,
        return_slider: bool = False,
    ):
        """Create a labelled slider row.  Returns the value label (and
        optionally the Scale widget itself when return_slider=True)."""

        row = tk.Frame(parent, bg=PANEL_BG)
        row.pack(fill=tk.X, pady=2)

        # Static name on the left, live value on the right
        tk.Label(
            row, text=label_prefix, bg=PANEL_BG, fg=TEXT_FG,
            font=("Segoe UI", 8), anchor="w",
        ).pack(side=tk.LEFT)

        val_label = tk.Label(
            row, textvariable=variable, bg=PANEL_BG, fg=ACCENT,
            font=("Segoe UI", 8, "bold"), width=5, anchor="e",
        )
        val_label.pack(side=tk.RIGHT)

        slider = tk.Scale(
            parent,
            variable=variable,
            from_=from_, to=to, resolution=resolution,
            orient=tk.HORIZONTAL,
            bg=SLIDER_BG, fg=TEXT_FG, troughcolor="#555",
            highlightthickness=0, bd=0, showvalue=False,
        )
        slider.pack(fill=tk.X, pady=(0, 4))

        if return_slider:
            return val_label, slider
        return val_label

    def _build_canvas_area(self, parent: tk.Frame):
        """Build the dual-panel image canvas (original | result)."""

        # Column headers
        header = tk.Frame(parent, bg=PANEL_BG)
        header.pack(fill=tk.X)
        for text in ("Original", "Cartoon Result"):
            tk.Label(
                header, text=text, bg=PANEL_BG, fg=TEXT_FG,
                font=("Segoe UI", 9, "bold"),
            ).pack(side=tk.LEFT, expand=True)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(2, 4))

        # The canvas itself (single widget, we draw both halves on it)
        self._canvas = tk.Canvas(
            parent,
            bg="#1e1e1e",
            width=CANVAS_MAX_W,
            height=CANVAS_MAX_H,
            highlightthickness=1,
            highlightbackground=DIVIDER,
        )
        self._canvas.pack(fill=tk.BOTH, expand=True)

        # Placeholder text shown before any image is loaded
        self._canvas.create_text(
            CANVAS_MAX_W // 2, CANVAS_MAX_H // 2,
            text="Open an image to begin",
            fill="#555555",
            font=("Segoe UI", 14),
            tags="placeholder",
        )

    def _build_status_bar(self):
        bar = tk.Frame(self, bg="#1c1c1c", pady=4)
        bar.pack(side=tk.BOTTOM, fill=tk.X)

        tk.Label(
            bar, textvariable=self._status_var,
            bg="#1c1c1c", fg="#aaaaaa",
            font=("Segoe UI", 8), anchor="w", padx=10,
        ).pack(fill=tk.X)

    # ── event handlers ────────────────────────────────────────────────────────

    def _on_style_changed(self):
        """When the user picks a different style, load its defaults and refresh."""
        key = self._style_var.get()
        d = DEFAULTS[key]
        self._sigma_s_var.set(d["sigma_s"])
        self._sigma_r_var.set(d["sigma_r"])
        self._shade_var.set(d["shade_factor"])
        self._refresh_controls()

    def _refresh_controls(self):
        """Enable/disable widgets based on current state."""
        has_image   = self._bgr_original is not None
        has_result  = self._bgr_result   is not None
        is_pencil   = self._style_var.get() in SHADE_ENABLED_STYLES

        # Apply button – needs an image loaded
        self._apply_btn.config(state=tk.NORMAL if has_image else tk.DISABLED)

        # Shade-factor slider – only meaningful for pencil styles
        shade_state = tk.NORMAL if is_pencil else tk.DISABLED
        self._shade_slider.config(state=shade_state)
        self._shade_label.config(
            fg=ACCENT if is_pencil else "#555555"
        )

    def _open_image(self):
        """Open a file-chooser dialog and load the selected image."""
        path_str = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("All files",   "*.*"),
            ],
        )
        if not path_str:
            return                              # user cancelled

        path = Path(path_str)
        bgr  = cv2.imread(str(path))
        if bgr is None:
            messagebox.showerror(
                "Load Error",
                f"Could not read image:\n{path}\n\n"
                "Make sure it is a valid image file.",
            )
            return

        self._source_path   = path
        self._bgr_original  = bgr
        self._bgr_result    = None

        self._set_status(f"Loaded  {path.name}  ({bgr.shape[1]}×{bgr.shape[0]} px)")
        self._render_canvas()
        self._refresh_controls()

    def _apply(self):
        """Run the selected cartoon style in a background thread."""
        if self._bgr_original is None:
            return

        self._apply_btn.config(state=tk.DISABLED, text="Processing…")
        self._set_status("Processing…")

        def worker():
            try:
                result = apply_style(
                    bgr_image    = self._bgr_original,
                    style_key    = self._style_var.get(),
                    sigma_s      = self._sigma_s_var.get(),
                    sigma_r      = self._sigma_r_var.get(),
                    shade_factor = self._shade_var.get(),
                )
                self._bgr_result = result
                self.after(0, self._on_apply_done)          # back to main thread
            except Exception as exc:                        # noqa: BLE001
                self.after(0, lambda: self._on_apply_error(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _on_apply_done(self):
        style_label = next(
            k for k, v in STYLES.items() if v == self._style_var.get()
        )
        self._set_status(
            f"Done — {style_label}  |  "
            f"σ_s={self._sigma_s_var.get():.0f}  "
            f"σ_r={self._sigma_r_var.get():.2f}  "
            f"shade={self._shade_var.get():.3f}"
        )
        self._render_canvas()
        self._apply_btn.config(state=tk.NORMAL, text="▶  Apply")
        self._refresh_controls()

    def _on_apply_error(self, exc: Exception):
        self._set_status(f"Error: {exc}")
        self._apply_btn.config(state=tk.NORMAL, text="▶  Apply")
        messagebox.showerror("Processing Error", str(exc))

    def _save_result(self):
        """Write the latest result image to a user-chosen path."""
        if self._bgr_result is None:
            messagebox.showinfo("Nothing to Save", "Apply a style first.")
            return

        # Suggest a sensible default file name
        stem   = self._source_path.stem if self._source_path else "image"
        style  = self._style_var.get()
        default_name = f"{stem}_cartoon_{style}.png"

        save_path = filedialog.asksaveasfilename(
            title="Save cartoon image",
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[
                ("PNG",  "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("BMP",  "*.bmp"),
                ("All",  "*.*"),
            ],
        )
        if not save_path:
            return                              # user cancelled

        ok = cv2.imwrite(save_path, self._bgr_result)
        if ok:
            self._set_status(f"Saved → {Path(save_path).name}")
        else:
            messagebox.showerror("Save Error", f"Could not write to:\n{save_path}")

    # ── canvas rendering ──────────────────────────────────────────────────────

    def _render_canvas(self):
        """Redraw both canvas panels (original left, result right)."""
        self._canvas.delete("all")

        cw = self._canvas.winfo_width()  or CANVAS_MAX_W
        ch = self._canvas.winfo_height() or CANVAS_MAX_H

        half_w = cw // 2 - 4            # 4 px gap around the centre divider

        # ── original ──────────────────────────────────────────────────────
        if self._bgr_original is not None:
            pil_orig = fit_image(cv_to_pil(self._bgr_original), half_w, ch)
            self._tk_original = ImageTk.PhotoImage(pil_orig)
            self._canvas.create_image(
                half_w // 2 + 2, ch // 2,          # centred in left half
                anchor=tk.CENTER,
                image=self._tk_original,
            )
        else:
            self._canvas.create_text(
                half_w // 2 + 2, ch // 2,
                text="Original", fill="#444444",
                font=("Segoe UI", 12),
            )

        # centre divider
        self._canvas.create_line(cw // 2, 0, cw // 2, ch, fill=DIVIDER, width=2)

        # ── result ────────────────────────────────────────────────────────
        if self._bgr_result is not None:
            pil_result = fit_image(cv_to_pil(self._bgr_result), half_w, ch)
            self._tk_result = ImageTk.PhotoImage(pil_result)
            self._canvas.create_image(
                cw // 2 + half_w // 2 + 2, ch // 2,   # centred in right half
                anchor=tk.CENTER,
                image=self._tk_result,
            )
        else:
            self._canvas.create_text(
                cw // 2 + half_w // 2 + 2, ch // 2,
                text="Apply a style to see result",
                fill="#444444",
                font=("Segoe UI", 12),
            )

    # ── utility ───────────────────────────────────────────────────────────────

    def _set_status(self, text: str):
        self._status_var.set(text)
        self.update_idletasks()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = CartoonifyApp()
    app.mainloop()
