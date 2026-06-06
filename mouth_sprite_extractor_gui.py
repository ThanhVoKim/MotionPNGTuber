#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mouth_sprite_extractor_gui.py

GUI tool to automatically extract mouth sprites (5 types of PNG) from a video.

Features:
1. Select a video (drag & drop supported)
2. Automatically analyze and prepare mouth tracking after selecting a video
3. Manually add candidate frames while playing/stopping/stepping in the candidate selection player in a separate window
4. Auxiliary use of automatic candidate selection and mouth shape assignment if necessary
5. Adjust crop range (top/bottom/left/right) and feather width separately
6. Check preview updates, live tests, and output comprehensively

Usage:
    python mouth_sprite_extractor_gui.py
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
import traceback
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Tkinter imports
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Optional: drag & drop support
_HAS_TK_DND = False
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    _HAS_TK_DND = True
except Exception:
    _HAS_TK_DND = False

# PIL for image display in Tkinter
try:
    from PIL import Image, ImageTk
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False
    print("[warn] PIL not installed. Preview will be limited.")

try:
    import sounddevice as sd
    _HAS_SOUNDDEVICE = True
except Exception:
    sd = None  # type: ignore
    _HAS_SOUNDDEVICE = False

# Core extractor module
from motionpngtuber.mouth_sprite_extractor import (
    MouthSpriteExtractor,
    MouthFrameInfo,
    get_unique_output_dir,
    quad_wh,
    warp_frame_to_norm,
    make_ellipse_mask,
    feather_mask,
    ensure_even_ge2,
    write_image_file,
)

# Auto classification modules
from motionpngtuber.mouth_feature_analyzer import MouthFeatureAnalyzer, MouthFeatures
from motionpngtuber.mouth_auto_classifier import MouthAutoClassifier
from motionpngtuber.auto_crop_estimator import AutoCropEstimator
from motionpngtuber.lipsync_core import one_pole_beta
from mouth_track_gui.services import list_input_devices
from motionpngtuber.audio_linux import (
    apply_audio_resolution_for_current_process,
    cleanup_audio_device_resolution,
    resolve_audio_device_spec,
)
from motionpngtuber.platform_open import open_path_with_default_app


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_TITLE = "Mouth Sprite Extractor"
BASE_CANDIDATE_COUNT = 10
OPENING_SEQ_COUNT = 10
CANDIDATE_COUNT = BASE_CANDIDATE_COUNT + OPENING_SEQ_COUNT  # Candidate frames count
CANDIDATE_ROWS = 2
CANDIDATE_PER_ROW = (CANDIDATE_COUNT + CANDIDATE_ROWS - 1) // CANDIDATE_ROWS
MOUTH_SHAPES = ("open", "closed", "half", "e", "u")
MOUTH_ASSIGNMENT_OPTIONS = ("Unassigned",) + MOUTH_SHAPES
THUMB_SIZE = 70       # Thumbnail size
PREVIEW_SIZE = 150    # Preview size (1.5x zoom)
PLAYER_PREVIEW_MAX_W = 640
PLAYER_PREVIEW_MAX_H = 420
DEFAULT_FEATHER = 15
DEFAULT_CROP = 0      # Default crop margin
MAX_CROP = 100        # Maximum crop range
MAX_FEATHER = 40


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def create_checkerboard(w: int, h: int, cell_size: int = 10) -> np.ndarray:
    """Generate a checkerboard background for transparent display"""
    board = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, cell_size):
        for x in range(0, w, cell_size):
            if (x // cell_size + y // cell_size) % 2 == 0:
                board[y:y+cell_size, x:x+cell_size] = 200
            else:
                board[y:y+cell_size, x:x+cell_size] = 255
    return board


def composite_on_checkerboard(rgba: np.ndarray) -> np.ndarray:
    """Composite RGBA on checkerboard and return RGB image"""
    h, w = rgba.shape[:2]
    board = create_checkerboard(w, h)
    
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    rgb = rgba[:, :, :3].astype(np.float32)
    
    result = board.astype(np.float32) * (1.0 - alpha) + rgb * alpha
    return result.astype(np.uint8)


def numpy_to_photoimage(
    img: np.ndarray,
    size: int,
    *,
    color_order: str = "BGR",
) -> Optional["ImageTk.PhotoImage"]:
    """Convert BGR/RGB numpy array to PhotoImage"""
    if not _HAS_PIL:
        return None

    if color_order == "RGB":
        rgb = img
    elif color_order == "BGR":
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported color_order: {color_order}")
    img = Image.fromarray(rgb)
    
    # Resize maintaining aspect ratio
    w, h = img.size
    scale = min(size / max(w, 1), size / max(h, 1))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    return ImageTk.PhotoImage(img)


def numpy_to_photoimage_fit(
    img: np.ndarray,
    max_w: int,
    max_h: int,
    *,
    color_order: str = "BGR",
) -> Optional["ImageTk.PhotoImage"]:
    """Convert BGR/RGB numpy array to PhotoImage that fits within a specified frame"""
    if not _HAS_PIL:
        return None

    if color_order == "RGB":
        rgb = img
    elif color_order == "BGR":
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported color_order: {color_order}")
    img = Image.fromarray(rgb)
    w, h = img.size
    scale = min(max_w / max(w, 1), max_h / max(h, 1))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(img)


def first_non_none(*values):
    """Return the first non-None value to avoid truth value testing of numpy arrays."""
    for v in values:
        if v is not None:
            return v
    return None


def draw_mouth_quad_overlay(
    frame_bgr: np.ndarray,
    mf: Optional[MouthFrameInfo],
) -> np.ndarray:
    """Overlay mouth quad for player display"""
    out = frame_bgr.copy()
    if mf is None:
        cv2.putText(
            out,
            "track: unavailable",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 128, 255),
            2,
            cv2.LINE_AA,
        )
        return out

    color = (0, 220, 0) if mf.valid else (0, 0, 255)
    pts = np.asarray(mf.quad, dtype=np.float32).reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(
        out,
        f"F:{mf.frame_idx} valid={int(mf.valid)} conf={mf.confidence:.2f}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    return out


def crop_frame_around_mouth(
    frame_bgr: np.ndarray,
    mf: Optional[MouthFrameInfo],
    *,
    margin_scale: float = 4.0,
) -> np.ndarray:
    """Crop around the mouth quad to make it easier to see"""
    if mf is None:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    cx, cy = mf.center.astype(np.float32)
    crop_w = max(96.0, float(mf.width) * float(margin_scale))
    crop_h = max(96.0, float(mf.height) * float(margin_scale))

    x0 = int(round(cx - crop_w * 0.5))
    x1 = int(round(cx + crop_w * 0.5))
    y0 = int(round(cy - crop_h * 0.5))
    y1 = int(round(cy + crop_h * 0.5))

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w, max(x0 + 2, x1))
    y1 = min(h, max(y0 + 2, y1))
    if x1 <= x0 or y1 <= y0:
        return frame_bgr
    return frame_bgr[y0:y1, x0:x1].copy()


def pick_opening_sequence(
    mouth_frames: List[MouthFrameInfo],
    preselected: Optional[set[int]] = None,
    window: int = OPENING_SEQ_COUNT,
) -> List[MouthFrameInfo]:
    """Select consecutive frames near the beginning of mouth opening."""
    if window <= 0:
        return []
    n_frames = len(mouth_frames)
    if n_frames < window:
        return []

    heights = np.array([mf.height for mf in mouth_frames], dtype=np.float32)
    valid = np.array([mf.valid for mf in mouth_frames], dtype=bool)
    if int(valid.sum()) < window:
        return []

    min_h = float(np.min(heights[valid]))
    max_h = float(np.max(heights[valid]))
    range_h = max(1e-6, max_h - min_h)
    low_threshold = min_h + 0.3 * range_h
    preselected = preselected or set()

    def pick_window(require_rise: bool) -> Optional[int]:
        best_start = None
        best_score = -1e9
        for start in range(0, n_frames - window + 1):
            if not valid[start:start + window].all():
                continue
            start_h = heights[start]
            end_h = heights[start + window - 1]
            increase = end_h - start_h
            if require_rise and increase <= 0:
                continue
            overlap = sum(
                1 for i in range(start, start + window)
                if mouth_frames[i].frame_idx in preselected
            )
            low_penalty = max(0.0, start_h - low_threshold)
            score = increase - 0.5 * low_penalty - overlap * 0.1 * range_h
            if score > best_score:
                best_score = score
                best_start = start
        return best_start

    best_start = pick_window(require_rise=True)
    if best_start is None:
        best_start = pick_window(require_rise=False)
    if best_start is None:
        return []

    return [mouth_frames[i] for i in range(best_start, best_start + window)]


def extract_sprite_with_crop(
    frame_bgr: np.ndarray,
    quad: np.ndarray,
    unified_w: int,
    unified_h: int,
    crop_top: int = 0,
    crop_bottom: int = 0,
    crop_left: int = 0,
    crop_right: int = 0,
    feather_px: int = 15,
) -> np.ndarray:
    """
    Extract sprites by applying crop range and feather.
    """
    # Convert to normalized space
    patch_bgr = warp_frame_to_norm(frame_bgr, quad, unified_w, unified_h)
    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    
    # Elliptical mask with crop range applied
    # Shift ellipse center and adjust radius
    # Crop top = shift ellipse down, crop bottom = shift ellipse up
    cx = unified_w // 2 + (crop_left - crop_right) // 2
    cy = unified_h // 2 + (crop_top - crop_bottom) // 2  # Fix direction
    rx = (unified_w - crop_left - crop_right) // 2
    ry = (unified_h - crop_top - crop_bottom) // 2
    
    rx = max(1, min(rx, unified_w // 2 - 1))
    ry = max(1, min(ry, unified_h // 2 - 1))
    
    mask = np.zeros((unified_h, unified_w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0.0, 0.0, 360.0, 255, -1)
    
    # Apply feather
    mask_f = feather_mask(mask, feather_px)
    
    # Generate RGBA image
    rgba = np.zeros((unified_h, unified_w, 4), dtype=np.uint8)
    rgba[:, :, :3] = patch_rgb
    rgba[:, :, 3] = (mask_f * 255).astype(np.uint8)
    
    return rgba


# ---------------------------------------------------------------------------
# Main GUI Application
# ---------------------------------------------------------------------------

class MouthSpriteExtractorApp(tk.Tk if not _HAS_TK_DND else TkinterDnD.Tk):
    """Mouth Sprite Extractor GUI Application"""
    
    def __init__(self):
        super().__init__()
        
        self.title(APP_TITLE)
        self.resizable(True, True)
        self._configure_initial_window_size()
        
        # State
        self.video_path: str = ""
        self.extractor: Optional[MouthSpriteExtractor] = None
        self.valid_frames: List[MouthFrameInfo] = []
        self._mouth_frame_by_idx: Dict[int, MouthFrameInfo] = {}
        self.candidate_frames: List[MouthFrameInfo] = []  # Candidate frames (in order of openness)
        self.candidate_images: List[Optional["ImageTk.PhotoImage"]] = []
        self.assignments: Dict[int, int] = {}  # candidate_idx -> mouth_type (1-5)
        self.preview_sprites: Dict[str, np.ndarray] = {}
        self.preview_images: Dict[str, "ImageTk.PhotoImage"] = {}
        self.unified_size: Optional[Tuple[int, int]] = None
        self.is_analyzing = False
        self.busy_mode: str = ""
        self.selected_candidate_idx = 0
        self.preview_state_code = "empty"
        self.preview_state_var = tk.StringVar(value="Preview Not Updated")
        self.workflow_state_var = tk.StringVar(value="State: Please select a video")
        self.busy_status_var = tk.StringVar(value="Processing State: Standby")
        self._suspend_preview_traces = False
        
        # Cached video capture for preview
        self._cached_cap: Optional[cv2.VideoCapture] = None
        self._player_cap: Optional[cv2.VideoCapture] = None
        self.player_window: Optional[tk.Toplevel] = None
        self.live_test_window: Optional[tk.Toplevel] = None
        self.player_total_frames = 0
        self.player_fps = 30.0
        self.player_current_frame_idx = 0
        self.player_playing = False
        self.player_job: Optional[str] = None
        self.player_image: Optional["ImageTk.PhotoImage"] = None
        self._updating_player_scale = False
        self._ignore_next_seek = False
        self._live_stream = None
        self._live_job: Optional[str] = None
        self._live_feat_q: "queue.Queue[tuple[float, float]]" = queue.Queue(maxsize=256)
        self._live_audio_items: List[dict] = []
        self._live_audio_state: Optional[dict] = None
        self._live_audio_resolution: Optional[dict] = None
        self._live_audio_apply_state: Optional[dict] = None
        self.live_preview_image: Optional["ImageTk.PhotoImage"] = None
        
        # Log queue for thread-safe logging
        self.log_queue: queue.Queue[str] = queue.Queue()
        
        # Build UI
        self._build_ui()
        self._build_player_window()
        self._build_live_test_window()
        self._setup_state_traces()
        self._refresh_workflow_state()
        
        # Start log polling
        self._poll_logs()

    def _configure_initial_window_size(self):
        """Set the initial window size larger according to the screen size"""
        self.update_idletasks()
        screen_w = max(800, int(self.winfo_screenwidth() or 0))
        screen_h = max(700, int(self.winfo_screenheight() or 0))

        width = min(1500, max(1180, screen_w - 120))
        height = min(1100, max(920, screen_h - 120))
        width = min(width, screen_w)
        height = min(height, screen_h)

        x = max(0, (screen_w - width) // 2)
        y = max(0, (screen_h - height) // 2)

        self.geometry(f"{width}x{height}+{x}+{y}")
        self.minsize(1100, 900)
    
    def _build_ui(self):
        """Build UI"""
        # Main scrollable area (for shorter screens)
        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True)

        main_canvas = tk.Canvas(container, highlightthickness=0)
        main_scroll = ttk.Scrollbar(
            container, orient=tk.VERTICAL, command=main_canvas.yview
        )
        main_canvas.configure(yscrollcommand=main_scroll.set)
        main_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        main_frame = ttk.Frame(main_canvas, padding=10)
        main_window = main_canvas.create_window(
            (0, 0), window=main_frame, anchor=tk.NW
        )

        main_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all")),
        )
        main_canvas.bind(
            "<Configure>",
            lambda e: main_canvas.itemconfigure(main_window, width=e.width),
        )
        
        # --- Video selection ---
        video_frame = ttk.LabelFrame(main_frame, text="Video File", padding=5)
        video_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.video_var = tk.StringVar()
        video_entry = ttk.Entry(video_frame, textvariable=self.video_var, state="readonly")
        video_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        video_btn = ttk.Button(video_frame, text="Select...", command=self._on_select_video)
        video_btn.pack(side=tk.RIGHT)
        
        # Drag & drop support
        if _HAS_TK_DND:
            video_entry.drop_target_register(DND_FILES)
            video_entry.dnd_bind("<<Drop>>", self._on_drop_video)

        guide_frame = ttk.LabelFrame(main_frame, text="Quick Guide", padding=6)
        guide_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(
            guide_frame,
            text=(
                "1. Select a video (analyzed automatically)\n"
                "2. Add good mouth shapes to candidates using the player\n"
                "3. Assign open / closed / half / e / u to each candidate\n"
                "4. Update Preview -> Live Test -> Output"
            ),
            justify=tk.LEFT,
        ).pack(anchor=tk.W)

        ttk.Label(
            main_frame,
            textvariable=self.workflow_state_var,
            font=("", 9, "bold"),
        ).pack(fill=tk.X, pady=(0, 10))

        # --- Analyze button ---
        self.analyze_btn = ttk.Button(
            main_frame, text="Analyze/Re-analyze", command=self._on_analyze
        )
        self.analyze_btn.pack(fill=tk.X, pady=(0, 10))

        busy_frame = ttk.LabelFrame(main_frame, text="Processing State", padding=5)
        busy_frame.pack(fill=tk.X, pady=(0, 10))
        self.busy_progress = ttk.Progressbar(
            busy_frame,
            mode="indeterminate",
        )
        self.busy_progress.pack(fill=tk.X)
        ttk.Label(
            busy_frame,
            textvariable=self.busy_status_var,
            font=("", 9),
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(6, 0))

        # --- Player launcher area ---
        player_frame = ttk.LabelFrame(
            main_frame,
            text="Player (Separate Window)",
            padding=5,
        )
        player_frame.pack(fill=tk.X, pady=(0, 10))

        player_bar = ttk.Frame(player_frame)
        player_bar.pack(fill=tk.X)

        self.player_status_var = tk.StringVar(value="F:0 / 0")
        ttk.Label(
            player_bar,
            textvariable=self.player_status_var,
            font=("", 9),
        ).pack(side=tk.LEFT)

        self.open_player_btn = ttk.Button(
            player_bar,
            text="Open Candidate Selection Player (Recommended)",
            command=self._show_player_window,
        )
        self.open_player_btn.pack(side=tk.RIGHT)

        ttk.Label(
            player_frame,
            text="Player: Find good frames and register them as candidates. Main screen: Assign, preview, and output.",
            font=("", 9),
            wraplength=1000,
            justify=tk.LEFT,
        ).pack(fill=tk.X, pady=(6, 0))
         
        # --- Candidate frames area ---
        cand_frame = ttk.LabelFrame(
            main_frame,
            text="Candidate Frames",
            padding=5,
        )
        cand_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(
            cand_frame,
            text="Find good mouth shapes in the player and add them to candidates. You can also register by double-clicking. Auto-selection is an auxiliary function.",
            font=("", 9),
            wraplength=1000,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 6))

        # Candidate area
        cand_canvas = tk.Canvas(cand_frame, height=260)
        cand_canvas.pack(fill=tk.X, expand=True)

        self.cand_inner = ttk.Frame(cand_canvas)
        cand_canvas.create_window((0, 0), window=self.cand_inner, anchor=tk.NW)
        self.cand_inner.bind(
            "<Configure>",
            lambda e: cand_canvas.configure(scrollregion=cand_canvas.bbox("all")),
        )
        
        # Candidate slots
        self.cand_slot_titles: List[ttk.Label] = []
        self.cand_labels: List[ttk.Label] = []
        self.cand_entries: List[ttk.Combobox] = []
        self.cand_vars: List[tk.StringVar] = []
        self.cand_frame_labels: List[ttk.Label] = []
        
        per_row = CANDIDATE_PER_ROW
        for i in range(CANDIDATE_COUNT):
            row = i // per_row
            col = i % per_row
            col_frame = ttk.Frame(self.cand_inner)
            col_frame.grid(row=row, column=col, padx=3, pady=3, sticky=tk.N)

            slot_title = ttk.Label(col_frame, text=f"Candidate{i+1}", font=("", 8, "bold"))
            slot_title.pack()
            self.cand_slot_titles.append(slot_title)
             
            # Thumbnail
            thumb_label = ttk.Label(col_frame, text="", width=10, anchor=tk.CENTER, relief=tk.SUNKEN)
            thumb_label.pack()
            self.cand_labels.append(thumb_label)
            
            # Frame Number
            frame_label = ttk.Label(col_frame, text="", font=("", 8))
            frame_label.pack()
            self.cand_frame_labels.append(frame_label)
             
            # Assignment Input
            var = tk.StringVar(value=MOUTH_ASSIGNMENT_OPTIONS[0])
            entry = ttk.Combobox(
                col_frame,
                textvariable=var,
                values=MOUTH_ASSIGNMENT_OPTIONS,
                state="readonly",
                width=7,
                justify=tk.CENTER,
            )
            entry.pack()
            self.cand_vars.append(var)
            self.cand_entries.append(entry)

            for widget in (col_frame, slot_title, thumb_label, frame_label, entry):
                widget.bind("<Button-1>", lambda _e, idx=i: self._select_candidate_slot(idx))
                widget.bind("<Double-Button-1>", lambda _e, idx=i: self._on_candidate_double_click(idx))
            entry.bind("<FocusIn>", lambda _e, idx=i: self._select_candidate_slot(idx))
         
        # Legend
        legend_frame = ttk.Frame(cand_frame)
        legend_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(legend_frame, text="Select mouth shape names directly under each candidate", font=("", 9)).pack(side=tk.LEFT)
        self.selected_slot_var = tk.StringVar(value="Selected Candidate: 1")
        ttk.Label(legend_frame, textvariable=self.selected_slot_var, font=("", 9)).pack(side=tk.LEFT, padx=(12, 0))
        self.auto_fill_btn = ttk.Button(
            legend_frame,
            text="Auto-select Candidates",
            command=self._on_fill_auto_candidates,
            state=tk.DISABLED,
        )
        self.auto_fill_btn.pack(side=tk.RIGHT)
        self.auto_assign_btn = ttk.Button(
            legend_frame,
            text="Auto-assign Mouth Shapes to Candidates",
            command=self._on_auto_assign,
            state=tk.DISABLED,
        )
        self.auto_assign_btn.pack(side=tk.RIGHT, padx=(0, 6))
        
        # --- Crop settings ---
        crop_frame = ttk.LabelFrame(main_frame, text="Crop Range (Reduce Margin)", padding=5)
        crop_frame.pack(fill=tk.X, pady=(0, 10))
        
        crop_grid = ttk.Frame(crop_frame)
        crop_grid.pack()
        
        self.crop_vars = {
            "top": tk.IntVar(value=DEFAULT_CROP),
            "bottom": tk.IntVar(value=DEFAULT_CROP),
            "left": tk.IntVar(value=DEFAULT_CROP),
            "right": tk.IntVar(value=DEFAULT_CROP),
        }
        
        # Top
        ttk.Label(crop_grid, text="Top:").grid(row=0, column=0, sticky=tk.E)
        ttk.Scale(crop_grid, from_=0, to=MAX_CROP, variable=self.crop_vars["top"], 
                  orient=tk.HORIZONTAL, length=100).grid(row=0, column=1)
        self.crop_labels = {}
        self.crop_labels["top"] = ttk.Label(crop_grid, text="0px", width=5)
        self.crop_labels["top"].grid(row=0, column=2)
        
        # Bottom
        ttk.Label(crop_grid, text="Bottom:").grid(row=1, column=0, sticky=tk.E)
        ttk.Scale(crop_grid, from_=0, to=MAX_CROP, variable=self.crop_vars["bottom"],
                  orient=tk.HORIZONTAL, length=100).grid(row=1, column=1)
        self.crop_labels["bottom"] = ttk.Label(crop_grid, text="0px", width=5)
        self.crop_labels["bottom"].grid(row=1, column=2)
        
        # Left
        ttk.Label(crop_grid, text="Left:").grid(row=0, column=3, sticky=tk.E, padx=(20, 0))
        ttk.Scale(crop_grid, from_=0, to=MAX_CROP, variable=self.crop_vars["left"],
                  orient=tk.HORIZONTAL, length=100).grid(row=0, column=4)
        self.crop_labels["left"] = ttk.Label(crop_grid, text="0px", width=5)
        self.crop_labels["left"].grid(row=0, column=5)
        
        # Right
        ttk.Label(crop_grid, text="Right:").grid(row=1, column=3, sticky=tk.E, padx=(20, 0))
        ttk.Scale(crop_grid, from_=0, to=MAX_CROP, variable=self.crop_vars["right"],
                  orient=tk.HORIZONTAL, length=100).grid(row=1, column=4)
        self.crop_labels["right"] = ttk.Label(crop_grid, text="0px", width=5)
        self.crop_labels["right"].grid(row=1, column=5)

        # Auto-optimization Button
        auto_crop_frame = ttk.Frame(crop_frame)
        auto_crop_frame.pack(fill=tk.X, pady=(5, 0))
        self.auto_crop_btn = ttk.Button(
            auto_crop_frame, text="Auto-adjust Crop", command=self._on_auto_crop, state=tk.DISABLED
        )
        self.auto_crop_btn.pack(side=tk.RIGHT)

        # --- Feather slider ---
        feather_frame = ttk.LabelFrame(main_frame, text="Feather Width", padding=5)
        feather_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.feather_var = tk.IntVar(value=DEFAULT_FEATHER)
        self.feather_slider = ttk.Scale(
            feather_frame,
            from_=0,
            to=MAX_FEATHER,
            orient=tk.HORIZONTAL,
            variable=self.feather_var,
        )
        self.feather_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        self.feather_label = ttk.Label(feather_frame, text=f"{DEFAULT_FEATHER}px", width=6)
        self.feather_label.pack(side=tk.RIGHT)
        
        # --- Update button ---
        self.update_btn = ttk.Button(
            main_frame, text="Update Preview", command=self._on_update_preview, state=tk.DISABLED
        )
        self.update_btn.pack(fill=tk.X, pady=(0, 4))

        ttk.Label(
            main_frame,
            textvariable=self.preview_state_var,
            font=("", 9),
        ).pack(anchor=tk.W, pady=(0, 8))

        self.live_test_btn = ttk.Button(
            main_frame,
            text="Live Test (Check Mouth Shape and Mic Input)",
            command=self._show_live_test_window,
            state=tk.DISABLED,
        )
        self.live_test_btn.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(
            main_frame,
            text="The live test uses the current preview results.",
            font=("", 9),
        ).pack(anchor=tk.W, pady=(0, 10))
        
        # --- Preview area ---
        preview_frame = ttk.LabelFrame(main_frame, text="Output Preview", padding=5)
        preview_frame.pack(fill=tk.X, pady=(0, 10))
        
        preview_inner = ttk.Frame(preview_frame)
        preview_inner.pack(expand=True)
        
        self.preview_labels: Dict[str, ttk.Label] = {}
        self.out_frame_labels: Dict[str, ttk.Label] = {}
        
        mouth_names = ["open", "closed", "half", "e", "u"]
        
        for i, name in enumerate(mouth_names):
            col_frame = ttk.Frame(preview_inner)
            col_frame.grid(row=0, column=i, padx=5, pady=5)
            
            preview_label = ttk.Label(
                col_frame,
                text="(Unselected)",
                width=12,
                anchor=tk.CENTER,
                relief=tk.SUNKEN,
            )
            preview_label.pack()
            self.preview_labels[name] = preview_label
            
            name_label = ttk.Label(col_frame, text=name, font=("", 9, "bold"))
            name_label.pack()
            
            frame_label = ttk.Label(col_frame, text="", font=("", 8))
            frame_label.pack()
            self.out_frame_labels[name] = frame_label
        
        # --- Output button ---
        self.output_btn = ttk.Button(
            main_frame, text="Output", command=self._on_output, state=tk.DISABLED
        )
        self.output_btn.pack(fill=tk.X, pady=(0, 10))
        
        # --- Log area ---
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(log_frame, height=5, state=tk.DISABLED, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self._update_candidate_slot_highlight()
        self.bind("<KeyPress>", self._on_key_press)
    
    def log(self, message: str):
        """Add message to log (thread-safe)"""
        self.log_queue.put(message)

    def _start_busy_state(self, mode: str, detail: str):
        """Start progress indicator"""
        self.busy_mode = mode
        self.busy_status_var.set(f"Processing State: {mode} - {detail}")
        self.busy_progress.start(10)

    def _update_busy_state(self, detail: str):
        """Update progress indicator text"""
        if not self.busy_mode:
            return
        self.busy_status_var.set(f"Processing State: {self.busy_mode} - {detail}")

    def _finish_busy_state(self, detail: str = "Standby"):
        """Stop progress indicator"""
        self.busy_progress.stop()
        self.busy_mode = ""
        self.busy_status_var.set(f"Processing State: {detail}")
    
    def _poll_logs(self):
        """Poll log queue to update UI"""
        while not self.log_queue.empty():
            try:
                msg = self.log_queue.get_nowait()
                self.log_text.configure(state=tk.NORMAL)
                self.log_text.insert(tk.END, msg + "\n")
                self.log_text.see(tk.END)
                self.log_text.configure(state=tk.DISABLED)
                if self.busy_mode:
                    self._update_busy_state(msg)
            except queue.Empty:
                break
        
        # Update crop labels
        for key in self.crop_vars:
            val = self.crop_vars[key].get()
            self.crop_labels[key].configure(text=f"{val}px")
        
        # Update feather label
        self.feather_label.configure(text=f"{self.feather_var.get()}px")
        
        self.after(100, self._poll_logs)

    def _setup_state_traces(self):
        """Monitor input changes that require preview update"""
        for var in self.cand_vars:
            var.trace_add("write", self._on_assignment_var_changed)
        for var in self.crop_vars.values():
            var.trace_add("write", self._on_crop_or_feather_changed)
        self.feather_var.trace_add("write", self._on_crop_or_feather_changed)

    def _set_preview_state_code(self, code: str):
        mapping = {
            "empty": "Preview Not Updated",
            "dirty": "Candidates changed (please update again)",
            "ready": "Preview updated",
        }
        self.preview_state_code = code
        self.preview_state_var.set(mapping.get(code, "Preview Not Updated"))
        self._refresh_workflow_state()

    def _refresh_workflow_state(self):
        if self.is_analyzing:
            self.workflow_state_var.set("State: Analyzing...")
            return
        if not self.video_path:
            self.workflow_state_var.set("State: Please select a video")
            return
        if not self.unified_size or not self.valid_frames:
            self.workflow_state_var.set("State: Video Selected / Waiting for Analysis")
            return

        parts = [f"State: Analysis Complete / Candidates {len(self.candidate_frames)}items"]
        if not self.candidate_frames:
            parts.append("Next: Add candidates in player")
        elif self.preview_state_code == "ready":
            parts.append("Preview updated")
        elif self.preview_state_code == "dirty":
            parts.append("Waiting for preview update")
        else:
            parts.append("Update preview after assignment")
        if self.preview_sprites:
            parts.append("Live Test Available")
        self.workflow_state_var.set(" / ".join(parts))

    def _on_assignment_var_changed(self, *_args):
        if self._suspend_preview_traces:
            return
        self._invalidate_preview(dirty=True)

    def _on_crop_or_feather_changed(self, *_args):
        if self._suspend_preview_traces:
            return
        self._invalidate_preview(dirty=True)

    def _build_player_window(self):
        """Build player UI in separate window"""
        self.player_focus_var = tk.BooleanVar(value=True)
        self.player_seek_var = tk.DoubleVar(value=0.0)
        self.player_speed_var = tk.StringVar(value="1.0x")
        self.player_frame_var = tk.StringVar(value="0")

        win = tk.Toplevel(self)
        win.title("Sprite Picker Player")
        win.resizable(True, True)
        win.geometry("900x760")
        win.minsize(760, 620)
        win.protocol("WM_DELETE_WINDOW", self._hide_player_window)
        self.player_window = win

        frame = ttk.Frame(win, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        self.player_viewport = ttk.Frame(frame, height=PLAYER_PREVIEW_MAX_H + 12)
        self.player_viewport.pack(fill=tk.BOTH, expand=True)
        self.player_viewport.pack_propagate(False)

        self.player_view = ttk.Label(
            self.player_viewport,
            text="Please select a video",
            anchor=tk.CENTER,
            relief=tk.SUNKEN,
        )
        self.player_view.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            frame,
            text=(
                "Usage: Find good frames by play/stop or step, "
                "Register with 'Add/Overwrite Current Frame to Candidates'."
            ),
            font=("", 9),
            wraplength=820,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(8, 4))

        info = ttk.Frame(frame)
        info.pack(fill=tk.X, pady=(8, 2))
        ttk.Label(info, textvariable=self.player_status_var, font=("", 9)).pack(side=tk.LEFT)
        ttk.Checkbutton(
            info,
            text="Zoom Mouth",
            variable=self.player_focus_var,
            command=lambda: self._show_player_frame(self.player_current_frame_idx),
        ).pack(side=tk.RIGHT)

        self.player_seek = ttk.Scale(
            frame,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            variable=self.player_seek_var,
            command=self._on_player_seek,
            state=tk.DISABLED,
        )
        self.player_seek.pack(fill=tk.X, pady=(0, 8))

        controls = ttk.Frame(frame)
        controls.pack(fill=tk.X)

        self.prev_frame_btn = ttk.Button(
            controls, text="◀ 1F", command=lambda: self._step_player(-1), state=tk.DISABLED
        )
        self.prev_frame_btn.pack(side=tk.LEFT)

        self.play_btn = ttk.Button(
            controls, text="Play", command=self._toggle_player, state=tk.DISABLED
        )
        self.play_btn.pack(side=tk.LEFT, padx=4)

        self.next_frame_btn = ttk.Button(
            controls, text="1F ▶", command=lambda: self._step_player(1), state=tk.DISABLED
        )
        self.next_frame_btn.pack(side=tk.LEFT)

        self.add_candidate_btn = ttk.Button(
            controls,
            text="Add/Overwrite Current Frame to Candidates",
            command=self._on_add_current_frame,
            state=tk.DISABLED,
        )
        self.add_candidate_btn.pack(side=tk.LEFT, padx=(12, 4))

        self.remove_candidate_btn = ttk.Button(
            controls,
            text="Remove Selected Candidate",
            command=self._on_remove_selected_candidate,
            state=tk.DISABLED,
        )
        self.remove_candidate_btn.pack(side=tk.LEFT, padx=4)

        self.clear_candidates_btn = ttk.Button(
            controls,
            text="Clear Candidates",
            command=self._on_clear_candidates,
            state=tk.DISABLED,
        )
        self.clear_candidates_btn.pack(side=tk.LEFT, padx=4)

        extras = ttk.Frame(frame)
        extras.pack(fill=tk.X, pady=(8, 0))

        ttk.Label(extras, text="Speed").pack(side=tk.LEFT)
        self.player_speed_combo = ttk.Combobox(
            extras,
            width=8,
            state="readonly",
            textvariable=self.player_speed_var,
            values=["0.25x", "0.5x", "1.0x", "1.5x", "2.0x", "4.0x"],
        )
        self.player_speed_combo.pack(side=tk.LEFT, padx=(4, 12))
        self.player_speed_combo.bind("<<ComboboxSelected>>", lambda _e=None: self._on_player_speed_change())

        ttk.Label(extras, text="Frame").pack(side=tk.LEFT)
        self.player_frame_entry = ttk.Entry(
            extras,
            width=8,
            textvariable=self.player_frame_var,
        )
        self.player_frame_entry.pack(side=tk.LEFT, padx=(4, 4))
        self.player_frame_entry.bind("<Return>", lambda _e=None: self._jump_to_frame())

        self.jump_frame_btn = ttk.Button(
            extras,
            text="Jump",
            command=self._jump_to_frame,
            state=tk.DISABLED,
        )
        self.jump_frame_btn.pack(side=tk.LEFT)

        hint = ttk.Label(
            frame,
            text="Shortcuts: Space=Play/Stop  J/K/Left/Right=Step  Double-click candidate=Register current frame",
            font=("", 9),
        )
        hint.pack(anchor=tk.W, pady=(8, 0))

        win.bind("<KeyPress>", self._on_key_press)

    def _hide_player_window(self):
        if self.player_window and self.player_window.winfo_exists():
            self.player_window.withdraw()

    def _show_player_window(self):
        if self.player_window is None or not self.player_window.winfo_exists():
            self._build_player_window()
        assert self.player_window is not None
        self.player_window.deiconify()
        self.player_window.lift()
        try:
            self.player_window.focus_force()
        except Exception:
            pass

    def _build_live_test_window(self):
        """Build simple live test window for mouth materials"""
        self.live_audio_device_var = tk.StringVar(value="")
        self.live_status_var = tk.StringVar(value="Stopped")
        self.live_level_var = tk.DoubleVar(value=0.0)
        self.live_shape_var = tk.StringVar(value="closed")

        win = tk.Toplevel(self)
        win.title("Mouth Live Test")
        win.resizable(True, True)
        win.geometry("560x720")
        win.minsize(420, 520)
        win.protocol("WM_DELETE_WINDOW", self._hide_live_test_window)
        self.live_test_window = win

        frame = ttk.Frame(win, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            frame,
            text=(
                "In this test, using the updated preview mouth materials\n"
                "switch open / closed / half / e / u with mic input."
            ),
            font=("", 9),
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 8))

        top = ttk.Frame(frame)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Audio Input").pack(side=tk.LEFT)
        self.live_audio_combo = ttk.Combobox(
            top,
            state="readonly",
            textvariable=self.live_audio_device_var,
            width=42,
        )
        self.live_audio_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 6))
        self.live_audio_combo.bind("<<ComboboxSelected>>", lambda _e=None: None)

        self.live_audio_refresh_btn = ttk.Button(
            top,
            text="Reload",
            command=self._refresh_live_audio_devices,
        )
        self.live_audio_refresh_btn.pack(side=tk.LEFT)

        ctrl = ttk.Frame(frame)
        ctrl.pack(fill=tk.X, pady=(8, 8))

        self.live_start_btn = ttk.Button(
            ctrl,
            text="Start Test",
            command=self._toggle_live_test,
            state=tk.DISABLED,
        )
        self.live_start_btn.pack(side=tk.LEFT)

        ttk.Label(ctrl, textvariable=self.live_status_var).pack(side=tk.LEFT, padx=(10, 10))
        ttk.Label(ctrl, text="mouth").pack(side=tk.LEFT)
        ttk.Label(ctrl, textvariable=self.live_shape_var, font=("", 10, "bold")).pack(side=tk.LEFT, padx=(4, 0))

        meter_frame = ttk.Frame(frame)
        meter_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(meter_frame, text="Input Level").pack(side=tk.LEFT)
        self.live_level_bar = ttk.Progressbar(
            meter_frame,
            variable=self.live_level_var,
            maximum=1.0,
            mode="determinate",
        )
        self.live_level_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))

        self.live_preview_label = ttk.Label(
            frame,
            text="Please update preview on main screen first to prepare mouth materials",
            anchor=tk.CENTER,
            relief=tk.SUNKEN,
        )
        self.live_preview_label.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            frame,
            text="If you update the preview on the main screen first, you can verify it here immediately.",
            font=("", 9),
        ).pack(anchor=tk.W, pady=(8, 0))

        if not _HAS_SOUNDDEVICE:
            self.live_status_var.set("Not available (sounddevice missing)")
        self._refresh_live_audio_devices()
        self._hide_live_test_window()

    def _hide_live_test_window(self):
        self._stop_live_test()
        if self.live_test_window and self.live_test_window.winfo_exists():
            self.live_test_window.withdraw()

    def _show_live_test_window(self):
        if self.live_test_window is None or not self.live_test_window.winfo_exists():
            self._build_live_test_window()
        assert self.live_test_window is not None
        self.live_test_window.deiconify()
        self.live_test_window.lift()
        self._refresh_live_audio_devices()
        self._update_live_test_preview(self._resolve_live_test_shape())
        try:
            self.live_test_window.focus_force()
        except Exception:
            pass

    def _refresh_live_audio_devices(self):
        """Update audio input list for live test"""
        self._live_audio_items = list_input_devices() if _HAS_SOUNDDEVICE else []
        values = [str(item.get("display", "")) for item in self._live_audio_items]
        self.live_audio_combo["values"] = values
        if values:
            current = self.live_audio_device_var.get()
            if current not in values:
                self.live_audio_device_var.set(values[0])
        else:
            self.live_audio_device_var.set("")
        self._refresh_live_test_button_state()

    def _refresh_live_test_button_state(self):
        has_sprites = bool(self.preview_sprites)
        has_audio = bool(self._live_audio_items) and _HAS_SOUNDDEVICE
        can_start = has_sprites and has_audio
        self.live_test_btn.configure(state=(tk.NORMAL if has_sprites else tk.DISABLED))
        self.live_start_btn.configure(state=(tk.NORMAL if can_start else tk.DISABLED))
        if not has_sprites:
            self.live_status_var.set("Preview update is required first")
        elif not has_audio:
            self.live_status_var.set("Audio input device not found")
        elif self._live_stream is None:
            self.live_status_var.set("Stopped")
        self._refresh_workflow_state()

    def _selected_live_device_item(self) -> Optional[dict]:
        cur = self.live_audio_device_var.get().strip()
        if not cur:
            return None
        for item in self._live_audio_items:
            if str(item.get("display", "")).strip() == cur:
                return item
        return None

    def _resolve_live_test_shape(self) -> str:
        if self._live_audio_state is not None:
            return str(self._live_audio_state.get("mouth_shape_now", "closed"))
        return "closed"

    def _resolve_live_test_sprites(self) -> Optional[Dict[str, np.ndarray]]:
        if not self.preview_sprites:
            return None
        base = dict(self.preview_sprites)
        closed = base.get("closed")
        open_ = first_non_none(base.get("open"), closed)
        half = first_non_none(base.get("half"), open_, closed)
        e_img = first_non_none(base.get("e"), open_, half, closed)
        u_img = first_non_none(base.get("u"), open_, half, closed)
        closed = first_non_none(closed, half, open_, e_img, u_img)
        if closed is None or open_ is None:
            return None
        return {
            "open": open_,
            "closed": closed,
            "half": half if half is not None else closed,
            "e": e_img if e_img is not None else open_,
            "u": u_img if u_img is not None else open_,
        }

    def _update_live_test_preview(self, mouth_shape: str):
        sprites = self._resolve_live_test_sprites()
        if sprites is None:
            self.live_preview_label.configure(
                image="",
                text="Please update preview on main screen first to prepare mouth materials",
            )
            self.live_shape_var.set("-")
            self._refresh_live_test_button_state()
            return

        sprite = sprites.get(mouth_shape, sprites["closed"])
        composited = composite_on_checkerboard(sprite)
        photo = numpy_to_photoimage_fit(composited, 420, 420, color_order="RGB")
        self.live_preview_image = photo
        if photo:
            self.live_preview_label.configure(image=photo, text="")
        else:
            self.live_preview_label.configure(image="", text=mouth_shape)
        self.live_shape_var.set(mouth_shape)
        self._refresh_live_test_button_state()

    def _create_live_audio_state(self) -> dict:
        return {
            "noise": 1e-4,
            "peak": 1e-3,
            "peak_decay": 0.995,
            "silence_gate_rms": 0.002,
            "rms_smooth_q": deque(maxlen=3),
            "env_lp": 0.0,
            "env_hist": deque(maxlen=1000),
            "cent_hist": deque(maxlen=1000),
            "TALK_TH": 0.06,
            "HALF_TH": 0.30,
            "OPEN_TH": 0.52,
            "U_TH": 0.16,
            "E_TH": 0.20,
            "current_open_shape": "open",
            "last_vowel_change_t": -999.0,
            "e_prev2": 0.0,
            "e_prev1": 0.0,
            "mouth_shape_now": "closed",
        }

    def _toggle_live_test(self):
        if self._live_stream is not None:
            self._stop_live_test()
        else:
            self._start_live_test()

    def _start_live_test(self):
        sprites = self._resolve_live_test_sprites()
        if sprites is None:
            messagebox.showwarning("Warning", "Please update preview on main screen before starting live test.")
            return
        if not _HAS_SOUNDDEVICE:
            messagebox.showerror("Error", "sounddevice is required")
            return
        device_item = self._selected_live_device_item()
        if device_item is None:
            messagebox.showwarning("Warning", "Please select an audio input device")
            return

        raw_spec = str(device_item.get("spec", "")).strip()
        resolution: dict | None = None
        apply_state: dict | None = None
        device_idx: int | None = None
        samplerate = 0

        def _resolve_live_input(*, prefer_default_source: bool = False) -> tuple[dict, dict, int, int]:
            nonlocal resolution, apply_state
            cleanup_audio_device_resolution(resolution or {}, apply_state)
            resolution = resolve_audio_device_spec(
                raw_spec,
                sd,  # type: ignore[arg-type]
                fallback_index=device_item.get("index"),
                prefer_default_source=prefer_default_source,
            )
            resolved_index = resolution.get("resolved_index")
            if resolved_index is None:
                raise RuntimeError(
                    f"Could not resolve input device: {device_item.get('display', '')}"
                )
            apply_state = apply_audio_resolution_for_current_process(resolution)
            dev = sd.query_devices(resolved_index, "input")  # type: ignore[union-attr]
            return resolution, apply_state, int(resolved_index), int(dev["default_samplerate"])

        try:
            resolution, apply_state, device_idx, samplerate = _resolve_live_input(
                prefer_default_source=False,
            )
        except Exception as e:
            if raw_spec.startswith("pa:"):
                try:
                    self.log(f"[audio] Live test primary resolution failed. Trying default-source fallback: {e}")
                    resolution, apply_state, device_idx, samplerate = _resolve_live_input(
                        prefer_default_source=True,
                    )
                except Exception as fallback_error:
                    cleanup_audio_device_resolution(resolution or {}, apply_state)
                    messagebox.showerror("Error", f"Could not open input device: {fallback_error}")
                    return
            else:
                cleanup_audio_device_resolution(resolution or {}, apply_state)
                messagebox.showerror("Error", f"Could not open input device: {e}")
                return

        while not self._live_feat_q.empty():
            try:
                self._live_feat_q.get_nowait()
            except queue.Empty:
                break

        self._live_audio_state = self._create_live_audio_state()
        audio_hz = 100
        hop = max(int(samplerate / audio_hz), 256)
        window = np.hanning(hop).astype(np.float32)
        freqs = np.fft.rfftfreq(hop, d=1.0 / samplerate)

        def audio_cb(indata, frames, time_info, status):
            x = indata.astype(np.float32)
            if x.ndim == 2:
                x = x.mean(axis=1)
            if len(x) < hop:
                x = np.pad(x, (0, hop - len(x)))
            elif len(x) > hop:
                x = x[:hop]
            rms_raw = float(np.sqrt(np.mean(x * x) + 1e-12))
            mag = np.abs(np.fft.rfft(x * window)) + 1e-9
            centroid = float((freqs * mag).sum() / mag.sum())
            centroid = float(np.clip(centroid / (samplerate * 0.5), 0.0, 1.0))
            try:
                self._live_feat_q.put_nowait((rms_raw, centroid))
            except queue.Full:
                pass

        try:
            self._live_stream = sd.InputStream(  # type: ignore[union-attr]
                samplerate=samplerate,
                channels=1,
                blocksize=hop,
                dtype="float32",
                callback=audio_cb,
                device=device_idx,
                latency="low",
            )
            self._live_stream.start()
        except Exception as e:
            if raw_spec.startswith("pa:") and not bool((resolution or {}).get("needs_default_source_switch")):
                try:
                    self.log(f"[audio] Live test stream open failed. Trying default-source fallback: {e}")
                    resolution, apply_state, device_idx, samplerate = _resolve_live_input(
                        prefer_default_source=True,
                    )
                    self._live_stream = sd.InputStream(  # type: ignore[union-attr]
                        samplerate=samplerate,
                        channels=1,
                        blocksize=hop,
                        dtype="float32",
                        callback=audio_cb,
                        device=device_idx,
                        latency="low",
                    )
                    self._live_stream.start()
                except Exception as fallback_error:
                    cleanup_audio_device_resolution(resolution or {}, apply_state)
                    self._live_stream = None
                    messagebox.showerror("Error", f"Failed to start live test: {fallback_error}")
                    return
            else:
                cleanup_audio_device_resolution(resolution or {}, apply_state)
                self._live_stream = None
                messagebox.showerror("Error", f"Failed to start live test: {e}")
                return

        self._live_audio_resolution = resolution
        self._live_audio_apply_state = apply_state
        self.live_status_var.set(f"Running: {self.live_audio_device_var.get()}")
        self.live_start_btn.configure(text="Stop")
        self._refresh_live_test_button_state()
        self._schedule_live_test_tick()
        self.log(f"Live test started: device={device_idx}")

    def _stop_live_test(self):
        if self._live_job:
            try:
                self.after_cancel(self._live_job)
            except Exception:
                pass
            self._live_job = None
        if self._live_stream is not None:
            try:
                self._live_stream.stop()
            except Exception:
                pass
            try:
                self._live_stream.close()
            except Exception:
                pass
            self._live_stream = None
        cleanup_audio_device_resolution(
            self._live_audio_resolution or {},
            self._live_audio_apply_state,
        )
        self._live_audio_resolution = None
        self._live_audio_apply_state = None
        self.log("Stop live test")
        self.live_level_var.set(0.0)
        self.live_start_btn.configure(text="Start Test")
        self._live_audio_state = None
        self._update_live_test_preview("closed")
        self._refresh_live_test_button_state()

    def _schedule_live_test_tick(self):
        self._poll_live_test_audio()
        if self._live_stream is not None:
            self._live_job = self.after(30, self._schedule_live_test_tick)

    def _poll_live_test_audio(self):
        if self._live_audio_state is None:
            return
        state = self._live_audio_state
        changed = False

        while True:
            try:
                rms_raw, cent = self._live_feat_q.get_nowait()
            except queue.Empty:
                break

            if rms_raw < state["noise"] + 0.0005:
                state["noise"] = 0.99 * state["noise"] + 0.01 * rms_raw
            else:
                state["noise"] = 0.999 * state["noise"] + 0.001 * rms_raw

            state["peak"] = max(rms_raw, state["peak"] * state["peak_decay"], state["noise"] + state["silence_gate_rms"])
            denom = max(state["peak"] - state["noise"], state["silence_gate_rms"])
            rms_norm = float(np.clip((rms_raw - state["noise"]) / denom, 0.0, 1.0) ** 0.5)
            if rms_raw < state["noise"] + state["silence_gate_rms"]:
                rms_norm = 0.0

            state["rms_smooth_q"].append(rms_norm)
            rms_sm = float(np.mean(state["rms_smooth_q"]))
            beta = one_pole_beta(8.0, 100)
            state["env_lp"] = state["env_lp"] + beta * (rms_sm - state["env_lp"])
            env = float(np.clip(0.75 * state["env_lp"] + 0.25 * rms_sm, 0.0, 1.0))
            self.live_level_var.set(env)

            state["env_hist"].append(env)
            state["cent_hist"].append(float(cent))

            if len(state["env_hist"]) > 300 and (len(state["env_hist"]) % 100 == 0):
                vals = np.array(state["env_hist"], dtype=np.float32)
                k = max(1, int(0.2 * len(vals)))
                noise_floor_env = float(np.median(np.sort(vals)[:k]))
                state["TALK_TH"] = float(np.clip(noise_floor_env + 0.05, 0.03, 0.18))
                talk_vals = vals[vals > state["TALK_TH"]]
                if len(talk_vals) > 20:
                    state["HALF_TH"] = max(float(np.percentile(talk_vals, 25)), state["TALK_TH"] + 0.02)
                    state["OPEN_TH"] = max(float(np.percentile(talk_vals, 58)), state["HALF_TH"] + 0.05)
                    cents = np.array(state["cent_hist"], dtype=np.float32)
                    open_mask = vals >= state["OPEN_TH"]
                    cent_open = cents[open_mask] if open_mask.sum() > 20 else cents[vals > state["TALK_TH"]]
                    if len(cent_open) > 20:
                        state["U_TH"] = float(np.percentile(cent_open, 20))
                        state["E_TH"] = float(np.percentile(cent_open, 80))

            if env < state["HALF_TH"]:
                mouth_level = "closed"
            elif env < state["OPEN_TH"]:
                mouth_level = "half"
            else:
                mouth_level = "open"

            now_t = time.perf_counter()
            if mouth_level == "open":
                is_peak = (
                    state["e_prev2"] < state["e_prev1"]
                    and state["e_prev1"] >= env
                    and state["e_prev1"] > state["OPEN_TH"] + 0.02
                )
                if is_peak and (now_t - state["last_vowel_change_t"]) >= 0.12:
                    if len(state["cent_hist"]) >= 5:
                        cm = float(np.mean(list(state["cent_hist"])[-5:]))
                    else:
                        cm = float(cent)
                    if cm < state["U_TH"]:
                        state["current_open_shape"] = "u"
                    elif cm > state["E_TH"]:
                        state["current_open_shape"] = "e"
                    else:
                        state["current_open_shape"] = "open"
                    state["last_vowel_change_t"] = now_t

                new_shape = state["current_open_shape"]
            elif mouth_level == "half":
                new_shape = "half"
            else:
                new_shape = "closed"

            state["e_prev2"], state["e_prev1"] = state["e_prev1"], env
            if new_shape != state["mouth_shape_now"]:
                state["mouth_shape_now"] = new_shape
                changed = True

        if changed or self.live_preview_image is None:
            self._update_live_test_preview(self._resolve_live_test_shape())

    def _update_candidate_slot_highlight(self):
        """Update selected candidate slot display"""
        self.selected_slot_var.set(f"Selected Candidate: {self.selected_candidate_idx + 1}")
        for i in range(CANDIDATE_COUNT):
            selected = (i == self.selected_candidate_idx)
            relief = tk.SOLID if selected else tk.SUNKEN
            self.cand_labels[i].configure(relief=relief)
            title = f"[Candidate{i+1}]" if selected else f"Candidate{i+1}"
            self.cand_slot_titles[i].configure(text=title)

    def _select_candidate_slot(self, idx: int):
        """Select candidate slot"""
        idx = max(0, min(int(idx), CANDIDATE_COUNT - 1))
        self.selected_candidate_idx = idx
        self._update_candidate_slot_highlight()

    def _focus_candidate_slot(self, idx: int):
        """Select candidate slot and focus"""
        self._select_candidate_slot(idx)
        if 0 <= idx < len(self.cand_entries):
            try:
                self.cand_entries[idx].focus_set()
                self.cand_entries[idx].selection_range(0, tk.END)
            except Exception:
                pass

    def _on_candidate_double_click(self, idx: int):
        """Register current frame on double-clicking candidate slot"""
        self._select_candidate_slot(idx)
        if self.add_candidate_btn.cget("state") == tk.NORMAL:
            self._on_add_current_frame()

    def _set_player_enabled(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.player_seek.configure(state=state)
        self.prev_frame_btn.configure(state=state)
        self.play_btn.configure(state=state)
        self.next_frame_btn.configure(state=state)
        self.player_speed_combo.configure(state=("readonly" if enabled else tk.DISABLED))
        self.player_frame_entry.configure(state=state)
        self.jump_frame_btn.configure(state=state)

    def _stop_player(self):
        """Stop player playback"""
        self.player_playing = False
        if self.player_job:
            try:
                self.after_cancel(self.player_job)
            except Exception:
                pass
            self.player_job = None
        self.play_btn.configure(text="Play")

    def _close_player_capture(self):
        if self._player_cap:
            self._player_cap.release()
            self._player_cap = None

    def _open_player_capture(self) -> Optional[cv2.VideoCapture]:
        if not self.video_path:
            return None
        if self._player_cap is None or not self._player_cap.isOpened():
            self._player_cap = cv2.VideoCapture(self.video_path)
        return self._player_cap

    def _load_player_metadata(self):
        """Load video info for player"""
        self.player_total_frames = 0
        self.player_fps = 30.0
        if not self.video_path:
            self._set_player_enabled(False)
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.player_view.configure(text="Cannot open video", image="")
            self._set_player_enabled(False)
            return

        try:
            self.player_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.player_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        finally:
            cap.release()

        if self.player_total_frames > 0:
            self.player_seek.configure(to=max(0, self.player_total_frames - 1))
            self._set_player_enabled(True)
        else:
            self._set_player_enabled(False)

    def _show_player_frame(self, frame_idx: int):
        """Show specified frame in player"""
        if not self.video_path or self.player_total_frames <= 0:
            return

        cap = self._open_player_capture()
        if cap is None or not cap.isOpened():
            return

        frame_idx = max(0, min(int(frame_idx), self.player_total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            self.log(f"Warning: Frame read failure F:{frame_idx}")
            return

        self.player_current_frame_idx = frame_idx
        mf = self._mouth_frame_by_idx.get(frame_idx)
        display = draw_mouth_quad_overlay(frame, mf)
        if self.player_focus_var.get():
            display = crop_frame_around_mouth(display, mf, margin_scale=4.0)
        photo = numpy_to_photoimage_fit(
            display,
            PLAYER_PREVIEW_MAX_W,
            PLAYER_PREVIEW_MAX_H,
        )
        self.player_image = photo
        if photo:
            self.player_view.configure(image=photo, text="")
        else:
            self.player_view.configure(image="", text=f"F:{frame_idx}")

        valid_text = ""
        if mf is not None:
            valid_text = f"  valid={int(mf.valid)} conf={mf.confidence:.2f}"
        self.player_status_var.set(
            f"F:{frame_idx} / {max(0, self.player_total_frames - 1)}{valid_text}"
        )
        self._updating_player_scale = True
        try:
            self.player_seek_var.set(float(frame_idx))
            self.player_frame_var.set(str(frame_idx))
        finally:
            self._updating_player_scale = False

    def _player_speed_value(self) -> float:
        label = str(self.player_speed_var.get() or "1.0x").strip().lower()
        if label.endswith("x"):
            label = label[:-1]
        try:
            return max(0.05, float(label))
        except Exception:
            return 1.0

    def _on_player_speed_change(self):
        """Update display after speed change"""
        if self.player_playing:
            self._stop_player()
            self.player_playing = True
            self.play_btn.configure(text="Stop")
            self._schedule_player_tick()

    def _on_player_seek(self, value: str):
        if self._updating_player_scale or self._ignore_next_seek or self.player_total_frames <= 0:
            return
        self._stop_player()
        try:
            idx = int(float(value))
        except Exception:
            return
        if idx != self.player_current_frame_idx:
            self._show_player_frame(idx)

    def _step_player(self, delta: int):
        """Step by 1 frame"""
        if self.player_total_frames <= 0:
            return
        self._stop_player()
        self._show_player_frame(self.player_current_frame_idx + int(delta))

    def _schedule_player_tick(self):
        if not self.player_playing:
            return

        next_idx = self.player_current_frame_idx + 1
        if next_idx >= self.player_total_frames:
            self._stop_player()
            return

        self._show_player_frame(next_idx)
        delay_ms = max(15, int(1000.0 / max(1.0, self.player_fps * self._player_speed_value())))
        self.player_job = self.after(delay_ms, self._schedule_player_tick)

    def _toggle_player(self):
        """Toggle Play/Stop"""
        if self.player_total_frames <= 0:
            return
        if self.player_playing:
            self._stop_player()
            return

        self.player_playing = True
        self.play_btn.configure(text="Stop")
        self._schedule_player_tick()

    def _jump_to_frame(self):
        """Jump by typing frame number directly"""
        if self.player_total_frames <= 0:
            return
        self._stop_player()
        try:
            idx = int(str(self.player_frame_var.get()).strip())
        except Exception:
            messagebox.showwarning("Warning", "Please enter an integer for the frame number")
            self.player_frame_var.set(str(self.player_current_frame_idx))
            return
        idx = max(0, min(idx, self.player_total_frames - 1))
        self._show_player_frame(idx)

    def _on_key_press(self, event):
        """Player shortcuts"""
        if self.player_total_frames <= 0:
            return

        focus = self.focus_get()
        focus_in_text_entry = isinstance(focus, (tk.Entry, ttk.Entry, ttk.Combobox))
        keysym = str(event.keysym or "").lower()

        if keysym in ("j",):
            self._step_player(-1)
            return
        if keysym in ("k",):
            self._step_player(1)
            return
        if keysym == "space" and not focus_in_text_entry:
            self._toggle_player()
            return
        if focus_in_text_entry:
            return
        if keysym == "left":
            self._step_player(-1)
        elif keysym == "right":
            self._step_player(1)

    def _enable_manual_pick_controls(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.add_candidate_btn.configure(state=state)
        self.auto_fill_btn.configure(state=state)

    def _refresh_candidate_buttons(self):
        has_candidates = len(self.candidate_frames) > 0
        state = tk.NORMAL if has_candidates else tk.DISABLED
        self.update_btn.configure(state=state)
        self.auto_crop_btn.configure(state=state)
        self.auto_assign_btn.configure(state=state)
        self.remove_candidate_btn.configure(state=state)
        self.clear_candidates_btn.configure(state=state)
        if not has_candidates:
            self.output_btn.configure(state=tk.DISABLED)
            self.live_test_btn.configure(state=tk.DISABLED)
            if self.preview_state_code != "empty":
                self._set_preview_state_code("empty")
        self._refresh_workflow_state()

    def _refresh_candidates_ui(self):
        """Regenerate and display candidate thumbnails"""
        self._thumbnail_patches = self._generate_thumbnail_patches()
        self._create_thumbnail_images()
        self._update_candidates_ui()
        self._refresh_candidate_buttons()

    def _on_add_current_frame(self):
        """Add/overwrite current frame to candidates"""
        if not self.extractor or not self.valid_frames:
            messagebox.showwarning("Warning", "Please run analysis first")
            return

        mf = self._mouth_frame_by_idx.get(self.player_current_frame_idx)
        if mf is None or not mf.valid:
            messagebox.showwarning(
                "Warning",
                "Current frame does not have a valid mouth track. Please select another frame.",
            )
            return

        target_idx = min(self.selected_candidate_idx, len(self.candidate_frames))
        if target_idx >= CANDIDATE_COUNT:
            messagebox.showwarning("Warning", f"Candidates max is {CANDIDATE_COUNT} frames")
            return

        if target_idx < len(self.candidate_frames):
            self.candidate_frames[target_idx] = mf
            self.cand_vars[target_idx].set(MOUTH_ASSIGNMENT_OPTIONS[0])
            self.log(f"Overwrote candidate: Candidate {target_idx+1} <- F:{mf.frame_idx}")
        else:
            self.candidate_frames.append(mf)
            target_idx = len(self.candidate_frames) - 1
            self.log(f"Added candidate: Candidate {target_idx+1} <- F:{mf.frame_idx}")

        next_idx = min(target_idx + 1, CANDIDATE_COUNT - 1)
        self._focus_candidate_slot(next_idx)
        self._invalidate_preview(dirty=True)
        self._refresh_candidates_ui()

    def _on_remove_selected_candidate(self):
        """Remove Selected Candidate"""
        if not self.candidate_frames:
            return
        idx = min(self.selected_candidate_idx, len(self.candidate_frames) - 1)
        removed = self.candidate_frames.pop(idx)
        self._clear_assignments()
        self.log(f"Removed candidate: Candidate {idx+1} / F:{removed.frame_idx}")
        self._select_candidate_slot(max(0, min(idx, len(self.candidate_frames))))
        self._invalidate_preview(dirty=True)
        self._refresh_candidates_ui()

    def _on_clear_candidates(self):
        """Clear all candidates"""
        self.candidate_frames = []
        self._clear_assignments()
        self._clear_candidates()
        self._clear_preview("empty")
        self._refresh_candidate_buttons()
        self.log("Candidates cleared")
    
    def _on_select_video(self):
        """Select Video File"""
        if sys.platform == "darwin":  # Mac
            path = filedialog.askopenfilename(title="Select Video File")
        else:  # Windows/Linux
            path = filedialog.askopenfilename(
                title="Select Video File",
                filetypes=[
                    ("Video File", "*.mp4 *.avi *.mov *.mkv *.webm"),
                    ("All Files", "*.*"),
                ],
            )
        if path:
            self._set_video(path)
    
    def _on_drop_video(self, event):
        """Set video by drag & drop"""
        path = event.data
        if path.startswith("{") and path.endswith("}"):
            path = path[1:-1]
        
        if os.path.isfile(path):
            self._set_video(path)
    
    def _set_video(self, path: str):
        """Set video path"""
        self._stop_player()
        self.video_path = path
        self.video_var.set(path)
        self.extractor = None
        self.valid_frames = []
        self._mouth_frame_by_idx = {}
        self.candidate_frames = []
        self._clear_candidates()
        self._clear_preview("empty")
        self.update_btn.configure(state=tk.DISABLED)
        self.output_btn.configure(state=tk.DISABLED)
        self.auto_crop_btn.configure(state=tk.DISABLED)
        self.auto_fill_btn.configure(state=tk.DISABLED)
        self.auto_assign_btn.configure(state=tk.DISABLED)
        self._enable_manual_pick_controls(False)
        self._refresh_candidate_buttons()
        self._refresh_workflow_state()

        # Close cached capture
        if self._cached_cap:
            self._cached_cap.release()
            self._cached_cap = None
        self._close_player_capture()

        self.player_current_frame_idx = 0
        self._load_player_metadata()
        if self.player_total_frames > 0:
            self._show_player_frame(0)
        else:
            self.player_view.configure(text="Cannot open video", image="")
        
        self.log(f"Video selected: {os.path.basename(path)}")
        if self.player_total_frames > 0:
            self.log("Automatically starting analysis after video selection...")
            self.after(50, self._on_analyze)
    
    def _clear_candidates(self):
        """Clear candidates display"""
        self.candidate_images = []
        self._suspend_preview_traces = True
        try:
            for i in range(CANDIDATE_COUNT):
                self.cand_labels[i].configure(image="", text="")
                self.cand_frame_labels[i].configure(text="")
                self.cand_vars[i].set(MOUTH_ASSIGNMENT_OPTIONS[0])
        finally:
            self._suspend_preview_traces = False
        self._update_candidate_slot_highlight()
    
    def _clear_preview(self, state_code: str = "empty"):
        """Clear preview"""
        self._stop_live_test()
        self.preview_images = {}
        self.preview_sprites = {}
        for name, label in self.preview_labels.items():
            label.configure(image="", text="(Unselected)")
            self.out_frame_labels[name].configure(text="")
        self.output_btn.configure(state=tk.DISABLED)
        self._update_live_test_preview("closed")
        self._set_preview_state_code(state_code)

    def _invalidate_preview(self, dirty: bool):
        next_state = "dirty" if dirty and self.preview_state_code in ("ready", "dirty") else "empty"
        self._clear_preview(next_state)
    
    def _on_analyze(self):
        """Run Analysis"""
        if not self.video_path:
            messagebox.showwarning("Warning", "Please select a video file")
            return
        
        if self.is_analyzing:
            return
        
        self.is_analyzing = True
        self._refresh_workflow_state()
        self._start_busy_state("Analyzing", "Analyzing mouth positions. Please wait...")
        self.analyze_btn.configure(state=tk.DISABLED)
        self.update_btn.configure(state=tk.DISABLED)
        self.output_btn.configure(state=tk.DISABLED)
        self.auto_crop_btn.configure(state=tk.DISABLED)
        self.auto_fill_btn.configure(state=tk.DISABLED)
        self.auto_assign_btn.configure(state=tk.DISABLED)
        self._enable_manual_pick_controls(False)
        self._clear_candidates()
        self._clear_preview("empty")
        
        thread = threading.Thread(target=self._analyze_worker, daemon=True)
        thread.start()
    
    def _analyze_worker(self):
        """Analysis worker thread"""
        try:
            self.log("Starting analysis...")
            
            self.extractor = MouthSpriteExtractor(self.video_path)
            self.extractor.analyze(callback=self.log)
             
            valid_frames = [mf for mf in self.extractor.mouth_frames if mf.valid]
            self.valid_frames = valid_frames
            self._mouth_frame_by_idx = {
                mf.frame_idx: mf for mf in self.extractor.mouth_frames
            }
             
            if len(valid_frames) == 0:
                self.log("Error: No valid frames")
                self.after(0, lambda: self._finish_busy_state("No valid frames found"))
                return
             
            # Calculate unified size (from all valid frames)
            if valid_frames:
                max_w = max(mf.width for mf in valid_frames)
                max_h = max(mf.height for mf in valid_frames)
                self.unified_size = (
                    ensure_even_ge2(int(max_w * 1.1)),
                    ensure_even_ge2(int(max_h * 1.1)),
                )
            
            self.log("Analysis complete. Please manually add candidate frames in the player.")
            self.log("If necessary, you can also use traditional auto-extraction via 'Auto-select Candidates'.")
            self.after(0, lambda: self._finish_busy_state("Analysis Complete"))
            self.after(0, lambda: self._enable_manual_pick_controls(True))
            self.after(0, lambda: self.auto_fill_btn.configure(state=tk.NORMAL))
            self.after(0, lambda: self._show_player_frame(self.player_current_frame_idx))
              
        except Exception as e:
            self.log(f"Error: {e}")
            self.after(0, lambda: self._finish_busy_state("Analysis error"))
            traceback.print_exc()
        
        finally:
            self.is_analyzing = False
            self.after(0, lambda: self.analyze_btn.configure(state=tk.NORMAL))
            self.after(0, self._refresh_workflow_state)
    
    def _get_video_capture(self) -> cv2.VideoCapture:
        """Get cached VideoCapture"""
        if self._cached_cap is None or not self._cached_cap.isOpened():
            self._cached_cap = cv2.VideoCapture(self.video_path)
        return self._cached_cap
    
    def _generate_thumbnail_patches(self) -> List[Optional[np.ndarray]]:
        """Generate numpy arrays for candidate frame thumbnails (for worker thread)"""
        if not self.candidate_frames or not self.unified_size:
            return []

        # Use independent VideoCapture in worker thread (thread-safe)
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return []

        unified_w, unified_h = self.unified_size
        patches = []

        try:
            for mf in self.candidate_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(mf.frame_idx))
                ok, frame = cap.read()
                if not ok or frame is None:
                    patches.append(None)
                    continue

                # Convert to normalized space (numpy array only)
                patch = warp_frame_to_norm(frame, mf.quad, unified_w, unified_h)
                patches.append(patch)
        finally:
            cap.release()

        return patches

    def _create_thumbnail_images(self):
        """Generate PhotoImage from numpy array (executed in UI thread)"""
        self.candidate_images = []
        patches = getattr(self, '_thumbnail_patches', [])

        for patch in patches:
            if patch is None:
                self.candidate_images.append(None)
            else:
                # PhotoImage is generated in UI thread
                photo = numpy_to_photoimage(patch, THUMB_SIZE)
                self.candidate_images.append(photo)

        # Release memory
        self._thumbnail_patches = []
    
    def _update_candidates_ui(self):
        """Update Candidates UI"""
        for i in range(CANDIDATE_COUNT):
            if i >= len(self.candidate_frames):
                self.cand_labels[i].configure(image="", text="")
                self.cand_frame_labels[i].configure(text="")
        for i, mf in enumerate(self.candidate_frames):
            if i < len(self.candidate_images) and self.candidate_images[i]:
                self.cand_labels[i].configure(image=self.candidate_images[i], text="")
            self.cand_frame_labels[i].configure(text=f"F:{mf.frame_idx}")
        self._update_candidate_slot_highlight()

    def _build_auto_candidates(self) -> List[MouthFrameInfo]:
        """Build auto candidates using traditional logic"""
        if not self.extractor:
            return []

        valid_frames = self.valid_frames
        if not valid_frames:
            return []

        heights = np.array([mf.height for mf in valid_frames])
        widths = np.array([mf.width for mf in valid_frames])
        aspect_ratios = widths / np.maximum(heights, 1e-6)

        selected_indices = set()
        candidates: List[Tuple[MouthFrameInfo, str]] = []

        def pick_by_score(scores, count, maximize=True, label=""):
            sorted_idx = np.argsort(scores)
            if maximize:
                sorted_idx = sorted_idx[::-1]

            picked = 0
            for idx in sorted_idx:
                if idx not in selected_indices and picked < count:
                    selected_indices.add(idx)
                    candidates.append((valid_frames[idx], label))
                    picked += 1
                if picked >= count:
                    break

        pick_by_score(heights, 2, maximize=True, label="open candidate")
        pick_by_score(heights, 2, maximize=False, label="closed candidate")

        median_h = np.median(heights)
        half_scores = -np.abs(heights - median_h)
        pick_by_score(half_scores, 2, maximize=True, label="half candidate")

        pick_by_score(aspect_ratios, 2, maximize=True, label="e candidate")
        pick_by_score(widths, 2, maximize=False, label="u candidate")

        preselected = {mf.frame_idx for mf, _ in candidates}
        opening_seq = pick_opening_sequence(
            self.extractor.mouth_frames,
            preselected=preselected,
            window=OPENING_SEQ_COUNT,
        )
        if opening_seq:
            candidates.extend((mf, "opening sequence") for mf in opening_seq)

        self.log(
            "Auto Candidate: open candidate x2, closed candidate x2, half candidate x2, "
            "e candidate x2, u candidate x2, Opening sequence "
            f"{len(opening_seq)} items"
        )
        return [mf for mf, _ in candidates][:CANDIDATE_COUNT]

    def _on_fill_auto_candidates(self):
        """Apply traditional auto candidate extraction to current candidates"""
        if not self.extractor or not self.valid_frames:
            messagebox.showwarning("Warning", "Please run analysis first")
            return
        self._clear_assignments()
        self.candidate_frames = self._build_auto_candidates()
        self._select_candidate_slot(0)
        self._clear_preview("empty")
        self._refresh_candidates_ui()
        self.log(f"Auto-select Candidates: {len(self.candidate_frames)} items set")

    def _on_auto_assign(self):
        """Execute auto-assignment for current candidates"""
        if not self.candidate_frames or not self.unified_size:
            messagebox.showwarning("Warning", "No candidates yet. Please add them in the player or use "Auto-select Candidates".")
            return

        self.log("Processing auto-assignment of mouth shapes...")
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open video")
            return
        try:
            self._run_auto_assign_internal(cap)
        finally:
            cap.release()
    
    def _on_update_preview(self):
        """Update preview"""
        if not self.candidate_frames or not self.unified_size:
            messagebox.showwarning("Warning", "No candidates yet. Please add them in the player or use "Auto-select Candidates".")
            return
        
        # Analyze assignments
        assignments: Dict[str, int] = {}
        for i, var in enumerate(self.cand_vars):
            val = var.get().strip()
            if val in ("", MOUTH_ASSIGNMENT_OPTIONS[0]):
                continue
            if i >= len(self.candidate_frames):
                self.log(f"Warning: Candidate slot does not exist {i+1}")
                continue
            if val in assignments:
                self.log(f"Warning: {val} is duplicated")
            assignments[val] = i
        
        if len(assignments) == 0:
            messagebox.showwarning("Warning", "No mouth shape assignments. Please choose open / closed / half / e / u.")
            return
        
        # Get parameters
        crop_top = self.crop_vars["top"].get()
        crop_bottom = self.crop_vars["bottom"].get()
        crop_left = self.crop_vars["left"].get()
        crop_right = self.crop_vars["right"].get()
        feather_px = self.feather_var.get()
        
        unified_w, unified_h = self.unified_size
        cap = self._get_video_capture()
        
        self._clear_preview("empty")
        
        for name, cand_idx in assignments.items():
            mf = self.candidate_frames[cand_idx]
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(mf.frame_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            
            # Extract sprites
            rgba = extract_sprite_with_crop(
                frame, mf.quad, unified_w, unified_h,
                crop_top, crop_bottom, crop_left, crop_right, feather_px
            )
            self.preview_sprites[name] = rgba
            
            # Preview display
            composited = composite_on_checkerboard(rgba)
            photo = numpy_to_photoimage(composited, PREVIEW_SIZE, color_order="RGB")
            if photo:
                self.preview_images[name] = photo
                self.preview_labels[name].configure(image=photo, text="")
                self.out_frame_labels[name].configure(text=f"F:{mf.frame_idx}")
        
        self.output_btn.configure(state=tk.NORMAL)
        self.live_test_btn.configure(state=tk.NORMAL)
        self._set_preview_state_code("ready")
        self._update_live_test_preview(self._resolve_live_test_shape())
        self.log(f"Update PreviewComplete ({len(self.preview_sprites)} frames)")

    def _run_auto_assign_internal(self, cap: cv2.VideoCapture):
        """
        Internal auto-assignment process (callable from worker thread)

        Args:
            cap: Open VideoCapture (managed by caller)
        """
        if not self.candidate_frames or not self.unified_size:
            return

        unified_w, unified_h = self.unified_size

        # Initialize feature analyzer and classifier
        analyzer = MouthFeatureAnalyzer((unified_w, unified_h))
        classifier = MouthAutoClassifier()

        # Calculate features for each candidate frame
        for mf in self.candidate_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(mf.frame_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            # Generate normalized patch
            patch = warp_frame_to_norm(frame, mf.quad, unified_w, unified_h)

            # Extract features
            features = analyzer.analyze_frame(patch)

            # Set features in MouthFrameInfo
            mf.inner_darkness = features.inner_darkness
            mf.opening_ratio = features.opening_ratio
            mf.horizontal_stretch = features.horizontal_stretch
            mf.vertical_compression = features.vertical_compression
            mf.lip_curvature = features.lip_curvature

        # Execute auto-classification
        selected = classifier.auto_select_5_types(self.candidate_frames)

        if not selected:
            self.log("Warning: Insufficient valid frames")
            return

        # Reflect selection results to UI
        # Create index map of candidate frames
        frame_to_cand_idx = {mf.frame_idx: i for i, mf in enumerate(self.candidate_frames)}

        # Clear assignments
        self.after(0, self._clear_assignments)

        for type_name, frame_idx in selected.items():
            if frame_idx in frame_to_cand_idx:
                cand_idx = frame_to_cand_idx[frame_idx]
                if type_name in MOUTH_SHAPES and cand_idx < len(self.cand_vars):
                    self.after(0, lambda idx=cand_idx, name=type_name: self.cand_vars[idx].set(name))
                    self.log(f"  {type_name} -> Candidate{cand_idx+1} (F:{frame_idx})")

        self.log("Auto-assignment complete")

    def _clear_assignments(self):
        """Clear assignment input"""
        self._suspend_preview_traces = True
        try:
            for var in self.cand_vars:
                var.set(MOUTH_ASSIGNMENT_OPTIONS[0])
        finally:
            self._suspend_preview_traces = False

    def _on_auto_crop(self):
        """Run auto-crop parameter optimization in background"""
        if self.is_analyzing:
            return
        if not self.candidate_frames or not self.unified_size:
            messagebox.showwarning("Warning", "Please run analysis first")
            return

        self.is_analyzing = True
        self._start_busy_state("Auto-cropping", "Estimating crop range from candidate frames")
        self.auto_crop_btn.configure(state=tk.DISABLED)
        self.log("Processing auto-crop adjustment...")

        thread = threading.Thread(target=self._auto_crop_worker, daemon=True)
        thread.start()

    def _auto_crop_worker(self):
        """Auto-crop worker thread"""
        cap = None
        try:
            if not self.candidate_frames or not self.unified_size:
                return

            # Use independent VideoCapture in worker thread (thread-safe)
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.log("Error: Could not open video")
                self.after(0, lambda: self._finish_busy_state("Auto-crop error"))
                return

            unified_w, unified_h = self.unified_size

            # Initialize auto-crop estimator
            estimator = AutoCropEstimator((unified_w, unified_h))

            # Collect patches of valid frames
            patches = []
            for mf in self.candidate_frames:
                if not mf.valid:
                    continue

                cap.set(cv2.CAP_PROP_POS_FRAMES, float(mf.frame_idx))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue

                # Generate normalized patch
                patch = warp_frame_to_norm(frame, mf.quad, unified_w, unified_h)
                patches.append(patch)

            if not patches:
                self.log("Warning: No valid patches")
                self.after(0, lambda: self._finish_busy_state("No targets for auto-cropping"))
                return

            # Estimate crop parameters
            margins = estimator.estimate_crop_params(patches)

            # Reflect to UI
            self.after(0, lambda: self.crop_vars["top"].set(margins.get("top", 0)))
            self.after(0, lambda: self.crop_vars["bottom"].set(margins.get("bottom", 0)))
            self.after(0, lambda: self.crop_vars["left"].set(margins.get("left", 0)))
            self.after(0, lambda: self.crop_vars["right"].set(margins.get("right", 0)))

            self.log(f"Auto-crop adjustment complete: Top={margins['top']}px, Bottom={margins['bottom']}px, "
                     f"Left={margins['left']}px, Right={margins['right']}px")
            self.after(0, lambda: self._finish_busy_state("Auto-crop Complete"))

            # Execute preview update in UI thread
            self.after(100, self._on_update_preview)

        except Exception as e:
            self.log(f"Auto-crop adjustment error: {e}")
            self.after(0, lambda: self._finish_busy_state("Auto-crop error"))
            traceback.print_exc()

        finally:
            if cap is not None:
                cap.release()
            self.is_analyzing = False
            self.after(0, lambda: self.auto_crop_btn.configure(state=tk.NORMAL))

    def _on_output(self):
        """Execute output"""
        if not self.preview_sprites:
            messagebox.showwarning("Warning", "Please update preview first")
            return
        
        # Determine output directory
        video_dir = os.path.dirname(os.path.abspath(self.video_path))
        base_output = os.path.join(video_dir, "mouth")
        output_dir = get_unique_output_dir(base_output)
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            for name, rgba in self.preview_sprites.items():
                filepath = os.path.join(output_dir, f"{name}.png")
                # Internal representation is RGBA, so reorder to BGRA before OpenCV save
                if not write_image_file(
                    filepath,
                    cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA),
                ):
                    raise RuntimeError(f"Failed to save: {filepath}")
                self.log(f"Saved: {name}.png")
            
            self.log(f"OutputComplete: {output_dir}")
            
            if messagebox.askyesno("Complete", f"Output has been completed.\n{output_dir}\n\nDo you want to open the folder?"):
                open_path_with_default_app(output_dir)
            
        except Exception as e:
            self.log(f"OutputError: {e}")
            messagebox.showerror("Error", str(e))
    
    def destroy(self):
        """Cleanup"""
        self._stop_live_test()
        self._stop_player()
        if self._cached_cap:
            self._cached_cap.release()
        self._close_player_capture()
        if self.player_window and self.player_window.winfo_exists():
            self.player_window.destroy()
        if self.live_test_window and self.live_test_window.winfo_exists():
            self.live_test_window.destroy()
        super().destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Main function"""
    app = MouthSpriteExtractorApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
