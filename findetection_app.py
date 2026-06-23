"""Simple desktop app for running the local fin detection pipeline.

This wraps the existing preprocessing helper, local detection server, and crop
pipeline in one Tk interface. It intentionally does not touch training.
"""

from __future__ import annotations

import base64
import json
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from scripts.picture_preprocessing import (
    RAW_EXTENSIONS,
    format_file_size,
    get_target_path,
    iter_source_files,
    rawpy,
    save_as_jpeg,
    scan_input_files,
    validate_gc_interval,
    validate_jpeg_quality,
)


APP_DIR = Path(__file__).resolve().parent
ASSETS_DIR = APP_DIR / "assets"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_BASE_URL = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/api/inference"
DEFAULT_JPEG_QUALITY = 75
DEFAULT_GC_INTERVAL = 500
REQUEST_TIMEOUT_SECONDS = 120
MAX_REQUEST_RETRIES = 3
SERVER_ERROR_MARKERS = (
    "ERROR",
    "CRITICAL",
    "Traceback",
    "Exception",
    "Error",
    "failed",
    "Failed",
    "not found",
    "No such file",
    "Address already in use",
)

MODEL_OPTIONS = {
    "Risso": APP_DIR / "deployment_model_risso.pt",
    "Orca": APP_DIR / "deployment_model_orca.pt",
}

MODEL_BADGE_MAX_SIZE = (160, 86)
MODEL_BADGE_CANDIDATES = {
    "Orca": (
        ASSETS_DIR / "orca_model.png",
        ASSETS_DIR / "orca_model.jpg",
        ASSETS_DIR / "orca_model.jpeg",
        ASSETS_DIR / "norwegian_orca_survey.png",
        ASSETS_DIR / "norwegian_orca_survey.jpg",
        ASSETS_DIR / "norwegian_orca_survey.jpeg",
    ),
    "Risso": (
        ASSETS_DIR / "risso_model.png",
        ASSETS_DIR / "risso_model.jpg",
        ASSETS_DIR / "risso_model.jpeg",
        ASSETS_DIR / "risso_dolphin.png",
        ASSETS_DIR / "risso_dolphin.jpg",
        ASSETS_DIR / "risso_dolphin.jpeg",
    ),
}

IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


class PipelineStopped(Exception):
    """Raised when the user asks the running pipeline to stop."""


@dataclass(frozen=True)
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    preprocess: bool
    base_url: str = DEFAULT_BASE_URL


@dataclass(frozen=True)
class PipelineSummary:
    completed: bool
    converted: int = 0
    conversion_failed: int = 0
    processed: int = 0
    crops_saved: int = 0
    detection_failed: int = 0
    output_dir: Path | None = None
    preprocessed_dir: Path | None = None


class FinDetectionApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("FinDetection")
        self.root.geometry("980x720")
        self.root.minsize(860, 620)

        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.stop_event = threading.Event()
        self.pipeline_thread: threading.Thread | None = None
        self.pipeline_running = False
        self.pending_pipeline_start = False
        self.pending_pipeline_config: PipelineConfig | None = None

        self.server_process: subprocess.Popen[str] | None = None
        self.server_starting = False
        self.server_stopping = False
        self.external_server_active = False
        self.server_had_error = False
        self.running_model_name: str | None = None

        self.input_dir_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.preprocess_var = tk.BooleanVar(value=False)
        self.model_var = tk.StringVar(value="Risso")
        self.server_status_var = tk.StringVar(value="Server stopped")
        self.pipeline_status_var = tk.StringVar(value="Ready")
        self.progress_label_var = tk.StringVar(value="Waiting to start")

        self._build_styles()
        self.model_badges = self.create_model_badges()
        self._build_layout()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self.poll_events)
        self.root.after(350, self.refresh_server_status)

    def _build_styles(self) -> None:
        self.colors = {
            "bg": "#f4f7fb",
            "panel": "#ffffff",
            "border": "#d7dde8",
            "text": "#172033",
            "muted": "#64748b",
            "accent": "#2563eb",
            "accent_hover": "#1d4ed8",
            "success": "#15803d",
            "danger": "#b91c1c",
            "log_bg": "#111827",
            "log_fg": "#d1d5db",
        }

        self.root.configure(bg=self.colors["bg"])
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("App.TFrame", background=self.colors["bg"])
        style.configure("Panel.TFrame", background=self.colors["panel"], relief="flat")
        style.configure(
            "Header.TLabel",
            background=self.colors["bg"],
            foreground=self.colors["text"],
            font=("Helvetica", 24, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=self.colors["bg"],
            foreground=self.colors["muted"],
            font=("Helvetica", 12),
        )
        style.configure(
            "Section.TLabel",
            background=self.colors["panel"],
            foreground=self.colors["text"],
            font=("Helvetica", 13, "bold"),
        )
        style.configure(
            "Body.TLabel",
            background=self.colors["panel"],
            foreground=self.colors["text"],
            font=("Helvetica", 11),
        )
        style.configure(
            "Muted.TLabel",
            background=self.colors["panel"],
            foreground=self.colors["muted"],
            font=("Helvetica", 10),
        )
        style.configure(
            "Status.TLabel",
            background=self.colors["panel"],
            foreground=self.colors["muted"],
            font=("Helvetica", 11, "bold"),
        )
        style.configure(
            "StatusMuted.TLabel",
            background=self.colors["panel"],
            foreground=self.colors["muted"],
            font=("Helvetica", 11, "bold"),
        )
        style.configure(
            "StatusOk.TLabel",
            background=self.colors["panel"],
            foreground=self.colors["success"],
            font=("Helvetica", 11, "bold"),
        )
        style.configure(
            "StatusError.TLabel",
            background=self.colors["panel"],
            foreground=self.colors["danger"],
            font=("Helvetica", 11, "bold"),
        )
        style.configure(
            "Accent.TButton",
            background=self.colors["accent"],
            foreground="#ffffff",
            bordercolor=self.colors["accent"],
            focusthickness=0,
            focuscolor=self.colors["accent"],
            padding=(14, 8),
            font=("Helvetica", 11, "bold"),
        )
        style.map(
            "Accent.TButton",
            background=[("active", self.colors["accent_hover"]), ("disabled", "#93a4bd")],
            foreground=[("disabled", "#f8fafc")],
        )
        style.configure("Tool.TButton", padding=(12, 7), font=("Helvetica", 10))
        style.configure("TEntry", padding=6)
        style.configure("TCombobox", padding=6)
        style.configure(
            "Horizontal.TProgressbar",
            troughcolor="#e2e8f0",
            background=self.colors["accent"],
            bordercolor="#e2e8f0",
            lightcolor=self.colors["accent"],
            darkcolor=self.colors["accent"],
        )

    def _build_layout(self) -> None:
        main = ttk.Frame(self.root, style="App.TFrame", padding=(24, 22, 24, 18))
        main.pack(fill=tk.BOTH, expand=True)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(4, weight=1)

        header_frame = ttk.Frame(main, style="App.TFrame")
        header_frame.grid(row=0, column=0, rowspan=2, sticky="ew", pady=(0, 18))
        header_frame.columnconfigure(0, weight=1)
        header_frame.columnconfigure(1, minsize=MODEL_BADGE_MAX_SIZE[0])

        ttk.Label(header_frame, text="FinDetection", style="Header.TLabel").grid(row=0, column=0, sticky="w")

        self.model_badge_frame = tk.Frame(
            header_frame,
            width=MODEL_BADGE_MAX_SIZE[0],
            height=MODEL_BADGE_MAX_SIZE[1],
            bg=self.colors["bg"],
            borderwidth=0,
            highlightthickness=0,
        )
        self.model_badge_frame.grid(row=0, column=1, rowspan=2, sticky="ne", padx=(18, 0))
        self.model_badge_frame.grid_propagate(False)
        self.model_badge_label = tk.Label(
            self.model_badge_frame,
            bg=self.colors["bg"],
            borderwidth=0,
            highlightthickness=0,
        )
        self.model_badge_label.place(relx=1.0, rely=0.0, anchor="ne")
        self.model_badge_label.place_forget()

        ttk.Label(
            header_frame,
            text="Choose folders, pick a fin model, start the local recognizer, and run the crop pipeline.",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        setup_panel = self._panel(main)
        setup_panel.grid(row=2, column=0, sticky="ew", pady=(0, 14))
        setup_panel.columnconfigure(1, weight=1)
        ttk.Label(setup_panel, text="Setup", style="Section.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 12)
        )

        self._path_row(
            setup_panel,
            row=1,
            label="Input folder",
            variable=self.input_dir_var,
            command=lambda: self.choose_folder(self.input_dir_var),
        )
        self._path_row(
            setup_panel,
            row=2,
            label="Output folder",
            variable=self.output_dir_var,
            command=lambda: self.choose_folder(self.output_dir_var),
        )

        ttk.Label(setup_panel, text="Model", style="Body.TLabel").grid(row=3, column=0, sticky="w", pady=(12, 0))
        self.model_combo = ttk.Combobox(
            setup_panel,
            textvariable=self.model_var,
            values=tuple(MODEL_OPTIONS.keys()),
            state="readonly",
            width=16,
        )
        self.model_combo.grid(row=3, column=1, sticky="w", pady=(12, 0))
        ttk.Checkbutton(
            setup_panel,
            text="Use picture preprocessing first",
            variable=self.preprocess_var,
        ).grid(row=3, column=2, sticky="e", pady=(12, 0))

        action_panel = self._panel(main)
        action_panel.grid(row=3, column=0, sticky="ew", pady=(0, 14))
        action_panel.columnconfigure(2, weight=1)
        ttk.Label(action_panel, text="Run", style="Section.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 12))

        self.server_button = ttk.Button(
            action_panel,
            text="Start recognition server",
            style="Tool.TButton",
            command=self.toggle_server,
        )
        self.server_button.grid(row=1, column=0, sticky="w", padx=(0, 10))

        self.pipeline_button = ttk.Button(
            action_panel,
            text="Start pipeline",
            style="Accent.TButton",
            command=self.toggle_pipeline,
        )
        self.pipeline_button.grid(row=1, column=1, sticky="w")

        status_frame = ttk.Frame(action_panel, style="Panel.TFrame")
        status_frame.grid(row=1, column=2, sticky="e")
        self.server_status_label = ttk.Label(
            status_frame,
            textvariable=self.server_status_var,
            style="StatusMuted.TLabel",
        )
        self.server_status_label.pack(anchor="e")
        ttk.Label(status_frame, textvariable=self.pipeline_status_var, style="Muted.TLabel").pack(anchor="e")

        self.progress_bar = ttk.Progressbar(action_panel, mode="determinate", maximum=100)
        self.progress_bar.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(18, 6))
        ttk.Label(action_panel, textvariable=self.progress_label_var, style="Muted.TLabel").grid(
            row=3, column=0, columnspan=3, sticky="w"
        )

        log_panel = self._panel(main)
        log_panel.grid(row=4, column=0, sticky="nsew")
        log_panel.rowconfigure(1, weight=1)
        log_panel.columnconfigure(0, weight=1)
        ttk.Label(log_panel, text="Activity", style="Section.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 10))

        self.log_display = scrolledtext.ScrolledText(
            log_panel,
            height=13,
            wrap=tk.WORD,
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            highlightcolor=self.colors["accent"],
            bg=self.colors["log_bg"],
            fg=self.colors["log_fg"],
            insertbackground=self.colors["log_fg"],
            font=("Menlo", 11),
        )
        self.log_display.grid(row=1, column=0, sticky="nsew")
        self.log_display.configure(state="disabled")

        self.log("Choose folders, start the recognition server, then start the pipeline.")

    def _panel(self, parent: ttk.Frame) -> ttk.Frame:
        return ttk.Frame(parent, style="Panel.TFrame", padding=18)

    def create_model_badges(self) -> dict[str, ImageTk.PhotoImage]:
        badges: dict[str, ImageTk.PhotoImage] = {}
        for model_name, candidate_paths in MODEL_BADGE_CANDIDATES.items():
            badge = load_model_badge(candidate_paths)
            if badge is not None:
                badges[model_name] = ImageTk.PhotoImage(badge)
        return badges

    def set_server_status(self, text: str, state: str = "muted") -> None:
        style_by_state = {
            "muted": "StatusMuted.TLabel",
            "ok": "StatusOk.TLabel",
            "error": "StatusError.TLabel",
        }
        self.server_status_var.set(text)
        self.server_status_label.configure(style=style_by_state.get(state, "StatusMuted.TLabel"))

    def show_model_badge(self, model_name: str | None) -> None:
        badge = self.model_badges.get(model_name or "")
        if badge is None:
            self.model_badge_label.configure(image="")
            self.model_badge_label.place_forget()
            return

        self.model_badge_label.configure(image=badge)
        self.model_badge_label.image = badge
        self.model_badge_label.place(relx=1.0, rely=0.0, anchor="ne")

    def _path_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        command: Any,
    ) -> None:
        ttk.Label(parent, text=label, style="Body.TLabel").grid(row=row, column=0, sticky="w", pady=4)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", padx=(12, 10), pady=4)
        ttk.Button(parent, text="Browse", style="Tool.TButton", command=command).grid(
            row=row, column=2, sticky="e", pady=4
        )

    def choose_folder(self, variable: tk.StringVar) -> None:
        path = filedialog.askdirectory()
        if path:
            variable.set(path)

    def post(self, event_name: str, payload: Any = None) -> None:
        self.events.put((event_name, payload))

    def poll_events(self) -> None:
        while True:
            try:
                event_name, payload = self.events.get_nowait()
            except queue.Empty:
                break
            self.handle_event(event_name, payload)
        self.root.after(100, self.poll_events)

    def handle_event(self, event_name: str, payload: Any) -> None:
        if event_name == "log":
            self.log(str(payload))
        elif event_name == "server_ready":
            self.on_server_ready(payload)
        elif event_name == "server_failed":
            self.on_server_failed(str(payload))
        elif event_name == "server_exited":
            self.on_server_exited(payload)
        elif event_name == "server_output_error":
            self.on_server_output_error(str(payload))
        elif event_name == "progress":
            current, total, label = payload
            self.set_progress(current, total, label)
        elif event_name == "progress_stop":
            self.progress_bar.stop()
            self.progress_bar.configure(mode="determinate")
        elif event_name == "pipeline_started":
            self.pipeline_running = True
            self.stop_event.clear()
            self.pipeline_button.configure(text="Stop pipeline")
            self.pipeline_status_var.set("Pipeline running")
        elif event_name == "pipeline_finished":
            self.on_pipeline_finished(payload)

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_display.configure(state="normal")
        self.log_display.insert(tk.END, f"{timestamp}  {message}\n")
        self.log_display.configure(state="disabled")
        self.log_display.yview(tk.END)

    def set_progress(self, current: int, total: int, label: str) -> None:
        self.progress_bar.stop()
        self.progress_bar.configure(mode="determinate", maximum=max(total, 1))
        self.progress_bar["value"] = min(current, max(total, 1))
        self.progress_label_var.set(label)

    def refresh_server_status(self) -> None:
        if self.server_process is not None and self.server_process.poll() is None:
            model_name = self.running_model_name or self.model_var.get()
            state = "error" if self.server_had_error else "ok"
            suffix = " (error)" if self.server_had_error else ""
            self.set_server_status(f"Server running: {model_name}{suffix}", state)
            self.server_button.configure(text="Stop recognition server")
            return

        def check() -> None:
            reachable = is_server_reachable(DEFAULT_BASE_URL, timeout=0.8)
            self.post("server_ready" if reachable else "server_exited", None)

        threading.Thread(target=check, daemon=True).start()

    def toggle_server(self) -> None:
        if self.server_process is not None and self.server_process.poll() is None:
            self.stop_server()
            return
        if self.external_server_active:
            messagebox.showinfo(
                "Server already running",
                "A recognition server is already reachable. Stop it from where it was started if you need to change models.",
            )
            return
        self.start_server()

    def start_server(self) -> bool:
        if self.server_process is not None and self.server_process.poll() is None:
            self.log("Recognition server is already starting or running.")
            return True

        model_name = self.model_var.get()
        model_path = MODEL_OPTIONS.get(model_name)
        if model_path is None:
            messagebox.showerror("Unknown model", "Choose Risso or Orca before starting the server.")
            return False
        if not model_path.exists():
            messagebox.showerror("Missing model", f"Could not find model weights:\n{model_path}")
            return False

        if is_server_reachable(DEFAULT_BASE_URL, timeout=0.8):
            self.external_server_active = True
            self.running_model_name = None
            self.server_had_error = False
            self.set_server_status("Server running", "ok")
            self.show_model_badge(None)
            self.server_button.configure(text="Server already running")
            self.log(f"Recognition server is already reachable at {DEFAULT_BASE_URL}.")
            if self.pending_pipeline_start:
                config = self.pending_pipeline_config
                self.pending_pipeline_start = False
                self.pending_pipeline_config = None
                self.root.after(0, lambda: self.launch_pipeline(config))
            return True

        command = [
            sys.executable,
            str(APP_DIR / "scripts" / "spawn_findetection_server.py"),
            "--model-path",
            str(model_path),
            "--host",
            DEFAULT_HOST,
            "--port",
            str(DEFAULT_PORT),
        ]
        self.log(f"Starting {model_name} recognition server...")
        self.server_had_error = False
        self.running_model_name = model_name
        self.set_server_status("Starting server", "muted")
        self.show_model_badge(None)
        self.server_button.configure(state="disabled")
        self.server_starting = True
        self.server_stopping = False
        self.progress_bar.stop()
        self.progress_bar.configure(mode="determinate")
        self.progress_bar["value"] = 0
        self.progress_label_var.set("Starting recognition server")

        try:
            process = subprocess.Popen(
                command,
                cwd=str(APP_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            self.server_starting = False
            self.server_had_error = True
            self.set_server_status("Server failed", "error")
            self.server_button.configure(state="normal")
            messagebox.showerror("Server failed to start", str(exc))
            return False

        self.server_process = process
        threading.Thread(target=self.read_server_output, args=(process,), daemon=True).start()
        threading.Thread(target=self.wait_for_server_ready, args=(process, model_name), daemon=True).start()
        return True

    def read_server_output(self, process: subprocess.Popen[str]) -> None:
        in_traceback = False
        if process.stdout is not None:
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue
                if "Traceback" in line:
                    in_traceback = True
                if in_traceback or is_server_error_line(line):
                    self.post("server_output_error", line)
                    if is_probable_exception_summary(line):
                        in_traceback = False
        return_code = process.wait()
        self.post("server_exited", (process, return_code))

    def wait_for_server_ready(self, process: subprocess.Popen[str], model_name: str) -> None:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if process.poll() is not None:
                self.post("server_failed", "The recognition server stopped before it became ready.")
                return
            if is_server_reachable(DEFAULT_BASE_URL, timeout=1.0):
                self.post("server_ready", model_name)
                return
            time.sleep(0.5)
        self.post("server_failed", "Timed out while waiting for the recognition server.")

    def on_server_ready(self, model_name: Any) -> None:
        self.server_starting = False
        self.server_button.configure(state="normal")
        if self.server_process is not None and self.server_process.poll() is None:
            label = str(model_name or self.model_var.get())
            self.running_model_name = label
            state = "error" if self.server_had_error else "ok"
            suffix = " (error)" if self.server_had_error else ""
            self.set_server_status(f"Server running: {label}{suffix}", state)
            self.show_model_badge(label)
            self.server_button.configure(text="Stop recognition server")
            self.progress_label_var.set("Recognition server ready")
            self.log(f"Recognition server ready at {DEFAULT_BASE_URL}.")
        else:
            self.external_server_active = True
            self.running_model_name = None
            self.set_server_status("Server running", "ok")
            self.show_model_badge(None)
            self.server_button.configure(text="Server already running")
            self.progress_label_var.set("Recognition server ready")

        if self.pending_pipeline_start:
            config = self.pending_pipeline_config
            self.pending_pipeline_start = False
            self.pending_pipeline_config = None
            self.launch_pipeline(config)

    def on_server_failed(self, message: str) -> None:
        self.server_starting = False
        self.pending_pipeline_start = False
        self.pending_pipeline_config = None
        self.server_had_error = True
        self.running_model_name = None
        self.server_button.configure(state="normal", text="Start recognition server")
        self.set_server_status("Server error", "error")
        self.show_model_badge(None)
        self.progress_label_var.set("Recognition server failed")
        self.log(message)
        messagebox.showerror("Server failed", message)

    def on_server_output_error(self, line: str) -> None:
        self.server_had_error = True
        model_name = self.running_model_name or self.model_var.get()
        if self.server_process is not None and self.server_process.poll() is None:
            self.set_server_status(f"Server running: {model_name} (error)", "error")
        else:
            self.set_server_status("Server error", "error")
        self.log(f"server error: {line}")

    def on_server_exited(self, payload: Any) -> None:
        if payload is None:
            self.external_server_active = False
            self.running_model_name = None
            self.server_had_error = False
            self.set_server_status("Server stopped", "muted")
            self.show_model_badge(None)
            self.server_button.configure(text="Start recognition server", state="normal")
            return

        process, return_code = payload
        if process is not self.server_process:
            return
        was_stopping = self.server_stopping
        self.server_process = None
        self.server_starting = False
        self.server_stopping = False
        self.external_server_active = False
        self.running_model_name = None
        self.show_model_badge(None)
        self.server_button.configure(text="Start recognition server", state="normal")
        if was_stopping or return_code in (0, None):
            self.server_had_error = False
            self.set_server_status("Server stopped", "muted")
            self.progress_label_var.set("Waiting to start")
            self.log("Recognition server stopped.")
        else:
            self.server_had_error = True
            self.set_server_status("Server error", "error")
            self.progress_label_var.set("Recognition server failed")
            self.log(f"Recognition server stopped with exit code {return_code}.")

    def stop_server(self) -> None:
        process = self.server_process
        if process is None:
            return
        self.server_button.configure(state="disabled")
        self.set_server_status("Stopping server", "muted")
        self.server_stopping = True
        self.log("Stopping recognition server...")
        threading.Thread(target=self.stop_server_worker, args=(process,), daemon=True).start()

    def stop_server_worker(self, process: subprocess.Popen[str]) -> None:
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def toggle_pipeline(self) -> None:
        if self.pipeline_running:
            self.stop_event.set()
            self.pipeline_button.configure(text="Stopping...", state="disabled")
            self.pipeline_status_var.set("Stopping after current file")
            self.log("Stopping pipeline after the current file finishes...")
            return
        self.start_pipeline()

    def start_pipeline(self) -> None:
        config = self.read_pipeline_config()
        if config is None:
            return
        if not is_server_reachable(config.base_url, timeout=1.5):
            if self.server_process is not None and self.server_process.poll() is None:
                self.pending_pipeline_start = True
                self.pending_pipeline_config = config
                self.log("Waiting for the recognition server to finish starting before running the pipeline.")
                return
            should_start = messagebox.askyesno(
                "Start recognition server?",
                "The recognition server is not reachable. Start it now and run the pipeline when it is ready?",
            )
            if not should_start:
                return
            self.pending_pipeline_start = True
            self.pending_pipeline_config = config
            if not self.start_server():
                self.pending_pipeline_start = False
                self.pending_pipeline_config = None
            return
        self.launch_pipeline(config)

    def launch_pipeline(self, config: PipelineConfig | None = None) -> None:
        if config is None:
            config = self.read_pipeline_config()
        if config is None:
            return
        self.stop_event.clear()
        self.pipeline_thread = threading.Thread(target=self.run_pipeline, args=(config,), daemon=True)
        self.pipeline_thread.start()

    def read_pipeline_config(self) -> PipelineConfig | None:
        input_text = self.input_dir_var.get().strip()
        output_text = self.output_dir_var.get().strip()
        if not input_text:
            messagebox.showerror("Input folder needed", "Choose an input folder first.")
            return None
        if not output_text:
            messagebox.showerror("Output folder needed", "Choose an output folder first.")
            return None

        input_dir = Path(input_text).expanduser().resolve()
        output_dir = Path(output_text).expanduser().resolve()
        if not input_dir.is_dir():
            messagebox.showerror("Input folder missing", f"Input folder does not exist:\n{input_dir}")
            return None
        if output_dir == input_dir:
            messagebox.showerror(
                "Choose a separate output folder",
                "The output folder must be different from the input folder so results do not get mixed back into the source images.",
            )
            return None

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            messagebox.showerror("Output folder error", f"Could not create output folder:\n{exc}")
            return None

        return PipelineConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            preprocess=self.preprocess_var.get(),
        )

    def run_pipeline(self, config: PipelineConfig) -> None:
        self.post("pipeline_started")
        summary = PipelineSummary(completed=False, output_dir=config.output_dir)
        try:
            working_input = config.input_dir
            converted = 0
            conversion_failed = 0
            preprocessed_dir = None

            self.post("log", f"Input folder: {config.input_dir}")
            self.post("log", f"Output folder: {config.output_dir}")

            if config.preprocess:
                converted, conversion_failed, preprocessed_dir = self.run_preprocessing(config)
                working_input = preprocessed_dir

            processed, crops_saved, detection_failed = self.run_detection(config, working_input)
            summary = PipelineSummary(
                completed=True,
                converted=converted,
                conversion_failed=conversion_failed,
                processed=processed,
                crops_saved=crops_saved,
                detection_failed=detection_failed,
                output_dir=config.output_dir,
                preprocessed_dir=preprocessed_dir,
            )
        except PipelineStopped:
            self.post("log", "Pipeline stopped by user.")
            summary = PipelineSummary(completed=False, output_dir=config.output_dir)
        except Exception as exc:
            self.post("log", f"Pipeline failed: {exc}")
            summary = PipelineSummary(completed=False, output_dir=config.output_dir)
        finally:
            self.post("pipeline_finished", summary)

    def run_preprocessing(self, config: PipelineConfig) -> tuple[int, int, Path]:
        jpeg_quality = validate_jpeg_quality(DEFAULT_JPEG_QUALITY)
        gc_interval = validate_gc_interval(DEFAULT_GC_INTERVAL)
        preprocessed_dir = config.output_dir / "preprocessed_images"
        if preprocessed_dir.exists():
            shutil.rmtree(preprocessed_dir)
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        exclude_dir = config.output_dir if is_relative_to(config.output_dir, config.input_dir) else preprocessed_dir

        if rawpy is None:
            first_raw_file = find_first_raw_file(config.input_dir, exclude_dir)
            if first_raw_file is not None:
                raise RuntimeError(
                    "RAW files were found but rawpy is not installed. "
                    "Close the app and open LaunchFinDetectionTool.command so it can install missing requirements. "
                    f"First RAW file: {first_raw_file.relative_to(config.input_dir)}"
                )

        total_files, total_input_size = scan_input_files(config.input_dir, exclude_dir)
        self.post(
            "log",
            f"Preprocessing {total_files} files to JPEG ({format_file_size(total_input_size)} source data).",
        )
        if total_files == 0:
            raise RuntimeError("No files found to preprocess.")

        converted = 0
        failed = 0
        target_path_counts: dict[Path, int] = {}
        for index, source_path in enumerate(iter_source_files(config.input_dir, exclude_dir), start=1):
            self.raise_if_stopped()
            relative_path = source_path.relative_to(config.input_dir)
            target_path = get_target_path(
                preprocessed_dir / relative_path.with_suffix(".jpg"),
                target_path_counts,
            )
            target_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                save_as_jpeg(source_path, target_path, jpeg_quality)
                shutil.copystat(source_path, target_path, follow_symlinks=True)
                converted += 1
            except Exception as exc:
                failed += 1
                self.post("log", f"Could not preprocess {relative_path}: {exc}")
            finally:
                if gc_interval and index % gc_interval == 0:
                    import gc

                    gc.collect()
                self.post(
                    "progress",
                    (
                        index,
                        total_files,
                        f"Preprocessing {index}/{total_files} files",
                    ),
                )

        self.post("log", f"Preprocessing done. Converted {converted}; failed {failed}.")
        self.post("log", f"Preprocessed JPEGs: {preprocessed_dir}")
        return converted, failed, preprocessed_dir

    def run_detection(self, config: PipelineConfig, input_dir: Path) -> tuple[int, int, int]:
        exclude_dir = (
            config.output_dir
            if input_dir == config.input_dir and is_relative_to(config.output_dir, config.input_dir)
            else None
        )
        image_paths = list(iter_image_files(input_dir, exclude_dir=exclude_dir))
        total_images = len(image_paths)
        self.post("log", f"Found {total_images} images for fin recognition.")
        if total_images == 0:
            raise RuntimeError("No supported images found for fin recognition.")

        processed = 0
        crops_saved = 0
        failed = 0
        for index, image_path in enumerate(image_paths, start=1):
            self.raise_if_stopped()
            relative_label = image_path.relative_to(input_dir)
            self.post("log", f"Processing {relative_label}")
            try:
                detections = request_detections(image_path, config.base_url)
            except Exception as exc:
                failed += 1
                self.post("log", f"Detection failed for {relative_label}: {exc}")
                detections = []

            if not detections:
                self.post("log", f"No fins detected in {relative_label}")
            else:
                output_dir = config.output_dir / image_path.parent.relative_to(input_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                for crop_index, detection in enumerate(detections):
                    output_file = output_dir / f"{image_path.stem}_cropped_{crop_index}.JPG"
                    save_base64_image(detection, output_file)
                    crops_saved += 1

            processed += 1
            self.post(
                "progress",
                (
                    index,
                    total_images,
                    f"Finding fins {index}/{total_images} images",
                ),
            )

        self.post(
            "log",
            f"Fin recognition done. Processed {processed}; saved {crops_saved} crops; failed {failed}.",
        )
        return processed, crops_saved, failed

    def raise_if_stopped(self) -> None:
        if self.stop_event.is_set():
            raise PipelineStopped()

    def on_pipeline_finished(self, summary: PipelineSummary) -> None:
        self.pipeline_running = False
        self.stop_event.clear()
        self.pipeline_button.configure(text="Start pipeline", state="normal")
        self.pipeline_status_var.set("Ready")
        self.post("progress_stop")

        if summary.completed:
            self.set_progress(1, 1, "Pipeline complete")
            self.log(
                "Done. "
                f"Processed {summary.processed} images and saved {summary.crops_saved} fin crops."
            )
            if summary.output_dir is not None:
                self.log(f"Results saved in: {summary.output_dir}")
        else:
            self.progress_label_var.set("Pipeline did not complete")

    def on_close(self) -> None:
        if self.pipeline_running:
            close_now = messagebox.askyesno(
                "Pipeline is running",
                "Stop the running pipeline and close the app?",
            )
            if not close_now:
                return
            self.stop_event.set()
        process = self.server_process
        if process is not None and process.poll() is None:
            process.terminate()
        self.root.destroy()


def is_server_reachable(base_url: str, timeout: float = 2.0) -> bool:
    try:
        response = requests.get(base_url, timeout=timeout)
    except requests.RequestException:
        return False
    return response.status_code < 500


def is_server_error_line(line: str) -> bool:
    return any(marker in line for marker in SERVER_ERROR_MARKERS)


def is_probable_exception_summary(line: str) -> bool:
    summary = line.split(":", 1)[0].strip()
    return summary.endswith(("Error", "Exception"))


def load_model_badge(candidate_paths: tuple[Path, ...]) -> Image.Image | None:
    for image_path in candidate_paths:
        if not image_path.is_file():
            continue
        with Image.open(image_path) as image:
            badge = image.convert("RGBA")
            badge.thumbnail(MODEL_BADGE_MAX_SIZE, Image.Resampling.LANCZOS)
            return badge
    return None


def find_first_raw_file(directory: Path, exclude_dir: Path | None = None) -> Path | None:
    excluded = exclude_dir.resolve() if exclude_dir else None
    for path in directory.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in RAW_EXTENSIONS:
            continue
        if excluded is not None and is_relative_to(path.resolve(), excluded):
            continue
        return path
    return None


def iter_image_files(directory: Path, exclude_dir: Path | None = None) -> list[Path]:
    excluded = exclude_dir.resolve() if exclude_dir else None
    images: list[Path] = []
    for path in directory.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if excluded is not None and is_relative_to(path.resolve(), excluded):
            continue
        images.append(path)
    return sorted(images)


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def request_detections(image_path: Path, base_url: str) -> list[str]:
    url = base_url.rstrip("/") + "/fin-detect"
    last_error: Exception | None = None
    for attempt in range(1, MAX_REQUEST_RETRIES + 1):
        try:
            with image_path.open("rb") as image_file:
                response = requests.post(
                    url,
                    files={"file": image_file},
                    verify=False,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
            if response.status_code != 200:
                raise RuntimeError(f"server returned {response.status_code}: {response.text[:300]}")
            content = json.loads(response.text)
            response_content = content.get("response", {})
            return response_content.get("extractedImages") or response_content.get("croppedImages", [])
        except Exception as exc:
            last_error = exc
            if attempt < MAX_REQUEST_RETRIES:
                time.sleep(1.0)
    raise RuntimeError(str(last_error))


def save_base64_image(base64_string: str, output_path: Path) -> None:
    image_data = base64.b64decode(base64_string)
    with Image.open(BytesIO(image_data)) as image:
        image.load()
        if image.mode != "RGB":
            converted = image.convert("RGB")
            try:
                converted.save(output_path, format="JPEG", quality=95)
            finally:
                converted.close()
        else:
            image.save(output_path, format="JPEG", quality=95)


def main() -> None:
    root = tk.Tk()
    app = FinDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
