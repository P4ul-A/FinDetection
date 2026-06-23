import base64
import glob
import json
import logging
import os
from datetime import datetime
import tkinter as tk
from io import BytesIO
from tkinter import scrolledtext, messagebox, Toplevel, Label, Entry, Button
from argparse import Namespace
import requests
import threading
import shutil
import queue

from PIL import Image

# Generate the filename with current datetime
log_filename = f"finwave_pipeline_image_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure log to file and stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

# Create a logger
logger = logging.getLogger(__name__)

# Default settings
settings = {
    "input_directory": "/Users/paul/Desktop/NOS/CreateFinDataset/input",
    "output_directory": "/Users/paul/Desktop/NOS/CreateFinDataset/output",
    "verify": False,
    "make_dir_for_crop": False,
    "base_url": "http://127.0.0.1:8000/api/inference",
    "detect_path": "/fin-detect",
    "identify_path": "/fin-identify",
    "keep_directory_structure": True,
    "max_retries": 10,
    "request_timeout": 120,
    "vvi_path": "/vvi-detect",
    "do_vvi": True,
    "invalid_path": ".invalid"
}

class FinwaveGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Finwave Pipeline")
        self.root.geometry("800x800")

        self.log_display = scrolledtext.ScrolledText(root, state='disabled', height=15)
        self.log_display.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        self.is_running = False  # To track if pipeline is running
        self.thread = None  # To hold the pipeline thread
        self.pipeline_events = queue.Queue()

        self.start_button = tk.Button(root, text="Start Pipeline", command=self.toggle_pipeline)
        self.start_button.pack(pady=5)

        self.settings_button = tk.Button(root, text="Settings", command=self.open_settings)
        self.settings_button.pack(pady=5)

        # Redirect log to the GUI
        self.setup_logging()
        self.root.after(100, self.poll_pipeline_events)

    def poll_pipeline_events(self):
        while True:
            try:
                event, completed = self.pipeline_events.get_nowait()
            except queue.Empty:
                break
            if event == "finished":
                self.finish_pipeline(completed)
        self.root.after(100, self.poll_pipeline_events)

    def setup_logging(self):
        log_queue = queue.Queue()

        class TextHandler(logging.Handler):
            def __init__(self, message_queue):
                super().__init__()
                self.message_queue = message_queue

            def emit(self, record):
                self.message_queue.put(self.format(record))

        def poll_log_queue():
            while True:
                try:
                    msg = log_queue.get_nowait()
                except queue.Empty:
                    break
                self.log_display.config(state='normal')
                self.log_display.insert(tk.END, msg + '\n')
                self.log_display.config(state='disabled')
                self.log_display.yview(tk.END)
            self.root.after(100, poll_log_queue)

        text_handler = TextHandler(log_queue)
        text_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(text_handler)
        logger.propagate = False  # Add this line
        self.root.after(100, poll_log_queue)

    def toggle_pipeline(self):
        if not self.is_running:

            logger.info("Starting pipeline")
            self.is_running = True
            self.start_button.config(text="Stop Pipeline")  # Change button text to Stop
            self.thread = threading.Thread(target=self.start_pipeline, daemon=True)
            self.thread.start()  # Start the pipeline in a new thread
        else:
            logger.info("Stopping pipeline")
            self.is_running = False
            self.start_button.config(text="Start Pipeline")  # Change button text back to Start
            # Optionally, signal the pipeline thread to stop here if needed

    def start_pipeline(self):
        completed = False
        try:
            logger.info("Checking server connectivity...")
            if not self.check_server(settings["base_url"]):
                logger.error("Server is not reachable. Please check the base URL.")
                return

            logger.info(f"Starting pipeline with settings: {settings}")
            logger.info("Step 1: Data loading...")
            args = Namespace(**settings)
            images = get_images(args.input_directory)
            logger.info(f"Found {len(images)} images")
            for idx, path in list(enumerate(images)):
                if not self.is_running:
                    logger.info("Pipeline stopped.")
                    return
                logger.info(f"[{idx + 1} \t / \t {len(images)}] Processing {path}")
                self.process_image(path, args)
            completed = True
            logger.info("Pipeline finished.")
        finally:
            self.pipeline_events.put(("finished", completed))

    def finish_pipeline(self, completed):
        self.is_running = False
        self.start_button.config(text="Start Pipeline")

    def process_image(self, image_path, ARGS, retry_count=0):
        url = ARGS.base_url + ARGS.detect_path

        if retry_count < ARGS.max_retries:
            try:
                with open(image_path, 'rb') as image_file:
                    response = requests.post(
                        url,
                        files={'file': image_file},
                        verify=ARGS.verify,
                        timeout=ARGS.request_timeout,
                    )
                if response.status_code == 200:
                    content = json.loads(response.text)
                    response_content = content["response"]
                    detections = response_content.get("extractedImages") or response_content.get("croppedImages", [])
                    if len(detections) == 0:
                        logger.info(f"No fins detected in {image_path}")
                    output = ARGS.output_directory
                    if output is None:
                        output = os.path.join(ARGS.input_directory, "CROPPED")
                    if ARGS.keep_directory_structure:
                        output_dir = get_path_diff(ARGS.input_directory, os.path.dirname(image_path))
                        output = os.path.join(output, output_dir)
                    if ARGS.make_dir_for_crop:
                        output = os.path.join(output, os.path.basename(image_path))

                    for i, detection in enumerate(detections):
                        img = load_image_from_base64(detection)
                        valid = True

                        os.makedirs(output, exist_ok=True)
                        output_file = os.path.basename(image_path).split(".")[0] + f"_cropped_{i}.JPG"
                        out_path = os.path.join(output, output_file)
                        img.save(out_path)
                        if ARGS.do_vvi:
                            valid = get_vvi(out_path, ARGS)
                        if not valid:
                            invalid_out = get_invalid_path(output_dir, ARGS)
                            new_path = os.path.join(invalid_out, output_file)
                            logger.warning(f"Image not valid. Moving to invalid folder at : {new_path}")
                            shutil.move(out_path, new_path)
                else:
                    logger.error(f"Could not send photo to pipeline: {response.text}")
                    self.process_image(image_path, ARGS, retry_count + 1)
            except Exception as e:
                logger.error(f"Connection error. Maybe retrying: {e}")
                self.process_image(image_path, ARGS, retry_count + 1)
        else:
            logger.error(f"Max retries reached for {image_path}")

    def open_settings(self):
        settings_window = Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("600x600")

        entries = {}

        for i, (key, value) in enumerate(settings.items()):
            Label(settings_window, text=key).grid(row=i, column=0, padx=10, pady=5, sticky='w')
            entry = Entry(settings_window)
            entry.insert(0, str(value))
            entry.grid(row=i, column=1, padx=10, pady=5)
            entries[key] = entry

        def save_settings():
            logger.info("Saving settings")
            for key, entry in entries.items():
                new_value = entry.get()
                if isinstance(settings[key], bool):
                    settings[key] = new_value.lower() in ('true', '1', 'yes')
                elif isinstance(settings[key], int):
                    try:
                        settings[key] = int(new_value)
                    except ValueError:
                        messagebox.showerror("Invalid input", f"{key} must be an integer")
                        return
                else:
                    settings[key] = new_value
            settings_window.destroy()

        Button(settings_window, text="Save", command=save_settings).grid(row=len(settings), columnspan=2, pady=10)
    
    def check_server(self, base_url):
        try:
            response = requests.get(base_url, timeout=5)
            logger.info(f"Server reachable. Status: {response.status_code}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Server unreachable: {e}")
            return False


def get_images(directory):
    extension_list = ['jpg', 'jpeg', 'JPG', 'JPEG']
    images = []
    for extension in extension_list:
        images.extend(glob.glob(directory + '/**/*.' + extension, recursive=True))
    images = sorted(list(set(images)))
    return images


def load_image_from_base64(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image


def get_path_diff(path1, path2):
    replaced = path2.replace(path1, "")
    if len(replaced) > 0 and replaced[0] == os.sep:
        replaced = replaced[1:]
    return replaced

def get_vvi(img_path, ARGS):
    url = ARGS.base_url + ARGS.vvi_path
    with open(img_path, 'rb') as image_file:
        response = requests.post(
            url,
            files={'file': image_file},
            verify=ARGS.verify,
            timeout=ARGS.request_timeout,
        )
    try:
        response_content = json.loads(response.content)["response"]
        return response_content["class"].lower() == "valid"
    except Exception as e:
        logging.error(f"Could not parse response from server: {e}")
    return response

def get_invalid_path(source, ARGS):
    diff = get_path_diff(ARGS.invalid_path, source)
    p = os.path.join(ARGS.output_directory, ARGS.invalid_path, diff)
    os.makedirs(p, exist_ok=True)
    return p


if __name__ == "__main__":
    root = tk.Tk()
    app = FinwaveGUI(root)
    root.mainloop()
