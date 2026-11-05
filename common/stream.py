"""Threaded video reader for RTSP / webcam / video file.

A background thread keeps reading the camera and always holds only the
LATEST frame. This stops the RTSP buffer from filling up, so the analytics
never lag behind the live picture even when the AI is slower than the camera.
It reconnects automatically if the stream drops.
"""
import os
import pathlib
import sys
import threading
import time

import numpy as np

# Use TCP for RTSP (more reliable than UDP on most networks).
# Must be set before OpenCV opens the stream.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2  # noqa: E402


BROWSER_DIR = pathlib.Path(__file__).resolve().parent.parent / "outputs" / "browser_cam"


class StreamReader:
    def __init__(self, source, reconnect_seconds=5):
        s = str(source).strip()
        self.source = int(s) if s.isdigit() else s
        self.reconnect_seconds = reconnect_seconds
        self._frame = None
        self._id = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self.connected = False

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def read(self):
        """Return (frame_id, frame_copy) of the newest frame, or (0, None)."""
        with self._lock:
            if self._frame is None:
                return 0, None
            return self._id, self._frame.copy()

    def _open(self):
        if isinstance(self.source, int):
            if sys.platform == "win32":  # DirectShow opens webcams faster/more reliably on Windows
                cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                if cap.isOpened():
                    return cap
                cap.release()
            return cv2.VideoCapture(self.source)
        return cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)

    def _loop_browser(self):
        """Source 'browser': another laptop's webcam. The dashboard's /browser-camera page saves its pictures to
        outputs/browser_cam/<slot>.jpg (see serve.py); this reads the newest one. 'connected' = a picture in the last 5 s."""
        slot = self.source.split(":", 1)[1] if ":" in self.source else "default"
        frame_file = BROWSER_DIR / (slot + ".jpg")
        last_m, last_t, warned = 0, 0.0, False
        while not self._stop.is_set():
            try:
                m = frame_file.stat().st_mtime_ns
                if m != last_m:
                    img = cv2.imdecode(np.frombuffer(frame_file.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                    last_m = m
                    if img is not None:
                        with self._lock:
                            self._frame = img
                            self._id += 1
                        last_t = time.time()
                        if not self.connected:
                            print("[stream] connected (browser webcam)")
                        self.connected, warned = True, False
            except OSError:
                pass
            if self.connected and time.time() - last_t > 5:
                self.connected = False
                print("[stream] browser webcam stopped sending, waiting ...")
            elif not self.connected and not warned and time.time() - last_t > 5:
                print("[stream] waiting for the browser webcam page to start sending (open the /browser-camera link on the other laptop)")
                warned = True
            self._stop.wait(0.05)

    def _loop(self):
        if isinstance(self.source, str) and (self.source == "browser" or self.source.startswith("browser:")):
            return self._loop_browser()
        is_file = isinstance(self.source, str) and os.path.isfile(self.source)
        while not self._stop.is_set():
            cap = self._open()
            if not cap.isOpened():
                print(f"[stream] cannot open source, retrying in {self.reconnect_seconds}s")
                cap.release()
                self._stop.wait(self.reconnect_seconds)
                continue
            print("[stream] connected")
            self.connected = True
            delay = 0.0
            if is_file:  # play test videos at their natural speed
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                delay = 1.0 / fps
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                with self._lock:
                    self._frame = frame
                    self._id += 1
                if delay:
                    time.sleep(delay)
            cap.release()
            self.connected = False
            if not self._stop.is_set():
                print(f"[stream] lost/ended, reconnecting in {self.reconnect_seconds}s")
                self._stop.wait(self.reconnect_seconds)
