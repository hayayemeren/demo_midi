"""Vision-Guided Grasping and Object Detection for SO-100 Follower Arm.

Implements Suggestion 3 (Vision-Guided Grasping via Wrist Camera) and
Suggestion 2 (Gripper Load / Position Grasp Verification) for Case 4.
"""

from __future__ import annotations

import logging
import http.server
import json
import math
import os
import socket
import socketserver
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any

import cv2
import numpy as np

logger = logging.getLogger("demo.vision")


@dataclass
class DetectionResult:
    """Result of sock detection in camera frame."""
    found: bool = False
    cx: float = 0.0           # Centroid X (pixel)
    cy: float = 0.0           # Centroid Y (pixel)
    error_x: float = 0.0      # Normalized error from center [-1.0, 1.0]
    error_y: float = 0.0      # Normalized error from center [-1.0, 1.0]
    angle: float = 0.0        # Orientation angle in degrees [-90.0, 90.0]
    area: float = 0.0         # Contour area (pixels)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0) # (x, y, w, h)
    annotated_frame: Optional[np.ndarray] = None
    raw_frame: Optional[np.ndarray] = None
    mask_frame: Optional[np.ndarray] = None
    combined_frame: Optional[np.ndarray] = None


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """Multi-threaded TCP server to handle simultaneous HTTP requests and persistent MJPEG streams."""
    daemon_threads = True
    allow_reuse_address = True


class VisionDebugHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler serving real-time MJPEG streams and snapshots for operator debugging."""

    def log_message(self, format, *args):
        pass  # Suppress continuous HTTP access logs in console

    def do_HEAD(self):
        srv = VisionDebugServer._instance
        if not srv:
            self.send_error(503, "Vision debug server not initialized")
            return
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
        elif path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
        else:
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.end_headers()

    def do_GET(self):
        srv = VisionDebugServer._instance
        if not srv:
            self.send_error(503, "Vision debug server not initialized")
            return

        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(srv.get_dashboard_html().encode("utf-8"))
        elif path in ("/stream/combined", "/stream.mjpg"):
            srv.stream_mjpeg(self, "combined")
        elif path == "/stream/raw":
            srv.stream_mjpeg(self, "raw")
        elif path in ("/stream/mask", "/stream/segmented"):
            srv.stream_mjpeg(self, "mask")
        elif path in ("/stream/detection", "/stream/annotated"):
            srv.stream_mjpeg(self, "detection")
        elif path == "/raw.jpg":
            srv.serve_image(self, srv._latest_raw, "ACTUAL CAMERA FRAME")
        elif path in ("/mask.jpg", "/segmented.jpg"):
            srv.serve_image(self, srv._latest_mask, "SEGMENTED MASK")
        elif path in ("/detection.jpg", "/annotated.jpg"):
            srv.serve_image(self, srv._latest_annotated, "DETECTION RETICLE")
        elif path in ("/combined.jpg", "/debug.jpg"):
            srv.serve_image(self, srv._latest_combined, "COMBINED DEBUG VIEW")
        elif path == "/status.json":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            with srv._lock:
                meta = dict(srv._latest_meta)
            self.wfile.write(json.dumps(meta).encode("utf-8"))
        else:
            self.send_error(404, "Not Found")


class VisionDebugServer:
    """Lightweight HTTP server publishing live MJPEG video streams and segmented masks."""
    _instance: Optional[VisionDebugServer] = None
    _lock = threading.Lock()

    def __init__(self, port: int = 8080):
        self.port = port
        self._latest_raw: Optional[bytes] = None
        self._latest_mask: Optional[bytes] = None
        self._latest_annotated: Optional[bytes] = None
        self._latest_combined: Optional[bytes] = None
        self._latest_meta: Dict[str, Any] = {"found": False, "cx": 0, "cy": 0, "area": 0, "fps": 0.0}
        self.server: Optional[socketserver.TCPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.started = False

    @classmethod
    def get_instance(cls, port: int = 8080) -> VisionDebugServer:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(port)
            return cls._instance

    @classmethod
    def update(
        cls,
        raw_frame: Optional[np.ndarray],
        mask_frame: Optional[np.ndarray],
        annotated_frame: Optional[np.ndarray],
        combined_frame: Optional[np.ndarray],
        meta: Optional[Dict[str, Any]] = None,
    ):
        inst = cls._instance
        if inst is None or not inst.started:
            return

        with cls._lock:
            if meta:
                inst._latest_meta.update(meta)
            if raw_frame is not None:
                _, buf = cv2.imencode(".jpg", raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                inst._latest_raw = buf.tobytes()
            if mask_frame is not None:
                _, buf = cv2.imencode(".jpg", mask_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                inst._latest_mask = buf.tobytes()
            if annotated_frame is not None:
                _, buf = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                inst._latest_annotated = buf.tobytes()
            if combined_frame is not None:
                _, buf = cv2.imencode(".jpg", combined_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                inst._latest_combined = buf.tobytes()

    def serve_image(self, handler, jpeg_bytes: Optional[bytes], placeholder_label: str):
        if jpeg_bytes is None:
            blank = np.zeros((300, 400, 3), dtype=np.uint8)
            cv2.putText(blank, placeholder_label, (25, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 2)
            cv2.putText(blank, "Awaiting Camera...", (25, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
            _, buf = cv2.imencode(".jpg", blank)
            jpeg_bytes = buf.tobytes()

        handler.send_response(200)
        handler.send_header("Content-Type", "image/jpeg")
        handler.send_header("Content-Length", str(len(jpeg_bytes)))
        handler.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()
        handler.wfile.write(jpeg_bytes)

    def stream_mjpeg(self, handler, stream_type: str = "combined"):
        """Serve continuous real-time MJPEG video stream over HTTP multipart."""
        try:
            handler.send_response(200)
            handler.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            handler.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            handler.send_header("Pragma", "no-cache")
            handler.send_header("Access-Control-Allow-Origin", "*")
            handler.end_headers()
        except Exception:
            return

        last_bytes = None
        while self.started:
            try:
                with self._lock:
                    if stream_type == "raw":
                        cur_bytes = self._latest_raw
                    elif stream_type == "mask":
                        cur_bytes = self._latest_mask
                    elif stream_type == "detection":
                        cur_bytes = self._latest_annotated
                    else:
                        cur_bytes = self._latest_combined

                if cur_bytes is not None and cur_bytes != last_bytes:
                    last_bytes = cur_bytes
                    header = (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(cur_bytes)).encode("ascii") + b"\r\n\r\n"
                    )
                    handler.wfile.write(header + cur_bytes + b"\r\n")
                    handler.wfile.flush()

                time.sleep(0.04)  # ~25 FPS max stream output rate
            except (BrokenPipeError, ConnectionResetError, socket.error):
                break
            except Exception:
                break

    def start(self):
        if self.started:
            return
        try:
            ThreadedTCPServer.allow_reuse_address = True
            self.server = ThreadedTCPServer(("", self.port), VisionDebugHandler)
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()
            self.started = True
            logger.info(">>> [VISION DEBUG] Live Web MJPEG Stream active at http://0.0.0.0:%d/ <<<", self.port)
        except Exception as e:
            logger.warning("Could not start Vision Debug Web Server on port %d: %s", self.port, e)

    def get_dashboard_html(self) -> str:
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MIDI Robot - Live Vision Debug Stream</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #121214; color: #e1e1e6; margin: 0; padding: 20px; }
        h1 { margin: 0 0 10px 0; font-size: 22px; color: #00d26a; }
        .meta-bar { background: #1e1e24; padding: 12px 18px; border-radius: 8px; margin-bottom: 20px; display: flex; gap: 20px; align-items: center; border: 1px solid #333; flex-wrap: wrap; }
        .badge { padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 14px; }
        .badge-live { background: #0088ff; color: #fff; }
        .badge-found { background: #00d26a; color: #000; }
        .badge-none { background: #e02424; color: #fff; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 20px; }
        .card { background: #1a1a20; border-radius: 8px; overflow: hidden; border: 1px solid #2d2d38; }
        .card-header { padding: 10px 14px; font-weight: 600; font-size: 15px; background: #22222c; border-bottom: 1px solid #333; display: flex; justify-content: space-between; }
        .card img { width: 100%; height: auto; display: block; background: #0a0a0c; min-height: 240px; }
        .full-width { grid-column: 1 / -1; }
        .btn { background: #00d26a; color: #000; border: none; padding: 6px 14px; border-radius: 4px; font-weight: 600; cursor: pointer; }
    </style>
</head>
<body>
    <h1>MIDI Robot: Live Wrist Camera & Segmentation Debugger</h1>
    <div class="meta-bar">
        <div>Video: <span class="badge badge-live">LIVE MJPEG</span></div>
        <div>Detection: <span id="status-badge" class="badge badge-none">SEARCHING FLOOR...</span></div>
        <div>Rate: <span id="fps-text" style="font-weight: bold; color: #00d26a;">-- FPS</span></div>
        <div>Offset: <span id="offset-text">eX: 0.00, eY: 0.00</span></div>
        <div>Angle: <span id="angle-text">0.0°</span></div>
        <div>Cloth Area: <span id="area-text">0 px</span></div>
        <div style="margin-left: auto;">
            <button class="btn" onclick="reloadStreams()">Reconnect Streams</button>
        </div>
    </div>

    <div class="grid">
        <div class="card full-width">
            <div class="card-header">
                <span>Multi-View Live Stream (Actual Raw | Segmented Mask | Annotated Reticle)</span>
                <a href="/stream/combined" target="_blank" style="color: #00d26a; text-decoration: none;">Direct Stream</a>
            </div>
            <img id="img-combined" src="/stream/combined" alt="Combined Multi-View Stream">
        </div>
        <div class="card">
            <div class="card-header">
                <span>1. Actual Camera Frame (Live)</span>
                <a href="/stream/raw" target="_blank" style="color: #00d26a; text-decoration: none;">Direct Stream</a>
            </div>
            <img id="img-raw" src="/stream/raw" alt="Actual Camera Stream">
        </div>
        <div class="card">
            <div class="card-header">
                <span>2. Segmented Mask (Live Filter)</span>
                <a href="/stream/mask" target="_blank" style="color: #00d26a; text-decoration: none;">Direct Stream</a>
            </div>
            <img id="img-mask" src="/stream/mask" alt="Segmented Mask Stream">
        </div>
        <div class="card">
            <div class="card-header">
                <span>3. Detection Overlay (Grasp Target)</span>
                <a href="/stream/detection" target="_blank" style="color: #00d26a; text-decoration: none;">Direct Stream</a>
            </div>
            <img id="img-detection" src="/stream/detection" alt="Annotated Detection Stream">
        </div>
    </div>

    <script>
        function checkStatus() {
            fetch('/status.json')
                .then(r => r.json())
                .then(d => {
                    const b = document.getElementById('status-badge');
                    if (d.found) {
                        b.className = 'badge badge-found';
                        b.innerText = 'SOCK DETECTED';
                    } else {
                        b.className = 'badge badge-none';
                        b.innerText = 'SEARCHING FLOOR...';
                    }
                    document.getElementById('offset-text').innerText = 'eX: ' + (d.error_x >= 0 ? '+' : '') + Number(d.error_x || 0).toFixed(2) + ', eY: ' + (d.error_y >= 0 ? '+' : '') + Number(d.error_y || 0).toFixed(2);
                    document.getElementById('angle-text').innerText = (d.angle !== undefined ? (d.angle >= 0 ? '+' : '') + Number(d.angle || 0).toFixed(1) + '°' : '--');
                    document.getElementById('area-text').innerText = Math.round(d.area || 0) + ' px';
                    if (d.fps !== undefined) {
                        document.getElementById('fps-text').innerText = Number(d.fps).toFixed(1) + ' FPS';
                    }
                })
                .catch(() => {});
        }

        function reloadStreams() {
            const t = Date.now();
            document.getElementById('img-combined').src = '/stream/combined?t=' + t;
            document.getElementById('img-raw').src = '/stream/raw?t=' + t;
            document.getElementById('img-mask').src = '/stream/mask?t=' + t;
            document.getElementById('img-detection').src = '/stream/detection?t=' + t;
        }

        setInterval(checkStatus, 300);
    </script>
</body>
</html>"""


class WristCamera:
    """Manages wrist camera video capture with fallback, background streaming, and dry-run simulation."""

    def __init__(
        self,
        device_path: Optional[str] = "/dev/video0",
        fallback_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        dry_run: bool = False,
    ):
        self.device_path = device_path or "/dev/video0"
        self.fallback_index = fallback_index
        self.width = width
        self.height = height
        self.fps = fps
        self.dry_run = dry_run
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_opened = False

        # Live background stream worker
        self._streaming = False
        self._stream_thread: Optional[threading.Thread] = None
        self._frame_lock = threading.Lock()
        self._latest_raw_frame: Optional[np.ndarray] = None
        self._detector: Optional[Any] = None
        self._target_fps: float = 20.0
        self._measured_fps: float = 0.0

    def connect(self) -> bool:
        """Initialize camera capture."""
        if self.dry_run:
            logger.info("[DRY RUN] Wrist camera initialized in simulation.")
            self.is_opened = True
            return True

        candidates = []
        if self.device_path:
            candidates.append(self.device_path)
            if "video" in str(self.device_path):
                try:
                    candidates.append(int(str(self.device_path).split("video")[-1]))
                except ValueError:
                    pass
        if self.fallback_index not in candidates:
            candidates.append(self.fallback_index)
        for alt in [0, 1, "/dev/video0", "/dev/video1"]:
            if alt not in candidates:
                candidates.append(alt)

        for target in candidates:
            try:
                # First try with CAP_V4L2 if on Linux, otherwise default
                backend = cv2.CAP_V4L2 if (hasattr(cv2, "CAP_V4L2") and os.name == "posix") else cv2.CAP_ANY
                self.cap = cv2.VideoCapture(target, backend)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(target)

                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                    for _ in range(5):
                        self.cap.read()
                    self.is_opened = True
                    logger.info("Wrist camera opened successfully on %s (%dx%d @ %d FPS)", str(target), self.width, self.height, self.fps)
                    return True
            except Exception as e:
                logger.debug("Failed opening camera candidate %s: %s", target, e)

        self.is_opened = False
        logger.error("Could not open wrist camera on any candidate: %s", candidates)
        return False

    def _generate_synthetic_frame(self) -> np.ndarray:
        """Generate a realistic synthetic camera frame with simulated sock."""
        frame = np.full((self.height, self.width, 3), 150, dtype=np.uint8)
        noise = np.random.randint(-15, 15, (self.height, self.width, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        cv2.ellipse(frame, (340, 220), (45, 25), 20, 0, 360, (30, 30, 40), -1)
        cv2.fillPoly(frame, [np.array([[180, 480], [240, 340], [280, 360], [220, 480]])], (180, 150, 0))
        cv2.fillPoly(frame, [np.array([[460, 480], [400, 340], [360, 360], [420, 480]])], (180, 150, 0))
        return frame

    def start_stream(self, detector: Optional[Any] = None, target_fps: float = 20.0):
        """Start continuous background video capture and live detection stream."""
        if self._streaming:
            if detector is not None:
                self._detector = detector
            return

        if not self.is_opened:
            if not self.connect():
                logger.warning("Cannot start camera live stream: camera failed to open.")
                return

        self._detector = detector
        self._target_fps = target_fps
        self._streaming = True
        self._stream_thread = threading.Thread(target=self._stream_worker, daemon=True)
        self._stream_thread.start()
        logger.info(">>> Wrist camera background live stream worker started (~%.1f FPS) <<<", self._target_fps)

    def stop_stream(self):
        """Stop background capture worker."""
        self._streaming = False
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=1.0)
            self._stream_thread = None

    def _stream_worker(self):
        """Background thread continuously pulling frames and updating debug stream at high FPS."""
        interval = 1.0 / max(1.0, self._target_fps)
        frame_count = 0
        fps_t0 = time.time()

        while self._streaming:
            loop_start = time.time()
            if self.dry_run:
                frame = self._generate_synthetic_frame()
                ret = True
            else:
                if self.cap is None or not self.cap.isOpened():
                    time.sleep(0.1)
                    continue
                ret, frame = self.cap.read()

            if ret and frame is not None:
                with self._frame_lock:
                    self._latest_raw_frame = frame

                frame_count += 1
                fps_elapsed = time.time() - fps_t0
                if fps_elapsed >= 1.0:
                    self._measured_fps = frame_count / fps_elapsed
                    frame_count = 0
                    fps_t0 = time.time()

                # Process detection continuously to update the web debugger in real time
                if self._detector is not None:
                    try:
                        self._detector.detect(frame)
                        srv = VisionDebugServer._instance
                        if srv and srv.started:
                            with srv._lock:
                                srv._latest_meta["fps"] = round(self._measured_fps, 1)
                    except Exception as e:
                        logger.debug("Background detector stream error: %s", e)

            elapsed = time.time() - loop_start
            sleep_t = interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    def capture_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame: immediately returns latest buffered frame if streaming, or queries cap."""
        if self._streaming:
            with self._frame_lock:
                if self._latest_raw_frame is not None:
                    return True, self._latest_raw_frame.copy()

        if self.dry_run:
            return True, self._generate_synthetic_frame()

        if not self.is_opened or self.cap is None:
            if not self.connect():
                return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            logger.warning("Wrist camera failed to capture frame!")
            return False, None

        with self._frame_lock:
            self._latest_raw_frame = frame
        return True, frame

    def release(self):
        """Cleanly release camera resources."""
        self.stop_stream()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_opened = False
        logger.debug("Wrist camera released.")


class SockDetector:
    """Robust sock detector using color segmentation, contrast saliency, and geometric filtering."""

    def __init__(
        self,
        mode: str = "auto",
        min_area: float = 1000.0,
        max_area: float = 80000.0,
        contrast_threshold: float = 38.0,
        border_margin: int = 12,
        save_debug_image: bool = True,
        debug_image_path: str = "/tmp/sock_detection.jpg",
        debug_raw_path: str = "/tmp/sock_raw.jpg",
        debug_mask_path: str = "/tmp/sock_mask.jpg",
        debug_combined_path: str = "/tmp/sock_combined.jpg",
        enable_web_stream: bool = True,
        web_stream_port: int = 8080,
        roi_top_pct: float = 0.10,
        roi_bottom_pct: float = 0.68,
        roi_left_pct: float = 0.12,
        roi_right_pct: float = 0.88,
    ):
        self.mode = mode.lower()
        self.min_area = min_area
        self.max_area = max_area
        self.contrast_threshold = contrast_threshold
        self.border_margin = border_margin
        self.save_debug_image = save_debug_image
        self.debug_image_path = debug_image_path
        self.debug_raw_path = debug_raw_path
        self.debug_mask_path = debug_mask_path
        self.debug_combined_path = debug_combined_path
        self.enable_web_stream = enable_web_stream
        self.web_stream_port = web_stream_port
        self.roi_top_pct = roi_top_pct
        self.roi_bottom_pct = roi_bottom_pct
        self.roi_left_pct = roi_left_pct
        self.roi_right_pct = roi_right_pct

        if self.enable_web_stream:
            server = VisionDebugServer.get_instance(self.web_stream_port)
            server.start()

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Process image frame to locate the sock on the floor."""
        if frame is None or frame.size == 0:
            return DetectionResult(found=False)

        h, w = frame.shape[:2]
        center_x = w / 2.0
        center_y = h / 2.0

        # 1. Define active floor Region of Interest (ROI)
        x_min = int(w * self.roi_left_pct)
        x_max = int(w * self.roi_right_pct)
        y_min = int(h * self.roi_top_pct)
        y_max = int(h * self.roi_bottom_pct)

        roi = frame[y_min:y_max, x_min:x_max]
        roi_h, roi_w = roi.shape[:2]

        # 2. Mask out Cyan/Teal gripper fingers (Hue 75-115 in OpenCV HSV)
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        cyan_mask = cv2.inRange(hsv_roi, np.array([75, 70, 70]), np.array([115, 255, 255]))

        # 3. Create foreground candidate mask
        if self.mode == "contrast" or self.mode == "auto":
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (9, 9), 0)
            
            # Detect objects darker or brighter than median floor background
            floor_median = np.median(blurred)
            diff = np.abs(blurred.astype(np.float32) - floor_median).astype(np.uint8)
            _, thresh = cv2.threshold(diff, self.contrast_threshold, 255, cv2.THRESH_BINARY)
        else:
            # Color mode: general dark / colored object mask
            sat = hsv_roi[:, :, 1]
            val = hsv_roi[:, :, 2]
            thresh = cv2.bitwise_or(
                cv2.inRange(sat, 50, 255),
                cv2.inRange(val, 0, 70)
            )

        # Exclude cyan gripper areas
        thresh = cv2.bitwise_and(thresh, cv2.bitwise_not(cyan_mask))

        # Morphological filtering to clean noise and bridge cloth folds
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

        # 4. Contour detection
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_contour = None
        best_area = 0.0
        best_center = (0.0, 0.0)
        best_bbox = (0, 0, 0, 0)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area <= area <= self.max_area:
                bx, by, bw, bh = cv2.boundingRect(cnt)

                # RULE A: Discard contours touching the ROI boundary (vignetting, lens borders, background objects)
                if (
                    bx <= self.border_margin
                    or by <= self.border_margin
                    or (bx + bw) >= (roi_w - self.border_margin)
                    or (by + bh) >= (roi_h - self.border_margin)
                ):
                    continue

                # RULE B: Discard abnormal aspect ratios (thin vertical/horizontal edge shadows or scratches)
                aspect = float(bw) / max(1, bh)
                if aspect < 0.25 or aspect > 4.0:
                    continue

                # RULE C: Discard non-compact contours (stringy noise or floor seams)
                extent = area / float(max(1, bw * bh))
                if extent < 0.25:
                    continue

                # Find centroid using image moments
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = float(M["m10"] / M["m00"]) + x_min
                    cy = float(M["m01"] / M["m00"]) + y_min
                    global_bx = bx + x_min
                    global_by = by + y_min

                    # Prioritize larger valid cloth contours
                    if area > best_area:
                        best_area = area
                        best_contour = cnt
                        best_center = (cx, cy)
                        best_bbox = (global_bx, global_by, bw, bh)

        # 5. Build full-frame segmented mask representation
        mask_full = np.zeros((h, w), dtype=np.uint8)
        mask_full[y_min:y_max, x_min:x_max] = cleaned
        mask_bgr = cv2.cvtColor(mask_full, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(mask_bgr, (x_min, y_min), (x_max, y_max), (70, 70, 70), 1, cv2.LINE_AA)

        # 6. Build annotation overlay
        annotated = frame.copy()
        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (100, 100, 100), 1, cv2.LINE_AA)
        cv2.drawMarker(annotated, (int(center_x), int(center_y)), (0, 255, 255), cv2.MARKER_CROSS, 20, 1)

        sock_angle = 0.0
        if best_contour is not None:
            cx, cy = best_center
            bx, by, bw, bh = best_bbox

            # Normalized error from center: [-1.0, 1.0]
            error_x = (cx - center_x) / center_x
            error_y = (cy - center_y) / center_y

            # Compute oriented bounding box and orientation angle (long axis)
            try:
                rect = cv2.minAreaRect(best_contour)
                (rcx, rcy), (rw, rh), rangle = rect
                if rw < rh:
                    raw_angle = rangle + 90.0
                else:
                    raw_angle = rangle
                while raw_angle > 90.0:
                    raw_angle -= 180.0
                while raw_angle < -90.0:
                    raw_angle += 180.0
                sock_angle = float(raw_angle)

                # Draw oriented bounding box in cyan
                box_pts = cv2.boxPoints(rect)
                box_pts = np.intp(box_pts) + np.array([x_min, y_min])
                cv2.drawContours(annotated, [box_pts], 0, (255, 255, 0), 2)
            except Exception:
                sock_angle = 0.0

            # Draw detection bounding box & centroid on annotated frame
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            cv2.circle(annotated, (int(cx), int(cy)), 6, (0, 0, 255), -1)
            cv2.line(annotated, (int(center_x), int(center_y)), (int(cx), int(cy)), (0, 255, 0), 2)

            label = f"SOCK: {int(best_area)}px [eX:{error_x:+.2f}, eY:{error_y:+.2f}, {sock_angle:+.1f}deg]"
            cv2.putText(annotated, label, (bx, max(20, by - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # Also draw bounding box and center on mask for visual clarity
            cv2.rectangle(mask_bgr, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            cv2.circle(mask_bgr, (int(cx), int(cy)), 6, (0, 0, 255), -1)

            result = DetectionResult(
                found=True,
                cx=cx,
                cy=cy,
                error_x=error_x,
                error_y=error_y,
                angle=sock_angle,
                area=best_area,
                bbox=best_bbox,
                annotated_frame=annotated,
                raw_frame=frame.copy(),
                mask_frame=mask_bgr,
            )
        else:
            cv2.putText(annotated, "NO SOCK DETECTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            result = DetectionResult(
                found=False,
                annotated_frame=annotated,
                raw_frame=frame.copy(),
                mask_frame=mask_bgr,
            )

        # 7. Build 3-panel combined debug frame: [ Actual Raw | Segmented Mask | Annotated Detection ]
        p_raw = frame.copy()
        cv2.putText(p_raw, "ACTUAL CAMERA", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        p_mask = mask_bgr.copy()
        cv2.putText(p_mask, "SEGMENTED MASK", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        p_ann = annotated.copy()
        cv2.putText(p_ann, "DETECTION & RETICLE", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        rw, rh = 426, 320
        c_raw = cv2.resize(p_raw, (rw, rh))
        c_mask = cv2.resize(p_mask, (rw, rh))
        c_ann = cv2.resize(p_ann, (rw, rh))
        combined = np.hstack([c_raw, c_mask, c_ann])
        result.combined_frame = combined

        # 8. Save debug images to disk
        if self.save_debug_image:
            try:
                if self.debug_image_path:
                    cv2.imwrite(self.debug_image_path, annotated)
                if self.debug_raw_path:
                    cv2.imwrite(self.debug_raw_path, frame)
                if self.debug_mask_path:
                    cv2.imwrite(self.debug_mask_path, mask_bgr)
                if self.debug_combined_path:
                    cv2.imwrite(self.debug_combined_path, combined)

                # Also save easy-access files in current workspace
                cv2.imwrite("debug_actual.jpg", frame)
                cv2.imwrite("debug_segmented.jpg", mask_bgr)
                cv2.imwrite("debug_combined.jpg", combined)
            except Exception as e:
                logger.debug("Error saving debug images: %s", e)

        # 9. Update live HTTP web stream
        if self.enable_web_stream:
            VisionDebugServer.update(
                raw_frame=frame,
                mask_frame=mask_bgr,
                annotated_frame=annotated,
                combined_frame=combined,
                meta={
                    "found": result.found,
                    "cx": result.cx,
                    "cy": result.cy,
                    "error_x": result.error_x,
                    "error_y": result.error_y,
                    "angle": round(result.angle, 1),
                    "area": result.area,
                }
            )

        return result


class VisionGraspPlanner:
    """Computes joint trajectory targets and visual servoing corrections for sock pickup and basket deposit."""

    def __init__(self, config: Dict[str, Any]):
        c4 = config.get("case4", {})
        serv = c4.get("servoing", {})
        self.pan_gain = float(serv.get("pan_gain", 22.0))
        self.lift_gain = float(serv.get("lift_gain", 15.0))
        self.align_tolerance = float(serv.get("align_tolerance", 0.12))
        self.grasp_verification_threshold = float(c4.get("grasp_verification_threshold_pct", 5.8))
        self.abort_on_miss = bool(c4.get("abort_on_miss", False))
        self.scan_sweep_angles = c4.get("scan_sweep_angles", [0.0, -28.0, 28.0])

        # Adaptive visual servoing parameters
        self.reach_elbow_gain = float(serv.get("reach_elbow_gain", 20.0))
        self.reach_lift_gain = float(serv.get("reach_lift_gain", 6.0))
        self.max_reach_delta_deg = float(serv.get("max_reach_delta_deg", 18.0))
        self.enable_roll_alignment = bool(serv.get("enable_roll_alignment", True))
        self.enable_hover_refinement = bool(serv.get("enable_hover_refinement", True))
        self.hover_lift_offset_deg = float(serv.get("hover_lift_offset_deg", -14.0))
        self.max_align_iterations = int(serv.get("max_align_iterations", 3))

        # Default poses
        self.scan_pose = c4.get("scan_pose", {
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": -50.0,
            "elbow_flex.pos": 35.0,
            "wrist_flex.pos": 70.0,
            "wrist_roll.pos": 0.0,
            "gripper.pos": 10.0,
        })
        # Basket is mounted directly behind the gripper centerline and slightly elevated
        self.basket_pose = c4.get("basket_pose", {
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": -80.0,
            "elbow_flex.pos": 20.0,
            "wrist_flex.pos": 50.0,
            "wrist_roll.pos": 0.0,
            "gripper.pos": 85.0,
        })
        # Calibrated floor reach to grasp sock from floor below the mobile base
        self.floor_grasp_pose = c4.get("floor_grasp_pose", {
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": -30.0,
            "elbow_flex.pos": 45.0,
            "wrist_flex.pos": 80.0,
            "wrist_roll.pos": 0.0,
            "gripper.pos": 0.0,
        })

        # Motion timing (seconds) - Slow and gentle for stability
        mot = c4.get("motion", {})
        self.scan_move_duration_s = float(mot.get("scan_move_duration_s", 1.5))
        self.descend_duration_s = float(mot.get("descend_duration_s", 2.5))
        self.clamp_duration_s = float(mot.get("clamp_duration_s", 1.2))
        self.lift_duration_s = float(mot.get("lift_duration_s", 1.5))
        self.basket_duration_s = float(mot.get("basket_duration_s", 3.2))
        self.release_duration_s = float(mot.get("release_duration_s", 0.8))
        self.stow_duration_s = float(mot.get("stow_duration_s", 2.5))

    def compute_aligned_pose(
        self,
        current_pose: Dict[str, float],
        detection: DetectionResult,
    ) -> Tuple[Dict[str, float], bool]:
        """Compute corrected pose to center gripper over sock."""
        if not detection.found:
            return dict(current_pose), False

        aligned = dict(current_pose)

        # Invert pan direction matching camera horizontal axis
        pan_correction = -self.pan_gain * detection.error_x
        aligned["shoulder_pan.pos"] = float(np.clip(
            aligned.get("shoulder_pan.pos", 0.0) + pan_correction,
            -110.0, 110.0
        ))

        # Vertical offset adjusts reach (shoulder_lift / elbow_flex)
        lift_correction = self.lift_gain * detection.error_y
        aligned["shoulder_lift.pos"] = float(np.clip(
            aligned.get("shoulder_lift.pos", -50.0) + lift_correction,
            -85.0, -20.0
        ))

        is_aligned = abs(detection.error_x) <= self.align_tolerance
        return aligned, is_aligned

    def compute_grasp_pose(
        self,
        aligned_pose: Dict[str, float],
        detection: Optional[DetectionResult] = None,
        is_hover: bool = False,
        descent_lift_deg: float = 16.0,
        descent_elbow_deg: float = 12.0,
    ) -> Dict[str, float]:
        """Compute final downward descent pose to hover right at floor level below chassis base,
        dynamically adapting reach (elbow/lift) and gripper orientation (roll) to detected cloth."""
        grasp = dict(aligned_pose)
        if self.floor_grasp_pose:
            base_lift = float(self.floor_grasp_pose.get("shoulder_lift.pos", -30.0))
            base_elbow = float(self.floor_grasp_pose.get("elbow_flex.pos", 45.0))
            base_wrist = float(self.floor_grasp_pose.get("wrist_flex.pos", 80.0))
            base_roll = float(self.floor_grasp_pose.get("wrist_roll.pos", 0.0))
        else:
            base_lift = float(np.clip(grasp.get("shoulder_lift.pos", -50.0) + descent_lift_deg, -90.0, 90.0))
            base_elbow = float(np.clip(grasp.get("elbow_flex.pos", 35.0) + descent_elbow_deg, -70.0, 70.0))
            base_wrist = float(np.clip(grasp.get("wrist_flex.pos", 70.0) + 10.0, -100.0, 100.0))
            base_roll = float(grasp.get("wrist_roll.pos", 0.0))

        grasp["shoulder_lift.pos"] = base_lift
        grasp["elbow_flex.pos"] = base_elbow
        grasp["wrist_flex.pos"] = base_wrist
        grasp["wrist_roll.pos"] = base_roll

        # Dynamic adaptation if detection is provided
        if detection is not None and detection.found:
            # 1. Fine-trim pan if residual error exists
            if abs(detection.error_x) > 0.03:
                pan_trim = -self.pan_gain * 0.3 * float(detection.error_x)
                grasp["shoulder_pan.pos"] = float(np.clip(grasp.get("shoulder_pan.pos", 0.0) + pan_trim, -110.0, 110.0))

            # 2. Longitudinal reach modulation (negative error_y = sock further away)
            longitudinal_error = -float(detection.error_y)
            elbow_reach_delta = float(np.clip(self.reach_elbow_gain * longitudinal_error, -self.max_reach_delta_deg, self.max_reach_delta_deg))
            lift_reach_delta = float(np.clip(self.reach_lift_gain * longitudinal_error, -self.max_reach_delta_deg * 0.5, self.max_reach_delta_deg * 0.5))
            wrist_reach_delta = float(np.clip(-0.5 * elbow_reach_delta, -15.0, 15.0))

            grasp["elbow_flex.pos"] = float(base_elbow + elbow_reach_delta)
            grasp["shoulder_lift.pos"] = float(base_lift + lift_reach_delta)
            grasp["wrist_flex.pos"] = float(base_wrist + wrist_reach_delta)

            # 3. Gripper roll orientation adaptation
            if self.enable_roll_alignment and abs(detection.angle) > 2.0:
                roll_offset = float(np.clip(detection.angle, -60.0, 60.0))
                grasp["wrist_roll.pos"] = float(np.clip(base_roll + roll_offset, -180.0, 180.0))

        # If hover pose: elevate shoulder and open gripper wide
        if is_hover:
            grasp["shoulder_lift.pos"] = float(grasp["shoulder_lift.pos"] + self.hover_lift_offset_deg)
            grasp["gripper.pos"] = 85.0
        else:
            grasp["gripper.pos"] = 0.0  # Clamp target

        return grasp

    def verify_grasp(self, measured_gripper_pct: float) -> bool:
        """Verify if sock is caught between jaws while elevated off floor.
        
        When elevated:
        - If empty, jaws close fully to < 8%.
        - If holding cloth, jaws are blocked at >= 12%.
        """
        return measured_gripper_pct >= self.grasp_verification_threshold

