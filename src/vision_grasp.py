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
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.kinematics import SO100Kinematics

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
    hsv_h_frame: Optional[np.ndarray] = None
    hsv_s_frame: Optional[np.ndarray] = None
    hsv_v_frame: Optional[np.ndarray] = None
    hsv_sv_inv_frame: Optional[np.ndarray] = None
    hsv_channels_frame: Optional[np.ndarray] = None
    gripper_mask_frame: Optional[np.ndarray] = None
    hsv_stats: Dict[str, Any] = field(default_factory=dict)


def update_config_hsv(
    config_path: str,
    hsv_lower: list[int],
    hsv_upper: list[int],
    gripper_hsv_lower: Optional[list[int]] = None,
    gripper_hsv_upper: Optional[list[int]] = None,
    mode: Optional[str] = None,
    contrast_threshold: Optional[float] = None,
    v_max: Optional[int] = None,
    max_floor_ratio: Optional[float] = None,
    min_solidity: Optional[float] = None,
    min_area: Optional[float] = None,
    max_area: Optional[float] = None,
    roi_top_pct: Optional[float] = None,
    roi_bottom_pct: Optional[float] = None,
    roi_left_pct: Optional[float] = None,
    roi_right_pct: Optional[float] = None,
) -> bool:
    """Save calibrated detection constants and HSV ranges into config.yaml, preserving comments."""
    import re
    import yaml

    if not os.path.exists(config_path):
        logger.error("Config file %s does not exist!", config_path)
        return False

    def clamp_hsv(vals: list[int], default: list[int]) -> list[int]:
        if not vals or len(vals) < 3:
            vals = default
        return [
            max(0, min(179, int(vals[0]))),
            max(0, min(255, int(vals[1]))),
            max(0, min(255, int(vals[2]))),
        ]

    low = clamp_hsv(hsv_lower, [0, 0, 0])
    high = clamp_hsv(hsv_upper, [179, 255, int(v_max) if v_max is not None else 75])
    if v_max is not None:
        high[2] = max(0, min(255, int(v_max)))
    g_low = clamp_hsv(gripper_hsv_lower, [75, 70, 70]) if gripper_hsv_lower is not None else None
    g_high = clamp_hsv(gripper_hsv_upper, [115, 255, 255]) if gripper_hsv_upper is not None else None

    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    def replace_or_insert_list(text: str, key: str, values: list[int], comment: str) -> str:
        formatted = f"[{values[0]}, {values[1]}, {values[2]}]"
        pat = rf"(^\s*{key}:\s*)\[[^\]]*\]"
        if re.search(pat, text, flags=re.MULTILINE):
            return re.sub(pat, rf"\g<1>{formatted}", text, count=1, flags=re.MULTILINE)
        anchor = r"(^\s*contrast_threshold:[^\n]*\n)"
        line_to_insert = f"    {key}: {formatted:<20} # {comment}\n"
        if re.search(anchor, text, flags=re.MULTILINE):
            return re.sub(anchor, lambda m: m.group(1) + line_to_insert, text, count=1, flags=re.MULTILINE)
        det_anchor = r"(^\s*detection:\s*\n)"
        if re.search(det_anchor, text, flags=re.MULTILINE):
            return re.sub(det_anchor, lambda m: m.group(1) + line_to_insert, text, count=1, flags=re.MULTILINE)
        return text

    def replace_or_insert_scalar(text: str, key: str, val_str: str, comment: str) -> str:
        pat = rf"(^\s*{key}:\s*)-?[0-9.]+"
        if re.search(pat, text, flags=re.MULTILINE):
            return re.sub(pat, rf"\g<1>{val_str}", text, count=1, flags=re.MULTILINE)
        anchor = r"(^\s*contrast_threshold:[^\n]*\n)"
        line_to_insert = f"    {key}: {val_str:<20} # {comment}\n"
        if re.search(anchor, text, flags=re.MULTILINE):
            return re.sub(anchor, lambda m: m.group(1) + line_to_insert, text, count=1, flags=re.MULTILINE)
        return text

    content = replace_or_insert_list(content, "hsv_lower", low, "Target sock HSV lower bound [H_min, S_min, V_min]")
    content = replace_or_insert_list(content, "hsv_upper", high, "Target sock HSV upper bound [H_max, S_max, V_max]")
    if g_low is not None:
        content = replace_or_insert_list(content, "gripper_hsv_lower", g_low, "Cyan gripper exclusion HSV lower bound [H_min, S_min, V_min]")
    if g_high is not None:
        content = replace_or_insert_list(content, "gripper_hsv_upper", g_high, "Cyan gripper exclusion HSV upper bound [H_max, S_max, V_max]")

    if mode is not None and str(mode).strip():
        clean_mode = str(mode).strip().lower()
        pat_mode = r'(^\s*mode:\s*)"[^"]*"'
        if re.search(pat_mode, content, flags=re.MULTILINE):
            content = re.sub(pat_mode, rf'\g<1>"{clean_mode}"', content, count=1, flags=re.MULTILINE)

    if contrast_threshold is not None:
        content = replace_or_insert_scalar(content, "contrast_threshold", f"{float(contrast_threshold):.1f}", "minimum signed darkness (Floor_Median - Gray)")
    if v_max is not None:
        content = replace_or_insert_scalar(content, "v_max", f"{int(v_max)}", "maximum sock brightness cap (0-255)")
    if max_floor_ratio is not None:
        content = replace_or_insert_scalar(content, "max_floor_ratio", f"{float(max_floor_ratio):.2f}", "adaptive brightness cap ratio relative to floor median")
    if min_solidity is not None:
        content = replace_or_insert_scalar(content, "min_solidity", f"{float(min_solidity):.2f}", "minimum contour solidity (Area / ConvexHullArea)")
    if min_area is not None:
        content = replace_or_insert_scalar(content, "min_area", f"{float(min_area):.1f}", "minimum contour area (pixels)")
    if max_area is not None:
        content = replace_or_insert_scalar(content, "max_area", f"{float(max_area):.1f}", "maximum contour area (pixels)")
    if roi_top_pct is not None:
        content = replace_or_insert_scalar(content, "roi_top_pct", f"{float(roi_top_pct):.2f}", "Active floor ROI top margin")
    if roi_bottom_pct is not None:
        content = replace_or_insert_scalar(content, "roi_bottom_pct", f"{float(roi_bottom_pct):.2f}", "Active floor ROI bottom margin")
    if roi_left_pct is not None:
        content = replace_or_insert_scalar(content, "roi_left_pct", f"{float(roi_left_pct):.2f}", "Active floor ROI left margin")
    if roi_right_pct is not None:
        content = replace_or_insert_scalar(content, "roi_right_pct", f"{float(roi_right_pct):.2f}", "Active floor ROI right margin")

    try:
        parsed = yaml.safe_load(content) or {}
        det_sec = parsed.setdefault("case4", {}).setdefault("detection", {})
        if det_sec.get("hsv_lower") != low or det_sec.get("hsv_upper") != high:
            det_sec["hsv_lower"] = low
            det_sec["hsv_upper"] = high
            if g_low is not None:
                det_sec["gripper_hsv_lower"] = g_low
            if g_high is not None:
                det_sec["gripper_hsv_upper"] = g_high
            if mode is not None:
                det_sec["mode"] = str(mode).strip().lower()
            if contrast_threshold is not None:
                det_sec["contrast_threshold"] = round(float(contrast_threshold), 1)
            if v_max is not None:
                det_sec["v_max"] = int(v_max)
            if max_floor_ratio is not None:
                det_sec["max_floor_ratio"] = round(float(max_floor_ratio), 2)
            if min_solidity is not None:
                det_sec["min_solidity"] = round(float(min_solidity), 2)
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(parsed, f, default_flow_style=False, sort_keys=False)
            return True
    except Exception as e:
        logger.error("Failed validating updated YAML: %s", e)
        return False

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(content)
    return True


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """Multi-threaded TCP server to handle simultaneous HTTP requests and persistent MJPEG streams."""
    daemon_threads = True
    allow_reuse_address = True


class VisionDebugHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler serving real-time MJPEG streams, locked sock snapshots, and interactive calibration/scan API."""

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
        elif path == "/stream/last_detected":
            srv.stream_mjpeg(self, "last_detected")
        elif path == "/stream/hsv_h":
            srv.stream_mjpeg(self, "hsv_h")
        elif path == "/stream/hsv_s":
            srv.stream_mjpeg(self, "hsv_s")
        elif path == "/stream/hsv_v":
            srv.stream_mjpeg(self, "hsv_v")
        elif path == "/stream/hsv_sv_inv":
            srv.stream_mjpeg(self, "hsv_sv_inv")
        elif path == "/stream/hsv_channels":
            srv.stream_mjpeg(self, "hsv_channels")
        elif path == "/stream/gripper_mask":
            srv.stream_mjpeg(self, "gripper_mask")
        elif path == "/raw.jpg":
            srv.serve_image(self, srv._latest_raw, "LIVE CAMERA FRAME")
        elif path in ("/mask.jpg", "/segmented.jpg"):
            srv.serve_image(self, srv._latest_mask, "MASKED CAMERA VIEW")
        elif path in ("/detection.jpg", "/annotated.jpg"):
            srv.serve_image(self, srv._latest_annotated, "SOCK DETECTION VIEW")
        elif path == "/last_detected.jpg":
            srv.serve_image(self, srv._latest_locked_detection, "AWAITING SOCK DETECTION...")
        elif path in ("/combined.jpg", "/debug.jpg"):
            srv.serve_image(self, srv._latest_combined, "COMBINED DEBUG VIEW")
        elif path == "/hsv_h.jpg":
            srv.serve_image(self, srv._latest_hsv_h, "HUE (H) CHANNEL & MASK")
        elif path == "/hsv_s.jpg":
            srv.serve_image(self, srv._latest_hsv_s, "SATURATION (S) CHANNEL & MASK")
        elif path == "/hsv_v.jpg":
            srv.serve_image(self, srv._latest_hsv_v, "VALUE (V) CHANNEL & MASK")
        elif path == "/hsv_sv_inv.jpg":
            srv.serve_image(self, srv._latest_hsv_sv_inv, "INVERTED V * S COMBINED")
        elif path == "/hsv_channels.jpg":
            srv.serve_image(self, srv._latest_hsv_channels, "SEPARATE H / S / V CHANNELS")
        elif path == "/gripper_mask.jpg":
            srv.serve_image(self, srv._latest_gripper_mask, "GRIPPER EXCLUSION MASK")
        elif path in ("/status.json", "/hsv.json"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            with srv._lock:
                meta = dict(srv._latest_meta)
                meta["locked_detection"] = dict(srv._latest_locked_meta)
            if srv.detector is not None:
                meta["hsv_config"] = srv.detector.get_hsv_config()
            self.wfile.write(json.dumps(meta).encode("utf-8"))
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        srv = VisionDebugServer._instance
        if not srv:
            self.send_error(503, "Vision debug server not initialized")
            return

        path = self.path.split("?")[0]
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(raw_body.decode("utf-8"))
        except Exception as e:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode("utf-8"))
            return

        if path == "/hsv.json":
            saved = False
            det = srv.detector
            if det is not None:
                det.apply_hsv_update(payload)
                # Re-run detection immediately on cached raw frame so streams reflect slider change right away
                if srv._latest_raw_ndarray is not None:
                    try:
                        det.detect(srv._latest_raw_ndarray.copy())
                    except Exception:
                        pass

                if payload.get("save"):
                    cfg_target = payload.get("config_path") or srv.config_path or "config.yaml"
                    saved = update_config_hsv(
                        config_path=cfg_target,
                        hsv_lower=list(det.hsv_lower),
                        hsv_upper=list(det.hsv_upper),
                        gripper_hsv_lower=list(det.gripper_hsv_lower),
                        gripper_hsv_upper=list(det.gripper_hsv_upper),
                        mode=det.mode,
                        contrast_threshold=det.contrast_threshold,
                        v_max=det.v_max,
                        max_floor_ratio=det.max_floor_ratio,
                        min_solidity=det.min_solidity,
                        min_area=det.min_area,
                        max_area=det.max_area,
                        roi_top_pct=det.roi_top_pct,
                        roi_bottom_pct=det.roi_bottom_pct,
                        roi_left_pct=det.roi_left_pct,
                        roi_right_pct=det.roi_right_pct,
                    )

            resp = {
                "ok": True,
                "saved": saved,
                "hsv_config": det.get_hsv_config() if det is not None else {},
            }
            with srv._lock:
                resp["hsv_stats"] = srv._latest_meta.get("hsv_stats", {})

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        elif path == "/scan.json":
            resp = srv.execute_web_scan_action(payload)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        else:
            self.send_error(404, "Not Found")


class VisionDebugServer:
    """Lightweight HTTP server publishing live MJPEG streams, locked sock snapshots, and interactive debug/scan UI."""
    _instance: Optional[VisionDebugServer] = None
    _lock = threading.Lock()

    def __init__(self, port: int = 8080):
        self.port = port
        self.detector: Optional[Any] = None
        self.arm: Optional[Any] = None
        self.camera: Optional[Any] = None
        self.planner: Optional[Any] = None
        self.base: Optional[Any] = None
        self._scan_lock = threading.Lock()
        self.config_path: str = "config.yaml"
        self._latest_raw_ndarray: Optional[np.ndarray] = None
        self._latest_raw: Optional[bytes] = None
        self._latest_mask: Optional[bytes] = None
        self._latest_annotated: Optional[bytes] = None
        self._latest_locked_detection: Optional[bytes] = None
        self._latest_locked_meta: Dict[str, Any] = {
            "has_locked": False,
            "label": "No sock locked yet",
            "area": 0,
            "r_cm": None,
            "pan_deg": None,
            "timestamp": "--",
        }
        self._latest_combined: Optional[bytes] = None
        self._latest_hsv_h: Optional[bytes] = None
        self._latest_hsv_s: Optional[bytes] = None
        self._latest_hsv_v: Optional[bytes] = None
        self._latest_hsv_sv_inv: Optional[bytes] = None
        self._latest_hsv_channels: Optional[bytes] = None
        self._latest_gripper_mask: Optional[bytes] = None
        self._latest_meta: Dict[str, Any] = {"found": False, "cx": 0, "cy": 0, "area": 0, "fps": 0.0, "hsv_stats": {}}
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
    def lock_detection(
        cls,
        annotated_frame: Optional[np.ndarray],
        label: str = "Detected",
        coords: Optional[Tuple[float, float, float]] = None,
        area: float = 0.0,
        cx: float = 0.0,
        cy: float = 0.0,
        meta: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """Lock and store the exact annotated camera frame when a sock is detected by the gripper/camera."""
        if annotated_frame is None:
            return
        inst = cls._instance if cls._instance is not None else cls.get_instance()
        if meta:
            label = str(meta.get("label", label))
            area = float(meta.get("area", area))
            cx = float(meta.get("cx", cx))
            cy = float(meta.get("cy", cy))
            if coords is None and isinstance(meta.get("coords"), (list, tuple)) and len(meta["coords"]) >= 3:
                coords = (float(meta["coords"][0]), float(meta["coords"][1]), float(meta["coords"][2]))

        snap = annotated_frame.copy()
        h, w = snap.shape[:2]
        is_fail = "FAIL" in label.upper() or "NO SOCK" in label.upper()
        banner_bg = (25, 25, 145) if is_fail else (15, 75, 30)
        cv2.rectangle(snap, (0, 0), (w, 36), banner_bg, -1)
        ts = time.strftime("%H:%M:%S")
        if is_fail:
            banner = f"{label} [{ts}] | Returning to normal position"
        elif coords is not None:
            r_m, pan_deg, _ = coords
            banner = f"CAPTURED SOCK @ {label} [{ts}] | R={r_m*100:.1f}cm, Pan={pan_deg:+.1f}deg | {int(area)}px"
        else:
            banner = f"CAPTURED SOCK @ {label} [{ts}] | Pos=({int(cx)},{int(cy)}) | Area={int(area)}px"
        cv2.putText(snap, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

        try:
            cv2.imwrite("/tmp/last_sock_detection.jpg", snap)
        except Exception:
            pass

        _, buf = cv2.imencode(".jpg", snap, [cv2.IMWRITE_JPEG_QUALITY, 85])
        with cls._lock:
            inst._latest_locked_detection = buf.tobytes()
            inst._latest_locked_meta = {
                "has_locked": True,
                "label": label,
                "area": round(float(area), 0),
                "cx": round(float(cx), 1),
                "cy": round(float(cy), 1),
                "r_cm": round(float(coords[0]) * 100.0, 1) if coords is not None else None,
                "pan_deg": round(float(coords[1]), 1) if coords is not None else None,
                "timestamp": ts,
            }

    def execute_web_scan_action(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle interactive Scan or Pose buttons from the Web Debug Screen."""
        action = str(payload.get("action", "sweep")).strip().lower()
        if not self._scan_lock.acquire(blocking=False):
            return {"ok": False, "error": "A scan or arm motion is already in progress."}
        try:
            if action == "snapshot":
                if self._latest_raw_ndarray is not None and self.detector is not None:
                    det = self.detector.detect(self._latest_raw_ndarray.copy(), force_save=True)
                    if det.annotated_frame is not None:
                        self.lock_detection(
                            det.annotated_frame,
                            label="Manual Snapshot",
                            area=det.area,
                            cx=det.cx,
                            cy=det.cy,
                        )
                    return {"ok": True, "found": det.found, "area": det.area, "message": "Captured snapshot."}
                return {"ok": False, "error": "No camera frame available."}

            # Ensure arm and planner are initialized for physical scan motions
            if self.planner is None:
                import yaml
                cfg = {}
                if os.path.exists(self.config_path):
                    with open(self.config_path, "r", encoding="utf-8") as f:
                        cfg = yaml.safe_load(f) or {}
                self.planner = VisionGraspPlanner(cfg)

            if self.arm is None:
                import yaml
                from src.arm_controller import ArmController
                cfg = {}
                if os.path.exists(self.config_path):
                    with open(self.config_path, "r", encoding="utf-8") as f:
                        cfg = yaml.safe_load(f) or {}
                arm_cfg = cfg.setdefault("arm", {})
                f_port = arm_cfg.get("follower_port", "")
                f_fb = arm_cfg.get("follower_fallback_port", "/dev/ttyACM0")
                if f_port and not os.path.exists(f_port) and os.path.exists(f_fb):
                    arm_cfg["follower_port"] = f_fb
                self.arm = ArmController(cfg, dry_run=False)

            if not self.arm.is_follower_connected():
                if not self.arm.connect_follower():
                    return {"ok": False, "error": f"Could not connect to Follower Arm on {self.arm.follower_port}"}

            if action == "pose":
                target_name = str(payload.get("target", "center")).strip().lower()
                if target_name == "stow":
                    self.arm.move_to_stow(duration_s=1.8)
                    return {"ok": True, "message": "Arm moved to Stow Pose."}
                scan_targets = dict(self.planner.scan_poses or {})
                pose_map = {
                    "center": ("Middle (Center)", scan_targets.get("center", self.planner.scan_pose)),
                    "left": ("Left", scan_targets.get("left", self.planner.scan_pose)),
                    "right": ("Right", scan_targets.get("right", self.planner.scan_pose)),
                }
                label, target_pose = pose_map.get(target_name, ("Middle (Center)", self.planner.scan_pose))
                cmd_pose = dict(target_pose)
                cmd_pose["gripper.pos"] = 85.0
                self.arm.move_to_pose(cmd_pose, duration_s=self.planner.scan_move_duration_s)
                time.sleep(0.35)
                if self.camera is not None and self.detector is not None:
                    ok, frame = self.camera.capture_frame(max_staleness_s=0.20, min_timestamp=time.time())
                    if ok and frame is not None:
                        det = self.detector.detect(frame, force_save=True)
                        if det.found and det.annotated_frame is not None:
                            meas = self.arm.get_follower_pose()
                            coords = self.planner.compute_target_coordinates(
                                det,
                                scan_pan_deg=meas.get("shoulder_pan.pos", cmd_pose.get("shoulder_pan.pos", 0.0)),
                                scan_pose=meas,
                            )
                            self.lock_detection(det.annotated_frame, label=label, coords=coords, area=det.area, cx=det.cx, cy=det.cy)
                            return {
                                "ok": True,
                                "found": True,
                                "label": label,
                                "area": round(det.area, 0),
                                "r_cm": round(coords[0] * 100.0, 1),
                                "pan_deg": round(coords[1], 1),
                                "message": f"Moved to {label}: SOCK DETECTED ({int(det.area)}px, R={coords[0]*100:.1f}cm, Pan={coords[1]:+.1f}°)",
                            }
                        return {"ok": True, "found": False, "label": label, "message": f"Moved to {label} Scan Pose (no sock in view)."}
                return {"ok": True, "found": False, "label": label, "message": f"Moved to {label} Scan Pose."}

            # Default action: "sweep" (3-Point Floor Scan Sweep across Middle, Left, Right)
            best = self.planner.execute_scan_sweep(
                arm=self.arm,
                camera=self.camera,
                detector=self.detector,
                open_gripper_pct=85.0,
            )
            if best is not None:
                det = best["det"]
                r_t, pan_t, roll_t = best["coords"]
                return {
                    "ok": True,
                    "found": True,
                    "label": best["label"],
                    "area": round(det.area, 0),
                    "r_cm": round(r_t * 100.0, 1),
                    "pan_deg": round(pan_t, 1),
                    "roll_deg": round(roll_t, 1),
                    "message": f"SOCK DETECTED at {best['label']}! Area={int(det.area)}px | Target R={r_t*100:.1f}cm, Pan={pan_t:+.1f}°",
                }
            return {"ok": True, "found": False, "message": "3-Point Scan complete: No sock detected on floor."}
        except Exception as e:
            logger.warning("Web scan action error: %s", e)
            return {"ok": False, "error": str(e)}
        finally:
            self._scan_lock.release()

    @classmethod
    def update(
        cls,
        raw_frame: Optional[np.ndarray],
        mask_frame: Optional[np.ndarray],
        annotated_frame: Optional[np.ndarray],
        combined_frame: Optional[np.ndarray],
        meta: Optional[Dict[str, Any]] = None,
        hsv_h_frame: Optional[np.ndarray] = None,
        hsv_s_frame: Optional[np.ndarray] = None,
        hsv_v_frame: Optional[np.ndarray] = None,
        hsv_sv_inv_frame: Optional[np.ndarray] = None,
        hsv_channels_frame: Optional[np.ndarray] = None,
        gripper_mask_frame: Optional[np.ndarray] = None,
    ):
        inst = cls._instance
        if inst is None or not inst.started:
            return

        with cls._lock:
            if meta:
                inst._latest_meta.update(meta)
            if raw_frame is not None:
                inst._latest_raw_ndarray = raw_frame
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
            if hsv_h_frame is not None:
                _, buf = cv2.imencode(".jpg", hsv_h_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                inst._latest_hsv_h = buf.tobytes()
            if hsv_s_frame is not None:
                _, buf = cv2.imencode(".jpg", hsv_s_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                inst._latest_hsv_s = buf.tobytes()
            if hsv_v_frame is not None:
                _, buf = cv2.imencode(".jpg", hsv_v_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                inst._latest_hsv_v = buf.tobytes()
            if hsv_sv_inv_frame is not None:
                _, buf = cv2.imencode(".jpg", hsv_sv_inv_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                inst._latest_hsv_sv_inv = buf.tobytes()
            if hsv_channels_frame is not None:
                _, buf = cv2.imencode(".jpg", hsv_channels_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                inst._latest_hsv_channels = buf.tobytes()
            if gripper_mask_frame is not None:
                _, buf = cv2.imencode(".jpg", gripper_mask_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                inst._latest_gripper_mask = buf.tobytes()

        # Automatically update the captured sock snapshot whenever a sock is detected
        if meta and meta.get("found") and annotated_frame is not None:
            cls.lock_detection(
                annotated_frame=annotated_frame,
                label="Live Detection",
                area=float(meta.get("area", 0.0)),
                cx=float(meta.get("cx", 0.0)),
                cy=float(meta.get("cy", 0.0)),
            )

    def serve_image(self, handler, jpeg_bytes: Optional[bytes], placeholder_label: str):
        if jpeg_bytes is None:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, placeholder_label, (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (140, 140, 140), 2)
            cv2.putText(blank, "Will capture automatically when sock is detected", (40, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 110), 2)
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
        last_send_time = time.time()
        while self.started:
            try:
                with self._lock:
                    if stream_type == "raw":
                        cur_bytes = self._latest_raw
                    elif stream_type == "mask":
                        cur_bytes = self._latest_mask
                    elif stream_type == "detection":
                        cur_bytes = self._latest_annotated
                    elif stream_type == "last_detected":
                        cur_bytes = self._latest_locked_detection
                        if cur_bytes is None:
                            blank = np.zeros((480, 640, 3), dtype=np.uint8)
                            cv2.putText(blank, "AWAITING SOCK DETECTION...", (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (140, 140, 140), 2)
                            cv2.putText(blank, "Snapshot locks automatically when sock is detected", (40, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 110), 1)
                            _, buf = cv2.imencode(".jpg", blank)
                            cur_bytes = buf.tobytes()
                    elif stream_type == "hsv_h":
                        cur_bytes = self._latest_hsv_h
                    elif stream_type == "hsv_s":
                        cur_bytes = self._latest_hsv_s
                    elif stream_type == "hsv_v":
                        cur_bytes = self._latest_hsv_v
                    elif stream_type == "hsv_sv_inv":
                        cur_bytes = self._latest_hsv_sv_inv
                    elif stream_type == "hsv_channels":
                        cur_bytes = self._latest_hsv_channels
                    elif stream_type == "gripper_mask":
                        cur_bytes = self._latest_gripper_mask
                    else:
                        cur_bytes = self._latest_combined

                now = time.time()
                # Send if new frame or keepalive heartbeat every 0.6s
                if cur_bytes is not None and (cur_bytes != last_bytes or (now - last_send_time >= 0.6)):
                    last_bytes = cur_bytes
                    last_send_time = now
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
            logger.info(">>> [VISION DEBUG] Live Web Debug & Scan Screen active at http://0.0.0.0:%d/ <<<", self.port)
        except Exception as e:
            logger.warning("Could not start Vision Debug Web Server on port %d: %s", self.port, e)

    def get_dashboard_html(self) -> str:
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MIDI Robot - Vision Debug &amp; Floor Scan Screen</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #121214; color: #e1e1e6; margin: 0; padding: 18px; }
        h1 { margin: 0 0 10px 0; font-size: 22px; color: #00d26a; }
        .meta-bar { background: #1e1e24; padding: 12px 16px; border-radius: 8px; margin-bottom: 14px; display: flex; gap: 18px; align-items: center; border: 1px solid #333; flex-wrap: wrap; }
        .badge { padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 13px; }
        .badge-live { background: #0088ff; color: #fff; }
        .badge-found { background: #00d26a; color: #000; }
        .badge-none { background: #e02424; color: #fff; }
        .scan-bar { background: #18222c; border: 1px solid #294866; border-radius: 8px; padding: 12px 16px; margin-bottom: 14px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
        .calib-panel { background: #1a1a22; border: 1px solid #323242; border-radius: 8px; padding: 14px 16px; margin-bottom: 16px; }
        .calib-header { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 12px; border-bottom: 1px solid #2d2d3a; padding-bottom: 10px; }
        .calib-cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }
        .slider-group { background: #14141a; border: 1px solid #282834; border-radius: 6px; padding: 10px 12px; }
        .slider-group h4 { margin: 0 0 10px 0; font-size: 14px; color: #00d26a; }
        .slider-row { display: grid; grid-template-columns: 155px 1fr 64px; gap: 8px; align-items: center; margin-bottom: 8px; font-size: 13px; }
        .slider-row-dual { display: grid; grid-template-columns: 95px 1fr 52px 1fr 52px; gap: 8px; align-items: center; margin-bottom: 8px; font-size: 13px; }
        .slider-row input[type=range], .slider-row-dual input[type=range] { width: 100%; accent-color: #00d26a; cursor: pointer; }
        .slider-row input[type=number], .slider-row-dual input[type=number] { width: 54px; background: #22222c; color: #fff; border: 1px solid #444; border-radius: 4px; padding: 3px 5px; font-family: monospace; text-align: center; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(440px, 1fr)); gap: 16px; }
        .card { background: #1a1a20; border-radius: 8px; overflow: hidden; border: 1px solid #2d2d38; }
        .card-header { padding: 9px 14px; font-weight: 600; font-size: 14px; background: #22222c; border-bottom: 1px solid #333; display: flex; justify-content: space-between; align-items: center; }
        .card img { width: 100%; height: auto; display: block; background: #0a0a0c; min-height: 240px; }
        .clickable-img { cursor: crosshair; }
        .btn { background: #00d26a; color: #000; border: none; padding: 7px 14px; border-radius: 4px; font-weight: 700; cursor: pointer; font-size: 13px; }
        .btn-sec { background: #2b2b38; color: #e1e1e6; border: 1px solid #454558; font-weight: 600; }
        .btn-blue { background: #0088ff; color: #fff; }
        .btn-gold { background: #ffcc00; color: #000; }
        .stat-pill { background: #121216; border: 1px solid #343444; border-radius: 5px; padding: 4px 9px; font-family: monospace; font-size: 13px; }
        #save-toast, #scan-status { font-size: 13px; font-weight: 600; color: #00d26a; }
    </style>
</head>
<body>
    <h1>MIDI Robot: Live Wrist Camera, Darkness Mask &amp; Floor Scan Debugger</h1>

    <div class="meta-bar">
        <div>Detection: <span id="status-badge" class="badge badge-none">SEARCHING FLOOR...</span></div>
        <div>Rate: <span id="fps-text" style="font-weight: bold; color: #00d26a;">-- FPS</span></div>
        <div class="stat-pill">Floor Med V: <strong id="floor-v">--</strong></div>
        <div class="stat-pill">Probe(x,y) V: <strong id="probe-v">--</strong> (DarkDiff: <strong id="probe-diff">--</strong>)</div>
        <div class="stat-pill">Sock V: <strong id="sock-v">--</strong> | Solidity: <strong id="sock-sol">--</strong></div>
        <div class="stat-pill">Offset: <span id="offset-text">eX: 0.00, eY: 0.00</span></div>
        <div class="stat-pill">Area: <strong id="area-text">0 px</strong></div>
        <div style="margin-left: auto;">
            <button class="btn btn-sec" onclick="reloadStreams()">Reconnect Streams</button>
        </div>
    </div>

    <!-- FLOOR SCAN & ARM POSE BAR -->
    <div class="scan-bar">
        <strong style="color:#5ac8fa; font-size:14px;">Floor Scan Actions:</strong>
        <button class="btn btn-gold" onclick="runScanAction({action:'sweep'})">&#128269; Run 3-Point Floor Scan Sweep</button>
        <button class="btn btn-sec" onclick="runScanAction({action:'pose', target:'center'})">Move to Middle Scan</button>
        <button class="btn btn-sec" onclick="runScanAction({action:'pose', target:'left'})">Move to Left Scan</button>
        <button class="btn btn-sec" onclick="runScanAction({action:'pose', target:'right'})">Move to Right Scan</button>
        <button class="btn btn-sec" onclick="runScanAction({action:'pose', target:'stow'})">Park / Stow Arm</button>
        <span id="scan-status" style="margin-left: 8px;">Ready</span>
    </div>

    <!-- CONSTANTS TUNING PANEL -->
    <div class="calib-panel">
        <div class="calib-header">
            <div>
                <strong style="color:#00d26a; font-size:15px;">Signed Floor-Relative Darkness &amp; Grasp Constants</strong>
                <span style="font-size:12px; color:#9da3b4; margin-left:10px;">(Click anywhere on Live Camera to inspect pixel brightness &amp; darkness diff)</span>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
                <button class="btn btn-sec" onclick="resetProbeCenter()">Center Probe</button>
                <button class="btn btn-blue" onclick="pushConstantsUpdate(true)">&#128190; Save Constants to config.yaml</button>
                <span id="save-toast"></span>
            </div>
        </div>

        <div class="calib-cols">
            <div class="slider-group">
                <h4>1. Black Sock Darkness &amp; Shape Constants</h4>
                <div class="slider-row">
                    <span><strong>Darkness Diff (&ge;)</strong></span>
                    <input type="range" id="contrast-th" min="5" max="140" step="1" value="35" oninput="syncInput('contrast-th','contrast-th-num'); pushConstantsUpdate(false)">
                    <input type="number" id="contrast-th-num" min="5" max="140" step="1" value="35" onchange="syncInput('contrast-th-num','contrast-th'); pushConstantsUpdate(false)">
                </div>
                <div class="slider-row">
                    <span><strong>Max Brightness Cap (V_max)</strong></span>
                    <input type="range" id="v-max" min="10" max="220" step="1" value="75" oninput="syncInput('v-max','v-max-num'); pushConstantsUpdate(false)">
                    <input type="number" id="v-max-num" min="10" max="220" step="1" value="75" onchange="syncInput('v-max-num','v-max'); pushConstantsUpdate(false)">
                </div>
                <div class="slider-row">
                    <span><strong>Max Floor Ratio (0.2-0.95)</strong></span>
                    <input type="range" id="floor-ratio" min="0.20" max="0.95" step="0.05" value="0.65" oninput="syncInput('floor-ratio','floor-ratio-num'); pushConstantsUpdate(false)">
                    <input type="number" id="floor-ratio-num" min="0.20" max="0.95" step="0.05" value="0.65" onchange="syncInput('floor-ratio-num','floor-ratio'); pushConstantsUpdate(false)">
                </div>
                <div class="slider-row">
                    <span><strong>Min Solidity (Compactness)</strong></span>
                    <input type="range" id="min-solidity" min="0.10" max="0.90" step="0.05" value="0.45" oninput="syncInput('min-solidity','min-solidity-num'); pushConstantsUpdate(false)">
                    <input type="number" id="min-solidity-num" min="0.10" max="0.90" step="0.05" value="0.45" onchange="syncInput('min-solidity-num','min-solidity'); pushConstantsUpdate(false)">
                </div>
                <div class="slider-row">
                    <span><strong>Min Sock Area (px)</strong></span>
                    <input type="range" id="min-area" min="100" max="8000" step="50" value="800" oninput="syncInput('min-area','min-area-num'); pushConstantsUpdate(false)">
                    <input type="number" id="min-area-num" min="100" max="8000" step="50" value="800" onchange="syncInput('min-area-num','min-area'); pushConstantsUpdate(false)">
                </div>
            </div>

            <div class="slider-group">
                <h4>2. Floor ROI Margins &amp; Cyan Gripper Exclusion <span id="grip-pass" style="float:right; font-weight:normal; color:#aaa; font-size:12px;">Gripper Masked: --%</span></h4>
                <div class="slider-row-dual">
                    <span><strong>ROI Top/Bot</strong></span>
                    <input type="range" id="roi-top" min="0.01" max="0.45" step="0.01" value="0.06" oninput="syncInput('roi-top','roi-top-num'); pushConstantsUpdate(false)">
                    <input type="number" id="roi-top-num" min="0.01" max="0.45" step="0.01" value="0.06" onchange="syncInput('roi-top-num','roi-top'); pushConstantsUpdate(false)">
                    <input type="range" id="roi-bot" min="0.50" max="0.98" step="0.01" value="0.86" oninput="syncInput('roi-bot','roi-bot-num'); pushConstantsUpdate(false)">
                    <input type="number" id="roi-bot-num" min="0.50" max="0.98" step="0.01" value="0.86" onchange="syncInput('roi-bot-num','roi-bot'); pushConstantsUpdate(false)">
                </div>
                <div class="slider-row-dual">
                    <span><strong>Grip H</strong></span>
                    <input type="range" id="gh-min" min="0" max="179" value="77" oninput="syncInput('gh-min','gh-min-num'); pushConstantsUpdate(false)">
                    <input type="number" id="gh-min-num" min="0" max="179" value="77" onchange="syncInput('gh-min-num','gh-min'); pushConstantsUpdate(false)">
                    <input type="range" id="gh-max" min="0" max="179" value="113" oninput="syncInput('gh-max','gh-max-num'); pushConstantsUpdate(false)">
                    <input type="number" id="gh-max-num" min="0" max="179" value="113" onchange="syncInput('gh-max-num','gh-max'); pushConstantsUpdate(false)">
                </div>
                <div class="slider-row-dual">
                    <span><strong>Grip S</strong></span>
                    <input type="range" id="gs-min" min="0" max="255" value="137" oninput="syncInput('gs-min','gs-min-num'); pushConstantsUpdate(false)">
                    <input type="number" id="gs-min-num" min="0" max="255" value="137" onchange="syncInput('gs-min-num','gs-min'); pushConstantsUpdate(false)">
                    <input type="range" id="gs-max" min="0" max="255" value="237" oninput="syncInput('gs-max','gs-max-num'); pushConstantsUpdate(false)">
                    <input type="number" id="gs-max-num" min="0" max="255" value="237" onchange="syncInput('gs-max-num','gs-max'); pushConstantsUpdate(false)">
                </div>
                <div class="slider-row-dual">
                    <span><strong>Grip V</strong></span>
                    <input type="range" id="gv-min" min="0" max="255" value="131" oninput="syncInput('gv-min','gv-min-num'); pushConstantsUpdate(false)">
                    <input type="number" id="gv-min-num" min="0" max="255" value="131" onchange="syncInput('gv-min-num','gv-min'); pushConstantsUpdate(false)">
                    <input type="range" id="gv-max" min="0" max="255" value="231" oninput="syncInput('gv-max','gv-max-num'); pushConstantsUpdate(false)">
                    <input type="number" id="gv-max-num" min="0" max="255" value="231" onchange="syncInput('gv-max-num','gv-max'); pushConstantsUpdate(false)">
                </div>
            </div>
        </div>
    </div>

    <!-- 4 CAMERA & DETECTION VIEWS -->
    <div class="grid">
        <div class="card">
            <div class="card-header">
                <span>1. Live Camera Feed (Click Any Pixel to Probe)</span>
                <a href="/stream/raw" target="_blank" style="color: #00d26a; text-decoration: none;">Direct Stream</a>
            </div>
            <img id="img-raw" class="clickable-img" src="/stream/raw" alt="Live Camera Feed" onclick="onImageClick(event, this)" onerror="setTimeout(reloadStreams, 2000)">
        </div>

        <div class="card">
            <div class="card-header">
                <span>2. Masked Camera View (Live Darkness + Gripper Filter)</span>
                <a href="/stream/mask" target="_blank" style="color: #00d26a; text-decoration: none;">Direct Stream</a>
            </div>
            <img id="img-mask" class="clickable-img" src="/stream/mask" alt="Masked Camera View" onclick="onImageClick(event, this)" onerror="setTimeout(reloadStreams, 2000)">
        </div>

        <div class="card">
            <div class="card-header">
                <span>3. Sock Detected Camera View (Live Grasp Target &amp; Core Circle)</span>
                <a href="/stream/detection" target="_blank" style="color: #00d26a; text-decoration: none;">Direct Stream</a>
            </div>
            <img id="img-detection" class="clickable-img" src="/stream/detection" alt="Sock Detected View" onclick="onImageClick(event, this)" onerror="setTimeout(reloadStreams, 2000)">
        </div>

        <div class="card" style="border: 2px solid #00d26a;">
            <div class="card-header" style="background: #142e20;">
                <span style="color:#00d26a;"><strong>4. Captured Sock Detected Snapshot</strong> <span id="locked-meta-text" style="font-weight:normal; color:#e1e1e6; font-size:12px;">(Locks when sock is detected)</span></span>
                <a href="/last_detected.jpg" target="_blank" style="color: #00d26a; text-decoration: none;">Snapshot JPG</a>
            </div>
            <img id="img-last-detected" src="/stream/last_detected" alt="Last Captured Sock Detection" onerror="setTimeout(reloadStreams, 2000)">
        </div>
    </div>

    <script>
        let slidersInitialized = false;

        function syncInput(srcId, dstId) {
            document.getElementById(dstId).value = document.getElementById(srcId).value;
        }

        function setSliderPair(baseId, val) {
            const s = document.getElementById(baseId);
            const n = document.getElementById(baseId + '-num');
            if (s && n && val !== undefined && val !== null) {
                s.value = val;
                n.value = val;
            }
        }

        function applyConfigToSliders(cfg) {
            if (!cfg) return;
            setSliderPair('contrast-th', cfg.contrast_threshold);
            setSliderPair('v-max', cfg.v_max);
            setSliderPair('floor-ratio', cfg.max_floor_ratio);
            setSliderPair('min-solidity', cfg.min_solidity);
            setSliderPair('min-area', cfg.min_area);
            setSliderPair('roi-top', cfg.roi_top_pct);
            setSliderPair('roi-bot', cfg.roi_bottom_pct);
            if (cfg.gripper_hsv_lower && cfg.gripper_hsv_lower.length === 3) {
                setSliderPair('gh-min', cfg.gripper_hsv_lower[0]);
                setSliderPair('gs-min', cfg.gripper_hsv_lower[1]);
                setSliderPair('gv-min', cfg.gripper_hsv_lower[2]);
            }
            if (cfg.gripper_hsv_upper && cfg.gripper_hsv_upper.length === 3) {
                setSliderPair('gh-max', cfg.gripper_hsv_upper[0]);
                setSliderPair('gs-max', cfg.gripper_hsv_upper[1]);
                setSliderPair('gv-max', cfg.gripper_hsv_upper[2]);
            }
        }

        function pushConstantsUpdate(saveToDisk) {
            const vmax = Number(document.getElementById('v-max-num').value || 75);
            const payload = {
                mode: 'auto',
                contrast_threshold: Number(document.getElementById('contrast-th-num').value || 35),
                v_max: vmax,
                max_floor_ratio: Number(document.getElementById('floor-ratio-num').value || 0.65),
                min_solidity: Number(document.getElementById('min-solidity-num').value || 0.45),
                min_area: Number(document.getElementById('min-area-num').value || 800),
                roi_top_pct: Number(document.getElementById('roi-top-num').value || 0.06),
                roi_bottom_pct: Number(document.getElementById('roi-bot-num').value || 0.86),
                hsv_upper: [179, 255, vmax],
                gripper_hsv_lower: [
                    Number(document.getElementById('gh-min-num').value),
                    Number(document.getElementById('gs-min-num').value),
                    Number(document.getElementById('gv-min-num').value)
                ],
                gripper_hsv_upper: [
                    Number(document.getElementById('gh-max-num').value),
                    Number(document.getElementById('gs-max-num').value),
                    Number(document.getElementById('gv-max-num').value)
                ],
                save: Boolean(saveToDisk)
            };
            fetch('/hsv.json', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            })
            .then(r => r.json())
            .then(res => {
                if (saveToDisk) {
                    const t = document.getElementById('save-toast');
                    t.innerText = res.saved ? 'Saved to config.yaml!' : 'Failed to save';
                    setTimeout(() => { t.innerText = ''; }, 3000);
                }
            })
            .catch(() => {});
        }

        function runScanAction(payload) {
            const st = document.getElementById('scan-status');
            st.style.color = '#ffcc00';
            st.innerText = 'Executing arm scan / motion...';
            fetch('/scan.json', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            })
            .then(r => r.json())
            .then(res => {
                if (!res.ok) {
                    st.style.color = '#ff6b6b';
                    st.innerText = 'Error: ' + (res.error || 'Failed');
                } else {
                    st.style.color = res.found ? '#00d26a' : '#5ac8fa';
                    st.innerText = res.message || 'Done';
                }
            })
            .catch(err => {
                st.style.color = '#ff6b6b';
                st.innerText = 'Request failed';
            });
        }

        function onImageClick(evt, imgElem) {
            const rect = imgElem.getBoundingClientRect();
            const rx = Math.max(0, Math.min(1, (evt.clientX - rect.left) / Math.max(1, rect.width)));
            const ry = Math.max(0, Math.min(1, (evt.clientY - rect.top) / Math.max(1, rect.height)));
            fetch('/hsv.json', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({probe_rx: rx, probe_ry: ry})
            });
        }

        function resetProbeCenter() {
            fetch('/hsv.json', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({probe_rx: 0.5, probe_ry: 0.5})
            });
        }

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
                    document.getElementById('area-text').innerText = Math.round(d.area || 0) + ' px';
                    if (d.fps !== undefined) {
                        document.getElementById('fps-text').innerText = Number(d.fps).toFixed(1) + ' FPS';
                    }
                    if (!slidersInitialized && d.hsv_config) {
                        applyConfigToSliders(d.hsv_config);
                        slidersInitialized = true;
                    }
                    const hs = d.hsv_stats || {};
                    if (hs.floor_median !== undefined) {
                        document.getElementById('floor-v').innerText = Math.round(hs.floor_median);
                    }
                    if (hs.probe_v !== undefined) {
                        document.getElementById('probe-v').innerText = hs.probe_v;
                        document.getElementById('probe-diff').innerText = hs.probe_dark_diff !== undefined ? hs.probe_dark_diff : '--';
                    }
                    if (hs.sock_v_mean !== undefined && hs.sock_v_mean !== null) {
                        document.getElementById('sock-v').innerText = Math.round(hs.sock_v_mean);
                        document.getElementById('sock-sol').innerText = hs.sock_solidity !== undefined && hs.sock_solidity !== null ? Number(hs.sock_solidity).toFixed(2) : '--';
                    } else {
                        document.getElementById('sock-v').innerText = '--';
                        document.getElementById('sock-sol').innerText = '--';
                    }
                    if (hs.gripper_pass_pct !== undefined) {
                        document.getElementById('grip-pass').innerText = 'Gripper Masked: ' + hs.gripper_pass_pct + '%';
                    }
                    const lk = d.locked_detection || {};
                    if (lk.has_locked) {
                        let txt = '[' + lk.timestamp + '] ' + lk.label + ' (' + lk.area + 'px)';
                        if (lk.r_cm !== null && lk.r_cm !== undefined) {
                            txt += ' -> R=' + lk.r_cm + 'cm, Pan=' + (lk.pan_deg >= 0 ? '+' : '') + lk.pan_deg + '°';
                        }
                        document.getElementById('locked-meta-text').innerText = txt;
                    }
                })
                .catch(() => {});
        }

        function reloadStreams() {
            const t = Date.now();
            document.getElementById('img-raw').src = '/stream/raw?t=' + t;
            document.getElementById('img-mask').src = '/stream/mask?t=' + t;
            document.getElementById('img-detection').src = '/stream/detection?t=' + t;
            document.getElementById('img-last-detected').src = '/stream/last_detected?t=' + t;
        }

        setInterval(checkStatus, 300);
    </script>
</body>
</html>"""


def discover_camera_indices() -> List[int]:
    """Discover all valid V4L2 video capture devices (excluding RPi codec nodes and metadata)."""
    indices = []
    v4l_dir = "/sys/class/video4linux"
    if os.path.exists(v4l_dir):
        for node in sorted(os.listdir(v4l_dir), key=lambda x: int(x.replace("video", "")) if x.replace("video", "").isdigit() else 999):
            if not node.startswith("video"):
                continue
            name_path = os.path.join(v4l_dir, node, "name")
            idx_path = os.path.join(v4l_dir, node, "index")
            if os.path.exists(name_path):
                try:
                    with open(name_path, "r") as f:
                        name = f.read().strip()
                    if "bcm2835" in name.lower():
                        continue
                    idx = 0
                    if os.path.exists(idx_path):
                        with open(idx_path, "r") as f:
                            idx = int(f.read().strip())
                    if idx == 0:
                        indices.append(int(node.replace("video", "")))
                except Exception:
                    pass
    return indices


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

        # Live background stream worker & health monitoring
        self._streaming = False
        self._stream_thread: Optional[threading.Thread] = None
        self._frame_lock = threading.Lock()
        self._latest_raw_frame: Optional[np.ndarray] = None
        self._latest_frame_time: float = 0.0
        self._consecutive_failures: int = 0
        self._detector: Optional[Any] = None
        self._target_fps: float = 20.0
        self._measured_fps: float = 0.0
        self._last_reconnect_time: float = 0.0

    def connect(self) -> bool:
        """Initialize camera capture with resilient auto-discovery and fallback."""
        if self.dry_run:
            logger.info("[DRY RUN] Wrist camera initialized in simulation.")
            self.is_opened = True
            with self._frame_lock:
                self._latest_raw_frame = self._generate_synthetic_frame()
                self._latest_frame_time = time.time()
            return True

        candidates = []
        if isinstance(self.device_path, str) and "video" in self.device_path:
            try:
                candidates.append(int(self.device_path.split("video")[-1]))
            except ValueError:
                pass
        elif isinstance(self.device_path, int):
            candidates.append(self.device_path)

        # Dynamically discover all active UVC cameras on the system
        for discovered in discover_camera_indices():
            if discovered not in candidates:
                candidates.append(discovered)

        if self.fallback_index not in candidates:
            candidates.append(self.fallback_index)
        if 0 not in candidates:
            candidates.append(0)

        # Filter out 1 (metadata node on Linux UVC)
        filtered_candidates = [c for c in candidates if c != 1]
        if not filtered_candidates:
            filtered_candidates = candidates

        for target in filtered_candidates:
            try:
                # Use CAP_V4L2 directly with integer index for 40ms fast open without FFMPEG hangs
                backend = cv2.CAP_V4L2 if (hasattr(cv2, "CAP_V4L2") and os.name == "posix") else cv2.CAP_ANY
                self.cap = cv2.VideoCapture(target, backend)

                if self.cap.isOpened():
                    try:
                        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
                        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                    # Warm up camera buffer
                    ret = False
                    for _ in range(5):
                        ret, frame = self.cap.read()
                        if ret and frame is not None:
                            with self._frame_lock:
                                self._latest_raw_frame = frame
                                self._latest_frame_time = time.time()

                    if ret:
                        self.is_opened = True
                        self._consecutive_failures = 0
                        logger.info("Wrist camera opened successfully on index %s (%dx%d @ %d FPS)", str(target), self.width, self.height, self.fps)
                        return True
                    else:
                        self.cap.release()
                        self.cap = None
            except Exception as e:
                logger.debug("Failed opening camera candidate %s: %s", target, e)
                if self.cap is not None:
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None

        self.is_opened = False
        logger.error("Could not open wrist camera on any candidate: %s", filtered_candidates)
        return False

    def _reconnect_camera(self) -> bool:
        """Safely reset and reconnect the camera device with rate-limiting and dynamic port recovery."""
        if self.dry_run:
            return True

        now = time.time()
        if now - getattr(self, "_last_reconnect_time", 0.0) < 5.0:
            return False
        self._last_reconnect_time = now

        logger.warning("[CAMERA RECOVERY] Resetting and reconnecting camera hardware...")
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.is_opened = False
        time.sleep(0.2)
        ok = self.connect()

        if not ok:
            # 1. Primary hardware recovery: power-cycle camera USB port wherever plugged in via uhubctl
            try:
                logger.info("[CAMERA RECOVERY] Power cycling camera USB port via uhubctl...")
                os.system("uhubctl -s 05a3:9230 -a cycle -d 1 >/dev/null 2>&1 || sudo -n uhubctl -s 05a3:9230 -a cycle -d 1 >/dev/null 2>&1")
                time.sleep(1.2)
                ok = self.connect()
            except Exception:
                pass

        if not ok:
            # 2. Secondary fallback: usbreset by device vendor/product ID
            try:
                logger.info("[CAMERA RECOVERY] Resetting USB camera via usbreset...")
                os.system("usbreset 05a3:9230 >/dev/null 2>&1 || sudo -n usbreset 05a3:9230 >/dev/null 2>&1")
                time.sleep(0.8)
                ok = self.connect()
            except Exception:
                pass

        if ok:
            logger.info("[CAMERA RECOVERY] Camera successfully reconnected and ready!")
            self._consecutive_failures = 0
        else:
            logger.warning("[CAMERA RECOVERY] Camera reconnect failed; will backoff.")
        return ok

    def ensure_healthy(self) -> bool:
        """Verify camera is alive and producing live frames; re-initialize if stalled."""
        if self.dry_run:
            return True
        now = time.time()
        # 1. If background stream thread died, restart it
        if self._streaming:
            if self._stream_thread is None or not self._stream_thread.is_alive():
                logger.warning("[CAMERA HEALTH] Stream worker thread was dead! Restarting...")
                self._streaming = False
                self.start_stream(detector=self._detector, target_fps=self._target_fps)
                time.sleep(0.15)

            # 2. If no fresh frame in last 0.6s, reconnect
            age = now - self._latest_frame_time
            if self._latest_raw_frame is None or age > 0.6:
                logger.warning("[CAMERA HEALTH] Camera stalled (last frame %.2fs ago). Triggering recovery...", age)
                self._reconnect_camera()
                time.sleep(0.15)
            return self.is_opened
        else:
            if not self.is_opened or self.cap is None or not self.cap.isOpened():
                return self.connect()
            return True

    def _generate_synthetic_frame(self) -> np.ndarray:
        """Generate a realistic synthetic camera frame with simulated sock (unless already picked up in simulation)."""
        frame = np.full((self.height, self.width, 3), 150, dtype=np.uint8)
        noise = np.random.randint(-15, 15, (self.height, self.width, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        if not getattr(self, "_sim_sock_picked", False):
            cv2.ellipse(frame, (340, 220), (45, 25), 20, 0, 360, (30, 30, 40), -1)
        cv2.fillPoly(frame, [np.array([[180, 480], [240, 340], [280, 360], [220, 480]])], (180, 150, 0))
        cv2.fillPoly(frame, [np.array([[460, 480], [400, 340], [360, 360], [420, 480]])], (180, 150, 0))
        return frame

    def start_stream(self, detector: Optional[Any] = None, target_fps: float = 20.0):
        """Start continuous background video capture and live detection stream."""
        if self._streaming and self._stream_thread is not None and self._stream_thread.is_alive():
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
        """Background thread continuously pulling frames with auto-recovery and live stream updates."""
        base_interval = 1.0 / max(1.0, self._target_fps)
        frame_count = 0
        fps_t0 = time.time()

        while self._streaming:
            loop_start = time.time()
            frame = None
            ret = False
            web_active = bool(self._detector is not None and getattr(self._detector, "enable_web_stream", False))
            # Throttle capture to ~10 FPS when web streaming is disabled to avoid saturating the Pi 4 shared USB 2.0 hub
            interval = base_interval if web_active else max(base_interval, 0.10)

            if self.dry_run:
                frame = self._generate_synthetic_frame()
                ret = True
            else:
                try:
                    if self.cap is None or not self.cap.isOpened():
                        self._consecutive_failures += 1
                        if self._consecutive_failures >= 3:
                            self._reconnect_camera()
                        time.sleep(0.2)
                        continue

                    ret, frame = self.cap.read()
                except Exception as e:
                    logger.debug("Camera read exception: %s", e)
                    ret = False
                    frame = None

            if ret and frame is not None:
                self._consecutive_failures = 0
                now = time.time()
                with self._frame_lock:
                    self._latest_raw_frame = frame
                    self._latest_frame_time = now

                frame_count += 1
                fps_elapsed = now - fps_t0
                if fps_elapsed >= 1.0:
                    self._measured_fps = frame_count / fps_elapsed
                    frame_count = 0
                    fps_t0 = now

                # Process detection continuously to update the web debugger in real time only if web stream is enabled
                if web_active:
                    try:
                        self._detector.detect(frame)
                        srv = VisionDebugServer._instance
                        if srv and srv.started:
                            with srv._lock:
                                srv._latest_meta["fps"] = round(self._measured_fps, 1)
                    except Exception as e:
                        logger.debug("Background detector stream error: %s", e)
            else:
                self._consecutive_failures += 1
                if self._consecutive_failures >= 8:  # ~0.4s of consecutive drops
                    logger.warning("[CAMERA] Lost %d consecutive frames. Re-establishing camera connection...", self._consecutive_failures)
                    self._reconnect_camera()

            elapsed = time.time() - loop_start
            sleep_t = interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    def capture_frame(
        self,
        max_staleness_s: float = 0.25,
        min_timestamp: Optional[float] = None,
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a fresh frame, waiting for post-timestamp update if streaming or flushing V4L2 queue."""
        now = time.time()
        if self._streaming:
            # Check if thread is healthy
            if self._stream_thread is None or not self._stream_thread.is_alive():
                logger.warning("[CAMERA] Stream thread inactive. Re-launching worker...")
                self._streaming = False
                self.start_stream(detector=self._detector, target_fps=self._target_fps)

            # Wait briefly for a fresh frame captured strictly after min_timestamp (if requested)
            deadline = time.time() + 0.35
            while True:
                now = time.time()
                with self._frame_lock:
                    has_frame = self._latest_raw_frame is not None
                    after_min = (min_timestamp is None) or (self._latest_frame_time >= min_timestamp)
                    is_fresh = has_frame and after_min and ((now - self._latest_frame_time) <= max_staleness_s)
                    if is_fresh:
                        return True, self._latest_raw_frame.copy()
                if self.dry_run:
                    frame = self._generate_synthetic_frame()
                    with self._frame_lock:
                        self._latest_raw_frame = frame
                        self._latest_frame_time = time.time()
                    return True, frame.copy()
                if time.time() >= deadline:
                    break
                time.sleep(0.02)

            now = time.time()
            # If still stale/older than min_timestamp, grab directly from hardware
            logger.warning("[CAMERA] Buffered frame is stale (age: %.2fs). Forcing fresh frame grab...", now - self._latest_frame_time)
            if not self.dry_run:
                if self.cap is None or not self.cap.isOpened():
                    self._reconnect_camera()
                if self.cap is not None and self.cap.isOpened():
                    try:
                        for _ in range(4):
                            self.cap.grab()
                        ret, frame = self.cap.read()
                        if ret and frame is not None:
                            with self._frame_lock:
                                self._latest_raw_frame = frame
                                self._latest_frame_time = time.time()
                            return True, frame
                    except Exception as e:
                        logger.debug("Direct read error: %s", e)

            # Fallback to cached frame if available
            with self._frame_lock:
                if self._latest_raw_frame is not None:
                    return True, self._latest_raw_frame.copy()

        if self.dry_run:
            return True, self._generate_synthetic_frame()

        if not self.is_opened or self.cap is None:
            if not self.connect():
                return False, None

        # Flush V4L2 ring buffer so single-shot captures never return pre-motion stale frames
        for _ in range(5):
            try:
                self.cap.grab()
            except Exception:
                break

        ret, frame = self.cap.read()
        if not ret or frame is None:
            logger.warning("Wrist camera failed to capture frame!")
            return False, None

        with self._frame_lock:
            self._latest_raw_frame = frame
            self._latest_frame_time = time.time()
        return True, frame

    def release(self):
        """Cleanly release camera resources."""
        self.stop_stream()
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.is_opened = False
        logger.debug("Wrist camera released.")

    disconnect = release


class SockDetector:
    """Robust sock detector using Signed Floor-Relative Darkness, Adaptive Brightness Cap,
    Solidity Filtering, and Distance-Transform Grasp Targeting."""

    def __init__(
        self,
        mode: str = "auto",
        min_area: float = 800.0,
        max_area: float = 80000.0,
        contrast_threshold: float = 35.0,
        v_max: int = 75,
        max_floor_ratio: float = 0.65,
        min_solidity: float = 0.45,
        border_margin: int = 4,
        save_debug_image: bool = True,
        debug_image_path: str = "/tmp/sock_detection.jpg",
        debug_raw_path: str = "/tmp/sock_raw.jpg",
        debug_mask_path: str = "/tmp/sock_mask.jpg",
        debug_combined_path: str = "/tmp/sock_combined.jpg",
        enable_web_stream: bool = False,
        web_stream_port: int = 8080,
        roi_top_pct: float = 0.06,
        roi_bottom_pct: float = 0.86,
        roi_left_pct: float = 0.04,
        roi_right_pct: float = 0.96,
        hsv_lower: Optional[list[int]] = None,
        hsv_upper: Optional[list[int]] = None,
        gripper_hsv_lower: Optional[list[int]] = None,
        gripper_hsv_upper: Optional[list[int]] = None,
        config_path: str = "config.yaml",
    ):
        self.mode = mode.lower()
        self.min_area = float(min_area)
        self.max_area = float(max_area)
        self.contrast_threshold = float(contrast_threshold)
        self.v_max = max(5, min(255, int(v_max)))
        self.max_floor_ratio = max(0.15, min(0.95, float(max_floor_ratio)))
        self.min_solidity = max(0.10, min(0.95, float(min_solidity)))
        self.border_margin = int(border_margin)
        self.save_debug_image = save_debug_image
        self.debug_image_path = debug_image_path
        self.debug_raw_path = debug_raw_path
        self.debug_mask_path = debug_mask_path
        self.debug_combined_path = debug_combined_path
        self.enable_web_stream = enable_web_stream
        self.web_stream_port = web_stream_port
        self.roi_top_pct = float(roi_top_pct)
        self.roi_bottom_pct = float(roi_bottom_pct)
        self.roi_left_pct = float(roi_left_pct)
        self.roi_right_pct = float(roi_right_pct)
        self.config_path = config_path
        self._last_disk_save_time: float = 0.0
        self._hsv_lock = threading.Lock()

        self._custom_hsv_set = (hsv_lower is not None) or (hsv_upper is not None)
        self.hsv_lower: list[int] = [
            max(0, min(179, int(hsv_lower[0]))) if hsv_lower and len(hsv_lower) >= 3 else 0,
            max(0, min(255, int(hsv_lower[1]))) if hsv_lower and len(hsv_lower) >= 3 else 0,
            max(0, min(255, int(hsv_lower[2]))) if hsv_lower and len(hsv_lower) >= 3 else 0,
        ]
        self.hsv_upper: list[int] = [
            max(0, min(179, int(hsv_upper[0]))) if hsv_upper and len(hsv_upper) >= 3 else 179,
            max(0, min(255, int(hsv_upper[1]))) if hsv_upper and len(hsv_upper) >= 3 else 255,
            max(0, min(255, int(hsv_upper[2]))) if hsv_upper and len(hsv_upper) >= 3 else 70,
        ]
        self.gripper_hsv_lower: list[int] = [
            max(0, min(179, int(gripper_hsv_lower[0]))) if gripper_hsv_lower and len(gripper_hsv_lower) >= 3 else 75,
            max(0, min(255, int(gripper_hsv_lower[1]))) if gripper_hsv_lower and len(gripper_hsv_lower) >= 3 else 70,
            max(0, min(255, int(gripper_hsv_lower[2]))) if gripper_hsv_lower and len(gripper_hsv_lower) >= 3 else 70,
        ]
        self.gripper_hsv_upper: list[int] = [
            max(0, min(179, int(gripper_hsv_upper[0]))) if gripper_hsv_upper and len(gripper_hsv_upper) >= 3 else 115,
            max(0, min(255, int(gripper_hsv_upper[1]))) if gripper_hsv_upper and len(gripper_hsv_upper) >= 3 else 255,
            max(0, min(255, int(gripper_hsv_upper[2]))) if gripper_hsv_upper and len(gripper_hsv_upper) >= 3 else 255,
        ]

        # Probe point for inspecting individual pixel values (normalized [0..1] or pixel coords)
        self.probe_rx: float = 0.5
        self.probe_ry: float = 0.5
        self.probe_px: Optional[Tuple[int, int]] = None

        if self.enable_web_stream:
            server = VisionDebugServer.get_instance(self.web_stream_port)
            server.detector = self
            server.config_path = self.config_path
            server.start()

    def get_hsv_config(self) -> Dict[str, Any]:
        """Return current detection & HSV threshold configuration."""
        with self._hsv_lock:
            return {
                "mode": self.mode,
                "contrast_threshold": round(float(self.contrast_threshold), 1),
                "v_max": int(self.v_max),
                "max_floor_ratio": round(float(self.max_floor_ratio), 2),
                "min_solidity": round(float(self.min_solidity), 2),
                "min_area": round(float(self.min_area), 1),
                "max_area": round(float(self.max_area), 1),
                "roi_top_pct": round(float(self.roi_top_pct), 2),
                "roi_bottom_pct": round(float(self.roi_bottom_pct), 2),
                "roi_left_pct": round(float(self.roi_left_pct), 2),
                "roi_right_pct": round(float(self.roi_right_pct), 2),
                "hsv_lower": list(self.hsv_lower),
                "hsv_upper": list(self.hsv_upper),
                "gripper_hsv_lower": list(self.gripper_hsv_lower),
                "gripper_hsv_upper": list(self.gripper_hsv_upper),
                "probe_rx": round(float(self.probe_rx), 4),
                "probe_ry": round(float(self.probe_ry), 4),
            }

    def apply_hsv_update(self, payload: Dict[str, Any]):
        """Update live detection constants, HSV ranges, ROI margins, or probe coordinates from web UI or CLI."""
        with self._hsv_lock:
            if "mode" in payload and payload["mode"]:
                self.mode = str(payload["mode"]).strip().lower()
            if "contrast_threshold" in payload and payload["contrast_threshold"] is not None:
                self.contrast_threshold = max(1.0, min(255.0, float(payload["contrast_threshold"])))
            if "v_max" in payload and payload["v_max"] is not None:
                self.v_max = max(5, min(255, int(float(payload["v_max"]))))
            if "max_floor_ratio" in payload and payload["max_floor_ratio"] is not None:
                self.max_floor_ratio = max(0.15, min(0.95, float(payload["max_floor_ratio"])))
            if "min_solidity" in payload and payload["min_solidity"] is not None:
                self.min_solidity = max(0.10, min(0.95, float(payload["min_solidity"])))
            if "min_area" in payload and payload["min_area"] is not None:
                self.min_area = max(50.0, min(50000.0, float(payload["min_area"])))
            if "max_area" in payload and payload["max_area"] is not None:
                self.max_area = max(500.0, min(300000.0, float(payload["max_area"])))
            if "roi_top_pct" in payload and payload["roi_top_pct"] is not None:
                self.roi_top_pct = max(0.0, min(0.50, float(payload["roi_top_pct"])))
            if "roi_bottom_pct" in payload and payload["roi_bottom_pct"] is not None:
                self.roi_bottom_pct = max(0.50, min(1.0, float(payload["roi_bottom_pct"])))
            if "roi_left_pct" in payload and payload["roi_left_pct"] is not None:
                self.roi_left_pct = max(0.0, min(0.45, float(payload["roi_left_pct"])))
            if "roi_right_pct" in payload and payload["roi_right_pct"] is not None:
                self.roi_right_pct = max(0.55, min(1.0, float(payload["roi_right_pct"])))
            if "hsv_lower" in payload and isinstance(payload["hsv_lower"], (list, tuple)) and len(payload["hsv_lower"]) >= 3:
                self.hsv_lower = [
                    max(0, min(179, int(payload["hsv_lower"][0]))),
                    max(0, min(255, int(payload["hsv_lower"][1]))),
                    max(0, min(255, int(payload["hsv_lower"][2]))),
                ]
                self._custom_hsv_set = True
            if "hsv_upper" in payload and isinstance(payload["hsv_upper"], (list, tuple)) and len(payload["hsv_upper"]) >= 3:
                self.hsv_upper = [
                    max(0, min(179, int(payload["hsv_upper"][0]))),
                    max(0, min(255, int(payload["hsv_upper"][1]))),
                    max(0, min(255, int(payload["hsv_upper"][2]))),
                ]
                self._custom_hsv_set = True
            if "gripper_hsv_lower" in payload and isinstance(payload["gripper_hsv_lower"], (list, tuple)) and len(payload["gripper_hsv_lower"]) >= 3:
                self.gripper_hsv_lower = [
                    max(0, min(179, int(payload["gripper_hsv_lower"][0]))),
                    max(0, min(255, int(payload["gripper_hsv_lower"][1]))),
                    max(0, min(255, int(payload["gripper_hsv_lower"][2]))),
                ]
            if "gripper_hsv_upper" in payload and isinstance(payload["gripper_hsv_upper"], (list, tuple)) and len(payload["gripper_hsv_upper"]) >= 3:
                self.gripper_hsv_upper = [
                    max(0, min(179, int(payload["gripper_hsv_upper"][0]))),
                    max(0, min(255, int(payload["gripper_hsv_upper"][1]))),
                    max(0, min(255, int(payload["gripper_hsv_upper"][2]))),
                ]
            if "probe_rx" in payload and "probe_ry" in payload:
                self.probe_rx = max(0.0, min(1.0, float(payload["probe_rx"])))
                self.probe_ry = max(0.0, min(1.0, float(payload["probe_ry"])))
                self.probe_px = None
            elif "probe_x" in payload and "probe_y" in payload:
                if payload["probe_x"] is None or payload["probe_y"] is None:
                    self.probe_px = None
                    self.probe_rx = 0.5
                    self.probe_ry = 0.5
                else:
                    self.probe_px = (int(payload["probe_x"]), int(payload["probe_y"]))

    @staticmethod
    def _hue_in_range(h_channel: np.ndarray, h_min: int, h_max: int) -> np.ndarray:
        """Compute single-channel Hue mask supporting standard [h_min <= h_max] and wrap-around [h_min > h_max]."""
        if h_min <= h_max:
            return cv2.inRange(h_channel, h_min, h_max)
        return cv2.bitwise_or(
            cv2.inRange(h_channel, h_min, 179),
            cv2.inRange(h_channel, 0, h_max),
        )

    def detect(
        self,
        frame: np.ndarray,
        force_save: bool = False,
        allow_border_touch: bool = False,
        max_area_override: Optional[float] = None,
        contrast_override: Optional[float] = None,
        build_hsv_views: bool = False,
    ) -> DetectionResult:
        """Process image frame to locate the sock on the floor using Signed Floor-Relative Darkness,
        Adaptive Brightness Cap, Solidity Filtering, and Distance-Transform Grasp Targeting."""
        if frame is None or frame.size == 0:
            return DetectionResult(found=False)

        h, w = frame.shape[:2]
        center_x = w / 2.0
        center_y = h / 2.0

        with self._hsv_lock:
            mode = self.mode
            min_area_cfg = self.min_area
            max_area_cfg = self.max_area
            contrast_cfg = self.contrast_threshold
            v_max_cap = self.v_max
            max_floor_ratio = self.max_floor_ratio
            min_solidity = self.min_solidity
            roi_top_cfg = self.roi_top_pct
            roi_bottom_cfg = self.roi_bottom_pct
            roi_left_cfg = self.roi_left_pct
            roi_right_cfg = self.roi_right_pct
            h_min, s_min, v_min = self.hsv_lower
            h_max, s_max, v_max = self.hsv_upper
            gh_min, gs_min, gv_min = self.gripper_hsv_lower
            gh_max, gs_max, gv_max = self.gripper_hsv_upper
            custom_hsv_set = self._custom_hsv_set
            if self.probe_px is not None:
                px = max(0, min(w - 1, int(self.probe_px[0])))
                py = max(0, min(h - 1, int(self.probe_px[1])))
            else:
                px = max(0, min(w - 1, int(round(self.probe_rx * (w - 1)))))
                py = max(0, min(h - 1, int(round(self.probe_ry * (h - 1)))))

        # 1. Define active floor Region of Interest (ROI)
        roi_top = 0.02 if allow_border_touch else roi_top_cfg
        roi_bottom = max(roi_bottom_cfg, 0.86) if allow_border_touch else roi_bottom_cfg
        x_min = max(0, min(w - 2, int(w * roi_left_cfg)))
        x_max = max(x_min + 2, min(w, int(w * roi_right_cfg)))
        y_min = max(0, min(h - 2, int(h * roi_top)))
        y_max = max(y_min + 2, min(h, int(h * roi_bottom)))

        roi = frame[y_min:y_max, x_min:x_max]
        roi_h, roi_w = roi.shape[:2]

        # 2. Convert full frame & ROI to HSV and extract H, S, V, Inverted V (255-V), and S*(255-V)
        hsv_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_full = hsv_full[:, :, 0]
        s_full = hsv_full[:, :, 1]
        v_full = hsv_full[:, :, 2]
        v_inv_full = cv2.bitwise_not(v_full)
        sv_inv_full = cv2.multiply(s_full, v_inv_full, scale=1.0 / 255.0)

        hsv_roi = hsv_full[y_min:y_max, x_min:x_max]
        h_roi = hsv_roi[:, :, 0]
        s_roi = hsv_roi[:, :, 1]
        v_roi = hsv_roi[:, :, 2]
        v_inv_roi = v_inv_full[y_min:y_max, x_min:x_max]
        sv_inv_roi = sv_inv_full[y_min:y_max, x_min:x_max]

        # Separate H, S, V masks inside ROI
        mask_h_roi = self._hue_in_range(h_roi, h_min, h_max)
        mask_s_roi = cv2.inRange(s_roi, s_min, s_max)
        mask_v_roi = cv2.inRange(v_roi, v_min, v_max)
        mask_hsv_roi = cv2.bitwise_and(mask_h_roi, cv2.bitwise_and(mask_s_roi, mask_v_roi))

        # Mask out Cyan/Teal gripper fingers using configurable gripper_hsv_lower / gripper_hsv_upper
        g_h_mask = self._hue_in_range(h_roi, gh_min, gh_max)
        g_s_mask = cv2.inRange(s_roi, gs_min, gs_max)
        g_v_mask = cv2.inRange(v_roi, gv_min, gv_max)
        cyan_mask = cv2.bitwise_and(g_h_mask, cv2.bitwise_and(g_s_mask, g_v_mask))

        eff_contrast = float(contrast_override) if contrast_override is not None else float(contrast_cfg)
        eff_max_area = float(max_area_override) if max_area_override is not None else float(max_area_cfg)

        # Combined S * (255 - V) threshold mask (intersected with HSV bounds and excluding cyan gripper)
        _, sv_inv_thresh = cv2.threshold(sv_inv_roi, eff_contrast, 255, cv2.THRESH_BINARY)
        mask_sv_inv_roi = cv2.bitwise_and(
            sv_inv_thresh,
            cv2.bitwise_and(mask_hsv_roi, cv2.bitwise_not(cyan_mask)),
        )

        # 3. Create foreground candidate mask via Signed Floor-Relative Darkness + Adaptive Brightness Cap
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_full = cv2.GaussianBlur(gray_full, (9, 9), 0)
        blurred = blurred_full[y_min:y_max, x_min:x_max]

        # Anchor floor median in the lower 70% of ROI so distant dark horizon/furniture
        # in upper frames cannot depress the median and light up the floor itself.
        floor_region = blurred[int(roi_h * 0.30):, :] if roi_h >= 20 else blurred
        floor_median = float(np.median(floor_region)) if floor_region.size else 150.0
        # Signed darkness relative to floor median: only pixels darker than floor produce > 0
        signed_diff_full = np.clip(floor_median - blurred_full.astype(np.float32), 0.0, 255.0).astype(np.uint8)
        signed_diff = signed_diff_full[y_min:y_max, x_min:x_max]

        diff = None
        if mode in ("auto", "contrast", "darkness"):
            diff = signed_diff
            _, dark_thresh = cv2.threshold(diff, eff_contrast, 255, cv2.THRESH_BINARY)
            # Adaptive absolute brightness cap: ensures candidate is genuinely dark even if floor_median is very bright
            eff_v_max = max(int(v_max_cap), int(round(floor_median * max_floor_ratio)))
            abs_dark_mask = cv2.inRange(blurred, 0, min(255, eff_v_max))
            thresh = cv2.bitwise_and(dark_thresh, abs_dark_mask)
        elif mode == "sv_inv":
            thresh = cv2.bitwise_and(sv_inv_thresh, mask_hsv_roi)
        elif mode == "hsv" or custom_hsv_set:
            thresh = mask_hsv_roi.copy()
        else:
            thresh = cv2.bitwise_or(
                cv2.inRange(s_roi, 50, 255),
                cv2.inRange(v_roi, 0, 70)
            )

        # Exclude cyan gripper areas
        thresh = cv2.bitwise_and(thresh, cv2.bitwise_not(cyan_mask))

        # Mask out bright/metallic robot chassis front bar in bottom-center between gripper jaws during scan
        if not allow_border_touch:
            y_bumper = max(0, int(h * 0.74) - y_min)
            x_b1 = max(0, int(w * 0.25) - x_min)
            x_b2 = min(roi_w, int(w * 0.75) - x_min)
            if y_bumper < roi_h and x_b2 > x_b1:
                bumper_sub = thresh[y_bumper:, x_b1:x_b2]
                gray_sub = blurred[y_bumper:, x_b1:x_b2]
                bumper_sub[gray_sub > 65] = 0

        # Morphological filtering to clean noise and bridge cloth folds
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

        # 4. Contour detection with Solidity & Contrast Filtering + Distance-Transform Grasp Targeting
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_contour = None
        best_area = 0.0
        best_score = 0.0
        best_center = (0.0, 0.0)
        best_bbox = (0, 0, 0, 0)
        best_solidity = 0.0
        best_inscribed_radius = 0.0
        best_dt_center: Optional[Tuple[float, float]] = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area_cfg <= area <= eff_max_area:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                aspect = float(bw) / max(1, bh)
                extent = area / float(max(1, bw * bh))

                hull = cv2.convexHull(cnt)
                hull_area = float(cv2.contourArea(hull))
                solidity = float(area) / max(1.0, hull_area)

                # Measure local contrast and intensity so faint floor shadows/glare cannot beat real cloth
                contrast_weight = 1.0
                mean_diff = eff_contrast
                mean_gray = 100.0
                patch_mask = np.zeros((bh, bw), dtype=np.uint8)
                cv2.drawContours(patch_mask, [cnt - np.array([bx, by])], -1, 255, -1)
                mean_gray = float(cv2.mean(blurred[by:by + bh, bx:bx + bw], mask=patch_mask)[0])
                if diff is not None:
                    mean_diff = float(cv2.mean(diff[by:by + bh, bx:bx + bw], mask=patch_mask)[0])
                    if allow_border_touch and mean_diff < max(eff_contrast, 50.0):
                        continue
                    contrast_weight = max(1.0, mean_diff / 40.0) ** 1.3

                touches_border = (
                    bx <= self.border_margin
                    or by <= self.border_margin
                    or (bx + bw) >= (roi_w - self.border_margin)
                    or (by + bh) >= (roi_h - self.border_margin)
                )
                # Allow a border-touching contour during scan ONLY if it is a strongly contrasting,
                # compact dark sock (e.g. near the bottom-left/right of Left/Right scan views)
                is_strong_border_sock = (
                    not allow_border_touch
                    and area >= 2500.0
                    and extent >= 0.52
                    and solidity >= max(0.50, min_solidity)
                    and mean_diff >= 90.0
                    and mean_gray < 48.0
                    and not (by <= self.border_margin and bw > (0.50 * roi_w))
                )
                # RULE A: Discard contours touching the ROI boundary during normal scan (unless strong sock),
                # or tiny vignette slivers in close-up hover mode
                if touches_border and ((not allow_border_touch and not is_strong_border_sock) or area < 2500.0):
                    continue

                # RULE A2: Discard wide top-edge horizon / room-background contours spanning
                # across the frame top or nearly full ROI width, unless in close-up hover directly over a compact dark sock
                is_closeup_hover_sock = (
                    allow_border_touch
                    and extent >= 0.55
                    and solidity >= max(0.55, min_solidity)
                    and mean_diff >= 115.0
                    and mean_gray < 35.0
                    and aspect < 2.5
                )
                if ((by <= self.border_margin and bw > (0.68 * roi_w)) or bw > (0.80 * roi_w)) and not is_closeup_hover_sock:
                    continue

                # RULE B: Discard abnormal aspect ratios (thin vertical/horizontal edge shadows or scratches)
                if aspect < 0.25 or aspect > 4.0:
                    continue

                # RULE C: Discard non-compact or low-solidity contours (stringy noise, cables, or branched shadows)
                if extent < 0.25 or solidity < min_solidity:
                    continue

                # Find centroid using image moments + Distance Transform peak (center of thickest part of sock)
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx_mom = float(M["m10"] / M["m00"]) + x_min
                    cy_mom = float(M["m01"] / M["m00"]) + y_min

                    # Compute Euclidean Distance Transform inside contour to find thickest grasp core
                    pad_mask = np.zeros((bh + 2, bw + 2), dtype=np.uint8)
                    pad_mask[1:bh + 1, 1:bw + 1] = patch_mask
                    dist_map = cv2.distanceTransform(pad_mask, cv2.DIST_L2, 5)
                    _, max_dist, _, max_loc = cv2.minMaxLoc(dist_map)
                    cx_dt = float(max_loc[0] - 1 + bx + x_min)
                    cy_dt = float(max_loc[1] - 1 + by + y_min)

                    # Check how deep the moment centroid sits inside the sock body
                    mx_local = max(0, min(bw + 1, int(round(cx_mom - x_min - bx + 1))))
                    my_local = max(0, min(bh + 1, int(round(cy_mom - y_min - by + 1))))
                    dist_at_mom = float(dist_map[my_local, mx_local])

                    # If the sock is L-shaped/curved so the moment centroid falls near the edge or outside,
                    # pull the grasp point strongly onto the thickest interior core (Distance Transform peak).
                    if max_dist > 2.0 and dist_at_mom < 0.65 * max_dist:
                        cx = 0.75 * cx_dt + 0.25 * cx_mom
                        cy = 0.75 * cy_dt + 0.25 * cy_mom
                    else:
                        cx = 0.35 * cx_dt + 0.65 * cx_mom
                        cy = 0.35 * cy_dt + 0.65 * cy_mom

                    global_bx = bx + x_min
                    global_by = by + y_min

                    solidity_weight = max(0.8, min(1.25, solidity / 0.75))
                    score = area * contrast_weight * solidity_weight
                    if score > best_score:
                        best_score = score
                        best_area = area
                        best_contour = cnt
                        best_center = (cx, cy)
                        best_bbox = (global_bx, global_by, bw, bh)
                        best_solidity = solidity
                        best_inscribed_radius = float(max_dist)
                        best_dt_center = (cx_dt, cy_dt)

        # 4b. Compute telemetry readouts
        py0, py1 = max(0, py - 2), min(h, py + 3)
        px0, px1 = max(0, px - 2), min(w, px + 3)
        p_h_patch = h_full[py0:py1, px0:px1]
        p_s_patch = s_full[py0:py1, px0:px1]
        p_v_patch = v_full[py0:py1, px0:px1]

        probe_h = int(h_full[py, px])
        probe_s = int(s_full[py, px])
        probe_v = int(v_full[py, px])
        probe_v_inv = int(v_inv_full[py, px])
        probe_sv_inv = int(sv_inv_full[py, px])
        probe_sv_inv_raw = int(probe_s * probe_v_inv)
        probe_dark_diff = round(float(signed_diff_full[py, px]), 1)

        roi_pixels = max(1, mask_h_roi.size)
        h_pass_pct = round(100.0 * float(np.count_nonzero(mask_h_roi)) / roi_pixels, 1)
        s_pass_pct = round(100.0 * float(np.count_nonzero(mask_s_roi)) / roi_pixels, 1)
        v_pass_pct = round(100.0 * float(np.count_nonzero(mask_v_roi)) / roi_pixels, 1)
        hsv_pass_pct = round(100.0 * float(np.count_nonzero(mask_hsv_roi)) / roi_pixels, 1)
        sv_inv_pass_pct = round(100.0 * float(np.count_nonzero(mask_sv_inv_roi)) / roi_pixels, 1)
        gripper_pass_pct = round(100.0 * float(np.count_nonzero(cyan_mask)) / roi_pixels, 1)
        mask_pass_pct = round(100.0 * float(np.count_nonzero(cleaned)) / roi_pixels, 1)

        sock_h_mean = sock_s_mean = sock_v_mean = sock_v_inv_mean = sock_sv_inv_mean = sock_dark_diff_mean = None
        sock_h_min = sock_h_max = sock_s_min = sock_s_max = sock_v_min = sock_v_max = None

        if best_contour is not None:
            bx_r, by_r, bw_r, bh_r = cv2.boundingRect(best_contour)
            cnt_mask = np.zeros((bh_r, bw_r), dtype=np.uint8)
            cv2.drawContours(cnt_mask, [best_contour - np.array([bx_r, by_r])], -1, 255, -1)
            sub_h = h_roi[by_r:by_r + bh_r, bx_r:bx_r + bw_r]
            sub_s = s_roi[by_r:by_r + bh_r, bx_r:bx_r + bw_r]
            sub_v = v_roi[by_r:by_r + bh_r, bx_r:bx_r + bw_r]
            sub_sv_inv = sv_inv_roi[by_r:by_r + bh_r, bx_r:bx_r + bw_r]
            sub_diff = signed_diff[by_r:by_r + bh_r, bx_r:bx_r + bw_r]
            pts_h = sub_h[cnt_mask > 0]
            pts_s = sub_s[cnt_mask > 0]
            pts_v = sub_v[cnt_mask > 0]
            pts_sv_inv = sub_sv_inv[cnt_mask > 0]
            pts_diff = sub_diff[cnt_mask > 0]
            if pts_h.size > 0:
                sock_h_mean = round(float(np.mean(pts_h)), 1)
                sock_s_mean = round(float(np.mean(pts_s)), 1)
                sock_v_mean = round(float(np.mean(pts_v)), 1)
                sock_v_inv_mean = round(255.0 - sock_v_mean, 1)
                sock_sv_inv_mean = round(float(np.mean(pts_sv_inv)), 1)
                sock_dark_diff_mean = round(float(np.mean(pts_diff)), 1)
                sock_h_min, sock_h_max = int(np.min(pts_h)), int(np.max(pts_h))
                sock_s_min, sock_s_max = int(np.min(pts_s)), int(np.max(pts_s))
                sock_v_min, sock_v_max = int(np.min(pts_v)), int(np.max(pts_v))

        hsv_stats: Dict[str, Any] = {
            "probe_x": px,
            "probe_y": py,
            "probe_h": probe_h,
            "probe_s": probe_s,
            "probe_v": probe_v,
            "probe_v_inv": probe_v_inv,
            "probe_sv_inv": probe_sv_inv,
            "probe_sv_inv_raw": probe_sv_inv_raw,
            "probe_dark_diff": probe_dark_diff,
            "floor_median": round(floor_median, 1),
            "probe_h_min": int(np.min(p_h_patch)),
            "probe_h_max": int(np.max(p_h_patch)),
            "probe_s_min": int(np.min(p_s_patch)),
            "probe_s_max": int(np.max(p_s_patch)),
            "probe_v_min": int(np.min(p_v_patch)),
            "probe_v_max": int(np.max(p_v_patch)),
            "roi_h_median": round(float(np.median(h_roi)), 1) if h_roi.size else 0.0,
            "roi_s_median": round(float(np.median(s_roi)), 1) if s_roi.size else 0.0,
            "roi_v_median": round(float(np.median(v_roi)), 1) if v_roi.size else 0.0,
            "roi_v_inv_median": round(float(np.median(v_inv_roi)), 1) if v_inv_roi.size else 0.0,
            "roi_sv_inv_median": round(float(np.median(sv_inv_roi)), 1) if sv_inv_roi.size else 0.0,
            "roi_h_mean": round(float(np.mean(h_roi)), 1) if h_roi.size else 0.0,
            "roi_s_mean": round(float(np.mean(s_roi)), 1) if s_roi.size else 0.0,
            "roi_v_mean": round(float(np.mean(v_roi)), 1) if v_roi.size else 0.0,
            "sock_h_mean": sock_h_mean,
            "sock_s_mean": sock_s_mean,
            "sock_v_mean": sock_v_mean,
            "sock_v_inv_mean": sock_v_inv_mean,
            "sock_sv_inv_mean": sock_sv_inv_mean,
            "sock_dark_diff_mean": sock_dark_diff_mean,
            "sock_solidity": round(best_solidity, 2) if best_contour is not None else None,
            "sock_inscribed_r": round(best_inscribed_radius, 1) if best_contour is not None else None,
            "sock_h_min": sock_h_min,
            "sock_h_max": sock_h_max,
            "sock_s_min": sock_s_min,
            "sock_s_max": sock_s_max,
            "sock_v_min": sock_v_min,
            "sock_v_max": sock_v_max,
            "h_pass_pct": h_pass_pct,
            "s_pass_pct": s_pass_pct,
            "v_pass_pct": v_pass_pct,
            "hsv_pass_pct": hsv_pass_pct,
            "sv_inv_pass_pct": sv_inv_pass_pct,
            "mask_pass_pct": mask_pass_pct,
            "gripper_pass_pct": gripper_pass_pct,
            "hsv_lower": [h_min, s_min, v_min],
            "hsv_upper": [h_max, s_max, v_max],
            "gripper_hsv_lower": [gh_min, gs_min, gv_min],
            "gripper_hsv_upper": [gh_max, gs_max, gv_max],
            "mode": mode,
            "contrast_threshold": round(eff_contrast, 1),
            "v_max": v_max_cap,
            "max_floor_ratio": round(max_floor_ratio, 2),
            "min_solidity": round(min_solidity, 2),
        }

        # 5. Build full-frame segmented mask representation
        mask_full = np.zeros((h, w), dtype=np.uint8)
        mask_full[y_min:y_max, x_min:x_max] = cleaned
        mask_bgr = cv2.cvtColor(mask_full, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(mask_bgr, (x_min, y_min), (x_max, y_max), (70, 70, 70), 1, cv2.LINE_AA)

        # 6. Build annotation overlay
        annotated = frame.copy()
        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (100, 100, 100), 1, cv2.LINE_AA)
        cv2.drawMarker(annotated, (int(center_x), int(center_y)), (0, 255, 255), cv2.MARKER_CROSS, 20, 1)

        # Draw Probe Reticle & Floor-Relative Darkness / HSV badge on annotated frame
        cv2.drawMarker(annotated, (px, py), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
        cv2.rectangle(annotated, (max(0, px - 3), max(0, py - 3)), (min(w - 1, px + 3), min(h - 1, py + 3)), (255, 0, 255), 1)
        hud_bg_y = max(0, h - 32)
        cv2.rectangle(annotated, (0, hud_bg_y), (w, h), (18, 18, 24), -1)
        sock_hud_str = (
            f"Sock V:{sock_v_mean:.0f} DarkDiff:{sock_dark_diff_mean:.0f} Sol:{best_solidity:.2f}"
            if sock_v_mean is not None and sock_dark_diff_mean is not None
            else "Sock: --"
        )
        hud_text = (
            f"FloorMed:{floor_median:.0f} | Probe({px},{py}) V:{probe_v:3d} DarkDiff:{probe_dark_diff:4.0f} | {sock_hud_str}"
        )
        cv2.putText(annotated, hud_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 230, 120), 1, cv2.LINE_AA)

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

            # Draw inscribed Distance-Transform grasp core circle if available
            if best_dt_center is not None and best_inscribed_radius >= 4.0:
                cv2.circle(
                    annotated,
                    (int(round(best_dt_center[0])), int(round(best_dt_center[1]))),
                    int(round(best_inscribed_radius)),
                    (255, 180, 0),
                    1,
                    cv2.LINE_AA,
                )

            # Draw detection bounding box & grasp target point on annotated frame
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            cv2.circle(annotated, (int(cx), int(cy)), 6, (0, 0, 255), -1)
            cv2.line(annotated, (int(center_x), int(center_y)), (int(cx), int(cy)), (0, 255, 0), 2)

            label = f"SOCK: {int(best_area)}px [eX:{error_x:+.2f}, eY:{error_y:+.2f}, Sol:{best_solidity:.2f}]"
            cv2.putText(annotated, label, (bx, max(20, by - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)

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
                hsv_stats=hsv_stats,
            )
        else:
            cv2.putText(annotated, "NO SOCK DETECTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            result = DetectionResult(
                found=False,
                annotated_frame=annotated,
                raw_frame=frame.copy(),
                mask_frame=mask_bgr,
                hsv_stats=hsv_stats,
            )

        # 7. Build 3-panel combined debug frame and optional auxiliary visual frames
        combined = None
        hsv_h_frame = None
        hsv_s_frame = None
        hsv_v_frame = None
        hsv_sv_inv_frame = None
        hsv_channels_frame = None
        gripper_mask_frame = None

        now = time.time()
        should_save = self.save_debug_image and (force_save or (now - self._last_disk_save_time >= 1.5))
        if self.enable_web_stream or build_hsv_views or (should_save and self.debug_combined_path):
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

        if build_hsv_views:
            hsv_chan_vis = hsv_full.copy()
            cv2.rectangle(hsv_chan_vis, (x_min, y_min), (x_max, y_max), (100, 100, 100), 1, cv2.LINE_AA)
            cv2.drawMarker(hsv_chan_vis, (px, py), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 16, 2)

            sv_inv_vis = cv2.cvtColor(sv_inv_full, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(sv_inv_vis, (x_min, y_min), (x_max, y_max), (100, 100, 100), 1, cv2.LINE_AA)
            cv2.drawMarker(sv_inv_vis, (px, py), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 16, 2)

            hsv_channels_frame = hsv_chan_vis
            hsv_sv_inv_frame = sv_inv_vis

            rw, rh = 426, 320
            h_color_hsv = np.dstack([h_full, np.full_like(h_full, 220), np.full_like(h_full, 220)])
            h_vis = cv2.cvtColor(h_color_hsv, cv2.COLOR_HSV2BGR)
            s_vis = cv2.cvtColor(s_full, cv2.COLOR_GRAY2BGR)
            v_vis = cv2.cvtColor(v_full, cv2.COLOR_GRAY2BGR)
            mh_full = np.zeros((h, w), dtype=np.uint8)
            ms_full = np.zeros((h, w), dtype=np.uint8)
            mv_full = np.zeros((h, w), dtype=np.uint8)
            mg_full = np.zeros((h, w), dtype=np.uint8)
            mh_full[y_min:y_max, x_min:x_max] = mask_h_roi
            ms_full[y_min:y_max, x_min:x_max] = mask_s_roi
            mv_full[y_min:y_max, x_min:x_max] = mask_v_roi
            mg_full[y_min:y_max, x_min:x_max] = cyan_mask
            hsv_h_frame = np.hstack([cv2.resize(h_vis, (rw, rh)), cv2.resize(cv2.cvtColor(mh_full, cv2.COLOR_GRAY2BGR), (rw, rh))])
            hsv_s_frame = np.hstack([cv2.resize(s_vis, (rw, rh)), cv2.resize(cv2.cvtColor(ms_full, cv2.COLOR_GRAY2BGR), (rw, rh))])
            hsv_v_frame = np.hstack([cv2.resize(v_vis, (rw, rh)), cv2.resize(cv2.cvtColor(mv_full, cv2.COLOR_GRAY2BGR), (rw, rh))])
            gripper_mask_frame = cv2.cvtColor(mg_full, cv2.COLOR_GRAY2BGR)

        result.combined_frame = combined
        result.hsv_h_frame = hsv_h_frame
        result.hsv_s_frame = hsv_s_frame
        result.hsv_v_frame = hsv_v_frame
        result.hsv_sv_inv_frame = hsv_sv_inv_frame
        result.hsv_channels_frame = hsv_channels_frame
        result.gripper_mask_frame = gripper_mask_frame

        # 8. Save debug images to RAM disk /tmp (throttled to at most once every 1.5s unless force_save=True)
        if should_save:
            self._last_disk_save_time = now
            try:
                if self.debug_image_path:
                    cv2.imwrite(self.debug_image_path, annotated)
                if self.debug_raw_path:
                    cv2.imwrite(self.debug_raw_path, frame)
                if self.debug_mask_path:
                    cv2.imwrite(self.debug_mask_path, mask_bgr)
                if self.debug_combined_path and combined is not None:
                    cv2.imwrite(self.debug_combined_path, combined)
            except Exception as e:
                logger.debug("Error saving debug images: %s", e)

        # 9. Update live HTTP web stream only if enabled
        if self.enable_web_stream and VisionDebugServer._instance and VisionDebugServer._instance.started:
            VisionDebugServer.update(
                raw_frame=frame,
                mask_frame=mask_bgr,
                annotated_frame=annotated,
                combined_frame=combined,
                hsv_h_frame=hsv_h_frame,
                hsv_s_frame=hsv_s_frame,
                hsv_v_frame=hsv_v_frame,
                hsv_sv_inv_frame=hsv_sv_inv_frame,
                hsv_channels_frame=hsv_channels_frame,
                gripper_mask_frame=gripper_mask_frame,
                meta={
                    "found": result.found,
                    "cx": result.cx,
                    "cy": result.cy,
                    "error_x": result.error_x,
                    "error_y": result.error_y,
                    "angle": round(result.angle, 1),
                    "area": result.area,
                    "hsv_stats": hsv_stats,
                },
            )

        return result


class VisionGraspPlanner:
    """Computes joint trajectory targets and visual servoing corrections for sock pickup and basket deposit."""

    def __init__(self, config: Dict[str, Any]):
        c4 = config.get("case4", {})
        serv = c4.get("servoing", {})
        self.pan_gain = float(serv.get("pan_gain", 60.0))
        self.lift_gain = float(serv.get("lift_gain", 15.0))
        self.align_tolerance = float(serv.get("align_tolerance", 0.12))
        self.grasp_verification_threshold = float(c4.get("grasp_verification_threshold_pct", 4.8))
        self.abort_on_miss = bool(c4.get("abort_on_miss", True))
        self.scan_sweep_angles = c4.get("scan_sweep_angles", [0.0, -28.0, 28.0])

        # Adaptive visual servoing parameters
        self.reach_elbow_gain = float(serv.get("reach_elbow_gain", 20.0))
        self.reach_lift_gain = float(serv.get("reach_lift_gain", 6.0))
        self.max_reach_delta_deg = float(serv.get("max_reach_delta_deg", 18.0))
        self.enable_roll_alignment = bool(serv.get("enable_roll_alignment", False))
        self.enable_hover_refinement = bool(serv.get("enable_hover_refinement", True))
        self.hover_lift_offset_deg = float(serv.get("hover_lift_offset_deg", -14.0))
        self.max_align_iterations = int(serv.get("max_align_iterations", 3))

        # Default poses
        self.scan_pose = c4.get("scan_pose", {
            "shoulder_pan.pos": -7.1,
            "shoulder_lift.pos": -63.7,
            "elbow_flex.pos": 17.3,
            "wrist_flex.pos": 96.0,
            "wrist_roll.pos": 1.5,
            "gripper.pos": 5.0,
        })
        self.scan_poses = c4.get("scan_poses", None)

        # Pan direction multiplier (+1.0 if right pan is positive, -1.0 if right pan is negative)
        configured_pan_dir = serv.get("pan_direction", None)
        if configured_pan_dir is not None:
            self.pan_direction = float(configured_pan_dir)
        elif self.scan_poses and isinstance(self.scan_poses, dict):
            r_pan = self.scan_poses.get("right", {}).get("shoulder_pan.pos")
            l_pan = self.scan_poses.get("left", {}).get("shoulder_pan.pos")
            if r_pan is not None and l_pan is not None:
                self.pan_direction = 1.0 if float(r_pan) > float(l_pan) else -1.0
            else:
                self.pan_direction = 1.0
        else:
            self.pan_direction = 1.0

        # Basket is mounted directly behind the gripper centerline and slightly elevated
        self.basket_pose = c4.get("basket_pose", {
            "shoulder_pan.pos": -3.3,
            "shoulder_lift.pos": -36.4,
            "elbow_flex.pos": -80.4,
            "wrist_flex.pos": -95.0,
            "wrist_roll.pos": 0.0,
            "gripper.pos": 85.0,
        })
        # Calibrated floor reach to grasp sock from floor below the mobile base
        self.floor_grasp_pose = c4.get("floor_grasp_pose", {
            "shoulder_pan.pos": -4.4,
            "shoulder_lift.pos": 97.5,
            "elbow_flex.pos": -58.2,
            "wrist_flex.pos": 59.6,
            "wrist_roll.pos": -0.7,
            "gripper.pos": 0.0,
        })

        # Inverse Kinematics Engine (SO-100 URDF Planar + Base-Yaw + 3D Ray Projection)
        kin_cfg = c4.get("kinematics", {})
        cam_cfg = c4.get("camera", {})
        self.use_ik = bool(kin_cfg.get("enabled", True))
        cam_offset = tuple(kin_cfg.get("cam_offset_gf", (-0.0700, 0.0110, -0.1260)))
        cam_w = float(cam_cfg.get("width", 640))
        cam_h = float(cam_cfg.get("height", 480))
        raw_base_h = kin_cfg.get("base_height_m", None)
        self.kinematics = SO100Kinematics(
            l1_m=float(kin_cfg.get("l1_m", 0.1160)),
            l2_m=float(kin_cfg.get("l2_m", 0.1350)),
            l3_m=float(kin_cfg.get("l3_m", 0.1594)),
            ground_clearance_m=float(kin_cfg.get("ground_clearance_m", 0.016 if raw_base_h is not None else 0.024)),
            hover_height_m=float(kin_cfg.get("hover_height_m", 0.115)),
            lift_height_m=float(kin_cfg.get("lift_height_m", 0.125)),
            radial_gain_m=float(kin_cfg.get("radial_gain_m", 0.050)),
            pan_gain_deg=float(serv.get("pan_gain", kin_cfg.get("pan_gain_deg", 60.0))),
            r_near_m=float(kin_cfg.get("r_min_m", kin_cfg.get("r_near_m", 0.145))),
            r_far_m=float(kin_cfg.get("r_max_m", kin_cfg.get("r_far_m", 0.315))),
            y_near_px=float(kin_cfg.get("y_near_px", 330.0)),
            y_far_px=float(kin_cfg.get("y_far_px", 60.0)),
            floor_grasp_pose=self.floor_grasp_pose,
            scan_pose=self.scan_pose,
            cam_offset_gf=cam_offset,
            cam_pitch_deg=float(kin_cfg.get("cam_pitch_deg", -30.0)),
            cam_fx=float(kin_cfg.get("cam_fx", 457.6)),
            cam_fy=float(kin_cfg.get("cam_fy", 457.6)),
            cam_cx=float(kin_cfg.get("cam_cx", cam_w / 2.0)),
            cam_cy=float(kin_cfg.get("cam_cy", cam_h / 2.0)),
            sock_height_offset_m=float(kin_cfg.get("sock_height_offset_m", 0.015)),
            base_height_m=float(raw_base_h) if raw_base_h is not None else None,
        )

        # Motion timing (seconds) - Slow and gentle for stability
        mot = c4.get("motion", {})
        self.scan_move_duration_s = float(mot.get("scan_move_duration_s", 1.5))
        self.descend_duration_s = float(mot.get("descend_duration_s", 2.5))
        self.clamp_duration_s = float(mot.get("clamp_duration_s", 1.2))
        self.lift_duration_s = float(mot.get("lift_duration_s", 1.5))
        self.basket_duration_s = float(mot.get("basket_duration_s", 3.2))
        self.release_duration_s = float(mot.get("release_duration_s", 0.8))
        self.stow_duration_s = float(mot.get("stow_duration_s", 2.5))

    def get_scan_targets(self) -> list[Tuple[str, Dict[str, float]]]:
        """Return sequential list of scan poses [(label, pose_dict)] for floor sweep.

        If 3-point scan_poses (center, left, right) are configured, returns those discrete poses.
        Otherwise falls back to applying scan_sweep_angles to the baseline scan_pose.
        """
        if self.scan_poses and isinstance(self.scan_poses, dict):
            targets = []
            if "center" in self.scan_poses and isinstance(self.scan_poses["center"], dict):
                targets.append(("Middle (Center)", dict(self.scan_poses["center"])))
            if "left" in self.scan_poses and isinstance(self.scan_poses["left"], dict):
                targets.append(("Left", dict(self.scan_poses["left"])))
            if "right" in self.scan_poses and isinstance(self.scan_poses["right"], dict):
                targets.append(("Right", dict(self.scan_poses["right"])))
            if targets:
                return targets

        # Fallback: multi-angle sweep angles applied to baseline scan_pose
        targets = []
        for angle in self.scan_sweep_angles:
            p = dict(self.scan_pose)
            p["shoulder_pan.pos"] = float(angle)
            label = "Middle (Center)" if abs(angle) < 1e-3 else (f"Left ({angle:+.1f}°)" if angle > 0 else f"Right ({angle:+.1f}°)")
            targets.append((label, p))
        return targets

    def compute_aligned_pose(
        self,
        current_pose: Dict[str, float],
        detection: DetectionResult,
    ) -> Tuple[Dict[str, float], bool]:
        """Compute corrected pose to center gripper over sock."""
        if not detection.found:
            return dict(current_pose), False

        aligned = dict(current_pose)

        # Rotate shoulder_pan towards the centroid of the sock using pan_direction
        pan_correction = self.pan_direction * self.pan_gain * detection.error_x
        aligned["shoulder_pan.pos"] = float(np.clip(
            aligned.get("shoulder_pan.pos", 0.0) + pan_correction,
            -110.0, 110.0
        ))

        # Vertical offset adjusts reach (shoulder_lift) around the current scan posture
        lift_correction = self.lift_gain * detection.error_y
        current_lift = float(aligned.get("shoulder_lift.pos", -50.0))
        aligned["shoulder_lift.pos"] = float(np.clip(
            current_lift + lift_correction,
            -100.0, 100.0
        ))

        is_aligned = abs(detection.error_x) <= self.align_tolerance
        return aligned, is_aligned

    def compute_target_coordinates(
        self,
        detection: Optional[DetectionResult],
        scan_pan_deg: float,
        scan_pose: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, float, float]:
        """Compute the 3D ground target coordinates (r_m, pan_deg, roll_deg) from detection."""
        return self.kinematics.compute_sock_target(
            detection,
            scan_pan_deg=scan_pan_deg,
            pan_direction=self.pan_direction,
            scan_pose=scan_pose,
        )

    def compute_grasp_pose(
        self,
        aligned_pose: Dict[str, float],
        detection: Optional[DetectionResult] = None,
        is_hover: bool = False,
        is_lift: bool = False,
        r_target: Optional[float] = None,
        pan_target: Optional[float] = None,
        roll_target: Optional[float] = None,
        scan_pose: Optional[Dict[str, float]] = None,
        descent_lift_deg: float = 16.0,
        descent_elbow_deg: float = 12.0,
    ) -> Dict[str, float]:
        """Compute downward descent pose to hover right at safe ground clearance below chassis,
        or elevated pre-grasp hover, or post-grasp vertical lift, using Inverse Kinematics."""
        target_pan = float(aligned_pose.get("shoulder_pan.pos", 0.0))

        if self.use_ik:
            if r_target is None or pan_target is None:
                r_calc, pan_calc, roll_calc = self.kinematics.compute_sock_target(
                    detection,
                    scan_pan_deg=target_pan,
                    pan_direction=self.pan_direction,
                    scan_pose=scan_pose,
                )
                if r_target is None:
                    r_target = r_calc
                if pan_target is None:
                    pan_target = pan_calc
                if roll_target is None:
                    roll_target = roll_calc

            if is_hover:
                z_target = self.kinematics.z_floor + self.kinematics.hover_height_m
                gripper_pct = 85.0
            elif is_lift:
                z_target = self.kinematics.z_floor + self.kinematics.lift_height_m
                gripper_pct = 0.0
            else:
                # Safe ground clearance: gripper tip hovers above floor without touching/scraping
                z_target = self.kinematics.z_floor + self.kinematics.ground_clearance_m
                gripper_pct = 0.0

            if self.enable_roll_alignment and detection is not None and getattr(detection, "found", False):
                sock_angle = float(getattr(detection, "angle", 0.0))
                if abs(sock_angle) > 2.0:
                    base_roll = float(self.floor_grasp_pose.get("wrist_roll.pos", 0.0))
                    roll_target = float(np.clip(base_roll + sock_angle, -175.0, 175.0))
                else:
                    roll_target = float(self.floor_grasp_pose.get("wrist_roll.pos", 0.0))
            else:
                roll_target = float(self.floor_grasp_pose.get("wrist_roll.pos", aligned_pose.get("wrist_roll.pos", 0.0)))

            return self.kinematics.compute_cartesian_pose(
                r_target=r_target,
                z_target=z_target,
                pan_target=pan_target,
                roll_target=roll_target,
                gripper_pct=gripper_pct,
                desired_pitch_deg=-90.0,
            )

        # Fallback heuristic if IK is disabled
        grasp = dict(aligned_pose)

        if self.floor_grasp_pose:
            base_lift = float(self.floor_grasp_pose.get("shoulder_lift.pos", -30.0))
            base_elbow = float(self.floor_grasp_pose.get("elbow_flex.pos", 45.0))
            base_wrist = float(self.floor_grasp_pose.get("wrist_flex.pos", 80.0))
            base_roll = float(self.floor_grasp_pose.get("wrist_roll.pos", 0.0))
        else:
            base_lift = float(np.clip(grasp.get("shoulder_lift.pos", -50.0) + descent_lift_deg, -100.0, 100.0))
            base_elbow = float(np.clip(grasp.get("elbow_flex.pos", 35.0) + descent_elbow_deg, -92.0, 92.0))
            base_wrist = float(np.clip(grasp.get("wrist_flex.pos", 70.0) + 10.0, -96.0, 96.0))
            base_roll = float(grasp.get("wrist_roll.pos", 0.0))

        grasp["shoulder_pan.pos"] = target_pan
        grasp["shoulder_lift.pos"] = base_lift
        grasp["elbow_flex.pos"] = base_elbow
        grasp["wrist_flex.pos"] = base_wrist
        grasp["wrist_roll.pos"] = base_roll

        if detection is not None and detection.found:
            if abs(detection.error_x) > 0.02:
                pan_trim = self.pan_direction * self.pan_gain * float(detection.error_x)
                grasp["shoulder_pan.pos"] = float(np.clip(target_pan + pan_trim, -110.0, 110.0))

            longitudinal_error = -float(detection.error_y)
            elbow_reach_delta = float(np.clip(self.reach_elbow_gain * longitudinal_error, -self.max_reach_delta_deg, self.max_reach_delta_deg))
            lift_reach_delta = float(np.clip(self.reach_lift_gain * longitudinal_error, -self.max_reach_delta_deg * 0.5, self.max_reach_delta_deg * 0.5))
            wrist_reach_delta = float(np.clip(-0.5 * elbow_reach_delta, -15.0, 15.0))

            grasp["elbow_flex.pos"] = float(base_elbow + elbow_reach_delta)
            grasp["shoulder_lift.pos"] = float(base_lift + lift_reach_delta)
            grasp["wrist_flex.pos"] = float(base_wrist + wrist_reach_delta)

            if self.enable_roll_alignment and abs(detection.angle) > 2.0:
                roll_offset = float(np.clip(detection.angle, -60.0, 60.0))
                grasp["wrist_roll.pos"] = float(np.clip(base_roll + roll_offset, -180.0, 180.0))
            else:
                grasp["wrist_roll.pos"] = base_roll

        if is_hover:
            grasp["shoulder_lift.pos"] = float(grasp["shoulder_lift.pos"] + self.hover_lift_offset_deg)
            grasp["gripper.pos"] = 85.0
        elif is_lift:
            grasp["shoulder_lift.pos"] = float(grasp["shoulder_lift.pos"] - 20.0)
            grasp["gripper.pos"] = 0.0
        else:
            grasp["gripper.pos"] = 0.0

        return grasp

    def verify_grasp(self, measured_gripper_pct: float) -> bool:
        """Verify if sock is caught between jaws while elevated off floor.
        
        When elevated:
        - If empty, jaws close fully to ~4.5%.
        - If holding cloth, jaws are blocked at >= 4.8%.
        """
        return measured_gripper_pct >= self.grasp_verification_threshold

    def verify_grasp_with_camera(
        self,
        arm: Any,
        camera: Any,
        detector: Any,
        feedback_cb: Optional[Callable[[float], None]] = None,
    ) -> Tuple[bool, Optional[DetectionResult]]:
        """Capture a fresh wrist-camera image after the gripper closes and lifts to verify
        whether the black sock was picked up off the floor.

        Because the wrist camera looks at the floor in front of the cyan gripper jaws while a
        grasped sock is tucked inside the closed jaws:
        - If the black sock is STILL detected on the floor (`det_verify.found == True`), the
          grasp failed (sock was unreachable or missed) -> returns `(False, det_verify)`.
        - If the black sock is NO LONGER on the floor (`det_verify.found == False`), the sock
          was successfully picked up off the floor -> returns `(True, det_verify)`.
        """
        is_dry = bool(getattr(arm, "dry_run", False))
        if camera is None or detector is None:
            cur_pose = arm.get_follower_pose() if arm is not None else {}
            measured_grip = float(cur_pose.get("gripper.pos", 25.0 if is_dry else 0.0))
            return self.verify_grasp(measured_grip), None

        if not is_dry:
            time.sleep(0.30)
        if feedback_cb is not None:
            feedback_cb(1.0)

        camera.ensure_healthy()
        t_settled = time.time()
        ok, v_frame = camera.capture_frame(max_staleness_s=0.15, min_timestamp=t_settled if not is_dry else None)
        if not ok or v_frame is None:
            logger.warning("[VERIFY CAMERA] Could not capture post-grasp verification frame; falling back to gripper position.")
            cur_pose = arm.get_follower_pose() if arm is not None else {}
            measured_grip = float(cur_pose.get("gripper.pos", 25.0 if is_dry else 0.0))
            return self.verify_grasp(measured_grip), None

        det_verify = detector.detect(
            v_frame,
            force_save=True,
            allow_border_touch=False,
        )

        if det_verify.found:
            logger.warning(
                "[VERIFY CAMERA] Black sock is STILL on the floor after closing gripper (Area=%.0fpx at (%.0f, %.0f))! "
                "Grasp failed / sock unreachable.",
                det_verify.area, det_verify.cx, det_verify.cy,
            )
            VisionDebugServer.lock_detection(
                annotated_frame=det_verify.annotated_frame,
                label="GRASP FAILED (Sock Still on Floor)",
                area=det_verify.area,
                cx=det_verify.cx,
                cy=det_verify.cy,
            )
            return False, det_verify

        logger.info(
            "[VERIFY CAMERA] Floor is clear (black sock is no longer on the floor) -> Sock successfully grasped!"
        )
        VisionDebugServer.lock_detection(
            annotated_frame=det_verify.annotated_frame,
            label="Grasp Verified (Sock Picked Up)",
            area=0.0,
            cx=0.0,
            cy=0.0,
        )
        return True, det_verify

    def execute_scan_sweep(
        self,
        arm: Any,
        camera: Any,
        detector: Any,
        feedback_cb: Optional[Any] = None,
        abort_cb: Optional[Any] = None,
        open_gripper_pct: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Sweep configured scan poses, capture post-settle frames, and return best floor detection.

        Returns a dict containing `det`, `coords` `(r_target, pan_target, roll_target)`,
        `label`, `pose`, and `frame`, or None if no sock was found.
        """
        if camera is None or detector is None:
            return None

        camera.ensure_healthy()
        candidates = []
        settle_s = 0.0 if getattr(arm, "dry_run", False) else 0.35

        for label, target in self.get_scan_targets():
            if abort_cb is not None and abort_cb():
                break

            scan_cmd = dict(target)
            if open_gripper_pct is not None:
                scan_cmd["gripper.pos"] = float(open_gripper_pct)

            pan_val = scan_cmd.get("shoulder_pan.pos", 0.0)
            logger.info("[SCAN] Moving camera to %s scan pose (pan: %+.1f°)...", label, pan_val)
            arm.move_to_pose(scan_cmd, duration_s=self.scan_move_duration_s, feedback_cb=feedback_cb)
            if settle_s > 0.0:
                time.sleep(settle_s)
            if feedback_cb is not None:
                feedback_cb(1.0)

            meas_pose = scan_cmd if getattr(arm, "dry_run", False) else arm.get_follower_pose()
            t_settled = time.time()
            ok, frame = camera.capture_frame(max_staleness_s=0.15, min_timestamp=t_settled)
            if not ok or frame is None:
                continue

            det = detector.detect(frame, force_save=True)
            if det.found:
                score = det.area * (1.0 - 0.4 * abs(det.error_x))
                logger.info(
                    "[SCAN CANDIDATE] Sock found at %s: Area=%.0fpx, eX=%+.2f, eY=%+.2f, Score=%.0f",
                    label, det.area, det.error_x, det.error_y, score,
                )
                candidates.append({
                    "score": score,
                    "det": det,
                    "pose": meas_pose,
                    "pan_val": meas_pose.get("shoulder_pan.pos", pan_val),
                    "label": label,
                    "frame": frame,
                })
            else:
                logger.info("[SCAN] No sock detected at %s. Continuing sweep...", label)

        if not candidates:
            return None

        best = max(candidates, key=lambda c: c["score"])
        best_det = best["det"]
        best_pose = best["pose"]
        r_target, pan_target, roll_target = self.kinematics.compute_sock_target(
            best_det,
            scan_pan_deg=best["pan_val"],
            pan_direction=self.pan_direction,
            scan_pose=best_pose,
        )
        best["coords"] = (r_target, pan_target, roll_target)

        if best.get("frame") is not None:
            locked_det = detector.detect(best["frame"], force_save=True)
            VisionDebugServer.lock_detection(
                annotated_frame=locked_det.annotated_frame,
                raw_frame=locked_det.raw_frame,
                mask_frame=locked_det.mask_frame,
                meta={
                    "found": True,
                    "label": f"Scan Lock ({best['label']})",
                    "cx": round(locked_det.cx, 1),
                    "cy": round(locked_det.cy, 1),
                    "area": round(locked_det.area, 0),
                    "error_x": round(locked_det.error_x, 3),
                    "error_y": round(locked_det.error_y, 3),
                    "coords": [round(r_target, 4), round(pan_target, 2), round(roll_target, 2)],
                },
            )

        return best

    def refine_target_in_hover(
        self,
        arm: Any,
        camera: Any,
        detector: Any,
        r_target: float,
        pan_target: float,
        roll_target: float,
        hover_pose: Dict[str, float],
        feedback_cb: Optional[Any] = None,
    ) -> Tuple[float, float, float, Optional[DetectionResult]]:
        """Close-up visual refinement while hovering above target sock before final descent."""
        if not self.enable_hover_refinement or camera is None or detector is None or getattr(arm, "dry_run", False):
            return r_target, pan_target, roll_target, None

        time.sleep(0.30)
        if feedback_cb is not None:
            feedback_cb(1.0)

        meas_hover = arm.get_follower_pose()
        t_settled = time.time()
        ok, h_frame = camera.capture_frame(max_staleness_s=0.15, min_timestamp=t_settled)
        if not ok or h_frame is None:
            return r_target, pan_target, roll_target, None

        det_hov = detector.detect(
            h_frame,
            force_save=True,
            allow_border_touch=True,
            max_area_override=220000.0,
        )
        if not det_hov.found:
            return r_target, pan_target, roll_target, det_hov

        # If the close-up sock contour fills a large portion of the hover frame or is clipped by the top/side
        # ROI boundary (which happens when the gripper is already hovering directly over the sock at 9cm),
        # keep the unclipped 3D target computed from the full-view scan pose.
        h_img, w_img = h_frame.shape[:2]
        bx, by, bw, bh = det_hov.bbox
        margin_top = int(h_img * 0.02) + getattr(detector, "border_margin", 4)
        margin_left = int(w_img * getattr(detector, "roi_left_pct", 0.04)) + getattr(detector, "border_margin", 4)
        margin_right = int(w_img * getattr(detector, "roi_right_pct", 0.96)) - getattr(detector, "border_margin", 4)
        if det_hov.area > 35000.0 or by <= margin_top or bx <= margin_left or (bx + bw) >= margin_right:
            logger.info(
                "[HOVER REFINE] Sock confirmed directly beneath gripper (area=%.0fpx); keeping unclipped scan target R=%.1fcm, Pan=%+.1f°",
                det_hov.area, r_target * 100.0, pan_target,
            )
            return r_target, pan_target, roll_target, det_hov

        r_new, pan_new, _ = self.kinematics.compute_sock_target(
            det_hov,
            scan_pan_deg=meas_hover.get("shoulder_pan.pos", pan_target),
            pan_direction=self.pan_direction,
            scan_pose=meas_hover,
        )

        # Reject noisy outliers far away from scan lock
        if abs(r_new - r_target) <= 0.06 and abs(pan_new - pan_target) <= 15.0:
            if abs(r_new - r_target) > 0.012 or abs(pan_new - pan_target) > 2.5:
                logger.info(
                    "[HOVER REFINE] Adjusting target: R %.1f->%.1fcm, Pan %+.1f->%+.1f°",
                    r_target * 100.0, r_new * 100.0, pan_target, pan_new,
                )
                refined_hover = self.kinematics.compute_cartesian_pose(
                    r_target=r_new,
                    z_target=self.kinematics.z_floor + self.kinematics.hover_height_m,
                    pan_target=pan_new,
                    roll_target=roll_target,
                    gripper_pct=85.0,
                )
                arm.move_to_pose(refined_hover, duration_s=0.8, feedback_cb=feedback_cb)
            VisionDebugServer.lock_detection(
                annotated_frame=det_hov.annotated_frame,
                raw_frame=det_hov.raw_frame,
                mask_frame=det_hov.mask_frame,
                meta={
                    "found": True,
                    "label": "Hover Refined Lock",
                    "cx": round(det_hov.cx, 1),
                    "cy": round(det_hov.cy, 1),
                    "area": round(det_hov.area, 0),
                    "error_x": round(det_hov.error_x, 3),
                    "error_y": round(det_hov.error_y, 3),
                    "coords": [round(r_new, 4), round(pan_new, 2), round(roll_target, 2)],
                },
            )
            return r_new, pan_new, roll_target, det_hov

        return r_target, pan_target, roll_target, det_hov

    def execute_basket_deposit(
        self,
        arm: Any,
        feedback_cb: Optional[Callable[[float], None]] = None,
    ) -> bool:
        """Deposit grasped sock into rear basket via 2-stage compact overhead clearance.

        Stage 1 moves through a compact overhead waypoint (shoulder_lift=-38°, elbow_flex=+18°,
        wrist_flex=-45°, wrist_roll=0°) while keeping gripper clamped (0%), avoiding the full
        extension gravity spike over the front edge. Stage 2 settles into rear basket_pose
        still clamped (0%), then verifies follower joints actually reached rear basket before
        opening the gripper (85%). Returns False without opening if stalled in front.
        """
        basket_t = float(getattr(self, "basket_duration_s", 3.2))
        release_t = float(getattr(self, "release_duration_s", 0.8))
        is_dry = bool(getattr(arm, "dry_run", False))

        # 1. Stage 1: Compact overhead clearance keeping gripper clamped (0%) and roll neutral
        stage1_t = 0.05 if is_dry else max(1.8, basket_t * 0.5)
        overhead_pose = {
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": -38.0,
            "elbow_flex.pos": 18.0,
            "wrist_flex.pos": -45.0,
            "wrist_roll.pos": 0.0,
            "gripper.pos": 0.0,
        }
        logger.info(
            "[BASKET] 1. Compact overhead clearance (pan=0°, lift=-38°, elbow=+18°, wrist=-45°, gripper=0%% over %.1fs)...",
            stage1_t,
        )
        arm.move_to_pose(overhead_pose, duration_s=stage1_t, feedback_cb=feedback_cb)

        if not is_dry:
            t_wait = time.time()
            while time.time() - t_wait < 0.5:
                cur = arm.get_follower_pose()
                if float(cur.get("shoulder_lift.pos", 999.0)) < 5.0:
                    break
                if feedback_cb:
                    feedback_cb(1.0)
                time.sleep(0.05)

        meas_stage1 = arm.get_follower_pose()
        stage1_lift = float(meas_stage1.get("shoulder_lift.pos", 999.0))
        if stage1_lift >= 5.0:
            logger.warning(
                "[BASKET] Safety gate: arm stalled in front during overhead lift "
                "(shoulder_lift=%.1f° >= 5.0°). Keeping gripper closed and aborting release!",
                stage1_lift,
            )
            return False

        # 2. Stage 2: Settle into rear basket_pose with gripper still clamped (0%)
        stage2_t = 0.05 if is_dry else max(1.4, basket_t * 0.5)
        transit_pose = dict(self.basket_pose)
        transit_pose["gripper.pos"] = 0.0
        logger.info(
            "[BASKET] 2. Settling over rear collection basket (%.1fs, gripper=0%%)...",
            stage2_t,
        )
        arm.move_to_pose(transit_pose, duration_s=stage2_t, feedback_cb=feedback_cb)

        target_elbow = float(transit_pose.get("elbow_flex.pos", -80.4))
        if not is_dry:
            t_wait = time.time()
            while time.time() - t_wait < 0.5:
                cur = arm.get_follower_pose()
                cur_l = float(cur.get("shoulder_lift.pos", 999.0))
                cur_e = float(cur.get("elbow_flex.pos", 999.0))
                if cur_l < 5.0 and (target_elbow >= -20.0 or cur_e < -20.0):
                    break
                if feedback_cb:
                    feedback_cb(1.0)
                time.sleep(0.05)

        meas_basket = arm.get_follower_pose()
        basket_lift = float(meas_basket.get("shoulder_lift.pos", 999.0))
        basket_elbow = float(meas_basket.get("elbow_flex.pos", 999.0))
        reached_rear = (basket_lift < 5.0) and (target_elbow >= -20.0 or basket_elbow < -20.0)
        if not reached_rear:
            logger.warning(
                "[BASKET] Safety gate: arm did not reach rear basket "
                "(shoulder_lift=%.1f°, elbow_flex=%.1f°). Keeping gripper closed!",
                basket_lift,
                basket_elbow,
            )
            return False

        # 3. Release Gate passed: open gripper over rear basket
        logger.info("[BASKET] 3. Releasing sock into basket (opening gripper to 85%% over %.1fs)...", release_t)
        drop_pose = dict(transit_pose)
        drop_pose["gripper.pos"] = 85.0
        arm.move_to_pose(drop_pose, duration_s=release_t, feedback_cb=feedback_cb)
        if not is_dry:
            time.sleep(0.5)
        if feedback_cb:
            feedback_cb(1.0)
        return True


def _get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def calibrate_hsv(
    config_path: str = "config.yaml",
    dry_run: bool = False,
    move_to_scan: bool = False,
    enable_gui: bool = True,
    duration_s: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Interactive Vision Debug & Floor Scan screen (./run_demo.sh 9) with live Web Dashboard."""
    import subprocess
    import yaml

    if not dry_run:
        try:
            ignore_pids = {str(os.getpid()), str(os.getppid())}
            out = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
            for line in out.splitlines():
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    pid, cmd = parts
                    if (
                        pid not in ignore_pids
                        and "vision_grasp.py" in cmd
                        and "python" in cmd
                        and " -c " not in cmd
                    ):
                        os.kill(int(pid), 15)
                        time.sleep(0.3)
        except Exception:
            pass

    if not os.path.exists(config_path):
        alt_path = os.path.join(_PROJECT_ROOT, "config.yaml")
        if os.path.exists(alt_path):
            config_path = alt_path

    cfg: Dict[str, Any] = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    c4 = cfg.get("case4", {})
    cam_cfg = c4.get("camera", {})
    det_cfg = c4.get("detection", {})

    planner = VisionGraspPlanner(cfg)
    arm = None
    try:
        from src.arm_controller import ArmController
        arm = ArmController(cfg, dry_run=dry_run)
        if move_to_scan and arm.connect_follower():
            scan_pose = c4.get("scan_pose", {})
            if scan_pose:
                print("[ARM] Moving Follower Arm to calibrated scan_pose for floor view...")
                arm.move_to_pose(scan_pose, duration_s=1.5)
    except Exception as e:
        print(f"[ARM] Note: Arm controller setup note: {e}")

    port = int(det_cfg.get("web_stream_port", 8080))
    camera = WristCamera(
        device_path=cam_cfg.get("device", "/dev/video0"),
        fallback_index=int(cam_cfg.get("fallback", 0)),
        width=int(cam_cfg.get("width", 640)),
        height=int(cam_cfg.get("height", 480)),
        fps=int(cam_cfg.get("fps", 30)),
        dry_run=dry_run,
    )
    detector = SockDetector(
        mode=det_cfg.get("mode", "auto"),
        min_area=float(det_cfg.get("min_area", 800.0)),
        max_area=float(det_cfg.get("max_area", 80000.0)),
        contrast_threshold=float(det_cfg.get("contrast_threshold", 35.0)),
        v_max=int(det_cfg.get("v_max", 75)),
        max_floor_ratio=float(det_cfg.get("max_floor_ratio", 0.65)),
        min_solidity=float(det_cfg.get("min_solidity", 0.45)),
        border_margin=int(det_cfg.get("border_margin", 4)),
        roi_top_pct=float(det_cfg.get("roi_top_pct", 0.06)),
        roi_bottom_pct=float(det_cfg.get("roi_bottom_pct", 0.86)),
        roi_left_pct=float(det_cfg.get("roi_left_pct", 0.04)),
        roi_right_pct=float(det_cfg.get("roi_right_pct", 0.96)),
        hsv_lower=det_cfg.get("hsv_lower", [0, 0, 0]),
        hsv_upper=det_cfg.get("hsv_upper", [179, 255, 70]),
        gripper_hsv_lower=det_cfg.get("gripper_hsv_lower", [75, 70, 70]),
        gripper_hsv_upper=det_cfg.get("gripper_hsv_upper", [115, 255, 255]),
        enable_web_stream=True,
        web_stream_port=port,
        config_path=config_path,
    )

    server = VisionDebugServer.get_instance(port)
    server.detector = detector
    server.camera = camera
    server.planner = planner
    server.arm = arm
    server.config_path = config_path

    if not camera.connect():
        print("[ERROR] Could not open wrist camera for Vision Debug & Scan.")
        return None

    camera.start_stream(detector=detector, target_fps=20.0)
    robot_ip = _get_local_ip()

    print("\n" + "=" * 78)
    print("         MIDI ROBOT - CASE 9: VISION DEBUG & FLOOR SCAN DASHBOARD")
    print("=" * 78)
    print(f"  * Live Web Debug Dashboard (4 Windows + Scan + Constants): http://{robot_ip}:{port}/")
    print(f"  * Window 1 - Live Camera Feed:             http://{robot_ip}:{port}/stream/raw")
    print(f"  * Window 2 - Masked Camera View:           http://{robot_ip}:{port}/stream/mask")
    print(f"  * Window 3 - Sock Detected Camera View:    http://{robot_ip}:{port}/stream/detection")
    print(f"  * Window 4 - Captured Sock Snapshot:       http://{robot_ip}:{port}/stream/last_detected")
    print("-" * 78)
    print("  Instructions:")
    print("  1. Open http://" + f"{robot_ip}:{port}/ in your browser.")
    print("  2. Tune detection constants (Contrast, V_max, Floor Ratio, Solidity, Area, ROI)")
    print("     live and click 'Save Constants to config.yaml'.")
    print("  3. Click 'Run 3-Point Floor Scan Sweep' in the browser to scan Middle/Left/Right")
    print("     and lock the detected sock snapshot in Window 4.")
    print("  4. Press Ctrl+C in this terminal to save & exit.")
    print("=" * 78 + "\n")

    t_start = time.time()
    last_print = 0.0
    last_stats: Dict[str, Any] = {}
    prev_found = False

    try:
        while True:
            ok, frame = camera.capture_frame(max_staleness_s=0.35)
            if ok and frame is not None:
                res = detector.detect(frame, build_hsv_views=(duration_s is not None))
                hs = res.hsv_stats
                last_stats = hs
                # Auto-capture snapshot when a sock transitions to detected or has strong lock in debug mode
                if res.found and (not prev_found or VisionDebugServer._instance._latest_locked_detection is None):
                    VisionDebugServer.lock_detection(
                        annotated_frame=res.annotated_frame,
                        label="Live Detection Lock",
                        area=res.area,
                        cx=res.cx,
                        cy=res.cy,
                    )
                prev_found = res.found

                now = time.time()
                if now - last_print >= 0.5:
                    last_print = now
                    sock_str = (
                        f"FOUND Area={res.area:5.0f}px V={hs['sock_v_mean']:3.0f} Diff={hs['sock_dark_diff_mean']:3.0f} Sol={hs['sock_solidity']:.2f}"
                        if res.found and hs.get("sock_v_mean") is not None
                        else "NO SOCK"
                    )
                    print(
                        f"\r[VISION] FloorMed={hs.get('floor_median', 0):5.1f} | "
                        f"Probe({hs['probe_x']:3d},{hs['probe_y']:3d}): V={hs['probe_v']:3d} DarkDiff={hs.get('probe_dark_diff', 0):4.0f} | "
                        f"Mask={hs.get('mask_pass_pct', 0):4.1f}% | {sock_str}       ",
                        end="",
                        flush=True,
                    )

            if duration_s is not None and (time.time() - t_start) >= duration_s:
                print()
                break
            time.sleep(0.04)
    except KeyboardInterrupt:
        print("\n\n[VISION DEBUG] Stopped by user.")
        if sys.stdin.isatty() and duration_s is None:
            cfg_now = detector.get_hsv_config()
            print(
                f"  Current Constants: mode={cfg_now['mode']}, contrast={cfg_now['contrast_threshold']}, "
                f"v_max={cfg_now['v_max']}, floor_ratio={cfg_now['max_floor_ratio']}, solidity={cfg_now['min_solidity']}"
            )
            try:
                ans = input(f"\nSave these constants to '{config_path}'? [Y/n]: ").strip().lower()
                if ans in ("", "y", "yes"):
                    if update_config_hsv(
                        config_path=config_path,
                        hsv_lower=cfg_now["hsv_lower"],
                        hsv_upper=cfg_now["hsv_upper"],
                        gripper_hsv_lower=cfg_now["gripper_hsv_lower"],
                        gripper_hsv_upper=cfg_now["gripper_hsv_upper"],
                        mode=cfg_now["mode"],
                        contrast_threshold=cfg_now["contrast_threshold"],
                        v_max=cfg_now["v_max"],
                        max_floor_ratio=cfg_now["max_floor_ratio"],
                        min_solidity=cfg_now["min_solidity"],
                        min_area=cfg_now["min_area"],
                        max_area=cfg_now["max_area"],
                        roi_top_pct=cfg_now["roi_top_pct"],
                        roi_bottom_pct=cfg_now["roi_bottom_pct"],
                        roi_left_pct=cfg_now["roi_left_pct"],
                        roi_right_pct=cfg_now["roi_right_pct"],
                    ):
                        print(f"[SUCCESS] Saved calibrated detection constants to '{config_path}'!")
            except (EOFError, KeyboardInterrupt):
                pass
    finally:
        camera.release()

    return last_stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MIDI Wrist Camera Vision Debug & Floor Scan Tool")
    parser.add_argument("-c", "--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("-d", "--dry-run", action="store_true", help="Run in simulation mode without hardware camera")
    parser.add_argument("--move-to-scan", action="store_true", help="Move follower arm to calibrated scan_pose first")
    parser.add_argument("--no-gui", action="store_true", help="Disable local OpenCV X11 window (use Web Dashboard + CLI)")
    parser.add_argument("--duration", type=float, default=None, help="Optional run duration in seconds")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    calibrate_hsv(
        config_path=args.config,
        dry_run=args.dry_run,
        move_to_scan=args.move_to_scan,
        enable_gui=not args.no_gui,
        duration_s=args.duration,
    )

