"""Vision-Guided Grasping and Object Detection for SO-100 Follower Arm.

Implements Suggestion 3 (Vision-Guided Grasping via Wrist Camera) and
Suggestion 2 (Gripper Load / Position Grasp Verification) for Case 4.
"""

from __future__ import annotations

import logging
import math
import os
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
    area: float = 0.0         # Contour area (pixels)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0) # (x, y, w, h)
    annotated_frame: Optional[np.ndarray] = None


class WristCamera:
    """Manages wrist camera video capture with fallback and dry-run simulation."""

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

    def connect(self) -> bool:
        """Initialize camera capture."""
        if self.dry_run:
            logger.info("[DRY RUN] Wrist camera initialized in simulation.")
            self.is_opened = True
            return True

        # Try device path first if exists
        target = self.device_path if (self.device_path and os.path.exists(self.device_path)) else self.fallback_index
        logger.info("Opening wrist camera on %s...", target)

        try:
            self.cap = cv2.VideoCapture(target)
            if not self.cap.isOpened():
                if target != self.fallback_index:
                    logger.warning("Failed on %s, trying fallback index %d...", target, self.fallback_index)
                    self.cap = cv2.VideoCapture(self.fallback_index)

            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                # Warm up camera buffer
                for _ in range(5):
                    self.cap.read()
                self.is_opened = True
                logger.info("Wrist camera opened successfully (%dx%d @ %d FPS)", self.width, self.height, self.fps)
                return True
        except Exception as e:
            logger.error("Error opening wrist camera: %s", e)

        self.is_opened = False
        return False

    def capture_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame from the camera."""
        if self.dry_run:
            # Generate a realistic synthetic camera frame with simulated sock
            frame = np.full((self.height, self.width, 3), 150, dtype=np.uint8)
            # Add synthetic floor texture
            noise = np.random.randint(-15, 15, (self.height, self.width, 3), dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            # Draw synthetic dark sock near center
            cv2.ellipse(frame, (340, 220), (45, 25), 20, 0, 360, (30, 30, 40), -1)
            # Draw synthetic cyan gripper tips at bottom
            cv2.fillPoly(frame, [np.array([[180, 480], [240, 340], [280, 360], [220, 480]])], (180, 150, 0))
            cv2.fillPoly(frame, [np.array([[460, 480], [400, 340], [360, 360], [420, 480]])], (180, 150, 0))
            return True, frame

        if not self.is_opened or self.cap is None:
            if not self.connect():
                return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            logger.warning("Wrist camera failed to capture frame!")
            return False, None

        return True, frame

    def release(self):
        """Cleanly release camera resources."""
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
        max_area: float = 60000.0,
        contrast_threshold: float = 38.0,
        border_margin: int = 12,
        save_debug_image: bool = True,
        debug_image_path: str = "/tmp/sock_detection.jpg",
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
        self.roi_top_pct = roi_top_pct
        self.roi_bottom_pct = roi_bottom_pct
        self.roi_left_pct = roi_left_pct
        self.roi_right_pct = roi_right_pct

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Process image frame to locate the sock on the floor."""
        if frame is None or frame.size == 0:
            return DetectionResult(found=False)

        h, w = frame.shape[:2]
        center_x = w / 2.0
        center_y = h / 2.0

        # 1. Define active floor Region of Interest (ROI)
        # Excludes lens edge vignetting (left/right) and cyan gripper finger tips (bottom)
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
                cv2.inRange(sat, 50, 255),        # Chromatic / colored socks
                cv2.inRange(val, 0, 70)           # Dark / black socks
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

        # 5. Build annotation overlay
        annotated = frame.copy()
        # Draw search ROI bounds
        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (100, 100, 100), 1, cv2.LINE_AA)
        # Draw central grasp reticle
        cv2.drawMarker(annotated, (int(center_x), int(center_y)), (0, 255, 255), cv2.MARKER_CROSS, 20, 1)

        if best_contour is not None:
            cx, cy = best_center
            bx, by, bw, bh = best_bbox

            # Normalized error from center: [-1.0, 1.0]
            error_x = (cx - center_x) / center_x
            error_y = (cy - center_y) / center_y

            # Draw detection bounding box & centroid
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            cv2.circle(annotated, (int(cx), int(cy)), 6, (0, 0, 255), -1)
            cv2.line(annotated, (int(center_x), int(center_y)), (int(cx), int(cy)), (0, 255, 0), 2)

            label = f"SOCK: {int(best_area)}px [eX:{error_x:+.2f}, eY:{error_y:+.2f}]"
            cv2.putText(annotated, label, (bx, max(20, by - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            result = DetectionResult(
                found=True,
                cx=cx,
                cy=cy,
                error_x=error_x,
                error_y=error_y,
                area=best_area,
                bbox=best_bbox,
                annotated_frame=annotated,
            )
        else:
            cv2.putText(annotated, "NO SOCK DETECTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            result = DetectionResult(found=False, annotated_frame=annotated)

        # Save debug inspection snapshot
        if self.save_debug_image and self.debug_image_path:
            try:
                cv2.imwrite(self.debug_image_path, annotated)
            except Exception:
                pass

        return result


class VisionGraspPlanner:
    """Computes joint trajectory targets and visual servoing corrections for sock pickup and basket deposit."""

    def __init__(self, config: Dict[str, Any]):
        c4 = config.get("case4", {})
        serv = c4.get("servoing", {})
        self.pan_gain = float(serv.get("pan_gain", 22.0))
        self.lift_gain = float(serv.get("lift_gain", 15.0))
        self.align_tolerance = float(serv.get("align_tolerance", 0.12))
        self.grasp_verification_threshold = float(c4.get("grasp_verification_threshold_pct", 12.0))
        self.scan_sweep_angles = c4.get("scan_sweep_angles", [0.0, -28.0, 28.0])

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
        descent_lift_deg: float = 16.0,
        descent_elbow_deg: float = 12.0,
    ) -> Dict[str, float]:
        """Compute final downward descent pose to hover right at floor level below chassis base."""
        grasp = dict(aligned_pose)
        if self.floor_grasp_pose:
            # Maintain visually aligned shoulder_pan, apply calibrated floor reach & pitch
            grasp["shoulder_lift.pos"] = float(self.floor_grasp_pose.get("shoulder_lift.pos", -30.0))
            grasp["elbow_flex.pos"] = float(self.floor_grasp_pose.get("elbow_flex.pos", 45.0))
            grasp["wrist_flex.pos"] = float(self.floor_grasp_pose.get("wrist_flex.pos", 80.0))
            grasp["wrist_roll.pos"] = float(self.floor_grasp_pose.get("wrist_roll.pos", 0.0))
        else:
            grasp["shoulder_lift.pos"] = float(np.clip(grasp.get("shoulder_lift.pos", -50.0) + descent_lift_deg, -90.0, 0.0))
            grasp["elbow_flex.pos"] = float(np.clip(grasp.get("elbow_flex.pos", 35.0) + descent_elbow_deg, -20.0, 70.0))
            grasp["wrist_flex.pos"] = float(np.clip(grasp.get("wrist_flex.pos", 70.0) + 10.0, 0.0, 100.0))
        grasp["gripper.pos"] = 0.0  # Clamp target
        return grasp

    def verify_grasp(self, measured_gripper_pct: float) -> bool:
        """Verify if sock is caught between jaws while elevated off floor.
        
        When elevated:
        - If empty, jaws close fully to < 8%.
        - If holding cloth, jaws are blocked at >= 12%.
        """
        return measured_gripper_pct >= self.grasp_verification_threshold
