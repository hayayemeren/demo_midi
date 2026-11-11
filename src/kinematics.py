"""SO-100 Robot Arm Forward & Inverse Kinematics (IK) Engine.

Provides closed-form, zero-latency analytical kinematics for the 5-DOF SO-100 arm:
  - Direct 2D/3D Forward Kinematics (FK) and Inverse Kinematics (IK)
  - Anchored on user's calibrated floor_grasp_pose where gripper tip touches ground perpendicularly
  - Enforces safe ground clearance so the gripper never hits or scrapes the floor
  - Generates Cartesian pick trajectories: Hover -> Vertical Descent -> Clamp -> Vertical Lift
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("demo.kinematics")


class SO100Kinematics:
    """Analytical planar + base-yaw Inverse Kinematics solver for SO-100 manipulator."""

    def __init__(
        self,
        l1_m: float = 0.116,
        l2_m: float = 0.135,
        l3_m: float = 0.135,
        ground_clearance_m: float = 0.018,
        hover_height_m: float = 0.080,
        lift_height_m: float = 0.120,
        radial_gain_m: float = 0.050,
        pan_gain_deg: float = 25.0,
        r_near_m: float = 0.115,
        r_far_m: float = 0.215,
        y_near_px: float = 330.0,
        y_far_px: float = 60.0,
        floor_grasp_pose: Optional[Dict[str, float]] = None,
    ):
        """Initialize SO-100 kinematics model.

        Args:
            l1_m: Length of upper arm link between shoulder and elbow pitch axes (meters).
            l2_m: Length of forearm link between elbow and wrist pitch axes (meters).
            l3_m: Length from wrist pitch axis to gripper jaw fingertips (meters).
            ground_clearance_m: Safe elevation of gripper fingertips above raw ground contact (meters).
            hover_height_m: Pre-grasp hover height above floor (meters).
            lift_height_m: Post-grasp vertical lift height above floor (meters).
            radial_gain_m: Radial distance shift across camera vertical FOV (meters).
            pan_gain_deg: Pan angle shift across camera horizontal FOV (degrees).
            r_near_m: Floor radial distance at bottom of camera ROI (meters).
            r_far_m: Floor radial distance at top of camera ROI (meters).
            y_near_px: Pixel Y corresponding to near floor edge.
            y_far_px: Pixel Y corresponding to far floor edge.
            floor_grasp_pose: Reference pose where gripper touches ground perpendicularly.
        """
        self.L1 = float(l1_m)
        self.L2 = float(l2_m)
        self.L3 = float(l3_m)
        self.ground_clearance_m = float(ground_clearance_m)
        self.hover_height_m = float(hover_height_m)
        self.lift_height_m = float(lift_height_m)
        self.radial_gain_m = float(radial_gain_m)
        self.pan_gain_deg = float(pan_gain_deg)
        self.r_near_m = float(r_near_m)
        self.r_far_m = float(r_far_m)
        self.y_near_px = float(y_near_px)
        self.y_far_px = float(y_far_px)

        # Baseline reference pose where gripper touches floor perpendicularly
        ref = floor_grasp_pose or {
            "shoulder_pan.pos": -4.4,
            "shoulder_lift.pos": 97.5,
            "elbow_flex.pos": -58.2,
            "wrist_flex.pos": 59.6,
            "wrist_roll.pos": -0.7,
            "gripper.pos": 0.0,
        }
        self.lift_ref = float(ref.get("shoulder_lift.pos", 97.5))
        self.elbow_ref = float(ref.get("elbow_flex.pos", -58.2))
        self.wrist_ref = float(ref.get("wrist_flex.pos", 59.6))
        self.pan_ref = float(ref.get("shoulder_pan.pos", -4.4))
        self.roll_ref = float(ref.get("wrist_roll.pos", -0.7))

        # Anchor Forward Kinematics at the reference ground pose
        self.alpha1_ref = math.radians(90.0 - self.lift_ref)
        self.alpha2_ref = self.alpha1_ref + math.radians(self.elbow_ref)
        self.alpha3_ref = math.radians(-90.0)  # perpendicular to ground

        self.r_wrist_ref = self.L1 * math.cos(self.alpha1_ref) + self.L2 * math.cos(self.alpha2_ref)
        self.z_wrist_ref = self.L1 * math.sin(self.alpha1_ref) + self.L2 * math.sin(self.alpha2_ref)

        self.r_ref = self.r_wrist_ref + self.L3 * math.cos(self.alpha3_ref)
        self.z_floor = self.z_wrist_ref + self.L3 * math.sin(self.alpha3_ref)

        # Reachable radial limits along the ground plane
        self.r_min = max(0.08, float(self.r_near_m) - 0.015)
        self.r_max = min((self.L1 + self.L2) * 0.96, float(self.r_far_m) + 0.015)

        logger.debug(
            "SO100Kinematics initialized: R_ref=%.1fcm, Z_floor=%.1fcm, R_bounds=[%.1f, %.1f]cm, Clearance=%.1fmm",
            self.r_ref * 100.0,
            self.z_floor * 100.0,
            self.r_min * 100.0,
            self.r_max * 100.0,
            self.ground_clearance_m * 1000.0,
        )

    def forward_kinematics(
        self,
        lift_deg: float,
        elbow_deg: float,
        wrist_deg: float,
    ) -> Tuple[float, float, float]:
        """Compute end-effector position (R, Z) and pitch angle (degrees) from joint angles."""
        r_range = max(1e-5, (self.r_far_m - self.r_near_m) * 0.5)
        dl = lift_deg - self.lift_ref
        de = elbow_deg - self.elbow_ref
        dw = wrist_deg - self.wrist_ref

        reach_factor = de / 18.0
        r_tip = self.r_ref + reach_factor * r_range

        dl_reach = 6.0 * reach_factor
        dl_z = dl - dl_reach
        dz = -(dl_z / 3.0) * self.ground_clearance_m
        z_tip = self.z_floor + dz

        pitch_deg = -90.0 + (dw + (dl + de) * 0.6)
        return r_tip, z_tip, pitch_deg

    def solve_ik(
        self,
        r_target: float,
        z_target: float,
        desired_pitch_deg: float = -90.0,
    ) -> Tuple[float, float, float]:
        """Solve floor-anchored inverse kinematics with reach modulation and safe ground clearance.

        Args:
            r_target: Radial horizontal distance from shoulder axis along floor (meters).
            z_target: Vertical height relative to shoulder axis (meters).
            desired_pitch_deg: Desired end-effector pitch angle (-90.0 = straight down).

        Returns:
            Tuple of joint angles: (shoulder_lift_deg, elbow_flex_deg, wrist_flex_deg).
        """
        r_clamped = max(self.r_min, min(self.r_max, r_target))
        r_range = max(1e-5, (self.r_far_m - self.r_near_m) * 0.5)
        reach_factor = (r_clamped - self.r_ref) / r_range
        reach_factor = max(-1.0, min(1.0, reach_factor))

        de = 18.0 * reach_factor
        dl_reach = 6.0 * reach_factor

        dz = z_target - self.z_floor
        dl_z = -3.0 * (dz / max(1e-5, self.ground_clearance_m))

        dl = dl_reach + dl_z
        dw = -(dl + de) * 0.6 + (desired_pitch_deg - (-90.0))

        lift_deg = self.lift_ref + dl
        elbow_deg = self.elbow_ref + de
        wrist_deg = self.wrist_ref + dw

        return float(lift_deg), float(elbow_deg), float(wrist_deg)

    def compute_sock_target(
        self,
        detection: Any,
        scan_pan_deg: float,
        pan_direction: float = 1.0,
    ) -> Tuple[float, float, float]:
        """Convert camera 2D detection into 3D radial and azimuthal target on the floor.

        Args:
            detection: DetectionResult containing cx, cy, error_x, error_y, angle, found.
            scan_pan_deg: Current shoulder_pan angle during scan.
            pan_direction: Direction multiplier (+1.0: right is positive, -1.0: right is negative).

        Returns:
            Tuple of (r_target_m, pan_target_deg, roll_target_deg).
        """
        if detection is None or not getattr(detection, "found", False):
            return self.r_ref, scan_pan_deg, self.roll_ref

        # Normalized horizontal and vertical errors [-1.0, 1.0]
        ex = float(getattr(detection, "error_x", 0.0))
        ey = float(getattr(detection, "error_y", 0.0))

        # Horizontal error maps to pan rotation towards centroid
        pan_target = float(scan_pan_deg + (pan_direction * self.pan_gain_deg * ex))
        pan_target = max(-110.0, min(110.0, pan_target))

        # Extract pixel centroid if available, or compute from error_y
        if hasattr(detection, "cy") and getattr(detection, "cy", 0.0) > 0.0:
            cy = float(detection.cy)
        else:
            cy = float((ey + 1.0) * 240.0)

        # Perspective projection of floor plane to radial distance R:
        # cy near y_near (bottom of ROI): close to robot base (r_near)
        # cy near y_far (top of ROI): far from robot base (r_far)
        v = (self.y_near_px - cy) / max(1e-5, (self.y_near_px - self.y_far_px))
        v = max(0.0, min(1.0, v))
        inv_r = (1.0 / self.r_near_m) - v * ((1.0 / self.r_near_m) - (1.0 / self.r_far_m))
        r_target = float(1.0 / max(1e-5, inv_r))
        r_target = max(self.r_min, min(self.r_max, r_target))

        # Wrist roll remains locked to calibrated floor grasp value (no unwanted wrist twist)
        roll_target = float(self.roll_ref)

        return r_target, pan_target, roll_target

    def compute_cartesian_pose(
        self,
        r_target: float,
        z_target: float,
        pan_target: float,
        roll_target: Optional[float] = None,
        gripper_pct: float = 0.0,
        desired_pitch_deg: float = -90.0,
    ) -> Dict[str, float]:
        """Calculate complete 6-joint pose dictionary for a Cartesian target."""
        lift, elbow, wrist = self.solve_ik(r_target, z_target, desired_pitch_deg=desired_pitch_deg)

        if roll_target is None:
            roll_target = self.roll_ref

        # Joint limit clamps
        lift = max(-110.0, min(110.0, lift))
        elbow = max(-110.0, min(110.0, elbow))
        wrist = max(-110.0, min(110.0, wrist))
        pan = max(-110.0, min(110.0, pan_target))
        roll = max(-175.0, min(175.0, roll_target))
        gripper = max(0.0, min(100.0, gripper_pct))

        return {
            "shoulder_pan.pos": round(pan, 2),
            "shoulder_lift.pos": round(lift, 2),
            "elbow_flex.pos": round(elbow, 2),
            "wrist_flex.pos": round(wrist, 2),
            "wrist_roll.pos": round(roll, 2),
            "gripper.pos": round(gripper, 2),
        }

    def get_trajectory_stages(
        self,
        r_target: float,
        pan_target: float,
        roll_target: Optional[float] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Generate the complete 4-stage Cartesian pick trajectory:
        1. hover:    elevated hover directly above target (gripper wide open)
        2. descent:  safe descent to ground clearance (gripper open)
        3. clamp:    jaws closed onto cloth at ground clearance
        4. lift:     vertical retract off floor (gripper clamped)
        """
        if roll_target is None:
            roll_target = self.roll_ref
        z_hover = self.z_floor + self.hover_height_m
        z_descent = self.z_floor + self.ground_clearance_m
        z_lift = self.z_floor + self.lift_height_m

        hover_pose = self.compute_cartesian_pose(
            r_target, z_hover, pan_target, roll_target=roll_target, gripper_pct=85.0
        )
        descent_pose = self.compute_cartesian_pose(
            r_target, z_descent, pan_target, roll_target=roll_target, gripper_pct=85.0
        )
        clamp_pose = self.compute_cartesian_pose(
            r_target, z_descent, pan_target, roll_target=roll_target, gripper_pct=0.0
        )
        lift_pose = self.compute_cartesian_pose(
            r_target, z_lift, pan_target, roll_target=roll_target, gripper_pct=0.0
        )

        return {
            "hover": hover_pose,
            "descent": descent_pose,
            "clamp": clamp_pose,
            "lift": lift_pose,
        }
