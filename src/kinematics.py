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
        self.r_min = max(0.08, abs(self.L1 - self.L2) + 0.01)
        self.r_max = (self.L1 + self.L2) * 0.96

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
        alpha1 = math.radians(90.0 - lift_deg)
        alpha2 = alpha1 + math.radians(elbow_deg)
        # Wrist angle delta relative to reference
        delta_wrist = math.radians(wrist_deg - self.wrist_ref)
        alpha3 = -math.pi / 2.0 + (alpha2 - self.alpha2_ref) + delta_wrist

        r_wrist = self.L1 * math.cos(alpha1) + self.L2 * math.cos(alpha2)
        z_wrist = self.L1 * math.sin(alpha1) + self.L2 * math.sin(alpha2)

        r_tip = r_wrist + self.L3 * math.cos(alpha3)
        z_tip = z_wrist + self.L3 * math.sin(alpha3)
        pitch_deg = math.degrees(alpha3)

        return r_tip, z_tip, pitch_deg

    def solve_ik(
        self,
        r_target: float,
        z_target: float,
        desired_pitch_deg: float = -90.0,
    ) -> Tuple[float, float, float]:
        """Solve analytical inverse kinematics for perpendicular or angled downward reach.

        Args:
            r_target: Radial horizontal distance from shoulder axis along floor (meters).
            z_target: Vertical height relative to shoulder axis (meters).
            desired_pitch_deg: Desired end-effector pitch angle (-90.0 = straight down).

        Returns:
            Tuple of joint angles: (shoulder_lift_deg, elbow_flex_deg, wrist_flex_deg).
        """
        # Clamp radial target to physically reachable limits
        r_clamped = max(self.r_min, min(self.r_max, r_target))

        alpha3 = math.radians(desired_pitch_deg)

        # 1. Wrist position from desired tip position
        r_w = r_clamped - self.L3 * math.cos(alpha3)
        z_w = z_target - self.L3 * math.sin(alpha3)

        # 2. Check reachability of wrist from shoulder (0, 0)
        D_sq = r_w**2 + z_w**2
        D = math.sqrt(D_sq)

        max_reach = (self.L1 + self.L2) * 0.995
        min_reach = abs(self.L1 - self.L2) * 1.005

        if D > max_reach:
            scale = max_reach / D
            r_w *= scale
            z_w *= scale
            D_sq = r_w**2 + z_w**2
            D = max_reach
        elif D < min_reach:
            scale = min_reach / max(1e-5, D)
            r_w *= scale
            z_w *= scale
            D_sq = r_w**2 + z_w**2
            D = min_reach

        # 3. Law of Cosines for elbow angle
        cos_gamma = (self.L1**2 + self.L2**2 - D_sq) / (2.0 * self.L1 * self.L2)
        cos_gamma = max(-1.0, min(1.0, cos_gamma))
        gamma = math.acos(cos_gamma)
        elbow_rad = -(math.pi - gamma)  # Elbow flex is negative for forward bend

        # 4. Shoulder lift angle
        psi = math.atan2(z_w, r_w)
        cos_beta = (self.L1**2 + D_sq - self.L2**2) / (2.0 * self.L1 * max(1e-5, D))
        cos_beta = max(-1.0, min(1.0, cos_beta))
        beta = math.acos(cos_beta)

        alpha1 = psi + beta
        lift_deg = 90.0 - math.degrees(alpha1)
        elbow_deg = math.degrees(elbow_rad)

        # 5. Wrist flex angle to enforce desired_pitch_deg
        alpha2 = alpha1 + elbow_rad
        wrist_deg = self.wrist_ref + (
            math.degrees(alpha3 - alpha2) - math.degrees(-math.pi / 2.0 - self.alpha2_ref)
        )

        return float(lift_deg), float(elbow_deg), float(wrist_deg)

    def compute_sock_target(
        self,
        detection: Any,
        scan_pan_deg: float,
        pan_direction: float = 1.0,
    ) -> Tuple[float, float, float]:
        """Convert camera 2D detection into 3D radial and azimuthal target on the floor.

        Args:
            detection: DetectionResult containing error_x, error_y, angle, found.
            scan_pan_deg: Current shoulder_pan angle during scan.
            pan_direction: Direction multiplier (+1.0: right is positive, -1.0: right is negative).

        Returns:
            Tuple of (r_target_m, pan_target_deg, roll_target_deg).
        """
        if detection is None or not getattr(detection, "found", False):
            return self.r_ref, scan_pan_deg, self.roll_ref

        # Horizontal error maps to pan rotation towards centroid
        ex = float(getattr(detection, "error_x", 0.0))
        pan_target = float(scan_pan_deg + (pan_direction * self.pan_gain_deg * ex))
        pan_target = max(-110.0, min(110.0, pan_target))

        # Vertical error: negative error_y = top of image = further away from base
        ey = float(getattr(detection, "error_y", 0.0))
        r_target = float(self.r_ref - (self.radial_gain_m * ey))
        r_target = max(self.r_min, min(self.r_max, r_target))

        # Oriented bounding box angle (if available)
        sock_angle = float(getattr(detection, "angle", 0.0))
        roll_target = float(self.roll_ref + max(-60.0, min(60.0, sock_angle)))

        return r_target, pan_target, roll_target

    def compute_cartesian_pose(
        self,
        r_target: float,
        z_target: float,
        pan_target: float,
        roll_target: float = 0.0,
        gripper_pct: float = 0.0,
        desired_pitch_deg: float = -90.0,
    ) -> Dict[str, float]:
        """Calculate complete 6-joint pose dictionary for a Cartesian target."""
        lift, elbow, wrist = self.solve_ik(r_target, z_target, desired_pitch_deg=desired_pitch_deg)

        # Joint limit clamps
        lift = max(-110.0, min(110.0, lift))
        elbow = max(-100.0, min(100.0, elbow))
        wrist = max(-96.0, min(96.0, wrist))
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
        roll_target: float = 0.0,
    ) -> Dict[str, Dict[str, float]]:
        """Generate the complete 4-stage Cartesian pick trajectory:
        1. hover:    elevated hover directly above target (gripper wide open)
        2. descent:  safe descent to ground clearance (gripper open)
        3. clamp:    jaws closed onto cloth at ground clearance
        4. lift:     vertical retract off floor (gripper clamped)
        """
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
