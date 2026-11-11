"""SO-100 Robot Arm Forward & Inverse Kinematics (IK) Engine.

Provides exact, zero-latency analytical kinematics calibrated to the physical
SO-100 / SO-101 URDF geometry (`so101_new_calib.urdf`):
  - Exact planar 2-link closed-form IK (shoulder_lift, elbow_flex, wrist_flex)
    selecting the physical elbow-up configuration
  - Smooth adaptive-pitch reach extension when far floor targets exceed vertical-wrist reach
  - 3D Forward Kinematics and calibrated wrist-camera ray-plane projection onto the floor
  - Anchored to the user's calibrated floor_grasp_pose Z height with configurable safe clearance
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("demo.kinematics")


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert URDF roll-pitch-yaw Euler angles (XYZ extrinsic) into a 3x3 rotation matrix."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def _make_tf(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    """Build a 4x4 homogeneous transform from URDF origin xyz and rpy."""
    T = np.eye(4, dtype=float)
    T[:3, :3] = _rpy_to_matrix(rpy[0], rpy[1], rpy[2])
    T[:3, 3] = np.asarray(xyz, dtype=float)
    return T


def _rot_z_4(rad: float) -> np.ndarray:
    """4x4 homogeneous rotation around local Z axis."""
    c, s = math.cos(rad), math.sin(rad)
    return np.array(
        [[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=float,
    )


class SO100Kinematics:
    """URDF-accurate 3D/planar Forward & Inverse Kinematics solver for SO-100 manipulator."""

    # Exact geometric constants extracted from SO-101 (`so101_new_calib.urdf`)
    PAN_X_OFFSET_M = 0.0388353
    R_SH_FROM_PAN_M = 0.0303992
    Z_SH_FROM_BASE_M = 0.11659995
    A1_ZERO_RAD = math.radians(76.03225038)
    A2_ZERO_RAD = math.radians(2.20749161)
    DX3_VERT_M = -0.00789788
    DZ3_VERT_M = -0.15922740

    # Fixed 3D URDF joint origin transforms for camera/tip ray-plane projection
    _URDF_CHAIN = (
        ("shoulder_pan", True, _make_tf([0.0388353, -8.97657e-09, 0.0624], [3.14159, 4.18253e-17, -3.14159])),
        ("shoulder_lift", True, _make_tf([-0.0303992, -0.0182778, -0.0542], [-1.5708, -1.5708, 0.0])),
        ("elbow_flex", True, _make_tf([-0.11257, -0.028, 1.73763e-16], [-3.63608e-16, 8.74301e-16, 1.5708])),
        ("wrist_flex", True, _make_tf([-0.1349, 0.0052, 3.62355e-17], [4.02456e-15, 8.67362e-16, -1.5708])),
        ("wrist_roll", True, _make_tf([5.55112e-17, -0.0611, 0.0181], [1.5708, 0.0486795, 3.14159])),
        ("gripper_frame", False, _make_tf([-0.0079, -0.000218121, -0.0981274], [0.0, 3.14159, 0.0])),
    )

    def __init__(
        self,
        l1_m: float = 0.1160,
        l2_m: float = 0.1350,
        l3_m: float = 0.1594,
        ground_clearance_m: float = 0.024,
        hover_height_m: float = 0.115,
        lift_height_m: float = 0.125,
        radial_gain_m: float = 0.050,
        pan_gain_deg: float = 60.0,
        r_near_m: float = 0.145,
        r_far_m: float = 0.315,
        y_near_px: float = 330.0,
        y_far_px: float = 60.0,
        floor_grasp_pose: Optional[Dict[str, float]] = None,
        scan_pose: Optional[Dict[str, float]] = None,
        cam_offset_gf: Tuple[float, float, float] = (-0.0900, -0.0010, -0.1045),
        cam_pitch_deg: float = -33.01,
        cam_fx: float = 531.5,
        cam_fy: Optional[float] = None,
        cam_cx: float = 320.0,
        cam_cy: float = 240.0,
        sock_height_offset_m: float = 0.015,
    ):
        """Initialize URDF-calibrated SO-100 kinematics and camera-floor projection model.

        Args:
            l1_m: Upper arm length from shoulder pitch to elbow pitch axis (meters).
            l2_m: Forearm length from elbow pitch to wrist pitch axis (meters).
            l3_m: Effective vertical length from wrist pitch to fingertips (meters).
            ground_clearance_m: Elevation above raw calibrated floor contact (meters).
            hover_height_m: Pre-grasp hover height above floor (meters).
            lift_height_m: Post-grasp vertical lift height above floor (meters).
            radial_gain_m: Legacy fallback radial modulation span (meters).
            pan_gain_deg: Azimuthal pan gain across normalized error_x [-1, 1] (degrees).
            r_near_m: Minimum safe radial reach from shoulder_pan axis (meters, clears bumper).
            r_far_m: Maximum radial reach from shoulder_pan axis along floor (meters).
            y_near_px: Fallback pixel Y for near floor boundary.
            y_far_px: Fallback pixel Y for far floor boundary.
            floor_grasp_pose: Calibrated floor-touch reference pose establishing Z_floor.
            scan_pose: Default camera observation pose for single-angle projections.
            cam_offset_gf: Wrist camera optical center offset in gripper_frame_link (meters).
            cam_pitch_deg: Wrist camera pitch tilt relative to gripper_frame_link (degrees).
            cam_fx: Camera horizontal focal length (pixels).
            cam_fy: Camera vertical focal length (pixels, defaults to cam_fx).
            cam_cx: Camera principal point X (pixels).
            cam_cy: Camera principal point Y (pixels).
            sock_height_offset_m: Target elevation of sock surface above bare floor for ray intersection.
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
        self.sock_height_offset_m = float(sock_height_offset_m)

        # Camera intrinsics & rigid mount transform on gripper_frame_link
        self.cam_offset_gf = np.asarray(cam_offset_gf, dtype=float)
        self.cam_pitch_deg = float(cam_pitch_deg)
        self.cam_fx = float(cam_fx)
        self.cam_fy = float(cam_fy if cam_fy is not None else cam_fx)
        self.cam_cx = float(cam_cx)
        self.cam_cy = float(cam_cy)

        p_rad = math.radians(self.cam_pitch_deg)
        r_gf_cam0 = np.column_stack([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        rx_pitch = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, math.cos(p_rad), -math.sin(p_rad)],
                [0.0, math.sin(p_rad), math.cos(p_rad)],
            ],
            dtype=float,
        )
        self._R_gf_cam = r_gf_cam0 @ rx_pitch

        # Calibrated baseline floor-touch pose
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

        # Default scan posture for 3D ray projection when caller only supplies scan_pan_deg
        self.default_scan_pose = dict(
            scan_pose
            or {
                "shoulder_pan.pos": -7.1,
                "shoulder_lift.pos": -63.7,
                "elbow_flex.pos": 17.3,
                "wrist_flex.pos": 96.0,
                "wrist_roll.pos": 1.5,
                "gripper.pos": 5.0,
            }
        )

        # Evaluate true URDF FK at the user-calibrated floor_grasp_pose
        self.r_ref, self.z_floor, self.pitch_ref = self.forward_kinematics(
            self.lift_ref, self.elbow_ref, self.wrist_ref
        )

        # Scale DZ3_VERT_M proportionally only if user supplies a custom non-default L3
        self._dz3_vert_m = -abs(self.L3) if abs(self.L3 - 0.1594) > 0.02 else self.DZ3_VERT_M

        # Safe radial limits from shoulder_pan axis (clears chassis front extrusion & respects max arm reach)
        self.r_min = max(0.12, float(self.r_near_m))
        self.r_max = min(0.325, float(self.r_far_m))

        logger.debug(
            "SO100Kinematics initialized: R_ref=%.1fcm, Z_floor=%.1fcm, Pitch_ref=%.1fdeg, "
            "R_bounds=[%.1f, %.1f]cm, Clearance=%.1fmm",
            self.r_ref * 100.0,
            self.z_floor * 100.0,
            self.pitch_ref,
            self.r_min * 100.0,
            self.r_max * 100.0,
            self.ground_clearance_m * 1000.0,
        )

    def _wrist_offset_for_pitch(self, pitch_deg: float) -> Tuple[float, float, float]:
        """Return (dx3, dz3, angle_sum_deg) of gripper tip relative to wrist_flex in RZ plane."""
        sum_deg = -pitch_deg  # -90 deg pitch => shoulder_lift + elbow_flex + wrist_flex = +90 deg
        dtheta = -math.radians(sum_deg - 90.0)
        dz3_0 = getattr(self, "_dz3_vert_m", self.DZ3_VERT_M)
        dx3 = self.DX3_VERT_M * math.cos(dtheta) - dz3_0 * math.sin(dtheta)
        dz3 = self.DX3_VERT_M * math.sin(dtheta) + dz3_0 * math.cos(dtheta)
        return dx3, dz3, sum_deg

    def forward_kinematics(
        self,
        lift_deg: float,
        elbow_deg: float,
        wrist_deg: float,
    ) -> Tuple[float, float, float]:
        """Compute planar end-effector position (r_pan, z_base) and pitch angle (degrees).

        Args:
            lift_deg: shoulder_lift angle in degrees.
            elbow_deg: elbow_flex angle in degrees.
            wrist_deg: wrist_flex angle in degrees.

        Returns:
            Tuple of (r_pan_m, z_base_m, pitch_deg) where r_pan_m is horizontal radial distance
            from shoulder_pan pivot axis, z_base_m is height wrt base_link origin, and
            pitch_deg = -90.0 means gripper points straight down (-Z).
        """
        a1 = self.A1_ZERO_RAD - math.radians(lift_deg)
        a2 = self.A2_ZERO_RAD - math.radians(lift_deg + elbow_deg)
        sum_deg = lift_deg + elbow_deg + wrist_deg
        pitch_deg = -sum_deg
        dx3, dz3, _ = self._wrist_offset_for_pitch(pitch_deg)

        r_tip = self.R_SH_FROM_PAN_M + self.L1 * math.cos(a1) + self.L2 * math.cos(a2) + dx3
        z_tip = self.Z_SH_FROM_BASE_M + self.L1 * math.sin(a1) + self.L2 * math.sin(a2) + dz3
        return float(r_tip), float(z_tip), float(pitch_deg)

    def forward_kinematics_3d(self, pose: Dict[str, float]) -> np.ndarray:
        """Compute full 4x4 world transform of gripper_frame_link in base_link frame."""
        T = np.eye(4, dtype=float)
        for jname, is_rev, T_orig in self._URDF_CHAIN:
            T = T @ T_orig
            if is_rev:
                ang_rad = math.radians(float(pose.get(f"{jname}.pos", 0.0)))
                T = T @ _rot_z_4(ang_rad)
        return T

    def _solve_2link_at_pitch(
        self,
        r_target: float,
        z_target: float,
        pitch_deg: float,
        allow_clamp: bool = False,
    ) -> Optional[Tuple[float, float, float]]:
        """Closed-form 2-link IK at fixed gripper pitch using the physical elbow-up branch."""
        dx3, dz3, sum_deg = self._wrist_offset_for_pitch(pitch_deg)
        wx = (r_target - dx3) - self.R_SH_FROM_PAN_M
        wz = (z_target - dz3) - self.Z_SH_FROM_BASE_M

        d_max = 0.993 * (self.L1 + self.L2)
        d_min = abs(self.L1 - self.L2) + 0.005
        d = math.hypot(wx, wz)

        if not allow_clamp and d > d_max:
            return None

        d_use = max(d_min, min(d_max, d))
        if d > 1e-6 and d != d_use:
            scale = d_use / d
            wx *= scale
            wz *= scale

        cos_psi = max(-1.0, min(1.0, (self.L1 ** 2 + d_use ** 2 - self.L2 ** 2) / (2.0 * self.L1 * d_use)))
        psi = math.acos(cos_psi)
        phi = math.atan2(wz, wx)

        # Elbow-up physical branch: a1 = phi + psi
        a1 = phi + psi
        lift_deg = math.degrees(self.A1_ZERO_RAD - a1)

        ex = wx - self.L1 * math.cos(a1)
        ez = wz - self.L1 * math.sin(a1)
        a2 = math.atan2(ez, ex)
        elbow_deg = math.degrees(self.A2_ZERO_RAD - a2) - lift_deg
        wrist_deg = sum_deg - lift_deg - elbow_deg

        if not allow_clamp:
            if not (-98.0 <= lift_deg <= 98.0 and -90.0 <= elbow_deg <= 90.0 and -95.0 <= wrist_deg <= 95.0):
                return None
        else:
            lift_deg = max(-98.0, min(98.0, lift_deg))
            elbow_deg = max(-90.0, min(90.0, elbow_deg))
            wrist_deg = max(-95.0, min(95.0, wrist_deg))

        return float(lift_deg), float(elbow_deg), float(wrist_deg)

    def solve_ik(
        self,
        r_target: float,
        z_target: float,
        desired_pitch_deg: float = -90.0,
        max_forward_tilt_deg: float = 35.0,
    ) -> Tuple[float, float, float]:
        """Solve exact SO-100 inverse kinematics for (r_pan, z_base, pitch).

        Solves cleanly with exact `desired_pitch_deg` (-90.0 = perpendicular to floor) across
        all points inside the vertical-wrist workspace. If `r_target` lies further forward
        beyond perpendicular reach, finds the steepest feasible downward pitch via binary
        search so the arm still hits `(r_target, z_target)` without stalling short.
        """
        r_clamped = max(self.r_min, min(self.r_max, float(r_target)))

        # 1. Try exact desired pitch (perpendicular -90 deg)
        exact = self._solve_2link_at_pitch(r_clamped, z_target, desired_pitch_deg, allow_clamp=False)
        if exact is not None:
            return exact

        # 2. Smoothly allow minimum necessary forward tilt (toward -90 + max_forward_tilt_deg) for far reach
        lo, hi = 0.0, float(max_forward_tilt_deg)
        best: Optional[Tuple[float, float, float]] = None
        for _ in range(18):
            mid = 0.5 * (lo + hi)
            cand = self._solve_2link_at_pitch(r_clamped, z_target, desired_pitch_deg + mid, allow_clamp=False)
            if cand is not None:
                best = cand
                hi = mid
            else:
                lo = mid

        if best is not None:
            return best

        # 3. Also test slight backward tilt (-mid) in case target is near inner boundary
        for tilt in range(2, 20, 2):
            cand = self._solve_2link_at_pitch(r_clamped, z_target, desired_pitch_deg - tilt, allow_clamp=False)
            if cand is not None:
                return cand

        # 4. Graceful boundary projection
        return self._solve_2link_at_pitch(
            r_clamped, z_target, desired_pitch_deg + max_forward_tilt_deg * 0.5, allow_clamp=True
        )

    def project_pixel_to_floor(
        self,
        pose: Dict[str, float],
        cx: float,
        cy: float,
        z_plane: Optional[float] = None,
    ) -> Optional[Tuple[float, float]]:
        """Project a camera pixel (cx, cy) onto the horizontal floor plane in base_link frame.

        Args:
            pose: 6-joint robot pose (measured or commanded) when frame was captured.
            cx: Image horizontal pixel coordinate.
            cy: Image vertical pixel coordinate.
            z_plane: Horizontal intersection height (default: z_floor + sock_height_offset_m).

        Returns:
            Tuple of (r_pan_m, pan_target_deg) or None if ray points above horizon.
        """
        if z_plane is None:
            z_plane = self.z_floor + self.sock_height_offset_m

        T_gf = self.forward_kinematics_3d(pose)
        R_wc = T_gf[:3, :3] @ self._R_gf_cam
        t_wc = T_gf[:3, :3] @ self.cam_offset_gf + T_gf[:3, 3]

        ray_cam = np.array(
            [(float(cx) - self.cam_cx) / self.cam_fx, (float(cy) - self.cam_cy) / self.cam_fy, 1.0],
            dtype=float,
        )
        ray_w = R_wc @ ray_cam
        if ray_w[2] >= -1e-3:
            return None

        t = (float(z_plane) - t_wc[2]) / ray_w[2]
        if t <= 0.02 or t > 2.5:
            return None

        pt = t_wc + t * ray_w
        dx = pt[0] - self.PAN_X_OFFSET_M
        dy = pt[1]
        r_pan = math.hypot(dx, dy)
        pan_deg = -math.degrees(math.atan2(dy, dx))
        r_pan = max(self.r_min, min(self.r_max, r_pan))
        pan_deg = max(-110.0, min(110.0, pan_deg))
        return float(r_pan), float(pan_deg)

    def compute_sock_target(
        self,
        detection: Any,
        scan_pan_deg: float,
        pan_direction: float = 1.0,
        scan_pose: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, float, float]:
        """Convert camera 2D detection into 3D radial and azimuthal floor coordinates.

        Uses exact 3D URDF ray-plane intersection whenever valid image pixel coordinates
        (cx, cy) are available; falls back cleanly to angular pan/perspective projection
        for normalized-error-only test inputs.
        """
        if detection is None or not getattr(detection, "found", False):
            return self.r_ref, scan_pan_deg, self.roll_ref

        cx = float(getattr(detection, "cx", 0.0) or 0.0)
        cy = float(getattr(detection, "cy", 0.0) or 0.0)
        ex = float(getattr(detection, "error_x", 0.0) or 0.0)
        ey = float(getattr(detection, "error_y", 0.0) or 0.0)

        # Use full 3D ray-plane intersection when real pixel coordinates are present
        if cx > 0.0 or cy > 0.0:
            pose_for_proj = dict(scan_pose) if scan_pose is not None else dict(self.default_scan_pose)
            if scan_pose is None:
                pose_for_proj["shoulder_pan.pos"] = float(scan_pan_deg)
            proj = self.project_pixel_to_floor(pose_for_proj, cx, cy)
            if proj is not None:
                r_target, pan_target = proj
                if pan_direction < 0.0:
                    pan_target = scan_pan_deg - (pan_target - float(pose_for_proj.get("shoulder_pan.pos", scan_pan_deg)))
                pan_target = max(-110.0, min(110.0, pan_target))
                return float(r_target), float(pan_target), float(self.roll_ref)

        # Fallback normalized-error mapping
        pan_target = float(scan_pan_deg + (pan_direction * self.pan_gain_deg * ex))
        pan_target = max(-110.0, min(110.0, pan_target))

        if cy <= 0.0:
            cy = float((ey + 1.0) * self.cam_cy)

        v = (self.y_near_px - cy) / max(1e-5, (self.y_near_px - self.y_far_px))
        v = max(0.0, min(1.0, v))
        inv_r = (1.0 / self.r_min) - v * ((1.0 / self.r_min) - (1.0 / self.r_max))
        r_target = float(1.0 / max(1e-5, inv_r))
        r_target = max(self.r_min, min(self.r_max, r_target))
        return r_target, pan_target, float(self.roll_ref)

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

        lift = max(-100.0, min(100.0, lift))
        elbow = max(-92.0, min(92.0, elbow))
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
        roll_target: Optional[float] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Generate the 4-stage Cartesian pick trajectory: hover -> descent -> clamp -> lift."""
        if roll_target is None:
            roll_target = self.roll_ref
        z_hover = self.z_floor + self.hover_height_m
        z_descent = self.z_floor + self.ground_clearance_m
        z_lift = self.z_floor + self.lift_height_m

        return {
            "hover": self.compute_cartesian_pose(
                r_target, z_hover, pan_target, roll_target=roll_target, gripper_pct=85.0
            ),
            "descent": self.compute_cartesian_pose(
                r_target, z_descent, pan_target, roll_target=roll_target, gripper_pct=85.0
            ),
            "clamp": self.compute_cartesian_pose(
                r_target, z_descent, pan_target, roll_target=roll_target, gripper_pct=0.0
            ),
            "lift": self.compute_cartesian_pose(
                r_target, z_lift, pan_target, roll_target=roll_target, gripper_pct=0.0
            ),
        }
