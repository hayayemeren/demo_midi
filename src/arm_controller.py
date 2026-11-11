"""Arm Controller for SO-100 Follower and Leader Arms.

Manages connection lifecycle, smooth jerk-free trajectory interpolation,
stow/rest positioning, leader arm discovery, and real-time teleoperation.
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger("demo.arm")

DEFAULT_STOW_POSE = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": -95.0,
    "elbow_flex.pos": 25.0,
    "wrist_flex.pos": 95.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": 5.0,
}

DEFAULT_READY_POSE = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": -45.0,
    "elbow_flex.pos": 15.0,
    "wrist_flex.pos": 45.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": 50.0,
}

DEFAULT_JOINT_LIMITS = {
    "shoulder_pan.pos": (-110.0, 110.0),
    "shoulder_lift.pos": (-100.0, 100.0),  # Accommodates floor reach up to 95° safely inside [-104.1°, 104.1°]
    "elbow_flex.pos": (-92.0, 92.0),       # Accommodates basket (-80.4°) and floor (-56.2°) inside [-97.1°, 97.1°]
    "wrist_flex.pos": (-96.0, 96.0),       # 5.7° safety margin inside [-101.7°, 101.7°] to prevent motor buzzing against endstops
    "wrist_roll.pos": (-175.0, 175.0),     # 5.0° safety margin inside continuous [-180.0°, 180.0°]
    "gripper.pos": (0.0, 100.0),
}


def cosine_interpolate(
    start_pose: Dict[str, float],
    target_pose: Dict[str, float],
    progress: float,  # 0.0 to 1.0
) -> Dict[str, float]:
    """Calculate intermediate joint positions using S-curve (cosine) easing.
    
    Zero start velocity, zero end velocity, minimal mechanical stress.
    """
    alpha = 0.5 * (1.0 - math.cos(math.pi * max(0.0, min(1.0, progress))))
    interpolated = {}
    for k in start_pose:
        if k in target_pose:
            interpolated[k] = start_pose[k] + alpha * (target_pose[k] - start_pose[k])
        else:
            interpolated[k] = start_pose[k]
    return interpolated


class ArmController:
    """Manages SO-100 Follower and Leader arms via LeRobot."""

    def __init__(self, config: Dict[str, Any], dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run or config.get("dry_run", False)

        arm_cfg = config.get("arm", {})
        self.follower_port = arm_cfg.get(
            "follower_port",
            "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6084008-if00"
        )
        self.leader_port = arm_cfg.get(
            "leader_port",
            "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6084621-if00"
        )
        self.follower_id = arm_cfg.get("follower_id", "so100_follower")
        self.leader_id = arm_cfg.get("leader_id", "so100_leader")
        self.use_degrees = bool(arm_cfg.get("use_degrees", True))
        self.stow_pose = arm_cfg.get("stow_pose", DEFAULT_STOW_POSE)
        self.interpolation_time = float(arm_cfg.get("interpolation_time_s", 1.5))
        self.invert_shoulder_pan = bool(arm_cfg.get("invert_shoulder_pan", False))
        self.invert_wrist_roll = bool(arm_cfg.get("invert_wrist_roll", False))

        # Gamepad Arm Teleoperation Speeds & Limits (Case 2)
        gp_arm_cfg = config.get("arm_gamepad_teleop", {})
        self.pan_speed_dps = float(gp_arm_cfg.get("pan_speed_dps", 50.0))
        self.lift_speed_dps = float(gp_arm_cfg.get("lift_speed_dps", 45.0))
        self.elbow_speed_dps = float(gp_arm_cfg.get("elbow_speed_dps", 50.0))
        self.flex_speed_dps = float(gp_arm_cfg.get("flex_speed_dps", 45.0))
        self.roll_speed_dps = float(gp_arm_cfg.get("roll_speed_dps", 60.0))
        self.gripper_speed_pct_s = float(gp_arm_cfg.get("gripper_speed_pct_s", 60.0))
        self.ready_pose = arm_cfg.get("ready_pose", DEFAULT_READY_POSE)
        self.joint_limits = arm_cfg.get("joint_limits", DEFAULT_JOINT_LIMITS)
        self.current_gamepad_target: Dict[str, float] = dict(self.stow_pose)

        self.follower = None
        self.leader = None
        self.sim_follower_pose: Dict[str, float] = dict(self.stow_pose)
        self.sim_leader_pose: Dict[str, float] = dict(self.stow_pose)
        self.last_known_follower_pose: Dict[str, float] = dict(self.stow_pose)
        self._last_sent_gamepad_pose: Optional[Dict[str, float]] = None
        self._last_gamepad_tx_time: float = 0.0

    def is_follower_port_available(self) -> bool:
        """Check if follower serial device exists."""
        if self.dry_run:
            return True
        return os.path.exists(self.follower_port)

    def is_leader_port_available(self) -> bool:
        """Check if leader serial device exists."""
        if self.dry_run:
            return True
        return os.path.exists(self.leader_port)

    def connect_follower(self) -> bool:
        """Connect to the Follower robot arm."""
        if self.dry_run:
            logger.info("[DRY RUN] Follower Arm initialized in simulation.")
            return True

        if self.is_follower_connected():
            return True

        if not self.is_follower_port_available():
            logger.warning("Follower port %s does not exist!", self.follower_port)
            return False

        try:
            from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
            cfg = SOFollowerRobotConfig(
                id=self.follower_id,
                port=self.follower_port,
                use_degrees=self.use_degrees,
            )
            self.follower = SOFollower(cfg)
            self.follower.connect(calibrate=False)
            logger.info("Follower Arm connected successfully on %s", self.follower_port)
            return True
        except Exception as e:
            logger.error("Failed to connect Follower Arm: %s", e)
            self.follower = None
            return False

    def is_follower_connected(self) -> bool:
        """Return True if follower is active and connected."""
        if self.dry_run:
            return True
        if self.follower is None or not getattr(self.follower, "is_connected", True):
            return False
        return True

    def connect_leader(self) -> bool:
        """Connect to the Leader teleoperator arm."""
        if self.dry_run:
            logger.info("[DRY RUN] Leader Arm connected in simulation.")
            return True

        if not self.is_leader_port_available():
            logger.debug("Leader port %s not found yet.", self.leader_port)
            return False

        try:
            from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig
            cfg = SOLeaderTeleopConfig(
                id=self.leader_id,
                port=self.leader_port,
                use_degrees=self.use_degrees,
            )
            self.leader = SOLeader(cfg)
            self.leader.connect(calibrate=False)
            logger.info("Leader Arm connected successfully on %s", self.leader_port)
            return True
        except Exception as e:
            logger.error("Failed to connect Leader Arm: %s", e)
            self.leader = None
            return False

    def is_leader_connected(self) -> bool:
        """Return True if leader is active and responding."""
        if self.dry_run:
            return self.leader is not None
        if self.leader is None or not self.leader.is_connected:
            return False
        return True

    def get_follower_pose(self) -> Dict[str, float]:
        """Read current Follower joint angles."""
        if self.dry_run or self.follower is None:
            self.last_known_follower_pose = dict(self.sim_follower_pose)
            return dict(self.sim_follower_pose)

        try:
            obs = self.follower.get_observation()
            pose = {}
            for k, v in obs.items():
                if not hasattr(v, "shape"):  # filter out camera images if any
                    pose[k] = float(v)
            self.last_known_follower_pose = dict(pose)
            return pose
        except Exception as e:
            logger.warning("Error reading follower pose: %s", e)
            return dict(self.last_known_follower_pose)

    def get_leader_pose(self) -> Dict[str, float]:
        """Read current Leader joint angles, applying joint inversions if configured."""
        if self.dry_run or self.leader is None:
            pose = dict(self.sim_leader_pose)
        else:
            try:
                act = self.leader.get_action()
                pose = {k: float(v) for k, v in act.items()}
            except Exception as e:
                logger.warning("Error reading leader pose: %s", e)
                pose = dict(self.sim_leader_pose)

        # Invert joints (e.g. left/right pan direction matching)
        if self.invert_shoulder_pan and "shoulder_pan.pos" in pose:
            pose["shoulder_pan.pos"] = -pose["shoulder_pan.pos"]
        if self.invert_wrist_roll and "wrist_roll.pos" in pose:
            pose["wrist_roll.pos"] = -pose["wrist_roll.pos"]

        return pose

    def clamp_to_joint_limits(self, pose: Dict[str, float]) -> Dict[str, float]:
        """Clamp joint positions strictly within safe hardware limits to prevent mechanical strain/buzzing."""
        clamped = {}
        for k, v in pose.items():
            if k in self.joint_limits:
                j_min, j_max = self.joint_limits[k]
                clamped[k] = float(max(j_min, min(j_max, v)))
            else:
                clamped[k] = float(v)
        return clamped

    def send_follower_action(self, action: Dict[str, float]):
        """Send joint position command to Follower arm, clamped to safe hardware limits."""
        clamped = self.clamp_to_joint_limits(action)
        self.last_known_follower_pose.update(clamped)
        if self.dry_run or self.follower is None:
            self.sim_follower_pose.update(clamped)
            return

        try:
            self.follower.send_action(clamped)
        except Exception as e:
            logger.warning("Error sending action to follower: %s", e)

    def smooth_trajectory(
        self,
        start_pose: Dict[str, float],
        target_pose: Dict[str, float],
        duration_s: float = 1.5,
        step_dt: float = 0.02,
        feedback_cb: Optional[Callable[[float], None]] = None,
    ):
        """Execute smooth cosine trajectory from start_pose to target_pose."""
        effective_duration = min(duration_s, 0.05) if self.dry_run else duration_s
        effective_dt = min(step_dt, 0.01) if self.dry_run else step_dt
        t0 = time.time()
        while True:
            step_start = time.time()
            elapsed = step_start - t0
            progress = min(1.0, elapsed / max(0.001, effective_duration))
            cmd_pose = cosine_interpolate(start_pose, target_pose, progress)
            self.send_follower_action(cmd_pose)

            if feedback_cb:
                feedback_cb(progress)

            if progress >= 1.0:
                break
            rest = effective_dt - (time.time() - step_start)
            if rest > 0:
                time.sleep(rest)

    def move_to_stow(self, duration_s: Optional[float] = None, feedback_cb = None):
        """Smoothly park the follower arm into the safe stowed travel pose.
        
        Handles both rear/basket postures (uncurling elbow/wrist forward) and forward
        floor-reach postures (raising shoulder above chassis before folding wrist/elbow
        so the jaws never hook underneath the front chassis bar).
        """
        duration = duration_s or self.interpolation_time
        start_pose = self.get_follower_pose()

        lift = start_pose.get("shoulder_lift.pos", -95.0)
        elbow = start_pose.get("elbow_flex.pos", 25.0)
        wrist = start_pose.get("wrist_flex.pos", 95.0)

        if lift > 15.0:
            logger.info("[ARM] Elevating from forward floor reach cleanly before stowing...")
            clear_front = dict(start_pose)
            clear_front["shoulder_pan.pos"] = 0.0
            clear_front["shoulder_lift.pos"] = -35.0
            # Keep forearm gently folded upward/compact to reduce shoulder gravity torque without tucking under bar
            clear_front["elbow_flex.pos"] = 10.0
            clear_front["wrist_flex.pos"] = -35.0
            clear_front["wrist_roll.pos"] = 0.0
            clear_front["gripper.pos"] = min(20.0, start_pose.get("gripper.pos", 20.0))
            clear_t = 0.05 if self.dry_run else max(2.0, duration)
            self.smooth_trajectory(start_pose, clear_front, duration_s=clear_t, feedback_cb=feedback_cb)
            start_pose = self.get_follower_pose() if not self.dry_run else clear_front
        elif elbow < -10.0 or wrist < -10.0:
            logger.info("[ARM] Clearing chassis/basket before parking into stow pose...")
            clearance_pose = dict(start_pose)
            # Elevate shoulder slightly, uncurl elbow and wrist forward, reset roll
            cur_lift = start_pose.get("shoulder_lift.pos", -35.0)
            clearance_pose["shoulder_lift.pos"] = min(-30.0, max(-60.0, cur_lift + 15.0 if cur_lift < -45.0 else -35.0))
            clearance_pose["elbow_flex.pos"] = 15.0
            clearance_pose["wrist_flex.pos"] = 50.0
            clearance_pose["wrist_roll.pos"] = 0.0
            clearance_pose["gripper.pos"] = min(20.0, start_pose.get("gripper.pos", 20.0))
            self.smooth_trajectory(start_pose, clearance_pose, duration_s=1.2, feedback_cb=feedback_cb)
            start_pose = clearance_pose

        logger.info("[ARM] Moving Follower to Stow Pose over %.1fs...", duration)
        self.smooth_trajectory(start_pose, self.stow_pose, duration, feedback_cb=feedback_cb)
        if not self.dry_run:
            # Allow brief settle check so shoulder passes behind center of gravity before releasing
            t_wait = time.time()
            while time.time() - t_wait < 0.6:
                cur_l = self.get_follower_pose().get("shoulder_lift.pos", 0.0)
                if cur_l <= self.stow_pose.get("shoulder_lift.pos", -95.0) + 15.0:
                    break
                time.sleep(0.05)
        self.current_gamepad_target = dict(self.stow_pose)
        logger.info("[ARM] Follower is parked safely at Stow Pose.")

    def sync_to_leader(self, duration_s: Optional[float] = None, feedback_cb = None):
        """Smoothly align Follower arm with the Leader arm's current pose."""
        duration = duration_s or self.interpolation_time
        start_pose = self.get_follower_pose()
        target_pose = self.get_leader_pose()
        logger.info("[ARM] Syncing Follower to Leader Pose over %.1fs...", duration)
        self.smooth_trajectory(start_pose, target_pose, duration, feedback_cb=feedback_cb)
        logger.info("[ARM] Arms aligned! Ready for teleoperation.")

    def teleop_step(self) -> Dict[str, float]:
        """Perform one 1:1 mirroring step: Leader -> Follower."""
        action = self.get_leader_pose()
        self.send_follower_action(action)
        return action

    def init_gamepad_teleop(self):
        """Initialize gamepad target pose from current follower observation."""
        cur = self.get_follower_pose()
        self.current_gamepad_target = {
            k: cur.get(k, self.stow_pose.get(k, 0.0))
            for k in self.stow_pose
        }
        self._last_sent_gamepad_pose = None
        logger.info("[ARM] Gamepad teleop initialized at: %s",
                    {k: round(v, 1) for k, v in self.current_gamepad_target.items()})

    def move_to_ready(self, duration_s: Optional[float] = None, feedback_cb = None):
        """Smoothly move follower arm to ready/work pose."""
        duration = duration_s or self.interpolation_time
        start_pose = self.get_follower_pose()
        logger.info("[ARM] Moving Follower to Ready Pose over %.1fs...", duration)
        self.smooth_trajectory(start_pose, self.ready_pose, duration, feedback_cb=feedback_cb)
        self.current_gamepad_target = dict(self.ready_pose)
        self._last_sent_gamepad_pose = dict(self.ready_pose)
        logger.info("[ARM] Follower is ready for manipulation.")

    def move_to_pose(
        self,
        target_pose: Dict[str, float],
        duration_s: Optional[float] = None,
        feedback_cb: Optional[Callable[[float], None]] = None,
    ):
        """Smoothly move follower arm to arbitrary target pose using S-curve trajectory."""
        duration = duration_s or self.interpolation_time
        start_pose = self.get_follower_pose()
        # Ensure all joints exist in target_pose (fallback to start_pose) and clamp to safe hardware limits
        full_target = {k: target_pose.get(k, start_pose.get(k, 0.0)) for k in start_pose}
        clamped_target = self.clamp_to_joint_limits(full_target)
        self.smooth_trajectory(start_pose, clamped_target, duration, feedback_cb=feedback_cb)
        self.current_gamepad_target = dict(clamped_target)

    def teleop_step_gamepad(self, gp: Any, dt: float) -> Dict[str, float]:
        """Perform one rate-control teleoperation step from Gamepad inputs.
        
        Controls:
          - Right Stick X: Shoulder Pan (Left / Right)
          - Right Stick Y: Shoulder Lift (Up / Down)
          - Left Stick Y: Elbow Flex (Reach / Pull back)
          - Left Stick X: Wrist Roll (Roll CW / CCW)
          - D-Pad Y: Wrist Flex (Pitch Up / Down)
          - RB (held): Close Gripper
          - LB (held): Open Gripper
          - Button A: Toggle Gripper (Open <-> Close)
          - Button X: Move to Ready Pose
          - Button B: Emergency Freeze
        """
        if not getattr(gp, "connected", True) or getattr(gp, "brake", False) or getattr(gp, "button_b_held", False):
            return dict(self.current_gamepad_target)

        # 1. Preset shortcut: Button X moves to Ready Pose
        if gp.button_x_pressed:
            logger.info("[ARM] Button [X] pressed -> Moving to Ready Pose...")
            self.move_to_ready(duration_s=1.0)
            return dict(self.current_gamepad_target)

        # 2. Right stick: Pan (X) and Lift (Y)
        # Push Right (+rs_x) -> Pan Right (+), Push Left (-rs_x) -> Pan Left (-)
        pan_delta = gp.right_stick_x * self.pan_speed_dps * dt
        # Push Up (+rs_y) -> Lift Up (+), Pull Down (-rs_y) -> Lift Down (-)
        lift_delta = gp.right_stick_y * self.lift_speed_dps * dt

        # 3. Left stick: Elbow Flex (Y) and Wrist Roll (X)
        # Push Up (+ls_y) -> Extend (+), Pull Down (-ls_y) -> Flex (-)
        elbow_delta = gp.left_stick_y * self.elbow_speed_dps * dt
        # Push Right (+ls_x) -> Roll CW (+), Push Left (-ls_x) -> Roll CCW (-)
        roll_delta = gp.left_stick_x * self.roll_speed_dps * dt

        # 4. D-Pad: Wrist Flex pitch (Up / Down) and fine roll (Left / Right)
        flex_delta = float(gp.dpad_y) * self.flex_speed_dps * dt
        if gp.dpad_x != 0:
            roll_delta += float(gp.dpad_x) * (self.roll_speed_dps * 0.5) * dt

        # 5. Integrate joint targets
        self.current_gamepad_target["shoulder_pan.pos"] += pan_delta
        self.current_gamepad_target["shoulder_lift.pos"] += lift_delta
        self.current_gamepad_target["elbow_flex.pos"] += elbow_delta
        self.current_gamepad_target["wrist_flex.pos"] += flex_delta
        self.current_gamepad_target["wrist_roll.pos"] += roll_delta

        # 6. Gripper rate control: RB closes, LB opens
        if gp.button_rb_held:
            self.current_gamepad_target["gripper.pos"] += self.gripper_speed_pct_s * dt
        elif gp.button_lb_held:
            self.current_gamepad_target["gripper.pos"] -= self.gripper_speed_pct_s * dt

        # Button A toggles gripper
        if gp.button_a_pressed:
            if self.current_gamepad_target["gripper.pos"] > 40.0:
                self.current_gamepad_target["gripper.pos"] = 0.0
            else:
                self.current_gamepad_target["gripper.pos"] = 85.0

        # 7. Clamp to safe calibrated joint limits
        for joint, (j_min, j_max) in self.joint_limits.items():
            if joint in self.current_gamepad_target:
                self.current_gamepad_target[joint] = max(
                    j_min, min(j_max, self.current_gamepad_target[joint])
                )

        # 8. Command follower arm (skip redundant 50Hz serial writes when target hasn't changed)
        now = time.time()
        changed = (
            self._last_sent_gamepad_pose is None
            or any(abs(self.current_gamepad_target.get(k, 0.0) - self._last_sent_gamepad_pose.get(k, 0.0)) > 1e-3 for k in self.current_gamepad_target)
        )
        if self.dry_run or changed or (now - self._last_gamepad_tx_time >= 0.15):
            self.send_follower_action(self.current_gamepad_target)
            self._last_sent_gamepad_pose = dict(self.current_gamepad_target)
            self._last_gamepad_tx_time = now
        return dict(self.current_gamepad_target)

    def disconnect(self):
        """Safely disconnect arms and disable torque."""
        logger.info("Disconnecting arms...")
        if self.follower:
            try:
                self.follower.disconnect()
            except Exception:
                pass
            self.follower = None

        if self.leader:
            try:
                self.leader.disconnect()
            except Exception:
                pass
            self.leader = None
        logger.info("Arms disconnected safely.")
