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

        self.follower = None
        self.leader = None
        self.sim_follower_pose: Dict[str, float] = dict(self.stow_pose)
        self.sim_leader_pose: Dict[str, float] = dict(self.stow_pose)

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
            return dict(self.sim_follower_pose)

        try:
            obs = self.follower.get_observation()
            pose = {}
            for k, v in obs.items():
                if not hasattr(v, "shape"):  # filter out camera images if any
                    pose[k] = float(v)
            return pose
        except Exception as e:
            logger.warning("Error reading follower pose: %s", e)
            return dict(self.sim_follower_pose)

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

    def send_follower_action(self, action: Dict[str, float]):
        """Send joint position command to Follower arm."""
        if self.dry_run or self.follower is None:
            self.sim_follower_pose.update(action)
            return

        try:
            self.follower.send_action(action)
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
        t0 = time.time()
        while True:
            elapsed = time.time() - t0
            progress = min(1.0, elapsed / max(0.001, duration_s))
            cmd_pose = cosine_interpolate(start_pose, target_pose, progress)
            self.send_follower_action(cmd_pose)

            if feedback_cb:
                feedback_cb(progress)

            if progress >= 1.0:
                break
            time.sleep(step_dt)

    def move_to_stow(self, duration_s: Optional[float] = None, feedback_cb = None):
        """Smoothly park the follower arm into the safe stowed travel pose."""
        duration = duration_s or self.interpolation_time
        start_pose = self.get_follower_pose()
        logger.info("[ARM] Moving Follower to Stow Pose over %.1fs...", duration)
        self.smooth_trajectory(start_pose, self.stow_pose, duration, feedback_cb=feedback_cb)
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
