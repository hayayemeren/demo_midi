"""SO-100 Leader Arm Reader & Joystick Kinematics Mapper.

Connects to the SO-100 Leader arm via Feetech STS3215 bus (scservo_sdk),
reads real-time joint positions with calibration normalization,
and maps joint deflections to differential-drive mobile base commands.
"""

from __future__ import annotations

import glob
import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("demo.leader")

try:
    import scservo_sdk as scs
    SCSERVO_AVAILABLE = True
except ImportError:
    SCSERVO_AVAILABLE = False
    scs = None

# Default calibration file search locations
DEFAULT_CALIBRATION_PATHS = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calibration", "so100_leader.json"),
    os.path.expanduser("~/lerobot/calibration/so100_leader.json"),
    os.path.expanduser("~/demo_midi/calibration/so100_leader.json"),
    "/home/midi/midi_lerobot/calibration/so100_leader.json",
]

DEFAULT_NEUTRAL_POSE = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": -45.0,
    "elbow_flex.pos": 15.0,
    "wrist_flex.pos": 45.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": 20.0,
}

MOTOR_NAME_TO_ID = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}

# STS3215 Control Table
ADDR_PRESENT_POSITION = 56
LEN_PRESENT_POSITION = 2
BAUDRATE_1M = 1_000_000


class SO100LeaderReader:
    """Reads joint positions from SO-100 Leader arm hardware or simulation."""

    def __init__(
        self,
        port: Optional[str] = None,
        calibration_path: Optional[str] = None,
        dry_run: bool = False,
    ):
        self.port = port
        self.calibration_path = calibration_path
        self.dry_run = dry_run

        self.port_handler = None
        self.packet_handler = None
        self.sync_reader = None
        self.is_connected = False
        self.calibration: Dict[str, Any] = {}

        self.sim_pose: Dict[str, float] = dict(DEFAULT_NEUTRAL_POSE)
        self.neutral_pose: Dict[str, float] = dict(DEFAULT_NEUTRAL_POSE)

        self._load_calibration()

    def _load_calibration(self):
        """Find and load servo calibration parameters."""
        paths = [self.calibration_path] if self.calibration_path else DEFAULT_CALIBRATION_PATHS
        for p in paths:
            if p and os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        self.calibration = json.load(f)
                    logger.info("Loaded SO-100 leader calibration from %s", p)
                    return
                except Exception as e:
                    logger.warning("Could not read calibration from %s: %s", p, e)

        # Fallback default calibration if file not found
        logger.warning("No calibration file found! Using default STS3215 calibration.")
        self.calibration = {
            "shoulder_pan": {"id": 1, "drive_mode": 0, "range_min": 726, "range_max": 3436},
            "shoulder_lift": {"id": 2, "drive_mode": 0, "range_min": 857, "range_max": 3217},
            "elbow_flex": {"id": 3, "drive_mode": 0, "range_min": 886, "range_max": 3104},
            "wrist_flex": {"id": 4, "drive_mode": 0, "range_min": 857, "range_max": 3159},
            "wrist_roll": {"id": 5, "drive_mode": 0, "range_min": 0, "range_max": 4095},
            "gripper": {"id": 6, "drive_mode": 0, "range_min": 2026, "range_max": 3282},
        }

    @staticmethod
    def auto_detect_port() -> Optional[str]:
        """Search available USB serial ports for leader arm."""
        patterns = [
            "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6084621-if00",
            "/dev/serial/by-id/*1a86*",
            "/dev/serial/by-id/*FTDI*",
            "/dev/ttyACM1",
            "/dev/ttyACM0",
            "/dev/ttyUSB1",
            "/dev/ttyUSB0",
        ]
        for pat in patterns:
            matches = glob.glob(pat)
            for m in matches:
                if os.path.exists(m):
                    return m
        return None

    def connect(self) -> bool:
        """Initialize serial connection to the Leader arm."""
        if self.dry_run:
            self.is_connected = True
            logger.info("[SIMULATION] SO-100 Leader Arm initialized in simulation mode.")
            return True

        if not SCSERVO_AVAILABLE:
            logger.warning("scservo_sdk is not available; falling back to simulation.")
            self.dry_run = True
            self.is_connected = True
            return True

        target_port = self.port or self.auto_detect_port()
        if not target_port:
            logger.warning("No SO-100 Leader Arm serial port detected.")
            return False

        try:
            self.port_handler = scs.PortHandler(target_port)
            self.packet_handler = scs.PacketHandler(0)  # Feetech protocol 0

            if not self.port_handler.openPort():
                logger.error("Failed to open leader arm serial port: %s", target_port)
                return False

            if not self.port_handler.setBaudRate(BAUDRATE_1M):
                logger.error("Failed to set 1M baudrate on %s", target_port)
                self.port_handler.closePort()
                return False

            # Set up GroupSyncRead for fast reading of all 6 servos simultaneously
            self.sync_reader = scs.GroupSyncRead(
                self.port_handler, self.packet_handler, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
            )
            for m_id in MOTOR_NAME_TO_ID.values():
                self.sync_reader.addParam(m_id)

            self.port = target_port
            self.is_connected = True
            logger.info("Connected to SO-100 Leader Arm on %s (1,000,000 baud)", target_port)
            return True
        except Exception as e:
            logger.error("Error connecting to leader arm: %s", e)
            self.is_connected = False
            return False

    def read_pose(self) -> Dict[str, float]:
        """Read all joint angles with calibration normalization.
        
        Returns:
            dict with keys: 'shoulder_pan.pos', 'shoulder_lift.pos',
            'elbow_flex.pos', 'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos'
        """
        if self.dry_run or not self.is_connected:
            return dict(self.sim_pose)

        try:
            comm_result = self.sync_reader.txRxPacket()
            if comm_result != scs.COMM_SUCCESS:
                # If sync read fails, return cached pose or try single read
                return dict(self.sim_pose)

            pose = {}
            for name, m_id in MOTOR_NAME_TO_ID.items():
                if self.sync_reader.isAvailable(m_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION):
                    raw_val = self.sync_reader.getData(m_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
                    cal = self.calibration.get(name, {})
                    min_val = cal.get("range_min", 0)
                    max_val = cal.get("range_max", 4095)
                    drive_mode = cal.get("drive_mode", 0)

                    key = f"{name}.pos"
                    if name == "gripper":
                        # Range 0 to 100%
                        span = max_val - min_val if max_val != min_val else 1
                        pct = ((raw_val - min_val) / span) * 100.0
                        pct = 100.0 - pct if drive_mode else pct
                        pose[key] = max(0.0, min(100.0, pct))
                    else:
                        # Degrees around center
                        mid = (min_val + max_val) / 2.0
                        deg = (raw_val - mid) * 360.0 / 4095.0
                        pose[key] = round(deg, 2)
                else:
                    pose[f"{name}.pos"] = self.sim_pose.get(f"{name}.pos", 0.0)

            self.sim_pose = dict(pose)
            return pose
        except Exception as e:
            logger.debug("Error during sync read: %s", e)
            return dict(self.sim_pose)

    def set_neutral_center(self, pose: Optional[Dict[str, float]] = None):
        """Set current pose as the zero/neutral reference for Base driving."""
        cur = pose or self.read_pose()
        self.neutral_pose = dict(cur)
        logger.info("[LEADER] Calibrated base neutral pose: Pan=%.1f°, Lift=%.1f°",
                    self.neutral_pose.get("shoulder_pan.pos", 0.0),
                    self.neutral_pose.get("shoulder_lift.pos", -45.0))

    def close(self):
        """Close serial port."""
        if self.port_handler:
            try:
                self.port_handler.closePort()
            except Exception:
                pass
            self.port_handler = None
        self.is_connected = False


def leader_to_base_drive(
    pose: Dict[str, float],
    neutral_pose: Dict[str, float],
    deadzone_deg: float = 6.0,
    max_pan_deg: float = 35.0,
    max_lift_deg: float = 35.0,
    turn_dir: int = -1,
) -> Tuple[float, float, bool]:
    """Map SO-100 leader arm deflection from neutral to mobile base driving commands.
    
    Args:
        pose: Current leader joint angles.
        neutral_pose: Reference neutral joint angles.
        deadzone_deg: Degrees around neutral that produce 0 motion.
        max_pan_deg: Degrees of pan deflection that produce 100% steering.
        max_lift_deg: Degrees of lift deflection that produce 100% forward/reverse.
        turn_dir: -1 to steer left when arm turned left.
        
    Returns:
        (forward: float [-1.0, 1.0], turn: float [-1.0, 1.0], brake: bool)
    """
    pan = pose.get("shoulder_pan.pos", 0.0)
    lift = pose.get("shoulder_lift.pos", -45.0)
    gripper = pose.get("gripper.pos", 0.0)

    n_pan = neutral_pose.get("shoulder_pan.pos", 0.0)
    n_lift = neutral_pose.get("shoulder_lift.pos", -45.0)

    # 1. Steering (Pan deflection)
    delta_pan = pan - n_pan
    abs_pan = abs(delta_pan)
    if abs_pan <= deadzone_deg:
        raw_turn = 0.0
    else:
        span_pan = max(1.0, max_pan_deg - deadzone_deg)
        norm_pan = min(1.0, (abs_pan - deadzone_deg) / span_pan)
        raw_turn = math.copysign(norm_pan, delta_pan)

    # Invert according to robot kinematics: Push Left -> turn left (+1.0)
    # turn_dir=-1: delta_pan < 0 (left) * -1 = +1.0 (turn left)
    turn = float(raw_turn * turn_dir)

    # 2. Forward / Reverse (Lift deflection)
    # Pushing arm forward: lift angle increases (tilts toward table)
    # Pulling arm back: lift angle decreases (tilts toward user)
    delta_lift = lift - n_lift
    abs_lift = abs(delta_lift)
    if abs_lift <= deadzone_deg:
        forward = 0.0
    else:
        span_lift = max(1.0, max_lift_deg - deadzone_deg)
        norm_lift = min(1.0, (abs_lift - deadzone_deg) / span_lift)
        forward = float(math.copysign(norm_lift, delta_lift))

    # 3. Squeeze Gripper = Active Brake
    # If gripper jaw is squeezed past 80%, engage electric brake
    brake = gripper >= 80.0

    return forward, turn, brake
