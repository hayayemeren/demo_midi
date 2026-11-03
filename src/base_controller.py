"""Base Controller for MIDI Mobile Base."""

from __future__ import annotations

import logging
import time
from typing import Dict, Any, Optional, Tuple

from src.ddsm import DDSMDriver, MotorFeedback
from src.diff_drive import DifferentialDrive, RobotDimensions, slew

logger = logging.getLogger("demo.base")


class BaseController:
    """Manages dual DDSM hub motors, kinematics, and electric braking."""

    def __init__(self, config: Dict[str, Any], dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run or config.get("dry_run", False)

        motors_cfg = config.get("motors", {})
        serial_cfg = config.get("serial", {})

        self.left_id = int(motors_cfg.get("left_id", 1))
        self.right_id = int(motors_cfg.get("right_id", 2))
        self.left_dir = -1 if bool(motors_cfg.get("invert_left", False)) else 1
        self.right_dir = -1 if bool(motors_cfg.get("invert_right", True)) else 1

        self.driver = DDSMDriver(
            port=serial_cfg.get("port", "/dev/ttyUSB0"),
            baudrate=int(serial_cfg.get("baudrate", 115200)),
            timeout=float(serial_cfg.get("timeout", 0.02)),
            dry_run=self.dry_run,
            motor_type=motors_cfg.get("type", "ddsm115"),
        )

        robot_cfg = config.get("robot", {})
        self.turn_ratio = float(robot_cfg.get("turn_ratio", 0.65))
        self.turn_dir = int(robot_cfg.get("turn_dir", 1))
        self.accel_rate = float(robot_cfg.get("accel_rate", 1500.0))    # RPM / s
        self.decel_rate = float(robot_cfg.get("decel_rate", 1000.0))   # RPM / s on release
        self.brake_rate = float(robot_cfg.get("brake_rate", 3000.0))   # RPM / s on brake

        self.cur_left_rpm = 0.0
        self.cur_right_rpm = 0.0
        self.is_locked = False
        self.last_fb_l: Optional[MotorFeedback] = None
        self.last_fb_r: Optional[MotorFeedback] = None

    def connect(self) -> bool:
        """Initialize RS485 communication and switch motors to Velocity Mode."""
        if not self.driver.connect():
            logger.error("Could not connect to RS485 on %s", self.driver.port)
            return False

        logger.info("Setting DDSM motors (ID %d, ID %d) to Velocity Mode...", self.left_id, self.right_id)
        self.driver.set_mode(self.left_id, DDSMDriver.MODE_VELOCITY)
        time.sleep(0.02)
        self.driver.set_mode(self.right_id, DDSMDriver.MODE_VELOCITY)
        time.sleep(0.02)
        return True

    def drive(self, forward: float, turn: float, max_rpm: float, dt: float, brake: bool = False):
        """Update motor velocities based on gamepad twist inputs and slew rates."""
        if self.is_locked or brake:
            target_l = 0.0
            target_r = 0.0
            dec = self.brake_rate
        else:
            scaled_turn = turn * self.turn_ratio * self.turn_dir
            l = forward + scaled_turn
            r = forward - scaled_turn
            scale = max(1.0, abs(l), abs(r))
            target_l = (l / scale) * max_rpm
            target_r = (r / scale) * max_rpm
            dec = self.decel_rate

        self.cur_left_rpm = slew(self.cur_left_rpm, target_l, self.accel_rate, dec, dt)
        self.cur_right_rpm = slew(self.cur_right_rpm, target_r, self.accel_rate, dec, dt)

        is_braking = self.is_locked or brake
        self.last_fb_l = self.driver.set_velocity(
            self.left_id,
            self.left_dir * self.cur_left_rpm,
            acceleration=1,
            brake=is_braking,
        )
        time.sleep(0.002)  # 2ms turnaround
        self.last_fb_r = self.driver.set_velocity(
            self.right_id,
            self.right_dir * self.cur_right_rpm,
            acceleration=1,
            brake=is_braking,
        )

    def lock_brakes(self):
        """Engage electric brakes and lock wheels completely stationary."""
        self.is_locked = True
        self.cur_left_rpm = 0.0
        self.cur_right_rpm = 0.0
        self.driver.emergency_brake(self.left_id)
        time.sleep(0.002)
        self.driver.emergency_brake(self.right_id)
        logger.info("[BASE] Electric brakes locked.")

    def release_brakes(self):
        """Release electric brakes to allow driving."""
        self.is_locked = False
        self.cur_left_rpm = 0.0
        self.cur_right_rpm = 0.0
        self.driver.set_velocity(self.left_id, 0.0, brake=False)
        time.sleep(0.002)
        self.driver.set_velocity(self.right_id, 0.0, brake=False)
        logger.info("[BASE] Electric brakes released.")

    def stop(self):
        """Gracefully decelerate motors to 0 and halt."""
        logger.info("Stopping base...")
        t0 = time.time()
        while (abs(self.cur_left_rpm) > 1 or abs(self.cur_right_rpm) > 1) and time.time() - t0 < 1.0:
            self.cur_left_rpm = slew(self.cur_left_rpm, 0.0, 0.0, self.brake_rate, 0.02)
            self.cur_right_rpm = slew(self.cur_right_rpm, 0.0, 0.0, self.brake_rate, 0.02)
            self.driver.set_velocity(self.left_id, self.left_dir * self.cur_left_rpm, acceleration=1)
            time.sleep(0.002)
            self.driver.set_velocity(self.right_id, self.right_dir * self.cur_right_rpm, acceleration=1)
            time.sleep(0.02)

        try:
            self.driver.emergency_brake(self.left_id)
            time.sleep(0.002)
            self.driver.emergency_brake(self.right_id)
        except Exception:
            pass

        self.driver.close()
        logger.info("Base stopped safely.")
