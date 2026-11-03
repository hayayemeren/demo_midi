"""Differential Drive Kinematics & Slew Rate Profiler for MIDI."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Tuple


def slew(cur: float, tgt: float, up: float, down: float, dt: float) -> float:
    """Asymmetric slew rate limiter for smooth acceleration and crisp stopping."""
    rate = up if abs(tgt) > abs(cur) and tgt * cur >= 0 else down
    s = rate * dt

    if tgt > cur:
        val = min(tgt, cur + s)
    else:
        val = max(tgt, cur - s)

    if abs(tgt) < 1e-4 and abs(val) < max(1.0, s):
        return 0.0

    return val


@dataclass
class RobotDimensions:
    """Mobile base physical geometric parameters."""
    wheel_radius: float = 0.0575   # meters (115mm diameter / 2)
    wheel_base: float = 0.350      # meters (distance between wheels)
    max_linear_speed: float = 0.8  # m/s (~2.88 km/h)
    max_angular_speed: float = 2.0 # rad/s (~115 deg/s)
    max_motor_rpm: float = 200.0   # Maximum safe RPM


class DifferentialDrive:
    """Converts twist commands (v, omega) to left/right motor RPMs."""

    def __init__(self, dimensions: RobotDimensions = None):
        self.dims = dimensions or RobotDimensions()

    def twist_to_rpm(self, linear_v: float, angular_w: float) -> Tuple[float, float]:
        """Convert linear (m/s) and angular (rad/s) velocities to (left_rpm, right_rpm)."""
        r = self.dims.wheel_radius
        b = self.dims.wheel_base

        v_left = linear_v - (angular_w * b / 2.0)
        v_right = linear_v + (angular_w * b / 2.0)

        left_rpm = (v_left / (2.0 * math.pi * r)) * 60.0
        right_rpm = (v_right / (2.0 * math.pi * r)) * 60.0

        max_val = max(abs(left_rpm), abs(right_rpm))
        if max_val > self.dims.max_motor_rpm:
            scale = self.dims.max_motor_rpm / max_val
            left_rpm *= scale
            right_rpm *= scale

        return left_rpm, right_rpm
