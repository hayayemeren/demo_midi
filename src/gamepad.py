"""Logitech Wireless Gamepad F710 Driver using Python evdev.

Provides synchronous, zero-latency polling matching hardware evdev capabilities.
Supports both XInput (X mode) and DirectInput (D mode) on Linux.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger("demo.gamepad")

try:
    import evdev
    from evdev import ecodes as E
    EVDEV_AVAILABLE = True
except ImportError:
    EVDEV_AVAILABLE = False
    E = None

# Gamepad Button Codes for XInput and DirectInput modes
BTN_CODE_BRAKE = {E.BTN_EAST, E.BTN_THUMB2, 0x131, 0x122} if E else {0x131, 0x122}  # Button B
BTN_CODE_LB    = {E.BTN_TL, E.BTN_TOP2, 0x136, 0x124} if E else {0x136, 0x124}      # Left Bumper (Speed UP)
BTN_CODE_RB    = {E.BTN_TR, E.BTN_PINKIE, 0x137, 0x125} if E else {0x137, 0x125}    # Right Bumper (Speed DOWN)
BTN_CODE_START = {E.BTN_START, E.BTN_BASE4, 0x13b, 0x129} if E else {0x13b, 0x129}  # START button

# Mode Switch Buttons
BTN_CODE_Y     = {E.BTN_WEST, 0x134, 0x123} if E else {0x134, 0x123}                # Button Y (Top button)
BTN_CODE_X     = {E.BTN_NORTH, 0x133, 0x120} if E else {0x133, 0x120}               # Button X (Left button)
BTN_CODE_A     = {E.BTN_SOUTH, 0x130, 0x121} if E else {0x130, 0x121}               # Button A (Bottom button)
BTN_CODE_BACK  = {E.BTN_SELECT, 0x13a, 0x128} if E else {0x13a, 0x128}             # Back / Select button


@dataclass
class GamepadState:
    """Snapshot of gamepad inputs."""
    forward: float = 0.0          # -1.0 (Reverse) to +1.0 (Forward)
    turn: float = 0.0             # -1.0 (Right/CW) to +1.0 (Left/CCW)
    brake: bool = False           # True if B is held
    # Extended 6-DOF Stick & D-Pad inputs for Arm Teleoperation
    left_stick_x: float = 0.0     # -1.0 (Left) to +1.0 (Right)
    left_stick_y: float = 0.0     # -1.0 (Down) to +1.0 (Up)
    right_stick_x: float = 0.0    # -1.0 (Left) to +1.0 (Right)
    right_stick_y: float = 0.0    # -1.0 (Down) to +1.0 (Up)
    dpad_x: int = 0               # -1 (Left), +1 (Right), 0 (Neutral)
    dpad_y: int = 0               # +1 (Up), -1 (Down), 0 (Neutral)
    button_lb_held: bool = False  # Continuous LB hold (Gripper Open)
    button_rb_held: bool = False  # Continuous RB hold (Gripper Close)
    button_y_pressed: bool = False      # Switch mode toggle (Y)
    button_x_pressed: bool = False      # Work pose toggle (X)
    button_a_pressed: bool = False      # Confirm / Gripper toggle (A)
    button_b_held: bool = False         # Emergency stop / freeze (B)
    button_lb_pressed: bool = False     # Speed UP
    button_rb_pressed: bool = False     # Speed DOWN
    button_start_pressed: bool = False  # Exit program
    button_back_pressed: bool = False   # Back button
    connected: bool = False
    device_name: str = "None"
    speed_level: int = 1                # Current speed level index
    speed_rpm: float = 100.0


class F710Gamepad:
    """Synchronous Gamepad reader using evdev (Zero-lag)."""

    def __init__(
        self,
        device_path: Optional[str] = None,
        deadzone: float = 0.10,
        speed_levels: Optional[List[float]] = None,
        default_speed_level: int = 1,
        switch_button: str = "Y",
    ):
        self.device_path = device_path
        self.deadzone = deadzone
        self.speed_levels = speed_levels or [50.0, 100.0, 150.0, 200.0]
        self.speed_level = max(0, min(len(self.speed_levels) - 1, default_speed_level))
        self.switch_button = switch_button.upper()

        self.dev = None
        self.ax: Dict[int, Tuple[int, int]] = {}     # code -> (min, max)
        self.val: Dict[int, int] = {}                # code -> current raw value
        self.btn: Set[int] = set()                   # currently held buttons
        self.pressed: List[int] = []                 # buttons pressed this cycle
        self.turn_code: Optional[int] = None
        self.connected = False
        self.device_name = "Not Connected"
        self._last_reconnect_attempt = 0.0

    @staticmethod
    def inspect(path: str):
        """Check if device has necessary sticks and keys for Logitech gamepad."""
        if not EVDEV_AVAILABLE:
            return None, {}, set(), False, "evdev library not installed"
        try:
            d = evdev.InputDevice(path)
            caps = dict(d.capabilities(absinfo=True))
            axes = {c: info for c, info in caps.get(E.EV_ABS, [])}
            keys = set(caps.get(E.EV_KEY, []))
            pad_keys = any(0x120 <= k <= 0x13F for k in keys)

            if not (E.ABS_X in axes and E.ABS_Y in axes):
                return d, axes, keys, False, "Left stick (ABS_X/ABS_Y) missing"
            if not pad_keys:
                return d, axes, keys, False, "Gamepad button keys missing"
            if not (E.ABS_RX in axes or E.ABS_Z in axes):
                return d, axes, keys, False, "Right stick (ABS_RX/ABS_Z) missing"

            return d, axes, keys, True, "OK"
        except Exception as e:
            return None, {}, set(), False, str(e)

    def connect(self, force_path: Optional[str] = None, verbose: bool = False) -> bool:
        """Find and open the Logitech F710 device."""
        if not EVDEV_AVAILABLE:
            logger.warning("evdev library is not available.")
            return False

        paths = [force_path] if force_path else evdev.list_devices()
        for path in paths:
            try:
                d, axes, keys, ok, why = self.inspect(path)
            except Exception as e:
                if verbose:
                    logger.debug("Device %s check failed: %s", path, e)
                continue

            if not ok:
                if verbose:
                    logger.debug("Skipping %s: %s", path, why)
                continue

            # Right stick X: XInput is ABS_RX, DirectInput is ABS_Z
            turn = E.ABS_RX if E.ABS_RX in axes else E.ABS_Z
            self.dev = d
            self.turn_code = turn
            self.ax = {c: (i.min, i.max) for c, i in axes.items()}
            self.val = {c: i.value for c, i in axes.items()}
            self.btn.clear()
            self.connected = True
            self.device_name = d.name

            mode_str = "XInput (X)" if turn == E.ABS_RX else "DirectInput (D)"
            logger.info("Gamepad connected: %s on %s (%s)", d.name, path, mode_str)
            return True

        self.connected = False
        return False

    def norm(self, code: int) -> float:
        """Normalize analog axis value using hardware calibrated range."""
        lo, hi = self.ax.get(code, (-32767, 32767))
        center = (lo + hi) / 2.0
        half_span = (hi - lo) / 2.0 or 1.0

        v = self.val.get(code, int(center))
        x = (v - center) / half_span
        x = max(-1.0, min(1.0, x))

        if abs(x) < self.deadzone:
            return 0.0

        return math.copysign((abs(x) - self.deadzone) / (1.0 - self.deadzone), x)

    def poll(self) -> GamepadState:
        """Synchronously drain all pending kernel events and return real-time state."""
        self.pressed = []
        now = time.time()

        if self.dev is None:
            if now - self._last_reconnect_attempt > 1.0:
                self._last_reconnect_attempt = now
                self.connect(self.device_path)
            if self.dev is None:
                return GamepadState(
                    connected=False,
                    speed_level=self.speed_level,
                    speed_rpm=self.speed_levels[self.speed_level],
                )

        try:
            # Drain all pending events in kernel buffer
            while True:
                ev = self.dev.read_one()
                if ev is None:
                    break
                if ev.type == E.EV_ABS:
                    self.val[ev.code] = ev.value
                elif ev.type == E.EV_KEY:
                    if ev.value == 1:
                        self.btn.add(ev.code)
                        self.pressed.append(ev.code)
                    elif ev.value == 0:
                        self.btn.discard(ev.code)

        except OSError:
            logger.warning("Gamepad disconnected!")
            self.dev = None
            self.connected = False
            return GamepadState(
                connected=False,
                speed_level=self.speed_level,
                speed_rpm=self.speed_levels[self.speed_level],
            )

        # Handle speed level adjustments: LB = speed up, RB = speed down
        for btn in self.pressed:
            if btn in BTN_CODE_LB:
                self.speed_level = min(len(self.speed_levels) - 1, self.speed_level + 1)
                logger.info("[GAMEPAD] Speed UP (LB): Level %d (%.0f RPM)", self.speed_level + 1, self.speed_levels[self.speed_level])
            elif btn in BTN_CODE_RB:
                self.speed_level = max(0, self.speed_level - 1)
                logger.info("[GAMEPAD] Speed DOWN (RB): Level %d (%.0f RPM)", self.speed_level + 1, self.speed_levels[self.speed_level])

        # Compute analog stick values:
        # Left Stick Y: Push Up = +1.0, Pull Down = -1.0
        ls_y = -self.norm(E.ABS_Y)
        # Left Stick X: Push Right = +1.0, Push Left = -1.0
        ls_x = self.norm(E.ABS_X)

        # Right Stick X: Push Right = +1.0, Push Left = -1.0
        rs_x = self.norm(self.turn_code) if self.turn_code else 0.0
        # Right Stick Y: Push Up = +1.0, Pull Down = -1.0
        ry_code = E.ABS_RY if E.ABS_RY in self.ax else E.ABS_RZ
        rs_y = -self.norm(ry_code) if ry_code in self.ax else 0.0

        # D-pad (POV Hat)
        dpad_x = int(self.val.get(E.ABS_HAT0X, 0))
        dpad_y = -int(self.val.get(E.ABS_HAT0Y, 0))  # Up = +1, Down = -1

        # Base driving inputs
        raw_fwd = ls_y
        raw_turn = -rs_x  # Push Left = +1.0, Push Right = -1.0

        # Buttons held and pressed
        brake_active = bool(BTN_CODE_BRAKE & self.btn)
        lb_held = bool(BTN_CODE_LB & self.btn)
        rb_held = bool(BTN_CODE_RB & self.btn)
        start_pressed = bool(BTN_CODE_START & set(self.pressed))
        y_pressed = bool(BTN_CODE_Y & set(self.pressed))
        x_pressed = bool(BTN_CODE_X & set(self.pressed))
        a_pressed = bool(BTN_CODE_A & set(self.pressed))
        back_pressed = bool(BTN_CODE_BACK & set(self.pressed))

        return GamepadState(
            forward=raw_fwd,
            turn=raw_turn,
            brake=brake_active,
            left_stick_x=ls_x,
            left_stick_y=ls_y,
            right_stick_x=rs_x,
            right_stick_y=rs_y,
            dpad_x=dpad_x,
            dpad_y=dpad_y,
            button_lb_held=lb_held,
            button_rb_held=rb_held,
            button_y_pressed=y_pressed,
            button_x_pressed=x_pressed,
            button_a_pressed=a_pressed,
            button_b_held=brake_active,
            button_lb_pressed=bool(BTN_CODE_LB & set(self.pressed)),
            button_rb_pressed=bool(BTN_CODE_RB & set(self.pressed)),
            button_start_pressed=start_pressed,
            button_back_pressed=back_pressed,
            connected=True,
            device_name=self.device_name,
            speed_level=self.speed_level,
            speed_rpm=self.speed_levels[self.speed_level],
        )

    def is_switch_pressed(self, state: GamepadState) -> bool:
        """Check if user pressed the configured mode switch button."""
        if self.switch_button == "Y":
            return state.button_y_pressed
        elif self.switch_button == "X":
            return state.button_x_pressed
        elif self.switch_button == "A":
            return state.button_a_pressed
        elif self.switch_button == "BACK":
            return state.button_back_pressed
        return state.button_y_pressed

    def close(self):
        if self.dev:
            try:
                self.dev.close()
            except Exception:
                pass
            self.dev = None
            self.connected = False
