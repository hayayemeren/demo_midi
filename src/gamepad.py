"""Logitech Wireless Gamepad F710 Driver using Python evdev.

Provides synchronous, zero-latency polling matching hardware evdev capabilities.
Supports both XInput (X mode) and DirectInput (D mode) on Linux.
"""

from __future__ import annotations

import errno
import logging
import math
import os
import select
import threading
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

ABS_CODE_X     = E.ABS_X if E else 0x00
ABS_CODE_Y     = E.ABS_Y if E else 0x01
ABS_CODE_Z     = E.ABS_Z if E else 0x02
ABS_CODE_RX    = E.ABS_RX if E else 0x03
ABS_CODE_RY    = E.ABS_RY if E else 0x04
ABS_CODE_RZ    = E.ABS_RZ if E else 0x05
ABS_CODE_HAT0X = E.ABS_HAT0X if E else 0x10
ABS_CODE_HAT0Y = E.ABS_HAT0Y if E else 0x11


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
    """Low-latency Gamepad reader using evdev."""

    def __init__(
        self,
        device_path: Optional[str] = None,
        deadzone: float = 0.10,
        speed_levels: Optional[List[float]] = None,
        default_speed_level: int = 1,
        switch_button: str = "Y",
        stale_timeout_s: float = 0.0,
    ):
        self.device_path = device_path
        self.deadzone = float(deadzone)
        self.speed_levels = speed_levels or [50.0, 100.0, 150.0, 200.0]
        self.speed_level = max(0, min(len(self.speed_levels) - 1, default_speed_level))
        self.switch_button = switch_button.upper()
        self.stale_timeout_s = float(stale_timeout_s)

        self.dev = None
        self.ax: Dict[int, Tuple[int, int]] = {}     # code -> (min, max)
        self.val: Dict[int, int] = {}                # code -> current raw value
        self.btn: Set[int] = set()                   # currently held buttons
        self.pressed: List[int] = []                 # buttons pressed this cycle
        self.turn_code: Optional[int] = None
        self.ry_code: Optional[int] = None
        self.connected = False
        self.device_name = "Not Connected"
        self._last_reconnect_attempt = 0.0

    @staticmethod
    def inspect(path: str):
        """Check if device has necessary sticks and keys for Logitech gamepad."""
        if not EVDEV_AVAILABLE:
            return None, {}, set(), False, "evdev library not installed"
        d = None
        try:
            d = evdev.InputDevice(path)
            # Reject Logitech F710 wireless sleep/pairing stub (046d:c22f)
            if getattr(d.info, "vendor", 0) == 0x046D and getattr(d.info, "product", 0) == 0xC22F:
                d.close()
                return None, {}, set(), False, "Logitech F710 sleep stub (046d:c22f)"

            caps = dict(d.capabilities(absinfo=True))
            axes = {c: info for c, info in caps.get(E.EV_ABS, [])}
            keys = set(caps.get(E.EV_KEY, []))
            pad_keys = any(0x120 <= k <= 0x13F for k in keys)

            if not (E.ABS_X in axes and E.ABS_Y in axes):
                d.close()
                return None, axes, keys, False, "Left stick (ABS_X/ABS_Y) missing"
            if not pad_keys:
                d.close()
                return None, axes, keys, False, "Gamepad button keys missing"
            if not (E.ABS_RX in axes or E.ABS_Z in axes):
                d.close()
                return None, axes, keys, False, "Right stick (ABS_RX/ABS_Z) missing"

            return d, axes, keys, True, "OK"
        except Exception as e:
            if d is not None:
                try:
                    d.close()
                except Exception:
                    pass
            return None, {}, set(), False, str(e)

    def _neutral_value_for_axis(self, code: int) -> int:
        """Return safe neutral center value for a given axis code."""
        if code in (ABS_CODE_HAT0X, ABS_CODE_HAT0Y):
            return 0
        lo, hi = self.ax.get(code, (-32768, 32767))
        return int(round((lo + hi) / 2.0))

    def _reset_state_to_neutral(self):
        """Reset all stick axes, D-pad, and held buttons to safe neutral values."""
        for code in list(self.ax.keys()):
            self.val[code] = self._neutral_value_for_axis(code)
        self.val[ABS_CODE_HAT0X] = 0
        self.val[ABS_CODE_HAT0Y] = 0
        self.btn.clear()

    def connect(self, force_path: Optional[str] = None, verbose: bool = False) -> bool:
        """Find and open the Logitech F710 device."""
        if not EVDEV_AVAILABLE:
            logger.warning("evdev library is not available.")
            return False

        all_paths = evdev.list_devices()
        if force_path:
            paths = [force_path] + [p for p in all_paths if p != force_path]
        else:
            paths = all_paths

        for path in paths:
            if not path:
                continue
            try:
                d, axes, keys, ok, why = self.inspect(path)
            except Exception as e:
                if verbose:
                    logger.debug("Device %s check failed: %s", path, e)
                continue

            if not ok or d is None:
                if verbose:
                    logger.debug("Skipping %s: %s", path, why)
                continue

            if self.dev is not None and self.dev is not d:
                try:
                    self.dev.close()
                except Exception:
                    pass

            # Right stick X: XInput is ABS_RX, DirectInput is ABS_Z
            turn = E.ABS_RX if E.ABS_RX in axes else E.ABS_Z
            ry = E.ABS_RY if E.ABS_RY in axes else E.ABS_RZ
            self.dev = d
            self.turn_code = turn
            self.ry_code = ry
            self.ax = {c: (i.min, i.max) for c, i in axes.items()}
            self.val = {}
            for c, i in axes.items():
                if c not in (ABS_CODE_HAT0X, ABS_CODE_HAT0Y) and i.min >= 0 and i.value == 0:
                    self.val[c] = self._neutral_value_for_axis(c)
                else:
                    self.val[c] = i.value
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

        v = self.val.get(code, int(round(center)))
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

        if hasattr(self.dev, "read_one"):
            try:
                # Drain all pending events in kernel buffer
                while True:
                    ev = self.dev.read_one()
                    if ev is None:
                        break
                    if E and ev.type == E.EV_ABS:
                        self.val[ev.code] = ev.value
                    elif E and ev.type == E.EV_KEY:
                        if ev.value == 1:
                            self.btn.add(ev.code)
                            self.pressed.append(ev.code)
                        elif ev.value == 0:
                            self.btn.discard(ev.code)
            except BlockingIOError:
                pass
            except OSError:
                logger.warning("Gamepad disconnected!")
                try:
                    self.dev.close()
                except Exception:
                    pass
                self.dev = None
                self.connected = False
                self._reset_state_to_neutral()
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
        ls_y = -self.norm(ABS_CODE_Y)
        # Left Stick X: Push Right = +1.0, Push Left = -1.0
        ls_x = self.norm(ABS_CODE_X)

        # Right Stick X: Push Right = +1.0, Push Left = -1.0
        rs_x = self.norm(self.turn_code) if self.turn_code else 0.0
        # Right Stick Y: Push Up = +1.0, Pull Down = -1.0
        ry_code = self.ry_code if self.ry_code is not None else (ABS_CODE_RY if ABS_CODE_RY in self.ax else ABS_CODE_RZ)
        rs_y = -self.norm(ry_code) if ry_code in self.ax else 0.0

        # D-pad (POV Hat)
        dpad_x = int(self.val.get(ABS_CODE_HAT0X, 0))
        dpad_y = -int(self.val.get(ABS_CODE_HAT0Y, 0))  # Up = +1, Down = -1

        # Base driving inputs (with D-Pad fallback in case F710 MODE swaps Left Stick & D-Pad)
        raw_fwd = ls_y if abs(ls_y) > 1e-3 else float(dpad_y)
        raw_turn = -rs_x if abs(rs_x) > 1e-3 else (-float(dpad_x) if dpad_x != 0 else 0.0)

        # Buttons held and pressed
        brake_active = bool(BTN_CODE_BRAKE & self.btn)
        lb_held = bool(BTN_CODE_LB & self.btn)
        rb_held = bool(BTN_CODE_RB & self.btn)
        pressed_set = set(self.pressed)
        start_pressed = bool(BTN_CODE_START & pressed_set)
        y_pressed = bool(BTN_CODE_Y & pressed_set)
        x_pressed = bool(BTN_CODE_X & pressed_set)
        a_pressed = bool(BTN_CODE_A & pressed_set)
        back_pressed = bool(BTN_CODE_BACK & pressed_set)

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
            button_lb_pressed=bool(BTN_CODE_LB & pressed_set),
            button_rb_pressed=bool(BTN_CODE_RB & pressed_set),
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
        self._reset_state_to_neutral()
