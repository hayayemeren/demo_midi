"""Unit Tests for MIDI Mobile Manipulator Demo."""

import math
import pytest
from src.arm_controller import cosine_interpolate, ArmController, DEFAULT_STOW_POSE
from src.base_controller import BaseController
from src.ddsm import calculate_crc8, MotorFeedback, DDSMDriver
from src.diff_drive import DifferentialDrive, RobotDimensions, slew
from src.gamepad import F710Gamepad, GamepadState
from src.state_machine import DemoStateMachine, DemoState


def test_ddsm_crc8_spec():
    """Verify standard DDSM115 CRC8 calculation."""
    payload = bytes([0x01, 0x74, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
    assert calculate_crc8(payload) == 0x04


def test_cosine_interpolation():
    """Verify S-curve cosine trajectory smoothing."""
    start = {"pan": 0.0, "lift": -90.0}
    target = {"pan": 40.0, "lift": 10.0}

    # At start (0.0)
    p0 = cosine_interpolate(start, target, 0.0)
    assert abs(p0["pan"] - 0.0) < 1e-5
    assert abs(p0["lift"] - (-90.0)) < 1e-5

    # At halfway (0.5) -> alpha = 0.5
    p_mid = cosine_interpolate(start, target, 0.5)
    assert abs(p_mid["pan"] - 20.0) < 1e-5
    assert abs(p_mid["lift"] - (-40.0)) < 1e-5

    # At end (1.0)
    p1 = cosine_interpolate(start, target, 1.0)
    assert abs(p1["pan"] - 40.0) < 1e-5
    assert abs(p1["lift"] - 10.0) < 1e-5


def test_differential_drive_straight():
    """Straight driving produces equal RPM commands."""
    dims = RobotDimensions(wheel_radius=0.0575, wheel_base=0.350)
    dd = DifferentialDrive(dims)

    left_rpm, right_rpm = dd.twist_to_rpm(0.5, 0.0)
    assert abs(left_rpm - right_rpm) < 1e-3
    assert left_rpm > 0


def test_differential_drive_pivot():
    """In-place pivot turn produces opposite RPM commands."""
    dims = RobotDimensions(wheel_radius=0.0575, wheel_base=0.350)
    dd = DifferentialDrive(dims)

    left_rpm, right_rpm = dd.twist_to_rpm(0.0, 1.0)
    assert abs(left_rpm + right_rpm) < 1e-3
    assert left_rpm < 0
    assert right_rpm > 0


def test_slew_rate_acceleration():
    """Test asymmetric slew rate limiting."""
    cur = 0.0
    tgt = 100.0
    up = 1000.0
    down = 500.0
    dt = 0.02

    # Accelerate
    cur = slew(cur, tgt, up, down, dt)
    assert abs(cur - 20.0) < 1e-4

    # Decelerate
    tgt = 0.0
    cur = slew(cur, tgt, up, down, dt)
    assert abs(cur - 10.0) < 1e-4


def test_state_machine_transitions_simulation():
    """Verify demo state transitions in dry-run simulation mode."""
    cfg = {
        "dry_run": True,
        "serial": {"port": "/dev/null"},
        "motors": {"left_id": 1, "right_id": 2},
        "arm": {
            "follower_port": "/dev/null",
            "leader_port": "/dev/null",
            "interpolation_time_s": 0.01,
        },
        "gamepad": {"switch_button": "Y"},
    }

    base = BaseController(cfg, dry_run=True)
    arm = ArmController(cfg, dry_run=True)
    gamepad = F710Gamepad()

    base.connect()
    arm.connect_follower()

    sm = DemoStateMachine(base, arm, gamepad, loop_hz=100.0)
    assert sm.state == DemoState.BASE_DRIVE

    # 1. Simulate button Y press when leader is connected
    arm.connect_leader()
    gp_state = GamepadState(button_y_pressed=True, connected=True)
    sm._handle_base_drive(gp_state, dt=0.02)
    assert sm.state == DemoState.SYNCING
    assert base.is_locked is True

    # 2. Complete syncing -> transitions to ARM_TELEOP
    sm._handle_syncing()
    assert sm.state == DemoState.ARM_TELEOP

    # 3. Simulate button Y press in ARM_TELEOP -> transitions to PARKING
    gp_state = GamepadState(button_y_pressed=True, connected=True)
    sm._handle_arm_teleop(gp_state)
    assert sm.state == DemoState.PARKING

    # 4. Complete parking -> transitions back to BASE_DRIVE and unlocks brakes
    sm._handle_parking()
    assert sm.state == DemoState.BASE_DRIVE
    assert base.is_locked is False

    # 5. Simulate button Y press when leader is NOT connected and NOT available -> goes to WAITING_FOR_LEADER
    arm.leader = None
    arm.is_leader_port_available = lambda: False
    gp_state = GamepadState(button_y_pressed=True, connected=True)
    sm._handle_base_drive(gp_state, dt=0.02)
    assert sm.state == DemoState.WAITING_FOR_LEADER
    assert base.is_locked is True
