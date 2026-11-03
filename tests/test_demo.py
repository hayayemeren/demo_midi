import math
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    assert base.is_locked is True

    # Verify that moving the gamepad sticks during ARM_TELEOP does NOT turn the wheels!
    stick_move_state = GamepadState(forward=1.0, turn=1.0, connected=True)
    sm._handle_arm_teleop(stick_move_state)
    assert base.is_locked is True
    assert base.cur_left_rpm == 0.0
    assert base.cur_right_rpm == 0.0

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


def test_base_steering_direction():
    """Verify that pushing stick left with turn_dir=-1 turns robot left (CCW)."""
    cfg = {
        "dry_run": True,
        "serial": {"port": "/dev/null"},
        "motors": {"left_id": 1, "right_id": 2},
        "robot": {"turn_ratio": 0.65, "turn_dir": -1},
    }
    base = BaseController(cfg, dry_run=True)
    base.connect()
    # Pushing stick left (turn = +1.0)
    base.drive(forward=0.0, turn=1.0, max_rpm=100.0, dt=0.02)
    # Left wheel should turn backward (negative), right wheel forward (positive) -> Turn Left!
    assert base.cur_left_rpm < 0.0
    assert base.cur_right_rpm > 0.0


def test_arm_joint_standard_1to1():
    """Verify that default arm rotation is standard 1:1 (unaltered)."""
    cfg = {
        "dry_run": True,
        "arm": {
            "invert_shoulder_pan": False,
        },
    }
    arm = ArmController(cfg, dry_run=True)
    arm.sim_leader_pose = {"shoulder_pan.pos": 35.0, "shoulder_lift.pos": -50.0}
    leader_pose = arm.get_leader_pose()
    assert leader_pose["shoulder_pan.pos"] == 35.0
    assert leader_pose["shoulder_lift.pos"] == -50.0


if __name__ == "__main__":
    test_ddsm_crc8_spec()
    print("  [PASS] test_ddsm_crc8_spec")
    test_cosine_interpolation()
    print("  [PASS] test_cosine_interpolation")
    test_differential_drive_straight()
    print("  [PASS] test_differential_drive_straight")
    test_differential_drive_pivot()
    print("  [PASS] test_differential_drive_pivot")
    test_slew_rate_acceleration()
    print("  [PASS] test_slew_rate_acceleration")
    test_base_steering_direction()
    print("  [PASS] test_base_steering_direction")
    test_arm_joint_standard_1to1()
    print("  [PASS] test_arm_joint_standard_1to1")
    test_state_machine_transitions_simulation()
    print("  [PASS] test_state_machine_transitions_simulation")
    print("\n>>> ALL TESTS PASSED SUCCESSFULLY! <<<")


