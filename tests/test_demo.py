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
from src.leader_reader import leader_to_base_drive, SO100LeaderReader, DEFAULT_NEUTRAL_POSE
from src.network import RobotServer, RobotClient
from src.state_machine import DemoStateMachine, DemoState
from src.vision_grasp import WristCamera, SockDetector, VisionGraspPlanner, DetectionResult


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


def test_case1_leader_teleop_simulation():
    """Verify Case 1: Base Drive + SO-100 Leader Arm Teleoperation."""
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

    sm = DemoStateMachine(base, arm, gamepad, demo_case=1, loop_hz=100.0)
    assert sm.state == DemoState.BASE_DRIVE

    # 1. Simulate button Y press when leader is connected
    arm.connect_leader()
    gp_state = GamepadState(button_y_pressed=True, connected=True)
    sm._handle_base_drive(gp_state, dt=0.02)
    assert sm.state == DemoState.SYNCING
    assert base.is_locked is True

    # 2. Complete syncing -> transitions to ARM_TELEOP_LEADER
    sm._handle_syncing()
    assert sm.state == DemoState.ARM_TELEOP_LEADER
    assert base.is_locked is True

    # Verify that moving the gamepad sticks during ARM_TELEOP_LEADER does NOT turn the wheels!
    stick_move_state = GamepadState(forward=1.0, turn=1.0, connected=True)
    sm._handle_arm_teleop_leader(stick_move_state)
    assert base.is_locked is True
    assert base.cur_left_rpm == 0.0
    assert base.cur_right_rpm == 0.0

    # 3. Simulate button Y press in ARM_TELEOP_LEADER -> transitions to PARKING
    gp_state = GamepadState(button_y_pressed=True, connected=True)
    sm._handle_arm_teleop_leader(gp_state)
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


def test_case2_gamepad_teleop_simulation():
    """Verify Case 2: Base Drive + Logitech Gamepad rate-controlled arm teleoperation."""
    cfg = {
        "dry_run": True,
        "serial": {"port": "/dev/null"},
        "motors": {"left_id": 1, "right_id": 2},
        "arm": {
            "follower_port": "/dev/null",
            "leader_port": "/dev/null",
            "interpolation_time_s": 0.01,
        },
        "arm_gamepad_teleop": {
            "pan_speed_dps": 50.0,
            "lift_speed_dps": 45.0,
            "elbow_speed_dps": 50.0,
            "flex_speed_dps": 45.0,
            "roll_speed_dps": 60.0,
            "gripper_speed_pct_s": 60.0,
        },
        "gamepad": {"switch_button": "Y"},
    }

    base = BaseController(cfg, dry_run=True)
    arm = ArmController(cfg, dry_run=True)
    gamepad = F710Gamepad()

    base.connect()
    arm.connect_follower()

    sm = DemoStateMachine(base, arm, gamepad, demo_case=2, loop_hz=100.0)
    assert sm.state == DemoState.BASE_DRIVE

    # 1. Drive base
    gp_drive = GamepadState(forward=0.8, turn=0.0, connected=True)
    sm._handle_base_drive(gp_drive, dt=0.02)
    assert base.cur_left_rpm > 0.0
    assert base.cur_right_rpm > 0.0
    assert base.is_locked is False

    # 2. Press Y -> Switches immediately to ARM_TELEOP_GAMEPAD (No leader arm required)
    gp_switch = GamepadState(button_y_pressed=True, connected=True)
    sm._handle_base_drive(gp_switch, dt=0.02)
    assert sm.state == DemoState.ARM_TELEOP_GAMEPAD
    assert base.is_locked is True

    # 3. Base wheels MUST be locked in ARM_TELEOP_GAMEPAD even if sticks are pushed
    gp_arm_move = GamepadState(
        forward=1.0, turn=1.0,
        right_stick_x=1.0,  # Pan right
        right_stick_y=0.5,  # Lift up
        connected=True
    )
    sm._handle_arm_teleop_gamepad(gp_arm_move, dt=0.1)
    assert base.is_locked is True
    assert base.cur_left_rpm == 0.0
    assert base.cur_right_rpm == 0.0

    # Arm target should have moved according to speeds and dt
    target = arm.current_gamepad_target
    assert target["shoulder_pan.pos"] > 0.0   # Moved positive (right)
    assert target["shoulder_lift.pos"] > -95.0 # Lifted upward from stow pose (-95.0)

    # 4. Gripper control: hold RB to close gripper
    prev_grip = target["gripper.pos"]
    gp_close_grip = GamepadState(button_rb_held=True, connected=True)
    sm._handle_arm_teleop_gamepad(gp_close_grip, dt=0.1)
    assert arm.current_gamepad_target["gripper.pos"] > prev_grip

    # 5. Gripper toggle: Button A
    gp_toggle_grip = GamepadState(button_a_pressed=True, connected=True)
    sm._handle_arm_teleop_gamepad(gp_toggle_grip, dt=0.1)
    assert arm.current_gamepad_target["gripper.pos"] in (0.0, 85.0)

    # 6. Button X: Ready pose
    gp_ready = GamepadState(button_x_pressed=True, connected=True)
    sm._handle_arm_teleop_gamepad(gp_ready, dt=0.1)
    for k, v in arm.ready_pose.items():
        assert abs(arm.current_gamepad_target[k] - v) < 1e-4

    # 7. Press Y -> Switches to PARKING
    gp_park = GamepadState(button_y_pressed=True, connected=True)
    sm._handle_arm_teleop_gamepad(gp_park, dt=0.1)
    assert sm.state == DemoState.PARKING
    assert base.is_locked is True

    # 8. Parking finishes -> Back to BASE_DRIVE, brakes released
    sm._handle_parking()
    assert sm.state == DemoState.BASE_DRIVE
    assert base.is_locked is False


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


def test_case3_leader_to_base_kinematics():
    """Verify mapping from leader arm deflection to base drive commands."""
    neutral = {"shoulder_pan.pos": 0.0, "shoulder_lift.pos": -45.0, "gripper.pos": 10.0}

    # 1. At neutral -> zero motion
    fwd, turn, brake = leader_to_base_drive(neutral, neutral)
    assert fwd == 0.0
    assert turn == 0.0
    assert brake is False

    # 2. Small deflection inside deadzone (<= 6 deg) -> zero motion
    pose_dz = {"shoulder_pan.pos": 4.0, "shoulder_lift.pos": -42.0, "gripper.pos": 10.0}
    fwd, turn, brake = leader_to_base_drive(pose_dz, neutral)
    assert fwd == 0.0
    assert turn == 0.0

    # 3. Turn left (pan negative, e.g. -25 deg) with turn_dir=-1 -> positive turn (CCW)
    pose_left = {"shoulder_pan.pos": -25.0, "shoulder_lift.pos": -45.0, "gripper.pos": 10.0}
    fwd, turn, brake = leader_to_base_drive(pose_left, neutral)
    assert turn > 0.0
    assert fwd == 0.0

    # 4. Turn right (pan positive, e.g. +25 deg) -> negative turn (CW)
    pose_right = {"shoulder_pan.pos": 25.0, "shoulder_lift.pos": -45.0, "gripper.pos": 10.0}
    fwd, turn, brake = leader_to_base_drive(pose_right, neutral)
    assert turn < 0.0

    # 5. Push forward (lift increases toward 0 deg, e.g. -20 deg) -> forward > 0
    pose_fwd = {"shoulder_pan.pos": 0.0, "shoulder_lift.pos": -20.0, "gripper.pos": 10.0}
    fwd, turn, brake = leader_to_base_drive(pose_fwd, neutral)
    assert fwd > 0.0
    assert turn == 0.0

    # 6. Pull backward (lift decreases toward -90 deg, e.g. -75 deg) -> forward < 0
    pose_rev = {"shoulder_pan.pos": 0.0, "shoulder_lift.pos": -75.0, "gripper.pos": 10.0}
    fwd, turn, brake = leader_to_base_drive(pose_rev, neutral)
    assert fwd < 0.0

    # 7. Squeeze gripper (>80%) -> brake engaged
    pose_grip = {"shoulder_pan.pos": 0.0, "shoulder_lift.pos": -45.0, "gripper.pos": 85.0}
    fwd, turn, brake = leader_to_base_drive(pose_grip, neutral)
    assert brake is True


def test_case3_network_teleop_simulation():
    """Verify Case 3: Remote Network Teleoperation between PC Client and Robot Server."""
    import time

    cfg = {
        "dry_run": True,
        "serial": {"port": "/dev/null"},
        "motors": {"left_id": 1, "right_id": 2},
        "arm": {
            "follower_port": "/dev/null",
            "interpolation_time_s": 0.01,
        },
    }

    test_port = 18765
    base = BaseController(cfg, dry_run=True)
    arm = ArmController(cfg, dry_run=True)
    gamepad = F710Gamepad()

    base.connect()
    arm.connect_follower()

    sm = DemoStateMachine(base, arm, gamepad, demo_case=3, network_port=test_port)
    assert sm.state == DemoState.NETWORK_SERVER
    assert sm.server is not None
    assert sm.server.start() is True

    # Connect client
    client = RobotClient(host="127.0.0.1", port=test_port)
    assert client.connect() is True
    time.sleep(0.05)

    # Poll server to register connection
    gp = GamepadState()
    sm._handle_network_server(gp, dt=0.02)
    assert sm.server.is_connected is True

    # 1. Test Base Drive command over network
    client.send_command({
        "mode": "base",
        "base": {"forward": 0.6, "turn": 0.0, "speed_rpm": 100.0, "brake": False}
    })
    time.sleep(0.02)
    sm._handle_network_server(gp, dt=0.02)
    assert base.cur_left_rpm > 0.0
    assert base.cur_right_rpm > 0.0
    assert base.is_locked is False

    # 2. Test Arm Teleop command over network
    arm_target = {
        "shoulder_pan.pos": 25.0,
        "shoulder_lift.pos": -35.0,
        "elbow_flex.pos": 20.0,
        "wrist_flex.pos": 40.0,
        "wrist_roll.pos": 0.0,
        "gripper.pos": 50.0,
    }
    client.send_command({"mode": "arm", "arm": arm_target})
    time.sleep(0.02)
    sm._handle_network_server(gp, dt=0.02)
    # Base wheels MUST be locked in arm mode!
    assert base.is_locked is True
    assert base.cur_left_rpm == 0.0
    assert base.cur_right_rpm == 0.0

    # 3. Test telemetry exchange
    time.sleep(0.02)
    telem = client.poll_telemetry()
    assert telem is not None
    assert telem["state"] == "NETWORK_SERVER"
    assert telem["mode"] == "arm"
    assert telem["base_locked"] is True

    # 4. Clean shutdown
    client.disconnect()
    sm.stop()


def test_case4_vision_grasp_components():
    """Verify Case 4 vision detector, camera, and grasp planner."""
    # 1. Test WristCamera dry-run frame capture
    cam = WristCamera(dry_run=True)
    assert cam.connect() is True
    ok, frame = cam.capture_frame()
    assert ok is True
    assert frame is not None
    assert frame.shape == (480, 640, 3)

    # 2. Test SockDetector on the synthetic frame
    detector = SockDetector(mode="auto", save_debug_image=False)
    det = detector.detect(frame)
    assert det.found is True
    assert det.area > 500.0
    assert abs(det.error_x) < 0.5  # near center

    # 3. Test VisionGraspPlanner alignment
    cfg = {
        "case4": {
            "servoing": {"pan_gain": 20.0, "lift_gain": 15.0, "align_tolerance": 0.10},
            "grasp_verification_threshold_pct": 10.0,
        }
    }
    planner = VisionGraspPlanner(cfg)
    cur_pose = {"shoulder_pan.pos": 0.0, "shoulder_lift.pos": -50.0, "gripper.pos": 10.0}
    aligned, is_aligned = planner.compute_aligned_pose(cur_pose, det)
    assert "shoulder_pan.pos" in aligned

    # 4. Test grasp pose computation (adaptive reach, orientation, and hover)
    grasp_pose = planner.compute_grasp_pose(aligned)
    assert grasp_pose["gripper.pos"] == 0.0
    assert grasp_pose["shoulder_lift.pos"] > aligned["shoulder_lift.pos"]  # lower down

    # 4b. Test adaptive longitudinal reach modulation
    det_far = DetectionResult(found=True, error_y=-0.4)
    det_near = DetectionResult(found=True, error_y=0.4)
    grasp_far = planner.compute_grasp_pose(aligned, det_far)
    grasp_near = planner.compute_grasp_pose(aligned, det_near)
    assert grasp_far["elbow_flex.pos"] > grasp_near["elbow_flex.pos"], "Elbow should extend further forward for far sock"

    # 4c. Test adaptive wrist roll orientation alignment
    det_angled = DetectionResult(found=True, angle=35.0)
    grasp_angled = planner.compute_grasp_pose(aligned, det_angled)
    assert abs(grasp_angled["wrist_roll.pos"] - grasp_pose["wrist_roll.pos"]) > 10.0, "Wrist roll should rotate to match sock orientation"

    # 4d. Test pre-grasp hover pose (elevated with open jaws)
    hover_pose = planner.compute_grasp_pose(aligned, det, is_hover=True)
    assert hover_pose["gripper.pos"] == 85.0
    assert hover_pose["shoulder_lift.pos"] < grasp_pose["shoulder_lift.pos"], "Hover pose must elevate above floor grasp"

    # 5. Test Suggestion 2 grasp verification
    assert planner.verify_grasp(25.0) is True   # holding cloth
    assert planner.verify_grasp(4.5) is False   # closed empty

    cam.release()


def test_case4_autonomous_state_machine_simulation():
    """Verify full Case 4 state machine autonomous pick-and-place flow in dry-run mode."""
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
        "case4": {
            "camera": {"device": "/dev/null"},
            "scan_pose": {"shoulder_pan.pos": 0.0, "shoulder_lift.pos": -50.0, "gripper.pos": 10.0},
            "basket_pose": {"shoulder_pan.pos": 0.0, "shoulder_lift.pos": -80.0, "elbow_flex.pos": 20.0, "wrist_flex.pos": 50.0, "wrist_roll.pos": 0.0, "gripper.pos": 85.0},
            "detection": {"mode": "auto", "save_debug_image": False},
        }
    }

    base = BaseController(cfg, dry_run=True)
    arm = ArmController(cfg, dry_run=True)
    gamepad = F710Gamepad()

    base.connect()
    arm.connect_follower()

    sm = DemoStateMachine(
        base=base,
        arm=arm,
        gamepad=gamepad,
        demo_case=4,
        loop_hz=50.0,
        config=cfg,
    )
    sm.running = True
    if sm.camera:
        sm.camera.connect()
    assert sm.state == DemoState.BASE_DRIVE

    # 1. Test Base Drive in Case 4
    gp = GamepadState(forward=0.8, turn=0.0, speed_rpm=100.0, connected=True)
    sm._handle_base_drive(gp, dt=0.02)
    assert base.cur_left_rpm > 0.0
    assert base.cur_right_rpm > 0.0
    assert base.is_locked is False

    # 2. Press [Y] button -> initiates autonomous pick routine
    gp_switch = GamepadState(button_y_pressed=True, connected=True)
    sm._handle_base_drive(gp_switch, dt=0.02)
    assert sm.state == DemoState.AUTONOMOUS_PICK_SCAN
    assert base.is_locked is True

    # 3. Autonomous scan step -> finds simulated sock -> switches to ALIGN
    sm._handle_autonomous_scan(gp)
    assert sm.state == DemoState.AUTONOMOUS_PICK_ALIGN
    assert sm.last_detection is not None
    assert sm.last_detection.found is True

    # 4. Autonomous align step -> aligns gripper -> switches to GRASP
    sm._handle_autonomous_align(gp)
    assert sm.state == DemoState.AUTONOMOUS_PICK_GRASP

    # 5. Autonomous grasp step -> descends and clamps -> switches to VERIFY
    sm._handle_autonomous_grasp(gp)
    assert sm.state == DemoState.AUTONOMOUS_PICK_VERIFY

    # 6. Grasp verification (Suggestion 2) -> confirms grip -> switches to BASKET
    sm._handle_autonomous_verify(gp)
    assert sm.state == DemoState.AUTONOMOUS_BASKET_DUMP

    # 7. Basket dump step -> swings to basket, deposits, parks -> returns to BASE_DRIVE
    sm._handle_autonomous_basket(gp)
    assert sm.state == DemoState.PARKING
    sm._handle_parking()
    assert sm.state == DemoState.BASE_DRIVE
    assert base.is_locked is False

    # 8. Test direct manual basket dump macro via Button [X]
    gp_x = GamepadState(button_x_pressed=True, connected=True)
    sm._handle_base_drive(gp_x, dt=0.02)
    assert sm.state == DemoState.AUTONOMOUS_BASKET_DUMP
    assert base.is_locked is True

    sm.stop()


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
    test_case1_leader_teleop_simulation()
    print("  [PASS] test_case1_leader_teleop_simulation")
    test_case2_gamepad_teleop_simulation()
    print("  [PASS] test_case2_gamepad_teleop_simulation")
    test_case3_leader_to_base_kinematics()
    print("  [PASS] test_case3_leader_to_base_kinematics")
    test_case3_network_teleop_simulation()
    print("  [PASS] test_case3_network_teleop_simulation")
    test_case4_vision_grasp_components()
    print("  [PASS] test_case4_vision_grasp_components")
    test_case4_autonomous_state_machine_simulation()
    print("  [PASS] test_case4_autonomous_state_machine_simulation")
    print("\n>>> ALL TESTS PASSED SUCCESSFULLY! <<<")


