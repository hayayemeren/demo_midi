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
    """Verify that pushing stick left with turn_dir=-1 turns robot left (CCW)
    and that relative wheel velocity is properly split so it does not double."""
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

    # Step forward until slew limiter reaches steady state target
    for _ in range(5):
        base.drive(forward=0.0, turn=1.0, max_rpm=100.0, dt=0.05)
    # At steady state, each wheel runs at 0.5 * 0.65 * 100 = 32.5 RPM
    assert abs(base.cur_left_rpm - (-32.5)) < 1e-3, f"Expected -32.5 RPM, got {base.cur_left_rpm}"
    assert abs(base.cur_right_rpm - 32.5) < 1e-3, f"Expected 32.5 RPM, got {base.cur_right_rpm}"
    # Relative velocity across both wheels is exactly 65 RPM (turn_ratio * max_rpm), NOT 130 RPM
    relative_rpm = base.cur_right_rpm - base.cur_left_rpm
    assert abs(relative_rpm - 65.0) < 1e-3, f"Expected 65.0 RPM relative, got {relative_rpm}"


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

    # 4b. Test adaptive longitudinal reach modulation (real URDF: far extends shoulder forward / elbow more negative)
    det_far = DetectionResult(found=True, error_y=-0.4)
    det_near = DetectionResult(found=True, error_y=0.4)
    grasp_far = planner.compute_grasp_pose(aligned, det_far)
    grasp_near = planner.compute_grasp_pose(aligned, det_near)
    r_far_fk, _, _ = planner.kinematics.forward_kinematics(
        grasp_far["shoulder_lift.pos"], grasp_far["elbow_flex.pos"], grasp_far["wrist_flex.pos"]
    )
    r_near_fk, _, _ = planner.kinematics.forward_kinematics(
        grasp_near["shoulder_lift.pos"], grasp_near["elbow_flex.pos"], grasp_near["wrist_flex.pos"]
    )
    assert r_far_fk > r_near_fk + 0.05, f"Far reach FK {r_far_fk:.3f}m must exceed near {r_near_fk:.3f}m"
    assert grasp_far["elbow_flex.pos"] < grasp_near["elbow_flex.pos"], (
        f"SO-101 elbow becomes more negative as forearm extends forward: {grasp_far['elbow_flex.pos']} vs {grasp_near['elbow_flex.pos']}"
    )

    # 4c. Test fixed wrist roll by default (no unwanted rolling during grasp)
    det_angled = DetectionResult(found=True, angle=35.0)
    grasp_angled = planner.compute_grasp_pose(aligned, det_angled)
    assert grasp_angled["wrist_roll.pos"] == grasp_pose["wrist_roll.pos"], "Wrist roll should remain fixed and flat without unwanted rolling"

    # 4c-2. Test opt-in wrist roll adaptation if explicitly enabled in configuration
    planner_roll = VisionGraspPlanner({"case4": {"servoing": {"enable_roll_alignment": True}}})
    grasp_rolled = planner_roll.compute_grasp_pose(aligned, det_angled)
    assert abs(grasp_rolled["wrist_roll.pos"] - grasp_pose["wrist_roll.pos"]) > 10.0, "Wrist roll rotates only when explicitly enabled"

    # 4d. Test pre-grasp hover pose (elevated with open jaws)
    hover_pose = planner.compute_grasp_pose(aligned, det, is_hover=True)
    assert hover_pose["gripper.pos"] == 85.0
    assert hover_pose["shoulder_lift.pos"] < grasp_pose["shoulder_lift.pos"], "Hover pose must elevate above floor grasp"

    # 5. Test Suggestion 2 grasp verification
    assert planner.verify_grasp(25.0) is True   # holding cloth
    assert planner.verify_grasp(4.5) is False   # closed empty

    # 6. Test camera health monitoring and staleness recovery
    assert cam.ensure_healthy() is True
    cam.start_stream(detector=detector)
    assert cam.ensure_healthy() is True
    ok, live_frame = cam.capture_frame(max_staleness_s=0.5)
    assert ok is True
    assert live_frame is not None

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


def test_arm_safety_and_stow_recovery():
    """Verify that joint limit clamping prevents mechanical stop contention,
    and move_to_stow safely clears rear/basket postures before stowing."""
    from src.arm_controller import ArmController
    cfg = {
        "dry_run": True,
        "arm": {
            "follower_port": "/dev/null",
            "interpolation_time_s": 0.01,
        }
    }
    arm = ArmController(cfg, dry_run=True)
    arm.connect_follower()

    # 1. Test joint limit clamping against hardware stops
    extreme_pose = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 95.0,
        "elbow_flex.pos": -85.0,
        "wrist_flex.pos": -101.5,   # At physical stop!
        "wrist_roll.pos": -179.5,   # Near continuous wrap limit!
        "gripper.pos": 85.0,
    }
    clamped = arm.clamp_to_joint_limits(extreme_pose)
    assert clamped["wrist_flex.pos"] == -96.0, f"Expected -96.0, got {clamped['wrist_flex.pos']}"
    assert clamped["wrist_roll.pos"] == -175.0, f"Expected -175.0, got {clamped['wrist_roll.pos']}"
    assert clamped["shoulder_lift.pos"] == 95.0, "Should allow 95.0 within [-100, 100]"

    # 2. Test safe stow transition from rear basket posture
    arm.sim_follower_pose = dict(clamped)
    arm.move_to_stow(duration_s=0.01)
    # Follower should end up at exact stow pose
    for k, v in arm.stow_pose.items():
        assert abs(arm.sim_follower_pose[k] - v) < 1e-3, f"Joint {k} did not reach stow pose"


def test_scan_poses_teaching_and_execution():
    """Verify 3-point floor scan poses configuration, clamping, planner targets, and state machine sweep."""
    import tempfile
    from main import update_config_scan_poses

    # 1. Test get_scan_targets fallback when scan_poses not present
    cfg_fallback = {
        "case4": {
            "scan_pose": {"shoulder_pan.pos": 0.0, "shoulder_lift.pos": -50.0},
            "scan_sweep_angles": [0.0, -25.0, 25.0],
        }
    }
    planner_fb = VisionGraspPlanner(cfg_fallback)
    targets_fb = planner_fb.get_scan_targets()
    assert len(targets_fb) == 3
    assert targets_fb[0][0] == "Middle (Center)"
    assert targets_fb[0][1]["shoulder_pan.pos"] == 0.0
    assert targets_fb[1][1]["shoulder_pan.pos"] == -25.0
    assert targets_fb[2][1]["shoulder_pan.pos"] == 25.0

    # 2. Test get_scan_targets when 3-point scan_poses are explicitly defined
    cfg_discrete = {
        "case4": {
            "scan_poses": {
                "center": {"shoulder_pan.pos": 0.0, "shoulder_lift.pos": -48.0},
                "left": {"shoulder_pan.pos": 30.0, "shoulder_lift.pos": -52.0},
                "right": {"shoulder_pan.pos": -30.0, "shoulder_lift.pos": -50.0},
            }
        }
    }
    planner_disc = VisionGraspPlanner(cfg_discrete)
    targets_disc = planner_disc.get_scan_targets()
    assert len(targets_disc) == 3
    labels = [t[0] for t in targets_disc]
    assert labels == ["Middle (Center)", "Left", "Right"]
    assert targets_disc[1][1]["shoulder_pan.pos"] == 30.0
    assert targets_disc[2][1]["shoulder_pan.pos"] == -30.0

    # 3. Test update_config_scan_poses with a temporary config file
    with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=False) as tf:
        tf.write("""# Demo test config
case4:
  scan_pose:
    shoulder_pan.pos: 0.0
    shoulder_lift.pos: -50.0
    elbow_flex.pos: 35.0
    wrist_flex.pos: 70.0
    wrist_roll.pos: 0.0
    gripper.pos: 10.0
  scan_sweep_angles: [0.0, -28.0, 28.0]
""")
        temp_cfg_path = tf.name

    try:
        center_raw = {"shoulder_pan.pos": 1.5, "shoulder_lift.pos": -52.0, "wrist_flex.pos": 75.0, "wrist_roll.pos": 0.0}
        left_raw = {"shoulder_pan.pos": 29.0, "shoulder_lift.pos": -50.0, "wrist_flex.pos": 72.0, "wrist_roll.pos": 0.0}
        right_raw = {"shoulder_pan.pos": -29.0, "shoulder_lift.pos": -51.0, "wrist_flex.pos": 73.0, "wrist_roll.pos": -190.0} # Needs clamping!

        ok = update_config_scan_poses(temp_cfg_path, center_raw, left_raw, right_raw)
        assert ok is True

        import yaml
        with open(temp_cfg_path, "r") as f:
            saved_cfg = yaml.safe_load(f)

        assert "scan_poses" in saved_cfg["case4"]
        sp = saved_cfg["case4"]["scan_poses"]
        assert sp["center"]["shoulder_pan.pos"] == 1.5
        assert sp["left"]["shoulder_pan.pos"] == 29.0
        assert sp["right"]["shoulder_pan.pos"] == -29.0
        # Check wrist_roll was clamped to max -175.0
        assert sp["right"]["wrist_roll.pos"] == -175.0
        # Check baseline scan_pose matches center
        assert saved_cfg["case4"]["scan_pose"]["shoulder_pan.pos"] == 1.5
    finally:
        if os.path.exists(temp_cfg_path):
            os.remove(temp_cfg_path)


def test_shoulder_pan_preservation_and_centroid_alignment():
    """Verify that compute_grasp_pose preserves shoulder_pan at Left/Right scan angles
    and dynamically centers shoulder_pan onto sock centroid in the correct direction."""
    # Test Robot A (User's robot: Right is positive, pan_direction = +1.0)
    cfg_user = {
        "case4": {
            "scan_poses": {
                "center": {"shoulder_pan.pos": -7.1},
                "left": {"shoulder_pan.pos": -67.3},
                "right": {"shoulder_pan.pos": 65.1},
            },
            "floor_grasp_pose": {
                "shoulder_pan.pos": -4.4,
                "shoulder_lift.pos": 97.5,
                "elbow_flex.pos": -58.2,
                "wrist_flex.pos": 59.6,
                "wrist_roll.pos": -0.7,
                "gripper.pos": 0.0,
            },
            "servoing": {
                "pan_gain": 22.0,
                "pan_direction": 1.0,
            }
        }
    }
    planner_user = VisionGraspPlanner(cfg_user)
    assert planner_user.pan_direction == 1.0

    # 1. Sock on right side (Right scan pose: pan = +65.1°)
    right_aligned = {
        "shoulder_pan.pos": 65.1,
        "shoulder_lift.pos": -63.7,
        "elbow_flex.pos": 17.3,
        "wrist_flex.pos": 96.0,
        "wrist_roll.pos": 1.5,
        "gripper.pos": 85.0,
    }
    # Descend to floor without extra detection trim: must stay at +65.1, NOT 0.0!
    grasp_right = planner_user.compute_grasp_pose(right_aligned)
    assert abs(grasp_right["shoulder_pan.pos"] - 65.1) < 1e-3

    # 2. Sock on right with residual horizontal offset to the right (error_x = +0.10)
    # On this robot (pan_direction = 1.0), pan must INCREASE towards right: 65.1 + 22.0*0.10 = +67.3°
    det_right = DetectionResult(found=True, error_x=0.10, error_y=0.0)
    grasp_trimmed = planner_user.compute_grasp_pose(right_aligned, detection=det_right)
    assert abs(grasp_trimmed["shoulder_pan.pos"] - 67.3) < 1e-2, (
        f"Expected pan ~ 67.3 (rotating towards centroid), got {grasp_trimmed['shoulder_pan.pos']}"
    )

    # 3. compute_aligned_pose centering towards centroid
    aligned_pose, is_aligned = planner_user.compute_aligned_pose(right_aligned, det_right)
    assert abs(aligned_pose["shoulder_pan.pos"] - 67.3) < 1e-2

    # 4. Sock on left with residual horizontal offset to the left (error_x = -0.10)
    # Pan must DECREASE towards left: -67.3 + 22.0*(-0.10) = -69.5°
    left_aligned = dict(right_aligned)
    left_aligned["shoulder_pan.pos"] = -67.3
    det_left = DetectionResult(found=True, error_x=-0.10, error_y=0.0)
    aligned_left, _ = planner_user.compute_aligned_pose(left_aligned, det_left)
    assert abs(aligned_left["shoulder_pan.pos"] - (-69.5)) < 1e-2


def test_so100_analytical_kinematics_and_clearance():
    """Verify URDF-calibrated SO-100 Forward/Inverse Kinematics, constant safe ground clearance
    across near/mid/far workspace, adaptive pitch at far reach, and 4-stage trajectory generation."""
    from src.kinematics import SO100Kinematics

    ref_pose = {
        "shoulder_pan.pos": -4.4,
        "shoulder_lift.pos": 97.5,
        "elbow_flex.pos": -58.2,
        "wrist_flex.pos": 59.6,
        "wrist_roll.pos": -0.7,
        "gripper.pos": 0.0,
    }
    kin = SO100Kinematics(floor_grasp_pose=ref_pose, ground_clearance_m=0.024)

    # 1. Forward Kinematics at reference ground contact matches URDF geometry
    r_ref, z_floor, pitch_contact = kin.forward_kinematics(97.5, -58.2, 59.6)
    expected_pitch = -(97.5 - 58.2 + 59.6)
    assert abs(pitch_contact - expected_pitch) < 1e-3, f"Expected pitch {expected_pitch}°, got {pitch_contact}"
    assert 0.20 < r_ref < 0.23, f"Expected URDF r_pan ~0.214m, got {r_ref:.4f}m"
    assert abs(z_floor - (-0.1634)) < 0.002, f"Expected URDF z_floor ~-0.1634m, got {z_floor:.4f}m"

    # 2. Inverse Kinematics reproduces exact reference pose and perpendicular floor poses across radii
    l_sol, e_sol, w_sol = kin.solve_ik(kin.r_ref, kin.z_floor, desired_pitch_deg=pitch_contact)
    assert abs(l_sol - 97.5) < 0.05
    assert abs(e_sol - (-58.2)) < 0.05
    assert abs(w_sol - 59.6) < 0.05

    # 3. Ground Clearance (+2.4cm above raw floor contact) across ALL near/mid/far radii!
    z_clear = kin.z_floor + kin.ground_clearance_m
    near_joints = kin.solve_ik(0.16, z_clear, desired_pitch_deg=-90.0)
    far_joints = kin.solve_ik(0.24, z_clear, desired_pitch_deg=-90.0)
    assert abs(near_joints[1] - far_joints[1]) > 40.0, (
        "Near vs far reach must fold/extend elbow distinctly instead of picking the same full-extension pose"
    )
    for r_test in (0.15, 0.19, 0.23, 0.25, 0.29):
        l_c, e_c, w_c = kin.solve_ik(r_test, z_clear, desired_pitch_deg=-90.0)
        r_chk, z_chk, pitch_chk = kin.forward_kinematics(l_c, e_c, w_c)
        assert abs(z_chk - z_clear) < 1e-3, f"R={r_test}: height mismatch {z_chk*100:.2f}cm vs {z_clear*100:.2f}cm"
        assert abs(r_chk - r_test) < 1e-3, f"R={r_test}: radial mismatch {r_chk*100:.2f}cm"
        if r_test <= 0.25:
            assert abs(pitch_chk - (-90.0)) < 0.1, f"R={r_test}: must stay perpendicular -90°, got {pitch_chk:.2f}°"
        else:
            assert -90.0 <= pitch_chk <= -55.0, f"Extended reach pitch must stay steeply downward: {pitch_chk:.1f}°"

    # 4. Vertical Cartesian Lift (+12.5cm)
    z_lift = kin.z_floor + kin.lift_height_m
    l_lift, e_lift, w_lift = kin.solve_ik(0.21, z_lift, desired_pitch_deg=-90.0)
    r_lchk, z_lchk, pitch_lchk = kin.forward_kinematics(l_lift, e_lift, w_lift)
    assert abs(z_lchk - z_lift) < 1e-3
    assert abs(r_lchk - 0.21) < 1e-3
    assert abs(pitch_lchk - (-90.0)) < 0.1

    # 5. Normalized error fallback mapping
    det_top = DetectionResult(found=True, error_x=-0.2, error_y=-0.5)
    r_target, pan_target, _ = kin.compute_sock_target(det_top, scan_pan_deg=10.0, pan_direction=1.0)
    assert r_target > 0.21, "Negative error_y (top of image) must yield far reach"
    assert pan_target < 10.0, "Negative error_x must rotate pan towards left"

    # 6. Complete 4-stage trajectory generation
    stages = kin.get_trajectory_stages(0.22, 0.0)
    assert "hover" in stages and "descent" in stages and "clamp" in stages and "lift" in stages
    assert stages["hover"]["gripper.pos"] == 85.0
    assert stages["descent"]["gripper.pos"] == 85.0
    assert stages["clamp"]["gripper.pos"] == 0.0
    assert stages["lift"]["gripper.pos"] == 0.0
    assert stages["hover"]["shoulder_lift.pos"] < stages["descent"]["shoulder_lift.pos"]


def test_centroid_to_cartesian_floor_targeting():
    """Verify that 2D centroid (cx, cy) from camera frame maps cleanly to 3D Cartesian
    floor coordinates (R, Pan) across near/far and left/right positions using 3D ray projection,
    while keeping identical target coordinates across all trajectory stages."""
    import yaml
    from src.vision_grasp import VisionGraspPlanner, DetectionResult

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    planner = VisionGraspPlanner(cfg)
    kin = planner.kinematics

    # 1. Near vs Far radial mapping using scan_pose ray projection
    det_near = DetectionResult(found=True, cx=320.0, cy=310.0, error_x=0.0, error_y=(310 - 240) / 240)
    det_far = DetectionResult(found=True, cx=320.0, cy=70.0, error_x=0.0, error_y=(70 - 240) / 240)

    r_near, pan_near, _ = kin.compute_sock_target(det_near, scan_pan_deg=-7.1)
    r_far, pan_far, _ = kin.compute_sock_target(det_far, scan_pan_deg=-7.1)

    assert r_near < 0.16, f"Expected near reach <= 16cm, got {r_near*100:.1f}cm"
    assert r_far > 0.27, f"Expected far reach > 27cm, got {r_far*100:.1f}cm"
    assert r_far > r_near + 0.11, "Far reach must exceed near reach by > 11cm"

    # Real URDF joint behavior: near folds elbow positive (+30°), far extends elbow negative (-55°)
    descent_near = kin.get_trajectory_stages(r_near, pan_near)["descent"]
    descent_far = kin.get_trajectory_stages(r_far, pan_far)["descent"]
    assert descent_far["elbow_flex.pos"] < descent_near["elbow_flex.pos"], (
        f"SO-101 elbow flexes more negative for far reach: {descent_far['elbow_flex.pos']} vs {descent_near['elbow_flex.pos']}"
    )
    assert descent_far["shoulder_lift.pos"] > descent_near["shoulder_lift.pos"], (
        f"SO-101 shoulder pitches forward for far reach: {descent_far['shoulder_lift.pos']} vs {descent_near['shoulder_lift.pos']}"
    )

    # 2. Left vs Right azimuthal pan mapping
    det_left = DetectionResult(found=True, cx=120.0, cy=200.0, error_x=(120 - 320) / 320, error_y=0.0)
    det_right = DetectionResult(found=True, cx=520.0, cy=200.0, error_x=(520 - 320) / 320, error_y=0.0)

    _, pan_l, _ = kin.compute_sock_target(det_left, scan_pan_deg=0.0, pan_direction=1.0)
    _, pan_r, _ = kin.compute_sock_target(det_right, scan_pan_deg=0.0, pan_direction=1.0)

    assert pan_l < -20.0, f"Left sock must turn pan CCW/left strongly: {pan_l:.1f}°"
    assert pan_r > 20.0, f"Right sock must turn pan CW/right strongly: {pan_r:.1f}°"

    # 3. Trajectory Stage Consistency (NO double/triple pan accumulation)
    stages = kin.get_trajectory_stages(r_near, pan_l)
    for stage_name in ("hover", "descent", "clamp", "lift"):
        assert abs(stages[stage_name]["shoulder_pan.pos"] - round(pan_l, 2)) < 1e-2, (
            f"Stage {stage_name} pan {stages[stage_name]['shoulder_pan.pos']} must match target pan {pan_l}"
        )

    # 4. Safe ground clearance preserved across both near and far reaches
    for d_pose in (descent_near, descent_far):
        _, zfk, pitch = kin.forward_kinematics(
            d_pose["shoulder_lift.pos"],
            d_pose["elbow_flex.pos"],
            d_pose["wrist_flex.pos"],
        )
        assert abs(zfk - (kin.z_floor + kin.ground_clearance_m)) < 2e-3, (
            f"Descent height {zfk*100:.2f}cm must match clearance height {(kin.z_floor + kin.ground_clearance_m)*100:.2f}cm"
        )
        assert -91.0 <= pitch <= -55.0


def test_real_floor_image_and_kinematic_descent():
    """Verify detection and 3D floor kinematics on captured real hardware images."""
    import yaml
    import cv2
    from src.vision_grasp import SockDetector, VisionGraspPlanner

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    planner = VisionGraspPlanner(cfg)
    detector = SockDetector(
        min_area=float(cfg["case4"]["detection"]["min_area"]),
        max_area=float(cfg["case4"]["detection"]["max_area"]),
        contrast_threshold=float(cfg["case4"]["detection"]["contrast_threshold"]),
        border_margin=int(cfg["case4"]["detection"]["border_margin"]),
        roi_top_pct=float(cfg["case4"]["detection"]["roi_top_pct"]),
        roi_bottom_pct=float(cfg["case4"]["detection"]["roi_bottom_pct"]),
        roi_left_pct=float(cfg["case4"]["detection"]["roi_left_pct"]),
        roi_right_pct=float(cfg["case4"]["detection"]["roi_right_pct"]),
        save_debug_image=False,
    )

    if os.path.exists("/tmp/scan_middle.jpg"):
        img = cv2.imread("/tmp/scan_middle.jpg")
        det = detector.detect(img)
        assert det.found is True, "Must detect the black sock in /tmp/scan_middle.jpg"
        r_tgt, pan_tgt, roll_tgt = planner.kinematics.compute_sock_target(
            det,
            scan_pan_deg=-7.1,
            pan_direction=planner.pan_direction,
            scan_pose=cfg["case4"]["scan_poses"]["center"],
        )
        assert -34.0 <= pan_tgt <= -22.0, f"Expected scan_middle pan toward sock (~ -26°..-30°), got {pan_tgt:.1f}°"
        assert 0.25 <= r_tgt <= 0.315, f"Expected reach toward sock [25, 31.5]cm, got {r_tgt*100:.1f}cm"
        assert roll_tgt == -0.7

        stages = planner.kinematics.get_trajectory_stages(r_tgt, pan_tgt, roll_tgt)
        descent = stages["descent"]
        _, z_fk, _ = planner.kinematics.forward_kinematics(
            descent["shoulder_lift.pos"], descent["elbow_flex.pos"], descent["wrist_flex.pos"]
        )
        assert z_fk >= planner.kinematics.z_floor + 0.020, (
            f"Descent Z={z_fk*100:.2f}cm must stay >= 2.0cm above raw floor contact {planner.kinematics.z_floor*100:.2f}cm"
        )
        assert stages["hover"]["shoulder_lift.pos"] < descent["shoulder_lift.pos"]

    if os.path.exists("/tmp/cl_hover1.jpg"):
        img_hov = cv2.imread("/tmp/cl_hover1.jpg")
        det_hov = detector.detect(
            img_hov, allow_border_touch=True, max_area_override=220000.0
        )
        assert det_hov.found is True, "Must detect close-up black sock touching top edge in /tmp/cl_hover1.jpg"
        assert det_hov.cy < 150.0, f"Must lock onto real dark sock at top (cy < 150), not floor shadow (got cy={det_hov.cy:.1f})"


def test_arm_front_stow_clearance_recovery():
    """Verify move_to_stow safely clears forward floor postures without tucking jaws backward into chassis."""
    cfg = {
        "dry_run": True,
        "arm": {"follower_port": "/dev/null", "interpolation_time_s": 0.01},
    }
    arm = ArmController(cfg, dry_run=True)
    arm.connect_follower()
    arm.sim_follower_pose = {
        "shoulder_pan.pos": -30.0,
        "shoulder_lift.pos": 88.0,
        "elbow_flex.pos": -50.0,
        "wrist_flex.pos": 35.0,
        "wrist_roll.pos": -0.7,
        "gripper.pos": 85.0,
    }
    arm.move_to_stow(duration_s=0.01)
    for k, v in arm.stow_pose.items():
        assert abs(arm.sim_follower_pose[k] - v) < 1e-3, f"Joint {k} did not reach stow pose"


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
    test_arm_safety_and_stow_recovery()
    print("  [PASS] test_arm_safety_and_stow_recovery")
    test_scan_poses_teaching_and_execution()
    print("  [PASS] test_scan_poses_teaching_and_execution")
    test_shoulder_pan_preservation_and_centroid_alignment()
    print("  [PASS] test_shoulder_pan_preservation_and_centroid_alignment")
    test_so100_analytical_kinematics_and_clearance()
    print("  [PASS] test_so100_analytical_kinematics_and_clearance")
    test_centroid_to_cartesian_floor_targeting()
    print("  [PASS] test_centroid_to_cartesian_floor_targeting")
    test_real_floor_image_and_kinematic_descent()
    print("  [PASS] test_real_floor_image_and_kinematic_descent")
    test_arm_front_stow_clearance_recovery()
    print("  [PASS] test_arm_front_stow_clearance_recovery")
    print("\n>>> ALL TESTS PASSED SUCCESSFULLY! <<<")


