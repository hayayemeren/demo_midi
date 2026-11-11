#!/usr/bin/env python3
"""MIDI Mobile Manipulator Demonstration: Base Drive & Arm Teleoperation Switch.

Main application entrypoint.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import sys
import threading
import time
from typing import Optional
import yaml

try:
    import cv2
except ImportError:
    cv2 = None

# Add project root to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from src.arm_controller import ArmController
from src.base_controller import BaseController
from src.gamepad import F710Gamepad
from src.state_machine import DemoStateMachine
from src.vision_grasp import VisionGraspPlanner, WristCamera, SockDetector


def setup_logging(verbose: bool = False):
    """Configure formatted console logging."""
    level = logging.DEBUG if verbose else logging.INFO
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=log_format, datefmt="%H:%M:%S")


def resolve_port(primary: str, fallback: str) -> str:
    """Return primary port if exists on filesystem; otherwise fallback."""
    if primary and os.path.exists(primary):
        return primary
    if fallback and os.path.exists(fallback):
        return fallback
    return primary or fallback


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file and resolve ports."""
    if not os.path.exists(config_path):
        logging.error("Configuration file '%s' not found!", config_path)
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        try:
            cfg = yaml.safe_load(f) or {}
        except Exception as e:
            logging.error("Failed to parse YAML configuration: %s", e)
            sys.exit(1)

    # Resolve ports
    serial_cfg = cfg.setdefault("serial", {})
    serial_cfg["port"] = resolve_port(
        serial_cfg.get("port", ""),
        serial_cfg.get("fallback_port", "/dev/ttyUSB0"),
    )

    arm_cfg = cfg.setdefault("arm", {})
    arm_cfg["follower_port"] = resolve_port(
        arm_cfg.get("follower_port", ""),
        arm_cfg.get("follower_fallback_port", "/dev/ttyACM0"),
    )
    arm_cfg["leader_port"] = resolve_port(
        arm_cfg.get("leader_port", ""),
        arm_cfg.get("leader_fallback_port", "/dev/ttyACM1"),
    )

    return cfg


def select_case_interactive(default_case: Optional[int] = None) -> int:
    """Display interactive menu to choose demonstration case."""
    if not sys.stdin.isatty():
        return default_case if default_case in (1, 2) else 1

    print("\n" + "=" * 64)
    print("              SELECT DEMONSTRATION CASE")
    print("=" * 64)
    print("  [1] Case 1: Base Drive + SO-100 Leader Arm Teleoperation (Local on Pi)")
    print("      - Base drive with Logitech F710 Gamepad")
    print("      - Press [Y] -> Lock wheels & mirror physical Leader Arm (1:1)")
    print()
    print("  [2] Case 2: Base Drive + Logitech Gamepad Arm Teleoperation (All-in-One)")
    print("      - Base drive with Logitech F710 Gamepad")
    print("      - Press [Y] -> Lock wheels & control SO-100 Arm with Gamepad")
    print("        * Right Stick: Pan (L/R) & Lift (U/D)")
    print("        * Left Stick:  Wrist Roll (L/R) & Elbow Flex (U/D)")
    print("        * D-Pad:       Wrist Flex Pitch (U/D) & Fine Roll (L/R)")
    print("        * Bumpers:     LB (Open Gripper), RB (Close Gripper)")
    print("        * Buttons:     [A] Toggle Gripper, [X] Move to Ready Pose")
    print()
    print("  [3] Case 3: Remote Network Teleoperation (Leader Arm on PC -> Robot)")
    print("      - Leader Arm is connected to PC; Pi runs Robot Server (port 8765)")
    print("      - Operator controls both Base and Arm from PC using Leader Arm")
    print("      - Press [SPACE] on computer to switch between Base Drive & Arm Teleop")
    print()
    print("  [4] Case 4: Vision-Guided Autonomous Sock Collection & Basket Tidy (Wrist Cam)")
    print("      - Base drive with Logitech F710 Gamepad")
    print("      - Press [Y] -> Lock wheels, wrist camera locates sock on floor")
    print("      - Visual servoing aligns gripper, descends & grasps sock")
    print("      - Verifies grasp & deposits sock into chassis-mounted basket!")
    print("      - Press [X] -> Direct manual 'Deposit to Basket' macro")
    print()
    print("  [5] Teach Chassis Basket Pose (--teach-basket)")
    print("      - Move limp arm by hand directly over basket, press ENTER to save")
    print()
    print("  [6] Teach Floor Grasp Pose (--teach-grasp)")
    print("      - Move limp arm by hand down to sock on floor below base, press ENTER to save")
    print()
    print("  [7] Test Floor Pickup & Basket Deposit Sequence (--test-pick)")
    print("      - Executes floor descend -> clamp -> lift verify -> basket drop test")
    print()
    print("  [8] Teach 3-Point Floor Scan Poses (--teach-scan)")
    print("      - Move limp arm to Middle, Left, and Right scan poses with live camera feed")
    print()
    print("  [Q] Quit Demonstration")
    print("=" * 64)

    default_prompt = f" [{default_case}]" if default_case in (1, 2, 3, 4) else " [1]"
    while True:
        try:
            choice = input(f"Enter choice (1, 2, 3, 4, 5, 6, 7, 8, or Q){default_prompt}: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            sys.exit(0)

        if not choice:
            return default_case if default_case in (1, 2, 3, 4) else 1
        if choice in ("1", "case 1", "c1"):
            return 1
        elif choice in ("2", "case 2", "c2"):
            return 2
        elif choice in ("3", "case 3", "c3"):
            return 3
        elif choice in ("4", "case 4", "c4"):
            return 4
        elif choice in ("5", "teach basket", "basket"):
            return 5
        elif choice in ("6", "teach grasp", "grasp"):
            return 6
        elif choice in ("7", "test pick", "test"):
            return 7
        elif choice in ("8", "teach scan", "scan"):
            return 8
        elif choice in ("q", "quit", "exit"):
            print("Exiting.")
            sys.exit(0)
        else:
            print("Invalid selection. Please enter 1, 2, 3, 4, 5, 6, 7, 8, or Q.")


def update_config_basket_pose(config_path: str, pose: dict) -> bool:
    """Save calibrated basket pose into config.yaml, clamping to safe limits and preserving file comments."""
    import re
    if not os.path.exists(config_path):
        logging.error("Config file %s does not exist!", config_path)
        return False

    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Clamp wrist joints safely within mechanical hard stops
    w_flex = max(-96.0, min(96.0, float(pose.get("wrist_flex.pos", 50.0))))
    w_roll = max(-175.0, min(175.0, float(pose.get("wrist_roll.pos", 0.0))))
    s_lift = max(-100.0, min(100.0, float(pose.get("shoulder_lift.pos", -80.0))))
    e_flex = max(-92.0, min(92.0, float(pose.get("elbow_flex.pos", 20.0))))

    new_basket_block = (
        "  basket_pose:\n"
        f"    shoulder_pan.pos: {pose.get('shoulder_pan.pos', 0.0):.1f}\n"
        f"    shoulder_lift.pos: {s_lift:.1f}\n"
        f"    elbow_flex.pos: {e_flex:.1f}\n"
        f"    wrist_flex.pos: {w_flex:.1f}\n"
        f"    wrist_roll.pos: {w_roll:.1f}\n"
        "    gripper.pos: 85.0"
    )

    pattern = r"  basket_pose:\n(?:    [a-z_.]+: -?[0-9.]+\n?)+"
    if re.search(pattern, content):
        new_content = re.sub(pattern, new_basket_block, content, count=1)
    else:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            cfg.setdefault("case4", {})["basket_pose"] = {
                "shoulder_pan.pos": round(float(pose.get("shoulder_pan.pos", 0.0)), 1),
                "shoulder_lift.pos": round(s_lift, 1),
                "elbow_flex.pos": round(e_flex, 1),
                "wrist_flex.pos": round(w_flex, 1),
                "wrist_roll.pos": round(w_roll, 1),
                "gripper.pos": 85.0,
            }
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
            return True
        except Exception as e:
            logging.error("Failed to update config.yaml: %s", e)
            return False

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    return True


def update_config_grasp_pose(config_path: str, pose: dict) -> bool:
    """Save calibrated floor grasp pose into config.yaml, preserving file comments and formatting."""
    import re
    if not os.path.exists(config_path):
        logging.error("Config file %s does not exist!", config_path)
        return False

    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    w_flex = max(-96.0, min(96.0, float(pose.get("wrist_flex.pos", 80.0))))
    w_roll = max(-175.0, min(175.0, float(pose.get("wrist_roll.pos", 0.0))))
    s_lift = max(-100.0, min(100.0, float(pose.get("shoulder_lift.pos", -30.0))))
    e_flex = max(-92.0, min(92.0, float(pose.get("elbow_flex.pos", 45.0))))

    new_grasp_block = (
        "  floor_grasp_pose:\n"
        f"    shoulder_pan.pos: {pose.get('shoulder_pan.pos', 0.0):.1f}\n"
        f"    shoulder_lift.pos: {s_lift:.1f}\n"
        f"    elbow_flex.pos: {e_flex:.1f}\n"
        f"    wrist_flex.pos: {w_flex:.1f}\n"
        f"    wrist_roll.pos: {w_roll:.1f}\n"
        "    gripper.pos: 0.0"
    )

    pattern = r"  floor_grasp_pose:\n(?:    [a-z_.]+: -?[0-9.]+\n?)+"
    if re.search(pattern, content):
        new_content = re.sub(pattern, new_grasp_block, content, count=1)
    else:
        pattern_basket = r"  basket_pose:"
        if re.search(pattern_basket, content):
            new_content = re.sub(pattern_basket, new_grasp_block + "\n\n  basket_pose:", content, count=1)
        else:
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                cfg.setdefault("case4", {})["floor_grasp_pose"] = {
                    "shoulder_pan.pos": round(float(pose.get("shoulder_pan.pos", 0.0)), 1),
                    "shoulder_lift.pos": round(s_lift, 1),
                    "elbow_flex.pos": round(e_flex, 1),
                    "wrist_flex.pos": round(w_flex, 1),
                    "wrist_roll.pos": round(w_roll, 1),
                    "gripper.pos": 0.0,
                }
                with open(config_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
                return True
            except Exception as e:
                logging.error("Failed to update config.yaml: %s", e)
                return False

    # Also update ground_clearance_m so IK descends to the exact taught height above z_floor (-0.155m)
    try:
        from src.kinematics import SO100Kinematics
        cfg_parsed = yaml.safe_load(new_content) or {}
        kin_cfg = cfg_parsed.get("case4", {}).get("kinematics", {})
        base_h = float(kin_cfg.get("base_height_m", 0.155))
        kin = SO100Kinematics(base_height_m=base_h)
        _, z_taught, _ = kin.forward_kinematics(s_lift, e_flex, w_flex)
        new_clearance = max(-0.015, min(0.060, z_taught - (-abs(base_h))))
        new_content = re.sub(
            r"(ground_clearance_m:\s*)-?[0-9.]+",
            f"\\g<1>{new_clearance:.3f}",
            new_content,
            count=1,
        )
    except Exception as e:
        logging.debug("Could not auto-compute ground_clearance_m from taught pose: %s", e)

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    return True


def update_config_scan_poses(
    config_path: str,
    center_pose: dict,
    left_pose: dict,
    right_pose: dict,
) -> bool:
    """Save 3-point calibrated scan poses (center, left, right) and baseline scan_pose into config.yaml."""
    import re
    if not os.path.exists(config_path):
        logging.error("Config file %s does not exist!", config_path)
        return False

    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    def clamp_pose(p: dict) -> dict:
        return {
            "shoulder_pan.pos": max(-110.0, min(110.0, float(p.get("shoulder_pan.pos", 0.0)))),
            "shoulder_lift.pos": max(-100.0, min(100.0, float(p.get("shoulder_lift.pos", -50.0)))),
            "elbow_flex.pos": max(-92.0, min(92.0, float(p.get("elbow_flex.pos", 35.0)))),
            "wrist_flex.pos": max(-96.0, min(96.0, float(p.get("wrist_flex.pos", 70.0)))),
            "wrist_roll.pos": max(-175.0, min(175.0, float(p.get("wrist_roll.pos", 0.0)))),
            "gripper.pos": max(0.0, min(100.0, float(p.get("gripper.pos", 10.0)))),
        }

    c = clamp_pose(center_pose)
    l = clamp_pose(left_pose)
    r = clamp_pose(right_pose)

    new_scan_pose_block = (
        "  scan_pose:\n"
        f"    shoulder_pan.pos: {c['shoulder_pan.pos']:.1f}\n"
        f"    shoulder_lift.pos: {c['shoulder_lift.pos']:.1f}\n"
        f"    elbow_flex.pos: {c['elbow_flex.pos']:.1f}\n"
        f"    wrist_flex.pos: {c['wrist_flex.pos']:.1f}\n"
        f"    wrist_roll.pos: {c['wrist_roll.pos']:.1f}\n"
        f"    gripper.pos: {c['gripper.pos']:.1f}\n"
    )

    new_scan_poses_block = (
        "  scan_poses:\n"
        "    center:\n"
        f"      shoulder_pan.pos: {c['shoulder_pan.pos']:.1f}\n"
        f"      shoulder_lift.pos: {c['shoulder_lift.pos']:.1f}\n"
        f"      elbow_flex.pos: {c['elbow_flex.pos']:.1f}\n"
        f"      wrist_flex.pos: {c['wrist_flex.pos']:.1f}\n"
        f"      wrist_roll.pos: {c['wrist_roll.pos']:.1f}\n"
        f"      gripper.pos: {c['gripper.pos']:.1f}\n"
        "    left:\n"
        f"      shoulder_pan.pos: {l['shoulder_pan.pos']:.1f}\n"
        f"      shoulder_lift.pos: {l['shoulder_lift.pos']:.1f}\n"
        f"      elbow_flex.pos: {l['elbow_flex.pos']:.1f}\n"
        f"      wrist_flex.pos: {l['wrist_flex.pos']:.1f}\n"
        f"      wrist_roll.pos: {l['wrist_roll.pos']:.1f}\n"
        f"      gripper.pos: {l['gripper.pos']:.1f}\n"
        "    right:\n"
        f"      shoulder_pan.pos: {r['shoulder_pan.pos']:.1f}\n"
        f"      shoulder_lift.pos: {r['shoulder_lift.pos']:.1f}\n"
        f"      elbow_flex.pos: {r['elbow_flex.pos']:.1f}\n"
        f"      wrist_flex.pos: {r['wrist_flex.pos']:.1f}\n"
        f"      wrist_roll.pos: {r['wrist_roll.pos']:.1f}\n"
        f"      gripper.pos: {r['gripper.pos']:.1f}\n"
    )

    pattern_scan_poses = r"  scan_poses:\n(?:    [a-z0-9_]+:\n(?:      [a-z_.]+: -?[0-9.]+\n)+)+"
    if re.search(pattern_scan_poses, content):
        content = re.sub(pattern_scan_poses, lambda _: new_scan_poses_block, content, count=1)
    else:
        pattern_sweep = r"(  scan_sweep_angles:[^\n]*\n)"
        if re.search(pattern_sweep, content):
            content = re.sub(pattern_sweep, lambda m: new_scan_poses_block + "\n" + m.group(1), content, count=1)
        else:
            pattern_sp = r"(  scan_pose:\n(?:    [a-z_.]+: -?[0-9.]+\n)+)"
            if re.search(pattern_sp, content):
                content = re.sub(pattern_sp, lambda m: m.group(1) + "\n" + new_scan_poses_block, content, count=1)
            else:
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        cfg = yaml.safe_load(f) or {}
                    cfg.setdefault("case4", {})["scan_pose"] = {k: round(v, 1) for k, v in c.items()}
                    cfg.setdefault("case4", {})["scan_poses"] = {
                        "center": {k: round(v, 1) for k, v in c.items()},
                        "left": {k: round(v, 1) for k, v in l.items()},
                        "right": {k: round(v, 1) for k, v in r.items()},
                    }
                    with open(config_path, "w", encoding="utf-8") as f:
                        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
                    return True
                except Exception as e:
                    logging.error("Failed to update config.yaml: %s", e)
                    return False

    pattern_scan_pose = r"  scan_pose:\n(?:    [a-z_.]+: -?[0-9.]+\n)+"
    if re.search(pattern_scan_pose, content):
        content = re.sub(pattern_scan_pose, lambda _: new_scan_pose_block, content, count=1)

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(content)
    return True


def teach_basket(config_path: str, dry_run: bool = False):
    """Interactive utility to teach and record the chassis basket drop-off pose."""
    print("\n" + "=" * 64)
    print("         TEACH CHASSIS BASKET POSE (CASE 4 CALIBRATION)")
    print("=" * 64)
    print("1. Connecting to SO-100 Follower arm...")
    config = load_config(config_path)
    arm = ArmController(config, dry_run=dry_run)

    if not arm.connect_follower():
        print(f"Error: Could not connect to Follower Arm on {arm.follower_port}")
        return

    print("2. Disabling motor torque so you can move the arm freely by hand...")
    if not dry_run and arm.follower is not None:
        try:
            arm.follower.bus.disable_torque()
            print("   >>> TORQUE DISABLED! Arm is now limp. <<<")
        except Exception as e:
            print(f"   Warning: Could not disable torque automatically: {e}")
    else:
        print("   [DRY RUN] Simulation mode.")

    print("\n3. Instructions:")
    print("   - Physically guide the follower arm over your chassis basket.")
    print("   - Make sure it is centered over the basket and elevated enough.")
    print("   - Press [ENTER] when the arm is in the desired release position.")
    print("   - Or press [Ctrl+C] to cancel.\n")

    try:
        input("Press [ENTER] to capture current arm pose... ")
    except (KeyboardInterrupt, EOFError):
        print("\nTeaching cancelled.")
        return

    pose = arm.get_follower_pose()
    print("\n>>> CAPTURED BASKET POSE: <<<")
    for joint in ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos"]:
        val = round(float(pose.get(joint, 0.0)), 1)
        print(f"  {joint}: {val:+.1f}°")
    print("  gripper.pos: 85.0% (Open for release)")

    confirm = input("\nSave this pose to config.yaml as case4.basket_pose? [Y/n]: ").strip().lower()
    if confirm in ("", "y", "yes"):
        if update_config_basket_pose(config_path, pose):
            print(f"\n[SUCCESS] New basket pose saved to '{config_path}'!")
            print("The autonomous sequence in Case 4 will now drop socks at this location.\n")
        else:
            print(f"\n[ERROR] Failed to save pose to '{config_path}'.")
    else:
        print("\nPose was NOT saved.")


def teach_grasp(config_path: str, dry_run: bool = False):
    """Interactive utility to teach and record the floor grasp pose (reaching below base)."""
    print("\n" + "=" * 64)
    print("       TEACH FLOOR GRASP POSE (CASE 4 FLOOR REACH CALIBRATION)")
    print("=" * 64)
    print("1. Connecting to SO-100 Follower arm...")
    config = load_config(config_path)
    arm = ArmController(config, dry_run=dry_run)

    if not arm.connect_follower():
        print(f"Error: Could not connect to Follower Arm on {arm.follower_port}")
        return

    print("2. Disabling motor torque so you can move the arm freely by hand...")
    if not dry_run and arm.follower is not None:
        try:
            arm.follower.bus.disable_torque()
            print("   >>> TORQUE DISABLED! Arm is now limp. <<<")
        except Exception as e:
            print(f"   Warning: Could not disable torque automatically: {e}")
    else:
        print("   [DRY RUN] Simulation mode.")

    print("\n3. Instructions:")
    print("   - Place a sock on the floor directly in front of the robot.")
    print("   - Physically lower the follower arm down until the gripper jaws")
    print("     reach the floor and surround the sock.")
    print("   - Ensure the gripper fingers hover right at floor level without digging in.")
    print("   - Press [ENTER] when the arm is in position.")
    print("   - Or press [Ctrl+C] to cancel.\n")

    try:
        input("Press [ENTER] to capture current floor grasp pose... ")
    except (KeyboardInterrupt, EOFError):
        print("\nTeaching cancelled.")
        return

    pose = arm.get_follower_pose()
    print("\n>>> CAPTURED FLOOR GRASP POSE: <<<")
    for joint in ["shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos"]:
        val = round(float(pose.get(joint, 0.0)), 1)
        print(f"  {joint}: {val:+.1f}°")
    print("  shoulder_pan.pos: 0.0° (Center baseline)")
    print("  gripper.pos: 0.0% (Closed clamp)")

    confirm = input("\nSave this pose to config.yaml as case4.floor_grasp_pose? [Y/n]: ").strip().lower()
    if confirm in ("", "y", "yes"):
        if update_config_grasp_pose(config_path, pose):
            print(f"\n[SUCCESS] New floor grasp pose saved to '{config_path}'!")
            print("Autonomous Case 4 will now reach this exact floor level below the chassis base.")

            test_confirm = input("\nWould you like to test the pickup sequence now? [Y/n]: ").strip().lower()
            if test_confirm in ("", "y", "yes"):
                test_pick(config_path, dry_run=dry_run)
        else:
            print(f"\n[ERROR] Failed to save pose to '{config_path}'.")
    else:
        print("\nPose was NOT saved.")


def teach_scan(config_path: str, dry_run: bool = False):
    """Interactive utility to teach and record 3 floor scan poses (Middle, Left, Right) with live camera feed."""
    print("\n" + "=" * 64)
    print("      TEACH 3-POINT FLOOR SCAN POSES (CASE 4 CALIBRATION)")
    print("=" * 64)
    print("1. Connecting to SO-100 Follower arm...")
    config = load_config(config_path)
    arm = ArmController(config, dry_run=dry_run)

    if not arm.connect_follower():
        print(f"Error: Could not connect to Follower Arm on {arm.follower_port}")
        return

    print("2. Disabling motor torque so you can move the arm freely by hand...")
    if not dry_run and arm.follower is not None:
        try:
            arm.follower.bus.disable_torque()
            print("   >>> TORQUE DISABLED! Arm is now limp. <<<")
        except Exception as e:
            print(f"   Warning: Could not disable torque automatically: {e}")
    else:
        print("   [DRY RUN] Simulation mode.")

    def get_local_ip() -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    # 3. Connect Camera & Start Live Stream
    camera = None
    stop_gui = threading.Event()
    gui_thread = None
    try:
        cam_cfg = config.get("case4", {}).get("camera", {})
        det_cfg = config.get("case4", {}).get("detection", {})
        camera = WristCamera(
            device_path=cam_cfg.get("device", "/dev/video0"),
            fallback_index=int(cam_cfg.get("fallback", 0)),
            width=int(cam_cfg.get("width", 640)),
            height=int(cam_cfg.get("height", 480)),
            fps=int(cam_cfg.get("fps", 30)),
            dry_run=dry_run,
        )
        detector = SockDetector(
            mode=det_cfg.get("mode", "auto"),
            min_area=float(det_cfg.get("min_area", 800.0)),
            max_area=float(det_cfg.get("max_area", 80000.0)),
            contrast_threshold=float(det_cfg.get("contrast_threshold", 35.0)),
            border_margin=int(det_cfg.get("border_margin", 4)),
            roi_top_pct=float(det_cfg.get("roi_top_pct", 0.06)),
            roi_bottom_pct=float(det_cfg.get("roi_bottom_pct", 0.86)),
            roi_left_pct=float(det_cfg.get("roi_left_pct", 0.04)),
            roi_right_pct=float(det_cfg.get("roi_right_pct", 0.96)),
            enable_web_stream=True,
            web_stream_port=int(det_cfg.get("web_stream_port", 8080)),
        )
        if camera.connect():
            camera.start_stream(detector=detector)
            robot_ip = get_local_ip()
            print("\n" + "=" * 64)
            print("   >>> [LIVE CAMERA FEED ACTIVE] <<<")
            print(f"   * Web Browser: http://{robot_ip}:8080/ (open on phone, laptop, or PC)")
            print(f"   * Snapshots:   /tmp/sock_raw.jpg & /tmp/sock_detection.jpg")
            if os.environ.get("DISPLAY") and cv2 is not None:
                print("   * Local Window: Opening X11 preview window...")
                def _gui_worker():
                    try:
                        while not stop_gui.is_set():
                            with camera._frame_lock:
                                f = camera._latest_raw_frame
                            if f is not None:
                                cv2.imshow("Wrist Camera Feed - Teach Scan", f)
                                cv2.waitKey(30)
                            else:
                                time.sleep(0.05)
                    except Exception:
                        pass
                    finally:
                        try:
                            cv2.destroyAllWindows()
                        except Exception:
                            pass
                gui_thread = threading.Thread(target=_gui_worker, daemon=True)
                gui_thread.start()
            print("=" * 64)
    except Exception as e:
        print(f"Note: Camera live stream setup: {e}")

    points = [
        ("center", "MIDDLE (CENTER)", "Aim camera down at the floor directly in front of the robot chassis."),
        ("left", "LEFT", "Aim camera down towards the floor on the LEFT side of the robot chassis."),
        ("right", "RIGHT", "Aim camera down towards the floor on the RIGHT side of the robot chassis."),
    ]
    captured_poses = {}

    try:
        for idx, (key, name, hint) in enumerate(points, 1):
            print(f"\n[{idx}/3] {name} SCAN POSE:")
            print(f"   - {hint}")
            print("   - Check the live camera stream to ensure proper floor visibility.")
            try:
                input(f"   Press [ENTER] to capture {name} pose (or Ctrl+C to cancel)... ")
            except (KeyboardInterrupt, EOFError):
                print("\nTeaching cancelled.")
                return

            p = arm.get_follower_pose()
            captured_poses[key] = p
            print(f"   Captured: pan={float(p.get('shoulder_pan.pos', 0.0)):+.1f}°, lift={float(p.get('shoulder_lift.pos', 0.0)):+.1f}°, elbow={float(p.get('elbow_flex.pos', 0.0)):+.1f}°, wrist_flex={float(p.get('wrist_flex.pos', 0.0)):+.1f}°")

        print("\n" + "=" * 68)
        print(">>> CAPTURED 3-POINT SCAN POSES SUMMARY <<<")
        print("=" * 68)
        joints = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos"]
        print(f"{'Joint':<20} | {'Middle (Center)':<16} | {'Left':<12} | {'Right':<12}")
        print("-" * 68)
        for j in joints:
            c_v = f"{float(captured_poses['center'].get(j, 0.0)):+.1f}°"
            l_v = f"{float(captured_poses['left'].get(j, 0.0)):+.1f}°"
            r_v = f"{float(captured_poses['right'].get(j, 0.0)):+.1f}°"
            print(f"{j:<20} | {c_v:<16} | {l_v:<12} | {r_v:<12}")
        print("=" * 68)

        confirm = input("\nSave these 3 poses to config.yaml as case4.scan_poses? [Y/n]: ").strip().lower()
        if confirm in ("", "y", "yes"):
            if update_config_scan_poses(config_path, captured_poses["center"], captured_poses["left"], captured_poses["right"]):
                print(f"\n[SUCCESS] 3-point scan poses saved to '{config_path}'!")
                print("Autonomous Case 4 will now sweep through these 3 calibrated poses during sock search.\n")
            else:
                print(f"\n[ERROR] Failed to save poses to '{config_path}'.")
        else:
            print("\nPoses were NOT saved.")

    finally:
        stop_gui.set()
        if gui_thread and gui_thread.is_alive():
            gui_thread.join(timeout=1.0)
        if camera is not None:
            camera.release()


def test_pick(config_path: str, dry_run: bool = False):
    """Test full pick-and-place sequence from floor to basket with gentle motions and live web debug stream."""
    print("\n" + "=" * 64)
    print("      TEST FLOOR PICKUP & BASKET DEPOSIT SEQUENCE")
    print("=" * 64)
    config = load_config(config_path)

    base = BaseController(config, dry_run=dry_run)
    base.connect()
    base.hold_lock()

    arm = ArmController(config, dry_run=dry_run)
    if not arm.connect_follower():
        print("Error: Could not connect to Follower Arm.")
        base.release_brakes()
        return

    # Start live camera and vision web debugger if available
    camera = None
    detector = None
    try:
        cam_cfg = config.get("case4", {}).get("camera", {})
        det_cfg = config.get("case4", {}).get("detection", {})
        camera = WristCamera(
            device_path=cam_cfg.get("device", "/dev/video0"),
            fallback_index=int(cam_cfg.get("fallback", 0)),
            width=int(cam_cfg.get("width", 640)),
            height=int(cam_cfg.get("height", 480)),
            fps=int(cam_cfg.get("fps", 30)),
            dry_run=dry_run,
        )
        detector = SockDetector(
            mode=det_cfg.get("mode", "auto"),
            min_area=float(det_cfg.get("min_area", 800.0)),
            max_area=float(det_cfg.get("max_area", 80000.0)),
            contrast_threshold=float(det_cfg.get("contrast_threshold", 35.0)),
            border_margin=int(det_cfg.get("border_margin", 4)),
            roi_top_pct=float(det_cfg.get("roi_top_pct", 0.06)),
            roi_bottom_pct=float(det_cfg.get("roi_bottom_pct", 0.86)),
            roi_left_pct=float(det_cfg.get("roi_left_pct", 0.04)),
            roi_right_pct=float(det_cfg.get("roi_right_pct", 0.96)),
            enable_web_stream=True,
            web_stream_port=int(det_cfg.get("web_stream_port", 8080)),
        )
        if camera.connect():
            camera.start_stream(detector=detector)
            print(">>> [VISION DEBUG] Live Web MJPEG Stream active at http://<robot-ip>:8080/ <<<\n")
    except Exception as e:
        print(f"Note: Camera live stream not available: {e}")

    try:
        planner = VisionGraspPlanner(config)
        descend_t = planner.descend_duration_s
        clamp_t = planner.clamp_duration_s
        lift_t = planner.lift_duration_s
        stow_t = planner.stow_duration_s

        print("\n[TEST] 1. Executing 3-point floor scan sweep (gripper open 85%)...")
        best = None
        if camera is not None and detector is not None:
            best = planner.execute_scan_sweep(
                arm=arm,
                camera=camera,
                detector=detector,
                feedback_cb=lambda _: base.hold_lock(),
                open_gripper_pct=85.0,
            )

        if best is not None:
            det = best["det"]
            r_tgt, pan_tgt, roll_tgt = best["coords"]
            print(f"\n>>> [TEST VISION] Sock detected at {best['label']} ({det.cx:.0f}, {det.cy:.0f}), area={det.area:.0f}px! <<<")
            print(f"[TEST] Initial Floor Target: R={r_tgt*100:.1f}cm, Pan={pan_tgt:+.1f}°, Roll={roll_tgt:+.1f}°")
            stages = planner.kinematics.get_trajectory_stages(r_tgt, pan_tgt, roll_tgt)
            hover_p = stages["hover"]
            print(f"[TEST] 1b. Moving to pre-grasp hover (~{planner.kinematics.hover_height_m*100:.0f}cm above sock)...")
            arm.move_to_pose(hover_p, duration_s=max(1.0, descend_t * 0.5), feedback_cb=lambda _: base.hold_lock())
            r_tgt, pan_tgt, roll_tgt, det_hov = planner.refine_target_in_hover(
                arm=arm,
                camera=camera,
                detector=detector,
                r_target=r_tgt,
                pan_target=pan_tgt,
                roll_target=roll_tgt,
                hover_pose=hover_p,
                feedback_cb=lambda _: base.hold_lock(),
            )
            if det_hov is not None and det_hov.found:
                stages = planner.kinematics.get_trajectory_stages(r_tgt, pan_tgt, roll_tgt)
            grasp_open = stages["descent"]
            grasp_clamp = stages["clamp"]
            lift_p = stages["lift"]
        else:
            scan_p = dict(planner.scan_pose)
            scan_p["gripper.pos"] = 85.0
            arm.move_to_pose(scan_p, duration_s=1.8, feedback_cb=lambda _: base.hold_lock())
            grasp_p = planner.compute_grasp_pose(scan_p)
            grasp_open = dict(grasp_p)
            grasp_open["gripper.pos"] = 85.0
            grasp_clamp = dict(grasp_open)
            grasp_clamp["gripper.pos"] = 0.0
            lift_p = planner.compute_grasp_pose(grasp_clamp, is_lift=True)

        print(f"[TEST] 2. Descending to floor grasp pose ({descend_t:.1f}s smooth motion)...")
        print(f"       Joints: Pan={grasp_open.get('shoulder_pan.pos', 0.0):.1f}°, Lift={grasp_open.get('shoulder_lift.pos', 0.0):.1f}°, Elbow={grasp_open.get('elbow_flex.pos', 0.0):.1f}°")
        arm.move_to_pose(grasp_open, duration_s=descend_t, feedback_cb=lambda _: base.hold_lock())

        print(f"[TEST] 3. Clamping gripper jaws onto sock (0%) over {clamp_t:.1f}s...")
        arm.move_to_pose(grasp_clamp, duration_s=clamp_t, feedback_cb=lambda _: base.hold_lock())
        time.sleep(0.3)
        base.hold_lock()

        print(f"[TEST] 4. Elevating arm off floor via IK to verify grip ({lift_t:.1f}s)...")
        arm.move_to_pose(lift_p, duration_s=lift_t, feedback_cb=lambda _: base.hold_lock())
        time.sleep(0.3)
        base.hold_lock()

        cur_pose = arm.get_follower_pose()
        measured_grip = cur_pose.get("gripper.pos", 0.0)
        if dry_run:
            measured_grip = 25.0

        print(f"\n>>> Measured gripper aperture: {measured_grip:.1f}% (Verification Threshold: {planner.grasp_verification_threshold:.1f}%) <<<")
        is_caught = planner.verify_grasp(measured_grip)

        if is_caught:
            print("[SUCCESS] Cloth securely held between jaws!")
        else:
            print(f"[INFO] Grip aperture reading: {measured_grip:.1f}% (Threshold: {planner.grasp_verification_threshold:.1f}%).")

        print("\n[TEST] 5. Moving via compact overhead clearance to rear basket with position-gated release...")
        deposited = planner.execute_basket_deposit(arm=arm, feedback_cb=lambda _: base.hold_lock())
        if deposited:
            print("\n>>> [PASSED] Sock successfully picked from floor and deposited into basket! <<<\n")
        else:
            print("\n>>> [ABORTED] Arm did not reach rear basket; gripper kept closed to protect sock! <<<\n")

        print(f"[TEST] Returning to stow pose over {stow_t:.1f}s...")
        arm.move_to_stow(duration_s=stow_t, feedback_cb=lambda _: base.hold_lock())
    finally:
        if camera is not None:
            camera.release()
        base.release_brakes()
        print("Done.\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="MIDI Mobile Manipulator: Base Drive & Arm Teleop Switch"
    )
    parser.add_argument(
        "-C", "--case",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        default=None,
        help="Demo case to run (1: Leader Arm, 2: Gamepad Arm, 3: Network Teleop, 4: Vision Autonomous Grasp, 5: Teach Basket, 6: Teach Grasp, 7: Test Pick, 8: Teach Scan). If omitted, displays interactive menu."
    )
    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "-d", "--dry-run",
        action="store_true",
        help="Run in simulation mode without physical hardware connected"
    )
    parser.add_argument(
        "--teach-basket",
        action="store_true",
        help="Interactively calibrate and save the chassis basket drop-off pose in config.yaml"
    )
    parser.add_argument(
        "--teach-grasp",
        action="store_true",
        help="Interactively calibrate and save the floor grasp pose (reaching below base) in config.yaml"
    )
    parser.add_argument(
        "--teach-scan",
        action="store_true",
        help="Interactively calibrate and save 3-point floor scan poses (Middle, Left, Right) with live camera view in config.yaml"
    )
    parser.add_argument(
        "--test-pick",
        action="store_true",
        help="Execute a test pick-and-place sequence from floor to basket to verify reach and grip"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging output"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)

    print("================================================================")
    print("      MIDI MOBILE MANIPULATOR DEMO: DRIVE & TELEOP SWITCH       ")
    print("================================================================")

    config = load_config(args.config)
    dry_run = args.dry_run or config.get("dry_run", False)
    if dry_run:
        logging.warning("[MODE] Running in DRY-RUN SIMULATION mode.")

    if args.teach_basket:
        teach_basket(args.config, dry_run=dry_run)
        return

    if args.teach_grasp:
        teach_grasp(args.config, dry_run=dry_run)
        return

    if args.teach_scan:
        teach_scan(args.config, dry_run=dry_run)
        return

    if args.test_pick:
        test_pick(args.config, dry_run=dry_run)
        return

    # Determine demonstration case
    case = args.case
    if case is None:
        cfg_case = config.get("case")
        if cfg_case in (1, 2, 3, 4, 5, 6, 7, 8):
            case = cfg_case
        else:
            case = select_case_interactive(default_case=1)

    if case == 5:
        teach_basket(args.config, dry_run=dry_run)
        return
    elif case == 6:
        teach_grasp(args.config, dry_run=dry_run)
        return
    elif case == 7:
        test_pick(args.config, dry_run=dry_run)
        return
    elif case == 8:
        teach_scan(args.config, dry_run=dry_run)
        return

    case_names = {
        1: "Case 1: Base Drive + SO-100 Leader Arm Teleoperation (Local on Pi)",
        2: "Case 2: Base Drive + Logitech Gamepad Arm Teleoperation (All-in-One)",
        3: "Case 3: Remote Network Teleoperation (Pi Robot Server <- PC Client)",
        4: "Case 4: Vision-Guided Autonomous Sock Collection & Basket Tidy (Wrist Cam)",
    }
    case_name = case_names.get(case, f"Case {case}")
    print(f"\n>>> Running {case_name} <<<\n")

    # 1. Initialize Base Controller
    base = BaseController(config, dry_run=dry_run)
    if not base.connect():
        if not dry_run:
            logging.error("Could not connect to DDSM base motors. Exiting.")
            sys.exit(1)

    # 2. Initialize Gamepad
    gp_cfg = config.get("gamepad", {})
    gamepad = F710Gamepad(
        device_path=gp_cfg.get("device_path", None),
        deadzone=float(gp_cfg.get("deadzone", 0.10)),
        speed_levels=gp_cfg.get("speed_levels", [50.0, 100.0, 150.0, 200.0]),
        default_speed_level=int(gp_cfg.get("default_speed_level", 1)),
        switch_button=gp_cfg.get("switch_button", "Y"),
    )
    if not gamepad.connect() and not dry_run and case != 3:
        logging.warning("Logitech Gamepad not detected at startup. Will retry automatically.")

    # 3. Initialize Arm Controller
    arm = ArmController(config, dry_run=dry_run)
    follower_ok = arm.connect_follower()
    if not follower_ok and not dry_run:
        logging.warning(
            "Follower Arm not detected on %s. Ensure it is plugged in and powered.",
            arm.follower_port,
        )

    # 4. Initialize State Machine
    ctrl_cfg = config.get("control", {})
    sm = DemoStateMachine(
        base=base,
        arm=arm,
        gamepad=gamepad,
        demo_case=case,
        loop_hz=float(ctrl_cfg.get("loop_hz", 50.0)),
        telemetry_interval=float(ctrl_cfg.get("telemetry_interval", 0.20)),
        network_port=int(ctrl_cfg.get("network_port", 8765)),
        config=config,
    )

    def handle_signal(sig, frame):
        print(f"\nCaught signal {sig}. Stopping demo...")
        sm.running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        sm.start()
    except KeyboardInterrupt:
        pass
    finally:
        sm.stop()


if __name__ == "__main__":
    main()
