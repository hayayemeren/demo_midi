#!/usr/bin/env python3
"""MIDI Mobile Manipulator Demonstration: Base Drive & Arm Teleoperation Switch.

Main application entrypoint.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from typing import Optional
import yaml

# Add project root to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from src.arm_controller import ArmController
from src.base_controller import BaseController
from src.gamepad import F710Gamepad
from src.state_machine import DemoStateMachine


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
    print("  [Q] Quit Demonstration")
    print("=" * 64)

    default_prompt = f" [{default_case}]" if default_case in (1, 2, 3, 4) else " [1]"
    while True:
        try:
            choice = input(f"Enter choice (1, 2, 3, 4, or Q){default_prompt}: ").strip().lower()
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
        elif choice in ("q", "quit", "exit"):
            print("Exiting.")
            sys.exit(0)
        else:
            print("Invalid selection. Please enter 1, 2, 3, 4, or Q.")


def update_config_basket_pose(config_path: str, pose: dict) -> bool:
    """Save calibrated basket pose into config.yaml, preserving file comments and formatting."""
    import re
    if not os.path.exists(config_path):
        logging.error("Config file %s does not exist!", config_path)
        return False

    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    new_basket_block = (
        "  basket_pose:\n"
        f"    shoulder_pan.pos: {pose.get('shoulder_pan.pos', 0.0):.1f}\n"
        f"    shoulder_lift.pos: {pose.get('shoulder_lift.pos', -80.0):.1f}\n"
        f"    elbow_flex.pos: {pose.get('elbow_flex.pos', 20.0):.1f}\n"
        f"    wrist_flex.pos: {pose.get('wrist_flex.pos', 50.0):.1f}\n"
        f"    wrist_roll.pos: {pose.get('wrist_roll.pos', 0.0):.1f}\n"
        f"    gripper.pos: 85.0"
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
                "shoulder_lift.pos": round(float(pose.get("shoulder_lift.pos", -80.0)), 1),
                "elbow_flex.pos": round(float(pose.get("elbow_flex.pos", 20.0)), 1),
                "wrist_flex.pos": round(float(pose.get("wrist_flex.pos", 50.0)), 1),
                "wrist_roll.pos": round(float(pose.get("wrist_roll.pos", 0.0)), 1),
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="MIDI Mobile Manipulator: Base Drive & Arm Teleop Switch"
    )
    parser.add_argument(
        "-C", "--case",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="Demo case to run (1: Leader Arm Mirroring, 2: Gamepad Arm Rate Control, 3: Network Teleop Server, 4: Vision Autonomous Grasp). If omitted, displays interactive menu."
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

    # Determine demonstration case
    case = args.case
    if case is None:
        cfg_case = config.get("case")
        if cfg_case in (1, 2, 3, 4):
            case = cfg_case
        else:
            case = select_case_interactive(default_case=1)

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
        sm.stop()
        sys.exit(0)

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
