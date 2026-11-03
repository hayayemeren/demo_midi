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
    print("  [Q] Quit Demonstration")
    print("=" * 64)

    default_prompt = f" [{default_case}]" if default_case in (1, 2, 3) else " [1]"
    while True:
        try:
            choice = input(f"Enter choice (1, 2, 3, or Q){default_prompt}: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            sys.exit(0)

        if not choice:
            return default_case if default_case in (1, 2, 3) else 1
        if choice in ("1", "case 1", "c1"):
            return 1
        elif choice in ("2", "case 2", "c2"):
            return 2
        elif choice in ("3", "case 3", "c3"):
            return 3
        elif choice in ("q", "quit", "exit"):
            print("Exiting.")
            sys.exit(0)
        else:
            print("Invalid selection. Please enter 1, 2, 3, or Q.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="MIDI Mobile Manipulator: Base Drive & Arm Teleop Switch"
    )
    parser.add_argument(
        "-C", "--case",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Demo case to run (1: Leader Arm Mirroring, 2: Gamepad Arm Rate Control, 3: Network Teleoperation Server). If omitted, displays interactive menu."
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

    # Determine demonstration case
    case = args.case
    if case is None:
        cfg_case = config.get("case")
        if cfg_case in (1, 2, 3):
            case = cfg_case
        else:
            case = select_case_interactive(default_case=1)

    case_names = {
        1: "Case 1: Base Drive + SO-100 Leader Arm Teleoperation (Local on Pi)",
        2: "Case 2: Base Drive + Logitech Gamepad Arm Teleoperation (All-in-One)",
        3: "Case 3: Remote Network Teleoperation (Pi Robot Server <- PC Client)",
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
