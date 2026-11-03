#!/usr/bin/env python3
"""MIDI Case 3: PC Teleoperation Client.

Runs on this computer connected to the SO-100 Leader Arm.
Communicates with the Raspberry Pi robot server over TCP network.
Press [SPACE] or [M] on the keyboard to switch between:
  1. Base Drive Mode: Tilt and turn the leader arm to drive mobile base.
  2. Follower Arm Teleoperation Mode: Mirrors the leader arm 1:1 in real time.
"""

from __future__ import annotations

import argparse
import logging
import os
import select
import sys
import termios
import time
import tty
from typing import Dict, Optional

# Add project root to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from src.leader_reader import SO100LeaderReader, leader_to_base_drive, DEFAULT_NEUTRAL_POSE
from src.network import RobotClient, DEFAULT_PORT

logger = logging.getLogger("demo.pc_client")

# ANSI Color Codes
COLOR_RESET   = "\033[0m"
COLOR_BOLD    = "\033[1m"
COLOR_GREEN   = "\033[92m"
COLOR_YELLOW  = "\033[93m"
COLOR_CYAN    = "\033[96m"
COLOR_MAGENTA = "\033[95m"
COLOR_RED     = "\033[91m"
CLEAR_SCREEN  = "\033[2J\033[H"

SPEED_LEVELS = [50.0, 100.0, 150.0, 200.0]


class NonBlockingKeyboard:
    """Reads single characters from stdin non-blockingly."""

    def __init__(self):
        self.is_tty = sys.stdin.isatty()
        self.old_settings = None

    def __enter__(self):
        if self.is_tty:
            try:
                self.old_settings = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
            except Exception:
                self.old_settings = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.old_settings is not None and self.is_tty:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            except Exception:
                pass

    def get_char(self) -> Optional[str]:
        """Return pressed character or None if no input."""
        if not self.is_tty:
            return None
        try:
            r, _, _ = select.select([sys.stdin], [], [], 0.0)
            if r:
                return sys.stdin.read(1)
        except Exception:
            pass
        return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="MIDI Case 3: PC Client (Leader Arm Base Drive & Teleop Switch)"
    )
    parser.add_argument(
        "-R", "--robot",
        default="144.122.83.216",
        help="Raspberry Pi IP address (default: 144.122.83.216)",
    )
    parser.add_argument(
        "-P", "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"TCP port number (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "-l", "--leader-port",
        default=None,
        help="Serial port for SO-100 Leader arm (default: auto-detect)",
    )
    parser.add_argument(
        "-d", "--dry-run",
        action="store_true",
        help="Run in simulation mode without physical Leader Arm or Robot",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def render_dashboard(
    robot_ip: str,
    robot_port: int,
    connected: bool,
    mode: str,
    pose: Dict[str, float],
    fwd: float,
    turn: float,
    speed_rpm: float,
    brake: bool,
    telemetry: Optional[Dict],
):
    """Draw a clean, flicker-free terminal dashboard."""
    mode_str = (
        f"{COLOR_BOLD}{COLOR_CYAN}[ MODE A: BASE DRIVE (Leader Arm Joystick) ]{COLOR_RESET}"
        if mode == "base"
        else f"{COLOR_BOLD}{COLOR_MAGENTA}[ MODE B: FOLLOWER ARM TELEOPERATION (1:1 Mirror) ]{COLOR_RESET}"
    )

    conn_str = (
        f"{COLOR_BOLD}{COLOR_GREEN}CONNECTED ({robot_ip}:{robot_port}){COLOR_RESET}"
        if connected
        else f"{COLOR_BOLD}{COLOR_RED}CONNECTING TO {robot_ip}:{robot_port}...{COLOR_RESET}"
    )

    pan = pose.get("shoulder_pan.pos", 0.0)
    lift = pose.get("shoulder_lift.pos", -45.0)
    elbow = pose.get("elbow_flex.pos", 15.0)
    flex = pose.get("wrist_flex.pos", 45.0)
    roll = pose.get("wrist_roll.pos", 0.0)
    grip = pose.get("gripper.pos", 0.0)

    f_pose = (telemetry or {}).get("follower_pose", {})
    f_pan = f_pose.get("shoulder_pan.pos", 0.0)
    f_lift = f_pose.get("shoulder_lift.pos", 0.0)
    f_grip = f_pose.get("gripper.pos", 0.0)
    left_rpm = (telemetry or {}).get("left_rpm", 0.0)
    right_rpm = (telemetry or {}).get("right_rpm", 0.0)
    base_locked = (telemetry or {}).get("base_locked", False)

    lock_tag = f"{COLOR_RED}[WHEELS FIRMLY LOCKED]{COLOR_RESET}" if base_locked else f"{COLOR_GREEN}[WHEELS UNLOCKED]{COLOR_RESET}"
    brake_tag = f"{COLOR_RED}ON{COLOR_RESET}" if brake else f"{COLOR_GREEN}OFF{COLOR_RESET}"

    out = (
        f"\033[H"
        f"================================================================================\n"
        f"       MIDI MOBILE MANIPULATOR CASE 3: REMOTE LEADER ARM TELEOPERATION          \n"
        f"================================================================================\n"
        f" Target Robot: {conn_str}\n"
        f" Active Mode:  {mode_str}\n"
        f" Status:       {lock_tag}\n"
        f"--------------------------------------------------------------------------------\n"
        f" {COLOR_BOLD}SO-100 Leader Arm Joint Angles (Local PC):{COLOR_RESET}\n"
        f"   Pan:   {pan:+6.1f}°  |  Lift:  {lift:+6.1f}°  |  Elbow: {elbow:+6.1f}°\n"
        f"   Flex:  {flex:+6.1f}°  |  Roll:  {roll:+6.1f}°  |  Grip:  {grip:5.1f}%\n"
        f"--------------------------------------------------------------------------------\n"
    )

    if mode == "base":
        out += (
            f" {COLOR_BOLD}Base Motion Commands Sent to Robot:{COLOR_RESET}\n"
            f"   Forward: {fwd:+5.2f} (Lift tilt)  | Steering: {turn:+5.2f} (Pan turn)\n"
            f"   Gear:    {speed_rpm:.0f} RPM               | Brake:    {brake_tag} (Squeeze gripper)\n"
            f"   Feedback: L={int(left_rpm):+4d} RPM | R={int(right_rpm):+4d} RPM\n"
        )
    else:
        out += (
            f" {COLOR_BOLD}Follower Arm Mirroring (Real-Time 1:1 Stream):{COLOR_RESET}\n"
            f"   Follower Pan: {f_pan:+6.1f}° | Lift: {f_lift:+6.1f}° | Grip: {f_grip:5.1f}%\n"
            f"   Base State:   Wheels strictly held at 0 RPM with active electric brakes\n"
            f"   Notice:       Move the leader arm in 3D space to teleoperate follower arm\n"
        )

    out += (
        f"================================================================================\n"
        f" {COLOR_BOLD}Computer Keyboard Controls:{COLOR_RESET}\n"
        f"   {COLOR_YELLOW}[SPACE / M]{COLOR_RESET} Toggle Mode (Base Drive <-> Arm Teleop)\n"
        f"   {COLOR_YELLOW}[C]{COLOR_RESET}         Recenter Neutral Point for Base Joystick\n"
        f"   {COLOR_YELLOW}[1 - 4]{COLOR_RESET}     Set Speed Gear (50 / 100 / 150 / 200 RPM)\n"
        f"   {COLOR_YELLOW}[B]{COLOR_RESET}         Toggle Emergency Brake\n"
        f"   {COLOR_YELLOW}[Q / ESC]{COLOR_RESET}   Safely Park Arm & Quit\n"
        f"================================================================================\n"
    )

    sys.stdout.write(out)
    sys.stdout.flush()


def run_pc_client():
    args = parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    print(CLEAR_SCREEN)
    print("================================================================")
    print("     MIDI CASE 3: PC CLIENT (LEADER ARM REMOTE CONTROLLER)      ")
    print("================================================================")

    # 1. Initialize Leader Arm Reader
    leader = SO100LeaderReader(
        port=args.leader_port,
        dry_run=args.dry_run,
    )
    if not leader.connect():
        if not args.dry_run:
            print(f"\n{COLOR_YELLOW}[NOTICE] SO-100 Leader Arm not detected on serial USB.{COLOR_RESET}")
            print("Plug it into this computer via USB and press ENTER to retry, or run with --dry-run.")
            try:
                input("Press ENTER to retry...")
                if not leader.connect():
                    print("Falling back to simulation mode.")
                    leader.dry_run = True
                    leader.connect()
            except KeyboardInterrupt:
                return

    # 2. Initialize Network Client
    client = RobotClient(host=args.robot, port=args.port)
    print(f"\nConnecting to Robot at {args.robot}:{args.port}...")
    client.connect()

    # Application state
    mode = "base"
    speed_idx = 1
    speed_rpm = SPEED_LEVELS[speed_idx]
    manual_brake = False
    running = True

    # Calibrate initial neutral pose for joystick
    initial_pose = leader.read_pose()
    leader.set_neutral_center(initial_pose)

    print(CLEAR_SCREEN)

    with NonBlockingKeyboard() as kbd:
        try:
            loop_period = 0.02  # 50 Hz
            t_last_render = 0.0

            while running:
                t_start = time.time()

                # A. Ensure connection to robot
                if not client.is_connected:
                    client.connect()

                # B. Read leader arm pose
                pose = leader.read_pose()

                # C. Check keyboard input
                ch = kbd.get_char()
                if ch:
                    if ch in (" ", "m", "M", "\r", "\n"):
                        # Mode Switch Button pressed on computer!
                        if mode == "base":
                            mode = "arm"
                            # Send transition into Arm mode
                            client.send_command({"mode": "arm", "arm": pose})
                        else:
                            # Switch back to Base Drive: park arm first
                            client.send_command({"mode": "base", "park": True})
                            leader.set_neutral_center(pose)
                            mode = "base"

                    elif ch in ("c", "C"):
                        # Recenter neutral
                        leader.set_neutral_center(pose)

                    elif ch in ("1", "2", "3", "4"):
                        speed_idx = int(ch) - 1
                        speed_rpm = SPEED_LEVELS[speed_idx]

                    elif ch in ("b", "B"):
                        manual_brake = not manual_brake

                    elif ch in ("q", "Q", "\x1b"):  # Q or ESC
                        print("\nExiting demonstration...")
                        client.send_command({"mode": "base", "park": True})
                        time.sleep(0.5)
                        break

                # D. Generate and send commands based on active mode
                fwd, turn, grip_brake = 0.0, 0.0, False
                if mode == "base":
                    fwd, turn, grip_brake = leader_to_base_drive(
                        pose=pose,
                        neutral_pose=leader.neutral_pose,
                        deadzone_deg=6.0,
                        max_pan_deg=35.0,
                        max_lift_deg=35.0,
                        turn_dir=-1,
                    )
                    effective_brake = manual_brake or grip_brake
                    cmd = {
                        "mode": "base",
                        "base": {
                            "forward": fwd,
                            "turn": turn,
                            "speed_rpm": speed_rpm,
                            "brake": effective_brake,
                        },
                    }
                    client.send_command(cmd)

                elif mode == "arm":
                    cmd = {
                        "mode": "arm",
                        "arm": pose,
                    }
                    client.send_command(cmd)

                # E. Poll telemetry feedback from robot
                telemetry = client.poll_telemetry()

                # F. Render Dashboard at 10 Hz
                if t_start - t_last_render >= 0.10:
                    t_last_render = t_start
                    render_dashboard(
                        robot_ip=args.robot,
                        robot_port=args.port,
                        connected=client.is_connected,
                        mode=mode,
                        pose=pose,
                        fwd=fwd,
                        turn=turn,
                        speed_rpm=speed_rpm,
                        brake=manual_brake or grip_brake,
                        telemetry=telemetry,
                    )

                # Sleep to maintain 50 Hz control rate
                elapsed = time.time() - t_start
                if loop_period > elapsed:
                    time.sleep(loop_period - elapsed)

        except KeyboardInterrupt:
            pass
        finally:
            print("\nShutting down client...")
            try:
                client.send_command({"mode": "base", "park": True})
            except Exception:
                pass
            client.disconnect()
            leader.close()
            print("Client stopped cleanly.")


if __name__ == "__main__":
    run_pc_client()
