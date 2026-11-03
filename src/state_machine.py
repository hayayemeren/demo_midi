"""Demo State Machine: Seamless Switch between Base Drive & Arm Teleoperation.

States:
  - BASE_DRIVE: Logitech F710 drives the mobile base; arm holds stow pose.
  - WAITING_FOR_LEADER: Base locked; waiting for leader arm to be connected.
  - SYNCING: Base locked; follower arm smoothly aligns with leader arm pose.
  - ARM_TELEOP: Base locked; people can play with the leader arm (1:1 follower mirror).
  - PARKING: Base locked; follower arm smoothly returns to safe stow pose.
"""

from __future__ import annotations

import enum
import logging
import sys
import time
from typing import Optional

from src.arm_controller import ArmController
from src.base_controller import BaseController
from src.gamepad import F710Gamepad, GamepadState

logger = logging.getLogger("demo.statemachine")


class DemoState(enum.Enum):
    BASE_DRIVE = "BASE_DRIVE"
    WAITING_FOR_LEADER = "WAITING_FOR_LEADER"
    SYNCING = "SYNCING"
    ARM_TELEOP_LEADER = "ARM_TELEOP_LEADER"
    ARM_TELEOP_GAMEPAD = "ARM_TELEOP_GAMEPAD"
    PARKING = "PARKING"
    STOPPING = "STOPPING"


# ANSI Color Codes for terminal UI
COLOR_RESET  = "\033[0m"
COLOR_BOLD   = "\033[1m"
COLOR_GREEN  = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_CYAN   = "\033[96m"
COLOR_MAGENTA= "\033[95m"
COLOR_RED    = "\033[91m"


class DemoStateMachine:
    """Coordinates mode transitions, hardware handshakes, and control loops."""

    def __init__(
        self,
        base: BaseController,
        arm: ArmController,
        gamepad: F710Gamepad,
        demo_case: int = 1,
        loop_hz: float = 50.0,
        telemetry_interval: float = 0.20,
    ):
        self.base = base
        self.arm = arm
        self.gamepad = gamepad
        self.demo_case = demo_case
        self.loop_hz = loop_hz
        self.telemetry_interval = telemetry_interval

        self.state = DemoState.BASE_DRIVE
        self.running = False
        self.last_telemetry_time = 0.0
        self.last_leader_check_time = 0.0
        self.transition_start_time = 0.0

    def start(self):
        """Initial check and launch of the control loop."""
        self.running = True
        logger.info("Initializing Demonstration State Machine for Case %d...", self.demo_case)

        # 1. Start Follower at stow pose
        logger.info("Setting Follower Arm to safe Stow Pose...")
        self.arm.move_to_stow(duration_s=1.0)

        # 2. Try connecting leader in background if Case 1 and available
        if self.demo_case == 1 and self.arm.is_leader_port_available():
            self.arm.connect_leader()

        case_desc = (
            f"Case 1: {COLOR_MAGENTA}SO-100 LEADER ARM TELEOPERATION{COLOR_RESET}"
            if self.demo_case == 1
            else f"Case 2: {COLOR_MAGENTA}LOGITECH GAMEPAD ARM TELEOPERATION (All-in-One){COLOR_RESET}"
        )

        logger.info(
            f"{COLOR_BOLD}{COLOR_GREEN}=== MIDI DEMO READY (Case {self.demo_case}) ==={COLOR_RESET}\n"
            f"Mode 1: {COLOR_CYAN}BASE DRIVE{COLOR_RESET} (Use Logitech sticks to drive base)\n"
            f"Toggle Button: {COLOR_YELLOW}[Y]{COLOR_RESET} on Gamepad -> Switch to {case_desc}\n"
            f"Brake: {COLOR_RED}[B]{COLOR_RESET} | Exit: {COLOR_RED}[START]{COLOR_RESET}\n"
        )

        self._run_loop()

    def _run_loop(self):
        period = 1.0 / self.loop_hz
        t_prev = time.time()

        while self.running:
            now = time.time()
            dt = max(0.001, min(0.1, now - t_prev))
            t_prev = now

            # Poll gamepad events
            gp = self.gamepad.poll()

            # Global emergency exit
            if gp.button_start_pressed:
                logger.info("START button pressed. Exiting demonstration safely...")
                self.state = DemoState.STOPPING
                break

            # State Dispatcher
            if self.state == DemoState.BASE_DRIVE:
                self._handle_base_drive(gp, dt)
            elif self.state == DemoState.WAITING_FOR_LEADER:
                self._handle_waiting_for_leader(gp, now)
            elif self.state == DemoState.SYNCING:
                self._handle_syncing()
            elif self.state == DemoState.ARM_TELEOP_LEADER:
                self._handle_arm_teleop_leader(gp)
            elif self.state == DemoState.ARM_TELEOP_GAMEPAD:
                self._handle_arm_teleop_gamepad(gp, dt)
            elif self.state == DemoState.PARKING:
                self._handle_parking()

            # Periodic Status Line
            if now - self.last_telemetry_time >= self.telemetry_interval:
                self.last_telemetry_time = now
                self._print_telemetry(gp)

            # Sleep remaining frame time
            rest = period - (time.time() - now)
            if rest > 0:
                time.sleep(rest)

    def _handle_base_drive(self, gp: GamepadState, dt: float):
        """Mode 1: Base driving with F710; Arm is stowed."""
        # Check for mode switch trigger
        if self.gamepad.is_switch_pressed(gp):
            # 1. Lock base immediately
            self.base.lock_brakes()

            # Case 1: Switch to Leader Arm Teleoperation
            if self.demo_case == 1:
                logger.info("\n[SWITCH] Button [Y] pressed -> Switching to LEADER ARM TELEOPERATION!")
                if not self.arm.is_leader_connected():
                    if self.arm.is_leader_port_available() and self.arm.connect_leader():
                        self.state = DemoState.SYNCING
                    else:
                        logger.info("\n[STATUS] Waiting for Leader Arm to be connected and turned on...")
                        self.state = DemoState.WAITING_FOR_LEADER
                else:
                    self.state = DemoState.SYNCING

            # Case 2: Switch to Logitech Gamepad Arm Teleoperation
            else:
                logger.info("\n[SWITCH] Button [Y] pressed -> Switching to GAMEPAD ARM TELEOPERATION (Case 2)!")
                self.arm.init_gamepad_teleop()
                logger.info(
                    f"\n{COLOR_BOLD}{COLOR_MAGENTA}>>> GAMEPAD ARM TELEOPERATION ACTIVE! <<<{COLOR_RESET}\n"
                    f"Right Stick: Pan (L/R) & Lift (U/D) | Left Stick: Elbow (U/D) & Roll (L/R)\n"
                    f"D-Pad: Wrist Flex (U/D) | RB: Close Grip | LB: Open Grip | A: Toggle Grip | X: Ready Pose\n"
                    f"Press {COLOR_YELLOW}[Y]{COLOR_RESET} anytime to park the arm and return to {COLOR_CYAN}BASE DRIVE{COLOR_RESET}.\n"
                )
                self.state = DemoState.ARM_TELEOP_GAMEPAD
            return

        # Drive the base
        if gp.connected:
            self.base.drive(gp.forward, gp.turn, gp.speed_rpm, dt, brake=gp.brake)
        else:
            self.base.drive(0.0, 0.0, 0.0, dt, brake=True)

    def _handle_waiting_for_leader(self, gp: GamepadState, now: float):
        """Base is locked; waiting for leader arm to be plugged in (Case 1)."""
        self.base.hold_lock()

        # Allow operator to cancel and go back to driving
        if self.gamepad.is_switch_pressed(gp):
            logger.info("\n[CANCEL] Button [Y] pressed -> Returning to BASE DRIVE.")
            self.base.release_brakes()
            self.state = DemoState.BASE_DRIVE
            return

        # Check for leader connection periodically (every 0.5s)
        if now - self.last_leader_check_time >= 0.5:
            self.last_leader_check_time = now
            if self.arm.is_leader_port_available():
                logger.info("\n[LEADER DETECTED] Attempting connection on %s...", self.arm.leader_port)
                if self.arm.connect_leader():
                    logger.info("[LEADER CONNECTED] Ready to sync arms!")
                    self.state = DemoState.SYNCING

    def _handle_syncing(self):
        """Base is locked; smoothly move follower arm to match leader arm pose (Case 1)."""
        self.base.hold_lock()
        logger.info("\n[SYNC] Aligning Follower arm with Leader arm pose (zero-jerk transition)...")
        self.arm.sync_to_leader(
            duration_s=self.arm.interpolation_time,
            feedback_cb=lambda _: self.base.hold_lock(),
        )
        logger.info(
            f"\n{COLOR_BOLD}{COLOR_MAGENTA}>>> TELEOPERATION ACTIVE! People can now play with the arm! <<<{COLOR_RESET}\n"
            f"Press {COLOR_YELLOW}[Y]{COLOR_RESET} anytime to park the arm and return to {COLOR_CYAN}BASE DRIVE{COLOR_RESET}.\n"
        )
        self.state = DemoState.ARM_TELEOP_LEADER

    def _handle_arm_teleop_leader(self, gp: GamepadState):
        """Mode 2 (Case 1): Follower arm mirrors Leader arm in real time."""
        self.base.hold_lock()

        # Check if user wants to switch back to Base Drive
        if self.gamepad.is_switch_pressed(gp):
            logger.info("\n[SWITCH] Button [Y] pressed -> Parking arm and returning to BASE DRIVE!")
            self.state = DemoState.PARKING
            return

        # If Leader arm disconnected, handle gracefully
        if not self.arm.is_leader_connected():
            logger.warning("\n[WARNING] Leader Arm disconnected! Halting teleop and returning to stow pose...")
            self.state = DemoState.PARKING
            return

        # Stream teleoperation action (1:1 mirror)
        self.arm.teleop_step()

    def _handle_arm_teleop_gamepad(self, gp: GamepadState, dt: float):
        """Mode 2 (Case 2): Follower arm controlled directly from Logitech Gamepad."""
        # Wheels strictly locked in teleop mode
        self.base.hold_lock()

        # Check if user wants to switch back to Base Drive
        if self.gamepad.is_switch_pressed(gp):
            logger.info("\n[SWITCH] Button [Y] pressed -> Parking arm and returning to BASE DRIVE!")
            self.state = DemoState.PARKING
            return

        # Rate-control teleoperation step from gamepad sticks/buttons
        self.arm.teleop_step_gamepad(gp, dt)

    def _handle_parking(self):
        """Base is locked; smoothly park follower arm back to safe stow pose."""
        self.base.hold_lock()
        logger.info("\n[PARK] Returning Follower arm to safe Stow Pose...")
        self.arm.move_to_stow(
            duration_s=self.arm.interpolation_time,
            feedback_cb=lambda _: self.base.hold_lock(),
        )
        self.base.release_brakes()
        logger.info(
            f"\n{COLOR_BOLD}{COLOR_CYAN}>>> BASE DRIVE ACTIVE! Mobile base is ready to drive! <<<{COLOR_RESET}\n"
        )
        self.state = DemoState.BASE_DRIVE

    def _print_telemetry(self, gp: GamepadState):
        """Render a single-line dynamic terminal dashboard."""
        mode_color = COLOR_CYAN if self.state == DemoState.BASE_DRIVE else COLOR_MAGENTA
        if self.state == DemoState.WAITING_FOR_LEADER:
            mode_color = COLOR_YELLOW
        elif self.state == DemoState.SYNCING or self.state == DemoState.PARKING:
            mode_color = COLOR_YELLOW

        state_tag = f"{mode_color}[{self.state.value:^18}]{COLOR_RESET}"
        conn_tag = f"{COLOR_GREEN}GP:OK{COLOR_RESET}" if gp.connected else f"{COLOR_RED}GP:DISC{COLOR_RESET}"
        leader_tag = f"{COLOR_GREEN}LDR:OK{COLOR_RESET}" if self.arm.is_leader_connected() else f"{COLOR_YELLOW}LDR:--{COLOR_RESET}"

        if self.state == DemoState.BASE_DRIVE:
            info = f"Cmd: L={int(self.base.cur_left_rpm):+4d} R={int(self.base.cur_right_rpm):+4d} RPM | Gear: {gp.speed_rpm:.0f} RPM"
        elif self.state == DemoState.ARM_TELEOP_LEADER:
            f_pose = self.arm.get_follower_pose()
            pan = f_pose.get("shoulder_pan.pos", 0.0)
            lift = f_pose.get("shoulder_lift.pos", 0.0)
            grip = f_pose.get("gripper.pos", 0.0)
            info = f"Leader Mirror | Pan:{pan:+5.1f}° Lift:{lift:+5.1f}° Grip:{grip:4.1f}% | Base: LOCKED"
        elif self.state == DemoState.ARM_TELEOP_GAMEPAD:
            f_pose = self.arm.get_follower_pose()
            pan = f_pose.get("shoulder_pan.pos", 0.0)
            lift = f_pose.get("shoulder_lift.pos", 0.0)
            elbow = f_pose.get("elbow_flex.pos", 0.0)
            grip = f_pose.get("gripper.pos", 0.0)
            info = f"GP Arm | Pan:{pan:+4.0f}° Lift:{lift:+4.0f}° Elb:{elbow:+4.0f}° Grp:{grip:3.0f}% | Base: LOCKED"
        elif self.state == DemoState.WAITING_FOR_LEADER:
            info = f"{COLOR_YELLOW}Waiting for Leader Arm... Press [Y] to cancel{COLOR_RESET}"
        elif self.state == DemoState.SYNCING:
            info = f"{COLOR_CYAN}Syncing arms smoothly to avoid jerk...{COLOR_RESET}"
        elif self.state == DemoState.PARKING:
            info = f"{COLOR_CYAN}Parking arm into stow pose...{COLOR_RESET}"
        else:
            info = ""

        sys.stdout.write(f"\r{state_tag} {conn_tag} {leader_tag} | {info:<55}  ")
        sys.stdout.flush()

    def stop(self):
        """Safely bring everything to a stop."""
        self.running = False
        print("\n\nShutting down demonstration...")
        try:
            self.arm.move_to_stow(duration_s=1.0)
        except Exception:
            pass
        self.arm.disconnect()
        self.base.stop()
        self.gamepad.close()
        print("Demo shut down cleanly.")
