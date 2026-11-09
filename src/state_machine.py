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
from typing import Optional, Dict, Any

from src.arm_controller import ArmController
from src.base_controller import BaseController
from src.gamepad import F710Gamepad, GamepadState
from src.network import RobotServer
from src.vision_grasp import WristCamera, SockDetector, VisionGraspPlanner, DetectionResult

logger = logging.getLogger("demo.statemachine")


class DemoState(enum.Enum):
    BASE_DRIVE = "BASE_DRIVE"
    WAITING_FOR_LEADER = "WAITING_FOR_LEADER"
    SYNCING = "SYNCING"
    ARM_TELEOP_LEADER = "ARM_TELEOP_LEADER"
    ARM_TELEOP_GAMEPAD = "ARM_TELEOP_GAMEPAD"
    NETWORK_SERVER = "NETWORK_SERVER"
    AUTONOMOUS_PICK_SCAN = "AUTONOMOUS_PICK_SCAN"
    AUTONOMOUS_PICK_ALIGN = "AUTONOMOUS_PICK_ALIGN"
    AUTONOMOUS_PICK_GRASP = "AUTONOMOUS_PICK_GRASP"
    AUTONOMOUS_PICK_VERIFY = "AUTONOMOUS_PICK_VERIFY"
    AUTONOMOUS_BASKET_DUMP = "AUTONOMOUS_BASKET_DUMP"
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
        network_port: int = 8765,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.base = base
        self.arm = arm
        self.gamepad = gamepad
        self.demo_case = demo_case
        self.loop_hz = loop_hz
        self.telemetry_interval = telemetry_interval
        self.network_port = network_port
        self.config = config or getattr(base, "config", {}) or {}

        self.state = DemoState.NETWORK_SERVER if demo_case == 3 else DemoState.BASE_DRIVE
        self.running = False
        self.last_telemetry_time = 0.0
        self.last_leader_check_time = 0.0
        self.transition_start_time = 0.0

        # Case 3 Network Server
        self.server: Optional[RobotServer] = None
        self._net_mode = "base"
        self._net_arm_synced = False
        if demo_case == 3:
            self.server = RobotServer(host="0.0.0.0", port=network_port)

        # Case 4 Autonomous Vision & Grasp Planner
        self.camera: Optional[WristCamera] = None
        self.detector: Optional[SockDetector] = None
        self.planner: Optional[VisionGraspPlanner] = None
        self.last_detection: Optional[DetectionResult] = None
        self.scan_retries = 0
        self.max_scan_retries = 3
        self.align_iterations = 0
        self.max_align_iterations = 2

        if demo_case == 4:
            c4 = self.config.get("case4", {})
            cam_cfg = c4.get("camera", {})
            self.camera = WristCamera(
                device_path=cam_cfg.get("device", "/dev/video0"),
                fallback_index=int(cam_cfg.get("fallback", 0)),
                width=int(cam_cfg.get("width", 640)),
                height=int(cam_cfg.get("height", 480)),
                fps=int(cam_cfg.get("fps", 30)),
                dry_run=self.base.dry_run,
            )
            det_cfg = c4.get("detection", {})
            self.detector = SockDetector(
                mode=det_cfg.get("mode", "auto"),
                min_area=float(det_cfg.get("min_area", 1000.0)),
                max_area=float(det_cfg.get("max_area", 80000.0)),
                contrast_threshold=float(det_cfg.get("contrast_threshold", 38.0)),
                border_margin=int(det_cfg.get("border_margin", 12)),
                save_debug_image=bool(det_cfg.get("save_debug_image", True)),
                debug_image_path=det_cfg.get("debug_image_path", "/tmp/sock_detection.jpg"),
                debug_raw_path=det_cfg.get("debug_raw_path", "/tmp/sock_raw.jpg"),
                debug_mask_path=det_cfg.get("debug_mask_path", "/tmp/sock_mask.jpg"),
                debug_combined_path=det_cfg.get("debug_combined_path", "/tmp/sock_combined.jpg"),
                enable_web_stream=bool(det_cfg.get("enable_web_stream", True)),
                web_stream_port=int(det_cfg.get("web_stream_port", 8080)),
            )
            self.planner = VisionGraspPlanner(self.config)

    def start(self):
        """Initial check and launch of the control loop."""
        self.running = True
        logger.info("Initializing Demonstration State Machine for Case %d...", self.demo_case)

        # 1. Start Follower at stow pose
        logger.info("Setting Follower Arm to safe Stow Pose...")
        self.arm.move_to_stow(duration_s=1.0)

        # 2. Case-specific initialization
        if self.demo_case == 1 and self.arm.is_leader_port_available():
            self.arm.connect_leader()
        elif self.demo_case == 3 and self.server is not None:
            self.server.start()
        elif self.demo_case == 4 and self.camera is not None:
            self.camera.connect()

        if self.demo_case == 1:
            case_desc = f"Case 1: {COLOR_MAGENTA}SO-100 LEADER ARM TELEOPERATION (Pi Local){COLOR_RESET}"
        elif self.demo_case == 2:
            case_desc = f"Case 2: {COLOR_MAGENTA}LOGITECH GAMEPAD ARM TELEOPERATION (All-in-One){COLOR_RESET}"
        elif self.demo_case == 3:
            case_desc = f"Case 3: {COLOR_MAGENTA}REMOTE LEADER ARM TELEOPERATION (PC Client -> Pi Server){COLOR_RESET}"
        else:
            case_desc = f"Case 4: {COLOR_MAGENTA}VISION-GUIDED AUTONOMOUS SOCK COLLECTION & BASKET TIDY{COLOR_RESET}"

        logger.info(
            f"{COLOR_BOLD}{COLOR_GREEN}=== MIDI DEMO READY (Case {self.demo_case}) ==={COLOR_RESET}\n"
            f"Active: {case_desc}\n"
            f"Exit: {COLOR_RED}[Ctrl+C] or Gamepad [START]{COLOR_RESET}\n"
        )

        self._run_loop()

    def _run_loop(self):
        period = 1.0 / self.loop_hz
        t_prev = time.time()

        while self.running:
            now = time.time()
            dt = max(0.001, min(0.1, now - t_prev))
            t_prev = now

            # Poll gamepad events (if connected)
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
            elif self.state == DemoState.NETWORK_SERVER:
                self._handle_network_server(gp, dt)
            elif self.state == DemoState.AUTONOMOUS_PICK_SCAN:
                self._handle_autonomous_scan(gp)
            elif self.state == DemoState.AUTONOMOUS_PICK_ALIGN:
                self._handle_autonomous_align(gp)
            elif self.state == DemoState.AUTONOMOUS_PICK_GRASP:
                self._handle_autonomous_grasp(gp)
            elif self.state == DemoState.AUTONOMOUS_PICK_VERIFY:
                self._handle_autonomous_verify(gp)
            elif self.state == DemoState.AUTONOMOUS_BASKET_DUMP:
                self._handle_autonomous_basket(gp)
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
            elif self.demo_case == 2:
                logger.info("\n[SWITCH] Button [Y] pressed -> Switching to GAMEPAD ARM TELEOPERATION (Case 2)!")
                self.arm.init_gamepad_teleop()
                logger.info(
                    f"\n{COLOR_BOLD}{COLOR_MAGENTA}>>> GAMEPAD ARM TELEOPERATION ACTIVE! <<<{COLOR_RESET}\n"
                    f"Right Stick: Pan (L/R) & Lift (U/D) | Left Stick: Elbow (U/D) & Roll (L/R)\n"
                    f"D-Pad: Wrist Flex (U/D) | RB: Close Grip | LB: Open Grip | A: Toggle Grip | X: Ready Pose\n"
                    f"Press {COLOR_YELLOW}[Y]{COLOR_RESET} anytime to park the arm and return to {COLOR_CYAN}BASE DRIVE{COLOR_RESET}.\n"
                )
                self.state = DemoState.ARM_TELEOP_GAMEPAD

            # Case 4: Switch to Vision-Guided Autonomous Sock Collection
            elif self.demo_case == 4:
                logger.info(
                    f"\n{COLOR_BOLD}{COLOR_MAGENTA}>>> [CASE 4] INITIATING AUTONOMOUS SOCK COLLECTION! <<<{COLOR_RESET}\n"
                    f"Base wheels locked with electric brakes. Starting vision inspection...\n"
                )
                self.scan_retries = 0
                self.align_iterations = 0
                self.state = DemoState.AUTONOMOUS_PICK_SCAN
            return

        # Case 4 shortcut: Button X triggers manual direct basket dump macro
        if self.demo_case == 4 and gp.button_x_pressed:
            self.base.lock_brakes()
            logger.info("\n[CASE 4] Button [X] pressed -> Initiating Direct Basket Deposit macro!")
            self.state = DemoState.AUTONOMOUS_BASKET_DUMP
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

    def _handle_network_server(self, gp: GamepadState, dt: float):
        """Mode 3 (Case 3): Remote PC Client controls Base and Arm over TCP network."""
        if self.server is None:
            return

        cmd, watchdog_trip = self.server.poll_commands()

        # 1. If no client connected, actively hold base lock
        if not self.server.is_connected:
            self.base.hold_lock()
            return

        # 2. Watchdog timeout: silence/dropped packet -> emergency stop
        if watchdog_trip:
            self.base.drive(0.0, 0.0, 0.0, dt, brake=True)
            return

        if not cmd:
            self.base.hold_lock()
            return

        mode = cmd.get("mode", "base")
        self._net_mode = mode

        # Handle explicit park request
        if cmd.get("park", False):
            self.base.hold_lock()
            self.arm.move_to_stow(duration_s=1.0, feedback_cb=lambda _: self.base.hold_lock())
            self._net_arm_synced = False
            return

        if mode == "base":
            # Mobile base driving mode: release brakes if locked, wheels active, arm stowed
            if self.base.is_locked:
                self.base.release_brakes()
            self._net_arm_synced = False
            base_cmd = cmd.get("base", {})
            fwd = float(base_cmd.get("forward", 0.0))
            turn = float(base_cmd.get("turn", 0.0))
            brake = bool(base_cmd.get("brake", False))
            speed = float(base_cmd.get("speed_rpm", 100.0))
            self.base.drive(fwd, turn, speed, dt, brake=brake)

        elif mode == "arm":
            # Arm teleoperation mode: base strictly locked with electric brakes
            self.base.hold_lock()
            arm_target = cmd.get("arm", {})

            if arm_target:
                # If first frame of arm mode, smoothly sync to target pose (zero-jerk transition)
                if not self._net_arm_synced:
                    logger.info("\n[NETWORK] First arm frame received: syncing follower to leader pose...")
                    cur_pose = self.arm.get_follower_pose()
                    self.arm.smooth_trajectory(cur_pose, arm_target, duration_s=1.0, feedback_cb=lambda _: self.base.hold_lock())
                    self._net_arm_synced = True
                else:
                    self.arm.send_follower_action(arm_target)

        # Send live telemetry frame to PC Client
        if self.server.is_connected:
            telemetry = {
                "type": "telemetry",
                "state": "NETWORK_SERVER",
                "mode": mode,
                "base_locked": self.base.is_locked,
                "left_rpm": round(self.base.cur_left_rpm, 1),
                "right_rpm": round(self.base.cur_right_rpm, 1),
                "follower_pose": self.arm.get_follower_pose(),
            }
            self.server.send_telemetry(telemetry)

    def _handle_autonomous_scan(self, gp: GamepadState):
        """Case 4: Locks base, sweeps wrist camera across floor (center, left, right), detects sock."""
        self.base.hold_lock()

        # Allow user to abort with B or switch button
        if gp.brake or gp.button_b_held or self.gamepad.is_switch_pressed(gp):
            logger.info("\n[ABORT] Aborting autonomous scan -> Returning to BASE DRIVE.")
            self.state = DemoState.PARKING
            return

        if self.planner is None or self.detector is None or self.camera is None:
            logger.error("Case 4 components not initialized!")
            self.state = DemoState.PARKING
            return

        sweep_angles = getattr(self.planner, "scan_sweep_angles", [0.0, -28.0, 28.0])
        logger.info(
            f"\n{COLOR_BOLD}{COLOR_MAGENTA}>>> [CASE 4] EXECUTING FULL FLOOR SCAN SWEEP (Angles: {sweep_angles}) <<<{COLOR_RESET}"
        )

        detected_sock = None
        for angle in sweep_angles:
            if (not self.running and not self.arm.dry_run) or gp.brake or gp.button_b_held:
                break

            logger.info(f"[SCAN] Sweeping camera to pan angle: {angle:+.1f}°...")
            scan_target = dict(self.planner.scan_pose)
            scan_target["shoulder_pan.pos"] = float(angle)
            self.arm.move_to_pose(
                scan_target,
                duration_s=0.9,
                feedback_cb=lambda _: self.base.hold_lock(),
            )
            time.sleep(0.20)
            self.base.hold_lock()

            ok, frame = self.camera.capture_frame()
            if not ok or frame is None:
                continue

            det = self.detector.detect(frame)
            if det.found:
                detected_sock = det
                logger.info(
                    f"\n{COLOR_BOLD}{COLOR_GREEN}>>> [VISION] SOCK DETECTED AT {angle:+.1f}° SWEEP! <<<{COLOR_RESET}\n"
                    f"Position: ({det.cx:.0f}, {det.cy:.0f}) | Offset: eX={det.error_x:+.2f}, eY={det.error_y:+.2f} | Area: {det.area:.0f}px\n"
                    f"Debug image saved to: {self.detector.debug_image_path}\n"
                )
                break
            else:
                logger.info(f"[SCAN] No sock at {angle:+.1f}°. Continuing sweep...")

        if detected_sock is not None:
            self.last_detection = detected_sock
            self.align_iterations = 0
            self.state = DemoState.AUTONOMOUS_PICK_ALIGN
        else:
            logger.warning(
                f"\n{COLOR_BOLD}{COLOR_YELLOW}>>> [SCAN COMPLETE] NO SOCK DETECTED ON FLOOR! <<<{COLOR_RESET}\n"
                f"Halting autonomous sequence. Returning to BASE DRIVE mode.\n"
            )
            self.state = DemoState.PARKING

    def _handle_autonomous_align(self, gp: GamepadState):
        """Case 4: Adjusts pan and reach to center the sock in the gripper's line of approach."""
        self.base.hold_lock()

        if gp.brake or gp.button_b_held or self.gamepad.is_switch_pressed(gp):
            logger.info("\n[ABORT] Aborting autonomous align -> Returning to BASE DRIVE.")
            self.state = DemoState.PARKING
            return

        if self.planner is None or self.last_detection is None:
            self.state = DemoState.PARKING
            return

        cur_pose = self.arm.get_follower_pose()
        aligned_pose, is_aligned = self.planner.compute_aligned_pose(cur_pose, self.last_detection)

        logger.info(
            f"[ALIGN] Servoing arm to center sock: pan={aligned_pose.get('shoulder_pan.pos', 0.0):.1f}°, "
            f"lift={aligned_pose.get('shoulder_lift.pos', 0.0):.1f}°..."
        )
        self.arm.move_to_pose(
            aligned_pose,
            duration_s=0.8,
            feedback_cb=lambda _: self.base.hold_lock(),
        )

        self.align_iterations += 1

        # Check alignment with fresh frame if iterations remain
        if self.camera is not None and self.detector is not None and not is_aligned and self.align_iterations < self.max_align_iterations:
            time.sleep(0.1)
            ok, frame = self.camera.capture_frame()
            if ok and frame is not None:
                det = self.detector.detect(frame)
                if det.found:
                    self.last_detection = det

        logger.info(
            f"\n{COLOR_BOLD}{COLOR_CYAN}>>> [ALIGN] Sock aligned with gripper center! Initiating Grasp... <<<{COLOR_RESET}\n"
        )
        self.state = DemoState.AUTONOMOUS_PICK_GRASP

    def _handle_autonomous_grasp(self, gp: GamepadState):
        """Case 4: Opens gripper, descends to floor height, clamps the sock, and elevates off floor."""
        self.base.hold_lock()

        if gp.brake or gp.button_b_held:
            logger.info("\n[ABORT] Aborting autonomous grasp.")
            self.state = DemoState.PARKING
            return

        if self.planner is None:
            self.state = DemoState.PARKING
            return

        logger.info("\n[GRASP] 1. Opening gripper wide (85%)...")
        cur_pose = self.arm.get_follower_pose()
        open_pose = dict(cur_pose)
        open_pose["gripper.pos"] = 85.0
        self.arm.move_to_pose(open_pose, duration_s=0.6, feedback_cb=lambda _: self.base.hold_lock())

        logger.info("[GRASP] 2. Lowering arm down to floor grasp pose...")
        grasp_pose = self.planner.compute_grasp_pose(open_pose)
        grasp_descend = dict(grasp_pose)
        grasp_descend["gripper.pos"] = 85.0
        self.arm.move_to_pose(grasp_descend, duration_s=1.0, feedback_cb=lambda _: self.base.hold_lock())

        logger.info("[GRASP] 3. Clamping gripper jaws onto sock (0%)...")
        grasp_clamp = dict(grasp_descend)
        grasp_clamp["gripper.pos"] = 0.0
        self.arm.move_to_pose(grasp_clamp, duration_s=0.8, feedback_cb=lambda _: self.base.hold_lock())
        time.sleep(0.3)
        self.base.hold_lock()

        logger.info("[GRASP] 4. Elevating arm off floor to verify grasp...")
        lift_verify_pose = dict(grasp_clamp)
        lift_verify_pose["shoulder_lift.pos"] -= 15.0  # elevate shoulder
        lift_verify_pose["elbow_flex.pos"] -= 10.0     # pull up elbow
        self.arm.move_to_pose(lift_verify_pose, duration_s=0.6, feedback_cb=lambda _: self.base.hold_lock())
        time.sleep(0.3)
        self.base.hold_lock()

        self.state = DemoState.AUTONOMOUS_PICK_VERIFY

    def _handle_autonomous_verify(self, gp: GamepadState):
        """Case 4 (Suggestion 2): Checks gripper position while elevated to verify that cloth was caught."""
        self.base.hold_lock()

        cur_pose = self.arm.get_follower_pose()
        measured_grip = cur_pose.get("gripper.pos", 0.0)
        if self.arm.dry_run:
            measured_grip = 25.0

        is_caught = self.planner.verify_grasp(measured_grip) if self.planner else True
        abort_on_miss = getattr(self.planner, "abort_on_miss", False)

        if is_caught:
            logger.info(
                f"\n{COLOR_BOLD}{COLOR_GREEN}>>> [GRASP VERIFIED] SUCCESS! Sock securely held (Grip: {measured_grip:.1f}% >= {self.planner.grasp_verification_threshold:.1f}%). <<<{COLOR_RESET}\n"
                f"Transporting sock to basket with clamped jaws..."
            )
            self.state = DemoState.AUTONOMOUS_BASKET_DUMP
        elif not abort_on_miss:
            # Proceed to basket even if thin cloth grip reading is close to threshold, NEVER drop before going up!
            logger.info(
                f"\n{COLOR_BOLD}{COLOR_YELLOW}>>> [GRASP INFO] Grip reading: {measured_grip:.1f}% (Threshold: {self.planner.grasp_verification_threshold:.1f}%). Proceeding to basket with clamped jaws... <<<{COLOR_RESET}\n"
            )
            self.state = DemoState.AUTONOMOUS_BASKET_DUMP
        else:
            logger.warning(
                f"\n{COLOR_BOLD}{COLOR_YELLOW}>>> [GRASP MISSED] Gripper is empty (Grip: {measured_grip:.1f}% < {self.planner.grasp_verification_threshold:.1f}%). Aborting as requested. <<<{COLOR_RESET}\n"
                f"Returning to driving."
            )
            cur_pose["gripper.pos"] = 40.0
            self.arm.move_to_pose(cur_pose, duration_s=0.5, feedback_cb=lambda _: self.base.hold_lock())
            self.state = DemoState.PARKING

    def _handle_autonomous_basket(self, gp: GamepadState):
        """Case 4: Smoothly moves arm over chassis-mounted basket (behind and a little up), drops sock, and parks."""
        self.base.hold_lock()

        if self.planner is None:
            self.state = DemoState.PARKING
            return

        logger.info("\n[BASKET] 1. Moving arm over collection basket (behind gripper, elevated)...")
        transit_pose = dict(self.planner.basket_pose)
        transit_pose["gripper.pos"] = 0.0  # Hold sock firmly during transit
        self.arm.move_to_pose(
            transit_pose,
            duration_s=1.2,
            feedback_cb=lambda _: self.base.hold_lock(),
        )

        logger.info("[BASKET] 2. Releasing sock into basket (opening gripper to 85%)...")
        drop_pose = dict(transit_pose)
        drop_pose["gripper.pos"] = 85.0
        self.arm.move_to_pose(drop_pose, duration_s=0.5, feedback_cb=lambda _: self.base.hold_lock())
        time.sleep(0.5)
        self.base.hold_lock()

        logger.info(
            f"\n{COLOR_BOLD}{COLOR_GREEN}>>> [BASKET COMPLETE] Sock deposited successfully into basket! <<<{COLOR_RESET}\n"
        )
        self.state = DemoState.PARKING

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
        if self.state in (DemoState.WAITING_FOR_LEADER, DemoState.SYNCING, DemoState.PARKING):
            mode_color = COLOR_YELLOW
        elif self.state == DemoState.NETWORK_SERVER:
            mode_color = COLOR_CYAN if self._net_mode == "base" else COLOR_MAGENTA
        elif self.state in (
            DemoState.AUTONOMOUS_PICK_SCAN,
            DemoState.AUTONOMOUS_PICK_ALIGN,
            DemoState.AUTONOMOUS_PICK_GRASP,
            DemoState.AUTONOMOUS_PICK_VERIFY,
            DemoState.AUTONOMOUS_BASKET_DUMP,
        ):
            mode_color = COLOR_GREEN

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
        elif self.state == DemoState.NETWORK_SERVER:
            pc_ok = self.server and self.server.is_connected
            conn_tag = f"{COLOR_GREEN}PC:OK{COLOR_RESET}" if pc_ok else f"{COLOR_YELLOW}PC:WAIT{COLOR_RESET}"
            if self._net_mode == "base":
                info = f"Net Base | Cmd: L={int(self.base.cur_left_rpm):+4d} R={int(self.base.cur_right_rpm):+4d} RPM"
            else:
                f_pose = self.arm.get_follower_pose()
                pan = f_pose.get("shoulder_pan.pos", 0.0)
                lift = f_pose.get("shoulder_lift.pos", 0.0)
                info = f"Net Arm  | Pan:{pan:+5.1f}° Lift:{lift:+5.1f}° | Base: LOCKED"
        elif self.state in (
            DemoState.AUTONOMOUS_PICK_SCAN,
            DemoState.AUTONOMOUS_PICK_ALIGN,
            DemoState.AUTONOMOUS_PICK_GRASP,
            DemoState.AUTONOMOUS_PICK_VERIFY,
            DemoState.AUTONOMOUS_BASKET_DUMP,
        ):
            f_pose = self.arm.get_follower_pose()
            grip = f_pose.get("gripper.pos", 0.0)
            det_str = f"SOCK@({self.last_detection.cx:.0f},{self.last_detection.cy:.0f})" if (self.last_detection and self.last_detection.found) else "SEARCHING"
            info = f"Auton: {det_str} | Grip:{grip:3.0f}% | Base: LOCKED"
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
        if self.server:
            try:
                self.server.close()
            except Exception:
                pass
        if self.camera:
            try:
                self.camera.release()
            except Exception:
                pass
        try:
            self.arm.move_to_stow(duration_s=1.0)
        except Exception:
            pass
        self.arm.disconnect()
        self.base.stop()
        self.gamepad.close()
        print("Demo shut down cleanly.")
