# MIDI Mobile Manipulator Demonstration Suite

Integrated demonstration suite for the **MIDI Mobile Manipulator**, seamlessly combining the **DDSM direct-drive mobile base** and the **SO-100 robotic arm** for interactive live demonstrations.

---

## 🎯 Demonstration Cases

The system features three selectable demonstration workflows accessible via an interactive menu or command-line flags:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   DEMONSTRATION CASE SELECTION MENU                     │
├────────────────────────────────────────────────────────────────────────┤
│ [1] Case 1: Base Drive + SO-100 Leader Arm Teleoperation (Local on Pi) │
│     • Drive mobile base with Logitech F710 Gamepad                     │
│     • Press [Y] -> Lock wheels with electric brakes                    │
│     • 1:1 real-time mirror: Follower mirrors physical Leader Arm       │
│                                                                        │
│ [2] Case 2: Base Drive + Logitech Gamepad Arm Teleoperation (All-in-1) │
│     • Drive mobile base with Logitech F710 Gamepad                     │
│     • Press [Y] -> Lock wheels with electric brakes                    │
│     • 6-DOF Rate control of SO-100 arm entirely from Logitech Gamepad  │
│                                                                        │
│ [3] Case 3: Remote Network Teleoperation (Leader Arm on PC -> Robot)   │
│     • SO-100 Leader Arm is plugged into your PC                        │
│     • PC communicates with Raspberry Pi over low-latency TCP network   │
│     • Press [SPACE] on computer to switch between Base Drive and Arm!  │
│       - In Base Drive: Tilt/turn leader arm like a joystick to drive   │
│       - In Arm Teleop: Wheels locked; 1:1 real-time arm mirroring      │
└────────────────────────────────────────────────────────────────────────┘
```

---

### Workflow Breakdown

```
                   ┌──────────────────────────────────────────────┐
                   │             Mode 1: BASE DRIVE               │
                   │  • Logitech F710 Gamepad (Cases 1 & 2)       │
                   │  • OR Leader Arm Joystick (Case 3 from PC)   │
                   │  • SO-100 Follower held in safe STOW pose    │
                   └──────────────────────┬───────────────────────┘
                                          │
                         Press Button     │    [Y] on Gamepad (Cases 1, 2)
                                          │    OR [SPACE] on PC (Case 3)
                                          ▼
                   ┌──────────────────────────────────────────────┐
                   │               THE MODE SWITCH                │
                   │  • DDSM Base locks active electric brakes    │
                   │  • Wheels held at strictly 0 RPM             │
                   └──────┬───────────────┼───────────────┬───────┘
                          │               │               │
              [Case 1]    │   [Case 2]    │   [Case 3]    │
                          ▼               ▼               ▼
         ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
         │ LEADER ARM SYNC  │  │ GAMEPAD RATE CTL │  │ PC NETWORK ARM   │
         │ • 1:1 Mirroring  │  │ • 6-DOF sticks   │  │ • 1:1 Mirroring  │
         │   physical arm   │  │ • Presets & grip │  │   over TCP stream│
         └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
                  │                     │                     │
                  └─────────────────────┼─────────────────────┘
                                        │
                         Press Button   │    [Y] on Gamepad
                                        │    OR [SPACE] on PC
                                        ▼
                   ┌──────────────────────────────────────────────┐
                   │                State: PARKING                │
                   │  • Follower smoothly parks to STOW pose      │
                   │  • Electric brakes released                  │
                   │  • Return to Mode 1: BASE DRIVE              │
                   └──────────────────────────────────────────────┘
```

---

## 🎮 Gamepad Controls

### Mode 1: Base Drive Mode (Common to Case 1 & Case 2)

| Control Input | Function | Description |
| :--- | :--- | :--- |
| **Button Y** (Top) | **MODE SWITCH** | **Toggle between Base Drive and Arm Teleoperation** |
| **Left Stick (Vertical)** | **Base Forward / Reverse** | Push forward to drive forward, pull back to reverse |
| **Right Stick (Horizontal)** | **Base Steering** | Push left to steer CCW/Left, push right to steer CW/Right |
| **Left Bumper (LB)** | **Speed UP (+)** | Step up gear: `50` → `100` → `150` → `200` RPM |
| **Right Bumper (RB)** | **Speed DOWN (-)** | Step down gear: `200` → `150` → `100` → `50` RPM |
| **Button B** | **Brake / E-Stop** | Hold to engage active electric brake |
| **START** | **Exit** | Safely parks arm, halts base, and exits program |

---

### Mode 2: Arm Teleoperation Controls

#### Case 1: Leader Arm Teleoperation
* Base is locked with electric brakes.
* Operator or audience holds the physical SO-100 Leader Arm.
* The Follower arm tracks the Leader arm 1:1 in real time.
* Press **[Y]** anytime to park the arm and resume base driving.

#### Case 2: Logitech Gamepad Arm Teleoperation

| Gamepad Input | Joint / Function | Action |
| :--- | :--- | :--- |
| **Right Stick (Horizontal)** | **Shoulder Pan** | Push Left = Pan Left (-), Push Right = Pan Right (+) |
| **Right Stick (Vertical)** | **Shoulder Lift** | Push Up = Lift Up (+), Pull Down = Lower Down (-) |
| **Left Stick (Vertical)** | **Elbow Flex** | Push Up = Extend Elbow (+), Pull Down = Flex In (-) |
| **Left Stick (Horizontal)** | **Wrist Roll** | Push Left = Roll CCW (-), Push Right = Roll CW (+) |
| **D-Pad (Up / Down)** | **Wrist Flex (Pitch)** | D-Pad Up = Pitch Up (+), D-Pad Down = Pitch Down (-) |
| **D-Pad (Left / Right)** | **Fine Wrist Roll** | Fine angle adjustment |
| **Right Bumper (RB)** | **Close Gripper** | Hold RB to continuously close gripper clamp |
| **Left Bumper (LB)** | **Open Gripper** | Hold LB to continuously open gripper clamp |
| **Button A** | **Gripper Toggle** | Quick toggle between Open (`85%`) and Fully Closed (`0%`) |
| **Button X** | **Ready Pose Preset** | Smoothly moves arm to optimal ready/working pose |
| **Button Y** | **Switch to Drive** | Parks arm to stow pose and unlocks mobile base |
| **Button B** | **Emergency Halt** | Freezes arm target in place |

---

#### Case 3: Remote Network Teleoperation (Leader Arm on PC -> Robot)

In Case 3, the **SO-100 Leader Arm** is connected via USB directly to **your PC**, while the mobile robot runs the TCP Robot Server on the Raspberry Pi (`144.122.83.216:8765`).

##### Computer Keyboard Controls:
| Computer Key | Function | Description |
| :--- | :--- | :--- |
| **[SPACE]** or **[M]** | **MODE SWITCH** | **Toggle between Base Drive and Follower Arm Teleoperation** |
| **[C]** | **Center Neutral** | Calibrate current leader arm position as the neutral center for Base driving |
| **[1] - [4]** | **Speed Gear** | Select Base driving gear (`50` / `100` / `150` / `200` RPM) |
| **[B]** | **Brake Toggle** | Toggle active electric brake lock on base |
| **[Q]** or **[ESC]** | **Quit & Park** | Smoothly parks follower arm to stow pose and exits cleanly |

##### Leader Arm Motion Mapping:
* **In Base Drive Mode:**
  * **Turn Base:** Turn leader arm's `shoulder_pan` left (CCW) or right (CW).
  * **Drive Base:** Tilt leader arm's `shoulder_lift` forward (drive forward) or pull back (reverse).
  * **Squeeze Brake:** Squeezing the leader gripper past 80% engages electric brakes.
  * *Follower arm on the robot stays safely stowed in rest pose.*
* **In Arm Teleoperation Mode:**
  * **Base Wheels:** Firmly locked at 0 RPM with active electric brakes.
  * **Follower Arm:** Tracks the leader arm 1:1 in real time with zero-jerk smoothing.

---

## 🔌 Hardware Architecture & Ports

| Subsystem | Device | Persistent by-id Port | Fallback Port |
| :--- | :--- | :--- | :--- |
| **Mobile Base** | Dual DDSM Hub Motors (RS485) | `/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_BH002HMM-if00-port0` | `/dev/ttyUSB0` |
| **Follower Arm** | SO-100 Follower (6x STS3215) | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6084008-if00` | `/dev/ttyACM0` |
| **Leader Arm** | SO-100 Leader (6x STS3215) | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6084621-if00` | `/dev/ttyACM1` |
| **Controller** | Logitech F710 Wireless Gamepad | `/dev/input/by-id/usb-Logitech_Wireless_Gamepad_F710_DE659D52-event-joystick` | `/dev/input/js0` |

---

## 🚀 How to Run the Demonstrations

### Running Case 1 or Case 2 (Robot-side with Gamepad)

1. **SSH into the Raspberry Pi:**
   ```bash
   ssh midi@144.122.83.216
   ```
   *(Password: `bekoromer`)*

2. **Launch the Demo:**
   ```bash
   cd ~/demo_midi
   ./run_demo.sh         # Interactive menu
   # OR:
   ./run_demo.sh 1       # Case 1 (Base Drive + Leader Arm Teleop on Pi)
   ./run_demo.sh 2       # Case 2 (Base Drive + Gamepad Arm Teleop)
   ```

---

### Running Case 3 (Remote Teleoperation: Leader Arm on PC -> Robot)

1. **Plug the SO-100 Leader Arm into your PC** via USB.
2. **Start the Robot Server on the Raspberry Pi:**
   ```bash
   ssh midi@144.122.83.216
   cd ~/demo_midi
   ./run_demo.sh 3
   ```
   *(The Raspberry Pi homes the follower arm to stow pose and listens on port 8765).*

3. **Start the PC Teleoperation Client on your computer:**
   ```bash
   cd ~/demo_midi
   ./run_pc_client.sh    # Automatically connects to 144.122.83.216:8765
   ```
4. **Control:**
   * Tilt and turn the leader arm to drive the robot around!
   * Press **[SPACE]** on your computer keyboard to switch into Arm Teleoperation and play with the arm.
   * Press **[SPACE]** again to park the arm and resume driving.

*(Testing & Simulation Options)*:
* `./run_demo.sh --dry-run 3` : Run Robot Server in dry-run simulation mode.
* `./run_pc_client.sh 127.0.0.1 --dry-run` : Run PC Client in simulation mode.
* `python3 tests/test_demo.py` : Run automated test suite verifying all 3 cases.

---

## ⚙️ Configuration (`config.yaml`)

Edit [`config.yaml`](file:///home/maksimgorki/demo_midi/config.yaml) to tune speeds, smooth trajectory timings, or change default poses:

```yaml
# Set default case (1, 2, or null for interactive menu):
case: null

# Gamepad Arm Teleoperation (Case 2) Speeds:
arm_gamepad_teleop:
  pan_speed_dps: 50.0           # Shoulder pan speed (deg/s)
  lift_speed_dps: 45.0          # Shoulder lift speed (deg/s)
  elbow_speed_dps: 50.0         # Elbow flex speed (deg/s)
  flex_speed_dps: 45.0          # Wrist flex speed (deg/s)
  roll_speed_dps: 60.0          # Wrist roll speed (deg/s)
  gripper_speed_pct_s: 60.0     # Gripper speed (%/s)

arm:
  interpolation_time_s: 1.5     # S-curve transition time to prevent sudden jerks
  stow_pose:
    shoulder_pan.pos: 0.0
    shoulder_lift.pos: -95.0
    elbow_flex.pos: 25.0
    wrist_flex.pos: 95.0
    wrist_roll.pos: 0.0
    gripper.pos: 5.0
```

---

## 🛡️ Safety Handshakes

* **Collision & Jerk Guard:** Transitions between stow pose and working/teleoperation poses always use an S-curve cosine trajectory to eliminate sudden mechanical jerk.
* **Firm Electric Brake Lock:** In both teleoperation modes (Leader arm or Gamepad rate control), the DDSM base continuously commands 0 RPM with active electric braking every control cycle, ensuring the mobile base remains immovable while the arm is active.
* **Graceful Exit:** Pressing `Ctrl+C` or the gamepad `START` button parks the arm into stow pose, stops the motors, and closes all serial buses cleanly.
