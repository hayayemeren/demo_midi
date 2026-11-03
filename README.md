# MIDI Mobile Manipulator Demonstration

Integrated control software for the **MIDI Mobile Manipulator**, seamlessly combining the **DDSM direct-drive mobile base** and the **SO-100 robotic arm** for interactive live demonstrations.

---

## 🎯 Demo Concept & Workflow

The demonstration features an intuitive **two-mode state machine** with a live hardware handshake:

```
                  ┌──────────────────────────────────────────────┐
                  │            Mode 1: BASE DRIVE                │
                  │  • Logitech F710 Gamepad drives wheels       │
                  │  • SO-100 Follower held in safe STOW pose    │
                  └──────────────────────┬───────────────────────┘
                                         │
                         Press Button    │    [Y] on Gamepad
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │          State: WAITING FOR LEADER           │
                  │  • DDSM Base locks electric brakes (0 RPM)   │
                  │  • Scans and detects Leader arm connection   │
                  └──────────────────────┬───────────────────────┘
                                         │
                         Leader Arm      │    Detected & Connected
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │               State: SYNCING                 │
                  │  • Smooth zero-jerk S-curve interpolation    │
                  │  • Follower aligns to Leader pose (1.5s)     │
                  └──────────────────────┬───────────────────────┘
                                         │
                         Arms Aligned    │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │          Mode 2: ARM TELEOPERATION           │
                  │  • DDSM Base stays locked & stable           │
                  │  • 1:1 real-time mirror: Leader -> Follower  │
                  │  • Audience / visitors play with the arm!    │
                  └──────────────────────┬───────────────────────┘
                                         │
                         Press Button    │    [Y] on Gamepad
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │               State: PARKING                 │
                  │  • Follower smoothly parks to STOW pose      │
                  │  • Brakes released -> Return to BASE DRIVE   │
                  └──────────────────────────────────────────────┘
```

1. **Mode 1: Base Drive Mode (`BASE_DRIVE`)**
   - The operator drives the mobile base around the demo area using the Logitech F710 gamepad sticks.
   - The SO-100 follower arm stays safely stowed in its natural rest pose (`shoulder_lift: -95°`, `elbow: 25°`, `wrist: 95°`) to prevent collisions or swinging during driving.
2. **The Switch (`WAITING_FOR_LEADER` & `SYNCING`)**
   - The operator presses **Button `Y`** on the gamepad.
   - The base immediately halts and locks its electric brakes (`DDSMDriver.emergency_brake`).
   - If the leader arm is not yet plugged in or turned on, the system displays a clear waiting message. As soon as the leader arm is detected on USB, it handshakes automatically.
   - To prevent the follower arm from violently jumping to the leader's position, the system executes a smooth **1.5-second S-curve (cosine) trajectory**, gently moving the follower arm to match the leader arm's exact pose.
3. **Mode 2: Arm Teleoperation Mode (`ARM_TELEOP`)**
   - Electric brakes keep the mobile base firmly anchored.
   - The follower arm mirrors the leader arm at 50 Hz.
   - Spectators, judges, or visitors can take the leader arm and freely play with it (pick-and-place, waving, grasping objects).
4. **Switching Back (`PARKING`)**
   - Pressing **Button `Y`** again smoothly moves the follower arm back into the stowed travel pose.
   - Electric brakes are released and base driving is restored.

---

## 🎮 Logitech F710 Gamepad Controls

| Control Input | Function | Description |
| :--- | :--- | :--- |
| **Button Y** (Top) | **MODE SWITCH** | **Toggle between Base Drive and Arm Teleoperation** |
| **Left Stick (Vertical)** | **Base Fwd / Rev** | Push forward to drive forward, pull back to reverse |
| **Right Stick (Horizontal)** | **Base Steering** | Push left to steer CCW, push right to steer CW |
| **Left Bumper (LB)** | **Speed UP (+)** | Step up gear: `50` → `100` → `150` → `200` RPM |
| **Right Bumper (RB)** | **Speed DOWN (-)** | Step down gear: `200` → `150` → `100` → `50` RPM |
| **Button B** | **Brake / E-Stop** | Hold to engage active electric brake |
| **START** | **Exit** | Safely parks arm, halts base, and exits program |

---

## 🔌 Hardware Architecture & Ports

| Subsystem | Device | Persistent by-id Port | Fallback Port |
| :--- | :--- | :--- | :--- |
| **Mobile Base** | Dual DDSM Hub Motors (RS485) | `/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_BH002HMM-if00-port0` | `/dev/ttyUSB0` |
| **Follower Arm** | SO-100 Follower (6x STS3215) | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6084008-if00` | `/dev/ttyACM0` |
| **Leader Arm** | SO-100 Leader (6x STS3215) | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6084621-if00` | `/dev/ttyACM1` |
| **Controller** | Logitech F710 Wireless Gamepad | `/dev/input/by-id/usb-Logitech_Wireless_Gamepad_F710_DE659D52-event-joystick` | `/dev/input/js0` |

---

## 🚀 How to Run the Demo on the Raspberry Pi

### 1. SSH into the Raspberry Pi
```bash
ssh midi@144.122.83.216
```
*(Password: `bekoromer`)*

### 2. Enter the Demo Directory
```bash
cd ~/demo_midi
```

### 3. Run the Demo
```bash
./run_demo.sh
```

*(Optional flags)*:
* `./run_demo.sh --dry-run` : Run in software simulation mode without physical hardware.
* `./run_demo.sh --verbose` : Enable detailed debug logging.
* `./run_demo.sh -c config.yaml` : Specify custom configuration.

---

## ⚙️ Configuration (`config.yaml`)

Edit [`config.yaml`](file:///home/maksimgorki/demo_midi/config.yaml) to tune speeds, smooth trajectory timings, or change default poses:

```yaml
gamepad:
  switch_button: "Y"            # Mode toggle button ("Y", "X", "A", "BACK")
  speed_levels: [50.0, 100.0, 150.0, 200.0]

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

* **Collision & Jerk Guard:** Switching between modes uses an S-curve cosine trajectory. Even if the leader arm is pointing in an extreme direction when switching, the follower arm smoothly glides to match it over 1.5 seconds.
* **Firm Brake Lock:** Whenever the robot is in teleoperation mode, both DDSM direct-drive motors actively hold 0 RPM with electric braking engaged, preventing the base from shifting when people bump or push the arm.
* **Graceful Signal Handling:** Pressing `Ctrl+C` or the gamepad `START` button parks the arm into stow pose, stops the motors, and closes all serial buses cleanly.
