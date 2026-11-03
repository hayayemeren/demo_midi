#!/usr/bin/env bash
# ==============================================================================
# MIDI Case 3: PC Client Launcher (Remote Leader Arm Controller)
# Connect SO-100 Leader Arm to this PC, and connect to Raspberry Pi
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Robot IP: can be passed as argument or default to 144.122.83.216
ROBOT_IP="${1:-144.122.83.216}"

# Ensure permissions on USB serial ports if connected locally
sudo chmod 666 /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || true

echo "Starting MIDI Case 3 PC Teleoperation Client..."
echo "Target Robot: $ROBOT_IP:8765"

python3 pc_client.py --robot "$ROBOT_IP" "${@:2}"
