#!/usr/bin/env bash
# ==============================================================================
# MIDI Mobile Manipulator Demo Launcher
# Automatically activates env_lerobot and runs main.py
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Activate Python Environment
if [ -f "/home/midi/env_lerobot/bin/activate" ]; then
    echo "[ENV] Activating /home/midi/env_lerobot..."
    source "/home/midi/env_lerobot/bin/activate"
elif [ -n "$VIRTUAL_ENV" ]; then
    echo "[ENV] Using active virtualenv: $VIRTUAL_ENV"
else
    echo "[ENV] Using system python3..."
fi

# 2. Stop conflicting background services & ensure device permissions
if systemctl is-active --quiet midi-robot.service 2>/dev/null; then
    echo "[SERVICE] Stopping background midi-robot.service..."
    sudo systemctl stop midi-robot.service 2>/dev/null || true
fi

sudo chmod 666 /dev/ttyUSB* /dev/ttyACM* /dev/input/event* /dev/input/js* /dev/video* 2>/dev/null || true

# 3. Parse arguments: convert positional '1'..'7' into '--case 1..7'
ARGS=()
for arg in "$@"; do
    if [ "$arg" = "1" ] || [ "$arg" = "2" ] || [ "$arg" = "3" ] || [ "$arg" = "4" ] || [ "$arg" = "5" ] || [ "$arg" = "6" ] || [ "$arg" = "7" ]; then
        ARGS+=("--case" "$arg")
    else
        ARGS+=("$arg")
    fi
done

# 4. Launch Demo Application
echo "Starting MIDI Mobile Manipulator Demo..."
python3 main.py "${ARGS[@]}"
