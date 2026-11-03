"""DDSM (Direct Drive Smart Motor) RS485 Communication Driver.

Supports DDSM115 and DDSM210 direct drive hub motors over RS485 / UART.
Implements the 10-byte frame protocol with CRC-8/MAXIM checksum.
Synchronous Request-Response transactions prevent half-duplex RS485 bus collisions.
"""

from __future__ import annotations

import logging
import os
import struct
import threading
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

logger = logging.getLogger("demo.ddsm")


def calculate_crc8(data: bytes) -> int:
    """Calculate CRC-8/MAXIM (Dallas/Maxim, poly 0x8C, init 0x00, reflected)."""
    crc = 0x00
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8C if crc & 1 else crc >> 1
    return crc & 0xFF


def set_usb_latency_1ms(port: str):
    """Set FTDI / USB-serial converter latency timer from default 16ms to 1ms on Linux."""
    yol = f"/sys/bus/usb-serial/devices/{os.path.basename(port)}/latency_timer"
    try:
        with open(yol, "w") as f:
            f.write("1")
        logger.info("USB serial latency timer set to 1 ms on %s", yol)
    except Exception:
        # Requires root or udev rule; continue gracefully if not writable
        pass


@dataclass
class MotorFeedback:
    """Telemetry data returned by a DDSM motor."""
    motor_id: int = 0
    mode: int = 0
    current_raw: int = 0
    current_amps: float = 0.0
    velocity_rpm: float = 0.0
    position_raw: int = 0
    error_code: int = 0
    sensor_fault: bool = False
    over_current: bool = False
    phase_over_current: bool = False
    stall_fault: bool = False
    over_temp: bool = False
    timestamp: float = 0.0

    @classmethod
    def from_bytes(cls, data: bytes, speed_scale: float = 1.0) -> Optional[MotorFeedback]:
        """Parse 10-byte feedback frame from DDSM motor."""
        if len(data) != 10:
            return None

        # Verify CRC
        expected_crc = calculate_crc8(data[:9])
        if data[9] != expected_crc:
            return None

        motor_id = data[0]
        mode = data[1]
        raw_current = struct.unpack(">h", data[2:4])[0]
        raw_velocity = struct.unpack(">h", data[4:6])[0]
        raw_position = struct.unpack(">h", data[6:8])[0]
        err = data[8]

        current_amps = (raw_current * 8.0) / 32767.0
        velocity_rpm = raw_velocity * speed_scale

        return cls(
            motor_id=motor_id,
            mode=mode,
            current_raw=raw_current,
            current_amps=round(current_amps, 3),
            velocity_rpm=round(velocity_rpm, 2),
            position_raw=raw_position,
            error_code=err,
            sensor_fault=bool(err & 0x01),
            over_current=bool(err & 0x02),
            phase_over_current=bool(err & 0x04),
            stall_fault=bool(err & 0x08),
            over_temp=bool(err & 0x10),
            timestamp=time.time(),
        )


class DDSMDriver:
    """RS485 Bus Driver for dual DDSM motors."""

    MODE_CURRENT = 0x01
    MODE_VELOCITY = 0x02
    MODE_POSITION = 0x03

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 115200,
        timeout: float = 0.02,
        dry_run: bool = False,
        motor_type: str = "ddsm115",
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.dry_run = dry_run
        self.motor_type = motor_type.lower()
        self.speed_scale = 0.1 if "210" in self.motor_type else 1.0

        self.serial = None
        self._lock = threading.Lock()
        self.latest_feedback: Dict[int, MotorFeedback] = {}

    def connect(self) -> bool:
        """Open serial port and configure low-latency settings."""
        if self.dry_run:
            logger.info("[DRY RUN] DDSMDriver initialized on %s (Simulation)", self.port)
            return True

        # Resolve real path if it's a by-id symlink
        real_port = os.path.realpath(self.port) if os.path.exists(self.port) else self.port

        try:
            import serial
            set_usb_latency_1ms(real_port)
            self.serial = serial.Serial(
                port=real_port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                write_timeout=self.timeout,
            )
            # Flush existing buffer garbage
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
            logger.info("Connected to DDSM RS485 on %s at %d baud", real_port, self.baudrate)
            return True
        except Exception as e:
            logger.error("Failed to open RS485 serial port %s: %s", self.port, e)
            self.serial = None
            return False

    def close(self):
        """Close serial port safely."""
        with self._lock:
            if self.serial and self.serial.is_open:
                try:
                    self.serial.close()
                except Exception:
                    pass
            self.serial = None

    def _transact(self, tx_bytes: bytes, rx_len: int = 10) -> Optional[bytes]:
        """Atomic Half-Duplex RS485 Request-Response Transaction."""
        if self.dry_run or not self.serial:
            return None

        with self._lock:
            try:
                self.serial.reset_input_buffer()
                self.serial.write(tx_bytes)
                self.serial.flush()

                # Read response frame
                rx = self.serial.read(rx_len)
                if len(rx) == rx_len:
                    return rx
                return None
            except Exception as e:
                logger.debug("RS485 transaction error: %s", e)
                return None

    def set_mode(self, motor_id: int, mode: int = MODE_VELOCITY) -> bool:
        """Switch motor operating mode."""
        pkt = bytearray(10)
        pkt[0] = motor_id & 0xFF
        pkt[1] = 0xA0
        pkt[2] = 0x00
        pkt[3] = mode & 0xFF
        pkt[9] = calculate_crc8(bytes(pkt[:9]))

        if self.dry_run:
            return True

        rx = self._transact(bytes(pkt), 10)
        return rx is not None

    def set_velocity(
        self,
        motor_id: int,
        target_rpm: float,
        acceleration: int = 1,
        brake: bool = False,
    ) -> Optional[MotorFeedback]:
        """Send target velocity and read immediate feedback frame."""
        if brake:
            raw_val = 0
            brake_byte = 0xFF
        else:
            raw_val = int(target_rpm / self.speed_scale)
            raw_val = max(-32768, min(32767, raw_val))
            brake_byte = 0x00

        pkt = bytearray(10)
        pkt[0] = motor_id & 0xFF
        pkt[1] = 0x64
        struct.pack_into(">h", pkt, 2, raw_val)
        pkt[4] = 0x00
        pkt[5] = 0x00
        pkt[6] = max(0, min(255, acceleration))
        pkt[7] = brake_byte
        pkt[8] = 0x00
        pkt[9] = calculate_crc8(bytes(pkt[:9]))

        if self.dry_run:
            fb = MotorFeedback(
                motor_id=motor_id,
                mode=self.MODE_VELOCITY,
                velocity_rpm=target_rpm if not brake else 0.0,
                timestamp=time.time(),
            )
            self.latest_feedback[motor_id] = fb
            return fb

        rx = self._transact(bytes(pkt), 10)
        if rx:
            fb = MotorFeedback.from_bytes(rx, speed_scale=self.speed_scale)
            if fb:
                self.latest_feedback[motor_id] = fb
                return fb
        return None

    def emergency_brake(self, motor_id: int) -> bool:
        """Command electric brake to actively hold position / halt immediately."""
        fb = self.set_velocity(motor_id, 0.0, brake=True)
        return fb is not None
