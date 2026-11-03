"""Network Protocol for Remote Teleoperation (Case 3).

Provides robust, low-latency TCP communication between PC Client and Robot Server.
Features JSON-lines message framing, TCP_NODELAY for sub-millisecond latency,
and a watchdog failsafe that halts robot motion if communication drops.
"""

from __future__ import annotations

import json
import logging
import select
import socket
import sys
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("demo.network")

DEFAULT_PORT = 8765
DEFAULT_WATCHDOG_TIMEOUT = 0.50  # Seconds before safety halt if connection goes silent


class RobotServer:
    """TCP Server running on the Raspberry Pi.
    
    Accepts connections from PC Client, receives motion/teleoperation commands,
    and streams telemetry feedback back to the PC.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
        watchdog_timeout: float = DEFAULT_WATCHDOG_TIMEOUT,
    ):
        self.host = host
        self.port = port
        self.watchdog_timeout = watchdog_timeout

        self.server_sock: Optional[socket.socket] = None
        self.client_sock: Optional[socket.socket] = None
        self.client_addr: Optional[Tuple[str, int]] = None
        self._rx_buffer = ""
        self.last_cmd_time = 0.0
        self.last_cmd: Optional[Dict[str, Any]] = None
        self.is_connected = False

    def start(self) -> bool:
        """Bind and listen on specified port."""
        try:
            self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_sock.bind((self.host, self.port))
            self.server_sock.listen(1)
            self.server_sock.setblocking(False)
            logger.info("Robot Network Server listening on %s:%d", self.host, self.port)
            return True
        except Exception as e:
            logger.error("Failed to start Robot Server on %s:%d: %s", self.host, self.port, e)
            self.server_sock = None
            return False

    def poll_connections(self) -> bool:
        """Check for incoming client connections (non-blocking)."""
        if self.server_sock is None:
            return False

        if self.client_sock is not None:
            return True  # Already connected

        try:
            r, _, _ = select.select([self.server_sock], [], [], 0.0)
            if r:
                conn, addr = self.server_sock.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                conn.setblocking(False)
                self.client_sock = conn
                self.client_addr = addr
                self.is_connected = True
                self.last_cmd_time = time.time()
                self._rx_buffer = ""
                logger.info("[NETWORK] PC Client connected from %s:%d", addr[0], addr[1])
                return True
        except Exception as e:
            logger.debug("Error accepting connection: %s", e)

        return False

    def poll_commands(self) -> Tuple[Optional[Dict[str, Any]], bool]:
        """Read latest command packet from client.
        
        Returns:
            (command_dict, is_watchdog_timed_out)
        """
        self.poll_connections()

        if self.client_sock is None:
            return None, True

        now = time.time()

        try:
            # Non-blocking read
            r, _, _ = select.select([self.client_sock], [], [], 0.0)
            if r:
                chunk = self.client_sock.recv(4096)
                if not chunk:
                    # Client disconnected
                    logger.warning("[NETWORK] PC Client disconnected.")
                    self._disconnect_client()
                    return None, True

                self._rx_buffer += chunk.decode("utf-8", errors="ignore")

                # Parse newline-delimited JSON messages
                while "\n" in self._rx_buffer:
                    line, self._rx_buffer = self._rx_buffer.split("\n", 1)
                    line = line.strip()
                    if line:
                        try:
                            msg = json.loads(line)
                            self.last_cmd = msg
                            self.last_cmd_time = now
                        except Exception as e:
                            logger.debug("Failed to decode JSON packet: %s", e)

        except (socket.error, OSError) as e:
            logger.warning("[NETWORK] Socket error reading client: %s", e)
            self._disconnect_client()
            return None, True

        # Check watchdog timer
        watchdog_trip = (now - self.last_cmd_time) > self.watchdog_timeout
        if watchdog_trip and self.last_cmd is not None:
            logger.warning("[WATCHDOG] No command from PC for %.2fs! Safety halt engaged.", now - self.last_cmd_time)

        return self.last_cmd, watchdog_trip

    def send_telemetry(self, data: Dict[str, Any]):
        """Send JSON telemetry frame to PC Client."""
        if self.client_sock is None:
            return

        try:
            payload = json.dumps(data) + "\n"
            self.client_sock.sendall(payload.encode("utf-8"))
        except (socket.error, OSError):
            self._disconnect_client()

    def _disconnect_client(self):
        """Clean up disconnected client socket."""
        if self.client_sock:
            try:
                self.client_sock.close()
            except Exception:
                pass
            self.client_sock = None
        self.is_connected = False
        self.last_cmd = None
        self._rx_buffer = ""

    def close(self):
        """Shut down server and close sockets."""
        self._disconnect_client()
        if self.server_sock:
            try:
                self.server_sock.close()
            except Exception:
                pass
            self.server_sock = None
        logger.info("Robot Network Server closed.")


class RobotClient:
    """TCP Client running on the operator PC.
    
    Connects to the Raspberry Pi, streams Leader Arm commands, and receives
    live telemetry feedback.
    """

    def __init__(self, host: str, port: int = DEFAULT_PORT, timeout_s: float = 3.0):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.sock: Optional[socket.socket] = None
        self.is_connected = False
        self._rx_buffer = ""
        self.last_telemetry: Optional[Dict[str, Any]] = None
        self.last_telemetry_time = 0.0

    def connect(self) -> bool:
        """Establish connection to Robot Server."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.settimeout(self.timeout_s)
            s.connect((self.host, self.port))
            s.setblocking(False)
            self.sock = s
            self.is_connected = True
            self._rx_buffer = ""
            logger.info("Connected to Robot Server at %s:%d", self.host, self.port)
            return True
        except Exception as e:
            logger.debug("Failed to connect to %s:%d: %s", self.host, self.port, e)
            self.sock = None
            self.is_connected = False
            return False

    def send_command(self, cmd: Dict[str, Any]) -> bool:
        """Send command packet to Robot Server."""
        if self.sock is None:
            return False

        try:
            payload = json.dumps(cmd) + "\n"
            self.sock.sendall(payload.encode("utf-8"))
            return True
        except (socket.error, OSError) as e:
            logger.warning("Error sending command to robot: %s", e)
            self.disconnect()
            return False

    def poll_telemetry(self) -> Optional[Dict[str, Any]]:
        """Read latest telemetry frame received from robot."""
        if self.sock is None:
            return None

        try:
            r, _, _ = select.select([self.sock], [], [], 0.0)
            if r:
                chunk = self.sock.recv(4096)
                if not chunk:
                    logger.warning("Robot server closed connection.")
                    self.disconnect()
                    return None

                self._rx_buffer += chunk.decode("utf-8", errors="ignore")

                while "\n" in self._rx_buffer:
                    line, self._rx_buffer = self._rx_buffer.split("\n", 1)
                    line = line.strip()
                    if line:
                        try:
                            self.last_telemetry = json.loads(line)
                            self.last_telemetry_time = time.time()
                        except Exception:
                            pass

        except (socket.error, OSError):
            self.disconnect()
            return None

        return self.last_telemetry

    def disconnect(self):
        """Close socket connection."""
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
        self.is_connected = False
