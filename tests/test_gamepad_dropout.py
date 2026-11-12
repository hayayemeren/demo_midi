import logging
import os
import sys
import time

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.base_controller import BaseController
from src.ddsm import MotorFeedback
from src.gamepad import F710Gamepad


class DroppedDev:
    """evdev device whose F710 receiver has just dropped off USB."""

    def read_one(self):
        raise OSError(19, "No such device")

    def close(self):
        pass


class QuietDev:
    def read_one(self):
        return None

    def close(self):
        pass


def make_gamepad_with_stick_forward(dev):
    gp = F710Gamepad()
    gp.ax = {1: (-32768, 32767)}
    gp.val = {1: -32768}  # Left stick held fully forward
    gp.dev = dev
    gp.connected = True
    return gp


def test_dropout_releases_the_stick():
    gp = make_gamepad_with_stick_forward(DroppedDev())
    assert gp.poll().connected is False
    assert gp.norm(1) == 0.0


def test_dropout_reconnect_is_retried_several_times_per_second(monkeypatch):
    gp = make_gamepad_with_stick_forward(DroppedDev())
    gp.poll()

    attempts = []
    monkeypatch.setattr(gp, "connect", lambda *args, **kwargs: attempts.append(time.time()) or False)
    deadline = time.time() + 0.5
    while time.time() < deadline:
        gp.poll()
        time.sleep(0.01)
    assert len(attempts) >= 2


def test_reconnect_logs_how_long_the_signal_was_gone(monkeypatch, caplog):
    gp = make_gamepad_with_stick_forward(DroppedDev())
    with caplog.at_level(logging.INFO, logger="demo.gamepad"):
        gp.poll()

        def reconnect(*args, **kwargs):
            gp.dev = QuietDev()
            return True

        monkeypatch.setattr(gp, "connect", reconnect)
        gp._last_reconnect_attempt = 0.0
        state = gp.poll()

    assert state.connected is True
    assert "Gamepad disconnected" in caplog.text
    assert "Back after" in caplog.text


class ScriptedDriver:
    """DDSM driver stand-in that returns preset feedback per motor id."""

    def __init__(self):
        self.replies = {}

    def set_velocity(self, motor_id, target_rpm, acceleration=1, brake=False):
        return self.replies.get(motor_id)


def make_base():
    cfg = {"dry_run": True, "serial": {"port": "/dev/null"}, "motors": {"left_id": 1, "right_id": 2}}
    base = BaseController(cfg, dry_run=True)
    base.driver = ScriptedDriver()
    return base


def drive_ticks(base, ticks):
    for _ in range(ticks):
        base.drive(forward=0.5, turn=0.0, max_rpm=100.0, dt=0.02)


def motor_lines(caplog):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("[MOTOR")]


def test_motor_fault_is_logged_once_and_recovery_is_logged(caplog):
    base = make_base()
    base.driver.replies = {1: MotorFeedback(stall_fault=True), 2: MotorFeedback()}
    with caplog.at_level(logging.INFO, logger="demo.base"):
        drive_ticks(base, 10)
        base.driver.replies[1] = MotorFeedback()
        drive_ticks(base, 10)

    assert motor_lines(caplog) == ["[MOTOR LEFT] stall", "[MOTOR LEFT] OK again"]


def test_motor_silence_is_reported_only_after_several_missed_replies(caplog):
    base = make_base()
    base.driver.replies = {1: MotorFeedback(), 2: MotorFeedback()}
    with caplog.at_level(logging.INFO, logger="demo.base"):
        drive_ticks(base, 3)
        del base.driver.replies[2]
        drive_ticks(base, 2)
        assert motor_lines(caplog) == []
        drive_ticks(base, 5)

    assert motor_lines(caplog) == ["[MOTOR RIGHT] not answering on RS485"]
