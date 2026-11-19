"""Poll Windows joysticks without Kit mappings or third-party dependencies.

Diagnostic: python windows_joystick.py --seconds 15
API: https://learn.microsoft.com/windows/win32/api/joystickapi/
"""

from __future__ import annotations

import argparse
import ctypes as ct
import json
import sys
import time
from dataclasses import dataclass


class JoyInfoEx(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in (
        "size", "flags", "x", "y", "z", "r", "u", "v", "buttons",
        "button_number", "pov", "reserved1", "reserved2",
    )]


class JoyCaps(ct.Structure):
    _fields_ = [("mid", ct.c_uint16), ("pid", ct.c_uint16), ("name", ct.c_wchar * 32)] + [
        (name, ct.c_uint32) for name in (
            "xmin", "xmax", "ymin", "ymax", "zmin", "zmax", "num_buttons",
            "period_min", "period_max", "rmin", "rmax", "umin", "umax", "vmin", "vmax",
            "caps", "max_axes", "num_axes", "max_buttons",
        )
    ] + [("reg_key", ct.c_wchar * 32), ("oem", ct.c_wchar * 260)]


def normalize_axis(value: int, low: int, high: int, dead_zone: float) -> float:
    if high <= low:
        raise ValueError("Joystick axis has an invalid range")
    value = max(-1.0, min(1.0, 2.0 * (value - low) / (high - low) - 1.0))
    return 0.0 if abs(value) <= dead_zone else value


@dataclass
class Sample:
    connected: bool
    axes: dict[str, float]
    buttons: int = 0


class WindowsJoystick:
    def __init__(self, device_id: int | None = None, dead_zone: float = 0.05):
        if sys.platform != "win32":
            raise RuntimeError("Windows joystick backend requires Windows")
        if not 0 <= dead_zone < 1:
            raise ValueError("dead_zone must be in [0, 1)")
        self.api = ct.WinDLL("winmm.dll")
        self.api.joyGetNumDevs.argtypes = []
        self.api.joyGetNumDevs.restype = ct.c_uint32
        self.api.joyGetDevCapsW.argtypes = [ct.c_size_t, ct.POINTER(JoyCaps), ct.c_uint32]
        self.api.joyGetDevCapsW.restype = ct.c_uint32
        self.api.joyGetPosEx.argtypes = [ct.c_uint32, ct.POINTER(JoyInfoEx)]
        self.api.joyGetPosEx.restype = ct.c_uint32
        self.dead_zone = dead_zone
        candidates = range(self.api.joyGetNumDevs()) if device_id is None else [device_id]
        for candidate in candidates:
            caps = JoyCaps()
            info = JoyInfoEx(size=ct.sizeof(JoyInfoEx), flags=0xFF)
            if self.api.joyGetDevCapsW(candidate, ct.byref(caps), ct.sizeof(caps)) == 0:
                if self.api.joyGetPosEx(candidate, ct.byref(info)) == 0:
                    self.device_id, self.caps = candidate, caps
                    break
        else:
            raise RuntimeError("No connected Windows joystick. Connect it and check joy.cpl, then relaunch.")
        self.axis_names = ["x", "y"] + [
            axis for axis, flag in (("z", 1), ("r", 2), ("u", 4), ("v", 8))
            if self.caps.caps & flag
        ]

    def describe(self) -> str:
        return (f"Windows joystick id={self.device_id}, name={self.caps.name!r}, "
                f"axes={self.axis_names}, buttons={self.caps.num_buttons}")

    def read(self) -> Sample:
        info = JoyInfoEx(size=ct.sizeof(JoyInfoEx), flags=0xFF)
        if self.api.joyGetPosEx(self.device_id, ct.byref(info)) != 0:
            return Sample(False, {})
        axes = {axis: normalize_axis(getattr(info, axis), getattr(self.caps, axis + "min"),
                                    getattr(self.caps, axis + "max"), self.dead_zone)
                for axis in self.axis_names}
        return Sample(True, axes, info.buttons)


class CommandMapper:
    """Edge-trigger buttons; require neutral sticks at startup/reconnect/reset."""

    def __init__(self, state, speed=0.6, yaw_rate=0.5, forward_axis="y", turn_axis="x",
                 forward_sign=-1, turn_sign=-1):
        self.state = state
        self.speed, self.yaw_rate = speed, yaw_rate
        self.forward_axis, self.turn_axis = forward_axis, turn_axis
        self.forward_sign, self.turn_sign = forward_sign, turn_sign
        self.previous_buttons = 0
        self.reset()

    def reset(self):
        self.armed = False

    def advance(self, sample: Sample) -> tuple[float, float, float]:
        if not sample.connected:
            self.reset()
            self.previous_buttons = 0
            return (0.0, 0.0, 0.0)
        forward = sample.axes[self.forward_axis] * self.forward_sign
        turn = sample.axes[self.turn_axis] * self.turn_sign
        pressed = sample.buttons & ~self.previous_buttons
        self.previous_buttons = sample.buttons
        # Generic USB buttons use Windows numbers, not Xbox letter labels.
        if pressed & 1:  # button 1: resume
            self.state.emergency_stop = False
        if pressed & 2:  # button 2: stop; wins if both are pressed
            self.state.emergency_stop = True
        if pressed & 4:  # button 3: reset, once per press
            self.state.reset_requested = True
        if not self.armed:
            self.armed = forward == 0.0 and turn == 0.0
            return (0.0, 0.0, 0.0)
        if self.state.emergency_stop or self.state.reset_requested:
            return (0.0, 0.0, 0.0)
        return (max(0.0, min(1.0, forward)) * self.speed, 0.0,
                max(-1.0, min(1.0, turn)) * self.yaw_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--device_id", type=int)
    args = parser.parse_args()
    joystick = WindowsJoystick(args.device_id)
    print(joystick.describe(), flush=True)
    print("Move sticks to identify axes; buttons are a bitmask (1, 2, 4 = buttons 1, 2, 3).", flush=True)
    until = time.monotonic() + args.seconds
    while time.monotonic() < until:
        sample = joystick.read()
        print(json.dumps(vars(sample)), flush=True)
        time.sleep(0.25)
