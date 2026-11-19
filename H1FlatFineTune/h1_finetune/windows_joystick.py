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


