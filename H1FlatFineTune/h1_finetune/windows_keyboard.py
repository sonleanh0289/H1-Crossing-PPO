"""Poll only control keys while the simulator's process owns the foreground window.

No global hooks, third-party dependencies or key logging. Missing focus fails closed.
Input commands are still smoothed by SmoothVelocityCommand in the environment.
"""
import ctypes as ct
from dataclasses import dataclass
import os
import sys


@dataclass(frozen=True)
class KeyboardSample:
    focused: bool
    keys: frozenset[str] = frozenset()


class WindowsKeyboard:
    KEY_CODES = {"W": 0x57, "S": 0x53, "A": 0x41, "D": 0x44,
                 "SHIFT": 0x10, "SPACE": 0x20, "ENTER": 0x0D, "R": 0x52}

    def __init__(self):
        if sys.platform != "win32":
            raise RuntimeError("Keyboard playback currently requires Windows")
        self.api = ct.WinDLL("user32.dll", use_last_error=True)
        self.api.GetForegroundWindow.argtypes = []
        self.api.GetForegroundWindow.restype = ct.c_void_p
        self.api.GetWindowThreadProcessId.argtypes = [ct.c_void_p, ct.POINTER(ct.c_uint32)]
        self.api.GetWindowThreadProcessId.restype = ct.c_uint32
        self.api.GetAsyncKeyState.argtypes = [ct.c_int]
        self.api.GetAsyncKeyState.restype = ct.c_short
        self.pid = os.getpid()

    def read(self):
        window = self.api.GetForegroundWindow()
        if not window:
            return KeyboardSample(False)
        pid = ct.c_uint32()
        thread_id = self.api.GetWindowThreadProcessId(window, ct.byref(pid))
        if not thread_id or pid.value != self.pid:
            return KeyboardSample(False)
        # High bit means currently held. Ignore the unreliable "pressed since last call" bit.
        keys = frozenset(name for name, code in self.KEY_CODES.items()
                         if self.api.GetAsyncKeyState(code) & 0x8000)
        return KeyboardSample(True, keys)
