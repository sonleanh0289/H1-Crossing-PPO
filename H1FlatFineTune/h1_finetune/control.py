"""Shared command distribution, acceleration limits and Windows input mapping."""
from dataclasses import dataclass
import math
import torch

LEGACY_STAGES = ("A", "B", "C", "D", "E")
GRADUAL_STAGES = ("A1", "A2", "A3", "A4", "A5")
CURRICULUM_STAGES = GRADUAL_STAGES + ("B", "C", "D", "E")
STAGES = {"A": (-0.5, 2.0), "B": (-0.625, 2.25), "C": (-0.75, 2.5),
          "D": (-0.875, 2.75), "E": (-1.0, 3.0),
          "A1": (-0.30, 1.25), "A2": (-0.35, 1.5), "A3": (-0.40, 1.5),
          "A4": (-0.45, 1.75), "A5": (-0.50, 2.0)}
DT = 0.02
COMMAND_PROFILES = ("baseline", "turn_focus")


def resolve_command_profile(requested=None, state=None):
    """Explicit profile opts into a new sampler; resume otherwise inherits it."""
    profile = requested or (state or {}).get("command_profile", "baseline")
    if profile not in COMMAND_PROFILES:
        raise ValueError(f"Unknown command profile: {profile}")
    return profile


def stage_sequence(stage):
    """A1-A5 continue into B-E; keep A as a legacy entry point."""
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    return LEGACY_STAGES if stage == "A" else CURRICULUM_STAGES


def is_next_stage(previous, target):
    sequence = stage_sequence(previous)
    index = sequence.index(previous)
    return index + 1 < len(sequence) and sequence[index + 1] == target


def sample_commands(count, stage, device="cpu", generator=None, profile="baseline"):
    """Return commands and category IDs: forward=0, reverse=1, stand=2, turn=3."""
    low, high = STAGES[stage]
    profile = resolve_command_profile(profile)
    r = torch.rand((count, 5), device=device, generator=generator)
    boundaries = [0.50, 0.70, 0.80] if profile == "turn_focus" else [0.60, 0.85, 0.95]
    kind = torch.bucketize(r[:, 0].contiguous(), r.new_tensor(boundaries))
    upper = r[:, 1] < 0.6
    forward = (r[:, 2] + upper.float()) * (high / 2)
    vx = torch.where(kind == 0, forward, torch.where(kind == 1, r[:, 2] * low, 0.0))
    yaw_limit = torch.where(vx.abs() <= 1.5, 0.5, 0.25)
    yaw = (2 * r[:, 4] - 1) * yaw_limit
    yaw *= ((kind < 2) & (r[:, 3] < 0.3)) | (kind == 3)
    if profile == "turn_focus":
        # 60% near the boundary; 40% retain low-speed turns, with independent sign.
        magnitude = torch.where(r[:, 3] < 0.6, 0.4 + 0.1 * r[:, 2], 0.4 * r[:, 2])
        sign = torch.where(r[:, 4] < 0.5, -1.0, 1.0)
        yaw = torch.where(kind == 3, sign * magnitude, yaw)
    return torch.stack((vx, torch.zeros_like(vx), yaw), dim=-1), kind


def clamp_commands(target, stage):
    if not torch.isfinite(target).all():
        raise ValueError("Command contains NaN/Inf")
    result = target.clone()
    result[..., 0].clamp_(*STAGES[stage])
    result[..., 1] = 0
    limit = torch.where(result[..., 0].abs() <= 1.5, 0.5, 0.25)
    result[..., 2] = result[..., 2].clamp(-limit, limit)
    return result


def slew(current, target, dt=DT):
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if not torch.isfinite(current).all() or not torch.isfinite(target).all():
        raise ValueError("Non-finite command")
    limits = current.new_tensor([2.0, 0.0, 1.0]) * dt
    adjusted = target.clone()
    # While decelerating from high speed, do not start the sharper slow-speed turn early.
    adjusted[..., 2] = torch.where(current[..., 0].abs() > 1.5,
                                   adjusted[..., 2].clamp(-0.25, 0.25), adjusted[..., 2])
    result = current + (adjusted - current).clamp(-limits, limits)
    # Wait at 1.5 m/s until yaw has ramped down; do not violate yaw acceleration by clipping it.
    waiting = (current[..., 0].abs() <= 1.5) & (result[..., 0].abs() > 1.5) & (result[..., 2].abs() > 0.25)
    result[..., 0] = torch.where(waiting, result[..., 0].clamp(-1.5, 1.5), result[..., 0])
    return result


@dataclass
class InputState:
    emergency_stop: bool = False
    reset_requested: bool = False


class KeyboardMapper:
    """Hold WASD; Shift uses stage limits; Space latches stop, Enter resumes, R resets."""
    def __init__(self, stage="A", forward_speed=1.0, reverse_speed=0.5):
        if stage not in STAGES:
            raise ValueError(f"Unknown stage: {stage}")
        if any(not math.isfinite(v) or v <= 0 for v in (forward_speed, reverse_speed)):
            raise ValueError("Keyboard speeds must be finite and positive")
        self.stage = stage
        self.forward_speed, self.reverse_speed = forward_speed, reverse_speed
        self.state = InputState()
        self.previous_keys = frozenset()
        self.armed = False

    def reset(self):
        self.armed = False
        self.state.reset_requested = False
        # Keep held-key edges: holding R must not reset every simulation frame.

    def advance(self, sample):
        if not sample.focused:
            self.armed = False
            self.previous_keys = frozenset()
            return (0.0, 0.0, 0.0)
        keys = frozenset(sample.keys)
        pressed = keys - self.previous_keys
        self.previous_keys = keys
        if "ENTER" in pressed:
            self.state.emergency_stop = False
            self.armed = False  # Release all control keys before resuming.
        if "SPACE" in keys:
            self.state.emergency_stop = True
        if "R" in pressed:
            self.state.reset_requested = True
        if self.state.emergency_stop or self.state.reset_requested:
            return (0.0, 0.0, 0.0)
        if not self.armed:
            self.armed = not keys
            return (0.0, 0.0, 0.0)
        low, high = STAGES[self.stage]
        forward = high if "SHIFT" in keys else min(self.forward_speed, high)
        backward = -low if "SHIFT" in keys else min(self.reverse_speed, -low)
        direction = int("W" in keys) - int("S" in keys)
        vx = direction * (forward if direction >= 0 else backward)
        yaw = (int("A" in keys) - int("D" in keys)) * (0.5 if abs(vx) <= 1.5 else 0.25)
        return (vx, 0.0, yaw)


class JoystickMapper:
    """Button 1 resume, 2 latched stop, 3 reset; neutral gate after reset/disconnect."""
    def __init__(self, stage="A", forward_axis="y", turn_axis="x", forward_sign=-1, turn_sign=-1):
        self.stage = stage
        self.forward_axis, self.turn_axis = forward_axis, turn_axis
        self.forward_sign, self.turn_sign = forward_sign, turn_sign
        self.state = InputState()
        self.previous_buttons = 0
        self.armed = False

    def reset(self):
        self.armed = False
        self.state.reset_requested = False

    def advance(self, sample):
        if not sample.connected or any(a not in sample.axes for a in (self.forward_axis, self.turn_axis)):
            self.armed = False
            self.previous_buttons = 0
            return (0.0, 0.0, 0.0)
        forward = sample.axes[self.forward_axis] * self.forward_sign
        turn = sample.axes[self.turn_axis] * self.turn_sign
        if not math.isfinite(forward) or not math.isfinite(turn):
            self.armed = False
            return (0.0, 0.0, 0.0)
        pressed = sample.buttons & ~self.previous_buttons
        self.previous_buttons = sample.buttons
        if pressed & 1:
            self.state.emergency_stop = False
        if pressed & 2:
            self.state.emergency_stop = True
        if pressed & 4:
            self.state.reset_requested = True
        if not self.armed:
            self.armed = abs(forward) <= 0.05 and abs(turn) <= 0.05
            return (0.0, 0.0, 0.0)
        if self.state.emergency_stop or self.state.reset_requested:
            return (0.0, 0.0, 0.0)
        low, high = STAGES[self.stage]
        vx = max(-1.0, min(1.0, forward)) * (high if forward >= 0 else -low)
        return (vx, 0.0, max(-1.0, min(1.0, turn)) * (0.5 if abs(vx) <= 1.5 else 0.25))
