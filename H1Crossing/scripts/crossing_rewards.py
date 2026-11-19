"""Batched rewards for the high-level crossing environment, without Kit.

Call step on PRE-RESET next states with real contact/fall flags, then reset done
environment ids. Does not change the existing manual preview or locomotion.
"""
from dataclasses import dataclass, fields
import math
from pathlib import Path
import torch

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "crossing_rewards.yaml"


@dataclass(frozen=True)
class CrossingRewardConfig:
    goal_reward: float = 1.0
    safe_episode_bonus: float = 1.0
    vehicle_collision_penalty: float = -2.0
    fall_penalty: float = 0.0
    timeout_penalty: float = -1.0
    out_of_bounds_penalty: float = -1.0
    progress_reward_per_m: float = 0.5
    goal_radius_m: float = 0.6
    stationary_radius_m: float = 0.10
    stationary_grace_s: float = 2.0
    stationary_penalty_growth_per_s2: float = 0.05
    stationary_max_penalty_per_s: float = 0.25

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{field.name} must be a finite number")
        if self.goal_reward < 0 or self.safe_episode_bonus < 0 or self.progress_reward_per_m < 0:
            raise ValueError("Positive event rewards cannot be negative")
        if any(getattr(self, name) > 0 for name in ("vehicle_collision_penalty", "fall_penalty",
                                                   "timeout_penalty", "out_of_bounds_penalty")):
            raise ValueError("Collision/fall/timeout/out-of-bounds penalties cannot be positive")
        if self.goal_radius_m <= 0 or self.stationary_radius_m <= 0:
            raise ValueError("Position radii must be positive")
        if min(self.stationary_grace_s, self.stationary_penalty_growth_per_s2,
               self.stationary_max_penalty_per_s) < 0:
            raise ValueError("Stationary penalty parameters cannot be negative")


def load_reward_config(path=DEFAULT_CONFIG):
    import yaml
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Reward YAML must contain a mapping")
    unknown = set(values) - {field.name for field in fields(CrossingRewardConfig)}
    if unknown:
        raise ValueError(f"Unknown reward config keys: {sorted(unknown)}")
    return CrossingRewardConfig(**values)


def stationary_penalty_integral(elapsed_s, cfg):
    """Integral of the positive ramped/capped rate since the last movement.

    Step cost = integral(t+dt)-integral(t), exactly across grace/cap boundaries.
    The returned cost already includes elapsed time; do not multiply by dt again.
    """
    elapsed = (elapsed_s - cfg.stationary_grace_s).clamp_min(0)
    growth, cap = cfg.stationary_penalty_growth_per_s2, cfg.stationary_max_penalty_per_s
    if growth == 0 or cap == 0:
        return torch.zeros_like(elapsed)
    ramp_time = elapsed.clamp_max(cap / growth)
    return 0.5 * growth * ramp_time.square() + cap * (elapsed - ramp_time)


class CrossingRewardTracker:
    """One-shot rewards and independent per-environment stationary timers.

    Safe bonus is paid once at successful goal, never on timeout/fall/contact.
    Collision/fall/out-of-bounds wins over simultaneous goal. Terminal events stay latched until
    reset. Administrative reset/stop is NOT a timeout eligible for a safe bonus.
    """
    def __init__(self, num_envs, cfg=None, *, device="cpu", dtype=torch.float32):
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Use float32 or float64 reward state")
        self.cfg = load_reward_config() if cfg is None else cfg
        self.anchor_xy = torch.zeros((num_envs, 2), device=device, dtype=dtype)
        self.previous_xy = torch.zeros_like(self.anchor_xy)
        self.stationary_elapsed_s = torch.zeros(num_envs, device=device, dtype=dtype)
        self.finished = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.initialized = torch.zeros_like(self.finished)

    def _position(self, value):
        if (not isinstance(value, torch.Tensor) or value.shape != self.anchor_xy.shape
                or value.device != self.anchor_xy.device or value.dtype != self.anchor_xy.dtype
                or not torch.isfinite(value).all()):
            raise ValueError("Positions must be finite matching (num_envs,2) tensors on the tracker device")

    def _flag(self, value):
        if (not isinstance(value, torch.Tensor) or value.shape != self.finished.shape
                or value.device != self.finished.device or value.dtype != torch.bool):
            raise ValueError("Event flags must be bool (num_envs,) tensors on the tracker device")

    def reset(self, h1_position_xy, env_ids=None):
        """Supply full position batch; reset only selected ids after physics reset."""
        self._position(h1_position_xy)
        selected = slice(None) if env_ids is None else env_ids
        self.anchor_xy[selected] = h1_position_xy[selected]
        self.previous_xy[selected] = h1_position_xy[selected]
        self.stationary_elapsed_s[selected] = 0
        self.finished[selected] = False
        self.initialized[selected] = True

    @torch.no_grad()
    def step(self, h1_position_xy, goal_position_xy, *, fallen, vehicle_contact, timeout, dt,
             out_of_bounds=None):
        """Return weighted terms, total reward, success, terminated and truncated.

        Inputs are after the transition, BEFORE automatic physics reset. Contact
        means actual H1-vehicle collision, not TTC or ground contact. The caller
        aggregates flags over low-level substeps. Positions share world XY axes
        and include env_origins. dt is simulated seconds per behavior transition.
        """
        self._position(h1_position_xy)
        self._position(goal_position_xy)
        if out_of_bounds is None:
            out_of_bounds = torch.zeros_like(self.finished)
        for flag in (fallen, vehicle_contact, timeout, out_of_bounds):
            self._flag(flag)
        if isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive simulated seconds")
        if not self.initialized.all():
            raise RuntimeError("reset() every environment before computing rewards")
        cfg = self.cfg
        active = ~self.finished
        moved = active & (torch.linalg.vector_norm(h1_position_xy - self.anchor_xy, dim=-1)
                          >= cfg.stationary_radius_m)
        previous_time = self.stationary_elapsed_s
        next_time = torch.where(moved, 0., previous_time + dt)
        stationary = torch.where(active & ~moved,
            -(stationary_penalty_integral(next_time, cfg) - stationary_penalty_integral(previous_time, cfg)), 0.)
        self.anchor_xy[moved] = h1_position_xy[moved]
        self.stationary_elapsed_s.copy_(torch.where(active, next_time, previous_time))

        failure = fallen | vehicle_contact | out_of_bounds
        distance = torch.linalg.vector_norm(goal_position_xy - h1_position_xy, dim=-1)
        previous_distance = torch.linalg.vector_norm(goal_position_xy - self.previous_xy, dim=-1)
        # Signed distance reduction, not a per-step proximity/survival bonus.
        # No extra dt: subdividing the same path preserves the undiscounted sum.
        progress = cfg.progress_reward_per_m * (previous_distance - distance)
        # Being knocked toward the goal on a failing transition earns no progress.
        progress = torch.where(failure, progress.clamp_max(0), progress)
        progress = torch.where(active, progress, 0.)
        self.previous_xy[active] = h1_position_xy[active]
        at_goal = distance <= cfg.goal_radius_m
        success = active & at_goal & ~failure
        terminated = active & (success | failure)
        truncated = active & timeout & ~terminated
        done = terminated | truncated
        goal = success.to(self.anchor_xy.dtype) * cfg.goal_reward
        safe = success.to(self.anchor_xy.dtype) * cfg.safe_episode_bonus
        timeout_cost = truncated.to(self.anchor_xy.dtype) * cfg.timeout_penalty
        collision = (active & vehicle_contact).to(self.anchor_xy.dtype) * cfg.vehicle_collision_penalty
        fall = (active & fallen).to(self.anchor_xy.dtype) * cfg.fall_penalty
        boundary_done = active & out_of_bounds
        boundary_cost = boundary_done.to(self.anchor_xy.dtype) * cfg.out_of_bounds_penalty
        self.finished |= done
        return {"goal": goal, "safe_episode": safe, "vehicle_collision": collision,
                "fall": fall, "stationary": stationary, "timeout": timeout_cost, "progress": progress,
                "out_of_bounds": boundary_cost, "out_of_bounds_done": boundary_done,
                "total": goal + safe + collision + fall + stationary + timeout_cost + progress + boundary_cost,
                "success": success, "terminated": terminated, "truncated": truncated,
                "stationary_elapsed_s": self.stationary_elapsed_s.clone()}
