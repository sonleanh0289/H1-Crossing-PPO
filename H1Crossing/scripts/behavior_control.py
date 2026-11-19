"""Pure tensor helpers for crossing behavior training; no simulator imports."""
from dataclasses import dataclass, asdict
from pathlib import Path
import math
import torch
from vehicle_motion import TrafficConfig

CONFIG = Path(__file__).resolve().parents[1] / "configs/crossing_behavior.yaml"
BEHAVIOR_FIELDS = (
    "car_dx/50", "car_dy/10", "goal_dx/10", "goal_dy/10", "car_vx/15", "car_vy/15", "ttc/10", "ttc_valid",
    "h1_vx/3", "h1_vy/3", "cos_yaw", "sin_yaw", "command_vx/3", "command_vy", "command_yaw/0.5",
    "stationary_anchor_dx/0.1", "stationary_anchor_dy/0.1", "stationary_time/episode_time", "remaining_time/episode_time",
)


@dataclass(frozen=True)
class BehaviorConfig:
    episode_seconds: float = 20.0
    behavior_decimation: int = 5
    action_deadband: float = .05
    heading_gain: float = 1.5
    contact_threshold_n: float = 1.
    fall_height_m: float = .55
    fall_up_z: float = .5
    env_spacing_m: float = 120.
    rollout_steps: int = 32
    learning_rate: float = 3e-4
    learning_epochs: int = 5
    mini_batches: int = 4
    entropy_coefficient: float = .005
    init_std: float = .5
    action_distribution: str = "gaussian"
    save_interval: int = 500
    guard_enabled: int = 0
    guard_mode: str = "conservative_v1"
    guard_ttc_threshold_s: float = 1.0
    guard_escape_speed_mps: float = 3.0
    guard_deceleration_mps2: float = 1.5
    guard_reaction_s: float = .2
    guard_time_margin_s: float = .5
    guard_position_margin_m: float = .5
    guard_recovery_s: float = .3

    def __post_init__(self):
        if self.guard_mode not in ("conservative_v1", "dual_loss_ttc_v1"):
            raise ValueError("Unknown crossing guard mode")
        if not 0 < self.guard_ttc_threshold_s or not 0 < self.guard_escape_speed_mps <= 3.:
            raise ValueError("Invalid dual-loss guard settings")
        if self.action_distribution not in ("gaussian", "beta"):
            raise ValueError("action_distribution must be gaussian or beta")
        if self.action_distribution == "beta" and not 0 < self.init_std < 1/math.sqrt(3):
            raise ValueError("Beta init_std must be below 1/sqrt(3)")
        if self.guard_enabled not in (0, 1) or min(self.guard_deceleration_mps2,
                self.guard_reaction_s, self.guard_time_margin_s,
                self.guard_position_margin_m, self.guard_recovery_s) <= 0:
            raise ValueError("Invalid crossing guard settings")
        if not all(not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)
                   for k, v in asdict(self).items() if k not in ("action_distribution", "guard_mode")):
            raise ValueError("Behavior settings must be finite numbers")
        for key in ("behavior_decimation", "rollout_steps", "learning_epochs", "mini_batches", "save_interval"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.behavior_decimation != 5 or self.env_spacing_m < 115:
            raise ValueError("This version uses 10 Hz behavior and spacing >=115m")
        if not 0 <= self.action_deadband < 1 or not -1 < self.fall_up_z < 1:
            raise ValueError("Invalid action deadband or fall tilt")
        if self.entropy_coefficient < 0 or min(self.episode_seconds, self.heading_gain, self.contact_threshold_n,
                self.fall_height_m, self.learning_rate, self.init_std) <= 0:
            raise ValueError("Invalid behavior coefficients")


def load_behavior_config(path=CONFIG):
    import yaml
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict) or set(values) - set(asdict(BehaviorConfig())):
        raise ValueError("Invalid or unknown behavior config keys")
    return BehaviorConfig(**values)


def speed_from_action(actions, deadband=.05):
    if actions.ndim != 2 or actions.shape[1] != 1 or not torch.isfinite(actions).all():
        raise ValueError("Behavior action must be finite (N,1)")
    a = actions[:, 0].clamp(-1, 1)
    a = torch.sign(a) * ((a.abs() - deadband) / (1 - deadband)).clamp_min(0)
    return torch.where(a >= 0, a * 3., a)  # certified range [-1,3] m/s; 0 means stop


def yaw_from_xyzw(quaternion):
    x, y, z, w = quaternion.unbind(-1)
    return torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def heading_command(position_xy, goal_xy, quaternion, speed, gain=1.5):
    delta = goal_xy - position_xy
    yaw = yaw_from_xyzw(quaternion)
    target = torch.atan2(delta[:, 1], delta[:, 0])
    error = torch.atan2(torch.sin(target-yaw), torch.cos(target-yaw))
    limit = torch.where(speed.abs() > 1.5, .25, .5)
    turn = (gain * error).clamp(-limit, limit)
    turn = torch.where(torch.linalg.vector_norm(delta, dim=1) < .05, 0., turn)
    return torch.stack((speed, torch.zeros_like(speed), turn), dim=1)


def behavior_observation(raw_crossing, h1_velocity_xy, quaternion, command, anchor_delta,
                         stationary_time, elapsed, episode_seconds):
    yaw = yaw_from_xyzw(quaternion)
    out = torch.cat((raw_crossing / raw_crossing.new_tensor([50,10,10,10,15,15,10,1]),
                     h1_velocity_xy/3, torch.stack((yaw.cos(), yaw.sin()), dim=1),
                     command / command.new_tensor([3,1,.5]), anchor_delta/.1,
                     (stationary_time/episode_seconds).unsqueeze(1),
                     ((episode_seconds-elapsed)/episode_seconds).clamp(0,1).unsqueeze(1)), dim=1)
    if out.shape[1] != len(BEHAVIOR_FIELDS) or not torch.isfinite(out).all():
        raise ValueError("Invalid behavior observation")
    return out.clamp(-10, 10)


class BatchedTraffic:
    """Independent +X traffic with bounded acceleration and partial resets."""
    def __init__(self, num_envs, device, seed, config=TrafficConfig()):
        config.validate()
        self.config = config
        self.x = torch.zeros(num_envs, device=device)
        self.speed = torch.zeros_like(self.x)
        self.target_speed = torch.zeros_like(self.x)
        self.target_remaining = torch.zeros_like(self.x)
        self.pass_count = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.will_yield = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.is_yielding = torch.zeros_like(self.will_yield)
        self.yield_clear_remaining = torch.zeros_like(self.x)

    def reset(self, ids):
        values = torch.rand((len(ids), 2), generator=self.generator, device=self.x.device)
        xlo, xhi = self.config.spawn_x_m
        vlo, vhi = self.config.speed_kmh
        self.x[ids] = xlo + (xhi-xlo)*values[:, 0]
        self.speed[ids] = (vlo+(vhi-vlo)*values[:, 1])/3.6
        self.target_speed[ids] = self.speed[ids]
        self.target_remaining[ids] = 0
        if self.config.mode == "variable":
            self._resample_target(ids)
        self.will_yield[ids] = False
        if self.config.yield_probability > 0:
            self.will_yield[ids] = torch.rand(len(ids), generator=self.generator, device=self.x.device) < self.config.yield_probability
        self.is_yielding[ids] = False
        self.yield_clear_remaining[ids] = 0
        self.pass_count[ids] += 1

    def _resample_target(self, ids):
        values = torch.rand((len(ids), 2), generator=self.generator, device=self.x.device)
        vlo, vhi = self.config.speed_kmh
        tlo, thi = self.config.target_interval_s
        self.target_speed[ids] = (vlo+(vhi-vlo)*values[:, 0])/3.6
        self.target_remaining[ids] = tlo+(thi-tlo)*values[:, 1]

    def advance(self, dt, h1_xy=None, vehicle_y=-1.75):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Traffic timestep must be finite and positive")
        if self.config.mode == "variable":
            expired = (self.target_remaining <= 1e-6).nonzero(as_tuple=False).flatten()
            if len(expired):
                self._resample_target(expired)
        cfg = self.config
        request = torch.zeros_like(self.will_yield)
        if h1_xy is not None:
            if h1_xy.shape != (len(self.x), 2) or not torch.isfinite(h1_xy).all():
                raise ValueError("H1 coordinates must be finite (num_envs, 2) local XY")
            request = ((h1_xy[:, 0].abs() <= cfg.yield_crossing_half_width_m)
                       & (h1_xy[:, 1] >= cfg.yield_wait_y_min_m)
                       & (h1_xy[:, 1] <= vehicle_y+.9+.35+cfg.yield_clearance_m))
        # Keep the original non-inference buffers: rollout runs in inference_mode,
        # but checkpoint restore resets these buffers outside inference_mode.
        self.yield_clear_remaining.copy_(torch.where(self.will_yield & request, cfg.yield_clear_delay_s,
                                                 (self.yield_clear_remaining-dt).clamp_min(0)))
        self.is_yielding.copy_(self.will_yield & (request | (self.yield_clear_remaining > 0))
                           & (self.x <= cfg.yield_crossing_half_width_m+2.1+.35))
        distance = (cfg.yield_stop_x_m-self.x-.25).clamp_min(0)
        allowed = (2*cfg.deceleration_mps2*distance).sqrt()
        target = torch.where(self.is_yielding, torch.minimum(self.target_speed, allowed), self.target_speed)
        required = self.speed.square() / (2*distance.clamp_min(.01))
        brake = torch.where(self.is_yielding, required.clamp(cfg.deceleration_mps2, cfg.emergency_deceleration_mps2),
                            cfg.deceleration_mps2)
        if cfg.mode == "variable" or cfg.yield_probability > 0:
            delta = target - self.speed
            rate = torch.where(delta >= 0, cfg.acceleration_mps2, brake)
            ramp_time = (delta.abs() / rate).clamp_max(dt)
            dv = delta.sign() * rate * ramp_time
            self.x += self.speed * dt + dv * (dt - .5*ramp_time)
            self.speed += dv
        else:
            self.x += self.speed * dt
        self.target_remaining -= dt
        ids = (self.x > self.config.exit_x_m).nonzero(as_tuple=False).flatten()
        if len(ids):
            self.reset(ids)
        return ids
