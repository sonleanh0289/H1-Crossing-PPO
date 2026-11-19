"""Sensor/odometry-only command governor. No vehicle simulator state access.

Versioned conservative and dual-loss TTC policies. No collision-free guarantee.
Locomotion slew remains in force, including when requesting escape speed.
"""
import math
import torch
from crossing_observations import planar_ttc


class CrossingCommandGuard:
    def __init__(self, count, device, settings):
        self.cfg = settings
        self.clear_seconds = torch.zeros(count, device=device)
        self.last_valid_ttc = torch.zeros(count, device=device)
        self.ttc_age = torch.zeros(count, device=device)
        self.has_ttc = torch.zeros(count, device=device, dtype=torch.bool)

    @torch.inference_mode()
    def reset(self, ids):
        self.clear_seconds[ids] = 0.
        self.last_valid_ttc[ids] = 0.
        self.ttc_age[ids] = 0.
        self.has_ttc[ids] = False

    @property
    def uses_dual_loss(self):
        return self.cfg.guard_mode == "dual_loss_ttc_v1"

    def apply(self, desired, obs, dt):
        # 22/23D policy observation, before the action's low-level application.
        if (obs.ndim != 2 or obs.shape[1] < 22 or obs.shape[0] != len(self.clear_seconds)
                or not torch.isfinite(obs).all()):
            raise ValueError("Guard requires finite camera observations")
        if (desired.shape != (len(obs), 3) or not torch.isfinite(desired).all()
                or not math.isfinite(dt) or dt <= 0):
            raise ValueError("Guard requires finite commands and positive simulation dt")
        if self.uses_dual_loss:
            return self._apply_dual_loss(desired, obs, dt)
        position_valid = obs[:, 19] > .5
        velocity_valid = obs[:, 20] > .5
        reliable = position_valid & velocity_valid
        relative = obs[:, :2] * obs.new_tensor([50., 10.])
        car_velocity = obs[:, 4:6] * 15.
        actual = obs[:, 8:10] * 3.
        heading = obs[:, 10:12]
        command_speed = obs[:, 12] * 3.
        speed = torch.maximum(torch.linalg.vector_norm(actual, dim=1), command_speed.abs())
        stopping_time = speed / self.cfg.guard_deceleration_mps2
        horizon = stopping_time + self.cfg.guard_reaction_s + self.cfg.guard_time_margin_s
        # Evaluate both current motion and the requested motion, so a stopped
        # robot does not interpret its zero current speed as permission to enter.
        requested_velocity = heading * desired[:, :1]
        risk = torch.zeros_like(reliable)
        minimum_ttc = torch.full_like(speed, 10.)
        for h1_velocity in (actual, requested_velocity):
            ttc, valid = planar_ttc(relative, car_velocity-h1_velocity,
                h1_radius_m=.35+self.cfg.guard_position_margin_m)
            ttc, valid = ttc[:, 0], valid[:, 0]
            risk |= valid & (ttc <= horizon)
            minimum_ttc = torch.minimum(minimum_ttc, ttc)
        # Also keep the actor's currently valid collision forecast conservative.
        risk |= (obs[:, 7] > .5) & (obs[:, 6]*10 <= horizon)
        unsafe = ~reliable | risk
        self.clear_seconds = torch.where(unsafe, 0., self.clear_seconds+dt)
        blocked = unsafe | (self.clear_seconds < self.cfg.guard_recovery_s-1e-6)
        result = desired.clone()
        result[blocked] = 0.  # Target only; never bypass certified slew/turn limits.
        diagnostics = dict(missing_position=~position_valid, missing_velocity=~velocity_valid,
            predicted_conflict=reliable & risk, recovery_hold=blocked & ~unsafe,
            blocked=blocked, proposed_vx=desired[:, 0].clone(), guarded_vx=result[:, 0].clone(),
            stopping_time_s=stopping_time, guard_horizon_s=horizon,
            predicted_ttc_s=minimum_ttc)
        return result, diagnostics

    def _apply_dual_loss(self, desired, obs, dt):
        position_valid = obs[:, 19] > .5
        velocity_valid = obs[:, 20] > .5
        # Cache only an actual sensor TTC, never the invalid 10-second sentinel.
        fresh_ttc = position_valid & velocity_valid & (obs[:, 7] > .5) & (obs[:, 6] >= 0.)
        self.ttc_age = torch.where(fresh_ttc, 0., self.ttc_age + dt)
        self.last_valid_ttc = torch.where(fresh_ttc, obs[:, 6] * 10., self.last_valid_ttc)
        self.has_ttc = self.has_ttc | fresh_ttc
        remaining = (self.last_valid_ttc - self.ttc_age).clamp_min(0.)
        active = ~position_valid & ~velocity_valid
        escape = active & self.has_ttc & (remaining <= self.cfg.guard_ttc_threshold_s)
        brake = active & ~escape
        result = desired.clone()
        # Forward targets only: retain goal-heading assistance and downstream slew.
        result[brake, 0] = 0.
        result[escape, 0] = self.cfg.guard_escape_speed_mps
        unknown = torch.full_like(remaining, -1.)
        diagnostics = dict(
            missing_position=~position_valid, missing_velocity=~velocity_valid,
            missing_both=active, guard_active=active, brake_requested=brake,
            escape_requested=escape, blocked=brake,
            ttc_memory_valid=self.has_ttc.clone(),
            ttc_memory_age_s=torch.where(self.has_ttc, self.ttc_age, unknown),
            estimated_ttc_s=torch.where(self.has_ttc, remaining, unknown),
            proposed_vx=desired[:, 0].clone(), guarded_vx=result[:, 0].clone())
        return result, diagnostics
