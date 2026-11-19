"""Stateful stability reward. Import only after AppLauncher."""
import math
import torch
from isaaclab.managers import ManagerTermBase


class TorsoAngularAccelerationPenalty(ManagerTermBase):
    """Squared world-horizontal angular acceleration at the policy rate.

    Samples match evaluation's (omega[t] - omega[t-1]) / env.step_dt, NOT the
    200 Hz physics substep. Intended yaw acceleration around world Z is excluded.
    First sample after each individual episode reset contributes zero; normal
    stop/reverse/turn commands are not masked. No filter or clipping hides spikes.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if not math.isfinite(env.step_dt) or env.step_dt <= 0:
            raise ValueError("Expected a finite positive policy timestep")
        scale = cfg.params["scale_radps2"]
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Angular acceleration scale must be finite and positive")
        entity = cfg.params["asset_cfg"]
        angular = env.scene[entity.name].data.body_ang_vel_w.torch[:, entity.body_ids, :]
        if angular.shape != (env.num_envs, 1, 3):
            raise ValueError("Acceleration reward must select exactly one torso_link body")
        self.previous_xy = torch.zeros((env.num_envs, 2), device=env.device)
        self.valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.valid[ids] = False
        self.previous_xy[ids] = 0.

    def __call__(self, env, asset_cfg, scale_radps2):
        angular_xy = env.scene[asset_cfg.name].data.body_ang_vel_w.torch[:, asset_cfg.body_ids, :2][:, 0]
        acceleration = (angular_xy - self.previous_xy) / env.step_dt
        penalty = (acceleration / scale_radps2).square().sum(dim=-1)
        result = torch.where(self.valid, penalty, torch.zeros_like(penalty))
        self.previous_xy.copy_(angular_xy)
        self.valid.fill_(True)
        return result
