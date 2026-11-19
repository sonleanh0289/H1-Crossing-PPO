"""Opt-in training objectives. Baseline and inference configs remain unchanged."""
from copy import deepcopy

TRAINING_PROFILES = ("baseline", "stability_v1", "stability_v2")
STABILITY_PROFILES = ("stability_v1", "stability_v2")
STABILITY_V1 = {
    "learning_rate": 5e-5,
    "entropy_coef": 0.001,
    "init_std": 0.15,
    "num_learning_epochs": 5,
    "ang_vel_xy_l2": -0.10,
    "action_rate_l2": -0.01,
    "torso_ang_vel_xy_l2": -0.05,
}

# Keep v1 immutable for existing checkpoints/full resumes. A new objective needs
# a new warm start; normalization is a reward scale, NOT an acceleration limit.
STABILITY_V2 = {
    **STABILITY_V1,
    "learning_rate": 2.5e-5,
    "action_rate_l2": -0.015,
    "torso_ang_acc_xy_l2": -0.05,
    "angular_acceleration_scale_radps2": 10.0,
}


def resolve_training_profile(requested=None, state=None):
    saved = (state or {}).get("training_profile", "baseline")
    name = requested or saved
    if name not in TRAINING_PROFILES:
        raise ValueError(f"Unknown training profile: {name}")
    if state is not None and name != saved:
        raise ValueError("Changing reward/PPO profile requires --init_checkpoint (new optimizer/iteration), not --resume")
    if state is not None and name != "baseline" and state.get("training_profile_settings") != profile_settings(name):
        raise ValueError("Saved stability profile differs from code; use an explicit new warm start")
    return name


def profile_settings(name):
    if name not in TRAINING_PROFILES:
        raise ValueError(f"Unknown training profile: {name}")
    return deepcopy({"baseline": {}, "stability_v1": STABILITY_V1, "stability_v2": STABILITY_V2}[name])


def torso_ang_vel_xy_l2(env, asset_cfg):
    """Penalize torso rotation about WORLD horizontal axes, not intended yaw."""
    angular = env.scene[asset_cfg.name].data.body_ang_vel_w.torch[:, asset_cfg.body_ids, :2]
    return angular.square().sum(dim=-1).mean(dim=-1)


def apply_training_profile(cfg, agent, name):
    settings = profile_settings(name)
    if not settings:
        return
    from isaaclab.managers import RewardTermCfg, SceneEntityCfg
    agent.algorithm.learning_rate = settings["learning_rate"]
    agent.algorithm.entropy_coef = settings["entropy_coef"]
    agent.algorithm.num_learning_epochs = settings["num_learning_epochs"]
    agent.algorithm.schedule = "fixed"
    agent.actor.distribution_cfg.init_std = settings["init_std"]
    cfg.rewards.ang_vel_xy_l2.weight = settings["ang_vel_xy_l2"]
    cfg.rewards.action_rate_l2.weight = settings["action_rate_l2"]
    cfg.rewards.torso_ang_vel_xy_l2 = RewardTermCfg(
        func=torso_ang_vel_xy_l2, weight=settings["torso_ang_vel_xy_l2"],
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"])})
    if name == "stability_v2":
        from .stability_rewards import TorsoAngularAccelerationPenalty
        cfg.rewards.torso_ang_acc_xy_l2 = RewardTermCfg(
            func=TorsoAngularAccelerationPenalty, weight=settings["torso_ang_acc_xy_l2"],
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
                    "scale_radps2": settings["angular_acceleration_scale_radps2"]})
