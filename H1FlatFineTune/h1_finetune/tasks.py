"""Isaac Lab 3.0 external tasks. Import ONLY after AppLauncher."""
import math
import gymnasium as gym
import torch
from isaaclab.envs import mdp
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.flat_env_cfg import H1FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.agents.rsl_rl_ppo_cfg import H1FlatPPORunnerCfg
from .control import sample_commands, clamp_commands, slew, STAGES


class SmoothVelocityCommand(mdp.UniformVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.target = torch.zeros_like(self.vel_command_b)
        self.manual = False
        self.dt = env.step_dt

    def _resample_command(self, env_ids):
        if self.manual:
            return
        self.target[env_ids], _ = sample_commands(
            len(env_ids), self.cfg.stage, self.device, profile=self.cfg.command_profile)

    def _update_command(self):
        self.vel_command_b[:] = slew(self.vel_command_b, self.target, self.dt)

    def reset(self, env_ids=None):
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids or slice(None)]
        self.vel_command_b[env_ids] = 0
        self.target[env_ids] = 0
        return super().reset(env_ids)

    def set_desired_velocity(self, vx, vy=0.0, yaw_rate=0.0, env_ids=None):
        """Body-frame command, filtered at 50 Hz; vy is intentionally disabled."""
        if vy != 0:
            raise ValueError("Run/retreat task requires vy=0")
        self.manual = True
        self.target[env_ids if env_ids is not None else slice(None)] = clamp_commands(
            self.target.new_tensor([vx, vy, yaw_rate]), self.cfg.stage)


@configclass
class SmoothVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    class_type: type = SmoothVelocityCommand
    stage: str = "A"
    command_profile: str = "baseline"


@configclass
class RunRetreatCfg(H1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Keep the native flat asset, observations, action mapping and actuators untouched.
        self.scene.num_envs = 128
        self.scene.env_spacing = 4.0
        self.sim.dt = 0.005
        self.decimation = 4
        self.episode_length_s = 20.0
        self.scene.robot.spawn.articulation_props.fix_root_link = False
        self.commands.base_velocity = SmoothVelocityCommandCfg(
            asset_name="robot", resampling_time_range=(4.0, 8.0),
            heading_command=False, rel_heading_envs=0.0, rel_standing_envs=0.0, debug_vis=False,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=STAGES["A"], lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.5, 0.5), heading=None))
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.events.reset_base.params["velocity_range"] = {
            name: (0.0, 0.0) for name in ("x", "y", "z", "roll", "pitch", "yaw")}
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (-math.pi, math.pi)}
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.feet_air_time.params["threshold"] = 0.4


@configclass
class RunRetreatPlayCfg(RunRetreatCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.viewer.eye = (7.0, -7.0, 4.0)
        self.viewer.lookat = (0.0, 0.0, 1.0)


@configclass
class RunRetreatPPORunnerCfg(H1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "h1_flat_run_retreat"
        self.max_iterations = 3000
        self.save_interval = 500
        self.obs_groups = {"actor": ["policy"], "critic": ["policy"]}
        self.actor.distribution_cfg.init_std = 0.3
        self.algorithm.learning_rate = 1e-4
        self.algorithm.schedule = "fixed"
        self.algorithm.entropy_coef = 0.005
        self.algorithm.num_learning_epochs = 5
        self.logger = "tensorboard"


for task_id, cfg_name in (("H1-Velocity-Flat-RunRetreat-v0", "RunRetreatCfg"),
                          ("H1-Velocity-Flat-RunRetreat-Play-v0", "RunRetreatPlayCfg")):
    if task_id not in gym.registry:
        gym.register(id=task_id, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
                     kwargs={"env_cfg_entry_point": f"{__name__}:{cfg_name}",
                             "rsl_rl_cfg_entry_point": f"{__name__}:RunRetreatPPORunnerCfg"})
