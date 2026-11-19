"""Runtime construction and executable contracts (import after AppLauncher)."""
from copy import deepcopy
import inspect
import json
import math
from importlib.metadata import version
import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.flat_env_cfg import H1FlatEnvCfg
from .tasks import RunRetreatCfg, RunRetreatPlayCfg, RunRetreatPPORunnerCfg
from .checkpoint import MODEL_CONTRACT
from .control import STAGES, resolve_command_profile
from .artifacts import sha256, write_json, provenance
from .numerics import finite


def plain(value):
    """JSON representation of resolved configs, including callable qualified names."""
    if hasattr(value, "to_dict") and not isinstance(value, type):
        return plain(value.to_dict())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if callable(value):
        return f"{value.__module__}:{value.__qualname__}"
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return str(value)


def validate_contract(env, cfg):
    native = H1FlatEnvCfg()
    robot = env.scene["robot"]
    action = env.action_manager.get_term("joint_pos")
    policy = cfg.observations.policy
    expected = MODEL_CONTRACT["observation_terms"]
    active = env.observation_manager.active_terms["policy"]
    checks = {
        "plane": cfg.scene.terrain.terrain_type == "plane" and cfg.scene.terrain.terrain_generator is None,
        "no_height_scan": cfg.scene.height_scanner is None and policy.height_scan is None,
        "observation_order": list(active) == expected,
        "observation_dim": tuple(env.observation_manager.group_obs_dim["policy"]) == (69,),
        "action_dim": env.action_manager.total_action_dim == 19,
        "action_config": plain(cfg.actions.to_dict()) == plain(native.actions.to_dict()),
        "joint_order": list(action._joint_names) == list(robot.joint_names) and len(robot.joint_names) == 19,
        "action_scale": action._scale == 0.5,
        "default_offset": torch.equal(action._offset, robot.data.default_joint_pos.torch),
        "actuators": plain(cfg.scene.robot.actuators) == plain(native.scene.robot.actuators),
        "asset": cfg.scene.robot.spawn.usd_path == native.scene.robot.spawn.usd_path,
        "free_root": cfg.scene.robot.spawn.articulation_props.fix_root_link is False,
        "timing": cfg.sim.dt == 0.005 and cfg.decimation == 4,
    }
    # Noise can differ in evaluation; functions, order, params and scales cannot.
    for name in expected:
        for key in ("func", "params", "scale", "clip"):
            checks[f"obs_{name}_{key}"] = plain(getattr(getattr(policy, name), key)) == plain(
                getattr(getattr(native.observations.policy, name), key))
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"Native flat checkpoint contract failed: {failed}")
    contract = deepcopy(MODEL_CONTRACT)
    contract.update(joint_names=list(robot.joint_names),
                    default_joint_positions=robot.data.default_joint_pos.torch[0].cpu().tolist(),
                    action_config=plain(cfg.actions.to_dict()),
                    observation_config={name: {key: plain(getattr(getattr(policy, name), key))
                                               for key in ("func", "params", "scale", "clip")} for name in expected},
                    actuator_config=plain(cfg.scene.robot.actuators),
                    asset_url=cfg.scene.robot.spawn.usd_path,
                    native_config_sha256=sha256(inspect.getfile(H1FlatEnvCfg)))
    return contract


def build(args, training=False):
    cfg = RunRetreatCfg() if training else RunRetreatPlayCfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.commands.base_velocity.stage = args.stage
    cfg.commands.base_velocity.ranges.lin_vel_x = STAGES[args.stage]
    if training:
        cfg.commands.base_velocity.command_profile = resolve_command_profile(
            getattr(args, "command_profile", None))
    agent = RunRetreatPPORunnerCfg()
    if training:
        agent.max_iterations = args.max_iterations
        from .training_profiles import apply_training_profile
        apply_training_profile(cfg, agent, getattr(args, "training_profile", None) or "baseline")
    agent = handle_deprecated_rsl_rl_cfg(agent, version("rsl-rl-lib"))
    agent.seed, agent.device = args.seed, args.device
    task = "H1-Velocity-Flat-RunRetreat-v0" if training else "H1-Velocity-Flat-RunRetreat-Play-v0"
    base = gym.make(task, cfg=cfg).unwrapped
    env = RslRlVecEnvWrapper(base, clip_actions=agent.clip_actions)
    contract = validate_contract(base, cfg)
    runner = OnPolicyRunner(env, deepcopy(agent.to_dict()), log_dir=None, device=args.device)
    return base, env, runner, contract, cfg, agent


def record_run(directory, args, checkpoint, contract, cfg, agent):
    write_json(directory / "provenance.json", provenance(args.seed, checkpoint))
    write_json(directory / "contract.json", contract)
    write_json(directory / "env_config.json", plain(cfg.to_dict()))
    write_json(directory / "ppo_config.json", plain(agent.to_dict()))
    write_json(directory / "arguments.json", plain(vars(args)))
