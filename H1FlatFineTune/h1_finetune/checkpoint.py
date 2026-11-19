"""Safe checkpoint I/O. Does not import Isaac Sim or launch a training loop."""
from copy import deepcopy
import json
from pathlib import Path
import torch
import math
from .control import STAGES
from .artifacts import output_path

MODEL_CONTRACT = {"observation_dim": 69, "action_dim": 19, "hidden_dims": [128, 128, 128],
                  "activation": "elu", "obs_normalization": False, "action_scale": 0.5,
                  "use_default_offset": True, "physics_dt": 0.005, "policy_dt": 0.02,
                  "observation_terms": ["base_lin_vel", "base_ang_vel", "projected_gravity",
                                        "velocity_commands", "joint_pos", "joint_vel", "actions"]}


def load_safe(path, device="cpu"):
    data = torch.load(path, map_location=device, weights_only=True)
    for group, output in (("actor_state_dict", 19), ("critic_state_dict", 1)):
        state = data.get(group, {})
        expected = {}
        for i, ins, outs in ((0, 69, 128), (2, 128, 128), (4, 128, 128), (6, 128, output)):
            expected[f"mlp.{i}.weight"] = (outs, ins)
            expected[f"mlp.{i}.bias"] = (outs,)
        if output == 19:
            expected["distribution.std_param"] = (19,)
        if set(state) != set(expected):
            raise ValueError(f"{group}: incompatible keys/normalizer/architecture")
        for name, shape in expected.items():
            value = state[name]
            if tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError(f"{group}/{name}: invalid shape or non-finite weights")
    if (data["actor_state_dict"]["distribution.std_param"] <= 0).any():
        raise ValueError("Gaussian exploration standard deviation must be positive")
    contract = data.get("fine_tune", {}).get("contract")
    if contract is not None and any(contract.get(key) != value for key, value in MODEL_CONTRACT.items()):
        raise ValueError("Checkpoint metadata violates the native flat model contract")
    return data


def initialize(alg, data, *, learning_rate=1e-4, init_std=0.3):
    """Warm start actor/critic only. Never restore the base optimizer or iteration."""
    if any(not math.isfinite(v) or v <= 0 for v in (learning_rate, init_std)):
        raise ValueError("Warm-start learning rate and std must be finite and positive")
    alg.actor.load_state_dict(data["actor_state_dict"], strict=True)
    alg.critic.load_state_dict(data["critic_state_dict"], strict=True)
    with torch.no_grad():
        alg.actor.distribution.std_param.fill_(init_std)
    alg.optimizer.state.clear()
    alg.learning_rate = learning_rate
    for group in alg.optimizer.param_groups:
        group["lr"] = alg.learning_rate


def resume_command_range(state, stage, checkpoint_path, allow_change=False):
    """Require explicit consent for a same-stage command-range change.

    Legacy checkpoints resolve their original range from adjacent saved env_config.
    This check never overwrites weights, optimizer state, or learned exploration std.
    """
    previous = state.get("command_range")
    if previous is None:
        config_path = Path(checkpoint_path).parent / "env_config.json"
        if config_path.is_file():
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            previous = cfg["commands"]["base_velocity"]["ranges"]["lin_vel_x"]
    current = list(STAGES[stage])
    if previous is not None:
        previous = list(previous)
    if previous != current and not allow_change:
        raise ValueError(f"Resume command range changed or unknown: {previous} -> {current}. "
                         "Use --allow_command_range_change to acknowledge a new training objective.")
    return previous, current


def restore(alg, data, contract):
    state = data.get("fine_tune")
    if not state or state.get("schema_version") != 1 or state.get("contract") != contract:
        raise ValueError("Resume requires a matching fine-tune checkpoint contract")
    if state["completed_iterations"] != data["iter"] or state["stage_iterations"] < 0:
        raise ValueError("Invalid iteration state")
    if (state["stage"] not in STAGES or not 0 <= state["stage_iterations"]
            or state["completed_iterations"] < state["stage_iterations"]
            or not math.isfinite(state["learning_rate"]) or state["learning_rate"] <= 0):
        raise ValueError("Invalid stage, iteration or learning-rate state")
    alg.actor.load_state_dict(data["actor_state_dict"], strict=True)
    alg.critic.load_state_dict(data["critic_state_dict"], strict=True)
    alg.optimizer.load_state_dict(data["optimizer_state_dict"])
    alg.learning_rate = state["learning_rate"]
    if any(group["lr"] != alg.learning_rate for group in alg.optimizer.param_groups):
        raise ValueError("Optimizer LR differs from checkpoint metadata")
    return deepcopy(state)


def save(alg, path, state):
    data = alg.save()
    data.update(iter=state["completed_iterations"], infos={}, fine_tune=deepcopy(state))
    torch.save(data, output_path(path))
