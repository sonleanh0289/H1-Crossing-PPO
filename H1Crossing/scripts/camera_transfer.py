"""Explicit camera warm-start rules. Never relax exact resume validation."""
from copy import deepcopy
from camera_perception import DETECTOR_ALGORITHMS, EXTRA_FIELDS, VELOCITY_VALIDITY_MODES


def validate_camera_transfer(data, current, behavior_fields):
    old = data["compatibility"]
    expected = list(behavior_fields)+list(EXTRA_FIELDS)
    if old["observation_fields"] != expected or current["observation_fields"] != expected:
        raise ValueError("Camera observation schema differs")
    for key in ("action", "reward_semantics", "rewards", "locomotion_policy_sha256",
                "locomotion_checkpoint_sha256", "locomotion_contract", "finite_horizon"):
        if old[key] != current[key]:
            raise ValueError(f"Warm-start incompatible {key}")
    remaining = []
    for item in (old, current):
        cfg = deepcopy(item["perception"])
        mode = cfg.pop("detector_mode", "green")
        if mode not in DETECTOR_ALGORITHMS or item["perception_algorithm"] != DETECTOR_ALGORITHMS[mode]:
            raise ValueError("Warm-start incompatible perception_algorithm")
        if tuple(cfg.pop("camera_names", ("front", "left"))) not in (("front", "left"), ("front", "left", "right")):
            raise ValueError("Unsupported camera rig")
        cfg.pop("frequency_hz")
        if cfg.pop("velocity_validity", "legacy_hits_v1") not in VELOCITY_VALIDITY_MODES:
            raise ValueError("Unsupported velocity validity mode")
        remaining.append(cfg)
    if remaining[0] != remaining[1]:
        raise ValueError("Warm-start incompatible perception geometry/tracker settings")
    remaining = []
    for item in (old, current):
        cfg = deepcopy(item["settings"])
        for key in ("learning_rate", "learning_epochs", "mini_batches", "entropy_coefficient",
                    "init_std", "rollout_steps", "save_interval"):
            cfg.pop(key)
        remaining.append(cfg)
    if remaining[0] != remaining[1]:
        raise ValueError("Warm-start incompatible environment/control settings")
    for name in ("actor", "critic"):
        if old["ppo"][name] != current["ppo"][name]:
            raise ValueError(f"Warm-start incompatible {name} architecture")
