"""Explicit train/play CLI for state or camera crossing PPO. No work at import time."""
import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
import json
import math
from pathlib import Path
import random
import sys
import time
import warnings
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "H1FlatFineTune"))
from crossing_bundle import validate_bundle, sha256
from crossing_rewards import load_reward_config, DEFAULT_CONFIG
from behavior_control import load_behavior_config, speed_from_action, CONFIG, BEHAVIOR_FIELDS
from h1_finetune.control import STAGES
from h1_finetune.validation import gate
from vehicle_motion import TrafficConfig


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def parse_args(mode, argv=None):
    parser = argparse.ArgumentParser(description="Frozen H1 locomotion + separate crossing behavior PPO")
    parser.add_argument("--bundle", type=Path, default=ROOT / "checkpoints/h1_flat_stability_v2_0500")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--observation_source", choices=("state", "groundtruth23", "camera", "camera_radar"), default="state")
    parser.add_argument("--asymmetric_critic", action="store_true",
                        help="Camera/radar actor 23D; training critic 29D with six current GT fields")
    parser.add_argument("--perception_config", type=Path, default=ROOT / "configs/crossing_perception.yaml")
    parser.add_argument("--perception_debug", action="store_true")
    parser.add_argument("--radar_shadow", action="store_true",
                        help="Play only: diagnostic torso radar rig; policy remains camera-only")
    parser.add_argument("--radar_config", type=Path, default=ROOT / "configs/crossing_radar.yaml")
    parser.add_argument("--init_checkpoint", type=Path, help="Warm-start 19D/22D -> camera 22D or camera_radar 23D; keep std, new optimizer/iteration 0")
    parser.add_argument("--reward_config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--traffic_mode", choices=("variable", "constant"), default="variable",
                        help="Bounded acceleration by default; constant preserves old traffic")
    parser.add_argument("--yield_probability", type=float, default=.5,
                        help="Probability of a yielding driver per spawn, from 0 to 1")
    parser.add_argument("--vehicle_palette", choices=("legacy_green", "multicolor"), default="legacy_green",
                        help="Multicolor cycles seven body/cabin colors per pass; use the multicolor perception config")
    if mode == "train":
        parser.add_argument("--fresh_behavior", action="store_true",
                            help="New random crossing actor/critic and optimizer at iteration 0; frozen locomotion bundle only")
        parser.add_argument("--max_iterations", type=int, help="Total completed-iteration target, including resume history")
        parser.add_argument("--smoke_test", action="store_true", help="4 envs x 2 iterations plus physical event checks")
        parser.add_argument("--resume", type=Path, help="Behavior checkpoint only; not the locomotion checkpoint")
    else:
        parser.add_argument("--trajectory", action="store_true", help="GT23 or asymmetric play: record decisions and 50Hz motion before resets")
        parser.add_argument("--keyboard", action="store_true", help="Manual GUI play; no crossing policy/checkpoint")
        parser.add_argument("--keyboard_speed", type=float, default=1.0)
        parser.add_argument("--keyboard_reverse_speed", type=float, default=1.0)
        parser.add_argument("--checkpoint", type=Path)
        parser.add_argument("--oracle_observations", action="store_true",
                            help="Evaluation only: replace sensor inputs by truth in the same 23D schema")
        parser.add_argument("--ground_truth_eval", action="store_true",
                            help="Play-only read-only scoring of camera/radar actor inputs against simulator truth")
        parser.add_argument("--steps", type=int, default=1000, help="Behavior steps at 10Hz; -1 runs until closed")
    args = parser.parse_args(argv)
    if args.asymmetric_critic:
        if args.observation_source != "camera_radar" or args.init_checkpoint or getattr(args, "oracle_observations", False) or getattr(args, "keyboard", False):
            parser.error("--asymmetric_critic requires camera_radar, fresh/matching resume, no oracle or keyboard")
        args.config = args.config or ROOT / "configs/crossing_behavior_asymmetric.yaml"
    if getattr(args, "trajectory", False) and ((args.observation_source != "groundtruth23" and not args.asymmetric_critic) or args.keyboard):
        parser.error("--trajectory requires groundtruth23 or asymmetric checkpoint play, without keyboard")
    if args.observation_source == "groundtruth23":
        args.config = args.config or ROOT / "configs/crossing_behavior_groundtruth.yaml"
        args.num_envs = args.num_envs or (8 if mode == "train" else 1)
        if args.perception_debug or args.radar_shadow or args.init_checkpoint:
            parser.error("groundtruth23 uses no sensors and accepts fresh training or matching --resume only")
    if getattr(args, "fresh_behavior", False) and (args.init_checkpoint or args.resume):
        parser.error("--fresh_behavior excludes --init_checkpoint and --resume")
    if args.radar_shadow and (mode != "play" or args.observation_source != "camera" or args.num_envs not in (None, 1)):
        parser.error("--radar_shadow is play-only and requires camera observations with one environment")
    if args.observation_source == "camera_radar":
        if getattr(args, "smoke_test", False):
            parser.error("Use --max_iterations 2 for radar training smoke, not the camera-only 4-env preset")
        if args.config is None:
            args.config = ROOT / "configs/crossing_behavior_camera_radar.yaml"
    args.config = args.config or (ROOT / "configs/crossing_behavior_camera.yaml" if args.observation_source != "state" else CONFIG)
    if args.init_checkpoint and (args.observation_source == "state" or getattr(args, "resume", None) or getattr(args, "checkpoint", None)):
        parser.error("--init_checkpoint requires camera and excludes --resume/--checkpoint")
    if mode == "play":
        if args.oracle_observations:
            if args.observation_source != "camera_radar" or args.keyboard or not args.checkpoint:
                parser.error("--oracle_observations requires camera_radar --checkpoint play without keyboard")
            args.ground_truth_eval = True
        if args.ground_truth_eval and (args.observation_source != "camera_radar" or args.keyboard):
            parser.error("--ground_truth_eval requires camera_radar policy play, without --keyboard")
        if args.keyboard:
            if args.checkpoint or args.init_checkpoint:
                parser.error("--keyboard excludes crossing --checkpoint/--init_checkpoint; locomotion bundle is retained")
            if args.headless or sys.platform != "win32" or args.num_envs not in (None, 1):
                parser.error("--keyboard requires the Windows GUI and one environment")
            if any(not math.isfinite(v) or v <= 0 for v in (args.keyboard_speed, args.keyboard_reverse_speed)):
                parser.error("Keyboard speeds must be finite and positive")
        elif not (args.checkpoint or args.init_checkpoint):
            parser.error("Play requires --checkpoint, --init_checkpoint or --keyboard")
    if args.perception_debug and args.observation_source == "state":
        parser.error("--perception_debug requires --observation_source camera")
    if not math.isfinite(args.yield_probability) or not 0 <= args.yield_probability <= 1:
        parser.error("--yield_probability must be between 0 and 1")
    if mode == "train":
        if args.observation_source in ("camera", "camera_radar", "groundtruth23"):
            if args.perception_debug:
                parser.error("Camera training is headless without debug panels; use play for perception_debug")
            if not (args.init_checkpoint or args.resume or args.fresh_behavior):
                parser.error("Choose --fresh_behavior, --init_checkpoint or --resume explicitly")
            args.headless = True
        if not args.smoke_test and args.max_iterations is None:
            parser.error("Specify --max_iterations or --smoke_test; no simulator/training started")
        if args.smoke_test:
            if args.resume or args.max_iterations is not None or args.num_envs not in (None, 4):
                parser.error("--smoke_test uses a fresh 4-env, 2-iteration run")
            args.num_envs, args.max_iterations = 4, 2
        if args.max_iterations <= 0:
            parser.error("--max_iterations must be positive")
    elif args.steps == 0 or args.steps < -1:
        parser.error("--steps must be -1 or positive")
    if args.num_envs is not None and args.num_envs <= 0:
        parser.error("--num_envs must be positive")
    return args


def ppo_config(bundle, settings, *, asymmetric_critic=False):
    config = json.loads((bundle / "ppo_config.json").read_text(encoding="utf-8"))
    config["obs_groups"] = {"actor": ["behavior"], "critic": ["behavior"]}
    if asymmetric_critic:
        config["obs_groups"]["critic"] = ["critic"]
    config["num_steps_per_env"] = settings.rollout_steps
    config["multi_gpu"] = None
    for key in ("actor", "critic"):
        config[key]["hidden_dims"] = [128, 128]
        config[key]["obs_normalization"] = False
    config["actor"]["distribution_cfg"]["init_std"] = settings.init_std
    if settings.action_distribution == "beta":
        config["actor"]["distribution_cfg"] = {
            "class_name": "crossing_action_distribution:BoundedBetaDistribution", "init_std": settings.init_std}
    config["algorithm"].update(learning_rate=settings.learning_rate, schedule="fixed",
        num_learning_epochs=settings.learning_epochs, num_mini_batches=settings.mini_batches,
        entropy_coef=settings.entropy_coefficient)
    return config


def compatibility(settings, rewards, manifest, config, traffic_config=TrafficConfig(), perception_config=None,
                  radar_observation_config=None, groundtruth23=False, asymmetric_critic=False):
    result = {"format": "h1-crossing-behavior-v1", "observation_fields": list(BEHAVIOR_FIELDS),
            "reward_semantics": "goal-only-safe-bonus-signed-progress-and-boundary-v3",
            "action": "one scalar: negative -> retreat to -1m/s, positive -> forward to 3m/s; goal heading assist",
            "finite_horizon": True, "settings": asdict(settings), "rewards": asdict(rewards),
            "locomotion_checkpoint_sha256": manifest["checkpoint_sha256"],
            "locomotion_policy_sha256": manifest["policy_sha256"], "locomotion_contract": manifest["contract"],
            "ppo": config, "traffic": traffic_config.contract()}
    if perception_config is not None:
        from camera_perception import EXTRA_FIELDS, DETECTOR_ALGORITHMS
        result.update(format="h1-crossing-camera-v1", observation_fields=list(BEHAVIOR_FIELDS)+list(EXTRA_FIELDS),
            perception=asdict(perception_config), perception_algorithm=DETECTOR_ALGORITHMS[perception_config.detector_mode])
    if radar_observation_config is not None:
        if perception_config is None:
            raise ValueError("Radar observations require camera position inputs")
        from radar_observations import FORMAT, DIRECTION_FIELD, observation_contract
        result.update(format=FORMAT, radar=observation_contract(radar_observation_config))
        result["observation_fields"].append(DIRECTION_FIELD)
    if groundtruth23:
        if perception_config is not None or radar_observation_config is not None:
            raise ValueError("Groundtruth23 contract cannot contain sensors")
        from oracle_observations import FORMAT, observation_fields, input_contract
        result.update(format=FORMAT, observation_fields=observation_fields(), groundtruth=input_contract())
    if asymmetric_critic:
        if groundtruth23 or perception_config is None or radar_observation_config is None:
            raise ValueError("Asymmetric contract requires camera/radar actor inputs")
        if config["obs_groups"] != {"actor": ["behavior"], "critic": ["critic"]}:
            raise ValueError("Asymmetric PPO must isolate actor and critic observation groups")
        from asymmetric_observations import critic_contract
        result.update(critic_contract(result["observation_fields"]))
    return result


def align_behavior_revision(saved, current, *, allow_changes):
    """Align only the known v2 -> v3 rollout change for play/explicit warm-start.

    The caller owns a deep copy. Other reward, schema, sensor, traffic and frozen
    locomotion differences remain subject to their existing compatibility checks.
    """
    changes = []
    # Old checkpoints predate the explicit Gaussian/Beta selector. Never migrate
    # between distributions implicitly, even during play or warm-start.
    for contract in (saved, current):
        if "settings" in contract:
            contract["settings"].setdefault("action_distribution", "gaussian")
    # Guard changes are explicit play/warm-start migrations, never exact resume.
    if "settings" in saved and "settings" in current:
        from behavior_control import BehaviorConfig
        defaults = asdict(BehaviorConfig())
        fields = [k for k in defaults if k.startswith("guard_")]
        for field in fields:
            if field in current["settings"]:
                saved["settings"].setdefault(field, defaults[field])
        if any(saved["settings"].get(k) != current["settings"].get(k) for k in fields):
            changes.append("crossing command guard policy/settings")
            if allow_changes:
                for field in fields:
                    if field in current["settings"]:
                        saved["settings"][field] = current["settings"][field]
    if "perception" in saved and "perception" in current and "motion_gate" in current["perception"]:
        saved["perception"].setdefault("motion_gate", 0)
        if saved["perception"]["motion_gate"] != current["perception"]["motion_gate"]:
            changes.append("camera measurement motion gate and confirmed births")
            if allow_changes:
                saved["perception"]["motion_gate"] = current["perception"]["motion_gate"]
    if saved.get("format") == current.get("format") and "perception" in saved and "perception" in current:
        from camera_perception import DETECTION_FILTER, TRACK_LIFECYCLE, TRACK_QUALITY, TRACK_QUALITY_FIELDS
        from camera_reset_barrier import RESET_CAPTURE_POLICY
        old_perception, new_perception = saved["perception"], current["perception"]
        if new_perception.get("track_quality") == "legacy_v1":
            old_quality = old_perception.get("track_quality", "legacy_v1")
            if old_quality == TRACK_QUALITY:
                changes.append("restored continuity-first camera track quality (legacy_v1)")
                if allow_changes:
                    for field in TRACK_QUALITY_FIELDS:
                        old_perception[field] = new_perception[field]
            elif old_quality == "legacy_v1":
                # Older contracts omit the quality revision and its inactive
                # tuning fields. Normalize missing metadata without changing
                # legacy tracking behavior or relaxing other resume checks.
                for field in TRACK_QUALITY_FIELDS:
                    old_perception.setdefault(field, new_perception[field])
        if (old_perception.get("track_quality", "legacy_v1") == "legacy_v1"
                and new_perception.get("track_quality") == TRACK_QUALITY):
            changes.append("confirmed camera births and uncertainty-limited prediction")
            if allow_changes:
                for field in TRACK_QUALITY_FIELDS:
                    old_perception[field] = new_perception[field]
        if ("reset_capture_policy" not in old_perception
                and new_perception.get("reset_capture_policy") == RESET_CAPTURE_POLICY):
            changes.append("camera post-reset scheduled-capture drain")
            if allow_changes:
                old_perception["reset_capture_policy"] = RESET_CAPTURE_POLICY
        if ("detection_filter" not in old_perception
                and new_perception.get("detection_filter") == DETECTION_FILTER):
            changes.append("mapped-road and pre-update camera innovation rejection")
            if allow_changes:
                for field in ("detection_filter", "road_margin_m", "max_innovation_m", "max_innovation_nis"):
                    old_perception[field] = new_perception[field]
        if ("track_lifecycle" not in old_perception
                and new_perception.get("track_lifecycle") == TRACK_LIFECYCLE):
            changes.append("single camera track retained until visual loss; no respawn-triggered camera reset")
            if allow_changes:
                old_perception["track_lifecycle"] = TRACK_LIFECYCLE
    if (saved.get("reward_semantics") == "goal-only-safe-bonus-and-signed-progress-v2"
            and current.get("reward_semantics") == "goal-only-safe-bonus-signed-progress-and-boundary-v3"
            and "out_of_bounds_penalty" not in saved.get("rewards", {})):
        changes.append("one-shot out-of-bounds penalty and terminal priority")
        if allow_changes:
            saved["reward_semantics"] = current["reward_semantics"]
            saved["rewards"]["out_of_bounds_penalty"] = current["rewards"]["out_of_bounds_penalty"]
    if saved.get("format") == current.get("format") == "h1-crossing-camera-radar-v1":
        previous = saved.get("radar", {})
        target = current.get("radar", {})
        from radar_observations import VELOCITY_SOURCE
        if (previous.get("velocity_source") == "radar_only_world_x_lane_assumption; vy=0"
                and target.get("velocity_source") == VELOCITY_SOURCE):
            changes.append("valid camera Kalman velocity now backs up unavailable radar velocity")
            if allow_changes:
                previous["velocity_source"] = VELOCITY_SOURCE
        # Legacy one-head checkpoints predate these inactive defaults. Adding
        # them does not change the one-head sensor and is safe for exact resume.
        from radar_processing import RadarConfig
        defaults = asdict(RadarConfig())
        for contract in (previous, target):
            for field in ("opposite_radar", "opposite_torso_offset", "max_head_disagreement_mps"):
                contract.get("sensor", {}).setdefault(field, defaults[field])
        rig_fields = ("opposite_radar", "opposite_torso_offset", "max_head_disagreement_mps",
                      "horizontal_fov_deg", "vertical_fov_deg")
        if any(previous.get("sensor", {}).get(k) != target.get("sensor", {}).get(k) for k in rig_fields):
            changes.append("radar field of view/complementary head changed (23D schema unchanged)")
            if allow_changes:
                for field in rig_fields:
                    previous["sensor"][field] = target["sensor"][field]
        if (previous.get("velocity_required_for_autonomy", True) is True
                and target.get("velocity_required_for_autonomy") is False):
            changes.append("radar velocity dropout no longer forces a stop")
            if allow_changes:
                previous["velocity_required_for_autonomy"] = False
    if changes:
        if not allow_changes:
            raise ValueError("Crossing rollout/reward behavior changed; exact --resume is incompatible. "
                             "Use --init_checkpoint for an explicit warm-start with a new optimizer and iteration 0.")
        warnings.warn("Applying updated crossing behavior: " + "; ".join(changes)
                      + ". This is play/warm-start, not an exact training resume; evaluate the policy again.", UserWarning)


def check_compatibility(saved, current, *, training):
    saved, current = deepcopy(saved), deepcopy(current)
    align_behavior_revision(saved, current, allow_changes=not training)
    # Checkpoint cadence is an I/O setting, not a policy/optimizer contract.
    # Permit changing it on resume/play while retaining all learning checks.
    for item in (saved, current):
        item.get("settings", {}).pop("save_interval", None)
    old_traffic = saved.pop("traffic", TrafficConfig(mode="constant", yield_probability=0).contract())
    new_traffic = current.pop("traffic", TrafficConfig(mode="constant", yield_probability=0).contract())
    # Legacy checkpoints have no camera_names field and used front/left only.
    # Canonicalize without mutating checkpoint data. Training stays strict.
    for item in (saved, current):
        if item.get("format") == "h1-crossing-camera-v1" and "perception" in item:
            item["perception"]["camera_names"] = tuple(item["perception"].get("camera_names", ("front", "left")))
            item["perception"].setdefault("detector_mode", "green")
            item["perception"].setdefault("velocity_validity", "legacy_hits_v1")
    # Play may probe rate/rig changes; this is not an exact training resume.
    if not training and saved.get("format") == current.get("format") == "h1-crossing-camera-v1":
        from camera_perception import DETECTOR_ALGORITHMS, VELOCITY_VALIDITY_MODES
        old_validity = saved["perception"]["velocity_validity"]
        new_validity = current["perception"]["velocity_validity"]
        if old_validity != new_validity and old_validity in VELOCITY_VALIDITY_MODES and new_validity in VELOCITY_VALIDITY_MODES:
            saved["perception"]["velocity_validity"] = new_validity
            warnings.warn(f"Playing with velocity validity {new_validity} instead of {old_validity}; "
                          "re-evaluation is required, not exact resume.", UserWarning)
        old_mode = saved["perception"]["detector_mode"]
        new_mode = current["perception"]["detector_mode"]
        if (old_mode != new_mode and old_mode in DETECTOR_ALGORITHMS and new_mode in DETECTOR_ALGORITHMS
                and saved.get("perception_algorithm") == DETECTOR_ALGORITHMS[old_mode]
                and current.get("perception_algorithm") == DETECTOR_ALGORITHMS[new_mode]):
            saved["perception"]["detector_mode"] = new_mode
            saved["perception_algorithm"] = current["perception_algorithm"]
            warnings.warn("Playing with changed color detector; re-evaluation is required, not exact resume.", UserWarning)
        old_rate = saved.get("perception", {}).get("frequency_hz")
        new_rate = current.get("perception", {}).get("frequency_hz")
        if old_rate is not None and new_rate is not None and old_rate != new_rate:
            saved["perception"]["frequency_hz"] = new_rate
            warnings.warn(f"Playing with camera rate {new_rate} Hz instead of checkpoint's {old_rate} Hz; "
                          "this is not an exact resume or a new performance validation.", UserWarning)
        old_names = saved.get("perception", {}).get("camera_names")
        new_names = current.get("perception", {}).get("camera_names")
        supported = (("front", "left"), ("front", "left", "right"))
        if old_names in supported and new_names in supported and old_names != new_names:
            saved["perception"]["camera_names"] = new_names
            warnings.warn(f"Playing with camera rig {new_names} instead of checkpoint's {old_names}; "
                          "22D schema is unchanged, but this sensor change needs evaluation and is not an exact resume.", UserWarning)
    if saved != current:
        raise ValueError("Behavior config/reward/schema/frozen bundle changed; refusing incompatible checkpoint")
    if old_traffic != new_traffic:
        if training:
            raise ValueError("Traffic dynamics changed; resume requires the checkpoint's --traffic_mode. "
                             "Original constant-speed checkpoints use --traffic_mode constant --yield_probability 0; "
                             "variable-speed checkpoints without yielding use --yield_probability 0.")
        warnings.warn("Playing checkpoint with changed traffic dynamics; previous training results "
                      "do not certify performance on this traffic.", UserWarning)


def load_behavior(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("compatibility", {}).get("format") not in ("h1-crossing-behavior-v1", "h1-crossing-camera-v1", "h1-crossing-camera-radar-v1", "h1-crossing-groundtruth23-v1", "h1-crossing-asymmetric-v1"):
        raise ValueError("Expected a crossing BEHAVIOR checkpoint, not a locomotion model")
    return data


def validate_warm_start(data, current):
    data = dict(data, compatibility=deepcopy(data["compatibility"]))
    old = data["compatibility"]
    align_behavior_revision(old, current, allow_changes=True)
    if old["settings"]["action_distribution"] != current["settings"]["action_distribution"]:
        raise ValueError("Changing action distribution requires fresh crossing training")
    if current["format"] == "h1-crossing-camera-radar-v1":
        from radar_observations import DIRECTION_FIELD
        from camera_perception import EXTRA_FIELDS
        expected = list(BEHAVIOR_FIELDS)+list(EXTRA_FIELDS)+[DIRECTION_FIELD]
        if current["observation_fields"] != expected:
            raise ValueError("Radar observation schema differs")
        if old["format"] == current["format"]:
            return check_compatibility(old, current, training=True)
        # Check all existing transfer invariants against the unchanged camera
        # prefix before explicitly migrating vehicle velocity semantics.
        camera_current = deepcopy(current)
        camera_current["format"] = "h1-crossing-camera-v1"
        camera_current["observation_fields"] = expected[:-1]
        camera_current.pop("radar")
        return validate_warm_start(data, camera_current)
    if old["format"] == current["format"] == "h1-crossing-camera-v1":
        from camera_transfer import validate_camera_transfer
        return validate_camera_transfer(data, current, BEHAVIOR_FIELDS)
    if old["format"] != "h1-crossing-behavior-v1" or current["format"] != "h1-crossing-camera-v1":
        raise ValueError("Warm-start supports state 19D -> camera 22D only; camera checkpoint uses --resume/--checkpoint")
    if old["observation_fields"] != list(BEHAVIOR_FIELDS):
        raise ValueError("State observation schema differs")
    for key in ("action", "reward_semantics", "rewards", "locomotion_policy_sha256", "locomotion_contract"):
        if old[key] != current[key]:
            raise ValueError(f"Warm-start incompatible {key}")
    for key in ("actor", "critic"):
        for field in ("hidden_dims", "activation", "obs_normalization"):
            if old["ppo"][key].get(field) != current["ppo"][key].get(field):
                raise ValueError(f"Warm-start incompatible {key}.{field}")


def warm_start(data, alg, contract):
    from camera_perception import expanded_state
    validate_warm_start(data, contract)
    for name in ("actor", "critic"):
        model = getattr(alg, name)
        source = data[name+"_state_dict"]
        weights = expanded_state(source, model.state_dict())
        model.load_state_dict(weights, strict=True)
    if alg.optimizer.state:
        raise ValueError("Warm-start requires a NEW optimizer")


def save_behavior(path, alg, completed, contract, env, seed):
    data = {**alg.save(), "completed_iterations": completed, "compatibility": contract,
            "num_envs": env.num_envs, "seed": seed, "learning_rate": alg.learning_rate,
            "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
            "traffic_rng": env.traffic.generator.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    temporary = path.with_suffix(".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def restore_behavior(data, alg, contract, env, seed, *, training):
    check_compatibility(data["compatibility"], contract, training=training)
    if training and (data["num_envs"] != env.num_envs or data["seed"] != seed):
        raise ValueError("Resume must retain num_envs and seed")
    alg.load(data, None if training else {"actor": True, "critic": True}, strict=True)
    if training:
        alg.learning_rate = float(data["learning_rate"])
        for group in alg.optimizer.param_groups:
            group["lr"] = alg.learning_rate
        torch.set_rng_state(data["torch_rng"].cpu())
        random.setstate(data["python_rng"])
        if data["cuda_rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in data["cuda_rng"]])
        env.traffic.generator.set_state(data["traffic_rng"].cpu())
    # Physics/rollout buffers are not serialized; explicitly start fresh episodes.
    env.reset()
    return int(data["completed_iterations"])


def load_asymmetric_play_actor(data, obs, contract, device):
    """Load actor alone; no critic network, GT input, optimizer or PPO at play."""
    from rsl_rl.utils import resolve_callable
    from asymmetric_observations import FORMAT
    if contract.get("format") != FORMAT:
        raise ValueError("Expected asymmetric checkpoint contract")
    check_compatibility(data["compatibility"], contract, training=False)
    cfg = deepcopy(contract["ppo"]["actor"])
    actor_type = resolve_callable(cfg.pop("class_name"))
    actor = actor_type(obs, {"actor": ["behavior"]}, "actor", 1, **cfg).to(device)
    actor.load_state_dict(data["actor_state_dict"], strict=True)
    return actor.eval()


def run(args, mode, app, output, settings, reward_cfg, manifest, loaded):
    from rsl_rl.algorithms import PPO
    from torch.utils.tensorboard import SummaryWriter
    from behavior_environment import CrossingBehaviorEnv
    from h1_finetune.runtime import plain
    from h1_finetune.numerics import finite
    env = CrossingBehaviorEnv(args.num_envs, args.device, args.seed, settings, reward_cfg,
                              args.bundle, manifest, args.headless,
                              traffic_config=TrafficConfig(mode=args.traffic_mode, yield_probability=args.yield_probability,
                                                           vehicle_palette=args.vehicle_palette),
                              perception_config=args.perception, perception_debug=args.perception_debug,
                              radar_config=args.radar, radar_output=output if args.radar else None,
                              radar_observations=args.observation_source == "camera_radar",
                              oracle_observations=getattr(args, "oracle_observations", False),
                              groundtruth23=args.observation_source == "groundtruth23",
                              privileged_critic=args.asymmetric_critic and mode == "train")
    play_keyboard = None
    truth_evaluation = None
    trajectory = None
    try:
        if mode == "play" and args.keyboard:
            from crossing_manual import run_keyboard
            write_json(output / "environment.json", plain(env.cfg.to_dict()))
            manual_result = run_keyboard(env, app, args, manifest)
            manual_result.update(observation_source=args.observation_source, observation_dim=env.observation_dim)
            write_json(output / "play_result.json", manual_result)
            return  # Do not construct/load a crossing actor, critic or optimizer.
        config = ppo_config(args.bundle, settings, asymmetric_critic=args.asymmetric_critic)
        contract = compatibility(settings, reward_cfg, manifest, config,
                                 TrafficConfig(mode=args.traffic_mode, yield_probability=args.yield_probability,
                                               vehicle_palette=args.vehicle_palette), args.perception,
                                 args.radar if args.observation_source == "camera_radar" else None,
                                 groundtruth23=args.observation_source == "groundtruth23",
                                 asymmetric_critic=args.asymmetric_critic)
        write_json(output / "environment.json", plain(env.cfg.to_dict()))
        write_json(output / "training_contract.json", contract)
        if mode == "play" and args.asymmetric_critic:
            from types import SimpleNamespace
            actor = load_asymmetric_play_actor(loaded, env.get_observations(), contract, args.device)
            alg = SimpleNamespace(actor=actor, eval_mode=actor.eval)
            completed = int(loaded["completed_iterations"])
            env.reset()  # Match the existing checkpoint-play episode initialization.
        else:
            alg = PPO.construct_algorithm(env.get_observations(), env, deepcopy(config), args.device)
        if mode == "play" and args.asymmetric_critic:
            pass  # Actor already loaded without constructing a critic or reading privileged state.
        elif args.init_checkpoint:
            warm_start(loaded, alg, contract)
            completed = 0
            save_behavior(output / "initialized_checkpoint.pt", alg, 0, contract, env, args.seed)
        else:
            completed = restore_behavior(loaded, alg, contract, env, args.seed, training=mode == "train") if loaded else 0
            if mode == "train" and args.fresh_behavior:
                if loaded is not None or completed != 0 or alg.optimizer.state:
                    raise AssertionError("Fresh behavior must not inherit a checkpoint or optimizer state")
                save_behavior(output / "initialized_checkpoint.pt", alg, 0, contract, env, args.seed)
                print("[Behavior] Fresh random crossing actor/critic, iteration 0; stability locomotion remains frozen.", flush=True)
        if mode == "play":
            alg.eval_mode()
            if args.trajectory:
                from behavior_trajectory import BehaviorTrajectory
                trajectory = BehaviorTrajectory(output, env.num_envs)
                env.trajectory_recorder = trajectory
            if args.oracle_observations:
                print("[Oracle evaluation] Ground truth feeds the 23D actor; sensors remain active for diagnostics.", flush=True)
            if args.ground_truth_eval:
                from ground_truth_evaluation import GroundTruthEvaluation
                truth_evaluation = GroundTruthEvaluation(output, env.num_envs,
                                                        ground_truth_to_actor=args.oracle_observations)
            manual_resets = 0
            if not args.headless:
                from play_reset_keyboard import PlayResetKeyboard
                play_keyboard = PlayResetKeyboard()
                print("[Play] Focus the Isaac Sim viewport and press R to reset all environments.", flush=True)
            counts = {k: 0 for k in ("success", "collision", "fall", "timeout", "out_of_bounds")}
            steps, obs = 0, env.get_observations()
            play_started = time.perf_counter()
            with torch.inference_mode():
                while app.is_running() and (args.steps < 0 or steps < args.steps):
                    if play_keyboard is not None and play_keyboard.consume():
                        obs = env.reset()
                        manual_resets += 1
                        print(f"[Play] R reset #{manual_resets}: H1, vehicle and camera tracks reset.", flush=True)
                    actions = alg.actor(obs, stochastic_output=False)
                    if trajectory:
                        trajectory.begin(env, obs, actions)
                    diagnostic = truth_evaluation.snapshot(env, obs, actions) if truth_evaluation else None
                    obs, reward, done, extras = env.step(actions)
                    if trajectory:
                        trajectory.finish(reward, done, extras)
                    if truth_evaluation:
                        truth_evaluation.record(diagnostic, done, extras)
                    for name in counts:
                        counts[name] += int(extras["events"][name].sum())
                    steps += 1
                    if args.perception and steps % 25 == 0:
                        print(f"[camera play] {steps}/{args.steps} behavior steps; "
                              f"detections={env.perception.detections}/{env.perception.exposures}", flush=True)
            if args.steps > 0 and steps != args.steps:
                raise RuntimeError("Play closed before requested steps")
            if truth_evaluation:
                truth_evaluation.completed = True
            if trajectory:
                trajectory.completed = True
            play_elapsed = time.perf_counter()-play_started
            write_json(output / "play_result.json", {"steps": steps, "finite": True, "events": counts,
                "ground_truth_to_actor": args.oracle_observations,
                "actor_input_source": "ground_truth_oracle" if args.oracle_observations else args.observation_source,
                "asymmetric_checkpoint": args.asymmetric_critic,
                "privileged_critic_active": env.privileged_critic,
                "guard_enabled": bool(settings.guard_enabled),
                "guard_mode": settings.guard_mode,
                "legacy_position_stop_enabled": bool(env.perception) and not (
                    settings.guard_enabled and settings.guard_mode == "dual_loss_ttc_v1"),
                "manual_resets": manual_resets,
                "wall_seconds": play_elapsed, "sim_seconds_per_env": steps*.1,
                "real_time_factor": steps*.1/max(play_elapsed, 1e-9),
                "note": "Not an evaluation acceptance gate; simultaneous fall/collision counts may overlap"})
            return
        if completed >= args.max_iterations:
            raise ValueError("--max_iterations must exceed the resumed completed iteration")
        if args.smoke_test and not args.perception:
            from smoke_behavior_checks import verify_physical_events
            write_json(output / "physical_smoke.json", verify_physical_events(env))
            env.reset()
        original = {k: v.detach().clone() for k, v in alg.actor.state_dict().items() if k.startswith("mlp.")}
        original_critic = {k: v.detach().clone() for k, v in alg.critic.state_dict().items()}
        frozen = {k: v.detach().clone() for k, v in env.locomotion.state_dict().items()}
        obs = env.get_observations()
        audit = None
        asymmetric_audit = None
        if args.asymmetric_critic:
            from asymmetric_training_checks import AsymmetricRolloutAudit
            asymmetric_audit = AsymmetricRolloutAudit()
        if args.smoke_test and env.perception:
            from camera_training_checks import CameraRolloutAudit
            audit = CameraRolloutAudit()
        rollout_seconds_total = update_seconds_total = 0.
        rollout_transitions_total = 0
        if args.perception and str(args.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(args.device)
        alg.train_mode()
        start = time.monotonic()
        with SummaryWriter(str(output / "tensorboard")) as writer, (output / "metrics.jsonl").open("w", encoding="utf-8") as log:
            while completed < args.max_iterations:
                if not app.is_running():
                    raise RuntimeError("Simulator closed before training completed")
                stats = {}
                reward_mean = 0.
                if args.perception and str(args.device).startswith("cuda"):
                    torch.cuda.synchronize(args.device)
                rollout_started = time.perf_counter()
                with torch.inference_mode():
                    for _ in range(settings.rollout_steps):
                        check_asymmetric = asymmetric_audit is not None and asymmetric_audit.batches < settings.rollout_steps
                        if check_asymmetric:
                            actions, asymmetric_input = asymmetric_audit.act(alg, obs)
                            asymmetric_index = alg.storage.step
                        elif audit:
                            actions, audited_input = audit.act(alg, obs)
                            storage_index = alg.storage.step
                        else:
                            actions = alg.act(obs)
                        requested = speed_from_action(actions, settings.action_deadband)
                        for key, value in {
                            "action/hard_boundary_fraction": (actions.abs() >= 1).float().mean(),
                            "action/near_max_speed_fraction": (requested >= 2.95).float().mean(),
                            "action/stop_fraction": (requested == 0).float().mean(),
                            "action/reverse_fraction": (requested < 0).float().mean(),
                            "action/mean_requested_speed_mps": requested.mean(),
                        }.items():
                            stats[key] = stats.get(key, 0.) + float(value)/settings.rollout_steps
                        obs, rewards, done, extras = env.step(actions)
                        alg.process_env_step(obs, rewards, done, {})  # finite horizon, no timeout bootstrap
                        if check_asymmetric:
                            asymmetric_audit.check_storage(alg, asymmetric_index, asymmetric_input)
                        if audit:
                            audit.check_storage(alg, storage_index, audited_input)
                        reward_mean += float(rewards.mean())
                        for key, value in extras["reward_terms"].items():
                            stats["reward/"+key] = stats.get("reward/"+key, 0.) + float(value.mean())
                        for key, value in extras["events"].items():
                            stats["episodes/"+key] = stats.get("episodes/"+key, 0.) + int(value.sum())
                    alg.compute_returns(obs)
                if args.perception and str(args.device).startswith("cuda"):
                    torch.cuda.synchronize(args.device)
                rollout_seconds = time.perf_counter()-rollout_started
                update_started = time.perf_counter()
                losses = alg.update()
                finite(losses, alg.actor.state_dict(), alg.critic.state_dict())
                if args.perception and str(args.device).startswith("cuda"):
                    torch.cuda.synchronize(args.device)
                update_seconds = time.perf_counter()-update_started
                rollout_seconds_total += rollout_seconds
                update_seconds_total += update_seconds
                rollout_transitions_total += env.num_envs*settings.rollout_steps
                completed += 1
                row = {"iteration": completed, "reward_mean": reward_mean/settings.rollout_steps,
                       "time/rollout_s": rollout_seconds, "time/ppo_update_s": update_seconds,
                       "throughput/transitions_per_s": env.num_envs*settings.rollout_steps/max(rollout_seconds,1e-9),
                       **{f"loss/{k}": float(v) for k, v in losses.items()}, **stats}
                if args.perception:
                    from h1_camera import gpu_memory_snapshot
                    for gpu, used_mib in (gpu_memory_snapshot() or {}).items():
                        row[f"memory/gpu_{gpu}_used_mib"] = used_mib
                    if str(args.device).startswith("cuda"):
                        row["memory/torch_peak_allocated_mib"] = torch.cuda.max_memory_allocated(args.device)/2**20
                log.write(json.dumps(row, allow_nan=False)+"\n")
                log.flush()
                for key, value in row.items():
                    if key != "iteration":
                        writer.add_scalar(key, value, completed)
                print(f"[behavior] {completed}/{args.max_iterations} {row}", flush=True)
                if completed % settings.save_interval == 0:
                    save_behavior(output / f"checkpoint_{completed:06d}.pt", alg, completed, contract, env, args.seed)
                    writer.flush()
            checkpoint = output / "checkpoint.pt"
            save_behavior(checkpoint, alg, completed, contract, env, args.seed)
        actor_updated = any(not torch.equal(v, alg.actor.state_dict()[k]) for k, v in original.items())
        critic_updated = any(not torch.equal(v, alg.critic.state_dict()[k]) for k, v in original_critic.items())
        frozen_unchanged = all(torch.equal(v, env.locomotion.state_dict()[k]) for k, v in frozen.items())
        if not actor_updated or not frozen_unchanged:
            raise AssertionError("Behavior actor must update and frozen locomotion must not")
        if asymmetric_audit:
            audit_result = asymmetric_audit.report()
            if not critic_updated or not audit_result["actor_critic_storage_match"]:
                raise AssertionError("Asymmetric critic must update with matching rollout storage")
            write_json(output / "asymmetric_training_audit.json", audit_result)
        if args.smoke_test and env.perception:
            if env.get_observations()["behavior"].shape != (env.num_envs, 22):
                raise AssertionError("Camera observation contract must be 22D")
            if not env.perception.detections or not env.perception.first_valid_times:
                raise AssertionError("Camera smoke must acquire a real image-based velocity track")
            audit_report = audit.report()
            if not audit_report["actor_and_storage_match"] or not audit_report["velocity_valid_transitions"]:
                raise AssertionError("Camera smoke must put valid image-derived tracks into PPO storage")
            write_json(output / "camera_training_audit.json", audit_report)
        data = load_behavior(checkpoint)
        before = {k: v.clone() for k, v in alg.actor.state_dict().items()}
        restore_behavior(data, alg, contract, env, args.seed, training=True)
        if not all(torch.equal(v, alg.actor.state_dict()[k]) for k, v in before.items()):
            raise AssertionError("Checkpoint round-trip changed actor weights")
        if args.asymmetric_critic:
            export = torch.jit.script(alg.actor.as_jit().cpu().eval())
            with torch.inference_mode():
                export_obs = env.get_observations(sensor_only=True)
                actor_input = export_obs["behavior"]
                expected = alg.actor(export_obs).cpu()
                torch.testing.assert_close(export(actor_input.cpu()), expected, rtol=1e-4, atol=1e-5)
            export.save(str(output / "actor_policy.pt"))
            write_json(output / "actor_policy.json", {"input_dim":23,"observation_fields":contract["observation_fields"],
                "action":contract["action"],"distribution":settings.action_distribution,
                "deterministic":True,"requires_vehicle_ground_truth":False,
                "note":"Crossing actor only; requires sensor preprocessing, command mapping and frozen locomotion"})
        write_json(output / "training_result.json", {"completed_iterations": completed, "finite": True,
            "actor_updated": actor_updated, "critic_updated": critic_updated, "locomotion_unchanged": frozen_unchanged,
            "checkpoint_roundtrip": True, "checkpoint_sha256": sha256(checkpoint),
            "elapsed_s": time.monotonic()-start, "smoke_only": args.smoke_test,
            "observation_source": args.observation_source, "headless": args.headless,
            "observation_dim": env.observation_dim,
            "critic_observation_dim": env.critic_observation_dim,
            "asymmetric_critic": args.asymmetric_critic,
            "rollout_seconds": rollout_seconds_total, "ppo_update_seconds": update_seconds_total,
            "rollout_transitions": rollout_transitions_total,
            "camera_rollout_audited": bool(audit),
            "note": "No crossing success claim; run an independent deterministic evaluation"})
    finally:
        if trajectory is not None:
            trajectory.close()
        if truth_evaluation is not None:
            truth_evaluation.close()
        if play_keyboard is not None:
            try:
                play_keyboard.close()
            except Exception as error:
                warnings.warn(f"Could not release reset hotkey: {error}", RuntimeWarning)
        if env.perception:
            write_json(output / "perception_summary.json", env.perception.report())
            from h1_camera import gpu_memory_snapshot
            write_json(output / "gpu_memory_after.json", gpu_memory_snapshot())
        if env.radar:
            radar_report = env.radar.report()
            if getattr(args, "keyboard", False):
                radar_report["policy_source"] = "manual keyboard; frozen locomotion only; no crossing actor"
            write_json(output / "radar_summary.json", radar_report)
        env.close()


def main(mode="train", argv=None):
    args = parse_args(mode, argv)
    args.bundle = args.bundle.resolve()
    settings, reward_cfg = load_behavior_config(args.config), load_reward_config(args.reward_config)
    if args.observation_source == "groundtruth23" and settings.guard_enabled:
        raise ValueError("Groundtruth23 baseline requires guard_enabled=0")
    manifest = validate_bundle(args.bundle, gate, STAGES)
    args.perception = None
    args.radar = None
    if args.observation_source in ("camera", "camera_radar"):
        from camera_perception import load_config
        args.perception = load_config(args.perception_config)
        if args.vehicle_palette == "multicolor" and args.perception.detector_mode != "multicolor":
            raise ValueError("Multicolor camera mode requires configs/crossing_perception_multicolor.yaml")
    if args.radar_shadow or args.observation_source == "camera_radar":
        from radar_processing import load_radar_config
        args.radar = load_radar_config(args.radar_config)
        if args.radar.sample_hz != args.perception.frequency_hz:
            raise ValueError("Radar shadow samples on the camera capture clock; configured frequencies must match")
    source = getattr(args, "resume", None) or getattr(args, "checkpoint", None) or args.init_checkpoint
    loaded = load_behavior(source) if source else None
    args.num_envs = args.num_envs or (loaded["num_envs"] if loaded and mode == "train" and not args.init_checkpoint else
        ((8 if args.observation_source == "camera_radar" else 4 if args.perception else 32) if mode == "train" else 1))
    args.seed = args.seed if args.seed is not None else (loaded["seed"] if loaded else 42)
    if args.num_envs*settings.rollout_steps < settings.mini_batches:
        raise ValueError("Rollout smaller than minibatch count")
    if loaded:
        current = compatibility(settings, reward_cfg, manifest, ppo_config(args.bundle, settings, asymmetric_critic=args.asymmetric_critic),
                          TrafficConfig(mode=args.traffic_mode, yield_probability=args.yield_probability,
                                        vehicle_palette=args.vehicle_palette), args.perception,
                          args.radar if args.observation_source == "camera_radar" else None,
                          groundtruth23=args.observation_source == "groundtruth23",
                          asymmetric_critic=args.asymmetric_critic)
        if args.init_checkpoint:
            validate_warm_start(loaded, current)
        else:
            check_compatibility(loaded["compatibility"], current, training=mode == "train")
    output = args.output or ROOT / "logs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_behavior_" + mode)
    output = output.resolve()
    if not output.is_relative_to(ROOT / "logs"):
        raise ValueError("Behavior outputs must be inside H1Crossing/logs")
    output.mkdir(parents=True, exist_ok=False)
    print(f"[Output] {output}", flush=True)
    versions = {}
    for name in ("torch", "rsl-rl-lib", "isaaclab", "isaacsim", "gymnasium"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "unavailable"
    write_json(output / "provenance.json", {"args": {k: asdict(v) if k in ("perception", "radar") and v else str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "package_versions": versions, "source_checkpoint_sha256": sha256(source) if source else None,
        "sources": {p.name: sha256(p) for p in (ROOT / "scripts").glob("*.py")}})
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    from isaaclab.app import AppLauncher
    from h1_camera import gpu_memory_snapshot
    write_json(output / "gpu_memory_before.json", gpu_memory_snapshot())
    kit_args = "--vulkan" if args.perception else ""
    experience = {}
    if args.radar:
        from h1_radar import MOTION_BVH_ARGS, radar_experience
        kit_args += " " + MOTION_BVH_ARGS
        experience["experience"] = radar_experience()
    launcher = AppLauncher(visualizer="none" if args.headless else "kit", device=args.device,
                          enable_cameras=bool(args.perception), kit_args=kit_args, **experience)
    exit_code = 0
    try:
        if args.radar:
            # Resolve Lab's lazy simulation/config exports before external task
            # imports when using the broader Isaac Sim reference experience.
            import isaaclab.sim
            from h1_radar import prepare_radar_clock
            prepare_radar_clock(args.radar.sample_hz)
        run(args, mode, launcher.app, output, settings, reward_cfg, manifest, loaded)
    except Exception as error:
        exit_code = 1
        import traceback
        write_json(output / "failure.json", {"error": str(error), "traceback": traceback.format_exc()})
        traceback.print_exc()
        raise
    except KeyboardInterrupt:
        exit_code = 130
        raise
    finally:
        # Kit fast shutdown otherwise masks Python errors with process status 0.
        launcher.app.close(exit_code=exit_code)
