"""Read-only 30 Hz camera-policy evaluation with isolated ground-truth scoring.

No optimizer updates or changes to production perception/environment modules.
Capture observer runs AFTER RGB-D perception and never changes observations.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from behavior_training import (ROOT, load_behavior, ppo_config, compatibility,
                               check_compatibility, write_json)
from behavior_control import load_behavior_config, behavior_observation
from camera_perception import load_config
from compare_camera_checkpoints import evaluate_episode, EVENTS
from crossing_bundle import validate_bundle, sha256
from crossing_rewards import load_reward_config
from crossing_observations import planar_ttc, compute_crossing_observation
from vehicle_motion import TrafficConfig
from h1_finetune.control import STAGES
from h1_finetune.validation import gate


def error_stats(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(values):
        return {"n": 0, "mean": None, "rmse": None, "p95": None, "max": None}
    if not np.isfinite(values).all():
        raise AssertionError("Nonfinite diagnostic error")
    return {"n": len(values), "mean": float(values.mean()),
            "rmse": float(np.sqrt(np.mean(values**2))),
            "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def summarize_frames(frames):
    n = len(frames)
    count = lambda field: sum(bool(f[field]) for f in frames)
    dt = np.diff([f["simulation_time_s"] for f in frames])
    if len(dt) and (not np.isfinite(dt).all() or (dt <= 0).any()):
        raise AssertionError("Non-monotonic capture timestamps")
    losses = sum(a["velocity_valid"] and not b["velocity_valid"] for a, b in zip(frames, frames[1:]))
    return {"exposures": n, "detected_exposures": count("detected"),
            "detection_rate": count("detected")/n if n else None,
            "position_valid_rate": count("position_valid")/n if n else None,
            "velocity_valid_rate": count("velocity_valid")/n if n else None,
            "velocity_track_losses": losses,
            "first_velocity_sim_s": next((f["simulation_time_s"] for f in frames if f["velocity_valid"]), None),
            "measured_capture_hz_sim": 1/float(dt.mean()) if len(dt) else None,
            "capture_interval_s": {"min": float(dt.min()), "max": float(dt.max())} if len(dt) else None,
            "position_error_m": error_stats([f["position_error_m"] for f in frames]),
            "velocity_error_mps": error_stats([f["velocity_error_mps"] for f in frames]),
            "ttc_error_s_both_valid": error_stats([f["ttc_error_s"] for f in frames]),
            "ground_truth_ttc_valid_exposures": count("gt_ttc_valid"),
            "gt_ttc_valid_camera_invalid": sum(f["gt_ttc_valid"] and not f["camera_ttc_valid"] for f in frames),
            "gt_ttc_invalid_camera_valid": sum(not f["gt_ttc_valid"] and f["camera_ttc_valid"] for f in frames)}


class CaptureScorer:
    """Observer owned by evaluator, not by policy, detector or tracker."""
    def __init__(self, env):
        self.env = env
        self.original_capture = env.perception.capture
        self.frames = []
        self.decisions = []
        self.enabled = False

    def decision(self, env, observation):
        """Score a COPY of the exact actor input; GT never enters the actor."""
        robot = env.robot.data
        h1 = robot.root_pos_w.torch[:, :2]
        h1_velocity = robot.root_lin_vel_w.torch[:, :2]
        car = env.vehicle.data.root_pos_w.torch[:, :2]
        velocity = torch.stack((env.traffic.speed, torch.zeros_like(env.traffic.speed)), dim=1)
        raw = compute_crossing_observation(h1, h1_velocity, car, velocity, env.goal_xy)
        truth = behavior_observation(raw, h1_velocity, robot.root_quat_w.torch,
            env.term.vel_command_b, h1-env.tracker.anchor_xy,
            env.tracker.stationary_elapsed_s, env.elapsed, env.settings.episode_seconds)
        # The 3 extra fields describe track quality, not physical GT quantities.
        shared = [2, 3, *range(8, 19)]
        torch.testing.assert_close(observation[:, shared], truth[:, shared], rtol=0, atol=1e-6)
        cam, gt = observation[0].cpu().tolist(), truth[0].cpu().tolist()
        pos_valid, vel_valid = bool(cam[19]), bool(cam[20])
        row = {"simulation_time_s": float(env.sim_time),
               "vehicle_pass": int(env.traffic.pass_count[0]),
               "camera_observation_22d": cam, "ground_truth_observation_19d": gt,
               "position_valid": pos_valid, "velocity_valid": vel_valid,
               "track_age_s": cam[21]*env.perception.cfg.dropout_seconds,
               "gt_ttc_valid": bool(gt[7]), "camera_ttc_valid": bool(cam[7]),
               "position_error_m": math.hypot((cam[0]-gt[0])*50, (cam[1]-gt[1])*10) if pos_valid else None,
               "velocity_error_mps": math.hypot((cam[4]-gt[4])*15, (cam[5]-gt[5])*15) if vel_valid else None,
               "ttc_error_s": abs(cam[6]-gt[6])*10 if cam[7] and gt[7] else None,
               "shared_fields_max_abs_error": max(abs(cam[i]-gt[i]) for i in shared)}
        json.dumps(row, allow_nan=False)
        self.decisions.append(row)

    def capture(self, now):
        result = self.original_capture(now)
        env = self.env
        if not self.enabled or bool(env.tracker.finished[0]):
            return result
        state, valid, velocity_valid, age = env.perception.tracks[0].estimate(now)
        robot = env.robot.data
        h1 = robot.root_pos_w.torch[0, :2].detach().cpu().numpy().copy()
        h1_vel = robot.root_lin_vel_w.torch[0, :2].detach().cpu().numpy().copy()
        truth = env.vehicle.data.root_pos_w.torch[0, :2].detach().cpu().numpy().copy()
        truth_velocity = np.array([float(env.traffic.speed[0]), 0.])
        relative = torch.tensor(np.stack((truth-h1, state[:2]-h1)), dtype=torch.float64)
        relative_vel = torch.tensor(np.stack((truth_velocity-h1_vel, state[2:]-h1_vel)), dtype=torch.float64)
        ttc, ttc_valid = planar_ttc(relative, relative_vel)
        gt_valid = bool(ttc_valid[0, 0])
        cam_valid = bool(valid and velocity_valid and ttc_valid[1, 0])
        views = env.perception.frames[0][1]
        row = {"simulation_time_s": float(now), "detected": any(v[-1] is not None for v in views),
               "front_detected": views[0][-1] is not None, "left_detected": views[1][-1] is not None,
               "right_detected": views[2][-1] is not None if len(views) > 2 else None,
               "position_valid": bool(valid), "velocity_valid": bool(velocity_valid),
               "track_age_s": float(age)*env.perception.cfg.dropout_seconds,
               "h1_xy": h1.tolist(), "h1_velocity_xy": h1_vel.tolist(),
               "h1_angular_velocity_xyz": robot.root_ang_vel_w.torch[0].detach().cpu().tolist(),
               "car_gt_xy": truth.tolist(), "car_gt_velocity_xy": truth_velocity.tolist(),
               "car_estimate_xy": state[:2].tolist() if valid else None,
               "car_estimate_velocity_xy": state[2:].tolist() if velocity_valid else None,
               "position_error_m": float(np.linalg.norm(state[:2]-truth)) if valid else None,
               "velocity_error_mps": float(np.linalg.norm(state[2:]-truth_velocity)) if velocity_valid else None,
               "gt_ttc_valid": gt_valid, "camera_ttc_valid": cam_valid,
               "gt_ttc_s": float(ttc[0, 0]) if gt_valid else None,
               "camera_ttc_s": float(ttc[1, 0]) if cam_valid else None,
               "ttc_error_s": abs(float(ttc[0, 0]-ttc[1, 0])) if gt_valid and cam_valid else None}
        if hasattr(env.traffic, "pass_count"):
            row["vehicle_pass"] = int(env.traffic.pass_count[0])
        # Validate before retaining diagnostic data. Nothing is fed back to env.
        json.dumps(row, allow_nan=False)
        self.frames.append(row)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--yield_probability", type=float, default=.5)
    parser.add_argument("--traffic_mode", choices=("constant", "variable"), default="variable")
    args = parser.parse_args()
    if len(args.seeds) != 5 or len(set(args.seeds)) != 5:
        parser.error("Provide exactly five unique seeds")
    if any(s < 0 or s > 2**31-1001 for s in args.seeds):
        parser.error("Seeds must be nonnegative 31-bit values")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT/"logs"):
        parser.error("Output must be a new directory under H1Crossing/logs")
    checkpoint = args.checkpoint.resolve()
    data = load_behavior(checkpoint)
    settings = load_behavior_config(ROOT/"configs/crossing_behavior_camera.yaml")
    rewards, perception = load_reward_config(), load_config()
    traffic = TrafficConfig(mode=args.traffic_mode, yield_probability=args.yield_probability)
    traffic.validate()
    if perception.frequency_hz != 30.:
        raise ValueError("This evaluation requires the current 30 Hz camera configuration")
    bundle = ROOT/"checkpoints/h1_flat_stability_v2_0500"
    manifest = validate_bundle(bundle, gate, STAGES)
    config = ppo_config(bundle, settings)
    contract = compatibility(settings, rewards, manifest, config, traffic, perception)
    # Read-only evaluation may intentionally probe a new camera rig/rate.
    check_compatibility(data["compatibility"], contract, training=False)
    checkpoint_hash = sha256(checkpoint)
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = {p.name: sha256(p) for p in (ROOT/"scripts").glob("*.py")}
    write_json(output/"protocol.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "seeds": args.seeds,
        "episodes": 5, "num_envs": 1, "deterministic_actor": True,
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
        "checkpoint_iterations": data["completed_iterations"], "contract": contract,
        "sources_sha256": source_hashes,
        "ground_truth": "Diagnostic observer after capture only; no estimator/policy feedback",
        "error_definition": "Euclidean XY errors on valid tracks; misses reported separately",
        "ttc_definition": "Same instantaneous constant-relative-velocity footprint test for GT and camera; not future realized collision time",
        "scope": "Five complete first-terminal episodes with torso-mounted cameras; not a controlled camera-shake ablation"})
    print(f"[Output] {output}", flush=True)
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(visualizer="none", device="cuda:0", enable_cameras=True, kit_args="--vulkan")
    env, code, rows, all_frames, all_decisions = None, 0, [], [], []
    try:
        from behavior_environment import CrossingBehaviorEnv
        from rsl_rl.algorithms import PPO
        from h1_camera import gpu_memory_snapshot
        env = CrossingBehaviorEnv(1, "cuda:0", args.seeds[0], settings, rewards, bundle, manifest, True,
                                  traffic_config=traffic, perception_config=perception)
        alg = PPO.construct_algorithm(env.get_observations(), env, deepcopy(config), "cuda:0")
        frozen = {k: v.clone() for k, v in env.locomotion.state_dict().items()}
        scorer = CaptureScorer(env)
        env.perception.capture = scorer.capture
        for seed in args.seeds:
            scorer.frames = []
            scorer.decisions = []
            scorer.enabled = True
            row = evaluate_episode(env, alg, data, seed, launcher.app, decision_observer=scorer.decision)
            scorer.enabled = False
            # Environment completes a behavior batch after first terminal low-level step.
            frames = [f for f in scorer.frames if f["simulation_time_s"] <= row["duration_s"]+1e-4]
            with (output/f"frames_seed_{seed}.jsonl").open("w", encoding="utf-8") as handle:
                for frame in frames:
                    handle.write(json.dumps(frame, allow_nan=False)+"\n")
            row["tracking"] = summarize_frames(frames)
            with (output/f"decisions_seed_{seed}.jsonl").open("w", encoding="utf-8") as handle:
                for decision in scorer.decisions:
                    handle.write(json.dumps(decision, allow_nan=False)+"\n")
            row["camera_count"] = len(perception.camera_names)
            row["capture_sets_per_wall_second"] = len(frames)/row["wall_seconds"]
            rows.append(row)
            all_frames.extend(frames)
            all_decisions.extend(scorer.decisions)
            write_json(output/"episodes.json", rows)
            print(f"[episode] seed={seed} outcome={row['outcome']} sim={row['duration_s']:.2f}s "
                  f"wall={row['wall_seconds']:.1f}s captures={len(frames)} "
                  f"tracking={row['tracking']}", flush=True)
        if not all(torch.equal(v, env.locomotion.state_dict()[k]) for k, v in frozen.items()):
            raise AssertionError("Frozen locomotion changed")
        if sha256(checkpoint) != checkpoint_hash:
            raise AssertionError("Source checkpoint changed")
        for name, original_hash in source_hashes.items():
            if sha256(ROOT/"scripts"/name) != original_hash:
                raise AssertionError(f"Source changed during evaluation: {name}")
        aggregate = {"complete": len(rows) == 5, "episodes": len(rows),
            "outcomes": {name: sum(r["outcome"] == name for r in rows) for name in EVENTS},
            "exposures": len(all_frames),
            "detection_rate": sum(f["detected"] for f in all_frames)/len(all_frames),
            "position_valid_rate": sum(f["position_valid"] for f in all_frames)/len(all_frames),
            "velocity_valid_rate": sum(f["velocity_valid"] for f in all_frames)/len(all_frames),
            "position_error_m": error_stats([f["position_error_m"] for f in all_frames]),
            "velocity_error_mps": error_stats([f["velocity_error_mps"] for f in all_frames]),
            "ttc_error_s_both_valid": error_stats([f["ttc_error_s"] for f in all_frames]),
            "total_wall_s": sum(r["wall_seconds"] for r in rows),
            "total_sim_s": sum(r["duration_s"] for r in rows),
            "gpu_memory_snapshot_mib": gpu_memory_snapshot(),
            "verification": {"finite": True, "locomotion_unchanged": True,
                             "checkpoint_unchanged": True, "sources_unchanged": True},
            "caveats": ["Small five-seed screen, not safety certification or causal shake attribution.",
                        "Conditional errors exclude invalid tracks; valid fractions include them.",
                        "Aggregates weight each captured frame, not each episode equally.",
                        "30 Hz is simulation capture frequency, not real-time FPS."]}
        aggregate["policy_input_comparison"] = {
            "decisions": len(all_decisions),
            "position_valid_rate": sum(d["position_valid"] for d in all_decisions)/len(all_decisions),
            "velocity_valid_rate": sum(d["velocity_valid"] for d in all_decisions)/len(all_decisions),
            "position_error_m": error_stats([d["position_error_m"] for d in all_decisions]),
            "velocity_error_mps": error_stats([d["velocity_error_mps"] for d in all_decisions]),
            "ttc_error_s_both_valid": error_stats([d["ttc_error_s"] for d in all_decisions]),
            "gt_ttc_valid": sum(d["gt_ttc_valid"] for d in all_decisions),
            "gt_ttc_valid_camera_invalid": sum(d["gt_ttc_valid"] and not d["camera_ttc_valid"] for d in all_decisions),
            "gt_ttc_invalid_camera_valid": sum(not d["gt_ttc_valid"] and d["camera_ttc_valid"] for d in all_decisions),
            "shared_fields_max_abs_error": max(d["shared_fields_max_abs_error"] for d in all_decisions),
            "note": "Exact 10 Hz actor inputs scored before act; GT is diagnostic-only. Invalid sentinel values excluded from conditional position/velocity errors but counted in valid fractions."}
        write_json(output/"summary.json", aggregate)
        print(f"[complete] {aggregate}", flush=True)
    except BaseException as error:
        import traceback
        code = 1
        write_json(output/"failure.json", {"error": str(error), "traceback": traceback.format_exc()})
        raise
    finally:
        if env is not None:
            env.close()
        launcher.app.close(exit_code=code)


if __name__ == "__main__":
    main()
