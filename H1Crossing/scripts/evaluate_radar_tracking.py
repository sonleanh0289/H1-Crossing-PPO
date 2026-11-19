"""Radar accuracy screen with isolated, exposure-aligned ground truth.

Scripted manual velocity commands use frozen locomotion; no crossing actor.
Truth is scored after radar processing and never fed to sensor/perception.
"""
import argparse
from collections import Counter, deque
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from behavior_training import ROOT, write_json
from behavior_control import load_behavior_config
from camera_perception import load_config
from crossing_bundle import validate_bundle, sha256
from crossing_rewards import load_reward_config
from radar_processing import load_radar_config
from vehicle_motion import TrafficConfig
from h1_finetune.control import STAGES
from h1_finetune.validation import gate


def command_at(t):
    """Repeatable commands, body frame, independent of radar/truth."""
    if t < 3:
        return "stand", (0., 0., 0.)
    if t < 7:
        return "forward", (1., 0., 0.)
    if t < 8:
        return "stand", (0., 0., 0.)
    if t < 11:
        return "reverse", (-.5, 0., 0.)
    if t < 12:
        return "stand", (0., 0., 0.)
    if t < 14:
        return "turn_left", (0., 0., .5)
    if t < 16:
        return "turn_right", (0., 0., -.5)
    return "forward", (1., 0., 0.)


def signed_stats(errors):
    x = np.asarray(errors, dtype=float)
    if not len(x):
        return {"n": 0, "mae_mps": None, "rmse_mps": None, "bias_mps": None,
                "p95_abs_mps": None, "max_abs_mps": None}
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite radar errors")
    return {"n": len(x), "mae_mps": float(np.abs(x).mean()),
            "rmse_mps": float(np.sqrt(np.mean(x*x))), "bias_mps": float(x.mean()),
            "p95_abs_mps": float(np.percentile(np.abs(x), 95)), "max_abs_mps": float(np.abs(x).max())}


def aligned_truth(history, row):
    """Match prior render poses to the native exposure; reject teleport intervals."""
    end = row["sensor_clock_reference_sim_s"]
    duration = (row["exposure_end_timestamp_ns"]-row["exposure_start_timestamp_ns"])*1e-9
    start = end-duration
    def match(t):
        candidates = [h for h in history if abs(h["t"]-t) <= 1e-5]
        return candidates[-1] if candidates else None
    a, b = match(start), match(end)
    if a is None or b is None:
        return {"gt_alignment_valid": False, "gt_alignment_reason": "missing_exposure_history"}
    if a["pass"] != b["pass"] or a["reset"] != b["reset"]:
        return {"gt_alignment_valid": False, "gt_alignment_reason": "respawn_or_reset"}
    displacement = np.asarray(b["car"])-a["car"]
    if not 0 < duration <= .1 or np.linalg.norm(displacement) > 3:
        return {"gt_alignment_valid": False, "gt_alignment_reason": "discontinuous_motion"}
    velocity = displacement / duration
    if abs(velocity[1]) > .01 or abs(velocity[2]) > .01:
        return {"gt_alignment_valid": False, "gt_alignment_reason": "non_lane_motion"}
    return {"gt_alignment_valid": True, "gt_alignment_reason": "aligned_render_exposure",
            "gt_exposure_start_sim_s": start, "gt_exposure_end_sim_s": end,
            "gt_lane_vx_exposure_mps": float(velocity[0]),
            "gt_lane_vx_end_mps": b["speed"], "gt_car_xy": b["car"][:2],
            "gt_vehicle_pass": b["pass"], "gt_is_yielding": b["yielding"],
            "gt_range_xy_m": float(np.linalg.norm(np.asarray(a["car"][:2])-row["sensor_position_world"][:2])),
            "gt_h1_speed_xy_mps": b["h1_speed"], "gt_h1_yaw_rate_mps": b["h1_yaw_rate"]}


def summarize(rows):
    measured = [r for r in rows if r["lane_speed_valid"] and r["gt_alignment_valid"]]
    def errors(subset):
        return signed_stats([r["error_lane_vx_mps"] for r in subset])
    n = len(rows)
    return {"frames": n, "radar_valid_frames": sum(r["lane_speed_valid"] for r in rows),
            "radar_valid_rate": sum(r["lane_speed_valid"] for r in rows)/n if n else None,
            "associated_rate": sum(r["associated"] for r in rows)/n if n else None,
            "scored_frames": len(measured), "scored_rate_all_frames": len(measured)/n if n else None,
            "invalid_radar_reasons": dict(Counter(r["reason"] for r in rows if not r["lane_speed_valid"])),
            "unscored_alignment_reasons": dict(Counter(r["gt_alignment_reason"] for r in rows if not r["gt_alignment_valid"])),
            "speed_error": errors(measured),
            "by_command": {name: errors([r for r in measured if r["command_phase"] == name])
                           for name in ("stand", "forward", "reverse", "turn_left", "turn_right")},
            "by_actual_h1_motion": {
                "near_stationary": errors([r for r in measured if r["gt_h1_speed_xy_mps"] < .15 and abs(r["gt_h1_yaw_rate_mps"]) < .1]),
                "translating": errors([r for r in measured if r["gt_h1_speed_xy_mps"] >= .15]),
                "turning": errors([r for r in measured if abs(r["gt_h1_yaw_rate_mps"]) >= .1])},
            "by_range": {label: errors([r for r in measured if lo <= r["gt_range_xy_m"] < hi])
                         for label, lo, hi in (("0_5m", 0, 5), ("5_15m", 5, 15), ("15_30m", 15, 30), ("30_50m", 30, 50), ("50_80m", 50, 80))}}


class RadarTruthObserver:
    """Ground truth observer only; original radar capture executes first."""
    def __init__(self, env):
        self.env = env
        self.before = env.radar.before_render
        self.capture_original = env.radar.capture
        self.reset_original = env.reset
        self.reset_id = 0
        self.history = deque(maxlen=12)
        self.rows = []
        self.phase = "stand"
        self.enabled = False

    def reset(self, *args, **kwargs):
        self.reset_id += 1
        return self.reset_original(*args, **kwargs)

    def before_render(self, now):
        env = self.env
        self.history.append({"t": float(now), "reset": self.reset_id,
            "pass": int(env.traffic.pass_count[0]),
            "car": env.vehicle.data.root_pos_w.torch[0].detach().cpu().tolist(),
            "speed": float(env.traffic.speed[0]), "yielding": bool(env.traffic.is_yielding[0]),
            "h1_speed": float(torch.linalg.vector_norm(env.robot.data.root_lin_vel_w.torch[0, :2])),
            "h1_yaw_rate": float(env.robot.data.root_ang_vel_w.torch[0, 2])})
        return self.before(now)

    def capture(self, now, camera_frames, vehicle_size):
        result = self.capture_original(now, camera_frames, vehicle_size)
        row = self.env.radar.latest
        if self.enabled and row is None:
            self.rows.append(dict(sim_capture_s=now, lane_speed_valid=False, associated=False,
                reason="missing_current_buffer", command_phase=self.phase, gt_alignment_valid=False,
                gt_alignment_reason="missing_current_buffer", error_lane_vx_mps=None, error_vs_gt_end_mps=None))
        elif self.enabled:
            scored = {**row, **aligned_truth(self.history, row), "command_phase": self.phase}
            scored["error_lane_vx_mps"] = (row["lane_vx_mps"]-scored["gt_lane_vx_exposure_mps"]
                if row["lane_speed_valid"] and scored["gt_alignment_valid"] else None)
            scored["error_vs_gt_end_mps"] = (row["lane_vx_mps"]-scored["gt_lane_vx_end_mps"]
                if row["lane_speed_valid"] and scored["gt_alignment_valid"] else None)
            self.rows.append(scored)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42,43,44,45,46])
    parser.add_argument("--steps", type=int, default=300, help="10 Hz behavior steps per seed")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--yield_probability", type=float, default=.5)
    parser.add_argument("--radar_config", type=Path, default=ROOT / "configs/crossing_radar.yaml")
    args = parser.parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s > 2**31-1001 for s in args.seeds):
        parser.error("Distinct nonnegative seeds required")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "logs"):
        parser.error("Output must be under H1Crossing/logs")
    traffic = TrafficConfig(vehicle_palette="multicolor", yield_probability=args.yield_probability)
    traffic.validate()
    settings = load_behavior_config(ROOT / "configs/crossing_behavior_camera.yaml")
    perception = load_config(ROOT / "configs/crossing_perception_multicolor.yaml")
    radar = load_radar_config(args.radar_config)
    if radar.sample_hz != perception.frequency_hz:
        raise ValueError("Radar/camera rates must match")
    bundle = ROOT / "checkpoints/h1_flat_stability_v2_0500"
    manifest = validate_bundle(bundle, gate, STAGES)
    policy_hash = sha256(bundle / "policy.pt")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": args.seeds, "steps_per_seed": args.steps, "num_envs": 1,
        "physics_hz": 200, "locomotion_hz": 50, "sensor_hz": radar.sample_hz,
        "crossing_checkpoint": None, "control": "scripted manual command sequence, frozen locomotion",
        "observation_source": "camera_radar", "observation_dim": 23,
        "commands": [{"start_s": t, "phase": command_at(t)[0], "command": command_at(t)[1]}
                     for t in (0,3,7,8,11,12,14,16)],
        "traffic": traffic.contract(), "radar": asdict(radar), "perception": asdict(perception),
        "locomotion_policy_sha256": policy_hash,
        "ground_truth": "Evaluation-only, average actual vehicle displacement over prior native exposure; resets excluded",
        "error_scope": "Conditional lane vx error on accepted radar frames; invalid rates use all returned frames",
        "sources_sha256": {p.name: sha256(p) for p in (ROOT / "scripts").glob("*.py")}})
    print(f"[Output] {output}", flush=True)
    from isaaclab.app import AppLauncher
    from h1_radar import radar_experience, MOTION_BVH_ARGS, prepare_radar_clock
    launcher = AppLauncher(visualizer="none", device="cuda:0", enable_cameras=True,
                          experience=radar_experience(), kit_args="--vulkan " + MOTION_BVH_ARGS)
    env = None
    code = 0
    all_rows, seed_results = [], []
    try:
        import isaaclab.sim
        prepare_radar_clock(radar.sample_hz)
        from behavior_environment import CrossingBehaviorEnv
        from h1_camera import gpu_memory_snapshot
        env = CrossingBehaviorEnv(1, "cuda:0", args.seeds[0], settings, load_reward_config(),
            bundle, manifest, True, traffic_config=traffic, perception_config=perception,
            radar_config=radar, radar_output=output, radar_observations=True)
        frozen = {k: v.clone() for k, v in env.locomotion.state_dict().items()}
        observer = RadarTruthObserver(env)
        env.radar.before_render, env.radar.capture, env.reset = observer.before_render, observer.capture, observer.reset
        for seed in args.seeds:
            torch.manual_seed(seed)
            env.traffic.generator.manual_seed(seed+1000)
            env.reset()
            observer.history.clear()
            observer.rows = []
            observer.enabled = True
            start_sim, start_wall = env.sim_time, time.perf_counter()
            counts = {k: 0 for k in ("success", "collision", "fall", "timeout", "out_of_bounds")}
            with torch.inference_mode():
                for step in range(args.steps):
                    if not launcher.app.is_running():
                        raise RuntimeError("App closed early")
                    observer.phase, command = command_at(step*.1)
                    obs, reward, done, extras = env.step(None, manual_command=[command])
                    if not torch.isfinite(obs["behavior"]).all() or not torch.isfinite(reward).all():
                        raise AssertionError("Nonfinite environment data")
                    for name in counts:
                        counts[name] += int(extras["events"][name].sum())
                    if (step+1) % 100 == 0:
                        print(f"[progress] seed={seed} steps={step+1}/{args.steps} frames={len(observer.rows)}", flush=True)
            observer.enabled = False
            rows = [{"seed": seed, **r} for r in observer.rows]
            with (output / f"frames_seed_{seed}.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, allow_nan=False)+"\n")
            summary = {"seed": seed, "sim_seconds": env.sim_time-start_sim,
                       "wall_seconds": time.perf_counter()-start_wall, "events": counts, **summarize(rows)}
            seed_results.append(summary)
            all_rows.extend(rows)
            write_json(output / "seeds.json", seed_results)
            print(f"[seed result] {json.dumps(summary['speed_error'])} valid={summary['radar_valid_rate']:.3f}", flush=True)
        unchanged = all(torch.equal(v, env.locomotion.state_dict()[k]) for k,v in frozen.items())
        if not unchanged or sha256(bundle / "policy.pt") != policy_hash:
            raise AssertionError("Locomotion changed during evaluation")
        report = {"complete": len(seed_results) == len(args.seeds), "seeds": seed_results,
            "aggregate": summarize(all_rows), "sensor": env.radar.report(),
            "total_wall_seconds": sum(r["wall_seconds"] for r in seed_results),
            "total_sim_seconds": sum(r["sim_seconds"] for r in seed_results),
            "verification": {"finite": True, "locomotion_unchanged": unchanged, "crossing_checkpoint_loaded": False},
            "gpu_memory": gpu_memory_snapshot(),
            "caveats": ["Scripted-motion screen, not crossing-policy performance.",
                        "MAE/RMSE exclude invalid radar or unmatched exposure frames; their rates are reported.",
                        "Lane vx assumes vehicle velocity along world X only.",
                        "Aggregates weight frames; seed means also available. Actual-motion cuts overlap."]}
        write_json(output / "summary.json", report)
        print("[complete]", json.dumps(report["aggregate"]), flush=True)
    except BaseException:
        code = 1
        import traceback
        write_json(output / "failure.json", {"traceback": traceback.format_exc()})
        raise
    finally:
        if env is not None:
            env.close()
        launcher.app.close(exit_code=code)


if __name__ == "__main__":
    main()
