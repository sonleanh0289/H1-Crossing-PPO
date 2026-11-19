"""Deterministic evaluation: 10 parallel trials x 5 seeds, per scenario."""
from pathlib import Path
import csv
import math
import torch
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from .artifacts import sha256, write_json, output_path
from .checkpoint import load_safe
from .control import STAGES, DT
from .runtime import build, record_run, finite
from .validation import SEEDS, scenarios, gate, passes_scenario
from .yaw_trace import YawTrace
from .stability import STABILITY_METRICS, STABILITY_PROTOCOL, squared_stability_samples


class Measurements:
    def __init__(self, base):
        self.base = base
        self.robot = base.scene["robot"]
        self.feet, _ = self.robot.find_bodies(".*_ankle_link")
        self.contact_feet, _ = base.scene["contact_forces"].find_bodies(".*_ankle_link")
        if len(self.feet) != 2 or len(self.contact_feet) != 2:
            raise ValueError("Expected two H1 ankle bodies for foot sliding metrics")

    def read(self, actions, previous):
        data = self.robot.data
        velocity = quat_apply_inverse(yaw_quat(data.root_quat_w.torch), data.root_lin_vel_w.torch)
        yaw = data.root_ang_vel_w.torch[:, 2]
        gravity = data.projected_gravity_b.torch
        height = data.root_pos_w.torch[:, 2] - self.base.scene.env_origins[:, 2]
        tilt = torch.acos((-gravity[:, 2]).clamp(-1, 1))
        foot_v = data.body_lin_vel_w.torch[:, self.feet, :2].norm(dim=-1)
        contacts = self.base.scene["contact_forces"].data.net_forces_w.torch[:, self.contact_feet].norm(dim=-1) > 1
        slip = (foot_v * contacts).sum(-1) / contacts.sum(-1).clamp_min(1)
        effort = data.applied_torque.torch
        limit = data.joint_effort_limits.torch
        saturation = ((effort.abs() / limit.clamp_min(1e-6)) >= 0.95).float().mean(-1)
        smooth = (actions - previous).square().mean(-1) / DT**2
        finite(velocity, yaw, gravity, height, tilt, slip, effort, limit, smooth)
        fallen = self.base.reset_terminated | (height < 0.65) | (tilt > math.pi / 3)
        return velocity, yaw, tilt, slip, saturation, smooth, fallen


def reset_batch(base, env, term, seed):
    term.manual = True
    base.reset(seed=seed)
    return env.get_observations()


def evaluate(args, app):
    # Validation protocol is fixed, not silently reduced by a smoke flag.
    args.num_envs = 10
    base, env, runner, contract, cfg, agent = build(args)
    directory = Path(args.output)
    checkpoint = load_safe(args.checkpoint, args.device)
    if checkpoint.get("fine_tune") and checkpoint["fine_tune"]["contract"] != contract:
        raise ValueError("Evaluation checkpoint/environment contract mismatch")
    runner.alg.actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    policy = runner.get_inference_policy(device=args.device)
    term = base.command_manager.get_term("base_velocity")
    metrics = Measurements(base)
    stability_enabled = getattr(args, "record_stability_metrics", False)
    if stability_enabled:
        torso_ids, _ = metrics.robot.find_bodies("torso_link")
        if len(torso_ids) != 1:
            raise ValueError("Expected exactly one torso_link for stability metrics")
        torso_id = torso_ids[0]
    record_run(directory, args, args.checkpoint, contract, cfg, agent)
    trial_rows, scenario_rows = [], []
    try:
        with torch.inference_mode(), YawTrace(directory / "yaw_trace.csv",
                enabled=getattr(args, "record_yaw_trace", False)) as trace:
            for name, (vx, yaw_target) in scenarios(args.stage).items():
                scenario_trials = []
                for seed in SEEDS:
                    obs = reset_batch(base, env, term, seed)
                    alive = torch.ones(10, dtype=torch.bool, device=args.device)
                    sums = torch.zeros((10, 8), device=args.device)
                    samples = torch.zeros(10, device=args.device)
                    previous = torch.zeros((10, 19), device=args.device)
                    stability_sum = torch.zeros((10, len(STABILITY_METRICS)), device=args.device)
                    previous_angular = None
                    for step in range(750):
                        if not app.is_running():
                            raise RuntimeError("Evaluation interrupted")
                        term.set_desired_velocity(vx, yaw_rate=yaw_target)
                        actions = policy(obs)
                        finite(actions)
                        obs, rewards, dones, _ = env.step(actions)
                        finite(obs, rewards)
                        vel, yaw, tilt, slip, saturation, smooth, fallen = metrics.read(actions, previous)
                        alive &= ~fallen
                        trace.record(name, seed, step, DT, yaw_target, term.vel_command_b[:, 2], yaw, alive)
                        previous.copy_(actions)
                        if stability_enabled:
                            data = metrics.robot.data
                            angular = data.body_ang_vel_w.torch[:, torso_id]
                            squared = squared_stability_samples(
                                data.body_link_pose_w.torch[:, torso_id, 3:], angular,
                                data.body_lin_vel_w.torch[:, torso_id], previous_angular, DT)
                            previous_angular = angular.clone()
                            if step >= 250:
                                stability_sum += squared * alive[:, None]
                        if step >= 250:
                            values = torch.stack(((vel[:, 0] - vx).abs(), vel[:, 1].abs(),
                                                  (yaw - yaw_target).abs(), vel[:, :2].norm(dim=-1),
                                                  tilt, slip, saturation, smooth), dim=-1)
                            sums += values * alive[:, None]
                            samples += alive
                    means = (sums / samples.clamp_min(1)[:, None]).cpu().tolist()
                    if stability_enabled:
                        stability_rms = (stability_sum / samples.clamp_min(1)[:, None]).sqrt().cpu().tolist()
                    for index in range(10):
                        row = dict(name=name, seed=seed, trial=index, survived=bool(alive[index]),
                                   samples=int(samples[index]), vx=vx, yaw=yaw_target,
                                   **dict(zip(("vx_mae", "vy_mae", "yaw_mae", "planar_speed", "tilt_rad",
                                               "foot_slip_mps", "torque_saturation_fraction", "action_rate_mse"), means[index])))
                        if stability_enabled:
                            row.update(zip(STABILITY_METRICS, stability_rms[index]))
                        scenario_trials.append(row)
                successful = [r for r in scenario_trials if r["survived"] and r["samples"] == 500]
                row = dict(name=name, vx=vx, yaw=yaw_target, rollouts=50, seeds=list(SEEDS),
                           finite=True, survival_rate=len(successful) / 50)
                for key in ("vx_mae", "vy_mae", "yaw_mae", "planar_speed", "tilt_rad",
                            "foot_slip_mps", "torque_saturation_fraction", "action_rate_mse"):
                    row[key] = sum(r[key] for r in successful) / len(successful) if successful else None
                row["passed"] = passes_scenario(row)
                if stability_enabled:
                    for key in STABILITY_METRICS:
                        row[key] = math.sqrt(sum(r[key] ** 2 for r in successful) / len(successful)) if successful else None
                scenario_rows.append(row)
                trial_rows.extend(scenario_trials)
                print(f"[evaluate] {name}: survival={row['survival_rate']:.0%}, pass={row['passed']}", flush=True)

            # 20 seconds: forward 4s -> stop 4s -> reverse 4s -> stop 4s -> forward 4s.
            low, high = STAGES[args.stage]
            sequence_successes = 0
            sequence_rows = []
            for seed in SEEDS:
                obs = reset_batch(base, env, term, seed)
                alive = torch.ones(10, dtype=torch.bool, device=args.device)
                movement_error = torch.zeros((10, 3), device=args.device)
                held = torch.zeros(10, device=args.device)
                stop_ok = torch.zeros((10, 2), dtype=torch.bool, device=args.device)
                previous = torch.zeros((10, 19), device=args.device)
                for step in range(1000):
                    if not app.is_running():
                        raise RuntimeError("Transition evaluation interrupted")
                    phase, offset = divmod(step, 200)
                    vx = (high, 0, low, 0, high)[phase]
                    if offset == 0:
                        held.zero_()
                    term.set_desired_velocity(vx)
                    actions = policy(obs)
                    finite(actions)
                    obs, rewards, dones, _ = env.step(actions)
                    finite(obs, rewards)
                    vel, yaw, tilt, slip, saturation, smooth, fallen = metrics.read(actions, previous)
                    alive &= ~fallen
                    previous.copy_(actions)
                    if phase in (1, 3):
                        held = torch.where(vel[:, :2].norm(dim=-1) < 0.15, held + 1, 0)
                        # A 1s interval completed by 4s implies its start was <=3s.
                        stop_ok[:, phase // 2] |= (held >= 50) & alive
                    elif offset >= 150 and step < 999:
                        # The final timeout resets robot state; exclude that post-reset sample.
                        movement_error[:, phase // 2] += (vel[:, 0] - vx).abs() / (49 if phase == 4 else 50)
                tolerances = movement_error.new_tensor([0.2 if high <= 1 else 0.3, 0.2,
                                                       0.2 if high <= 1 else 0.3])
                succeeded = alive & stop_ok.all(-1) & (movement_error <= tolerances).all(-1)
                sequence_successes += int(succeeded.sum())
                for index in range(10):
                    sequence_rows.append({"seed": seed, "trial": index, "success": bool(succeeded[index]),
                                          "survived": bool(alive[index]), "stops_passed": stop_ok[index].cpu().tolist(),
                                          "movement_vx_mae": movement_error[index].cpu().tolist()})
        report = {"schema_version": 1, "stage": args.stage, "checkpoint_sha256": sha256(args.checkpoint),
                  "checkpoint": str(Path(args.checkpoint).resolve()), "contract": contract,
                  "protocol": "50 trials/scenario; five seeds; fixed 5s ramp + 10s measure; transition 5x4s",
                  "scenarios": scenario_rows,
                  "transition": {"finite": True, "rollouts": 50, "seeds": list(SEEDS),
                                 "success_rate": sequence_successes / 50, "trials": sequence_rows}}
        report["passed"] = gate(report)
        if stability_enabled:
            report["stability_metrics_protocol"] = STABILITY_PROTOCOL
        write_json(directory / "evaluation.json", report)
        write_json(directory / "trials.json", trial_rows)
        with output_path(directory / "trials.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(trial_rows[0]))
            writer.writeheader()
            writer.writerows(trial_rows)
        print(f"[evaluate] stage={args.stage}, passed={report['passed']}: {directory / 'evaluation.json'}", flush=True)
    finally:
        env.close()
