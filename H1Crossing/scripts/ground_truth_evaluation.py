"""Read-only scoring at the exact pre-action simulation timestamp.

In normal sensor play, ground truth is consumed only after the actor action. Errors
measure the input available to the policy now, including sensor latency; they
are NOT exposure-aligned intrinsic Doppler accuracy measurements. Explicit oracle
evaluation logs sensor observations separately from the truth-fed actor input.
"""
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from radar_observations import OBS_CONFIG


def score_decision(obs, h1_xy, car_xy, car_velocity, camera_estimate, radar, now):
    obs = np.asarray(obs, dtype=float).copy()
    state, camera_position_valid, camera_velocity_valid, age = camera_estimate
    state = np.asarray(state, dtype=float).copy()
    car_xy, car_velocity = np.asarray(car_xy), np.asarray(car_velocity)
    radar = radar or {}
    vx, stamp = radar.get("lane_vx_mps"), radar.get("sensor_clock_reference_sim_s")
    radar_valid = bool(radar.get("lane_speed_valid") and vx is not None and stamp is not None
        and math.isfinite(vx) and math.isfinite(stamp)
        and 0 <= now-stamp <= OBS_CONFIG.max_velocity_age_s+1e-8)
    camera_valid = bool(camera_position_valid and camera_velocity_valid and np.isfinite(state[2:]).all())
    source = "radar" if radar_valid else "camera_fallback" if camera_valid else "invalid"
    expected = [vx, 0.] if radar_valid else state[2:] if camera_valid else [0., 0.]
    actual_velocity = obs[4:6]*15.
    position_valid, velocity_valid = bool(obs[19] > .5), bool(obs[20] > .5)
    # behavior_observation clamps normalized values to [-10, 10].
    expected = np.clip(expected, -150., 150.)
    if velocity_valid != (source != "invalid") or not np.allclose(actual_velocity, expected, atol=1e-4, rtol=1e-5):
        raise AssertionError("Diagnostic source selection disagrees with the actual actor input")
    actual_position = np.asarray(h1_xy)+obs[:2]*[50., 10.]
    error = lambda a, b: float(np.linalg.norm(np.asarray(a)-b))
    row = {
        "simulation_time_s": float(now), "observation": obs.tolist(),
        "gt_car_position_world_xy_m": car_xy.tolist(),
        "gt_car_velocity_world_xy_mps": car_velocity.tolist(),
        "h1_position_world_xy_m": np.asarray(h1_xy).tolist(),
        "camera_position_valid": position_valid,
        "camera_position_world_xy_m": actual_position.tolist() if position_valid else None,
        "camera_position_error_m": error(actual_position, car_xy) if position_valid else None,
        "camera_velocity_valid": camera_valid,
        "camera_velocity_world_xy_mps": state[2:].tolist() if camera_valid else None,
        "camera_velocity_error_mps": error(state[2:], car_velocity) if camera_valid else None,
        "camera_track_age_normalized": float(age),
        "radar_velocity_valid": radar_valid,
        "radar_velocity_world_xy_mps": [float(vx), 0.] if radar_valid else None,
        "radar_velocity_error_mps": error([vx, 0.], car_velocity) if radar_valid else None,
        "radar_age_s": float(now-stamp) if stamp is not None and math.isfinite(stamp) else None,
        "radar_reason": radar.get("reason", "no_frame"),
        "velocity_source": source, "selected_velocity_valid": velocity_valid,
        "selected_velocity_world_xy_mps": actual_velocity.tolist(),
        "selected_velocity_error_mps": error(actual_velocity, car_velocity) if velocity_valid else None,
    }
    json.dumps(row, allow_nan=False)
    return row


def error_summary(rows, field):
    values = np.asarray([r[field] for r in rows if r[field] is not None], dtype=float)
    n, total = len(values), len(rows)
    return {"total": total, "valid": n, "invalid": total-n,
        "valid_rate": n/total if total else None,
        "mean": float(values.mean()) if n else None,
        "rmse": float(np.sqrt(np.mean(values**2))) if n else None,
        "p95": float(np.percentile(values, 95)) if n else None,
        "max": float(values.max()) if n else None}


def attach_oracle_input(row, observation):
    """Keep sensor diagnostics intact, but identify the actual actor input."""
    obs = np.asarray(observation, dtype=float)
    if obs.shape != (23,) or not np.isfinite(obs).all() or not np.all(obs[19:21] > .5):
        raise ValueError("Oracle actor input must be finite 23D with valid position/velocity")
    row["sensor_observation"] = row["observation"]
    row["sensor_velocity_source"] = row["velocity_source"]
    row["sensor_selected_velocity_valid"] = row["selected_velocity_valid"]
    row["sensor_selected_velocity_error_mps"] = row["selected_velocity_error_mps"]
    row["observation"] = obs.tolist()
    row["ground_truth_to_actor"] = True
    row["actor_input_source"] = "ground_truth_oracle"
    position = np.asarray(row["h1_position_world_xy_m"])+obs[:2]*[50., 10.]
    velocity = obs[4:6]*15.
    row["actor_position_world_xy_m"] = position.tolist()
    row["actor_position_error_m"] = float(np.linalg.norm(position-row["gt_car_position_world_xy_m"]))
    row["velocity_source"] = "ground_truth_oracle"
    row["selected_velocity_valid"] = True
    row["selected_velocity_world_xy_mps"] = velocity.tolist()
    row["selected_velocity_error_mps"] = float(np.linalg.norm(velocity-row["gt_car_velocity_world_xy_mps"]))
    return row


def summarize(rows, completed, ground_truth_to_actor=False):
    fields = ("camera_position_error_m", "camera_velocity_error_mps",
              "radar_velocity_error_mps", "selected_velocity_error_mps")
    return {"completed": completed, "decision_rows": len(rows),
        "scope": "Pre-action errors against current world-XY truth, including latency. Not intrinsic exposure-aligned radar accuracy.",
        "ground_truth_velocity_source": "Scripted traffic.speed along world X; zero world Y",
        "ground_truth_to_actor": ground_truth_to_actor,
        "camera_metrics_source": "Sensor pipeline, even in oracle mode; selected velocity is the actor input",
        "errors": {f: error_summary(rows, f) for f in fields},
        "velocity_source_counts": dict(Counter(r["velocity_source"] for r in rows)),
        "selected_velocity_by_source": {source: error_summary(
            [r for r in rows if r["velocity_source"] == source], "selected_velocity_error_mps")
            for source in ("radar", "camera_fallback", "invalid", "ground_truth_oracle")}}


class GroundTruthEvaluation:
    def __init__(self, output, num_envs, ground_truth_to_actor=False):
        self.ground_truth_to_actor = ground_truth_to_actor
        self.output = Path(output)
        self.log = (self.output / "ground_truth_decisions.jsonl").open("x", encoding="utf-8")
        self.rows = []
        self.episodes = [0]*num_envs
        self.completed = False

    def snapshot(self, env, observation, actions):
        # Copies only. Never alter observations, tracks, actions or simulation.
        obs = observation["behavior"].detach().cpu().numpy().copy()
        sensor_obs = (env.get_observations(sensor_only=True)["behavior"].detach().cpu().numpy().copy()
                      if self.ground_truth_to_actor else obs)
        h1 = env.robot.data.root_pos_w.torch[:, :2].detach().cpu().numpy().copy()
        car = env.vehicle.data.root_pos_w.torch[:, :2].detach().cpu().numpy().copy()
        speed = env.traffic.speed.detach().cpu().numpy().copy()
        passes = env.traffic.pass_count.detach().cpu().tolist()
        action = actions.detach().cpu().tolist()
        h1_velocity = env.robot.data.root_lin_vel_w.torch[:, :2].detach().cpu().numpy().copy()
        radar = env.radar.latest
        radar = [radar] if radar is None or isinstance(radar, dict) else radar
        batch = []
        for i, track in enumerate(env.perception.tracks):
            row = score_decision(sensor_obs[i], h1[i], car[i], [float(speed[i]), 0.],
                                 track.estimate(env.sim_time), radar[i], env.sim_time)
            row.update(ground_truth_to_actor=False, actor_input_source="camera_radar")
            if self.ground_truth_to_actor:
                attach_oracle_input(row, obs[i])
            row.update(env_index=i, episode_index=self.episodes[i], vehicle_pass=int(passes[i]),
                       h1_speed_mps=float(np.linalg.norm(h1_velocity[i])),
                       decision_index=len(self.rows)//len(self.episodes), action=action[i],
                       camera_track_id=getattr(track, "track_id", None),
                       camera_track_update_status=getattr(track, "last_update_status", None))
            barrier = getattr(env.perception, "reset_barrier", None)
            if barrier is not None:
                # Diagnostics only, never appended to the actor observation.
                row.update(barrier.snapshot(i))
            if hasattr(track, "position_diagnostics"):
                row.update(track.position_diagnostics(env.sim_time))
            batch.append(row)
        return batch

    def record(self, batch, done, extras):
        dones = done.detach().cpu().tolist()
        events = {k: v.detach().cpu().tolist() for k, v in extras["events"].items()}
        for i, row in enumerate(batch):
            row["done_after_action"] = bool(dones[i])
            row["events_after_action"] = {k: bool(v[i]) for k, v in events.items()}
            row["command_guard_after_action"] = {
                k: v[i].item() for k, v in extras.get("command_guard", {}).items()}
            self.log.write(json.dumps(row, allow_nan=False)+"\n")
            self.rows.append(row)
            self.episodes[i] += int(bool(dones[i]))
        self.log.flush()

    def close(self):
        self.log.close()
        from motion_quality import sensor_motion_summary
        (self.output/'sensor_motion_quality_summary.json').write_text(json.dumps(
            dict(completed=self.completed,**sensor_motion_summary(self.rows)),indent=2,allow_nan=False),encoding='utf-8')
        (self.output / "ground_truth_summary.json").write_text(
            json.dumps(summarize(self.rows, self.completed, self.ground_truth_to_actor), indent=2, allow_nan=False), encoding="utf-8")
