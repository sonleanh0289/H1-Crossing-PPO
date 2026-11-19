"""Opt-in GT23 play diagnostics. Never feed diagnostic values back into control."""
import json
from pathlib import Path
import torch
from behavior_control import yaw_from_xyzw, speed_from_action
from crossing_observations import planar_ttc


def state_snapshot(env):
    robot = env.robot.data
    h1 = robot.root_pos_w.torch[:, :2]
    velocity = robot.root_lin_vel_w.torch[:, :2]
    car = env.vehicle.data.root_pos_w.torch[:, :2]
    car_velocity = torch.stack((env.traffic.speed, torch.zeros_like(env.traffic.speed)), 1)
    yaw = yaw_from_xyzw(robot.root_quat_w.torch)
    relative = car - h1
    ttc, valid = planar_ttc(relative, car_velocity - velocity,
                            vehicle_size_xy=env.layout.vehicle_size[:2])
    # Nominal axis-aligned footprint; diagnostic only, not a contact detector.
    extent = relative.new_tensor([env.layout.vehicle_size[0]/2+.35,
                                  env.layout.vehicle_size[1]/2+.35])
    outside = (relative.abs() - extent).clamp_min(0)
    values = {
        "h1_world_xy_m": h1, "h1_local_xy_m": h1-env.origins[:, :2],
        "h1_world_velocity_xy_mps": velocity,
        "h1_forward_velocity_mps": velocity[:, 0]*yaw.cos()+velocity[:, 1]*yaw.sin(),
        "h1_yaw_rad": yaw, "car_world_xy_m": car, "car_relative_xy_m": relative,
        "car_velocity_xy_mps": car_velocity, "vehicle_pass": env.traffic.pass_count,
        "driver_will_yield": env.traffic.will_yield, "driver_is_yielding": env.traffic.is_yielding,
        "command_body_vx_vy_yaw": env.term.vel_command_b,
        "ttc_s": ttc[:, 0], "ttc_valid": valid[:, 0],
        "nominal_footprint_gap_m": torch.linalg.vector_norm(outside, dim=1),
        "goal_distance_m": torch.linalg.vector_norm(env.goal_xy-h1, dim=1),
        "episode_elapsed_s": env.elapsed,
    }
    arrays = {k: v.detach().cpu().tolist() for k, v in values.items()}
    snapshots = [dict(simulation_time_s=float(env.sim_time), **{k: v[i] for k, v in arrays.items()})
                 for i in range(env.num_envs)]
    if 'torso_link' in getattr(env.robot,'body_names',()):
        from motion_quality import torso_sample
        samples=torso_sample(robot,list(env.robot.body_names).index('torso_link'))
        for state,sample in zip(snapshots,samples):state.update(sample)
    return snapshots


class BehaviorTrajectory:
    def __init__(self, output, num_envs):
        self.output = Path(output)
        self.stream = (self.output / "trajectory.jsonl").open("w", encoding="utf-8")
        self.episodes = [0]*num_envs
        self.index = 0
        self.rows = 0
        self.completed = False
        self.pending = None
        self.motion_samples = []

    def begin(self, env, obs, actions):
        before = state_snapshot(env)
        speeds = speed_from_action(actions, env.settings.action_deadband).cpu().tolist()
        observations, action_values = obs["behavior"].cpu().tolist(), actions.cpu().tolist()
        self.pending = [dict(decision_index=self.index, env_id=i, episode_id=self.episodes[i],
            observation=observations[i], action_raw=action_values[i], requested_speed_mps=speeds[i],
            before=before[i], lowlevel_steps=[]) for i in range(env.num_envs)]

    def substep(self, env, active, desired, target, applied, parts, collision, fallen, contact_peak):
        after = state_snapshot(env)
        vals = [x.detach().cpu().tolist() for x in (active, desired, target, applied,
            parts["success"], collision, fallen, parts["truncated"], parts["out_of_bounds_done"],
            parts["terminated"] | parts["truncated"], contact_peak)]
        for i, row in enumerate(self.pending):
            if not vals[0][i]:
                continue  # Do not record dead-body drift after the first terminal step.
            row["lowlevel_steps"].append(dict(
                command_time_s=float(env.sim_time-env.base.step_dt),
                desired_body_vx_vy_yaw=vals[1][i], target_body_vx_vy_yaw=vals[2][i],
                applied_body_vx_vy_yaw=vals[3][i], after=after[i],
                events=dict(success=vals[4][i], collision=vals[5][i], fall=vals[6][i],
                            timeout=vals[7][i], out_of_bounds=vals[8][i]),
                done=vals[9][i], contact_peak_since_decision_n=vals[10][i]))
            self.motion_samples.append((i,self.episodes[i],after[i],bool(vals[9][i])))

    def finish(self, rewards, done, extras):
        rewards, done = rewards.cpu().tolist(), done.cpu().tolist()
        events = {k: v.cpu().tolist() for k, v in extras["events"].items()}
        for i, row in enumerate(self.pending):
            row.update(reward=rewards[i], done=bool(done[i]), events={k: v[i] for k,v in events.items()})
            self.stream.write(json.dumps(row, allow_nan=False)+"\n")
            self.episodes[i] += int(bool(done[i]))
            self.rows += 1
        self.stream.flush()
        self.pending = None
        self.index += 1

    def close(self):
        self.stream.close()
        from motion_quality import vibration_summary
        (self.output/'motion_quality_summary.json').write_text(json.dumps(
            dict(completed=self.completed,**vibration_summary(self.motion_samples)),indent=2,allow_nan=False),encoding='utf-8')
        (self.output / "trajectory_summary.json").write_text(json.dumps(dict(
            schema="h1-behavior-trajectory-v1", completed=self.completed,
            decisions=self.index, rows=self.rows, completed_episodes_per_env=self.episodes,
            behavior_hz=10, lowlevel_hz=50,
            note="Substeps sample active environments before automatic reset. Contact peak accumulates 200Hz contacts; exact contact time is only localized to a 20ms interval. Vehicle pass changes mark respawn. Footprint gap is nominal, not measured mesh clearance. Driver flags are diagnostic only."), indent=2), encoding="utf-8")
