"""One-seed counterfactual radar screen. No production defaults are modified.

Run cases in separate processes with identical commands and seed; compare the
recorded robot/vehicle trajectories before attributing coverage differences.
The level case removes pitch/roll only, retaining yaw and origin translation.
Ground truth is appended by the evaluator AFTER sensor processing, never used
by radar association or commands. FOV checks are geometric, not visibility.
"""
import argparse
from dataclasses import replace
import math
import sys

import numpy as np


def yaw_rotation(rotation):
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def geometry(center, position, rotation, hfov, vfov):
    p = (np.asarray(center)-position) @ rotation
    r = float(np.linalg.norm(p))
    az = math.degrees(math.atan2(p[1], p[0]))
    el = math.degrees(math.atan2(p[2], math.hypot(p[0], p[1])))
    return {"az_deg": az, "el_deg": el, "range_m": r,
            "center_in_fov": abs(az) <= hfov/2 and abs(el) <= vfov/2 and .5 <= r <= 80,
            "abs_los_x": abs(float(np.asarray(center)[0]-position[0]))/max(r, 1e-9)}


def command_at(t):
    if t < 3:
        return "stand", (0., 0., 0.)
    if t < 6:
        return "walk", (1., 0., 0.)
    if t < 9:
        return "run", (3., 0., 0.)
    if t < 12:
        return "reverse", (-.5, 0., 0.)
    if t < 15:
        return "turn_left", (0., 0., .5)
    return "turn_right", (0., 0., -.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=("baseline", "level", "wide", "outward"))
    args, remaining = parser.parse_known_args()
    import evaluate_radar_tracking as screen
    from camera_perception import rotation_xyzw
    from radar_processing import mounted_pose, rotation_to_wxyz
    # This module has lazy simulator imports; all sensor construction is still
    # performed only after the evaluator starts AppLauncher.
    from h1_radar import H1RadarShadow
    original_load = screen.load_radar_config

    def load_config(path):
        cfg = original_load(path)
        if args.case == "wide":
            return replace(cfg, horizontal_fov_deg=180.)
        if args.case == "outward":
            return replace(cfg, torso_offset=(.22, .40, .35), opposite_torso_offset=(.22, -.40, .35))
        return cfg

    if args.case == "level":
        def follow(self, pose=None, velocity=None):
            if pose is None:
                pose = self.robot.data.body_link_pose_w.torch[self.env_index, self.body].detach().cpu().numpy().copy()
                velocity = self.robot.data.body_link_vel_w.torch[self.env_index, self.body].detach().cpu().numpy().copy()
            position, _, origin_velocity = mounted_pose(pose, velocity, self.cfg)
            torso_yaw = yaw_rotation(rotation_xyzw(pose[3:]))
            angle = math.radians(self.cfg.yaw_deg)
            c, s = math.cos(angle), math.sin(angle)
            rotation = torso_yaw @ np.array([[c,-s,0.], [s,c,0.], [0.,0.,1.]])
            self.pose = position, rotation, origin_velocity
            self.radar.set_world_poses(positions=[position], orientations=[rotation_to_wxyz(rotation)])
        H1RadarShadow.follow = follow

    original_observer = screen.RadarTruthObserver

    class DiagnosticObserver(original_observer):
        def before_render(self, now):
            result = super().before_render(now)
            env = self.env
            torso = env.robot.data.body_link_pose_w.torch[0, env.radar.body].detach().cpu().numpy().copy()
            rotation = rotation_xyzw(torso[3:])
            self.history[-1].update(
                torso_pose=torso.tolist(),
                root_pose=env.robot.data.root_pose_w.torch[0].detach().cpu().tolist(),
                torso_tilt_deg=math.degrees(math.acos(float(np.clip(rotation[2,2], -1, 1)))),
                head_poses=[{"position": h.pose[0].tolist(), "rotation": h.pose[1].tolist()}
                            for h in env.radar.sensors])
            return result

        def capture(self, now, camera_frames, vehicle_size):
            count = len(self.rows)
            result = super().capture(now, camera_frames, vehicle_size)
            if len(self.rows) == count:
                return result
            row = self.rows[-1]
            row["diagnostic_case"] = args.case
            end = row.get("sensor_clock_reference_sim_s", -1)
            duration = (row.get("exposure_end_timestamp_ns", 0)-row.get("exposure_start_timestamp_ns", 0))*1e-9
            a = next((h for h in reversed(self.history) if abs(h["t"]-(end-duration)) < 1e-5), None)
            b = next((h for h in reversed(self.history) if abs(h["t"]-end) < 1e-5), None)
            if a is not None and b is not None:
                row["diagnostic_truth"] = {k: b[k] for k in ("car", "speed", "reset", "pass", "torso_pose", "root_pose", "torso_tilt_deg")}
                row["head_geometry"] = [geometry(a["car"], np.array(h["position"]), np.array(h["rotation"]),
                    sensor.cfg.horizontal_fov_deg, sensor.cfg.vertical_fov_deg)
                    for h, sensor in zip(a["head_poses"], self.env.radar.sensors)]
            row["head_diagnostics"] = [dict(h.latest) if h.latest else None for h in self.env.radar.sensors]
            return result

    screen.command_at = command_at
    screen.load_radar_config = load_config
    screen.RadarTruthObserver = DiagnosticObserver
    # Add exact experiment metadata, including corrected command boundaries,
    # without editing the reusable production evaluator.
    original_write = screen.write_json
    def write_json(path, data):
        if path.name == "protocol.json":
            data.update(diagnostic_case=args.case,
                commands=[{"start_s": t, "phase": command_at(t)[0], "command": command_at(t)[1]}
                          for t in (0,3,6,9,12,15)],
                diagnostic_scope="Pitch/roll-only leveling retains sensor translation and torso yaw; FOV is not occlusion visibility")
        original_write(path, data)
    screen.write_json = write_json
    sys.argv = [sys.argv[0], *remaining]
    screen.main()


if __name__ == "__main__":
    main()
