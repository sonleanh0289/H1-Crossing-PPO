"""Complementary fixed radar heads, with exposure-aligned best-head selection.

Each head has its own clock and lane association. Never merge returns expressed
in different sensor frames, average conflicting targets, hold invalid velocity,
or consult simulator vehicle state. The public latest row remains the 23D
observation provider's measurement interface.
"""
from collections import Counter
from dataclasses import replace, asdict
import json
import math


def select_head(rows, now, cfg):
    """Choose a current, independently valid head; conflict invalidates the rig."""
    present = [(i, r) for i, r in enumerate(rows) if r is not None]
    if not present:
        return None
    valid = [(i, r) for i, r in present if r.get("lane_speed_valid")
             and r.get("timing_valid") and r.get("lane_vx_mps") is not None
             and math.isfinite(r["lane_vx_mps"])
             and 0 <= now-r.get("sensor_clock_reference_sim_s", -math.inf) <= .12+1e-8]
    # Both heads must describe the same exposure. Never combine a late buffer
    # with a newer reading (even when each individually passes the age limit).
    if valid:
        newest = max(r["sensor_clock_reference_sim_s"] for _, r in valid)
        valid = [(i, r) for i, r in valid if abs(r["sensor_clock_reference_sim_s"]-newest) < 1e-6]
    conflict = len(valid) > 1 and (max(r["lane_vx_mps"] for _, r in valid)
                                 - min(r["lane_vx_mps"] for _, r in valid) > cfg.max_head_disagreement_mps)
    pool = valid if valid else present
    i, chosen = max(pool, key=lambda item: (
        item[1].get("median_abs_los_x", 0.),
        -item[1].get("lane_speed_mad_mps", 1e9),
        item[1].get("observable_returns", 0.), -item[0]))
    row = dict(chosen)
    row.update(selected_head=i if valid and not conflict else None,
               head_reasons=[r.get("reason") if r else "missing_buffer" for r in rows],
               head_valid=[any(j == index for j, _ in valid) for index in range(len(rows))])
    if conflict or not valid:
        row.update(lane_speed_valid=False, lane_vx_mps=None,
                   reason="head_velocity_disagreement" if conflict else (
                       "no_current_valid_head" if chosen.get("lane_speed_valid") else row.get("reason", "no_valid_head")))
    return row


def create_radar(robot, cfg, output, **kwargs):
    from h1_radar import H1RadarShadow
    if not cfg.opposite_radar:
        return H1RadarShadow(robot, cfg, output, **kwargs)
    return H1RadarRig(robot, cfg, output, **kwargs)


class H1RadarRig:
    def __init__(self, robot, cfg, output, *, headless=True, observation_lane_y=None,
                 env_index=0, console_debug=False, sensor_factory=None):
        from h1_radar import H1RadarShadow
        factory = sensor_factory or H1RadarShadow
        self.robot, self.cfg, self.env_index = robot, cfg, env_index
        self.observation_mode = observation_lane_y is not None
        self.sensors, self.log = [], None
        self.last_render_s = self.exposure_clock_s = 0.
        self.latest = None
        self.frames = self.valid_frames = self.missing_frames = 0
        self.selections, self.reasons = Counter(), Counter()
        try:
            for name, child_cfg in (("Left", replace(cfg, opposite_radar=False)),
                    ("Right", replace(cfg, opposite_radar=False, yaw_deg=cfg.yaw_deg-180.,
                                      torso_offset=cfg.opposite_torso_offset))):
                self.sensors.append(factory(robot, child_cfg, output, headless=headless,
                    observation_lane_y=observation_lane_y, env_index=env_index,
                    console_debug=console_debug, head_name=name, log_suffix="_head_"+name.lower()))
            self.body = self.sensors[0].body
            name = "radar_frames.jsonl" if env_index == 0 else f"radar_frames_env_{env_index:03d}.jsonl"
            self.log = (output/name).open("w", encoding="utf-8")
        except BaseException:
            self.close()
            raise

    def follow(self, pose=None, velocity=None):
        if pose is None:
            pose = self.robot.data.body_link_pose_w.torch[self.env_index, self.body].detach().cpu().numpy()
            velocity = self.robot.data.body_link_vel_w.torch[self.env_index, self.body].detach().cpu().numpy()
        for sensor in self.sensors:
            sensor.follow(pose, velocity)

    def reset(self, now):
        self.latest = None
        for sensor in self.sensors:
            sensor.reset(now)

    def before_render(self, now):
        from h1_radar import set_radar_render_step
        set_radar_render_step(now-self.last_render_s)
        self.exposure_clock_s, self.last_render_s = self.last_render_s, now

    def capture(self, now, camera_frames, vehicle_size):
        for sensor in self.sensors:
            # Also works when the batch owns the shared render clock.
            sensor.exposure_clock_s, sensor.last_render_s = self.exposure_clock_s, self.last_render_s
            sensor.capture(now, camera_frames, vehicle_size)
        self.latest = select_head([sensor.latest for sensor in self.sensors], now, self.cfg)
        self.frames += 1
        if self.latest is None:
            self.missing_frames += 1
            return
        self.valid_frames += bool(self.latest["lane_speed_valid"])
        self.reasons[self.latest["reason"]] += 1
        if self.latest["selected_head"] is not None:
            self.selections[self.latest["selected_head"]] += 1
        self.log.write(json.dumps(self.latest, allow_nan=False)+"\n")
        if self.frames % 30 == 0:
            self.log.flush()

    def report(self):
        return {"config": asdict(self.cfg), "frames": self.frames,
                "valid_lane_speed_frames": self.valid_frames, "missing_frames": self.missing_frames,
                "selected_head_counts": dict(self.selections), "reasons": dict(self.reasons),
                "heads": [s.report() for s in self.sensors], "shared_render_clock": True,
                "policy_source": ("best valid radar head, camera position, 23D" if self.observation_mode
                                  else "camera only, radar never feeds observations/actions"),
                "physical_payload_added": False}

    def close(self):
        for sensor in self.sensors:
            sensor.close()
        if self.log is not None:
            self.log.close()
            self.log = None
