"""Independent per-environment radar sensors on ONE shared render clock."""
import math


def validate_sensor_spacing(num_envs, spacing, camera_cfg, radar_cfg):
    if num_envs <= 1:
        return
    # Camera clipping is optical depth, not Euclidean range. Account for the
    # diagonal frustum ray and two scene/robot envelopes (60 m radius each).
    tangent = math.tan(math.radians(camera_cfg.horizontal_fov_deg)/2)
    camera_reach = camera_cfg.far_m * math.sqrt(1+tangent**2+(tangent*camera_cfg.height/camera_cfg.width)**2)
    required = 120 + max(camera_reach, radar_cfg.max_range_m) + 2
    if spacing <= required:
        raise ValueError(f"Radar/camera env spacing must exceed {required:.1f} m to isolate scenes; use the 300 m radar config")


class H1RadarBatch:
    def __init__(self, robot, cfg, output, lane_y, *, headless=True, sensor_factory=None, render_step=None):
        from h1_radar import set_radar_render_step
        from radar_rig import create_radar
        factory = sensor_factory or create_radar
        self.render_step = render_step or set_radar_render_step
        self.robot, self.cfg = robot, cfg
        self.last_render_s = self.exposure_clock_s = 0.
        self.sensors = []
        try:
            for index, y in enumerate(lane_y):
                self.sensors.append(factory(robot, cfg, output, headless=headless or index != 0,
                    observation_lane_y=float(y), env_index=index))
        except BaseException:
            self.close()
            raise

    @property
    def latest(self):
        return [sensor.latest for sensor in self.sensors]

    def follow(self):
        body = self.sensors[0].body
        poses = self.robot.data.body_link_pose_w.torch[:, body].detach().cpu().numpy()
        velocities = self.robot.data.body_link_vel_w.torch[:, body].detach().cpu().numpy()
        for index, sensor in enumerate(self.sensors):
            sensor.follow(poses[index], velocities[index])

    def reset(self, now, env_ids=None):
        ids = range(len(self.sensors)) if env_ids is None else env_ids.tolist() if hasattr(env_ids, "tolist") else env_ids
        for index in ids:
            self.sensors[index].reset(now)
        # A partial reset must not rewind the shared render time or other sensors.

    def before_render(self, now):
        self.render_step(now-self.last_render_s)
        self.exposure_clock_s, self.last_render_s = self.last_render_s, now
        for sensor in self.sensors:
            sensor.exposure_clock_s = self.exposure_clock_s
            sensor.last_render_s = now

    def capture(self, now, camera_frames, vehicle_size):
        if len(camera_frames) != len(self.sensors):
            raise ValueError("One camera-frame set is required per radar environment")
        for index, sensor in enumerate(self.sensors):
            sensor.capture(now, camera_frames[index], vehicle_size)

    def report(self):
        return {"num_envs":len(self.sensors), "per_environment":[s.report() for s in self.sensors],
                "shared_render_clock":True, "independent_tracks_and_resets":True,
                "policy_source":"camera position + per-env radar velocity, 23D", "physical_payload_added":False}

    def close(self):
        for sensor in self.sensors:
            sensor.close()
