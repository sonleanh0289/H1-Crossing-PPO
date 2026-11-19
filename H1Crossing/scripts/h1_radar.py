"""Opt-in RTX radar mounted virtually on the torso; shadow or observation mode.

Import only after AppLauncher. No simulator vehicle handle, object IDs, policy,
reward or ground-truth velocity is accessible to this class.
"""
from dataclasses import asdict
import json
import numpy as np
from radar_processing import mounted_pose, rotation_to_wxyz, RadarClock, diagnose_returns, exposure_geometry
from camera_perception import fuse


MOTION_BVH_ARGS = ("--/renderer/raytracingMotion/enabled=true "
                   "--/renderer/raytracingMotion/enableHydraEngineMasking=true "
                   "--/renderer/raytracingMotion/enabledForHydraEngines=0,1,2,3,4 "
                   "--/app/settings/fabricDefaultStageFrameHistoryCount=3 "
                   "--/app/useFabricSceneDelegate=true "
                   "--/rtx/hydra/supportMultiTickRate=false --/rtx/rendering/perSensorTickTlas=false "
                   "--/isaaclab/cameras_enabled=true")


def radar_experience():
    """Use the installed base experience only for opt-in radar diagnostics.

    The local Lab headless experience produced only dummy returns, including in
    NVIDIA's static test scene. The base experience passed the same test and
    moving-target Doppler calibration. Do not modify either installed experience.
    """
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "apps/h1_radar.kit"
    if not path.is_file():
        raise RuntimeError(f"Radar reference experience not found: {path}")
    return str(path)


def prepare_radar_clock(sample_hz):
    """Call before environment/tensor initialization, never after sim.reset()."""
    # The reference experience eagerly loads a submodule whose name shadows
    # Lab's lazy-exported decorator. Restore that documented export in this
    # process only, without editing either installed package or external tasks.
    import types
    import isaaclab.utils as lab_utils
    from isaaclab.utils.configclass import configclass
    if isinstance(lab_utils.configclass, types.ModuleType):
        lab_utils.configclass = configclass
    from isaacsim.core.rendering_manager import RenderingManager
    import carb
    import omni.timeline
    RenderingManager.set_dt(1/sample_hz)
    carb.settings.get_settings().set_bool("/app/player/useFixedTimeStepping", False)
    omni.timeline.get_timeline_interface().play()


def set_radar_render_step(dt):
    """Advance render time by the actual 200 Hz scheduler interval.

    Do not call RenderingManager.set_dt during live physics: changing the USD
    timeCodesPerSecond metadata invalidates tensor views on this installation.
    """
    from omni.kit.loop import _loop
    if not 0 < dt <= .1:
        raise ValueError("Invalid radar render step")
    runner = _loop.acquire_loop_interface()
    runner.set_manual_step_size(dt)
    runner.set_manual_mode(True)


def enable_radar_extension():
    """Enable the installed extension in this app only; do not install packages."""
    import omni.kit.app
    manager = omni.kit.app.get_app().get_extension_manager()
    if not manager.set_extension_enabled_immediate("isaacsim.sensors.experimental.rtx", True):
        raise RuntimeError("Installed RTX radar extension could not be enabled")


def radar_attributes(cfg):
    prefix = "omni:sensor:WpmDmat:scan:s001:"
    return {"omni:sensor:WpmDmat:elementsCoordsType": "CARTESIAN",
            "omni:sensor:WpmDmat:outputFrameOfReference": "SENSOR",
            "omni:sensor:WpmDmat:instanceTimeOffsetUsec": 0,
            "omni:sensor:WpmDmat:traceTreeDepth": 3,
            prefix+"maxAzAngDeg": cfg.horizontal_fov_deg/2,
            prefix+"maxElAngDeg": cfg.vertical_fov_deg/2,
            prefix+"maxRangeM": cfg.max_range_m,
            prefix+"raysPerDeg": cfg.rays_per_degree}


class H1RadarShadow:
    def __init__(self, robot, cfg, output, *, headless=True, observation_lane_y=None, env_index=0,
                 console_debug=False, head_name="Left", log_suffix=""):
        import carb
        enable_radar_extension()
        from isaacsim.sensors.experimental.rtx import Radar, RadarSensor
        from pxr import UsdPhysics
        if not carb.settings.get_settings().get("/renderer/raytracingMotion/enabled"):
            raise RuntimeError("Radar requires Motion BVH at application startup")
        self.robot, self.cfg = robot, cfg
        self.env_index = env_index
        # Console diagnostics are opt-in; keep the JSONL measurements unchanged.
        self.console_debug = console_debug
        self.observation_mode = observation_lane_y is not None
        self.association = None
        if self.observation_mode:
            from radar_observations import LaneRadarAssociation
            self.association = LaneRadarAssociation(observation_lane_y)
        self.body = list(robot.body_names).index("torso_link")
        self.clock = RadarClock()
        self.latest = None
        self.pose = None
        self.last_render_s = 0.
        self.exposure_clock_s = 0.
        self.warm_until = 0.
        self.frames = self.empty_frames = self.associated_frames = self.valid_speed_frames = 0
        self.duplicates = self.reset_count = 0
        self.ratios = []
        self.times = []
        self.window = self.label = self.log = self.sensor = None
        self.radar = Radar(f"/World/envs/env_{env_index}/H1Radar{head_name}", aux_output_level="BASIC",
                           tick_rate=cfg.sample_hz, attributes=radar_attributes(cfg))
        prim = self.radar.prims[0]
        # A sensor Xform is not a payload. Do not add MassAPI/CollisionAPI or a
        # rigid body (including a zero-mass rigid body) to the articulation.
        for schema in (UsdPhysics.RigidBodyAPI, UsdPhysics.CollisionAPI, UsdPhysics.MassAPI):
            if prim.HasAPI(schema):
                raise RuntimeError("Radar preview must not add physics properties")
        try:
            self.follow()
            self.sensor = RadarSensor(self.radar)
            self.sensor.attach_annotators("generic-model-output")
            log_base = "radar_frames" if env_index == 0 else f"radar_frames_env_{env_index:03d}"
            log_name = log_base+log_suffix+".jsonl"
            self.log = (output/log_name).open("w", encoding="utf-8")
            if not headless:
                import omni.ui as ui
                self.window = ui.Window(f"H1 radar {head_name} - " + ("OBSERVATION" if self.observation_mode else "SHADOW ONLY"), width=570, height=160)
                with self.window.frame:
                    with ui.VStack():
                        ui.Label(f"{head_name} torso radar | no mass | " + ("23D observation velocity" if self.observation_mode else "NOT used by the policy"), height=24)
                        self.label = ui.Label("Warming up real RTX returns...", word_wrap=True)
            print(f"[Radar] Generic sensor, observation_mode={self.observation_mode}; no physical payload.", flush=True)
        except BaseException:
            self.close()
            raise

    def follow(self, pose=None, velocity=None):
        if pose is None:
            pose = self.robot.data.body_link_pose_w.torch[self.env_index, self.body].detach().cpu().numpy().copy()
            velocity = self.robot.data.body_link_vel_w.torch[self.env_index, self.body].detach().cpu().numpy().copy()
        self.pose = mounted_pose(pose, velocity, self.cfg)
        pos, rotation, _ = self.pose
        self.radar.set_world_poses(positions=[pos], orientations=[rotation_to_wxyz(rotation)])

    def reset(self, now):
        self.clock.reset()
        if self.association is not None:
            self.association.reset()
        self.latest = None
        self.warm_until = now+self.cfg.reset_warmup_s
        self.reset_count += 1
        if self.label is not None:
            self.label.text = "Reset: radar warming up; no current velocity estimate"

    def before_render(self, now):
        set_radar_render_step(now-self.last_render_s)
        # The synchronous native GMO buffer describes the preceding render
        # interval (verified by frameStart/frameEnd poses in the calibration).
        self.exposure_clock_s = self.last_render_s
        self.last_render_s = now

    def capture(self, now, camera_frames, vehicle_size):
        from isaacsim.sensors.experimental.rtx import parse_generic_model_output_data
        raw, _ = self.sensor.get_data("generic-model-output")
        if isinstance(raw, dict):
            raw = raw["data"]
        if raw is None:
            self.empty_frames += 1
            self.latest = None
            if self.label is not None:
                self.label.text = "No current radar buffer; velocity invalid"
            return
        gmo = parse_generic_model_output_data(raw)
        fresh, timing_valid, ratio = self.clock.accept(int(gmo.timestampNs), self.exposure_clock_s)
        if not fresh:
            self.duplicates += 1
            self.latest = None
            if self.label is not None:
                self.label.text = "Repeated radar buffer; velocity invalid"
            return
        count = int(gmo.numElements)
        points = np.column_stack((gmo.x, gmo.y, gmo.z)).copy() if count else np.empty((0, 3))
        radial = np.asarray(gmo.rv_ms).copy() if count else np.empty(0)
        if radial.shape != (count,):
            raise RuntimeError("Radar Doppler channel unavailable; BASIC auxiliary output is required")
        camera_time, frames = camera_frames if camera_frames is not None else (now, [])
        measurement = fuse([f[-1] for f in frames]) if frames else None
        # Native GMO contains an older exposure than the current torso state.
        # Its orientation is XYZW (unlike the Radar authoring API's WXYZ).
        try:
            position, rotation, velocity = exposure_geometry(
                gmo.frameStart.posM, gmo.frameStart.orientation, int(gmo.frameStart.timestampNs),
                gmo.frameEnd.posM, int(gmo.frameEnd.timestampNs))
        except ValueError:
            self.empty_frames += 1
            self.latest = None
            if self.label is not None:
                self.label.text = "Exposure metadata unavailable; velocity invalid"
            return
        common = (points, radial, position, rotation, velocity,
            measurement.center if measurement is not None else None, now-camera_time,
            vehicle_size, self.cfg)
        verified = timing_valid and now >= self.warm_until
        if self.association is not None:
            ego_xy = self.robot.data.root_pos_w.torch[self.env_index, :2].detach().cpu().numpy()
            result = self.association.process(*common, timing_valid=verified,
                now=self.exposure_clock_s, ego_xy=ego_xy)
        else:
            result = diagnose_returns(*common, timing_valid=verified)
        self.frames += 1
        self.empty_frames += count == 0
        self.associated_frames += result["associated"]
        self.valid_speed_frames += result["lane_speed_valid"]
        self.times.append(now)
        if ratio is not None:
            self.ratios.append(ratio)
        row = {"env_index": self.env_index, "sim_capture_s": now, "sensor_timestamp_ns": int(gmo.timestampNs),
               "sensor_clock_reference_sim_s": self.exposure_clock_s,
               "sensor_to_sim_dt_ratio": ratio, "timing_valid": timing_valid,
               "sensor_position_world": position.tolist(), "sensor_velocity_world": velocity.tolist(),
               "exposure_start_timestamp_ns": int(gmo.frameStart.timestampNs),
               "exposure_end_timestamp_ns": int(gmo.frameEnd.timestampNs),
               "camera_measurement_time_s": camera_time, "shadow_only": not self.observation_mode, **result}
        self.latest = row
        self.log.write(json.dumps(row, allow_nan=False)+"\n")
        if self.frames % 30 == 0:
            self.log.flush()
            if self.console_debug:
                print(f"[radar env={self.env_index}] returns={count}, matched={result['associated_returns']}, "
                      f"vr={result['radial_relative_mps']}, lane_vx={result['lane_vx_mps']}, "
                      f"status={result['reason']}", flush=True)
        if self.label is not None:
            self.label.text = (f"Returns: {count} | associated: {result['associated_returns']}\n"
                f"Radial relative: {result['radial_relative_mps']} m/s\n"
                f"Lane vx (assumption): {result['lane_vx_mps']} m/s\n"
                f"Status: {result['reason']} | " + ("velocity feeds 23D observation" if self.observation_mode else "policy remains camera-only"))

    def report(self):
        return {"config": asdict(self.cfg), "frames": self.frames, "empty_frames": self.empty_frames,
                "associated_frames": self.associated_frames, "valid_lane_speed_frames": self.valid_speed_frames,
                "duplicate_buffers": self.duplicates, "resets": self.reset_count,
                "observed_capture_hz_sim": (len(self.times)-1)/(self.times[-1]-self.times[0]) if len(self.times)>1 else None,
                "sensor_to_sim_dt_ratio_median": float(np.median(self.ratios)) if self.ratios else None,
                "policy_source": "camera position + radar velocity, 23D" if self.observation_mode else "camera only, radar never feeds observations/actions",
                "physical_payload_added": False,
                "scope": "Generic radar diagnostics; no sensor accuracy or crossing safety claim"}

    def close(self):
        if self.log is not None:
            self.log.close()
            self.log = None
        if self.window is not None:
            self.window.destroy()
            self.window = self.label = None
        if self.sensor is not None:
            self.sensor.detach_annotators("generic-model-output")
            # Installed 6.0.1 has no public close(); destruction releases its render product.
            self.sensor = None
