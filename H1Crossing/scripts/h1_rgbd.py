"""Physical torso-following RGB-D sensors. Import after AppLauncher only."""
import math
import cv2
import numpy as np
import torch
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils.math import quat_mul, quat_apply
from camera_perception import Track, RoadRegion, detect, fuse, observation_from_tracks, camera_frame_due, CAMERA_YAWS
from camera_preview import preview_size, preview_frame
from camera_reset_barrier import CameraResetBarrier


def add_cameras(scene_cfg, cfg):
    aperture = 20.955
    for name in cfg.camera_names:
        setattr(scene_cfg, name+"_rgbd", CameraCfg(
            # Capture is explicitly scheduled by the physics clock. A 1/30 sensor
            # throttle would incorrectly skip the 30-ms interval in a 7/7/6 cycle.
            prim_path="{ENV_REGEX_NS}/"+name.title()+"RGBD", update_period=0.,
            width=cfg.width, height=cfg.height, data_types=["rgb", "distance_to_image_plane"],
            update_latest_camera_pose=True,
            spawn=PinholeCameraCfg(focal_length=aperture/(2*math.tan(math.radians(cfg.horizontal_fov_deg)/2)),
                horizontal_aperture=aperture, vertical_aperture=aperture*cfg.height/cfg.width,
                clipping_range=(cfg.near_m, cfg.far_m))))


class RGBDPerception:
    def __init__(self, robot, cameras, origins, cfg, debug=False, headless=True, *, road_length, road_width):
        # Intentionally does NOT accept an environment, car, or traffic controller.
        self.robot, self.cameras, self.origins, self.cfg = robot, cameras, origins, cfg
        self.camera_names = cfg.camera_names
        if len(cameras) != len(self.camera_names):
            raise ValueError("Sensor count must match the configured camera rig")
        self.body = list(robot.body_names).index("torso_link")
        self.road_regions = [RoadRegion(tuple(origin[:2]), road_length, road_width, cfg.road_margin_m)
                             for origin in origins.detach().cpu().tolist()]
        self.tracks = [Track(cfg, region) for region in self.road_regions]
        self.detector_rejections = {}
        self.debug, self.headless = debug, headless
        self.frames = [None for _ in self.tracks]
        self.reset_barrier = CameraResetBarrier(len(self.tracks))
        self.exposures = 0
        self.detections = 0
        self.valid_frames = 0
        self.first_valid_times = []
        self.reset_times = [0. for _ in self.tracks]
        self.seen_velocity = [False for _ in self.tracks]
        self.was_valid = [False for _ in self.tracks]
        self.track_losses = 0
        self.reacquisitions = 0
        self.window = None
        self.image_provider = None
        if debug and not headless:
            import omni.ui as ui
            self.image_provider = ui.ByteImageProvider()
            tile_width, tile_height = preview_size(cfg.width, cfg.height)
            panel_width = tile_width*len(self.camera_names)
            # Dynamic provider uploads need a live window, not a cached blank
            # raster captured before the first camera exposure arrives.
            self.window = ui.Window("H1 RGB-D perception", width=panel_width+20,
                                    height=tile_height+65, raster_policy=ui.RasterPolicy.NEVER)
            with self.window.frame:
                with ui.VStack():
                    ui.Label("Sensor RGB + depth | "+" / ".join(self.camera_names)+" | no vehicle ground truth", height=22)
                    self.preview_image = ui.ImageWithProvider(self.image_provider,
                        width=panel_width, height=tile_height,
                        fill_policy=ui.IwpFillPolicy.IWP_STRETCH)

    def reset(self, ids, now):
        indices = ids.tolist()
        for i in indices:
            self.tracks[i].reset()
            self.frames[i] = None
            self.reset_times[i] = now
            self.seen_velocity[i] = False
            self.was_valid[i] = False
        self.reset_barrier.reset(indices, now)
        # base.reset() may already have rendered before crossing places its car.
        # Refresh mounts and invalidate sensor buffers AFTER that placement.
        # No extra render here: radar shares the regular simulation-clock cadence.
        self.follow()
        for camera in self.cameras:
            camera.reset(env_ids=ids)

    def due(self, physics_step, physics_dt):
        return camera_frame_due(physics_step, physics_dt, self.cfg.frequency_hz)

    def follow(self):
        pose = self.robot.data.body_link_pose_w.torch[:, self.body]
        offset = pose.new_tensor(self.cfg.torso_offset).expand(len(pose), -1)
        position = pose[:, :3]+quat_apply(pose[:, 3:], offset)
        for camera, name in zip(self.cameras, self.camera_names):
            yaw = CAMERA_YAWS[name]
            local = pose.new_tensor([0., 0., math.sin(yaw/2), math.cos(yaw/2)]).expand(len(pose), -1)
            camera.set_world_poses(position, quat_mul(pose[:, 3:], local), convention="world")

    def capture(self, now):
        views = []
        for camera in self.cameras:
            camera.update(0., force_recompute=True)
            data = camera.data
            # Clone as one exposure before any subsequent physics/render update.
            views.append(tuple(t.torch.detach().cpu().numpy().copy() for t in (
                data.output["rgb"], data.output["distance_to_image_plane"],
                data.intrinsic_matrices, data.pos_w, data.quat_w_ros)))
        for i, track in enumerate(self.tracks):
            self.exposures += 1
            if not self.reset_barrier.admit(i, now):
                self.frames[i] = None
                continue
            detections, frames = [], []
            for rgb, depth, intrinsics, pos, quat in views:
                candidates = detect(rgb[i], depth[i], intrinsics[i], pos[i], quat[i], self.cfg,
                              road_z=float(self.origins[i, 2]), road_region=self.road_regions[i],
                              diagnostics=self.detector_rejections, return_candidates=True)
                item = track.select_detection(now, candidates)
                detections.append(item)
                frames.append((rgb[i], depth[i], intrinsics[i], pos[i], quat[i], item))
            measurement = fuse(detections)
            track.update(now, measurement)
            state, valid, velocity_valid, age = track.estimate(now)
            if self.was_valid[i] and not velocity_valid:
                self.track_losses += 1
            if velocity_valid and not self.was_valid[i]:
                if self.seen_velocity[i]:
                    self.reacquisitions += 1
                else:
                    self.first_valid_times.append(now-self.reset_times[i])
                    self.seen_velocity[i] = True
            self.was_valid[i] = velocity_valid
            self.frames[i] = (now, frames)
            self.detections += measurement is not None
            self.valid_frames += valid
            if self.debug and i == 0 and not self.headless:
                images = []
                for name, frame in zip(self.camera_names, frames):
                    distance = np.linalg.norm(state[:2]-frame[3][:2])
                    picture = preview_frame(frame[0],
                        frame[-1].bbox if frame[-1] is not None else None,
                        (f"{name}: valid={valid}/{velocity_valid}",
                         f"d={distance:.1f}m vx={state[2]:.1f} age={age*.3:.2f}s"))
                    images.append(picture)
                rgba = cv2.cvtColor(np.concatenate(images, axis=1), cv2.COLOR_BGR2RGBA)
                self.image_provider.set_bytes_data(list(rgba.tobytes()), [rgba.shape[1], rgba.shape[0]])
                self.window.frame.invalidate_raster()

    def observations(self, now, h1_xy, h1_velocity, goal_xy):
        return observation_from_tracks(self.tracks, now, h1_xy, h1_velocity, goal_xy)

    def report(self):
        return {"camera_names": list(self.camera_names), "camera_count": len(self.camera_names),
                "reset_capture_barrier": self.reset_barrier.report(),
                "detection_filter": self.cfg.detection_filter,
                "detector_rejected_candidates": dict(self.detector_rejections),
                "tracker_rejections_by_env": [dict(t.rejections) for t in self.tracks],
                "track_lifecycle": self.cfg.track_lifecycle,
                "association_rejected_candidates_by_env": [dict(t.association_rejections) for t in self.tracks],
                "tracks_created_by_env": [t.tracks_created for t in self.tracks],
                "tracks_lost_by_env": [t.tracks_lost for t in self.tracks],
                "track_quality": self.cfg.track_quality,
                "tentatives_discarded_by_env": [t.tentatives_discarded for t in self.tracks],
                "exposures": self.exposures, "detections": self.detections,
                "detection_rate": self.detections/max(1, self.exposures),
                "valid_position_rate": self.valid_frames/max(1, self.exposures),
                "first_valid_velocity_latency_s": self.first_valid_times,
                "track_losses": self.track_losses, "reacquisitions": self.reacquisitions,
                "source": "RGB-D only; no ground-truth fallback"}

    def close(self):
        if self.window is not None:
            self.window.destroy()
            self.window = None
