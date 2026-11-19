"""Sensor-only radar diagnostics. No vehicle simulator state or policy access.

This is a shadow experiment, not a fused observation provider. Positive radial
velocity means increasing range. World-X speed is only identifiable here under
the explicit straight-lane assumption, and never near a transverse sight line.
"""
from dataclasses import dataclass, fields
import math
from pathlib import Path
import numpy as np
import yaml
from camera_perception import rotation_xyzw


@dataclass(frozen=True)
class RadarConfig:
    profile: str = "generic_shadow_v1"
    torso_offset: tuple = (.22, .10, .35)
    yaw_deg: float = 90.
    sample_hz: float = 30.
    horizontal_fov_deg: float = 120.
    vertical_fov_deg: float = 40.
    max_range_m: float = 80.
    min_range_m: float = .5
    rays_per_degree: float = 2.
    association_margin_m: float = .6
    max_camera_age_s: float = .08
    min_returns: int = 2
    min_abs_los_x: float = .35
    max_speed_mps: float = 30.
    max_speed_mad_mps: float = 1.5
    reset_warmup_s: float = .2
    opposite_radar: bool = False
    opposite_torso_offset: tuple = (.22, -.10, .35)
    max_head_disagreement_mps: float = 2.

    def __post_init__(self):
        if self.profile != "generic_shadow_v1":
            raise ValueError("Unsupported radar profile")
        if (type(self.opposite_radar) is not bool or len(self.opposite_torso_offset) != 3
                or not np.isfinite(self.opposite_torso_offset).all()
                or not math.isfinite(self.max_head_disagreement_mps) or self.max_head_disagreement_mps <= 0):
            raise ValueError("Invalid complementary radar configuration")
        if len(self.torso_offset) != 3 or not np.isfinite(self.torso_offset).all():
            raise ValueError("Radar offset must be three finite values")
        for name in ("yaw_deg", "sample_hz", "horizontal_fov_deg", "vertical_fov_deg", "max_range_m",
                     "min_range_m", "rays_per_degree", "association_margin_m", "max_camera_age_s",
                     "min_abs_los_x", "max_speed_mps", "max_speed_mad_mps", "reset_warmup_s"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"Non-finite radar setting: {name}")
        if not (0 < self.sample_hz <= 50 and 0 < self.min_range_m < self.max_range_m
                and 0 < self.horizontal_fov_deg <= 180 and 0 < self.vertical_fov_deg <= 90
                and 0 < self.rays_per_degree <= 8 and 0 < self.min_abs_los_x <= 1
                and self.association_margin_m >= 0 and self.max_camera_age_s > 0
                and self.max_speed_mps > 0 and self.max_speed_mad_mps > 0
                and self.reset_warmup_s > 0 and type(self.min_returns) is int and self.min_returns >= 2):
            raise ValueError("Invalid radar limits")


def load_radar_config(path=None):
    path = path or Path(__file__).resolve().parents[1]/"configs/crossing_radar.yaml"
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict) or set(values)-{f.name for f in fields(RadarConfig)}:
        raise ValueError("Unknown radar settings")
    if "torso_offset" in values:
        values["torso_offset"] = tuple(values["torso_offset"])
    if "opposite_torso_offset" in values:
        values["opposite_torso_offset"] = tuple(values["opposite_torso_offset"])
    return RadarConfig(**values)


def mounted_pose(link_pose_xyzw, link_velocity, cfg):
    """Return position, rotation matrix and velocity at the sensor origin.

    Link velocity is [linear at link origin, angular], in world coordinates.
    The omega-cross-offset term is essential for torso rotation.
    """
    pose, velocity = np.asarray(link_pose_xyzw), np.asarray(link_velocity)
    if (pose.shape != (7,) or velocity.shape != (6,) or not np.isfinite(pose).all()
            or not np.isfinite(velocity).all()):
        raise ValueError("Invalid torso odometry")
    r = rotation_xyzw(pose[3:])
    offset = r @ np.asarray(cfg.torso_offset)
    yaw = math.radians(cfg.yaw_deg)
    local = rotation_xyzw([0., 0., math.sin(yaw/2), math.cos(yaw/2)])
    return pose[:3]+offset, r @ local, velocity[:3]+np.cross(velocity[3:], offset)


def exposure_geometry(start_position, start_xyzw, start_ns, end_position, end_ns):
    """Use reported exposure pose and origin velocity, not the latest torso pose.

    The origin finite difference uses the same exposure clock as Doppler and
    includes mount movement from torso rotation. No target state is accessed.
    """
    start, end = np.asarray(start_position, dtype=float), np.asarray(end_position, dtype=float)
    dt = (end_ns-start_ns)*1e-9
    if (start.shape != (3,) or end.shape != (3,) or not np.isfinite(start).all()
            or not np.isfinite(end).all() or not 0 < dt <= .1):
        raise ValueError("Invalid radar exposure metadata")
    return start, rotation_xyzw(start_xyzw), (end-start)/dt


def rotation_to_wxyz(rotation):
    """Explicit conversion at the Isaac Sim API boundary (Lab uses XYZW)."""
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(rotation).as_quat()
    return np.array([w, x, y, z])


class RadarClock:
    """Reject repeated buffers and diagnose sensor/simulation clock mismatch.

    No inferred clock rescaling is used to 'repair' Doppler. A mismatched clock
    disables the derived speed, while raw returns remain inspectable.
    """
    def __init__(self):
        self.last = None
        self.good = 0

    def reset(self):
        self.last = None
        self.good = 0

    def accept(self, timestamp_ns, sim_s):
        if not math.isfinite(sim_s) or timestamp_ns < 0:
            raise ValueError("Invalid radar timestamp")
        if self.last is not None:
            ns, prev_sim = self.last
            if timestamp_ns <= ns or sim_s <= prev_sim:
                self.good = 0
                return False, False, None
            sensor_dt, sim_dt = (timestamp_ns-ns)*1e-9, sim_s-prev_sim
            aligned = abs(sensor_dt-sim_dt) <= .001
            self.good = self.good+1 if aligned else 0
            ratio = sensor_dt/sim_dt
        else:
            ratio = None
        self.last = (timestamp_ns, sim_s)
        return True, self.good >= 3, ratio


def diagnose_returns(points_sensor, radial_mps, sensor_position, sensor_rotation, sensor_velocity,
                     camera_center_xy, camera_age_s, vehicle_size, cfg, *, timing_valid):
    """Associate returns to a fresh image-derived box; never read truth or IDs.

    Camera association is heuristic and can be wrong. The derived lane speed
    assumes world-Y/Z vehicle velocity is zero; it is NOT a measured 2D velocity.
    """
    points, radial = np.asarray(points_sensor, dtype=float), np.asarray(radial_mps, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or radial.shape != (len(points),):
        raise ValueError("Radar points and Doppler must have matching rows")
    for value, shape in ((sensor_position, (3,)), (sensor_velocity, (3,)), (sensor_rotation, (3,3))):
        if np.asarray(value).shape != shape or not np.isfinite(value).all():
            raise ValueError("Invalid sensor odometry")
    result = {"returns": len(points), "associated_returns": 0, "associated": False,
              "radial_relative_mps": None, "radial_world_mps": None,
              "lane_vx_mps": None, "lane_speed_valid": False, "reason": "no_camera_detection"}
    if camera_center_xy is None or not 0 <= camera_age_s <= cfg.max_camera_age_s:
        return result
    ranges = np.linalg.norm(points, axis=1)
    world = points @ np.asarray(sensor_rotation).T + np.asarray(sensor_position)
    center = np.asarray(camera_center_xy)
    extent = np.asarray(vehicle_size[:2])/2 + cfg.association_margin_m
    keep = (np.isfinite(points).all(axis=1) & np.isfinite(radial)
            & (ranges >= cfg.min_range_m) & (ranges <= cfg.max_range_m)
            & (np.abs(world[:, :2]-center) <= extent).all(axis=1)
            & (world[:, 2] > .25) & (world[:, 2] < 2.))
    result["associated_returns"] = int(keep.sum())
    if keep.sum() < cfg.min_returns:
        result["reason"] = "insufficient_returns"
        return result
    result["associated"] = True
    los = (world[keep]-sensor_position)/ranges[keep, None]
    relative = radial[keep]
    world_radial = relative + los @ np.asarray(sensor_velocity)
    result["radial_relative_mps"] = float(np.median(relative))
    result["radial_world_mps"] = float(np.median(world_radial))
    observable = np.abs(los[:, 0]) >= cfg.min_abs_los_x
    result["observable_returns"] = int(observable.sum())
    if observable.sum() < cfg.min_returns:
        result["reason"] = "transverse_geometry"
        return result
    speeds = world_radial[observable]/los[observable, 0]
    result["median_abs_los_x"] = float(np.median(np.abs(los[observable, 0])))
    median = float(np.median(speeds))
    mad = float(np.median(np.abs(speeds-median)))
    result["lane_speed_mad_mps"] = mad
    if not timing_valid:
        result["reason"] = "sensor_clock_unverified"
    elif abs(median) > cfg.max_speed_mps or mad > cfg.max_speed_mad_mps:
        result["reason"] = "inconsistent_doppler"
    else:
        result.update(lane_vx_mps=median, lane_speed_valid=True, reason="lane_assumption_only")
    return result
