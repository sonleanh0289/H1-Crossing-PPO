"""23D camera-position / radar-lane-velocity observation contract.

Single known straight lane and one procedural vehicle only. No vehicle truth,
or object IDs. Valid camera Kalman velocity backs up unavailable radar velocity.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
import torch
from radar_processing import diagnose_returns
from crossing_observations import compute_crossing_observation

FORMAT = "h1-crossing-camera-radar-v1"
DIRECTION_FIELD = "vehicle_direction_world_x"
VELOCITY_SOURCE = "radar_world_x_vy_zero_else_valid_camera_kalman_world_xy_v1"


@dataclass(frozen=True)
class RadarObservationConfig:
    max_velocity_age_s: float = .12
    direction_deadband_mps: float = .2
    # Sparse radar can see only the two ends of the 4.2 m procedural body.
    cluster_gap_m: float = 4.2
    ego_exclusion_radius_m: float = .8
    confirmation_frames: int = 3
    association_version: str = "single_lane_unique_cluster_v1"


OBS_CONFIG = RadarObservationConfig()


def observation_contract(cfg):
    return {"sensor": asdict(cfg), "processing": asdict(OBS_CONFIG),
            "velocity_source": VELOCITY_SOURCE,
            "direction": "signed world-X motion: -1,0,+1; zero below deadband or invalid",
            "position_source": "camera_only", "position_required_for_autonomy": True,
            "velocity_required_for_autonomy": False}


class LaneRadarAssociation:
    """Acquire a unique above-ground lane cluster without camera detections.

Reject ambiguity, ego returns, discontinuities and unobservable Doppler. Camera
    boxes can select returns, but cannot supply velocity. Radar-only acquisition
    requires consecutive sensor confirmations; reset discards that history.
"""
    def __init__(self, lane_y, cfg=OBS_CONFIG):
        self.lane_y, self.cfg = float(lane_y), cfg
        self.reset()

    def reset(self):
        self.previous = None
        self.hits = 0

    def process(self, points, radial, position, rotation, ego_velocity, camera_center,
                camera_age, vehicle_size, radar_cfg, *, timing_valid, now, ego_xy):
        points, radial = np.asarray(points), np.asarray(radial)
        world = points @ np.asarray(rotation).T + position
        ranges = np.linalg.norm(points, axis=1)
        keep = (np.isfinite(points).all(axis=1) & np.isfinite(radial)
                & (ranges >= radar_cfg.min_range_m) & (ranges <= radar_cfg.max_range_m)
                & (np.abs(world[:, 1]-self.lane_y) <= vehicle_size[1]/2+radar_cfg.association_margin_m)
                & (world[:, 2] > .25) & (world[:, 2] < 2.)
                & (np.linalg.norm(world[:, :2]-ego_xy, axis=1) > self.cfg.ego_exclusion_radius_m))
        camera_fresh = camera_center is not None and 0 <= camera_age <= radar_cfg.max_camera_age_s
        if camera_fresh:
            guided = diagnose_returns(points[keep], radial[keep], position, rotation, ego_velocity,
                camera_center, camera_age, vehicle_size, radar_cfg, timing_valid=timing_valid)
            if guided["lane_speed_valid"]:
                self.previous = (now, np.asarray(camera_center), guided["lane_vx_mps"])
                self.hits = self.cfg.confirmation_frames
                guided.update(returns=len(points), association_source="camera_guided_lane")
                return guided
        ids = np.flatnonzero(keep)
        ids = ids[np.argsort(world[ids, 0])]
        clusters = np.split(ids, np.flatnonzero(np.diff(world[ids, 0]) > self.cfg.cluster_gap_m)+1)
        candidates = []
        for group in clusters:
            if len(group) < radar_cfg.min_returns:
                continue
            xy = world[group, :2]
            if np.any(np.ptp(xy, axis=0) > np.asarray(vehicle_size[:2])+2*radar_cfg.association_margin_m):
                continue
            center = (xy.min(axis=0)+xy.max(axis=0))/2
            candidates.append((group, center))
        if camera_fresh:
            extent = np.asarray(vehicle_size[:2])/2+radar_cfg.association_margin_m
            guided = [(g,c) for g,c in candidates if np.all(np.abs(c-camera_center) <= extent)]
            if guided:
                candidates = guided
        empty = dict(returns=len(points), associated_returns=0, associated=False,
                     radial_relative_mps=None, radial_world_mps=None, lane_vx_mps=None,
                     lane_speed_valid=False, association_source="radar_lane",
                     reason="ambiguous_lane_clusters" if len(candidates)>1 else "no_lane_cluster")
        if len(candidates) != 1:
            self.reset()
            return empty
        group, center = candidates[0]
        result = diagnose_returns(points[group], radial[group], position, rotation, ego_velocity,
            center, 0., vehicle_size, radar_cfg, timing_valid=timing_valid)
        result.update(returns=len(points), association_source="radar_lane")
        if not result["lane_speed_valid"]:
            self.reset()
            return result
        continuous = False
        if self.previous is not None:
            t, old_center, old_vx = self.previous
            dt = now-t
            predicted = old_center + np.array([old_vx*dt, 0.])
            continuous = 0 < dt <= self.cfg.max_velocity_age_s and np.linalg.norm(center-predicted) <= 2.
        self.hits = self.hits+1 if continuous else 1
        self.previous = (now, center, result["lane_vx_mps"])
        if self.hits < self.cfg.confirmation_frames:
            result.update(lane_speed_valid=False, lane_vx_mps=None, reason="radar_track_confirming")
        return result


def radar_observation(raw_camera, camera_flags, row, now, h1_xy, h1_velocity, goal_xy):
    """Prefer fresh radar; otherwise use valid camera Kalman world-XY velocity.

No stale-velocity hold or ground-truth fallback. Camera validity is supplied by
the tracker (including its age/confidence gates), never inferred from detection.
TTC requires camera position and a valid selected velocity. Radar remains usable
without a camera track; the camera age field retains its original meaning.
Age uses exposure time, not the wall clock or time of repeated observation reads.
"""
    count = len(h1_xy)
    rows = [row] if row is None or isinstance(row, dict) else row
    if raw_camera.shape != (count, 8) or camera_flags.shape != (count, 3) or len(rows) != count:
        raise ValueError("Camera, radar and odometry must have matching environment rows")
    speeds, validity = [], []
    for item in rows:
        valid, vx = False, 0.
        if item and item.get("lane_speed_valid"):
            measured = item.get("lane_vx_mps")
            stamp = item.get("sensor_clock_reference_sim_s")
            valid = (measured is not None and stamp is not None and math.isfinite(measured)
                     and math.isfinite(stamp) and 0 <= now-stamp <= OBS_CONFIG.max_velocity_age_s+1e-8)
            if valid:
                vx = float(measured)
        speeds.append([vx, 0.])
        validity.append(valid)
    velocity = h1_xy.new_tensor(speeds)
    valid = torch.tensor(validity, device=h1_xy.device, dtype=torch.bool)
    position_valid = camera_flags[:, 0] > .5
    camera_valid = (position_valid & (camera_flags[:, 1] > .5)
                    & torch.isfinite(raw_camera[:, 4:6]).all(dim=1))
    use_camera = ~valid & camera_valid
    velocity = torch.where(use_camera[:, None], raw_camera[:, 4:6], velocity)
    valid = valid | use_camera
    position = h1_xy + torch.where(position_valid[:, None], raw_camera[:, :2], 0.)
    raw = compute_crossing_observation(h1_xy, h1_velocity, position, velocity, goal_xy)
    flags = camera_flags.clone()
    flags[:, 1] = valid.to(flags.dtype)
    invalid = ~(valid & position_valid)
    raw[invalid, 6], raw[invalid, 7] = 10., 0.
    direction = torch.where(valid & (velocity[:, 0].abs() >= OBS_CONFIG.direction_deadband_mps),
                            velocity[:, 0].sign(), 0.)[:, None]
    return raw, flags, direction
