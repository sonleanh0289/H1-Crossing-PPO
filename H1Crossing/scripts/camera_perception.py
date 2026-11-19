"""RGB-D-only car perception. No simulator, vehicle-state or traffic imports.

Known: X-aligned box dimensions, road height, camera calibration and odometry.
Depth is optical Z (distance_to_image_plane), quaternions are XYZW.
This detector supports green-only and saturated multicolor procedural cars.
It is NOT a general vehicle detector; achromatic/low-saturation cars are unsupported.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import math
import numpy as np
import cv2
import torch
from crossing_observations import compute_crossing_observation
from camera_reset_barrier import RESET_CAPTURE_POLICY

CONFIG = Path(__file__).resolve().parents[1] / "configs/crossing_perception.yaml"
EXTRA_FIELDS = ("position_valid", "velocity_valid", "track_age/0.3s")
LEGACY_CAMERA_NAMES = ("front", "left")
CAMERA_YAWS = {"front": 0., "left": math.pi/2, "right": -math.pi/2}
DETECTOR_ALGORITHMS = {"green": "green-box-cv-kalman-v1",
                       "multicolor": "saturated-box-cv-kalman-v2"}
VELOCITY_VALIDITY_MODES = ("legacy_hits_v1", "confidence_v1")
DETECTION_FILTER = "road-innovation-v1"
TRACK_LIFECYCLE = "single-active-until-lost-v1"
TRACK_QUALITY = "confirmed-prediction-v1"
TRACK_QUALITY_FIELDS = ("track_quality", "birth_max_speed_mps", "birth_max_residual_m",
                        "prediction_max_age_s", "prediction_unreliable_age_s",
                        "prediction_max_std_m", "prediction_max_displacement_m")
# Versioned confidence_v1 thresholds. These gate the output flag only: no
# measurement rejection, state clipping, fusion changes, or GT feedback.
VELOCITY_MIN_SPAN_S = .25
VELOCITY_STABLE_SPAN_S = .15
VELOCITY_MAX_STD_MPS = 1.5
VELOCITY_MAX_INNOVATION_M = .75
VELOCITY_MAX_NIS = 9.21
VELOCITY_MAX_UPDATE_MPS = 1.


@dataclass(frozen=True)
class PerceptionConfig:
    width: int = 320
    height: int = 240
    horizontal_fov_deg: float = 100.
    frequency_hz: float = 30.
    near_m: float = .1
    far_m: float = 80.
    dropout_seconds: float = .3
    acceleration_noise: float = 8.
    measurement_noise_m: float = .25
    min_pixels: int = 4
    torso_offset: tuple = (.20, 0., .60)
    vehicle_size: tuple = (4.2, 1.8, .65)
    vehicle_center_z: float = .68
    # Missing field in old YAML/checkpoints means the original two-camera rig.
    camera_names: tuple = LEGACY_CAMERA_NAMES
    detector_mode: str = "green"
    velocity_validity: str = "confidence_v1"
    detection_filter: str = DETECTION_FILTER
    road_margin_m: float = .5
    max_innovation_m: float = 2.0
    max_innovation_nis: float = 9.21
    track_lifecycle: str = TRACK_LIFECYCLE
    reset_capture_policy: str = RESET_CAPTURE_POLICY
    track_quality: str = TRACK_QUALITY
    birth_max_speed_mps: float = 25.
    birth_max_residual_m: float = .75
    prediction_max_age_s: float = .15
    prediction_unreliable_age_s: float = .05
    prediction_max_std_m: float = 1.
    prediction_max_displacement_m: float = 2.
    motion_gate: int = 0

    def __post_init__(self):
        if self.motion_gate not in (0, 1):
            raise ValueError("motion_gate must be 0 or 1")
        if self.detector_mode not in DETECTOR_ALGORITHMS:
            raise ValueError("detector_mode must be green or multicolor")
        if self.velocity_validity not in VELOCITY_VALIDITY_MODES:
            raise ValueError("Unsupported velocity validity mode")
        if self.detection_filter != DETECTION_FILTER:
            raise ValueError("Unsupported detection filter")
        if self.track_lifecycle != TRACK_LIFECYCLE:
            raise ValueError("Unsupported track lifecycle")
        if self.reset_capture_policy != RESET_CAPTURE_POLICY:
            raise ValueError("Unsupported reset capture policy")
        if self.track_quality not in ("legacy_v1", TRACK_QUALITY):
            raise ValueError("Unsupported track quality policy")
        if not self.prediction_unreliable_age_s <= self.prediction_max_age_s <= self.dropout_seconds:
            raise ValueError("Prediction age limits must fit within identity dropout")
        if self.camera_names not in (LEGACY_CAMERA_NAMES, ("front", "left", "right")):
            raise ValueError("camera_names must be front/left or front/left/right in that order")
        numbers = [v for k, v in asdict(self).items() if k != "motion_gate" and isinstance(v, (int, float))]
        if not all(math.isfinite(v) and v > 0 for v in numbers):
            raise ValueError("Perception scalar settings must be finite and positive")
        if not 1 < self.horizontal_fov_deg < 179 or self.near_m >= self.far_m:
            raise ValueError("Invalid camera frustum")
        if self.frequency_hz not in (20., 30.) or self.dropout_seconds > .3:
            raise ValueError("Supported camera rates are 20/30 Hz; dropout must be <= 0.3 seconds")
        if len(self.vehicle_size) != 3 or min(self.vehicle_size) <= 0 or len(self.torso_offset) != 3:
            raise ValueError("Invalid dimensions/offset")


def load_config(path=CONFIG):
    import yaml
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("torso_offset", "vehicle_size", "camera_names"):
        if key in values:
            values[key] = tuple(values[key])
    return PerceptionConfig(**values)


def camera_frame_due(physics_step, physics_dt, frequency_hz):
    """Phase-locked capture, without cumulative drift when physics Hz/fps is fractional.

    At 200 Hz physics and 30 fps this alternates 7/7/6 physics ticks. Exposure
    timestamps retain the actual tick time (35/35/30 ms), not fictitious 1/30 s.
    """
    phase = physics_dt*frequency_hz
    if physics_step < 1 or not math.isfinite(phase) or not 0 < phase <= 1:
        raise ValueError("Invalid capture clock")
    return math.floor(physics_step*phase+1e-9) > math.floor((physics_step-1)*phase+1e-9)


def rotation_xyzw(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > 1e-3:
        raise ValueError("Expected a normalized XYZW quaternion")
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def backproject(u, v, depth, intrinsics, position, quaternion_ros):
    k = np.asarray(intrinsics)
    optical = np.column_stack(((u-k[0, 2])*depth/k[0, 0],
                               (v-k[1, 2])*depth/k[1, 1], depth))
    return optical @ rotation_xyzw(quaternion_ros).T + np.asarray(position)


@dataclass
class Detection:
    center: np.ndarray
    variance: float
    bbox: tuple
    pixels: int


@dataclass(frozen=True)
class RoadRegion:
    """Static mapped road in world XY, independent of observed vehicle state."""
    origin_xy: tuple
    length: float
    width: float
    margin: float = .5

    def __post_init__(self):
        values = np.asarray((*self.origin_xy, self.length, self.width, self.margin))
        if (len(self.origin_xy) != 2 or not np.isfinite(values).all()
                or min(self.length, self.width) <= 0 or self.margin < 0):
            raise ValueError("Invalid mapped road region")

    def contains(self, center):
        center = np.asarray(center)
        return bool(center.shape == (2,) and np.isfinite(center).all()
                    and np.all(np.abs(center-np.asarray(self.origin_xy))
                               <= np.array([self.length, self.width])/2+self.margin))


def fit_center(points, camera_position, cfg):
    """Fit known vertical box faces; exclude cabin, roof and ground before fitting.

    Near-visible faces constrain the center by HALF the box extent. An unseen
    tangential axis uses the silhouette midpoint and carries measurement noise.
    """
    from scipy.optimize import least_squares
    p = np.asarray(points)[:, :2]
    half = np.asarray(cfg.vehicle_size[:2]) / 2
    lo, hi = np.quantile(p, [.01, .99], axis=0)
    midpoint = (lo+hi)/2
    view = np.asarray(camera_position)[:2]-midpoint
    direction = view / max(np.linalg.norm(view), 1e-6)
    initial = midpoint.copy()
    for axis in range(2):
        if abs(direction[axis]) > .25:
            initial[axis] = (hi[axis]-half[axis] if view[axis] > 0 else lo[axis]+half[axis])
    def residual(center):
        # SDF of an axis-aligned rectangle: zero at either visible vertical face.
        d = np.abs(p-center)-half
        return np.maximum(d[:, 0], d[:, 1])
    fit = least_squares(residual, initial, loss="soft_l1", f_scale=.08,
                        bounds=(initial-.8, initial+.8), max_nfev=30)
    if np.mean(np.abs(residual(fit.x))) > .25:
        return None
    return fit.x


def color_mask(rgb, mode):
    """Image-only candidate mask. No palette, assigned color, labels or GT inputs.

    Multicolor removes the hue gate; depth/height/box checks still decide whether
    a saturated component resembles the known car. Black/white/gray are excluded.
    """
    hsv = cv2.cvtColor(np.ascontiguousarray(np.asarray(rgb)[..., :3]), cv2.COLOR_RGB2HSV)
    mask = (hsv[..., 1] >= 65) & (hsv[..., 2] >= 35)
    if mode == "green":
        mask &= (hsv[..., 0] >= 35) & (hsv[..., 0] <= 90)
    elif mode != "multicolor":
        raise ValueError("Unsupported detector mode")
    return mask


def detect(rgb, depth, k, position, quaternion_ros, cfg, road_z=0., *, road_region=None,
           diagnostics=None, return_candidates=False):
    """Inputs must all describe the SAME rendered exposure."""
    rgb = np.asarray(rgb)[..., :3]
    depth = np.asarray(depth).reshape(rgb.shape[:2])
    mask = (color_mask(rgb, cfg.detector_mode) &
            np.isfinite(depth) & (depth > cfg.near_m) & (depth < cfg.far_m))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    candidates = []
    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] < cfg.min_pixels:
            continue
        v, u = np.where(labels == label)
        points = backproject(u, v, depth[v, u], k, position, quaternion_ros)
        # Only chassis vertical faces: goal/sidewalk and cabin do not qualify.
        height = points[:, 2]-road_z
        keep = (height > cfg.vehicle_center_z-cfg.vehicle_size[2]/2+.015) & (height < .94)
        points = points[keep]
        # Retain pixel coordinates for exactly the 3D points used by the fit.
        # The original color component can include a connected ground/goal
        # patch: its unfiltered bbox must not become the vehicle's display box.
        u, v = u[keep], v[keep]
        if len(points) < cfg.min_pixels:
            continue
        span = np.ptp(points[:, :2], axis=0)
        if np.any(span > np.asarray(cfg.vehicle_size[:2])+.5):
            continue
        center = fit_center(points[::max(1, len(points)//600)], position, cfg)
        if center is None:
            continue
        if road_region is not None and not road_region.contains(center):
            if diagnostics is not None:
                diagnostics["outside_road"] = diagnostics.get("outside_road", 0)+1
            continue
        distance = np.linalg.norm(center-np.asarray(position)[:2])
        variance = (cfg.measurement_noise_m + .008*distance)**2
        x, y = int(u.min()), int(v.min())
        w, h = int(u.max())-x+1, int(v.max())-y+1
        candidates.append(Detection(center, variance, (int(x), int(y), int(w), int(h)), len(points)))
    return candidates if return_candidates else max(candidates, key=lambda item: item.pixels, default=None)


def fuse(detections):
    """One known car: fuse overlapping views, never create a second track."""
    items = [d for d in detections if d is not None]
    if not items:
        return None
    best = max(items, key=lambda d: d.pixels / d.variance)
    items = [d for d in items if np.linalg.norm(d.center-best.center) < 2.]
    weights = np.array([1/d.variance for d in items])
    center = np.average([d.center for d in items], axis=0, weights=weights)
    return Detection(center, float(1/weights.sum()), best.bbox, sum(d.pixels for d in items))


class Track:
    def __init__(self, cfg, road_region=None):
        self.cfg = cfg
        self.road_region = road_region
        self.rejections = {}
        self.association_rejections = {}
        self.tracks_created = 0
        self.tracks_lost = 0
        self.tentatives_discarded = 0
        self.reset()

    def reset(self):
        self.x = np.zeros(4)
        self.p = np.diag([1., 1., 100., 100.])
        self.time = None
        self.last_measurement = None
        self.hits = 0
        self.first_measurement = None
        self.velocity_good_since = None
        self.last_update_status = "reset"
        self.track_id = None
        self.confirmed = False
        self.capture_index = 0
        self.birth_samples = []
        self.last_position = None
        self.last_detection_position = None
        self._association_time = 0.

    def expired(self, now):
        timeout = (5/self.cfg.frequency_hz if self.cfg.track_quality == TRACK_QUALITY
                   and not self.confirmed else self.cfg.dropout_seconds)
        return (self.last_measurement is not None
                and now-self.last_measurement > timeout+1e-8)

    def recent_birth_samples(self, now, index):
        return [s for s in self.birth_samples if index-s[0] < 5
                and now-s[1] <= 5/self.cfg.frequency_hz+1e-8]

    def birth_rejection(self, now, detection, index):
        """Image-space-derived world motion only; never query traffic state."""
        samples = self.recent_birth_samples(now, index)
        if not samples:
            return None
        dt = now-samples[-1][1]
        distance = np.linalg.norm(detection.center-samples[-1][2])
        if dt <= 0 or distance > self.cfg.birth_max_speed_mps*dt + .35:
            return "birth_speed"
        if len(samples) >= 2:
            times = np.array([s[1]-now for s in samples])
            xy = np.array([s[2] for s in samples])
            model = np.linalg.lstsq(np.column_stack((np.ones(len(times)), times)), xy, rcond=None)[0]
            if (np.linalg.norm(model[1]) > self.cfg.birth_max_speed_mps
                    or np.linalg.norm(detection.center-model[0]) > self.cfg.birth_max_residual_m):
                return "birth_motion_residual"
        return None

    def prediction(self, now):
        dt = 0. if self.time is None else max(0., now-self.time)
        f = np.eye(4)
        f[:2, 2:] = np.eye(2)*dt
        g = np.vstack((np.eye(2)*dt*dt/2, np.eye(2)*dt))
        return f@self.x, f@self.p@f.T+g@g.T*self.cfg.acceleration_noise**2

    def rejection_reason(self, detection, state, covariance):
        center = np.asarray(detection.center)
        if (center.shape != (2,) or not np.isfinite(center).all()
                or not math.isfinite(detection.variance) or detection.variance <= 0):
            return "invalid_measurement"
        if self.road_region is not None and not self.road_region.contains(center):
            return "outside_road"
        if self.hits:
            innovation = center-state[:2]
            # Measurement-to-measurement bound cannot be widened by an
            # uncertain Kalman covariance or a corrupted velocity estimate.
            if self.cfg.motion_gate and self.last_detection_position is not None:
                dt = max(0., self._association_time-self.last_measurement)
                if np.linalg.norm(center-self.last_detection_position) > .35+self.cfg.birth_max_speed_mps*dt:
                    return "measurement_motion_jump"
            nis = float(innovation @ np.linalg.solve(covariance[:2, :2]
                        +np.eye(2)*detection.variance, innovation))
            if np.linalg.norm(innovation) > self.cfg.max_innovation_m:
                return "innovation_distance"
            if not math.isfinite(nis) or nis > self.cfg.max_innovation_nis:
                return "innovation_nis"
        return None

    def select_detection(self, now, candidates):
        """Associate every camera to the active track before choosing/fusing boxes.

        A larger unmatched candidate must not hide a still-visible tracked car.
        No candidate can replace an expired track in its retirement frame.
        """
        if self.expired(now):
            return None
        self._association_time = now
        state, covariance = self.prediction(now)
        matched = []
        for item in candidates:
            reason = self.rejection_reason(item, state, covariance)
            if reason is None and (self.cfg.motion_gate or self.cfg.track_quality == TRACK_QUALITY) and not self.confirmed:
                reason = self.birth_rejection(now, item, self.capture_index+1)
            if reason is None:
                matched.append(item)
            else:
                self.association_rejections[reason] = self.association_rejections.get(reason, 0)+1
        if self.hits:
            return min(matched, key=lambda d: float((d.center-state[:2]) @ np.linalg.solve(
                covariance[:2, :2]+np.eye(2)*d.variance, d.center-state[:2])), default=None)
        return max(matched, key=lambda d: d.pixels, default=None)

    def reject(self, reason):
        self.rejections[reason] = self.rejections.get(reason, 0)+1
        self.last_update_status = reason
        self.velocity_good_since = None

    def update(self, now, detection):
        if self.time is not None and now <= self.time:
            raise ValueError("Duplicate/out-of-order camera timestamp")
        if self.expired(now):
            # Retirement and birth are distinct camera frames. Never reuse the
            # candidate that caused loss as an immediate replacement track.
            was_confirmed = self.confirmed
            if was_confirmed:
                self.tracks_lost += 1
            else:
                self.tentatives_discarded += 1
            self.reset()
            self.time = now
            self.last_update_status = "lost" if was_confirmed else "tentative_expired"
            return
        self.capture_index += 1
        if self.hits:
            self.x, self.p = self.prediction(now)
        self.time = now
        self.last_update_status = "missing"
        if detection is None:
            if (self.last_measurement is not None and
                    now-self.last_measurement > 2.5/self.cfg.frequency_hz+1e-8):
                self.velocity_good_since = None
            return
        self._association_time = now
        reason = self.rejection_reason(detection, self.x, self.p)
        if reason is None and (self.cfg.motion_gate or self.cfg.track_quality == TRACK_QUALITY) and not self.confirmed:
            reason = self.birth_rejection(now, detection, self.capture_index)
        if reason is not None:
            self.reject(reason)
            return
        r = np.eye(2)*detection.variance
        if self.hits:
            innovation = detection.center-self.x[:2]
            nis = float(innovation @ np.linalg.solve(self.p[:2, :2]+r, innovation))
        previous_measurement = self.last_measurement
        previous_velocity = self.x[2:].copy()
        consistent = False
        if not self.hits:
            self.x[:2] = detection.center
            self.first_measurement = now
        else:
            gain = np.linalg.solve(self.p[:2, :2]+r, self.p[:, :2].T).T
            self.x += gain @ (detection.center-self.x[:2])
            h = np.eye(4)[:2]
            a = np.eye(4)-gain@h
            self.p = a@self.p@a.T + gain@r@gain.T
            consistent = (np.linalg.norm(innovation) <= VELOCITY_MAX_INNOVATION_M
                          and nis <= VELOCITY_MAX_NIS
                          and np.linalg.norm(self.x[2:]-previous_velocity) <= VELOCITY_MAX_UPDATE_MPS
                          and np.linalg.eigvalsh(self.p[2:, 2:]).max() <= VELOCITY_MAX_STD_MPS**2)
        self.hits += 1
        self.last_measurement = now
        self.last_detection_position = detection.center.copy()
        self.last_position = self.x[:2].copy()
        self.birth_samples = self.recent_birth_samples(now, self.capture_index)
        self.birth_samples.append((self.capture_index, now, detection.center.copy()))
        if not self.confirmed:
            ready = self.cfg.track_quality == "legacy_v1" and not self.cfg.motion_gate
            if not ready and len(self.birth_samples) >= 3:
                times = np.array([s[1]-now for s in self.birth_samples])
                xy = np.array([s[2] for s in self.birth_samples])
                design = np.column_stack((np.ones(len(times)), times))
                model = np.linalg.lstsq(design, xy, rcond=None)[0]
                ready = (now-self.birth_samples[0][1] >= 1.8/self.cfg.frequency_hz-1e-8
                         and np.linalg.norm(model[1]) <= self.cfg.birth_max_speed_mps
                         and np.linalg.norm(xy-design@model, axis=1).max() <= self.cfg.birth_max_residual_m)
            if ready:
                self.confirmed = True
                self.tracks_created += 1
                self.track_id = self.tracks_created
        self.last_update_status = "accepted" if self.confirmed else "tentative"
        if not consistent:
            self.velocity_good_since = None
        elif (self.velocity_good_since is None or previous_measurement is None
              or now-previous_measurement > 2.5/self.cfg.frequency_hz+1e-8):
            self.velocity_good_since = now

    def estimate(self, now):
        age = self.cfg.dropout_seconds if self.last_measurement is None else max(0., now-self.last_measurement)
        valid = self.hits > 0 and age <= self.cfg.dropout_seconds+1e-8
        if self.cfg.motion_gate:
            valid = valid and self.confirmed
        state = self.x.copy()
        if self.time is not None:
            state[:2] += state[2:]*max(0, now-self.time)
        if self.road_region is not None:
            valid = valid and self.road_region.contains(state[:2])
        if self.cfg.track_quality == TRACK_QUALITY:
            valid = valid and self.position_diagnostics(now)["camera_position_reason"] == "valid"
        if self.cfg.velocity_validity == "legacy_hits_v1":
            velocity_valid = valid and self.hits >= 3
        else:
            # Predict uncertainty to the consumer's timestamp without mutating
            # the filter. Repeated reads cannot advance confidence/warm-up.
            prediction_dt = 0. if self.time is None else max(0., now-self.time)
            variance = (np.linalg.eigvalsh(self.p[2:, 2:]).max()
                        + (self.cfg.acceleration_noise*prediction_dt)**2)
            velocity_valid = bool(valid and self.hits >= 3
                and self.first_measurement is not None
                and self.last_measurement-self.first_measurement >= VELOCITY_MIN_SPAN_S-1e-8
                and self.velocity_good_since is not None
                and self.last_measurement-self.velocity_good_since >= VELOCITY_STABLE_SPAN_S-1e-8
                and age <= 2.5/self.cfg.frequency_hz+1e-8
                and variance <= VELOCITY_MAX_STD_MPS**2)
        return state, valid, velocity_valid, min(age/self.cfg.dropout_seconds, 1.)

    def position_diagnostics(self, now):
        state, covariance = self.prediction(now)
        if not np.isfinite(state).all() or not np.isfinite(covariance).all():
            return {"camera_position_reason": "nonfinite", "camera_track_confirmed": self.confirmed,
                    "camera_birth_hits_in_window": 0, "camera_position_std_m": None,
                    "camera_prediction_age_limit_s": self.cfg.prediction_unreliable_age_s,
                    "camera_prediction_displacement_m": None}
        std = float(np.sqrt(max(0., np.linalg.eigvalsh(covariance[:2, :2]).max())))
        velocity_std = float(np.sqrt(max(0., np.linalg.eigvalsh(covariance[2:, 2:]).max())))
        reliable_motion = (self.first_measurement is not None and self.last_measurement is not None
                           and self.last_measurement-self.first_measurement >= VELOCITY_MIN_SPAN_S-1e-8
                           and velocity_std <= VELOCITY_MAX_STD_MPS)
        limit = self.cfg.prediction_max_age_s if reliable_motion else self.cfg.prediction_unreliable_age_s
        age = math.inf if self.last_measurement is None else max(0., now-self.last_measurement)
        displacement = 0. if self.last_position is None else float(np.linalg.norm(state[:2]-self.last_position))
        if not self.hits:
            reason = "no_track"
        elif (self.cfg.motion_gate or self.cfg.track_quality == TRACK_QUALITY) and not self.confirmed:
            reason = "unconfirmed"
        elif not np.isfinite(state).all() or not np.isfinite(covariance).all():
            reason = "nonfinite"
        elif self.road_region is not None and not self.road_region.contains(state[:2]):
            reason = "outside_road"
        elif age > self.cfg.dropout_seconds+1e-8:
            reason = "expired"
        elif self.cfg.track_quality == "legacy_v1":
            reason = "valid"
        elif std > self.cfg.prediction_max_std_m:
            reason = "position_uncertainty"
        elif age > limit+1e-8:
            reason = "prediction_age" if reliable_motion else "uncertain_velocity_age"
        elif displacement > self.cfg.prediction_max_displacement_m:
            reason = "prediction_displacement"
        else:
            reason = "valid"
        return {"camera_position_reason": reason, "camera_track_confirmed": self.confirmed,
                "camera_birth_hits_in_window": len(self.recent_birth_samples(now, self.capture_index)),
                "camera_position_std_m": std if math.isfinite(std) else None,
                "camera_prediction_age_limit_s": limit,
                "camera_prediction_displacement_m": displacement if math.isfinite(displacement) else None}


def observation_from_tracks(tracks, now, h1_xy, h1_velocity, goal_xy):
    """Only image tracks + H1/map data enter here; no simulator vehicle fallback."""
    estimates = [track.estimate(now) for track in tracks]
    values = h1_xy.new_tensor(np.stack([v[0] for v in estimates]))
    flags = h1_xy.new_tensor([v[1:] for v in estimates])
    pos_valid, vel_valid = flags[:, 0].bool(), flags[:, 1].bool()
    position = torch.where(pos_valid[:, None], values[:, :2], h1_xy)
    velocity = torch.where(vel_valid[:, None], values[:, 2:], 0.)
    raw = compute_crossing_observation(h1_xy, h1_velocity, position, velocity, goal_xy)
    valid = pos_valid & vel_valid
    raw[~valid, 6] = 10.
    raw[~valid, 7] = 0.
    return raw, flags


def expanded_state(old_state, new_state):
    """Append zero input columns (19->22/23 or 22->23), preserve all other weights/std."""
    if old_state.keys() != new_state.keys():
        raise ValueError("Model state keys differ")
    result = {}
    for name, target in new_state.items():
        source = old_state[name]
        if name == "mlp.0.weight":
            if (source.ndim != 2 or target.ndim != 2 or source.shape[0] != 128 or target.shape[0] != 128
                    or (source.shape[1], target.shape[1]) not in ((19,22),(19,23),(22,22),(22,23),(23,23))):
                raise ValueError("Warm-start requires supported 19/22/23 input and 128 hidden units")
            result[name] = torch.cat((source, source.new_zeros(128, target.shape[1]-source.shape[1])), dim=1)
        elif source.shape == target.shape:
            result[name] = source.clone()
        else:
            raise ValueError(f"Incompatible parameter {name}")
    return result
