"""Ground-truth inputs shared by oracle evaluation and sensor-free 23D training."""
import torch

from crossing_observations import compute_crossing_observation
from radar_observations import OBS_CONFIG

FORMAT = "h1-crossing-groundtruth23-v1"


def observation_fields():
    from behavior_control import BEHAVIOR_FIELDS
    from camera_perception import EXTRA_FIELDS
    from radar_observations import DIRECTION_FIELD
    return list(BEHAVIOR_FIELDS) + list(EXTRA_FIELDS) + [DIRECTION_FIELD]


def input_contract():
    return {"source": "simulator ground truth; scripted instantaneous vehicle velocity",
            "position_valid": 1, "velocity_valid": 1, "track_age": 0,
            "direction_deadband_mps": OBS_CONFIG.direction_deadband_mps,
            "direction": "signed world-X motion: -1,0,+1; zero below deadband",
            "sensors_enabled": False}


def oracle_inputs(h1_xy, h1_velocity, car_xy, car_velocity, goal_xy):
    """Return raw 8D state, [position valid, velocity valid, age], direction.

    Positions/velocities use world XY, exactly like the sensor pipeline. TTC is
    recomputed from truth; stopped cars retain valid zero velocity. Normalization
    remains the responsibility of behavior_observation (shared with sensors).
    """
    raw = compute_crossing_observation(h1_xy, h1_velocity, car_xy, car_velocity, goal_xy)
    flags = raw.new_zeros((len(raw), 3))
    flags[:, :2] = 1.
    vx = car_velocity[:, 0]
    direction = torch.where(vx.abs() >= OBS_CONFIG.direction_deadband_mps, vx.sign(), 0.)[:, None]
    return raw, flags, direction
