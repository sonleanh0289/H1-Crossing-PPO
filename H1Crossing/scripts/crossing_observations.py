"""State-only behavior observations; tensor math is testable without Isaac Sim.

World-aligned XY axes: +X along traffic, +Y across the road. Never append this
group to the frozen locomotion actor's 69-input `policy` group.
"""
import math
import torch


OBSERVATION_NAMES = (
    "vehicle_relative_x_m", "vehicle_relative_y_m",
    "goal_relative_x_m", "goal_relative_y_m",
    "vehicle_vx_mps", "vehicle_vy_mps", "ttc_s", "ttc_valid",
)
OBSERVATION_DIM = len(OBSERVATION_NAMES)
TTC_HORIZON_S = 10.0
H1_PLANAR_RADIUS_M = 0.35


def _check_xy(*values):
    first = values[0]
    for value in values:
        if (not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != 2
                or value.shape != first.shape or not value.is_floating_point()
                or value.device != first.device or value.dtype != first.dtype):
            raise ValueError("Expected matching floating tensors (num_envs, 2), device and dtype")
        if not torch.isfinite(value).all():
            raise ValueError("Crossing state contains NaN/Inf")


def planar_ttc(relative_position_xy, relative_velocity_xy, *, vehicle_size_xy=(4.2, 1.8),
               h1_radius_m=H1_PLANAR_RADIUS_M, horizon_s=TTC_HORIZON_S):
    """First future intersection with an inflated axis-aligned vehicle footprint.

Relative state = vehicle - H1. The car must remain aligned with world X.
H1 is approximated by a radius added to both rectangle half-extents (conservative
at corners). Constant velocities, no acceleration/braking/turn prediction.
Returns finite (N, 1) seconds and boolean validity; invalid => horizon, False.
Current overlap/touch => 0, True, including stationary or receding objects.
"""
    _check_xy(relative_position_xy, relative_velocity_xy)
    if (len(vehicle_size_xy) != 2 or not all(math.isfinite(v) and v > 0 for v in vehicle_size_xy)
            or not math.isfinite(h1_radius_m) or h1_radius_m < 0
            or not math.isfinite(horizon_s) or horizon_s <= 0):
        raise ValueError("Invalid TTC footprint or horizon")
    # Sum dimensions before casting once: an exactly touching float32 point
    # must not lie one ULP outside through repeated float32 rounding.
    extent = relative_position_xy.new_tensor([size / 2 + h1_radius_m for size in vehicle_size_xy])
    moving = relative_velocity_xy.abs() > 1e-6
    safe_velocity = torch.where(moving, relative_velocity_xy, torch.ones_like(relative_velocity_xy))
    first = (-extent - relative_position_xy) / safe_velocity
    last = (extent - relative_position_xy) / safe_velocity
    entry = torch.where(moving, torch.minimum(first, last), -torch.inf).amax(dim=1, keepdim=True)
    exit_time = torch.where(moving, torch.maximum(first, last), torch.inf).amin(dim=1, keepdim=True)
    blocked = ((~moving) & (relative_position_xy.abs() > extent)).any(dim=1, keepdim=True)
    entry = entry.clamp_min(0.0)
    valid = (~blocked) & (exit_time >= entry) & (entry <= horizon_s)
    return torch.where(valid, entry, torch.full_like(entry, horizon_s)), valid


def compute_crossing_observation(h1_position_xy, h1_velocity_xy, vehicle_position_xy,
                                 vehicle_velocity_xy, goal_position_xy, **ttc_options):
    """Raw SI values, batched (N,8), no image/joint state/noise/normalization."""
    _check_xy(h1_position_xy, h1_velocity_xy, vehicle_position_xy, vehicle_velocity_xy, goal_position_xy)
    relative = vehicle_position_xy - h1_position_xy
    ttc, valid = planar_ttc(relative, vehicle_velocity_xy - h1_velocity_xy, **ttc_options)
    return torch.cat((relative, goal_position_xy - h1_position_xy, vehicle_velocity_xy,
                      ttc, valid.to(ttc.dtype)), dim=-1)


def _tensor(value):
    # Isaac Lab 3 exposes physics tensors through `.torch`; CPU mocks use tensors.
    return value if isinstance(value, torch.Tensor) else value.torch


def set_scripted_vehicle_velocity(env, velocity_xy):
    """Publish commanded WORLD velocity after spawn/motion/reset, never finite-difference teleports.

    Future vectorized traffic must supply one row per environment, including new
    speeds for reset env_ids. This does not modify the vehicle's physical motion.
    """
    reference = _tensor(env.scene["robot"].data.root_pos_w)[:, :2]
    _check_xy(reference, velocity_xy)
    env.crossing_vehicle_velocity_xy = velocity_xy.clone()


def crossing_state(env, goal_position_local, vehicle_size_xy, h1_radius_m=H1_PLANAR_RADIUS_M,
                   horizon_s=TTC_HORIZON_S):
    """ObservationTerm adapter. Read physics poses, but analytic scripted car speed.

    Zero velocity is used only while the manager probes shapes during creation;
    the play lifecycle publishes traffic speed and recomputes observations before
    returning the first/reset sample to a consumer.
    """
    robot = env.scene["robot"].data
    vehicle = env.scene["vehicle"].data
    h1_xy = _tensor(robot.root_pos_w)[:, :2]
    car_velocity = getattr(env, "crossing_vehicle_velocity_xy", torch.zeros_like(h1_xy))
    goal_xy = _tensor(env.scene.env_origins)[:, :2] + h1_xy.new_tensor(goal_position_local[:2])
    return compute_crossing_observation(
        h1_xy, _tensor(robot.root_lin_vel_w)[:, :2], _tensor(vehicle.root_pos_w)[:, :2],
        car_velocity, goal_xy, vehicle_size_xy=vehicle_size_xy,
        h1_radius_m=h1_radius_m, horizon_s=horizon_s)


def add_crossing_observation_group(cfg, layout):
    """Call after AppLauncher; attach a separate `crossing` group, preserving policy."""
    from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
    group = ObservationGroupCfg(concatenate_terms=True, enable_corruption=False)
    group.state = ObservationTermCfg(func=crossing_state, params={
        "goal_position_local": layout.goal_position,
        "vehicle_size_xy": layout.vehicle_size[:2],
        "h1_radius_m": H1_PLANAR_RADIUS_M, "horizon_s": TTC_HORIZON_S,
    })
    cfg.observations.crossing = group


def observation_schema(layout):
    return {
        "version": 1, "group": "crossing", "shape": ["num_envs", OBSERVATION_DIM],
        "fields": list(OBSERVATION_NAMES), "frame": "world XY: +X traffic, +Y crossing",
        "relative_positions": "vehicle/goal centre minus H1 root XY; goal includes env_origin",
        "source": "simulator ground truth, not camera perception", "normalized": False,
        "vehicle_velocity": "scripted world velocity in m/s; not displacement across respawn",
        "ttc": {"horizon_s": TTC_HORIZON_S, "invalid_value_s": TTC_HORIZON_S,
                "validity": "1 if footprint intersection predicted in [0,horizon], otherwise 0",
                "vehicle_size_xy_m": list(layout.vehicle_size[:2]), "h1_radius_m": H1_PLANAR_RADIUS_M,
                "model": "constant relative velocity; X-aligned rectangle inflated by H1 radius",
                "limitation": "conservative corners; not exact PhysX contact or distance/speed arrival time"},
        "locomotion": "policy group stays 69D; frozen checkpoint consumes only policy",
    }
