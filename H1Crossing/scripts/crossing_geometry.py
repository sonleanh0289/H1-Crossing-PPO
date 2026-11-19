"""Crossing geometry reused from the legacy preview; importing never launches Kit."""
from __future__ import annotations
from dataclasses import dataclass
from math import sin, cos, pi
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

@dataclass(frozen=True)
class CrossingConfig:
    """World layout and deterministic motion settings."""

    road_length: float = 60.0
    road_width: float = 7.0
    road_thickness: float = 0.05
    sidewalk_width: float = 2.5
    sidewalk_height: float = 0.15

    h1_start_position: tuple[float, float, float] = (0.0, -4.6, 1.20)
    h1_start_yaw: float = pi / 2.0
    goal_position: tuple[float, float, float] = (0.0, 4.6, 0.17)

    vehicle_start_position: tuple[float, float, float] = (-20.0, -1.75, 0.68)
    vehicle_size: tuple[float, float, float] = (4.2, 1.8, 0.65)
    vehicle_orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    vehicle_wrap_min_x: float = -25.0
    vehicle_wrap_max_x: float = 25.0

    crosswalk_stripe_count: int = 10
    crosswalk_stripe_length_x: float = 0.30
    crosswalk_stripe_gap_x: float = 0.22

    camera_eye: tuple[float, float, float] = (18.0, -18.0, 16.0)
    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.7)

    @property
    def road_left_edge_y(self) -> float:
        return -0.5 * self.road_width

    @property
    def road_right_edge_y(self) -> float:
        return 0.5 * self.road_width

    @property
    def h1_start_orientation(self) -> tuple[float, float, float, float]:
        half_yaw = 0.5 * self.h1_start_yaw
        return (0.0, 0.0, sin(half_yaw), cos(half_yaw))


def validate_config(cfg: CrossingConfig) -> None:
    """Validate the left-to-right crossing convention before launching physics."""

    assert cfg.h1_start_position[1] < cfg.road_left_edge_y
    assert cfg.goal_position[1] > cfg.road_right_edge_y
    assert cfg.goal_position[1] > cfg.h1_start_position[1]
    assert abs(cfg.h1_start_position[0] - cfg.goal_position[0]) < 1.0e-6
    assert abs(cos(cfg.h1_start_yaw)) < 1.0e-6
    assert sin(cfg.h1_start_yaw) > 0.999


def make_static_cuboid(
    prim_path: str,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
    color: tuple[float, float, float],
    *,
    collision: bool = False,
    metallic: float = 0.0,
) -> AssetBaseCfg:
    """Create an InteractiveScene config for a static colored cuboid."""

    return AssetBaseCfg(
        prim_path=prim_path,
        init_state=AssetBaseCfg.InitialStateCfg(pos=position),
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg() if collision else None,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=metallic),
        ),
    )


def add_crossing_assets(env_cfg, cfg: CrossingConfig, *, flat_sidewalks: bool = False) -> None:
    """Add road, sidewalks, markings, pads, and the procedural vehicle."""

    env_ns = "{ENV_REGEX_NS}"
    white = (0.95, 0.95, 0.90)

    # The native Flat task's global plane remains the physical road surface.
    # Visual-only surfaces sit 2 mm above the plane to prevent z-fighting.
    # Physics still uses the native plane at Z=0, with no curb in flat mode.
    visual_offset = 0.002
    env_cfg.scene.road = make_static_cuboid(
        f"{env_ns}/Crossing/Road",
        (cfg.road_length, cfg.road_width, cfg.road_thickness),
        (0.0, 0.0, -0.5 * cfg.road_thickness + visual_offset),
        (0.055, 0.065, 0.075),
    )

    sidewalk_center_y = 0.5 * (cfg.road_width + cfg.sidewalk_width)
    sidewalk_z = (-0.5 if flat_sidewalks else 0.5) * cfg.sidewalk_height
    if flat_sidewalks:
        sidewalk_z += visual_offset
    env_cfg.scene.sidewalk_left = make_static_cuboid(
        f"{env_ns}/Crossing/SidewalkLeft",
        (cfg.road_length, cfg.sidewalk_width, cfg.sidewalk_height),
        (0.0, -sidewalk_center_y, sidewalk_z),
        (0.42, 0.44, 0.43),
        collision=not flat_sidewalks,
    )
    env_cfg.scene.sidewalk_right = make_static_cuboid(
        f"{env_ns}/Crossing/SidewalkRight",
        (cfg.road_length, cfg.sidewalk_width, cfg.sidewalk_height),
        (0.0, sidewalk_center_y, sidewalk_z),
        (0.42, 0.44, 0.43),
        collision=not flat_sidewalks,
    )

    marking_z = 0.006
    marking_thickness = 0.012
    for side_name, side_sign in (("left", -1.0), ("right", 1.0)):
        setattr(
            env_cfg.scene,
            f"road_edge_{side_name}",
            make_static_cuboid(
                f"{env_ns}/Crossing/Markings/Edge{side_name.title()}",
                (cfg.road_length, 0.08, marking_thickness),
                (0.0, side_sign * (0.5 * cfg.road_width - 0.10), marking_z),
                white,
            ),
        )

    stripe_pitch = cfg.crosswalk_stripe_length_x + cfg.crosswalk_stripe_gap_x
    first_stripe_x = -0.5 * (cfg.crosswalk_stripe_count - 1) * stripe_pitch
    for stripe_index in range(cfg.crosswalk_stripe_count):
        stripe_x = first_stripe_x + stripe_index * stripe_pitch
        setattr(
            env_cfg.scene,
            f"crosswalk_stripe_{stripe_index:02d}",
            make_static_cuboid(
                f"{env_ns}/Crossing/Crosswalk/Stripe_{stripe_index:02d}",
                (cfg.crosswalk_stripe_length_x, cfg.road_width - 0.30, marking_thickness),
                (stripe_x, 0.0, marking_z + 0.001),
                white,
            ),
        )

    dash_positions = sorted(sign * x for x in range(9, int(cfg.road_length / 2 - 1.5) + 1, 6)
                            for sign in (-1, 1))
    for dash_index, dash_x in enumerate(dash_positions):
        setattr(
            env_cfg.scene,
            f"center_dash_{dash_index:02d}",
            make_static_cuboid(
                f"{env_ns}/Crossing/Markings/CenterDash_{dash_index:02d}",
                (3.0, 0.10, marking_thickness),
                (dash_x, 0.0, marking_z),
                white,
            ),
        )

    pad_z = (0.0 if flat_sidewalks else cfg.sidewalk_height) + 0.008
    env_cfg.scene.start_pad = make_static_cuboid(
        f"{env_ns}/Crossing/StartPad",
        (1.2, 1.2, 0.016),
        (cfg.h1_start_position[0], cfg.h1_start_position[1], pad_z),
        (0.15, 0.55, 0.90),
        metallic=0.1,
    )
    env_cfg.scene.goal_pad = make_static_cuboid(
        f"{env_ns}/Crossing/GoalPad",
        (1.2, 1.2, 0.016),
        cfg.goal_position,
        (0.10, 0.80, 0.25),
        metallic=0.1,
    )

    # The chassis is the kinematic root and collision proxy. Cabin and wheels
    # are visual children, so they inherit its pose without extra rigid bodies.
    env_cfg.scene.vehicle = RigidObjectCfg(
        prim_path=f"{env_ns}/Vehicle",
        spawn=sim_utils.CuboidCfg(
            size=cfg.vehicle_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1400.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.05, 0.55, 0.20), metallic=0.25, roughness=0.35
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=cfg.vehicle_start_position,
            rot=cfg.vehicle_orientation,
        ),
    )
    env_cfg.scene.vehicle_cabin = make_static_cuboid(
        f"{env_ns}/Vehicle/Cabin",
        (2.2, 1.55, 0.65),
        (-0.15, 0.0, 0.60),
        (0.08, 0.65, 0.42),
        metallic=0.15,
    )

    wheel_rot = (sin(pi / 4.0), 0.0, 0.0, cos(pi / 4.0))
    for axle_name, wheel_x in (("rear", -1.35), ("front", 1.35)):
        for side_name, wheel_y in (("left", -0.94), ("right", 0.94)):
            setattr(
                env_cfg.scene,
                f"vehicle_wheel_{axle_name}_{side_name}",
                AssetBaseCfg(
                    prim_path=f"{env_ns}/Vehicle/Wheel{axle_name.title()}{side_name.title()}",
                    init_state=AssetBaseCfg.InitialStateCfg(
                        pos=(wheel_x, wheel_y, -0.33),
                        rot=wheel_rot,
                    ),
                    spawn=sim_utils.CylinderCfg(
                        radius=0.35,
                        height=0.30,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.025, 0.025, 0.030), roughness=0.82
                        ),
                    ),
                ),
            )
