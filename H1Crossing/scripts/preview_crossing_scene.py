"""Preview a flat H1 pedestrian-crossing scene.

The preview intentionally keeps H1 fixed at the left-side spawn point.  It is
used to validate the road layout and scripted vehicle before a locomotion
policy and a ManagerBasedRLEnv task are connected.

Example:

    D:\\IssacSim\\IsaacLab\\isaaclab.bat -p scripts\\preview_crossing_scene.py
    D:\\IssacSim\\IsaacLab\\isaaclab.bat -p scripts\\preview_crossing_scene.py --headless --steps 300
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Preview the flat Unitree H1 road-crossing scene.")
parser.add_argument(
    "--vehicle_speed",
    type=float,
    default=8.0,
    help="Scripted vehicle speed in metres per second along world +X.",
)
parser.add_argument(
    "--steps",
    type=int,
    default=-1,
    help="Number of physics steps to run. Use -1 to run until the simulator closes.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.vehicle_speed <= 0.0:
    parser.error("--vehicle_speed must be greater than zero so the vehicle travels along +X.")
if args_cli.steps == 0 or args_cli.steps < -1:
    parser.error("--steps must be -1 or a positive integer.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from dataclasses import dataclass
from math import cos, pi, sin

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext

from isaaclab_assets import H1_MINIMAL_CFG


@dataclass(frozen=True)
class CrossingSceneConfig:
    """Geometry and state shared by the preview and the future crossing task."""

    physics_dt: float = 0.005

    road_length: float = 60.0
    road_width: float = 7.0
    road_thickness: float = 0.05
    sidewalk_width: float = 2.5
    sidewalk_height: float = 0.15

    h1_start_position: tuple[float, float, float] = (0.0, -4.6, 1.20)
    h1_start_yaw: float = pi / 2.0
    goal_position: tuple[float, float, float] = (0.0, 4.6, 0.17)

    vehicle_start_position: tuple[float, float, float] = (-20.0, -1.75, 0.68)
    vehicle_orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    vehicle_speed: float = 8.0
    vehicle_wrap_min_x: float = -25.0
    vehicle_wrap_max_x: float = 25.0

    vehicle_cabin_offset: tuple[float, float, float] = (-0.15, 0.0, 0.60)
    vehicle_wheel_z_offset: float = -0.33

    crosswalk_stripe_count: int = 10
    crosswalk_stripe_length_x: float = 0.30
    crosswalk_stripe_gap_x: float = 0.22

    @property
    def road_left_edge_y(self) -> float:
        return -0.5 * self.road_width

    @property
    def road_right_edge_y(self) -> float:
        return 0.5 * self.road_width

    @property
    def h1_start_orientation(self) -> tuple[float, float, float, float]:
        """Return the start quaternion in Isaac Lab's (x, y, z, w) convention."""

        half_yaw = 0.5 * self.h1_start_yaw
        return (0.0, 0.0, sin(half_yaw), cos(half_yaw))

    @property
    def vehicle_wheel_orientation(self) -> tuple[float, float, float, float]:
        """Rotate cylinder wheels +90 degrees around X, in XYZW convention."""

        half_roll = 0.25 * pi
        return (sin(half_roll), 0.0, 0.0, cos(half_roll))


def validate_layout(cfg: CrossingSceneConfig) -> None:
    """Fail early when the start, goal, road, or crossing axes are inconsistent."""

    assert cfg.h1_start_position[1] < cfg.road_left_edge_y, "H1 must start on the Y<0 sidewalk."
    assert cfg.goal_position[1] > cfg.road_right_edge_y, "Goal must be on the Y>0 sidewalk."
    assert cfg.goal_position[1] > cfg.h1_start_position[1], "Goal must be to the right of H1."
    assert abs(cfg.h1_start_position[0] - cfg.goal_position[0]) < 1.0e-6
    assert abs(cfg.h1_start_position[0]) < 1.0e-6, "The start-goal line must cross the road at X=0."
    assert abs(cos(cfg.h1_start_yaw)) < 1.0e-6
    assert sin(cfg.h1_start_yaw) > 0.999, "H1 must face world +Y."
    assert abs(sum(value * value for value in cfg.h1_start_orientation) - 1.0) < 1.0e-6
    assert abs(sum(value * value for value in cfg.vehicle_orientation) - 1.0) < 1.0e-6
    assert abs(sum(value * value for value in cfg.vehicle_wheel_orientation) - 1.0) < 1.0e-6
    assert cfg.vehicle_cabin_offset[2] > 0.0, "Cabin must be above the chassis."
    assert cfg.vehicle_wheel_z_offset < 0.0, "Wheel centers must be below the chassis center."


def spawn_cuboid(
    prim_path: str,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
    color: tuple[float, float, float],
    *,
    collision: bool = False,
    metallic: float = 0.0,
) -> None:
    """Spawn a colored static cuboid, optionally with a collider."""

    cfg = sim_utils.CuboidCfg(
        size=size,
        collision_props=sim_utils.CollisionPropertiesCfg() if collision else None,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=metallic),
    )
    cfg.func(prim_path, cfg, translation=position)


def spawn_road(cfg: CrossingSceneConfig) -> None:
    """Build a two-lane flat road, sidewalks, markings, crossing, and pads."""

    sim_utils.create_prim("/World/Environment", "Xform")

    # A ground plane sits just below the visible road and sidewalks.
    ground_cfg = sim_utils.GroundPlaneCfg(
        color=(0.28, 0.30, 0.29),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.9),
    )
    ground_cfg.func("/World/Environment/Ground", ground_cfg, translation=(0.0, 0.0, -0.05))

    spawn_cuboid(
        "/World/Environment/Road",
        (cfg.road_length, cfg.road_width, cfg.road_thickness),
        (0.0, 0.0, -0.5 * cfg.road_thickness),
        (0.055, 0.065, 0.075),
        collision=True,
    )

    sidewalk_center_y = 0.5 * (cfg.road_width + cfg.sidewalk_width)
    for side_name, side_sign in (("Left", -1.0), ("Right", 1.0)):
        spawn_cuboid(
            f"/World/Environment/Sidewalk{side_name}",
            (cfg.road_length, cfg.sidewalk_width, cfg.sidewalk_height),
            (0.0, side_sign * sidewalk_center_y, 0.5 * cfg.sidewalk_height),
            (0.42, 0.44, 0.43),
            collision=True,
        )

    marking_z = 0.006
    marking_thickness = 0.012
    white = (0.95, 0.95, 0.90)

    # Solid road-edge lines.
    for side_name, side_sign in (("Left", -1.0), ("Right", 1.0)):
        spawn_cuboid(
            f"/World/Environment/Markings/Edge{side_name}",
            (cfg.road_length, 0.08, marking_thickness),
            (0.0, side_sign * (0.5 * cfg.road_width - 0.10), marking_z),
            white,
        )

    # Dashed center line. Leave a gap through the zebra crossing.
    dash_length = 3.0
    dash_spacing = 6.0
    dash_x = -0.5 * cfg.road_length + 0.5 * dash_length
    dash_index = 0
    crossing_half_width = 0.5 * (
        cfg.crosswalk_stripe_count
        * (cfg.crosswalk_stripe_length_x + cfg.crosswalk_stripe_gap_x)
        - cfg.crosswalk_stripe_gap_x
    )
    while dash_x < 0.5 * cfg.road_length:
        if abs(dash_x) > crossing_half_width + 0.5 * dash_length:
            spawn_cuboid(
                f"/World/Environment/Markings/CenterDash_{dash_index:02d}",
                (dash_length, 0.10, marking_thickness),
                (dash_x, 0.0, marking_z),
                white,
            )
        dash_x += dash_spacing
        dash_index += 1

    # Zebra stripes span the drivable road from left to right.
    stripe_pitch = cfg.crosswalk_stripe_length_x + cfg.crosswalk_stripe_gap_x
    first_stripe_x = -0.5 * (cfg.crosswalk_stripe_count - 1) * stripe_pitch
    for stripe_index in range(cfg.crosswalk_stripe_count):
        stripe_x = first_stripe_x + stripe_index * stripe_pitch
        spawn_cuboid(
            f"/World/Environment/Crosswalk/Stripe_{stripe_index:02d}",
            (cfg.crosswalk_stripe_length_x, cfg.road_width - 0.30, marking_thickness),
            (stripe_x, 0.0, marking_z + 0.001),
            white,
        )

    # Start and goal pads are visual-only and sit on top of the sidewalks.
    pad_z = cfg.sidewalk_height + 0.008
    spawn_cuboid(
        "/World/Environment/StartPad",
        (1.2, 1.2, 0.016),
        (cfg.h1_start_position[0], cfg.h1_start_position[1], pad_z),
        (0.15, 0.55, 0.90),
        metallic=0.1,
    )
    spawn_cuboid(
        "/World/Environment/GoalPad",
        (1.2, 1.2, 0.016),
        (cfg.goal_position[0], cfg.goal_position[1], pad_z),
        (0.10, 0.80, 0.25),
        metallic=0.1,
    )


def spawn_h1(cfg: CrossingSceneConfig) -> Articulation:
    """Spawn a fixed-root H1 at the left-side start pad."""

    h1_cfg = H1_MINIMAL_CFG.replace(prim_path="/World/H1")
    h1_cfg.init_state.pos = cfg.h1_start_position
    h1_cfg.init_state.rot = cfg.h1_start_orientation
    h1_cfg.spawn.articulation_props.fix_root_link = True
    return Articulation(cfg=h1_cfg)


def spawn_vehicle(cfg: CrossingSceneConfig) -> RigidObject:
    """Spawn a simple green kinematic car made from Isaac Lab primitives."""

    sim_utils.create_prim("/World/Vehicle", "Xform")
    vehicle_cfg = RigidObjectCfg(
        prim_path="/World/Vehicle/Chassis",
        spawn=sim_utils.CuboidCfg(
            size=(4.2, 1.8, 0.65),
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
    vehicle = RigidObject(cfg=vehicle_cfg)

    # Visual children inherit the kinematic chassis transform. The chassis is
    # the single collision proxy for this preview.
    cabin_cfg = sim_utils.CuboidCfg(
        size=(2.2, 1.55, 0.65),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.08, 0.65, 0.42), metallic=0.15, roughness=0.28
        ),
    )
    cabin_cfg.func(
        "/World/Vehicle/Chassis/Cabin",
        cabin_cfg,
        translation=cfg.vehicle_cabin_offset,
    )

    wheel_cfg = sim_utils.CylinderCfg(
        radius=0.35,
        height=0.30,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.025, 0.025, 0.030), metallic=0.0, roughness=0.82
        ),
    )
    for axle_name, wheel_x in (("Rear", -1.35), ("Front", 1.35)):
        for side_name, wheel_y in (("Left", -0.94), ("Right", 0.94)):
            wheel_cfg.func(
                f"/World/Vehicle/Chassis/Wheel{axle_name}{side_name}",
                wheel_cfg,
                translation=(wheel_x, wheel_y, cfg.vehicle_wheel_z_offset),
                orientation=cfg.vehicle_wheel_orientation,
            )

    return vehicle


def reset_scene(
    h1: Articulation,
    vehicle: RigidObject,
    cfg: CrossingSceneConfig,
    device: str,
) -> float:
    """Restore H1 to the left, keep the goal on the right, and reset the car."""

    h1_root_pose = h1.data.default_root_pose.torch.clone()
    h1_root_pose[:, :3] = torch.tensor(cfg.h1_start_position, device=device)
    h1_root_pose[:, 3:7] = torch.tensor(cfg.h1_start_orientation, device=device)
    h1.write_root_pose_to_sim_index(root_pose=h1_root_pose)
    h1.write_root_velocity_to_sim_index(root_velocity=torch.zeros_like(h1.data.default_root_vel.torch))
    h1.write_joint_position_to_sim_index(position=h1.data.default_joint_pos.torch.clone())
    h1.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(h1.data.default_joint_vel.torch))
    h1.reset()

    vehicle_root_pose = vehicle.data.default_root_pose.torch.clone()
    vehicle_root_pose[:, :3] = torch.tensor(cfg.vehicle_start_position, device=device)
    vehicle_root_pose[:, 3:7] = torch.tensor(cfg.vehicle_orientation, device=device)
    vehicle.write_root_pose_to_sim_index(root_pose=vehicle_root_pose)
    vehicle.write_root_velocity_to_sim_index(root_velocity=torch.zeros_like(vehicle.data.default_root_vel.torch))
    vehicle.reset()

    return cfg.vehicle_start_position[0]


def quaternions_equivalent(actual: torch.Tensor, expected: torch.Tensor, atol: float = 1.0e-3) -> bool:
    """Check XYZW quaternions while accepting the equivalent negated representation."""

    return bool(torch.allclose(actual, expected, atol=atol) or torch.allclose(actual, -expected, atol=atol))


def validate_runtime_state(
    h1: Articulation,
    vehicle: RigidObject,
    cfg: CrossingSceneConfig,
) -> None:
    """Validate that the robot and car remained upright during the preview."""

    h1_root_pose = h1.data.root_pose_w.torch[0]
    vehicle_root_pose = vehicle.data.root_pose_w.torch[0]
    h1_expected_quat = torch.tensor(cfg.h1_start_orientation, device=h1.device, dtype=h1_root_pose.dtype)
    vehicle_expected_quat = torch.tensor(
        cfg.vehicle_orientation,
        device=vehicle.device,
        dtype=vehicle_root_pose.dtype,
    )

    assert torch.allclose(
        h1_root_pose[:3],
        torch.tensor(cfg.h1_start_position, device=h1.device, dtype=h1_root_pose.dtype),
        atol=2.0e-3,
    ), f"H1 root moved from its fixed start pose: {h1_root_pose.tolist()}"
    assert quaternions_equivalent(h1_root_pose[3:7], h1_expected_quat), (
        f"H1 is not upright/facing +Y: {h1_root_pose[3:7].tolist()}"
    )
    assert quaternions_equivalent(vehicle_root_pose[3:7], vehicle_expected_quat), (
        f"Vehicle is not upright: {vehicle_root_pose[3:7].tolist()}"
    )

    max_joint_error = torch.max(
        torch.abs(h1.data.joint_pos.torch - h1.data.default_joint_pos.torch)
    ).item()
    assert max_joint_error < 0.15, f"H1 standing joints drifted too far: max error={max_joint_error:.4f} rad"

    print(
        "[INFO] Runtime validation passed: "
        f"H1 quat={h1_root_pose[3:7].tolist()}, "
        f"vehicle quat={vehicle_root_pose[3:7].tolist()}, "
        f"max_joint_error={max_joint_error:.5f} rad."
    )


def run_simulator(
    sim: SimulationContext,
    h1: Articulation,
    vehicle: RigidObject,
    cfg: CrossingSceneConfig,
    max_steps: int,
) -> None:
    """Hold H1 at the start pad and move the scripted car along world +X."""

    sim_dt = sim.get_physics_dt()
    vehicle_x = reset_scene(h1, vehicle, cfg, sim.device)
    vehicle_pose = vehicle.data.default_root_pose.torch.clone()
    default_joint_positions = h1.data.default_joint_pos.torch.clone()
    step_count = 0

    print(
        "[INFO] Scene ready: H1 starts at Y<0, goal is at Y>0, "
        f"vehicle speed={cfg.vehicle_speed:.2f} m/s."
    )

    while simulation_app.is_running() and (max_steps < 0 or step_count < max_steps):
        vehicle_x += cfg.vehicle_speed * sim_dt
        if vehicle_x > cfg.vehicle_wrap_max_x:
            vehicle_x = cfg.vehicle_wrap_min_x

        vehicle_pose[:, :3] = torch.tensor(
            (vehicle_x, cfg.vehicle_start_position[1], cfg.vehicle_start_position[2]),
            device=sim.device,
        )
        vehicle_pose[:, 3:7] = torch.tensor(cfg.vehicle_orientation, device=sim.device)
        vehicle.write_root_pose_to_sim_index(root_pose=vehicle_pose)

        # The root is fixed for this preview; position targets preserve the
        # default standing pose for all movable joints.
        h1.set_joint_position_target_index(target=default_joint_positions)
        h1.write_data_to_sim()
        vehicle.write_data_to_sim()

        sim.step()
        h1.update(sim_dt)
        vehicle.update(sim_dt)
        step_count += 1

    validate_runtime_state(h1, vehicle, cfg)
    print(f"[INFO] Preview completed after {step_count} physics steps; vehicle_x={vehicle_x:.3f}.")


def main() -> None:
    """Create and run the crossing preview."""

    scene_cfg = CrossingSceneConfig(vehicle_speed=args_cli.vehicle_speed)
    validate_layout(scene_cfg)

    sim_cfg = sim_utils.SimulationCfg(dt=scene_cfg.physics_dt, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # With this view, the Y<0 spawn side appears on the left of the frame.
    sim.set_camera_view(eye=(18.0, -18.0, 16.0), target=(0.0, 0.0, 0.7))

    dome_cfg = sim_utils.DomeLightCfg(intensity=2200.0, color=(0.78, 0.82, 0.88))
    dome_cfg.func("/World/Lights/Dome", dome_cfg)
    distant_cfg = sim_utils.DistantLightCfg(intensity=2800.0, color=(1.0, 0.94, 0.84), angle=0.55)
    distant_cfg.func("/World/Lights/Sun", distant_cfg, rotation=(0.35, -0.55, 0.25))

    spawn_road(scene_cfg)
    h1 = spawn_h1(scene_cfg)
    vehicle = spawn_vehicle(scene_cfg)

    sim.reset()
    run_simulator(sim, h1, vehicle, scene_cfg, args_cli.steps)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
