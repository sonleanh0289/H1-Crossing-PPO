"""Run the published H1 flat locomotion policy in the crossing scene.

This script intentionally reuses Isaac Lab's native H1 flat Manager-Based
environment.  Consequently, the policy receives the exact 69-dimensional
observation layout and produces the exact 19-dimensional action layout used
during training; the crossing geometry does not alter that contract.

Examples (from ``D:\\IssacSim\\IsaacLab``):

    isaaclab.bat -p ..\\H1Crossing\\scripts\\play_crossing_policy.py --viz kit --joystick
    isaaclab.bat -p ..\\H1Crossing\\scripts\\play_crossing_policy.py --viz none --steps 300
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import sys
from dataclasses import dataclass
from math import cos, pi, sin
from windows_joystick import WindowsJoystick, CommandMapper

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Play the H1 flat policy in the road-crossing scene.")
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Optional local checkpoint path. By default, use the published H1 Flat checkpoint.",
)
parser.add_argument(
    "--walk_speed",
    type=float,
    default=0.6,
    help="Forward command in m/s. The published policy was trained for 0 <= vx <= 1.",
)
parser.add_argument(
    "--vehicle_speed",
    type=float,
    default=8.0,
    help="Scripted vehicle speed in m/s along world +X.",
)
parser.add_argument(
    "--steps",
    type=int,
    default=-1,
    help="Number of policy steps to run. Use -1 to run until the simulator closes.",
)
parser.add_argument(
    "--stop_at_goal",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Command zero velocity after H1 reaches the right-side goal.",
)
parser.add_argument(
    "--joystick",
    action="store_true",
    help="Use gamepad commands instead of the automatic forward command. Requires --viz kit.",
)
parser.add_argument(
    "--joystick_dead_zone",
    type=float,
    default=0.05,
    help="Ignore stick magnitudes below this value.",
)
parser.add_argument(
    "--yaw_rate",
    type=float,
    default=0.5,
    help="Maximum joystick yaw command in rad/s.",
)
parser.add_argument("--joystick_backend", choices=("auto", "windows", "kit"), default="auto")
parser.add_argument("--joystick_id", type=int, default=None, help="Windows joystick device id; default first connected.")
parser.add_argument("--joystick_forward_axis", choices=tuple("xyzruv"), default="y")
parser.add_argument("--joystick_turn_axis", choices=tuple("xyzruv"), default="x")
parser.add_argument("--joystick_forward_sign", type=int, choices=(-1, 1), default=-1)
parser.add_argument("--joystick_turn_sign", type=int, choices=(-1, 1), default=-1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not 0.0 <= args_cli.walk_speed <= 1.0:
    parser.error("--walk_speed must be in the checkpoint's trained range [0.0, 1.0].")
if args_cli.vehicle_speed <= 0.0:
    parser.error("--vehicle_speed must be greater than zero.")
if args_cli.steps == 0 or args_cli.steps < -1:
    parser.error("--steps must be -1 or a positive integer.")
if not 0.0 <= args_cli.joystick_dead_zone < 1.0:
    parser.error("--joystick_dead_zone must be in [0.0, 1.0).")
if not 0.0 <= args_cli.yaw_rate <= 1.0:
    parser.error("--yaw_rate must be in the checkpoint's trained range [0.0, 1.0].")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch
from packaging import version
from rsl_rl.runners import OnPolicyRunner

import carb
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.devices import Se2Gamepad, Se2GamepadCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.agents.rsl_rl_ppo_cfg import (
    H1FlatPPORunnerCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.flat_env_cfg import H1FlatEnvCfg_PLAY


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


@dataclass
class JoystickState:
    """State changed by idempotent gamepad button callbacks."""

    emergency_stop: bool = False
    reset_requested: bool = False


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


def add_crossing_assets(env_cfg: H1FlatEnvCfg_PLAY, cfg: CrossingConfig) -> None:
    """Add road, sidewalks, markings, pads, and the procedural vehicle."""

    env_ns = "{ENV_REGEX_NS}"
    white = (0.95, 0.95, 0.90)

    # The native Flat task's global plane remains the physical road surface.
    # The dark road slab is visual-only to avoid coincident collision surfaces.
    env_cfg.scene.road = make_static_cuboid(
        f"{env_ns}/Crossing/Road",
        (cfg.road_length, cfg.road_width, cfg.road_thickness),
        (0.0, 0.0, -0.5 * cfg.road_thickness),
        (0.055, 0.065, 0.075),
    )

    sidewalk_center_y = 0.5 * (cfg.road_width + cfg.sidewalk_width)
    env_cfg.scene.sidewalk_left = make_static_cuboid(
        f"{env_ns}/Crossing/SidewalkLeft",
        (cfg.road_length, cfg.sidewalk_width, cfg.sidewalk_height),
        (0.0, -sidewalk_center_y, 0.5 * cfg.sidewalk_height),
        (0.42, 0.44, 0.43),
        collision=True,
    )
    env_cfg.scene.sidewalk_right = make_static_cuboid(
        f"{env_ns}/Crossing/SidewalkRight",
        (cfg.road_length, cfg.sidewalk_width, cfg.sidewalk_height),
        (0.0, sidewalk_center_y, 0.5 * cfg.sidewalk_height),
        (0.42, 0.44, 0.43),
        collision=True,
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

    for dash_index, dash_x in enumerate((-27.0, -21.0, -15.0, -9.0, 9.0, 15.0, 21.0, 27.0)):
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

    pad_z = cfg.sidewalk_height + 0.008
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


def build_environment_config(cfg: CrossingConfig) -> H1FlatEnvCfg_PLAY:
    """Create a deterministic, single-environment H1 Flat crossing config."""

    env_cfg = H1FlatEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 12.0
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = 42
    env_cfg.episode_length_s = 60.0

    env_cfg.scene.robot.init_state.pos = cfg.h1_start_position
    env_cfg.scene.robot.init_state.rot = cfg.h1_start_orientation
    env_cfg.scene.robot.spawn.articulation_props.fix_root_link = False

    # Every automatic reset returns H1 to the exact start pose with zero speed.
    env_cfg.events.reset_base.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    env_cfg.events.reset_base.params["velocity_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }

    # Commands are overwritten explicitly before every policy inference. These
    # zero ranges prevent the manager's periodic resampling from injecting yaw.
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.debug_vis = False
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)

    env_cfg.viewer.origin_type = "world"
    env_cfg.viewer.eye = cfg.camera_eye
    env_cfg.viewer.lookat = cfg.camera_target

    add_crossing_assets(env_cfg, cfg)
    return env_cfg


def initialize_gui_view(base_env: ManagerBasedRLEnv, cfg: CrossingConfig) -> None:
    """Force the Kit viewport onto the crossing after all renderers exist."""

    if not base_env.sim.has_gui and not base_env.sim.has_active_visualizers():
        return

    if base_env.viewport_camera_controller is not None:
        base_env.viewport_camera_controller.update_view_location(
            eye=cfg.camera_eye,
            lookat=cfg.camera_target,
        )
    else:
        base_env.sim.set_camera_view(eye=cfg.camera_eye, target=cfg.camera_target)

    # Pump a few Kit frames now. Without this, the window can remain on its
    # empty startup frame while actor/checkpoint initialization blocks Python.
    for _ in range(8):
        base_env.sim.render()
    print("[INFO] GUI viewport focused on the crossing scene.", flush=True)


def resolve_checkpoint() -> str:
    """Resolve an explicit checkpoint or the official published Flat H1 one."""

    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    checkpoint = get_published_pretrained_checkpoint("rsl_rl", "Isaac-Velocity-Flat-H1-v0")
    if not checkpoint:
        raise FileNotFoundError("Published checkpoint for Isaac-Velocity-Flat-H1-v0 is unavailable.")
    return checkpoint


class WindowsGamepad:
    """Convert polled Windows input to the native controller tensor contract."""

    def __init__(self, device: str, state: JoystickState):
        self.reader = WindowsJoystick(args_cli.joystick_id, args_cli.joystick_dead_zone)
        for axis in (args_cli.joystick_forward_axis, args_cli.joystick_turn_axis):
            if axis not in self.reader.axis_names:
                raise ValueError(f"Joystick axis {axis!r} unavailable: {self.reader.axis_names}")
        self.mapper = CommandMapper(state, args_cli.walk_speed, args_cli.yaw_rate,
                                    args_cli.joystick_forward_axis, args_cli.joystick_turn_axis,
                                    args_cli.joystick_forward_sign, args_cli.joystick_turn_sign)
        self.device = device
        self.last_sample = self.reader.read()
        print(f"[INFO] {self.reader.describe()}", flush=True)
        print("[INFO] Windows controls: left stick up=forward, left/right=turn (default axes); "
              "button 1=resume, 2=stop, 3=reset. Release sticks to neutral to arm.", flush=True)

    def advance(self):
        sample = self.reader.read()
        if sample.connected != self.last_sample.connected:
            print(f"[INFO] Joystick connected={sample.connected}; release sticks to neutral.", flush=True)
        self.last_sample = sample
        return torch.tensor(self.mapper.advance(sample), device=self.device, dtype=torch.float32)

    def reset(self):
        self.mapper.reset()


def create_gamepad(device: str) -> tuple[Se2Gamepad | WindowsGamepad, JoystickState]:
    """Create the native Isaac Lab gamepad and bind safety controls."""

    backend = args_cli.joystick_backend
    if backend == "auto":
        backend = "windows" if sys.platform == "win32" else "kit"
    if backend == "windows":
        state = JoystickState()
        return WindowsGamepad(device, state), state

    gamepad_cfg = Se2GamepadCfg(
        v_x_sensitivity=args_cli.walk_speed,
        v_y_sensitivity=0.0,
        omega_z_sensitivity=args_cli.yaw_rate,
        dead_zone=args_cli.joystick_dead_zone,
        sim_device=device,
    )
    gamepad = Se2Gamepad(gamepad_cfg)
    state = JoystickState()

    # Callbacks can be emitted for both button press and release. Each action is
    # therefore idempotent: A resumes, B stops, and X requests a scene reset.
    gamepad.add_callback(carb.input.GamepadInput.A, lambda: setattr(state, "emergency_stop", False))
    gamepad.add_callback(carb.input.GamepadInput.B, lambda: setattr(state, "emergency_stop", True))
    gamepad.add_callback(carb.input.GamepadInput.X, lambda: setattr(state, "reset_requested", True))

    print(gamepad)
    print(
        "[INFO] Joystick controls: left stick up=forward, left stick down=stop, "
        "right stick=turn, A=resume, B=emergency stop, X=reset."
    )
    return gamepad, state


def run_policy(
    base_env: ManagerBasedRLEnv,
    env: RslRlVecEnvWrapper,
    policy,
    cfg: CrossingConfig,
    gamepad: Se2Gamepad | WindowsGamepad | None = None,
    joystick_state: JoystickState | None = None,
) -> None:
    """Run policy inference, joystick/automatic commands, and car motion."""

    robot = base_env.scene["robot"]
    vehicle = base_env.scene["vehicle"]
    command_term = base_env.command_manager.get_term("base_velocity")
    command = torch.zeros((base_env.num_envs, 3), device=base_env.device)
    stopped_at_goal = torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)

    vehicle_x = cfg.vehicle_start_position[0]
    vehicle_pose = vehicle.data.root_pose_w.torch.clone()
    policy_dt = base_env.step_dt
    step_count = 0
    control_mode = "joystick" if gamepad is not None else "automatic"

    print(
        "[INFO] Crossing policy ready: "
        f"control={control_mode}, start={cfg.h1_start_position}, goal={cfg.goal_position}, "
        f"max/auto walk speed={args_cli.walk_speed:.2f} m/s, "
        f"vehicle speed={args_cli.vehicle_speed:.2f} m/s."
    )

    while simulation_app.is_running() and (args_cli.steps < 0 or step_count < args_cli.steps):
        if joystick_state is not None and joystick_state.reset_requested:
            env.reset()
            joystick_state.reset_requested = False
            joystick_state.emergency_stop = False
            stopped_at_goal.zero_()
            vehicle_x = cfg.vehicle_start_position[0]
            vehicle_pose = vehicle.data.root_pose_w.torch.clone()
            gamepad.reset()
            print("[INFO] Joystick reset: H1 and vehicle returned to their start poses.")

        root_pose = robot.data.root_pose_w.torch
        if args_cli.stop_at_goal:
            stopped_at_goal |= root_pose[:, 1] >= cfg.goal_position[1]

        command.zero_()
        if gamepad is not None:
            joystick_command = gamepad.advance()
            # Device adapters share a tensor contract; all command clamps below
            # remain identical to those used with the published flat checkpoint.
            # The published Flat H1 policy was trained with vx in [0, 1] and
            # vy fixed at zero. Pulling the stick down therefore means stop,
            # not reverse; lateral commands are deliberately locked out.
            command[:, 0] = torch.clamp(joystick_command[0], min=0.0, max=1.0)
            command[:, 1] = 0.0
            command[:, 2] = torch.clamp(joystick_command[2], min=-1.0, max=1.0)
            if joystick_state.emergency_stop:
                command.zero_()
        else:
            command[:, 0] = args_cli.walk_speed

        command[stopped_at_goal] = 0.0
        command_term.vel_command_b[:] = command

        # Recompute after setting the command so the policy observes the exact
        # desired [vx, vy, yaw_rate], not a value sampled by the command manager.
        obs = env.get_observations()
        with torch.inference_mode():
            actions = policy(obs)

        vehicle_x += args_cli.vehicle_speed * policy_dt
        if vehicle_x > cfg.vehicle_wrap_max_x:
            vehicle_x = cfg.vehicle_wrap_min_x
        vehicle_pose[:, :3] = torch.tensor(
            (vehicle_x, cfg.vehicle_start_position[1], cfg.vehicle_start_position[2]),
            device=base_env.device,
            dtype=vehicle_pose.dtype,
        )
        vehicle_pose[:, 3:7] = torch.tensor(
            cfg.vehicle_orientation,
            device=base_env.device,
            dtype=vehicle_pose.dtype,
        )
        vehicle.write_root_pose_to_sim_index(root_pose=vehicle_pose)

        _, _, dones, _ = env.step(actions)
        if version.parse(metadata.version("rsl-rl-lib")) >= version.parse("4.0.0"):
            policy.reset(dones)

        if torch.any(dones):
            stopped_at_goal[dones] = False
            vehicle_x = cfg.vehicle_start_position[0]

        if gamepad is not None and step_count % 100 == 0:
            if isinstance(gamepad, WindowsGamepad):
                print(f"[INFO] joystick connected={gamepad.last_sample.connected}, "
                      f"armed={gamepad.mapper.armed}, axes={gamepad.last_sample.axes}, "
                      f"buttons={gamepad.last_sample.buttons}", flush=True)
            print(
                f"[INFO] joystick command={command[0].tolist()}, "
                f"H1_y={root_pose[0, 1].item():.3f}, stop={joystick_state.emergency_stop}, "
                f"at_goal={stopped_at_goal[0].item()}", flush=True,
            )

        step_count += 1

    final_pose = robot.data.root_pose_w.torch[0]
    assert torch.isfinite(final_pose).all(), f"H1 root state contains NaN/Inf: {final_pose.tolist()}"
    print(
        f"[INFO] Crossing run completed after {step_count} policy steps; "
        f"H1 position={final_pose[:3].tolist()}, vehicle_x={vehicle_x:.3f}."
    )

def main() -> None:
    """Build the Manager-Based scene, load the Flat checkpoint, and run inference."""

    crossing_cfg = CrossingConfig()
    validate_config(crossing_cfg)
    checkpoint_path = resolve_checkpoint()

    env_cfg = build_environment_config(crossing_cfg)
    base_env = ManagerBasedRLEnv(cfg=env_cfg)
    initialize_gui_view(base_env, crossing_cfg)

    agent_cfg = H1FlatPPORunnerCfg()
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    agent_cfg.device = base_env.device

    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[INFO] Loading H1 Flat checkpoint: {checkpoint_path}", flush=True)
    # Inference only needs actor tensors. RSL-RL's runner.load() deserializes
    # the complete training checkpoint with weights_only=False, so load the
    # official actor state safely and skip critic/optimizer state altogether.
    checkpoint_data = torch.load(checkpoint_path, map_location=base_env.device, weights_only=True)
    runner.alg.actor.load_state_dict(checkpoint_data["actor_state_dict"], strict=True)
    print("[INFO] H1 Flat actor weights loaded.", flush=True)
    policy = runner.alg.actor

    obs = env.get_observations()
    policy_obs_shape = tuple(obs["policy"].shape)
    action_dim = base_env.action_manager.total_action_dim
    assert policy_obs_shape == (1, 69), (
        f"Expected checkpoint policy observation shape (1, 69), got {policy_obs_shape}"
    )
    assert action_dim == 19, f"Expected checkpoint action dimension 19, got {action_dim}"
    print(
        f"[INFO] Policy contract validated: observation={policy_obs_shape}, action_dim={action_dim}.",
        flush=True,
    )

    gamepad = None
    joystick_state = None
    if args_cli.joystick:
        gamepad, joystick_state = create_gamepad(str(base_env.device))

    try:
        run_policy(base_env, env, policy, crossing_cfg, gamepad, joystick_state)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
