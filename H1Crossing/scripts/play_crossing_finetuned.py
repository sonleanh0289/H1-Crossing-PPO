"""Validated flat H1 bundle in crossing geometry; no training or core modifications.

GUI: python scripts/play_crossing_finetuned.py --keyboard
Smoke: python scripts/play_crossing_finetuned.py --headless --auto --walk_speed 0 --steps 300
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FINETUNE = ROOT.parent / "H1FlatFineTune"
sys.path.insert(0, str(FINETUNE))
from crossing_bundle import validate_bundle, sha256
from vehicle_motion import VehicleMotion, TrafficConfig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "checkpoints/h1_flat_stability_v2_0500")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--keyboard", action="store_true")
    inputs.add_argument("--joystick", action="store_true")
    inputs.add_argument("--auto", action="store_true", help="Fixed forward command; no vehicle avoidance")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--h1_camera", action="store_true", help="Show an additional display-only H1 first-person viewport")
    parser.add_argument("--capture_viewport", action="store_true", help="Save a GUI viewport PNG at the end of play")
    parser.add_argument("--record_crossing_observations", action="store_true",
                        help="Record raw 8D crossing state at each 50 Hz policy step (no camera)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=-1)
    parser.add_argument("--walk_speed", type=float, default=1.0)
    parser.add_argument("--vehicle_speed", type=float, default=None,
                        help="Override random traffic with legacy fixed speed in m/s; 0 parks at X=-20")
    parser.add_argument("--vehicle_seed", type=int, default=None,
                        help="Replayable traffic RNG seed; omitted generates a new seed each launch")
    parser.add_argument("--traffic_mode", choices=("variable", "constant"), default="variable",
                        help="Variable speed by default; --vehicle_speed overrides both modes")
    parser.add_argument("--yield_probability", type=float, default=.5,
                        help="Probability of a yielding driver per spawn; fixed --vehicle_speed disables yielding")
    parser.add_argument("--keyboard_speed", type=float, default=1.0)
    parser.add_argument("--keyboard_reverse_speed", type=float, default=0.5)
    parser.add_argument("--joystick_id", type=int)
    parser.add_argument("--forward_axis", choices=tuple("xyzruv"), default="y")
    parser.add_argument("--turn_axis", choices=tuple("xyzruv"), default="x")
    parser.add_argument("--forward_sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--turn_sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--raised_sidewalks", action="store_true", help="Opt-in 15 cm curbs, NOT validated by flat tests")
    parser.add_argument("--stop_at_goal", action=argparse.BooleanOptionalAction, default=None,
                        help="Default on for auto, off for manual so you can reverse after crossing")
    args = parser.parse_args(argv)
    if not math.isfinite(args.yield_probability) or not 0 <= args.yield_probability <= 1:
        parser.error("--yield_probability must be between 0 and 1")
    if not (args.keyboard or args.joystick or args.auto):
        args.keyboard = not args.headless
        args.auto = args.headless
    if args.keyboard and (args.headless or sys.platform != "win32"):
        parser.error("Keyboard needs the Windows simulator GUI")
    if args.capture_viewport and args.headless:
        parser.error("--capture_viewport requires the GUI")
    if args.h1_camera and args.headless:
        parser.error("--h1_camera requires the GUI; no headless camera sensor is configured")
    if args.steps == 0 or args.steps < -1:
        parser.error("--steps must be -1 or positive")
    for name in ("walk_speed", "vehicle_speed", "keyboard_speed", "keyboard_reverse_speed"):
        value = getattr(args, name)
        if name == "vehicle_speed" and value is None:
            continue
        if not math.isfinite(value) or value < 0 or (name.startswith("keyboard") and value == 0):
            parser.error(f"Invalid --{name}")
    if args.stop_at_goal is None:
        args.stop_at_goal = args.auto
    return args


def build_config(args, stage, layout):
    from h1_finetune.tasks import RunRetreatPlayCfg
    from h1_finetune.control import STAGES
    from crossing_geometry import add_crossing_assets
    cfg = RunRetreatPlayCfg()
    cfg.seed = 42
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.episode_length_s = 60.0
    cfg.commands.base_velocity.stage = stage
    cfg.commands.base_velocity.ranges.lin_vel_x = STAGES[stage]
    cfg.scene.robot.init_state.pos = layout.h1_start_position
    cfg.scene.robot.init_state.rot = layout.h1_start_orientation
    cfg.events.reset_base.params["pose_range"] = {key: (0., 0.) for key in ("x", "y", "z", "roll", "pitch", "yaw")}
    cfg.viewer.origin_type = "world"
    cfg.viewer.eye, cfg.viewer.lookat = layout.camera_eye, layout.camera_target
    add_crossing_assets(cfg, layout, flat_sidewalks=not args.raised_sidewalks)
    from crossing_observations import add_crossing_observation_group
    add_crossing_observation_group(cfg, layout)
    if args.h1_camera:
        from isaaclab.assets import AssetBaseCfg
        import isaaclab.sim as sim_utils
        from h1_camera import H1CameraConfig
        camera_cfg = H1CameraConfig()
        cfg.scene.h1_view_camera = AssetBaseCfg(prim_path=camera_cfg.prim_path, spawn=sim_utils.PinholeCameraCfg())
    return cfg


def run(args, app, manifest, output):
    import csv
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from h1_finetune.runtime import validate_contract, plain
    from h1_finetune.numerics import finite
    from h1_finetune.control import KeyboardMapper, JoystickMapper
    from crossing_geometry import CrossingConfig, validate_config
    from crossing_observations import (OBSERVATION_DIM, OBSERVATION_NAMES,
                                       observation_schema, set_scripted_vehicle_velocity)

    layout = CrossingConfig()
    if args.vehicle_speed is None:
        layout = replace(layout, road_length=TrafficConfig().road_length_m)
    if not args.raised_sidewalks:
        layout = replace(layout, goal_position=(0., 4.6, .017))
    validate_config(layout)
    cfg = build_config(args, manifest["validated_stage"], layout)
    camera_window = None
    overview = None
    if not args.headless:
        from omni.kit.viewport.utility import get_active_viewport, create_viewport_window
        overview = get_active_viewport()
        if args.h1_camera:
            # Create all Hydra viewports before physics/Fabric initializes the scene.
            camera_window = create_viewport_window("H1 Camera", width=640, height=480, position_x=40, position_y=80)
    try:
        base = ManagerBasedRLEnv(cfg=cfg)
    except Exception:
        if camera_window is not None:
            camera_window.destroy()
        raise
    env = None
    h1_camera = None
    observation_stream = None
    try:
        env = RslRlVecEnvWrapper(base)
        contract = validate_contract(base, cfg)
        if contract != manifest["contract"]:
            raise ValueError("Crossing observation/action/actuator contract differs from exported policy")
        if tuple(base.observation_manager.group_obs_dim["crossing"]) != (OBSERVATION_DIM,):
            raise ValueError("Crossing observation group must be 8D, separate from 69D policy")
        schema = observation_schema(layout)
        (output / "crossing_observation_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
        (output / "env_config.json").write_text(json.dumps(plain(cfg.to_dict()), indent=2), encoding="utf-8")
        policy = torch.jit.load(str(args.bundle / "policy.pt"), map_location=base.device).eval()
        reader, mapper = None, None
        if args.keyboard:
            from h1_finetune.windows_keyboard import WindowsKeyboard
            reader = WindowsKeyboard()
            mapper = KeyboardMapper(manifest["validated_stage"], args.keyboard_speed, args.keyboard_reverse_speed)
            print("[Controls] Click viewport, release keys. WASD move/turn; Shift=max speed; Space=stop; Enter=resume; R=reset.", flush=True)
        elif args.joystick:
            from h1_finetune.windows_joystick import WindowsJoystick
            reader = WindowsJoystick(args.joystick_id)
            mapper = JoystickMapper(manifest["validated_stage"], args.forward_axis, args.turn_axis,
                                    args.forward_sign, args.turn_sign)
            print(reader.describe(), flush=True)
            print("[Controls] Neutral first; up/down=forward/reverse; button 1=resume, 2=stop, 3=reset.", flush=True)

        robot, vehicle = base.scene["robot"], base.scene["vehicle"]
        term = base.command_manager.get_term("base_velocity")
        term.manual = True
        traffic = VehicleMotion(seed=args.vehicle_seed, fixed_speed_mps=args.vehicle_speed,
                                config=TrafficConfig(mode=args.traffic_mode, yield_probability=args.yield_probability))
        reached, falls, resets = False, 0, 0
        pose = vehicle.data.root_pose_w.torch.clone()

        def record_vehicle_pass():
            event = traffic.events[-1]
            print(f"[Vehicle] {event['reason']}: X={traffic.x:.2f} m, "
                  f"speed={traffic.speed_mps * 3.6:.2f} km/h; seed={traffic.seed}", flush=True)
            (output / "vehicle_traffic.json").write_text(json.dumps(traffic.report(), indent=2), encoding="utf-8")

        def place_vehicle():
            pose[:, :3] = pose.new_tensor((traffic.x, *layout.vehicle_start_position[1:])) + base.scene.env_origins
            pose[:, 3:] = pose.new_tensor(layout.vehicle_orientation)
            vehicle.write_root_pose_to_sim_index(root_pose=pose)
            velocity_xy = pose.new_zeros((base.num_envs, 2))
            velocity_xy[:, 0] = traffic.speed_mps
            set_scripted_vehicle_velocity(base, velocity_xy)

        def reset_scene():
            nonlocal reached, resets
            base.reset()
            reached = False
            traffic.reset("initial" if resets == 0 else "manual_reset")
            record_vehicle_pass()
            place_vehicle()
            if mapper:
                mapper.reset()
            if h1_camera is not None:
                h1_camera.update()
            resets += 1
            return env.get_observations()

        obs = reset_scene()
        finite(obs)
        initial_crossing = obs["crossing"].cpu().tolist()
        observation_writer = None
        if args.record_crossing_observations:
            observation_stream = (output / "crossing_observations.csv").open("w", newline="", encoding="utf-8")
            observation_writer = csv.writer(observation_stream)
            observation_writer.writerow(["step", "sim_time_s", "reset_count", "vehicle_pass", *OBSERVATION_NAMES])
            observation_writer.writerow([0, traffic.sim_time_s, resets, len(traffic.events), *initial_crossing[0]])
        start_position = robot.data.root_pos_w.torch[0].clone()
        if not args.headless:
            if base.viewport_camera_controller is not None:
                base.viewport_camera_controller.update_view_location(eye=layout.camera_eye, lookat=layout.camera_target)
            else:
                base.sim.set_camera_view(eye=layout.camera_eye, target=layout.camera_target)
            if args.h1_camera:
                from h1_camera import H1Camera
                h1_camera = H1Camera(robot, base.sim.stage, camera_window)
                # Render once, after updating the camera from the latest physics pose.
                base.render_enabled = False
                print("[Camera] H1 Camera: 640x480, 90 deg HFOV; display only, not policy input.", flush=True)
            for _ in range(8):
                base.sim.render()
            if h1_camera is not None:
                overview.camera_path = cfg.viewer.cam_prim_path
                from isaaclab_physx.renderers.kit_viewport_utils import set_kit_renderer_camera_view
                set_kit_renderer_camera_view(layout.camera_eye, layout.camera_target, cfg.viewer.cam_prim_path)
                h1_camera.viewport.resolution = (h1_camera.cfg.width, h1_camera.cfg.height)
                h1_camera.update()
        print(f"[Ready] Stage {manifest['validated_stage']} [-1,3] m/s; flat sidewalks={not args.raised_sidewalks}; policy 50 Hz; physics 200 Hz.", flush=True)
        print("[Warning] Locomotion only: the policy does NOT see or avoid the vehicle.", flush=True)
        print("[Observations] crossing=8D ground truth; policy=69D frozen locomotion; no camera input.", flush=True)
        steps = 0
        from h1_camera import gpu_memory_snapshot
        gpu_start = gpu_memory_snapshot() if h1_camera is not None else None
        loop_start = time.monotonic()
        with torch.inference_mode():
            while app.is_running() and (args.steps < 0 or steps < args.steps):
                start = time.monotonic()
                command = (args.walk_speed, 0., 0.)
                if mapper:
                    command = mapper.advance(reader.read())
                    if mapper.state.reset_requested:
                        obs = reset_scene()
                        command = (0., 0., 0.)
                if args.stop_at_goal:
                    # Same left-to-right goal line as the legacy scene, in env-local coordinates.
                    reached |= bool((robot.data.root_pos_w.torch[:, 1] - base.scene.env_origins[:, 1]
                                     >= layout.goal_position[1]).any())
                    if reached:
                        command = (0., 0., 0.)
                term.set_desired_velocity(command[0], yaw_rate=command[2])
                actions = policy(obs["policy"])
                if tuple(actions.shape) != (1, 19):
                    raise ValueError(f"Invalid action shape {actions.shape}")
                finite(obs, actions)
                h1_local_xy = (robot.data.root_pos_w.torch[0, :2]-base.scene.env_origins[0, :2]).tolist()
                if traffic.step(base.step_dt, h1_xy=h1_local_xy, vehicle_y=layout.vehicle_start_position[1]):
                    record_vehicle_pass()
                place_vehicle()
                obs, rewards, done, _ = env.step(actions)
                finite(obs, rewards, robot.data.root_pose_w.torch)
                falls += int(base.reset_terminated.sum())
                if bool(done.any()):
                    # Automatic episode reset already restored H1; synchronize car and controls.
                    reached = False
                    traffic.reset("episode_reset")
                    record_vehicle_pass()
                    place_vehicle()
                    if mapper:
                        mapper.reset()
                    resets += 1
                    # env.step returned observations before the external traffic reset.
                    # Publish the new pass and recompute before any consumer sees it.
                    obs = env.get_observations()
                    finite(obs)
                if h1_camera is not None:
                    h1_camera.update()
                    # Physics did not pump Kit; display the latest pose exactly once.
                    base.sim.render()
                steps += 1
                if observation_writer is not None:
                    observation_writer.writerow([steps, traffic.sim_time_s, resets, len(traffic.events),
                                                 *obs["crossing"][0].cpu().tolist()])
                if steps % 100 == 0:
                    print(f"[Crossing] step={steps} command={term.vel_command_b[0].tolist()} H1={robot.data.root_pos_w.torch[0].tolist()} falls={falls}", flush=True)
                    if observation_stream is not None:
                        observation_stream.flush()
                        print(f"[Crossing obs] {dict(zip(OBSERVATION_NAMES, obs['crossing'][0].tolist()))}", flush=True)
                if not args.headless:
                    time.sleep(max(0., base.step_dt - (time.monotonic() - start)))
        loop_seconds = time.monotonic() - loop_start
        gpu_end = gpu_memory_snapshot() if h1_camera is not None else None
        if args.steps > 0 and steps != args.steps:
            raise RuntimeError("Simulator closed before the requested step count")
        if args.capture_viewport:
            from h1_camera import capture_viewports
            viewports = {"viewport.png": overview}
            if h1_camera is not None:
                viewports["h1_camera.png"] = h1_camera.viewport
            capture_viewports(viewports, output, base.sim, app)
        return {"steps": steps, "finite": True, "falls": falls, "resets_including_initial": resets,
                "start_position": start_position.tolist(), "final_position": robot.data.root_pos_w.torch[0].tolist(),
                "vehicle_x": traffic.x, "vehicle_traffic": traffic.report(), "goal_line_reached": reached,
                "crossing_observations": {"schema": "crossing_observation_schema.json",
                                          "initial": initial_crossing, "final": obs["crossing"].cpu().tolist(),
                                          "recorded": args.record_crossing_observations,
                                          "rows": steps + 1 if args.record_crossing_observations else 0},
                "camera": h1_camera.report() if h1_camera is not None else None,
                "performance": {"loop_seconds": loop_seconds, "control_steps_per_wall_second": steps / max(loop_seconds, 1e-9),
                                "gpu_total_used_mib_start": gpu_start, "gpu_total_used_mib_end": gpu_end,
                                "note": "Control loop throughput, not display FPS. GPU snapshots include other processes, not peak VRAM."},
                "note": "Manual crossing play/smoke, not a crossing safety or vehicle-avoidance validation."}
    finally:
        if observation_stream is not None:
            observation_stream.close()
        try:
            if h1_camera is not None:
                h1_camera.close()
            elif camera_window is not None:
                camera_window.destroy()
        finally:
            (env if env is not None else base).close()


def main(argv=None):
    args = parse_args(argv)
    from h1_finetune.control import STAGES
    from h1_finetune.validation import gate
    args.bundle = args.bundle.resolve()
    manifest = validate_bundle(args.bundle, gate, STAGES)
    low, high = manifest["validated_vx_range_mps"]
    if not low <= args.walk_speed <= high:
        raise ValueError("Walk speed is outside the bundle's validated range")
    output = ROOT / "logs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_crossing_finetuned")
    output.mkdir(parents=True, exist_ok=False)
    provenance = {"args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  "bundle_manifest": manifest, "sources": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"[Output] {output}", flush=True)
    from isaaclab.app import AppLauncher
    # The camera experience enables multi-view rendering; this creates no sensor tensors.
    launcher = AppLauncher(visualizer="none" if args.headless else "kit", device=args.device,
                           enable_cameras=args.h1_camera, kit_args="--vulkan" if args.h1_camera else "")
    try:
        result = run(args, launcher.app, manifest, output)
        (output / "play_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[Result] {result}", flush=True)
    except Exception as error:
        import traceback
        (output / "failure.json").write_text(json.dumps({"error": str(error), "traceback": traceback.format_exc()}, indent=2), encoding="utf-8")
        raise
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
