"""Fixed-command, Windows joystick or foreground-only keyboard play on a flat plane."""
from pathlib import Path
import time
import torch
from .checkpoint import load_safe
from .runtime import build, finite, record_run
from .control import JoystickMapper, KeyboardMapper
from .artifacts import write_json


def play(args, app):
    base, env, runner, contract, cfg, agent = build(args)
    data = load_safe(args.checkpoint, args.device)
    if data.get("fine_tune") and data["fine_tune"]["contract"] != contract:
        raise ValueError("Play checkpoint contract mismatch")
    runner.alg.actor.load_state_dict(data["actor_state_dict"], strict=True)
    policy = runner.get_inference_policy(args.device)
    term = base.command_manager.get_term("base_velocity")
    term.manual = True
    base.reset()
    joystick, keyboard, mapper = None, None, None
    if args.joystick:
        from .windows_joystick import WindowsJoystick
        joystick = WindowsJoystick(args.joystick_id)
        mapper = JoystickMapper(args.stage, args.forward_axis, args.turn_axis,
                                args.forward_sign, args.turn_sign)
        print(joystick.describe(), flush=True)
        print("Neutral first. Up=forward, down=reverse. Buttons: 1 resume, 2 stop, 3 reset.", flush=True)
    if getattr(args, "keyboard", False):
        from .windows_keyboard import WindowsKeyboard
        keyboard = WindowsKeyboard()
        mapper = KeyboardMapper(args.stage, args.keyboard_speed, args.keyboard_reverse_speed)
        print("Click the simulator viewport and release all control keys first.", flush=True)
        print("Hold W/S: forward/backward; A/D: left/right yaw; Shift: stage maximum speed.", flush=True)
        print("Release: stop command. Space: latched stop; Enter: resume after key release; R: reset.", flush=True)
        print("Switching to another app sends zero command; release keys before returning to control.", flush=True)
    record_run(Path(args.output), args, args.checkpoint, contract, cfg, agent)
    obs = env.get_observations()
    steps, falls = 0, 0
    try:
        with torch.inference_mode():
            while app.is_running() and (args.steps < 0 or steps < args.steps):
                start = time.monotonic()
                command = (args.vx, 0.0, args.yaw_rate)
                if joystick or keyboard:
                    command = mapper.advance((joystick or keyboard).read())
                    if mapper.state.reset_requested:
                        base.reset()
                        mapper.reset()
                        obs = env.get_observations()
                term.set_desired_velocity(command[0], yaw_rate=command[2])
                action = policy(obs)
                finite(action)
                obs, reward, done, _ = env.step(action)
                finite(obs, reward)
                falls += int(base.reset_terminated.sum())
                if mapper and bool(done.any()):
                    mapper.reset()
                steps += 1
                if not args.headless and steps % 5 == 0:
                    position = base.scene["robot"].data.root_pos_w.torch[0].cpu()
                    base.sim.set_camera_view(eye=(position + torch.tensor([7.0, -7.0, 4.0])).tolist(),
                                             target=position.tolist())
                if not args.headless:
                    time.sleep(max(0.0, base.step_dt - (time.monotonic() - start)))
        if args.steps > 0 and steps != args.steps:
            raise RuntimeError("Play closed before requested steps completed")
        write_json(Path(args.output) / "play_result.json", {"steps": steps, "finite": True, "falls": falls,
                   "note": "Playback smoke is not a speed-envelope validation."})
    finally:
        env.close()
