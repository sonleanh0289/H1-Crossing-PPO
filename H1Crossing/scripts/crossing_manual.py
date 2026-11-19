"""Keyboard-only crossing preview using frozen locomotion; no behavior model."""
import time
import torch
from h1_finetune.control import KeyboardMapper


def run_keyboard(env, app, args, manifest, reader=None):
    if reader is None:
        from h1_finetune.windows_keyboard import WindowsKeyboard
        reader = WindowsKeyboard()
    mapper = KeyboardMapper(manifest["validated_stage"], args.keyboard_speed, args.keyboard_reverse_speed)
    print("[Manual] No crossing checkpoint/actor loaded. Frozen locomotion only.", flush=True)
    print("[Controls] Focus viewport and release keys. W/S=forward/reverse; A/D=turn; "
          "Shift=max speed; Space=stop; Enter=resume; R=reset.", flush=True)
    print("[Manual] Sensors are diagnostic: no goal-heading assist or lost-track stop. "
          "Release keys after automatic episode reset. Collision avoidance is manual.", flush=True)
    counts = {k: 0 for k in ("success", "collision", "fall", "timeout", "out_of_bounds")}
    steps = manual_resets = automatic_resets = 0
    started = time.perf_counter()
    with torch.inference_mode():
        while app.is_running() and (args.steps < 0 or steps < args.steps):
            command = mapper.advance(reader.read())
            if mapper.state.reset_requested:
                env.reset()
                mapper.reset()
                command = (0., 0., 0.)
                manual_resets += 1
                print(f"[Manual] R reset #{manual_resets}: robot, traffic and sensor tracks.", flush=True)
            obs, reward, done, extras = env.step(None, manual_command=[command])
            from h1_finetune.numerics import finite
            finite(obs, reward)
            for name in counts:
                counts[name] += int(extras["events"][name].sum())
            if done.any():
                mapper.reset()
                automatic_resets += int(done.sum())
            steps += 1
    if args.steps > 0 and steps != args.steps:
        raise RuntimeError("Manual play closed before requested steps")
    elapsed = time.perf_counter() - started
    return {"mode": "keyboard", "crossing_checkpoint": None, "steps": steps,
            "finite": True, "events": counts, "manual_resets": manual_resets,
            "automatic_resets": automatic_resets, "wall_seconds": elapsed,
            "sim_seconds_per_env": steps * .1,
            "real_time_factor": steps * .1 / max(elapsed, 1e-9),
            "note": "Manual sensor preview, frozen locomotion; not a crossing policy evaluation"}
