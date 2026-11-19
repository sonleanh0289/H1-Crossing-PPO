"""Explicit rendered perception TEST; ground truth stays outside policy/perception.

Runs bounded probes, not training. Writes aggregate errors/misses and four
diagnostic contact sheets (not an image stream). Missing tracks are counted.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import math
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "H1FlatFineTune"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "logs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")+"_camera_benchmark")
    output = output.resolve()
    if not output.is_relative_to(ROOT / "logs"):
        raise ValueError("Benchmark outputs must be inside H1Crossing/logs")
    output.mkdir(parents=True, exist_ok=False)
    print(f"[Output] {output}", flush=True)
    from camera_perception import load_config
    from crossing_bundle import validate_bundle
    from crossing_rewards import load_reward_config
    from behavior_control import load_behavior_config
    from h1_finetune.validation import gate
    from h1_finetune.control import STAGES
    from h1_camera import gpu_memory_snapshot
    bundle = ROOT / "checkpoints/h1_flat_stability_v2_0500"
    manifest = validate_bundle(bundle, gate, STAGES)
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(visualizer="none" if args.headless else "kit", device="cuda:0",
                          enable_cameras=True, kit_args="--vulkan")
    from behavior_environment import CrossingBehaviorEnv
    env = None
    exit_code = 0
    rows = []
    try:
        env = CrossingBehaviorEnv(1, "cuda:0", 42, load_behavior_config(), load_reward_config(),
                                  bundle, manifest, args.headless, perception_config=load_config())
        scenarios = [(f"static_{d}m", float(d), "stand", "stop", 25) for d in (5,15,30,50)]
        scenarios += [(f"{body}_{motion}", 50., body, motion, 60)
                      for body, motion in (("stand","constant"), ("walk","accelerate"),
                          ("turn","brake"), ("stand","yield_stop"), ("stand","respawn"))]
        with torch.inference_mode():
            for name, distance, body, motion, exposures in scenarios:
                env.reset()
                start = env.sim_time
                x = -distance
                speeds = {"stop":0., "constant":8., "accelerate":6., "brake":10., "yield_stop":4., "respawn":8.}
                speed = speeds[motion]
                errors, velocity_errors, misses, velocity_missing = [], [], 0, 0
                detection_count, first_velocity = 0, None
                screenshots = False
                for frame in range(exposures):
                    substep = 0
                    while True:
                        if env.physics_steps % 4 == 0:
                            command = env.term.target.new_tensor([[.8 if body == "walk" else 0., 0., .5 if body == "turn" else 0.]])
                            env.term.target.copy_(command)
                            env.base.command_manager.compute(dt=.02)
                            action = env.locomotion(env.base.observation_manager.compute_group("policy"))
                            env.base.action_manager.process_action(action)
                        previous_speed = speed
                        if motion == "accelerate":
                            speed += .005*2
                        elif motion in ("brake", "yield_stop"):
                            speed = max(0., speed-.005*3)
                        x += .005*(previous_speed+speed)/2
                        if motion == "respawn" and frame == 30 and substep == 0:
                            x = -50.
                        env.traffic.x[:] = x
                        env.traffic.speed[:] = speed
                        env.place_vehicle()
                        env.base._sim_step_counter += 1
                        env.base.action_manager.apply_action()
                        env.base.scene.write_data_to_sim()
                        env.base.sim.step(render=False)
                        env.base.scene.update(.005)
                        env.physics_steps += 1
                        env.sim_time = env.physics_steps*.005
                        substep += 1
                        if env.perception.due(env.physics_steps, .005):
                            break
                    env.perception.follow()
                    env.base.sim.render()
                    env.perception.capture(env.sim_time)
                    estimate, valid, vel_valid, _ = env.perception.tracks[0].estimate(env.sim_time)
                    # GT is used only HERE to score observations, never to create them.
                    truth = env.vehicle.data.root_pos_w.torch[0,:2].cpu().numpy()
                    if valid:
                        errors.append(float(np.linalg.norm(estimate[:2]-truth)))
                    else:
                        misses += 1
                    if vel_valid:
                        velocity_errors.append(float(np.linalg.norm(estimate[2:]-[speed,0])))
                        if first_velocity is None:
                            first_velocity = env.sim_time-start
                    else:
                        velocity_missing += 1
                    frames = env.perception.frames[0][1]
                    detection_count += any(view[-1] is not None for view in frames)
                    if name.startswith("static_") and frame == 15 and not screenshots:
                        import cv2
                        images = []
                        for title, view in zip(env.perception.camera_names, frames):
                            image = cv2.cvtColor(view[0][..., :3], cv2.COLOR_RGB2BGR)
                            if view[-1] is not None:
                                bx, by, bw, bh = view[-1].bbox
                                cv2.rectangle(image,(bx,by),(bx+bw,by+bh),(0,0,255),1)
                            cv2.putText(image, f"{name} {title}", (4,18), 0,.5,(255,255,255),1)
                            images.append(image)
                        cv2.imwrite(str(output / (name+".png")), np.concatenate(images,axis=1))
                        screenshots = True
                row = {"scenario":name, "frames":exposures, "detected_frames":detection_count,
                       "position_invalid_frames":misses, "velocity_invalid_frames":velocity_missing,
                       "position_mae_m":float(np.mean(errors)) if errors else None,
                       "velocity_mae_mps":float(np.mean(velocity_errors)) if velocity_errors else None,
                       "first_velocity_latency_s":first_velocity,
                       "maximum_position_error_m":max(errors,default=None)}
                limit = 1. if distance <= 30 else 2.
                row["conditional_error_target_met"] = bool(errors and velocity_errors and
                    row["position_mae_m"] <= limit and row["velocity_mae_mps"] <= limit)
                rows.append(row)
                print(f"[perception benchmark] {row}", flush=True)
        report = {"scenarios":rows, "gpu_memory_mib":gpu_memory_snapshot(),
                  "note":"Conditional errors exclude invalid estimates; invalid/detection counts are always reported. These bounded probes do not certify crossing safety."}
        (output / "benchmark.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    except Exception:
        exit_code = 1
        import traceback
        (output / "failure.txt").write_text(traceback.format_exc(),encoding="utf-8")
        raise
    finally:
        if env is not None:
            env.close()
        launcher.app.close(exit_code=exit_code)


if __name__ == "__main__":
    main()
