"""Bounded diagnostic wrapper for normal camera play; never trains or changes policy.

Pass the same flags as play_crossing_behavior.py. Writes timing/robot-motion
samples, NOT images. Hooks are scoped to this process and restored afterwards.
"""
import json
import sys
import time
import numpy as np
import behavior_training as playback


def summarize(samples, loop_wall_seconds, counts):
    groups = {name: [] for name in ("all", "steady_after_2_sim_seconds", "moving", "moving_on_road", "stationary")}
    for previous, current in zip(samples, samples[1:]):
        wall_dt = current["wall_seconds"]-previous["wall_seconds"]
        sim_dt = current["sim_seconds"]-previous["sim_seconds"]
        if wall_dt <= 0 or sim_dt <= 0:
            raise ValueError("Non-increasing exposure timestamps")
        pair = (wall_dt, sim_dt)
        groups["all"].append(pair)
        if previous["sim_seconds"] < 2.:
            continue
        groups["steady_after_2_sim_seconds"].append(pair)
        if previous["episode"] != current["episode"]:
            continue  # A reset/teleport is not H1 locomotion.
        moving = min(previous["h1_speed_mps"],current["h1_speed_mps"]) > .2
        if moving:
            groups["moving"].append(pair)
            if previous["on_road"] and current["on_road"]:
                groups["moving_on_road"].append(pair)
        if max(previous["h1_speed_mps"],current["h1_speed_mps"]) < .1:
            groups["stationary"].append(pair)
    result = {}
    for name, values in groups.items():
        if not values:
            result[name] = {"intervals":0, "wall_fps_per_camera":None}
            continue
        wall, sim = np.array(values).T
        result[name] = {"intervals":len(values), "wall_seconds":float(wall.sum()),
            "sim_seconds":float(sim.sum()), "wall_fps_per_camera":float(len(values)/wall.sum()),
            "sim_fps_per_camera":float(len(values)/sim.sum()),
            "real_time_factor":float(sim.sum()/wall.sum()),
            "wall_frame_interval_median_ms":float(np.median(wall)*1000),
            "wall_frame_interval_p95_ms":float(np.percentile(wall,95)*1000),
            "sim_interval_min_ms":float(sim.min()*1000), "sim_interval_max_ms":float(sim.max()*1000)}
    episodes = []
    for index in sorted(set(s["episode"] for s in samples)):
        part = [s for s in samples if s["episode"] == index]
        episodes.append({"episode":index, "samples":len(part),
            "min_h1_y_m":min(s["h1_y_m"] for s in part),
            "max_h1_y_m":max(s["h1_y_m"] for s in part),
            "max_h1_speed_mps":max(s["h1_speed_mps"] for s in part)})
    return {"definition":"Each capture delivers one synchronized RGB-D set from all configured cameras; set rate equals EACH camera FPS, not the sum.",
        "loop_wall_seconds":loop_wall_seconds, "captures":len(samples), "events":counts,
        "groups":result, "episodes":episodes,
        "capture_and_perception_ms_mean":float(np.mean([s["capture_ms"] for s in samples])) if samples else None,
        "notes":["perf_counter wall timing includes rendering, CPU transfer, perception and debug UI work; excludes app/environment initialization.",
                 "Moving threshold is actual H1 planar speed >0.2 m/s; on-road uses physical road Y edges.",
                 "Steady phase excludes the first 2 simulated seconds. These are delivered sensor frames, NOT main viewport FPS.",
                 "Diagnostic pose reads add small synchronization overhead; no image stream is recorded."]}


def measured_run(args, mode, app, output, settings, reward_cfg, manifest, loaded):
    if mode != "play" or args.observation_source != "camera" or args.num_envs != 1 or args.steps < 1:
        raise ValueError("FPS probe requires camera play, one env and a positive bounded --steps")
    from behavior_environment import CrossingBehaviorEnv
    from h1_rgbd import RGBDPerception
    from h1_camera import gpu_memory_snapshot
    import torch
    samples, counts = [], {k:0 for k in ("success","collision","fall","timeout","out_of_bounds")}
    context = {"env":None,"loop_start":None,"loop_end":None,"episode":-1}
    old_capture, old_step, old_reset, old_close = (RGBDPerception.capture,
        CrossingBehaviorEnv.step, CrossingBehaviorEnv.reset, CrossingBehaviorEnv.close)

    def reset(env, ids=None):
        context["episode"] += 1
        return old_reset(env, ids)

    def capture(rig, now):
        start = time.perf_counter()
        old_capture(rig, now)
        end = time.perf_counter()
        env = context["env"]
        position = (rig.robot.data.root_pos_w.torch[0,:2]-rig.origins[0,:2]).cpu().tolist()
        speed = float(torch.linalg.vector_norm(rig.robot.data.root_lin_vel_w.torch[0,:2]))
        samples.append({"wall_seconds":end-context["loop_start"], "sim_seconds":now,
            "episode":context["episode"], "h1_x_m":position[0], "h1_y_m":position[1],
            "h1_speed_mps":speed, "on_road":abs(position[1]) <= env.layout.road_width/2,
            "capture_ms":(end-start)*1000})

    def step(env, actions):
        context["env"] = env
        if context["loop_start"] is None:
            context["loop_start"] = time.perf_counter()
        result = old_step(env,actions)
        context["loop_end"] = time.perf_counter()
        for key in counts:
            counts[key] += int(result[3]["events"][key].sum())
        return result

    def close(env):
        try:
            if samples:
                report = summarize(samples,context["loop_end"]-context["loop_start"],counts)
                report.update(headless=args.headless, perception_debug=args.perception_debug,
                    configured_camera_hz=rig_frequency(env), gpu_memory_mib=gpu_memory_snapshot())
                playback.write_json(output / "fps_report.json",report)
                playback.write_json(output / "fps_samples.json",samples)
                print("[FPS measured] "+json.dumps(report,allow_nan=False),flush=True)
        finally:
            old_close(env)

    RGBDPerception.capture, CrossingBehaviorEnv.step = capture, step
    CrossingBehaviorEnv.reset, CrossingBehaviorEnv.close = reset, close
    try:
        return original_run(args,mode,app,output,settings,reward_cfg,manifest,loaded)
    finally:
        RGBDPerception.capture, CrossingBehaviorEnv.step = old_capture, old_step
        CrossingBehaviorEnv.reset, CrossingBehaviorEnv.close = old_reset, old_close


def rig_frequency(env):
    return env.perception.cfg.frequency_hz


original_run = playback.run

if __name__ == "__main__":
    playback.run = measured_run
    playback.main("play",sys.argv[1:])
