"""Explicit nested wall timers; unaffected by Kit changing Python profiler hooks.

Diagnostic only: use the same bounded camera-play arguments. No timing changes,
no GPU synchronization inserted, no behavior/config/checkpoint modifications.
"""
from collections import defaultdict
import sys
import time
import measure_camera_fps as probe

original_run = probe.original_run


def timed_run(args, mode, app, output, settings, reward_cfg, manifest, loaded):
    from behavior_environment import CrossingBehaviorEnv
    saved_step, saved_close = CrossingBehaviorEnv.step, CrossingBehaviorEnv.close
    inclusive, exclusive, counts = defaultdict(float), defaultdict(float), defaultdict(int)
    stack, restored = [], []
    state = {"active":False,"installed":False,"steps":0,"wall":0.}

    def wrap(obj, attr, name):
        previous = getattr(obj,attr)
        def measured(*a,**kw):
            if not state["active"]:
                return previous(*a,**kw)
            frame = [0.]
            stack.append(frame)
            start = time.perf_counter()
            try:
                return previous(*a,**kw)
            finally:
                elapsed = time.perf_counter()-start
                stack.pop()
                inclusive[name] += elapsed
                exclusive[name] += elapsed-frame[0]
                counts[name] += 1
                if stack:
                    stack[-1][0] += elapsed
        setattr(obj,attr,measured)
        restored.append((obj,attr,previous))

    def step(env, actions):
        if not state["installed"]:
            for obj,attr,name in (
                (env.base.sim,"step","physics_step"), (env.base.sim,"render","render_and_GUI"),
                (env.base.scene,"write_data_to_sim","scene_write_actuators"),
                (env.base.scene,"update","scene_update"), (env,"place_vehicle","place_vehicle"),
                (env.traffic,"advance","traffic_advance"), (env.perception,"capture","camera_capture_perception_debug"),
                (env.perception,"follow","camera_pose_follow"),
                (env.base.action_manager,"apply_action","apply_action"),
                (env.base.command_manager,"compute","command_compute"),
                (env.base.observation_manager,"compute_group","observation_group"),
                (env,"get_observations","behavior_observation"), (env.tracker,"step","reward_step")):
                wrap(obj,attr,name)
            state["installed"] = True
        state["active"] = env.sim_time >= 2.-1e-8
        start = time.perf_counter()
        try:
            return saved_step(env,actions)
        finally:
            if state["active"]:
                state["wall"] += time.perf_counter()-start
                state["steps"] += 1
            state["active"] = False

    def close(env):
        try:
            rows = [{"stage":name,"calls":counts[name],"exclusive_seconds":exclusive[name],
                     "inclusive_seconds":inclusive[name],"share_percent":100*exclusive[name]/state["wall"]}
                    for name in exclusive] if state["wall"] else []
            other = state["wall"]-sum(exclusive.values())
            rows.append({"stage":"remaining_loop_and_checks","exclusive_seconds":other,
                         "share_percent":100*other/state["wall"] if state["wall"] else None})
            rows.sort(key=lambda row:row["exclusive_seconds"],reverse=True)
            report = {"profiled_behavior_steps":state["steps"],"sim_seconds":state["steps"]*.1,
                      "wall_seconds":state["wall"],"stages":rows,
                      "notes":"First 2 simulated seconds excluded. Exclusive rows sum to measured loop wall time. CPU timers include GPU waits at call sites; not pure GPU kernel timings."}
            probe.playback.write_json(output / "stage_timing.json",report)
            print("[Stage timing] "+str(report),flush=True)
        finally:
            saved_close(env)

    CrossingBehaviorEnv.step, CrossingBehaviorEnv.close = step, close
    try:
        return original_run(args,mode,app,output,settings,reward_cfg,manifest,loaded)
    finally:
        for obj,attr,previous in reversed(restored):
            setattr(obj,attr,previous)
        CrossingBehaviorEnv.step, CrossingBehaviorEnv.close = saved_step, saved_close


if __name__ == "__main__":
    probe.original_run = timed_run
    probe.playback.run = probe.measured_run
    probe.playback.main("play",sys.argv[1:])
