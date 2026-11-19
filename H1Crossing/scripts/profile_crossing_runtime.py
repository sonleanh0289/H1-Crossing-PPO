"""CPU wall-time attribution for a bounded camera play; no policy/config changes.

Uses measure_camera_fps flags. Profiles only env.step after 2 simulated seconds.
CUDA waits are charged to the calling CPU operation, not isolated GPU kernels.
"""
import cProfile
import io
import pstats
import sys
import measure_camera_fps as probe

original_run = probe.original_run


def profiled_run(args, mode, app, output, settings, reward_cfg, manifest, loaded):
    from behavior_environment import CrossingBehaviorEnv
    profile = cProfile.Profile()
    saved_step, saved_close = CrossingBehaviorEnv.step, CrossingBehaviorEnv.close
    profiled_steps = 0

    def step(env, actions):
        nonlocal profiled_steps
        if env.sim_time < 2.-1e-8:
            return saved_step(env, actions)
        profiled_steps += 1
        profile.enable()
        try:
            return saved_step(env, actions)
        finally:
            profile.disable()

    def close(env):
        try:
            profile.dump_stats(str(output / "runtime_profile.pstats"))
            stream = io.StringIO()
            stats = pstats.Stats(profile,stream=stream)
            stats.sort_stats("cumulative").print_stats(45)
            stats.sort_stats("tottime").print_stats(35)
            (output / "runtime_profile.txt").write_text(stream.getvalue(),encoding="utf-8")
            rows = []
            for (filename,line,function),(primitive,calls,self_s,cumulative_s,callers) in stats.stats.items():
                rows.append({"file":filename,"line":line,"function":function,"calls":calls,
                             "self_seconds":self_s,"inclusive_seconds":cumulative_s})
            rows.sort(key=lambda row:row["inclusive_seconds"],reverse=True)
            probe.playback.write_json(output / "runtime_profile.json",{
                "profiled_behavior_steps":profiled_steps,"profiled_sim_seconds":profiled_steps*.1,
                "profiled_wall_seconds":stats.total_tt,"top_functions":rows[:100],
                "warning":"Inclusive rows overlap; do not sum parent and child times. CPU wall profile includes GPU synchronization waits and profiling overhead."})
        finally:
            saved_close(env)

    CrossingBehaviorEnv.step, CrossingBehaviorEnv.close = step, close
    try:
        return original_run(args,mode,app,output,settings,reward_cfg,manifest,loaded)
    finally:
        CrossingBehaviorEnv.step, CrossingBehaviorEnv.close = saved_step, saved_close


if __name__ == "__main__":
    probe.original_run = profiled_run
    probe.playback.run = probe.measured_run
    probe.playback.main("play",sys.argv[1:])
