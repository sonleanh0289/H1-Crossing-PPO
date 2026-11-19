"""Bounded multi-env real-sensor test; constant-speed fixtures, no training.

Truth/fixture velocities are used only for assertions after sensor processing.
No crossing checkpoint is loaded. Partial reset and camera dropout affect env 0.
"""
import argparse
from datetime import datetime, timezone
import time
import torch
import numpy as np
from behavior_training import ROOT, write_json
from behavior_control import load_behavior_config
from camera_perception import load_config
from crossing_rewards import load_reward_config
from crossing_bundle import validate_bundle, sha256
from radar_processing import load_radar_config
from vehicle_motion import TrafficConfig
from h1_finetune.control import STAGES
from h1_finetune.validation import gate


def clock_states(sensor):
    """Inspect each native head, not a synthetic shared clock for a rig."""
    return [head.clock.last for head in getattr(sensor, "sensors", [sensor])]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs",type=int,default=8)
    parser.add_argument("--steps",type=int,default=30)
    parser.add_argument("--radar_config",type=str,default=str(ROOT/"configs/crossing_radar.yaml"))
    parser.add_argument("--perception_config",type=str,default=str(ROOT/"configs/crossing_perception_multicolor.yaml"))
    args=parser.parse_args()
    if not 2<=args.num_envs<=8 or args.steps<30:
        parser.error("Use 2..8 envs and at least 30 behavior steps")
    count=args.num_envs
    output=ROOT/"logs"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")+"_radar_batch_smoke")
    output.mkdir()
    print(f"[Output] {output}",flush=True)
    bundle=ROOT/"checkpoints/h1_flat_stability_v2_0500"
    manifest=validate_bundle(bundle,gate,STAGES)
    frozen_hash=sha256(bundle/"policy.pt")
    radar=load_radar_config(args.radar_config)
    from isaaclab.app import AppLauncher
    from h1_radar import radar_experience, MOTION_BVH_ARGS, prepare_radar_clock
    from h1_camera import gpu_memory_snapshot
    launcher=AppLauncher(visualizer="none",device="cuda:0",enable_cameras=True,
        experience=radar_experience(),kit_args="--vulkan "+MOTION_BVH_ARGS)
    env=None; code=0
    try:
        import isaaclab.sim
        prepare_radar_clock(radar.sample_hz)
        from behavior_environment import CrossingBehaviorEnv
        env=CrossingBehaviorEnv(count,"cuda:0",42,load_behavior_config(ROOT/"configs/crossing_behavior_camera_radar.yaml"),
            load_reward_config(),bundle,manifest,True,
            traffic_config=TrafficConfig(mode="constant",yield_probability=0,vehicle_palette="multicolor"),
            perception_config=load_config(args.perception_config),
            radar_config=radar,radar_output=output,radar_observations=True)
        assert len(env.radar.sensors)==count
        fixture_speed=torch.arange(count,device="cuda:0",dtype=torch.float32)+2.
        env.traffic.x[:]=-20.-torch.arange(count,device="cuda:0")
        env.traffic.speed.copy_(fixture_speed)
        env.traffic.target_speed.copy_(fixture_speed)
        env.place_vehicle()
        capture=env.perception.capture
        def dropout_capture(now):
            capture(now)
            if .7<=now<=1.4:
                stamp,frames=env.perception.frames[0]
                env.perception.frames[0]=(stamp,[(*f[:-1],None) for f in frames])
                env.perception.tracks[0].reset()
        env.perception.capture=dropout_capture
        errors=[[] for _ in range(count)]; validity=[0]*count
        radar_only=0; partial_reset_passed=False; rows=[]; memory=[]
        start=time.perf_counter()
        with torch.inference_mode():
            for step in range(args.steps):
                if not launcher.app.is_running(): raise RuntimeError("App closed early")
                if step==15:
                    before=[dict(row) if row else None for row in env.radar.latest]
                    clocks=[clock_states(sensor) for sensor in env.radar.sensors]
                    camera_states=[(track.time,track.last_measurement,track.x.copy()) for track in env.perception.tracks]
                    env.reset(torch.tensor([0],device="cuda:0"))
                    assert env.radar.latest[0] is None
                    assert env.radar.latest[1:]==before[1:]
                    assert [clock_states(s) for s in env.radar.sensors[1:]]==clocks[1:]
                    for i,track in enumerate(env.perception.tracks[1:],1):
                        assert (track.time,track.last_measurement)==camera_states[i][:2]
                        np.testing.assert_array_equal(track.x,camera_states[i][2])
                    assert env.get_observations()["behavior"][0,20]==0
                    partial_reset_passed=True
                obs,reward,done,_=env.step(None,manual_command=torch.zeros(count,3,device="cuda:0"))
                values=obs["behavior"]
                assert values.shape==(count,23) and torch.isfinite(values).all() and torch.isfinite(reward).all()
                assert not done.any(), "Unexpected episode termination in standing fixture"
                row=values.cpu().tolist()
                rows.append(row)
                for i in range(count):
                    # Test native radar mapping, not the camera fallback's accuracy.
                    native = env.radar.latest[i]
                    if (native and native.get("lane_speed_valid")
                            and 0 <= env.sim_time-native["sensor_clock_reference_sim_s"] <= .12+1e-8):
                        assert row[i][20] == 1
                        validity[i]+=1
                        if i!=0 or step<15:
                            errors[i].append(abs(row[i][4]*15.-float(fixture_speed[i])))
                if .7<=env.sim_time<=1.4 and row[0][19]==0 and row[0][20]==1:
                    radar_only+=1
                    assert row[0][7]==0
                if step%10==9:
                    memory.append(gpu_memory_snapshot())
                    print(f"[batch smoke] {step+1}/{args.steps}, valid={validity}, VRAM={memory[-1]}",flush=True)
        assert all(validity) and all(errors), "Every environment must acquire its own radar velocity"
        medians=[float(np.median(e)) for e in errors]
        assert all(e<.25 for e in medians), f"Radar mapping/speed mismatch: {medians}"
        assert radar_only>0 and partial_reset_passed
        assert sha256(bundle/"policy.pt")==frozen_hash
        write_json(output/"observations.json",rows)
        write_json(output/"result.json",dict(passed=True,num_envs=count,observation_shape=[count,23],
            steps=args.steps,valid_radar_velocity_steps=validity,median_abs_speed_error_mps=medians,
            perception_config=str(args.perception_config),
            env0_radar_only_steps=radar_only,partial_reset_isolated=True,
            locomotion_unchanged=True,training_started=False,wall_seconds=time.perf_counter()-start,
            gpu_memory_mib=memory,radar=env.radar.report()))
        print("[PASS] multi-env radar + camera observation, partial reset, velocity mapping",flush=True)
    except BaseException:
        code=1
        import traceback
        write_json(output/"failure.json",{"traceback":traceback.format_exc()})
        raise
    finally:
        if env is not None: env.close()
        launcher.app.close(exit_code=code)


if __name__=="__main__":
    main()
