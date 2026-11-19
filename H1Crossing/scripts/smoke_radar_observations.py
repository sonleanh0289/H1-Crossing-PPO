"""Bounded real-sensor 23D/dropout smoke. No crossing policy or training.

Force image detections/tracks unavailable during seconds 2-8 while preserving
real RTX radar buffers. The observer never reads vehicle simulator state.
"""
import argparse
from datetime import datetime, timezone
import json
import time
import torch
from behavior_training import ROOT, write_json
from behavior_control import load_behavior_config
from camera_perception import load_config
from crossing_rewards import load_reward_config
from crossing_bundle import validate_bundle, sha256
from radar_processing import load_radar_config
from vehicle_motion import TrafficConfig
from h1_finetune.control import STAGES
from h1_finetune.validation import gate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui",action="store_true")
    args=parser.parse_args()
    output=ROOT/"logs"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")+"_radar_observation_smoke")
    output.mkdir()
    print(f"[Output] {output}",flush=True)
    bundle=ROOT/"checkpoints/h1_flat_stability_v2_0500"
    manifest=validate_bundle(bundle,gate,STAGES)
    frozen_hash=sha256(bundle/"policy.pt")
    radar=load_radar_config()
    from isaaclab.app import AppLauncher
    from h1_radar import radar_experience, MOTION_BVH_ARGS, prepare_radar_clock
    launcher=AppLauncher(visualizer="kit" if args.gui else "none",device="cuda:0",enable_cameras=True,
        experience=radar_experience(),kit_args="--vulkan "+MOTION_BVH_ARGS)
    env=None; code=0
    try:
        import isaaclab.sim
        prepare_radar_clock(radar.sample_hz)
        from behavior_environment import CrossingBehaviorEnv
        env=CrossingBehaviorEnv(1,"cuda:0",42,load_behavior_config(ROOT/"configs/crossing_behavior_camera.yaml"),
            load_reward_config(),bundle,manifest,not args.gui,
            traffic_config=TrafficConfig(vehicle_palette="multicolor"),
            perception_config=load_config(ROOT/"configs/crossing_perception_multicolor.yaml"),
            radar_config=radar,radar_output=output,radar_observations=True)
        capture=env.perception.capture
        def dropout_capture(now):
            capture(now)
            if 2. <= now <= 8.:
                stamp,frames=env.perception.frames[0]
                env.perception.frames[0]=(stamp,[(*f[:-1],None) for f in frames])
                for track in env.perception.tracks:
                    track.reset()
        env.perception.capture=dropout_capture
        rows=[]; start=time.perf_counter()
        with torch.inference_mode():
            for step in range(100):
                if not launcher.app.is_running():
                    raise RuntimeError("App closed early")
                obs,_,_,_=env.step(None,manual_command=[(0. if step<30 else .5,0.,0.)])
                value=obs["behavior"]
                assert value.shape==(1,23) and torch.isfinite(value).all()
                row=dict(time_s=env.sim_time,observation=value[0].cpu().tolist(),
                    radar_valid=bool(env.radar.latest and env.radar.latest["lane_speed_valid"]),
                    association=env.radar.latest.get("association_source") if env.radar.latest else None)
                rows.append(row)
                if step%25==24:
                    print(f"[smoke] {step+1}/100 steps",flush=True)
        velocity_only=[r for r in rows if 2<=r["time_s"]<=8 and r["observation"][19]==0 and r["observation"][20]==1]
        write_json(output/"observations.json",rows)
        assert velocity_only, "No valid radar velocity observed during forced camera dropout"
        assert all(r["observation"][7]==0 for r in velocity_only), "TTC valid without position"
        assert any(abs(r["observation"][4])>.01 and r["observation"][22]!=0 for r in velocity_only)
        env.reset()
        assert env.get_observations()["behavior"][0,20]==0, "Reset retained velocity"
        assert sha256(bundle/"policy.pt")==frozen_hash
        result=dict(steps=100,observation_dim=23,forced_camera_dropout_s=[2,8],
            velocity_only_steps=len(velocity_only),finite=True,reset_clears_velocity=True,
            locomotion_unchanged=True,training_started=False,wall_seconds=time.perf_counter()-start,
            radar=env.radar.report())
        write_json(output/"result.json",result)
        print("[PASS] "+json.dumps(result),flush=True)
    except BaseException:
        code=1
        import traceback
        write_json(output/"failure.json",{"traceback":traceback.format_exc()})
        raise
    finally:
        if env is not None:
            env.close()
        launcher.app.close(exit_code=code)


if __name__=="__main__":
    main()
