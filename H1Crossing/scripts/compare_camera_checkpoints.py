"""Paired, deterministic, first-terminal-episode evaluation. No optimizer updates.

One environment avoids asynchronous reset RNG coupling. Each seed is restarted
for each model; reactive traffic can legitimately differ after H1 actions differ.
"""
import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import time
import torch

from behavior_training import (ROOT, load_behavior, ppo_config, compatibility,
                               check_compatibility, write_json)
from behavior_control import load_behavior_config
from camera_perception import load_config
from crossing_bundle import validate_bundle, sha256
from crossing_rewards import load_reward_config
from vehicle_motion import TrafficConfig
from h1_finetune.control import STAGES
from h1_finetune.validation import gate

EVENTS = ("success", "collision", "fall", "timeout", "out_of_bounds")


def outcome(events):
    # Count exactly one outcome per episode; retain overlapping raw flags too.
    for name in ("collision", "fall", "out_of_bounds", "timeout", "success"):
        if events[name]:
            return name
    raise ValueError("Finished episode has no recorded terminal event")


def verify_initial_pair(a, b):
    if a.keys() != b.keys():
        raise AssertionError("Initial-state keys differ")
    for key in a:
        torch.testing.assert_close(torch.tensor(a[key]), torch.tensor(b[key]), rtol=0, atol=1e-6,
                                   msg=f"Paired initial state differs: {key}")


def summarize(rows, seeds):
    models = {}
    for label in ("baseline", "candidate"):
        selected = [r for r in rows if r["model"] == label]
        n = len(selected)
        counts = {name: sum(r["outcome"] == name for r in selected) for name in EVENTS}
        success = [r for r in selected if r["outcome"] == "success"]
        models[label] = {"episodes": n, "outcomes": counts,
            "success_rate": counts["success"]/n if n else None,
            "mean_return": sum(r["return"] for r in selected)/n if n else None,
            "mean_success_time_s": sum(r["duration_s"] for r in success)/len(success) if success else None,
            "raw_events": {k:sum(r["events"][k] for r in selected) for k in EVENTS}}
    paired = []
    for seed in seeds:
        pair = {r["model"]:r for r in rows if r["seed"] == seed}
        if len(pair) == 2:
            a,b = pair["baseline"],pair["candidate"]
            verify_initial_pair(a["initial"],b["initial"])
            paired.append({"seed":seed,"baseline":a["outcome"],"candidate":b["outcome"],
                           "return_delta":b["return"]-a["return"],"initial_state_matched":True})
    return {"models":models,"pairs":paired,"complete":len(paired)==len(seeds),
            "caveats":["Small matched-seed screen, not a safety validation.",
                       "Seed 42 was the training seed; remaining seeds are additional scenarios.",
                       "Same seeded initial traffic and random stream, not identical future car trajectories: yielding reacts to H1.",
                       "Deterministic actor does not guarantee bit-identical GPU physics/rendering."]}


def initial_state(env):
    state = {"h1_root":env.robot.data.root_state_w.torch[0].cpu().tolist(),
             "joint_pos":env.robot.data.joint_pos.torch[0].cpu().tolist(),
             "joint_vel":env.robot.data.joint_vel.torch[0].cpu().tolist(),
             "goal_xy":env.goal_xy[0].cpu().tolist(),
             "command":env.term.vel_command_b[0].cpu().tolist()}
    for name in ("x","speed","target_speed","target_remaining","will_yield"):
        state["car_"+name] = getattr(env.traffic,name)[0].item()
    return state


def evaluate_episode(env, alg, data, seed, app, *, decision_observer=None):
    # Model construction/loading must not consume the episode's random stream.
    alg.load(data, {"actor":True,"critic":True}, strict=True)
    alg.eval_mode()
    env.base.seed(seed)
    env.traffic.generator.manual_seed(seed+1000)
    env.traffic.pass_count.zero_()
    env.physics_steps = 0
    env.sim_time = 0.
    obs = env.reset()
    if obs["behavior"].shape != (1,22) or bool(obs["behavior"][0,19:21].any()):
        raise AssertionError("Each episode must start with a cleared 22D camera track")
    start = initial_state(env)
    total = 0.
    terms = {}
    valid = 0
    wall_start = time.perf_counter()
    with torch.inference_mode():
        for step in range(1, math.ceil(env.settings.episode_seconds/.1)+2):
            if not app.is_running():
                raise RuntimeError("Simulator closed before paired evaluation completed")
            if decision_observer is not None:
                before = obs["behavior"].clone()
                decision_observer(env, before.clone())
                if not torch.equal(before, obs["behavior"]):
                    raise AssertionError("Diagnostic observer modified the policy observation")
            valid += int(obs["behavior"][0,20] > .5)
            actions = alg.actor(obs, stochastic_output=False)
            if not torch.isfinite(actions).all():
                raise AssertionError("Nonfinite deterministic action")
            obs, reward, done, extra = env.step(actions)
            total += float(reward[0])
            for name,value in extra["reward_terms"].items():
                terms[name] = terms.get(name,0.) + float(value[0])
            if bool(done[0]):
                flags = {k:bool(extra["events"][k][0]) for k in EVENTS}
                terminal = extra["terminal_observation"][0]
                return {"seed":seed,"outcome":outcome(flags),"events":flags,
                    "return":total,"reward_terms":terms,"behavior_steps":step,
                    "duration_s":float(extra["terminal_elapsed"][0]),
                    "final_goal_distance_m":float(torch.linalg.vector_norm(terminal[2:4]*10)),
                    "velocity_valid_fraction_at_decisions":valid/step,
                    "wall_seconds":time.perf_counter()-wall_start,"initial":start}
    raise RuntimeError("Episode exceeded configured horizon without termination")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",type=Path,required=True)
    parser.add_argument("--candidate",type=Path,required=True)
    parser.add_argument("--seeds",type=int,nargs="+",default=[42,43,44,45,46])
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--bundle",type=Path,default=ROOT/"checkpoints/h1_flat_stability_v2_0500")
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s > 2**31-1001 for s in args.seeds):
        parser.error("Seeds must be unique nonnegative 31-bit values")
    paths = {"baseline":args.baseline.resolve(),"candidate":args.candidate.resolve()}
    data = {label:load_behavior(path) for label,path in paths.items()}
    saved = data["baseline"]["compatibility"]
    if saved["format"] != "h1-crossing-camera-v1":
        raise ValueError("Compare the converted 22D camera checkpoint, not the 19D state checkpoint")
    check_compatibility(saved,data["candidate"]["compatibility"],training=True)
    settings = load_behavior_config(ROOT/"configs/crossing_behavior_camera.yaml")
    rewards = load_reward_config()
    perception = load_config()
    traffic = TrafficConfig()
    manifest = validate_bundle(args.bundle,gate,STAGES)
    config = ppo_config(args.bundle,settings)
    current = compatibility(settings,rewards,manifest,config,traffic,perception)
    check_compatibility(saved,current,training=False)
    output = (args.output or ROOT/"logs"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")+"_camera_paired_eval")).resolve()
    if not output.is_relative_to(ROOT/"logs"):
        raise ValueError("Outputs must be inside H1Crossing/logs")
    output.mkdir(parents=True,exist_ok=False)
    hashes = {label:sha256(path) for label,path in paths.items()}
    write_json(output/"protocol.json",{"seeds":args.seeds,"episodes_per_seed_per_model":1,
        "num_envs":1,"actor_deterministic":True,"headless":True,
        "checkpoints":{k:str(v) for k,v in paths.items()},"checkpoint_sha256":hashes,
        "iterations":{k:v["completed_iterations"] for k,v in data.items()},
        "contract":current,"evaluation_source_sha256":sha256(Path(__file__)),
        "sources":{p.name:sha256(p) for p in (ROOT/"scripts").glob("*.py")},
        "torch_version":str(torch.__version__)})
    print(f"[Output] {output}",flush=True)
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(visualizer="none",device=args.device,enable_cameras=True,kit_args="--vulkan")
    env = None
    code = 0
    rows = []
    try:
        from behavior_environment import CrossingBehaviorEnv
        from rsl_rl.algorithms import PPO
        from h1_camera import gpu_memory_snapshot
        env = CrossingBehaviorEnv(1,args.device,args.seeds[0],settings,rewards,args.bundle,manifest,True,
                                  traffic_config=traffic,perception_config=perception)
        alg = PPO.construct_algorithm(env.get_observations(),env,deepcopy(config),args.device)
        frozen = {k:v.clone() for k,v in env.locomotion.state_dict().items()}
        for index,seed in enumerate(args.seeds):
            # Alternate model order to reduce a systematic warm-cache/order effect.
            labels = ("baseline","candidate") if index%2 == 0 else ("candidate","baseline")
            for label in labels:
                row = evaluate_episode(env,alg,data[label],seed,launcher.app)
                row["model"] = label
                rows.append(row)
                write_json(output/"episodes.json",rows)
                write_json(output/"comparison.json",summarize(rows,args.seeds))
                print(f"[paired] {label} seed={seed}: {row['outcome']}, return={row['return']:.3f}, "
                      f"duration={row['duration_s']:.2f}s, wall={row['wall_seconds']:.1f}s",flush=True)
        if not all(torch.equal(v,env.locomotion.state_dict()[k]) for k,v in frozen.items()):
            raise AssertionError("Frozen locomotion changed")
        if any(sha256(paths[k]) != v for k,v in hashes.items()):
            raise AssertionError("Source checkpoint changed")
        with (output/"episodes.csv").open("w",encoding="utf-8",newline="") as handle:
            fields=["model","seed","outcome","return","duration_s","final_goal_distance_m",
                    "velocity_valid_fraction_at_decisions","wall_seconds"]
            writer=csv.DictWriter(handle,fieldnames=fields,extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        write_json(output/"gpu_memory_after.json",gpu_memory_snapshot())
        write_json(output/"verification.json",{"finite":True,"locomotion_unchanged":True,
            "source_checkpoints_unchanged":True,"all_initial_pairs_matched":True,"episodes":len(rows)})
    except BaseException as error:
        import traceback
        code=1
        write_json(output/"failure.json",{"error":str(error),"traceback":traceback.format_exc()})
        raise
    finally:
        if env is not None:
            env.close()
        launcher.app.close(exit_code=code)


if __name__ == "__main__":
    main()
