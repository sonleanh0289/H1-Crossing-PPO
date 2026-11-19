"""PPO worker; launched explicitly by train.py, never at import time."""
from pathlib import Path
import random
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from .checkpoint import load_safe, initialize, restore, save, resume_command_range
from .control import STAGES, is_next_stage
from .runtime import build, record_run, finite, plain
from .artifacts import write_json, sha256, source_hashes
from .validation import gate


def save_snapshot(alg, destination, state):
    """Save resume state without resetting the live environment or optimizer."""
    save(alg, destination, state)
    saved = load_safe(destination, "cpu")
    saved.update(torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                 cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    torch.save(saved, destination)


def run(args, app):
    from h1_finetune.training_profiles import resolve_training_profile, profile_settings
    base, env, runner, contract, cfg, agent = build(args, training=True)
    directory = Path(args.output)
    data = load_safe(args.resume or args.init_checkpoint, args.device)
    alg = runner.alg
    training_profile = resolve_training_profile(
        getattr(args, "training_profile", None), data.get("fine_tune") if args.resume else None)
    objective = profile_settings(training_profile)
    if objective and data.get("fine_tune", {}).get("contract") != contract:
        raise ValueError("Stability refinement requires a matching fine-tune checkpoint contract")
    training_settings = {"rollout_steps": agent.num_steps_per_env, "ppo": plain(agent.algorithm.to_dict())}
    if args.resume:
        state = restore(alg, data, contract)
        if state.get("training_settings") != training_settings:
            raise ValueError("Resume PPO settings mismatch; use an explicit new warm start to change settings")
        if state["num_envs"] != args.num_envs:
            raise ValueError("Resume num_envs mismatch. Start an explicit new warm-start run to change it.")
        if state["stage"] == args.stage:
            old_range, new_range = resume_command_range(
                state, args.stage, args.resume, getattr(args, "allow_command_range_change", False))
            if old_range != new_range:
                state.setdefault("command_range_history", []).append({
                    "at_iteration": state["completed_iterations"], "old_range": old_range,
                    "new_range": new_range, "source_checkpoint_sha256": sha256(args.resume)})
        if state["stage"] != args.stage:
            from .artifacts import read_json
            report = read_json(args.promotion_report) if args.promotion_report else {}
            if (not gate(report) or report.get("stage") != state["stage"]
                    or report.get("checkpoint_sha256") != sha256(args.resume)
                    or not is_next_stage(state["stage"], args.stage)):
                raise ValueError("Stage promotion requires passing evaluation of the exact previous checkpoint")
            state["stage"], state["stage_iterations"] = args.stage, 0
        if "torch_rng" in data:
            torch.set_rng_state(data["torch_rng"].cpu())
        if "cuda_rng" in data and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([x.cpu() for x in data["cuda_rng"]])
        if "python_rng" in data:
            random.setstate(data["python_rng"])
    else:
        if objective:
            initialize(alg, data, learning_rate=objective["learning_rate"], init_std=objective["init_std"])
        else:
            initialize(alg, data)
        state = {"schema_version": 1, "stage": args.stage, "stage_iterations": 0,
                 "completed_iterations": 0, "learning_rate": alg.learning_rate, "contract": contract,
                 "num_envs": args.num_envs, "seed": args.seed}
    state["training_profile"] = training_profile
    state["training_profile_settings"] = objective
    record_run(directory, args, args.resume or args.init_checkpoint, contract, cfg, agent)
    profile = getattr(args, "command_profile", None) or state.get("command_profile", "baseline")
    old_profile = state.get("command_profile", "baseline")
    if profile != old_profile:
        state.setdefault("command_profile_history", []).append({
            "at_iteration": state["completed_iterations"], "stage_iteration": state["stage_iterations"],
            "old_profile": old_profile, "new_profile": profile,
            "source_checkpoint_sha256": sha256(args.resume or args.init_checkpoint)})
    state["command_profile"] = profile
    state["source_sha256"] = source_hashes()
    state["training_settings"] = training_settings
    state["command_range"] = list(STAGES[args.stage])
    stage_cap = getattr(args, "max_iterations", 3000)
    state["stage_iteration_cap"] = stage_cap
    if state["stage_iterations"] + args.iterations > stage_cap:
        raise ValueError(f"Stage cap of {stage_cap} iterations would be exceeded")
    original = {k: v.detach().clone() for k, v in alg.actor.state_dict().items() if k.startswith("mlp.")}
    obs = env.get_observations().to(args.device)
    alg.train_mode()
    losses = []
    start = time.monotonic()
    target_iteration = state["stage_iterations"] + args.iterations
    try:
        with SummaryWriter(str(directory / "tensorboard")) as writer:
            for _ in range(args.iterations):
                if not app.is_running():
                    raise RuntimeError("Simulator closed before requested iterations completed")
                rewards_sum = 0.0
                with torch.inference_mode():
                    for _ in range(24):
                        actions = alg.act(obs)
                        finite(actions)
                        obs, rewards, dones, extras = env.step(actions.to(env.device))
                        obs, rewards, dones = obs.to(args.device), rewards.to(args.device), dones.to(args.device)
                        finite(obs, rewards)
                        alg.process_env_step(obs, rewards, dones, extras)
                        rewards_sum += rewards.mean().item()
                    alg.compute_returns(obs)
                loss = alg.update()
                finite(loss, alg.actor.state_dict(), alg.critic.state_dict())
                state["completed_iterations"] += 1
                state["stage_iterations"] += 1
                state["learning_rate"] = alg.learning_rate
                row = {"iteration": state["completed_iterations"], "reward_mean": rewards_sum / 24,
                       **{k: float(v) for k, v in loss.items()}}
                losses.append(row)
                for key, val in row.items():
                    writer.add_scalar(key, val, state["completed_iterations"])
                print(f"[fine-tune] {state['stage']} {state['stage_iterations']}/{target_iteration} {row}", flush=True)
                if state["stage_iterations"] % agent.save_interval == 0:
                    snapshot = directory / f"checkpoint_{state['stage_iterations']:04d}.pt"
                    save_snapshot(alg, snapshot, state)
                    write_json(directory / "training_progress.json", {
                        "status": "training", "stage": state["stage"],
                        "stage_iterations": state["stage_iterations"], "target_iteration": target_iteration,
                        "completed_iterations": state["completed_iterations"],
                        "checkpoint": str(snapshot), "checkpoint_sha256": sha256(snapshot), "losses": losses})
                    writer.flush()
                    print(f"[checkpoint] Saved {snapshot.name}; no intermediate evaluation or scene reload.", flush=True)
        changed = any(not torch.equal(v, alg.actor.state_dict()[k]) for k, v in original.items())
        if not changed:
            raise AssertionError("Actor MLP weights did not update")
        destination = directory / "checkpoint.pt"
        save_snapshot(alg, destination, state)
        # Round-trip full state and strict weights before marking the worker successful.
        restore(alg, load_safe(destination, args.device), contract)
        write_json(directory / "training_result.json", {"actor_updated": changed, "finite": True,
                   "iterations_in_worker": args.iterations, "stage_iterations": state["stage_iterations"],
                   "completed_iterations": state["completed_iterations"], "elapsed_s": time.monotonic() - start,
                   "checkpoint": str(destination), "checkpoint_sha256": sha256(destination), "losses": losses})
        write_json(directory / "training_progress.json", {
            "status": "completed", "stage": state["stage"], "stage_iterations": state["stage_iterations"],
            "target_iteration": target_iteration, "completed_iterations": state["completed_iterations"],
            "checkpoint": str(destination), "checkpoint_sha256": sha256(destination), "losses": losses})
    except torch.cuda.OutOfMemoryError:
        print(f"CUDA OOM. No automatic config change. Retry a NEW warm-start run with fewer envs, "
              f"e.g. --num_envs {max(1, args.num_envs // 2)}.", flush=True)
        raise
    finally:
        env.close()
