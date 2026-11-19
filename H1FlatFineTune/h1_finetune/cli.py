"""Safe CLI: help/invalid invocation exits before importing or launching Kit."""
import argparse
import math
from pathlib import Path
import subprocess
import sys
import traceback
from .artifacts import ROOT, BASE, run_dir, read_json, write_json, output_path
from .control import STAGES, stage_sequence, COMMAND_PROFILES, resolve_command_profile
from .training_profiles import TRAINING_PROFILES, STABILITY_PROFILES, resolve_training_profile


def parser_for(mode):
    parser = argparse.ArgumentParser(description=f"Isolated H1 flat fine-tune: {mode}")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", type=Path, help="Output directory inside H1FlatFineTune only")
    parser.add_argument("--num_envs", type=int, default=None,
                        help="Fresh training default: 128; resume inherits checkpoint; smoke uses 32")
    if mode == "train":
        parser.add_argument("--training_profile", choices=TRAINING_PROFILES, default=None,
                            help="Opt-in reward/PPO profile; resume inherits it. Changes require a new warm start")
        parser.add_argument("--command_profile", choices=COMMAND_PROFILES, default=None,
                            help="Sampler: inherit resume checkpoint if omitted, otherwise baseline")
        source = parser.add_mutually_exclusive_group()
        source.add_argument("--init_checkpoint", type=Path)
        source.add_argument("--resume", type=Path)
        parser.add_argument("--allow_command_range_change", action="store_true",
                            help="Resume optimizer/std while explicitly accepting a changed same-stage vx range")
        choice = parser.add_mutually_exclusive_group()
        choice.add_argument("--stage", choices=STAGES)
        choice.add_argument("--curriculum", action="store_true")
        choice.add_argument("--smoke_test", action="store_true")
        parser.add_argument("--max_iterations", type=int, default=3000, help="Total per-stage cap, not additional iterations")
        parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
        parser.add_argument("--iterations", type=int, default=3000, help=argparse.SUPPRESS)
        parser.add_argument("--promotion_report", type=Path, help="Passing prior-stage report for explicit promotion")
    else:
        parser.add_argument("--checkpoint", type=Path, default=BASE)
        parser.add_argument("--stage", choices=STAGES, default="A")
    if mode == "evaluate":
        parser.add_argument("--record_stability_metrics", action="store_true",
                            help="Measure torso tilt/angular motion without changing Stage gates")
        parser.add_argument("--record_yaw_trace", action="store_true",
                            help="Record both turn scenarios at 50 Hz in yaw_trace.csv; gates unchanged")
    if mode == "play":
        parser.add_argument("--steps", type=int, default=-1)
        parser.add_argument("--vx", type=float, default=0.0)
        parser.add_argument("--yaw_rate", type=float, default=0.0)
        inputs = parser.add_mutually_exclusive_group()
        inputs.add_argument("--joystick", action="store_true")
        inputs.add_argument("--keyboard", action="store_true", help="Windows GUI: hold WASD, Shift for stage maximum")
        parser.add_argument("--keyboard_speed", type=float, default=1.0, help="W speed in m/s (clamped to stage)")
        parser.add_argument("--keyboard_reverse_speed", type=float, default=0.5, help="S speed magnitude in m/s")
        parser.add_argument("--joystick_id", type=int)
        parser.add_argument("--forward_axis", default="y", choices=["x", "y", "z", "r", "u", "v"])
        parser.add_argument("--turn_axis", default="x", choices=["x", "y", "z", "r", "u", "v"])
        parser.add_argument("--forward_sign", type=int, choices=[-1, 1], default=-1)
        parser.add_argument("--turn_sign", type=int, choices=[-1, 1], default=-1)
    return parser


def child(script, flags):
    subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / f"{script}.py"), *map(str, flags)],
                   cwd=ROOT, check=True)


def child_result(script, flags, result_name):
    """Require success artifacts even if Kit shutdown returns exit code zero."""
    directory = Path(flags[flags.index("--output") + 1])
    process_error = None
    try:
        child(script, flags)
    except subprocess.CalledProcessError as error:
        process_error = error
    failure_path = directory / "failure.json"
    if failure_path.is_file():
        try:
            failure = read_json(failure_path)
            detail = f"{failure['error']}: {failure['message']}"
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise RuntimeError(f"{script} worker failed; cannot read failure report: {failure_path}") from error
        raise RuntimeError(f"{script} worker failed: {detail}. Details: {failure_path}") from process_error
    if process_error is not None:
        raise RuntimeError(f"{script} worker exited with code {process_error.returncode}; "
                           f"no failure.json was saved. Check the console output and {directory}.") from process_error
    result_path = directory / result_name
    if not result_path.is_file():
        raise RuntimeError(f"{script} worker exited without {result_name} or failure.json. "
                           f"Run is incomplete; check the console output and {directory}.")
    return read_json(result_path)


def train_controller(args):
    """Train one uninterrupted worker per stage; evaluate only at the stage cap."""
    from .checkpoint import load_safe
    from .validation import next_stage
    resumed = load_safe(args.resume).get("fine_tune") if args.resume else None
    if args.resume and not resumed:
        raise ValueError("--resume needs a fine-tune checkpoint. Use --init_checkpoint for the base model.")
    stage = args.stage or (resumed["stage"] if resumed else "A1")
    args.command_profile = resolve_command_profile(getattr(args, "command_profile", None), resumed)
    args.training_profile = resolve_training_profile(getattr(args, "training_profile", None), resumed)
    if resumed and resumed["stage"] == stage:
        from .checkpoint import resume_command_range
        resume_command_range(resumed, stage, args.resume, getattr(args, "allow_command_range_change", False))
    count = resumed["stage_iterations"] if resumed and resumed["stage"] == stage else 0
    checkpoint = args.resume or args.init_checkpoint
    promotion = args.promotion_report
    resume = bool(args.resume)
    while True:
        remaining = args.max_iterations - count
        if remaining > 0:
            iterations = remaining
            folder = Path(args.output) / f"{stage}_{count + iterations:04d}"
            folder.mkdir(parents=True, exist_ok=False)
            flags = ["--worker", "--stage", stage, "--iterations", iterations,
                     "--max_iterations", args.max_iterations,
                     "--num_envs", args.num_envs, "--seed", args.seed, "--device", args.device,
                     "--output", folder, "--resume" if resume else "--init_checkpoint", checkpoint]
            flags += ["--command_profile", args.command_profile]
            flags += ["--training_profile", args.training_profile]
            if args.headless:
                flags += ["--headless"]
            if resume and getattr(args, "allow_command_range_change", False):
                flags += ["--allow_command_range_change"]
            if promotion:
                flags += ["--promotion_report", promotion]
            result = child_result("train", flags, "training_result.json")
            checkpoint = folder / "checkpoint.pt"
            count = result["stage_iterations"]
        else:
            # A checkpoint at the cap still needs evaluation, including after an interrupted run.
            folder = Path(args.output) / f"{stage}_{count:04d}_recheck"
            folder.mkdir(parents=True, exist_ok=False)
        evaluation_dir = folder / "evaluation"
        evaluation_flags = ["--checkpoint", checkpoint, "--stage", stage, "--headless",
                            "--device", args.device, "--output", evaluation_dir]
        if args.command_profile == "turn_focus":
            evaluation_flags += ["--record_yaw_trace"]
        if args.training_profile in STABILITY_PROFILES:
            evaluation_flags += ["--record_stability_metrics"]
        report = child_result("evaluate", evaluation_flags, "evaluation.json")
        report_path = evaluation_dir / "evaluation.json"
        decision = next_stage(stage, report, count, args.max_iterations)
        write_json(Path(args.output) / "curriculum_state.json",
                   {"stage": stage, "stage_iterations": count, "checkpoint": str(checkpoint),
                    "evaluation": str(report_path), "decision": decision})
        if decision in ("stop", "complete") or (decision != stage and not args.curriculum):
            print(f"Stage {stage}: {decision}. Checkpoint: {checkpoint}", flush=True)
            return
        if decision != stage:
            stage, count, promotion = decision, 0, report_path
        else:
            promotion = None
        resume = True


def main(mode):
    parser = parser_for(mode)
    args = parser.parse_args()
    for name in ("resume", "init_checkpoint", "checkpoint", "promotion_report", "output"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, value.resolve())
    if mode == "train":
        if not (args.smoke_test or args.stage or args.curriculum):
            parser.print_help()
            print("\nNo training started. Choose --smoke_test, --stage or --curriculum explicitly.")
            return
        if args.max_iterations < 1 or args.iterations < 1:
            parser.error("Iteration limits must be positive; --max_iterations is the total per-stage cap")
        if args.allow_command_range_change and not args.resume:
            parser.error("--allow_command_range_change requires --resume")
        if args.curriculum:
            from .checkpoint import load_safe
            curriculum_start = (load_safe(args.resume).get("fine_tune", {}).get("stage", "A1")
                                if args.resume else "A1")
            ranges = [STAGES[s] for s in stage_sequence(curriculum_start)]
            if any(low > prev_low or high < prev_high
                   for (prev_low, prev_high), (low, high) in zip(ranges, ranges[1:])):
                parser.error("Curriculum ranges must expand monotonically; check STAGES.")
        if args.smoke_test and (args.resume or args.promotion_report):
            parser.error("Smoke test is a fresh warm start, not resume/promotion")
        args.init_checkpoint = args.init_checkpoint or (None if args.resume else BASE)
        if args.smoke_test:
            if args.num_envs not in (None, 32):
                parser.error("Smoke test is fixed at 32 env x 5 iterations")
            args.num_envs, args.stage, args.iterations = 32, "A", 5
        if args.resume:
            from .checkpoint import load_safe
            state = load_safe(args.resume).get("fine_tune", {})
            args.num_envs = args.num_envs or state.get("num_envs", 64)
            if state and args.seed != state["seed"]:
                parser.error("Resume must use the checkpoint seed")
        args.num_envs = args.num_envs or 128
        args.command_profile = resolve_command_profile(
            args.command_profile, state if args.resume else None)
        args.training_profile = resolve_training_profile(args.training_profile, state if args.resume else None)
        if args.training_profile in STABILITY_PROFILES:
            stage = args.stage or (state.get("stage") if args.resume else None)
            if args.smoke_test or args.curriculum or stage != "E":
                parser.error(f"{args.training_profile} is a separate Stage E refinement; specify --stage E")
        source = args.resume or args.init_checkpoint
    else:
        args.num_envs = args.num_envs or (10 if mode == "evaluate" else 1)
        if mode == "evaluate" and args.num_envs != 10:
            parser.error("Evaluation uses 10 parallel trials for each of 5 seeds")
        if mode == "play" and (args.steps == 0 or args.steps < -1):
            parser.error("--steps must be -1 or positive")
        if mode == "play" and args.keyboard:
            if args.headless:
                parser.error("--keyboard needs the simulator GUI; remove --headless")
            if sys.platform != "win32":
                parser.error("--keyboard currently requires Windows")
            if any(not math.isfinite(v) or v <= 0 for v in (args.keyboard_speed, args.keyboard_reverse_speed)):
                parser.error("Keyboard speeds must be finite and positive")
        source = args.checkpoint
    if args.num_envs <= 0 or not source.is_file():
        parser.error(f"Invalid env count or checkpoint does not exist: {source}")
    if mode == "train" and args.training_profile in STABILITY_PROFILES:
        from .checkpoint import load_safe
        source_state = load_safe(source).get("fine_tune", {})
        if source_state.get("stage") != "E":
            parser.error(f"{args.training_profile} needs a Stage E fine-tune checkpoint, not the original published base")
    if args.output:
        output_path(args.output / "_validate_path")
        args.output.mkdir(parents=True, exist_ok=True)
    else:
        args.output = run_dir(mode)
    # Prevent accidental replacement of an existing worker's model or evaluation.
    if any((args.output / n).exists() for n in ("checkpoint.pt", "evaluation.json", "play_result.json")):
        parser.error("Output directory already contains results; choose a new directory")
    if mode == "train" and not args.worker and not args.smoke_test:
        train_controller(args)
        return
    # Only explicit runtime modes get here. No app launches during import/help/unit tests.
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(visualizer="none" if args.headless else "kit", device=args.device)
    try:
        if mode == "train":
            from .train_worker import run
        elif mode == "evaluate":
            from .evaluation import evaluate as run
        else:
            from .playback import play as run
        run(args, launcher.app)
    except Exception as error:
        write_json(args.output / "failure.json", {"error": type(error).__name__, "message": str(error),
                   "traceback": traceback.format_exc(),
                   "passed": False, "note": "Failed/incomplete execution is never eligible for promotion/export."})
        if "out of memory" in str(error).lower():
            print(f"Insufficient VRAM. Retry a NEW warm-start run with fewer envs, e.g. "
                  f"--num_envs {max(1, args.num_envs // 2)}; no automatic change made.")
        raise
    finally:
        launcher.app.close()
