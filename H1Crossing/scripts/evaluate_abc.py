"""Run matched checkpoint/seed evaluation variants, sequentially, without training."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "logs/server_imports/camera_fallback_960/logs/20261020T043939_070344Z_behavior_train/checkpoint_001000.pt"
VARIANTS = {
    "A": {"guard": 0, "oracle": False, "description": "Sensors, new guard off; legacy missing-position stop remains"},
    "B": {"guard": 0, "oracle": True, "description": "23D oracle, new guard off; position and velocity always valid"},
    "C": {"guard": 1, "oracle": False, "description": "Sensors, new guard on"},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("all", "A", "B", "C"), default="all")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry_run", action="store_true", help="Validate and print commands without writing files or launching Isaac")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")
    if args.steps <= 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Steps must be positive and seeds must be unique")
    output = (args.output or ROOT / "logs" / ("eval_abc_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))).resolve()
    if not output.is_relative_to(ROOT / "logs") or output.exists():
        parser.error("Output must be a NEW directory inside H1Crossing/logs")
    config = yaml.safe_load((ROOT / "configs/crossing_behavior_camera_radar.yaml").read_text())
    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    if not args.dry_run:
        output.mkdir(parents=True)
        with checkpoint.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        (output / "experiment.json").write_text(json.dumps({
            "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
            "steps_per_seed": args.steps, "seeds": args.seeds,
            "variants": {v: VARIANTS[v] for v in variants},
            "note": "A/B differ in sensor information and resulting legacy stop activation. Same seeds do not guarantee identical traffic trajectories after differing resets/yield responses.",
        }, indent=2), encoding="utf-8")
    print(f"Output: {output}", flush=True)
    for variant in variants:
        spec = VARIANTS[variant]
        config_path = output / f"config_{variant}.yaml"
        if not args.dry_run:
            config_path.write_text(yaml.safe_dump(dict(config, guard_enabled=spec["guard"])), encoding="utf-8")
        for seed in args.seeds:
            name = f"{variant}_seed_{seed}"
            command = [sys.executable, "-u", "-B", str(ROOT / "scripts/play_crossing_behavior.py"),
                "--checkpoint", str(checkpoint), "--config", str(config_path),
                "--observation_source", "camera_radar",
                "--perception_config", str(ROOT / "configs/crossing_perception_multicolor_960x720.yaml"),
                "--radar_config", str(ROOT / "configs/crossing_radar.yaml"),
                "--vehicle_palette", "multicolor", "--traffic_mode", "variable",
                "--yield_probability", "0.5", "--num_envs", "1", "--seed", str(seed),
                "--headless", "--steps", str(args.steps), "--ground_truth_eval",
                "--output", str(output / name)]
            if spec["oracle"]:
                command.append("--oracle_observations")
            if args.dry_run:
                print(subprocess.list2cmdline(command))
                continue
            console = output / f"{name}.console.log"
            print(f"Testing {name}: {spec['description']}", flush=True)
            with console.open("w", encoding="utf-8") as stream:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                print(f"FAILED {name}. Inspect {console}", flush=True)
                return result.returncode
            print(f"Completed {name}", flush=True)
    print("Dry run complete" if args.dry_run else f"All requested evaluations completed: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
