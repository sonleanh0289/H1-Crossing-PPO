"""Measure whole-GPU VRAM for selected camera resolutions, without training.

Separate simulator processes; same seed, one env, three cameras and two radars.
GPU-wide measurements include desktop/other applications, not just this process.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sample():
    result = subprocess.run([
        "nvidia-smi", "--id=0",
        "--query-gpu=memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits"], capture_output=True, text=True,
        timeout=10, check=True)
    used, total, util = map(float, result.stdout.strip().split(","))
    return {"used_mib": used, "total_mib": total, "util_percent": util}


def child(config, output, steps):
    import evaluate_radar_tracking as screen
    original_load = screen.load_config
    screen.load_config = lambda _: original_load(config)
    original_observer = screen.RadarTruthObserver

    class Observer(original_observer):
        def __init__(self, env):
            super().__init__(env)
            (output / "environment_ready.json").write_text(
                json.dumps({"wall_time": time.time()}), encoding="utf-8")

    screen.RadarTruthObserver = Observer
    sys.argv = [sys.argv[0], "--seeds", "42", "--steps", str(steps),
                "--yield_probability", "0", "--output", str(output)]
    screen.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--child_config", type=Path)
    parser.add_argument("--resolutions", nargs="+", choices=("320x240", "640x480", "960x720"),
                        default=["320x240", "640x480"])
    args = parser.parse_args()
    output = args.output.resolve()
    if args.steps < 1 or not output.is_relative_to(ROOT / "logs"):
        parser.error("Positive steps and an output inside project logs are required")
    if args.child_config:
        child(args.child_config, output, args.steps)
        return
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for label in dict.fromkeys(args.resolutions):
        suffix = "" if label == "320x240" else f"_{label}"
        config = ROOT / f"configs/crossing_perception_multicolor{suffix}.yaml"
        target = output / label
        before = sample()
        print(f"[start] {label}: before={before}", flush=True)
        started = time.monotonic()
        samples = []
        with (output / f"{label}_console.log").open("w", encoding="utf-8") as console, \
                (output / f"{label}_gpu.jsonl").open("w", encoding="utf-8") as stream:
            process = subprocess.Popen([
                sys.executable, "-B", str(Path(__file__).resolve()),
                "--child_config", str(config), "--output", str(target),
                "--steps", str(args.steps)], stdout=console, stderr=subprocess.STDOUT,
                cwd=ROOT)
            while process.poll() is None:
                try:
                    row = {"elapsed_s": time.monotonic()-started, **sample(),
                           "active": (target / "environment_ready.json").exists()
                           and not (target / "summary.json").exists()}
                    samples.append(row)
                    stream.write(json.dumps(row)+"\n")
                    stream.flush()
                except (subprocess.SubprocessError, ValueError) as error:
                    stream.write(json.dumps({"sampling_error": str(error)})+"\n")
                    stream.flush()
                time.sleep(1)
        active = [r["used_mib"] for r in samples if r["active"]]
        summary_path = target / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
        result = {"resolution": label, "exit_code": process.returncode,
                  "before": before, "samples": len(samples), "active_samples": len(active),
                  "whole_process_peak_mib": max((r["used_mib"] for r in samples), default=None),
                  "active_median_mib": statistics.median(active) if active else None,
                  "active_peak_mib": max(active) if active else None,
                  "simulator_summary": summary,
                  "measurement_scope": "Whole GPU, sampled approximately once per second; not training VRAM"}
        results.append(result)
        (output / "benchmark.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[result] {label}: exit={process.returncode}, active_median={result['active_median_mib']}, peak={result['whole_process_peak_mib']}", flush=True)
        if process.returncode or summary is None or not summary.get("complete"):
            raise RuntimeError(f"Incomplete simulation: inspect {target} and console log")
        time.sleep(5)


if __name__ == "__main__":
    main()
