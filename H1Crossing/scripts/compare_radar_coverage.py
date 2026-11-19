"""Compare matched scripted radar screens; no simulator or policy execution."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np


def read_run(path):
    summary = json.loads((path/"summary.json").read_text(encoding="utf-8"))
    protocol = json.loads((path/"protocol.json").read_text(encoding="utf-8"))
    if not summary.get("complete"):
        raise ValueError(f"Incomplete screen: {path}")
    rows = [json.loads(line) for seed in protocol["seeds"]
            for line in (path/f"frames_seed_{seed}.jsonl").read_text(encoding="utf-8").splitlines()]
    return summary, protocol, rows


def coverage(rows):
    valid = [r for r in rows if r["lane_speed_valid"]]
    errors = [r["error_lane_vx_mps"] for r in valid if r["gt_alignment_valid"]]
    return dict(frames=len(rows), valid_frames=len(valid),
        availability=len(valid)/len(rows) if rows else None,
        scored_frames=len(errors), mae_mps=float(np.mean(np.abs(errors))) if errors else None,
        p95_abs_error_mps=float(np.percentile(np.abs(errors), 95)) if errors else None,
        invalid_reasons=dict(Counter(r["reason"] for r in rows if not r["lane_speed_valid"])))


def details(summary, rows):
    def side(row):
        if not row.get("gt_alignment_valid"):
            return "unscored"
        return "world_x_positive_of_sensor" if row["gt_car_xy"][0] >= row["sensor_position_world"][0] else "world_x_negative_of_sensor"
    return dict(aggregate=coverage(rows),
        by_command={phase: coverage([r for r in rows if r["command_phase"] == phase])
                    for phase in ("stand", "forward", "reverse", "turn_left", "turn_right")},
        by_side={name: coverage([r for r in rows if side(r) == name])
                 for name in ("world_x_negative_of_sensor", "world_x_positive_of_sensor", "unscored")},
        by_seed={str(seed): coverage([r for r in rows if r["seed"] == seed]) for seed in sorted({r["seed"] for r in rows})},
        sim_seconds=summary["total_sim_seconds"], wall_seconds=summary["total_wall_seconds"],
        gpu_memory=summary.get("gpu_memory"), verification=summary["verification"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    a, pa, ra = read_run(args.baseline)
    b, pb, rb = read_run(args.candidate)
    for field in ("seeds", "steps_per_seed", "num_envs", "physics_hz", "locomotion_hz", "sensor_hz",
                  "commands", "traffic", "perception", "locomotion_policy_sha256"):
        if pa[field] != pb[field]:
            raise ValueError(f"Unmatched comparison protocol: {field}")
    pairs = list(zip(ra, rb))
    if len(ra) != len(rb) or any(x["seed"] != y["seed"] or x["command_phase"] != y["command_phase"]
            or abs(x["sim_capture_s"]-y["sim_capture_s"]) > 1e-6 for x, y in pairs):
        raise ValueError("Frame sequences differ")
    paired = [(x, y) for x, y in pairs if x["gt_alignment_valid"] and y["gt_alignment_valid"]]
    result = dict(baseline_path=str(args.baseline.resolve()), candidate_path=str(args.candidate.resolve()),
        baseline=details(a, ra), candidate=details(b, rb),
        paired_ground_truth=dict(frames=len(paired),
            max_car_xy_difference_m=max((float(np.linalg.norm(np.array(x["gt_car_xy"])-y["gt_car_xy"])) for x, y in paired), default=None),
            max_speed_difference_mps=max((abs(x["gt_lane_vx_exposure_mps"]-y["gt_lane_vx_exposure_mps"]) for x, y in paired), default=None)),
        caveats=["Scripted commands, not trained crossing-policy performance.",
                 "Accuracy is conditional on accepted frames; availability counts all capture attempts.",
                 "World-X side is relative to the sensor, not target direction of motion.",
                 "Synthetic RTX radar, single procedural vehicle and straight-lane motion only."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps({k: result[k]["aggregate"] for k in ("baseline", "candidate")}, indent=2))


if __name__ == "__main__":
    main()
