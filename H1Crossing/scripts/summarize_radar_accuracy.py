"""Export completed radar statistics and reviewed inline chart inputs."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from evaluate_radar_tracking import signed_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--visualization_dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    report = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    if not report["complete"] or len(report["seeds"]) != 5:
        raise ValueError("A complete five-seed screen is required")
    frames = []
    for seed in report["seeds"]:
        frames.extend(json.loads(line) for line in (run / f"frames_seed_{seed['seed']}.jsonl").read_text(encoding="utf-8").splitlines())
    valid = [r for r in frames if r["error_lane_vx_mps"] is not None]
    moving = [r for r in valid if abs(r["gt_lane_vx_exposure_mps"]) >= .2]
    stopped = [r for r in valid if abs(r["gt_lane_vx_exposure_mps"]) < .2]
    comparison = {"aggregate": report["aggregate"], "vehicle_moving_threshold_mps": .2,
        "vehicle_moving_error": signed_stats([r["error_lane_vx_mps"] for r in moving]),
        "vehicle_stopped_error": signed_stats([r["error_lane_vx_mps"] for r in stopped]),
        "error_vs_instantaneous_exposure_end": signed_stats([r["error_vs_gt_end_mps"] for r in valid]),
        "total_wall_seconds": report["total_wall_seconds"], "total_sim_seconds": report["total_sim_seconds"],
        "seeds": [{"seed": s["seed"], "valid_rate": s["radar_valid_rate"],
                   "frames": s["frames"], **s["speed_error"]} for s in report["seeds"]]}
    chart_rows = [{"seed": str(s["seed"]), "metric": label, "error_mps": s[key],
                   "valid_pct": s["valid_rate"]*100, "scored_frames": s["n"]}
                  for s in comparison["seeds"] for label, key in (("MAE", "mae_mps"), ("P95", "p95_abs_mps"))]
    source = {"label": "H1 radar accuracy, seeds 42–46",
              "files": [{"label": "summary.json"}, {"label": "frames_seed_42–46.jsonl"}],
              "metricDefinitions": [{"name": "Lane-speed error", "definition": "Radar world-X speed minus actual car displacement divided by duration over the same native exposure."}],
              "filters": ["Three RGB-D cameras and one left torso radar", "Five seeds, 30 simulation seconds per seed", "Only accepted radar measurements with matched exposure ground truth"],
              "caveats": ["Invalid radar frames are excluded from conditional error; valid fraction is reported separately.",
                          "Procedural car in the current simulated straight lane; not a real-sensor accuracy claim."]}
    chart = {"schemaVersion": 1, "id": "radar-speed-error-five-seeds", "queryId": "radar-five-seeds",
             "title": "Radar: sai số vận tốc theo seed", "description": "MAE và P95 của sai số tuyệt đối trên các khung đo hợp lệ",
             "chart": {"type": "bar", "x": "seed", "y": "error_mps", "series": "metric",
                       "xLabel": "Seed", "yLabel": "Sai số (m/s)", "startAtZero": True},
             "rows": chart_rows, "columns": ["seed", "metric", "error_mps", "valid_pct", "scored_frames"],
             "source": source, "generatedAt": datetime.now(timezone.utc).isoformat()}
    receipt = {"schemaVersion": 1, "items": [{"id": "radar-accuracy", "title": "Độ chính xác và độ phủ radar",
        "queries": [{"id": "radar-five-seeds", "source": source, "rows": comparison["seeds"],
                     "columns": [{"field": "seed", "label": "Seed"}, {"field": "mae_mps", "label": "MAE (m/s)"},
                                 {"field": "p95_abs_mps", "label": "P95 (m/s)"}, {"field": "valid_rate", "label": "Valid fraction"}],
                     "reportingPeriod": "5 × 30 simulation seconds",
                     "preview": {"kind": "aggregate", "note": "Per-seed summary; frames are retained separately.", "totalRows": len(frames)}}]}]}
    for path, payload in ((run / "comparison.json", comparison),
                          (args.visualization_dir / "radar-speed-error.json", chart),
                          (args.visualization_dir / "radar-speed-sources.json", receipt)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
    with (run / "seed_metrics.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison["seeds"][0]))
        writer.writeheader()
        writer.writerows(comparison["seeds"])
    print(json.dumps({k:v for k,v in comparison.items() if k != "aggregate"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
