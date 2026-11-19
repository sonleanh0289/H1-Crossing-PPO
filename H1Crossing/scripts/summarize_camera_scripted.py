"""Independently verify a camera screen and prepare bounded inline evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    rows = [json.loads(line) for line in (run / "camera_frames.jsonl").read_text().splitlines()]
    report = json.loads((run / "camera_summary.json").read_text())
    protocol = json.loads((run / "protocol.json").read_text())
    if not report["complete"]:
        raise AssertionError("Incomplete run")
    valid = [r for r in rows if r["velocity_valid"]]
    errors = [math.dist(r["velocity_xy"], r["gt_velocity_xy"]) for r in valid]
    assert len(rows) == report["all"]["frames"]
    assert len(valid) == report["all"]["velocity_valid_frames"]
    for row, error in zip(valid, errors):
        assert math.isclose(error, row["velocity_error_mps"], abs_tol=1e-10)
    if errors:
        assert math.isclose(statistics.mean(errors), report["all"]["velocity_vector_error_mps"]["mae"], abs_tol=1e-10)
        assert math.isclose(math.sqrt(statistics.mean(e*e for e in errors)),
                            report["all"]["velocity_vector_error_mps"]["rmse"], abs_tol=1e-10)
    scripts = Path(__file__).resolve().parent
    changed = [name for name, checksum in protocol["sources_sha256"].items()
               if hashlib.sha256((scripts / name).read_bytes()).hexdigest() != checksum]
    assert not changed, changed
    sign_correct = sum(r["velocity_valid"] and abs(r["gt_velocity_xy"][0]) >= .5
                       and abs(r["velocity_xy"][0]) >= .5
                       and r["velocity_xy"][0]*r["gt_velocity_xy"][0] > 0 for r in rows)
    assert sign_correct == report["all"]["direction"]["correct_frames"]
    phases = [{"phase": phase, "frames": v["frames"], "valid": v["velocity_valid_frames"],
               "vector_mae_mps": v["velocity_vector_error_mps"]["mae"],
               "direction_correct": v["direction"]["correct_frames"]}
              for phase, v in report["by_phase"].items()]
    review = {"independent_recalculation": True, "sources_unchanged": True,
              "frames": len(rows), "valid_frames": len(valid), "by_phase": phases,
              "ground_truth_vx_range": [min(r["gt_velocity_xy"][0] for r in rows), max(r["gt_velocity_xy"][0] for r in rows)],
              "range_xy_m": [min(r["range_xy_m"] for r in rows), max(r["range_xy_m"] for r in rows)],
              "terminal_frames_retained": sum(r["terminal_already_reached"] for r in rows)}
    (run / "camera_review.json").write_text(json.dumps(review, indent=2), encoding="utf-8")
    source = {"label": "Camera 960x720 seed 42",
              "files": [{"label": "camera_frames.jsonl"}, {"label": "camera_summary.json"},
                        {"label": "evaluate_camera_scripted.py"}],
              "filters": ["Seed: 42", "One environment; three RGB-D cameras at 30 Hz", "Scripted H1 motion; no crossing actor", "Vehicle yielding disabled"],
              "caveats": ["Errors use velocity-valid frames; invalid frames remain in coverage counts.",
                          "Only positive-X vehicle motion was tested, not reverse traffic or turning.",
                          "Ground truth is used after image processing for evaluation only; radar does not supply the camera velocity.",
                          "One seed does not establish general tracking reliability or a resolution improvement."],
              "metricDefinitions": [{"label": "Velocity error", "definition": "Mean Euclidean XY velocity error on velocity-valid frames, in m/s."},
                                    {"label": "Direction", "definition": "World-X sign agreement when both measured and true longitudinal speeds have magnitude at least 0.5 m/s."}]}
    chart_rows = []
    for row in rows:
        chart_rows.append({"time_s": row["time_s"], "source": "Ground truth", "vx_mps": row["gt_velocity_xy"][0]})
        chart_rows.append({"time_s": row["time_s"], "source": "Camera (valid)",
                           "vx_mps": row["velocity_xy"][0] if row["velocity_valid"] else None})
    chart = {"schemaVersion": 1, "id": "camera-960-velocity", "queryId": "camera-960-frames",
             "title": "Camera 960×720: vận tốc dọc so với ground truth",
             "chart": {"type": "line", "x": "time_s", "y": "vx_mps", "series": "source",
                       "xLabel": "Thời gian mô phỏng (s)", "yLabel": "Vận tốc vx (m/s)", "startAtZero": True},
             "rows": chart_rows, "source": source, "theme": "codex-classic"}
    (run / "chart.json").write_text(json.dumps(chart, ensure_ascii=False), encoding="utf-8")
    receipt = {"schemaVersion": 1, "items": [{"id": "camera-tracking", "title": "Camera velocity, coverage and direction",
                "queries": [{"id": "camera-960-frames", "source": source,
                             "reportingPeriod": "18 seconds of simulated motion, one seed",
                             "rows": phases, "columns": ["phase", "frames", "valid", "vector_mae_mps", "direction_correct"],
                             "methods": [{"language": "calculation", "code": "Velocity error = mean(sqrt((vx_est - vx_gt)^2 + (vy_est - vy_gt)^2)) over velocity-valid frames. Coverage = velocity-valid frames / all captured frames. Direction = sign(vx_est) agrees with sign(vx_gt), with a 0.5 m/s deadband; unavailable estimates count as unknown."}],
                             "preview": {"kind": "aggregate", "note": "All captured frames grouped by scripted H1 command phase", "totalRows": len(rows)}}]}]}
    (run / "sources.json").write_text(json.dumps(receipt, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(review, indent=2))


if __name__ == "__main__":
    main()
