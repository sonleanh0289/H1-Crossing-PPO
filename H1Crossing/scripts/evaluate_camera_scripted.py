"""Camera-only velocity scoring in the unchanged camera/radar scene.

One seed and scripted H1 commands, no crossing actor or optimization. The
observer reads ground truth only after the original RGB-D capture completes.
Radar remains enabled for scene fidelity but is not an input to this score.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np


def stats(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {"n": 0, "mae": None, "rmse": None, "bias": None, "p95_abs": None, "max_abs": None}
    if not np.isfinite(a).all():
        raise AssertionError("Nonfinite score")
    return {"n": len(a), "mae": float(np.abs(a).mean()),
            "rmse": float(np.sqrt(np.mean(a*a))), "bias": float(a.mean()),
            "p95_abs": float(np.percentile(np.abs(a), 95)), "max_abs": float(np.abs(a).max())}


def summarize(rows):
    n = len(rows)
    valid = [r for r in rows if r["velocity_valid"]]
    moving = [r for r in rows if abs(r["gt_velocity_xy"][0]) >= .5]
    known = [r for r in moving if r["velocity_valid"] and abs(r["velocity_xy"][0]) >= .5]
    correct = [r for r in known if r["velocity_xy"][0]*r["gt_velocity_xy"][0] > 0]
    angular = []
    for r in valid:
        gt, est = np.array(r["gt_velocity_xy"]), np.array(r["velocity_xy"])
        if min(np.linalg.norm(gt), np.linalg.norm(est)) >= .5:
            angular.append(float(np.degrees(np.arccos(np.clip(
                gt @ est / (np.linalg.norm(gt)*np.linalg.norm(est)), -1, 1)))))
    result = {"frames": n, "detected_frames": sum(r["detected"] for r in rows),
              "position_valid_frames": sum(r["position_valid"] for r in rows),
              "velocity_valid_frames": len(valid),
              "velocity_valid_rate": len(valid)/n if n else None,
              "velocity_vector_error_mps": stats([r["velocity_error_mps"] for r in valid]),
              "vx_error_mps": stats([r["velocity_xy"][0]-r["gt_velocity_xy"][0] for r in valid]),
              "vy_error_mps": stats([r["velocity_xy"][1]-r["gt_velocity_xy"][1] for r in valid]),
              "position_error_m": stats([r["position_error_m"] for r in rows if r["position_valid"]]),
              "direction_angle_error_deg": stats(angular),
              "direction": {"speed_deadband_mps": .5, "gt_moving_frames": len(moving),
                  "available_frames": len(known), "correct_frames": len(correct),
                  "wrong_frames": len(known)-len(correct),
                  "unknown_frames": len(moving)-len(known),
                  "accuracy_when_available": len(correct)/len(known) if known else None,
                  "correct_fraction_all_moving": len(correct)/len(moving) if moving else None,
                  "gt_positive_frames": sum(r["gt_velocity_xy"][0] >= .5 for r in rows),
                  "gt_negative_frames": sum(r["gt_velocity_xy"][0] <= -.5 for r in rows)}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perception_config", type=Path, required=True)
    args, rest = parser.parse_known_args()
    import evaluate_radar_tracking as screen
    from diagnose_radar_motion import command_at
    original_load, original_write = screen.load_config, screen.write_json
    original_observer = screen.RadarTruthObserver
    camera_rows = []
    screen.load_config = lambda _: original_load(args.perception_config)
    screen.command_at = command_at

    class Observer(original_observer):
        def __init__(self, env):
            super().__init__(env)
            self.camera_capture = env.perception.capture
            env.perception.capture = self.capture_camera

        def capture_camera(self, now):
            result = self.camera_capture(now)
            if not self.enabled:
                return result
            env = self.env
            track = env.perception.tracks[0]
            state, pos_valid, vel_valid, age = track.estimate(now)
            stamp, views = env.perception.frames[0]
            if abs(stamp-now) > 1e-9:
                raise AssertionError("Camera timestamp mismatch")
            for view in views:
                if view[0].shape[:2] != (env.perception.cfg.height, env.perception.cfg.width):
                    raise AssertionError("Unexpected rendered resolution")
            truth = env.vehicle.data.root_pos_w.torch[0, :2].detach().cpu().numpy().copy()
            velocity = np.array([float(env.traffic.speed[0]), 0.])
            h1 = env.robot.data.root_pos_w.torch[0, :2].detach().cpu().numpy().copy()
            row = {"time_s": float(now), "phase": self.phase, "reset_id": self.reset_id,
                   "vehicle_pass": int(env.traffic.pass_count[0]),
                   "detected": any(v[-1] is not None for v in views),
                   "detections_by_camera": {name: v[-1] is not None
                        for name, v in zip(env.perception.camera_names, views)},
                   "position_valid": bool(pos_valid), "velocity_valid": bool(vel_valid),
                   "age_s": float(age)*env.perception.cfg.dropout_seconds,
                   "gt_xy": truth.tolist(), "gt_velocity_xy": velocity.tolist(),
                   "position_xy": state[:2].tolist() if pos_valid else None,
                   "velocity_xy": state[2:].tolist() if vel_valid else None,
                   "internal_velocity_xy": state[2:].tolist(),
                   "position_error_m": float(np.linalg.norm(state[:2]-truth)) if pos_valid else None,
                   "velocity_error_mps": float(np.linalg.norm(state[2:]-velocity)) if vel_valid else None,
                   "range_xy_m": float(np.linalg.norm(truth-h1)),
                   "h1_velocity_xy": env.robot.data.root_lin_vel_w.torch[0, :2].detach().cpu().tolist(),
                   "terminal_already_reached": bool(env.tracker.finished[0])}
            json.dumps(row, allow_nan=False)
            camera_rows.append(row)
            if len(camera_rows) % 90 == 0:
                print(f"[camera] frames={len(camera_rows)} t={now:.2f}s velocity_valid={vel_valid}", flush=True)
            return result

    def write_json(path, data):
        if path.name == "protocol.json":
            if len(data["seeds"]) != 1:
                raise ValueError("This screen requires exactly one seed")
            data["camera_score"] = "World XY camera track versus current render-time GT; radar velocity excluded"
            data["commands"] = [{"start_s": t, "phase": command_at(t)[0], "command": command_at(t)[1]}
                                for t in (0, 3, 6, 9, 12, 15)]
        if path.name == "summary.json":
            protocol = json.loads((path.parent / "protocol.json").read_text())
            expected = round(data["total_sim_seconds"]*protocol["sensor_hz"])
            if len(camera_rows) != expected:
                raise AssertionError(f"Expected {expected} captures, got {len(camera_rows)}")
            if any(b["time_s"] <= a["time_s"] for a, b in zip(camera_rows, camera_rows[1:])):
                raise AssertionError("Nonmonotonic camera clock")
            with (path.parent / "camera_frames.jsonl").open("w", encoding="utf-8") as handle:
                for row in camera_rows:
                    handle.write(json.dumps(row, allow_nan=False)+"\n")
            report = {"complete": data["complete"], "seed": protocol["seeds"][0],
                      "perception": protocol["perception"], "all": summarize(camera_rows),
                      "by_phase": {p: summarize([r for r in camera_rows if r["phase"] == p])
                          for p in dict.fromkeys(r["phase"] for r in camera_rows)},
                      "by_range": {f"{lo}_{hi}m": summarize([r for r in camera_rows if lo <= r["range_xy_m"] < hi])
                          for lo, hi in ((0,5), (5,15), (15,30), (30,50), (50,80))},
                      "wall_seconds": data["total_wall_seconds"], "sim_seconds": data["total_sim_seconds"],
                      "verification": data["verification"], "events": data["seeds"][0]["events"],
                      "caveats": ["One seed, scripted manual commands, not crossing-policy evaluation.",
                          "Velocity errors are conditional on velocity_valid; all missing frames are counted.",
                          "Direction is world-space velocity, not car body yaw or turning intention.",
                          "Current traffic travels along positive world X; negative-X performance is untested.",
                          "Terminal frames before automatic reset are retained; resets do not reset simulation time.",
                          "Ground truth is read only by the scoring observer after camera processing."]}
            original_write(path.parent / "camera_summary.json", report)
            print("[camera complete]", json.dumps(report["all"]), flush=True)
        original_write(path, data)

    screen.RadarTruthObserver = Observer
    screen.write_json = write_json
    sys.argv = [sys.argv[0], *rest]
    screen.main()


if __name__ == "__main__":
    main()
