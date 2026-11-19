"""Diagnostic torso metrics; no reward changes or relaxed locomotion gates."""
import math
import torch

STABILITY_METRICS = (
    "torso_tilt_rms_rad",
    "torso_ang_xy_rms_radps",
    "torso_ang_acc_xy_rms_radps2",
    "torso_vertical_speed_rms_mps",
)
STABILITY_PROTOCOL = "v1: torso_link; world XY angular motion; 50Hz; 5s warmup + 10s; survived trials only"


def squared_stability_samples(quat_xyzw, angular_w, linear_w, previous_angular_w, dt):
    """Return squared instantaneous quantities; average then sqrt for RMS.

    Angular acceleration is a finite difference at policy rate, not a direct
    physical acceleration sensor. Exclude reset/warmup data in the caller.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Stability metric dt must be positive and finite")
    norm = quat_xyzw.norm(dim=-1, keepdim=True)
    if (norm <= 0).any():
        raise ValueError("Invalid zero quaternion")
    q = quat_xyzw / norm
    up_z = 1 - 2 * q[..., :2].square().sum(-1)
    tilt = torch.acos(up_z.clamp(-1, 1))
    acceleration = torch.zeros_like(angular_w) if previous_angular_w is None else (
        angular_w - previous_angular_w) / dt
    result = torch.stack((tilt.square(), angular_w[..., :2].square().sum(-1),
                          acceleration[..., :2].square().sum(-1), linear_w[..., 2].square()), dim=-1)
    if not torch.isfinite(result).all():
        raise ValueError("Non-finite stability measurement")
    return result


def compare_reports(baseline, candidate):
    """Per-scenario diagnostics, not an automatic claim that balance improved."""
    from .validation import gate
    if not gate(baseline):
        raise ValueError("Baseline must pass the full locomotion protocol")
    for report in (baseline, candidate):
        if report.get("stability_metrics_protocol") != STABILITY_PROTOCOL:
            raise ValueError("Re-evaluate both checkpoints with --record_stability_metrics")
    if any(baseline.get(k) != candidate.get(k) for k in ("stage", "contract", "protocol")):
        raise ValueError("Reports must use the same stage, contract and protocol")
    reference = {r["name"]: r for r in baseline["scenarios"]}
    compared = {r["name"]: r for r in candidate["scenarios"]}
    if len(compared) != len(candidate["scenarios"]) or set(reference) != set(compared):
        raise ValueError("Scenario sets differ or contain duplicate IDs")
    keys = (*STABILITY_METRICS, "action_rate_mse", "foot_slip_mps",
            "torque_saturation_fraction", "vx_mae", "vy_mae", "yaw_mae")
    rows = []
    for name, old in reference.items():
        new = compared[name]
        if any(old.get(k) != new.get(k) for k in ("vx", "yaw", "seeds", "rollouts")):
            raise ValueError("Scenario command or sampling protocol differs")
        row = {"name": name, "baseline_survival": old["survival_rate"],
               "candidate_survival": new["survival_rate"], "metrics": {}}
        for key in keys:
            if key not in old or key not in new:
                raise ValueError(f"Missing metric: {name}/{key}")
            a, b = old.get(key), new.get(key)
            if any(v is not None and (not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0)
                   for v in (a, b)):
                raise ValueError(f"Invalid metric: {name}/{key}")
            row["metrics"][key] = {"baseline": a, "candidate": b,
                "change_percent": 100 * (b / a - 1) if a is not None and a > 0 and b is not None else None}
        rows.append(row)
    return {"baseline_checkpoint": baseline.get("checkpoint"),
            "baseline_sha256": baseline.get("checkpoint_sha256"),
            "candidate_checkpoint": candidate.get("checkpoint"),
            "candidate_sha256": candidate.get("checkpoint_sha256"),
            "candidate_locomotion_passed": gate(candidate), "scenarios": rows,
            "note": "Negative change means lower metric. No automatic stability pass; inspect all speeds, "
                    "survival and tracking. Survivor-only metrics may look better when more trials fall."}
