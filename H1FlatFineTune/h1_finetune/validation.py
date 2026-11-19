"""Fail-closed evaluation gates shared by curriculum, export and CPU tests."""
import math
from .control import STAGES, GRADUAL_STAGES, stage_sequence

SEEDS = (42, 142, 242, 342, 442)


def scenarios(stage):
    sequence = stage_sequence(stage)
    reached = sequence[:sequence.index(stage) + 1]
    speeds = sorted({0.5, *[v for s in reached for v in STAGES[s]]})
    # Preserve old stage-A speed regressions and explicitly validate the 1-2 m/s band.
    milestones = (-0.3, 1.0) if stage in GRADUAL_STAGES else (-0.3, 1.0, 1.5)
    speeds = sorted({*speeds, *milestones})
    fixed = {"stand": (0.0, 0.0), "turn_left": (0.0, 0.5), "turn_right": (0.0, -0.5)}
    # Keep existing two-decimal IDs, but do not round the new -0.625/-0.875 milestones.
    fixed.update({f"vx_{v:+.2f}" if round(v, 2) == v else f"vx_{v:+.3f}": (v, 0.0)
                  for v in speeds})
    return fixed


def passes_scenario(row):
    required = ("survival_rate", "vx_mae", "vy_mae", "yaw_mae", "planar_speed")
    if not row.get("finite", False) or any(not isinstance(row.get(k), (int, float))
                                          or not math.isfinite(row[k]) for k in required):
        return False
    if not 0.95 <= row["survival_rate"] <= 1.0 or any(row[k] < 0 for k in required[1:]):
        return False
    tolerance = 0.2 if abs(row["vx"]) <= 1.0 else 0.3
    if row["vx_mae"] > tolerance or row["vy_mae"] > 0.15 or row["yaw_mae"] > 0.20:
        return False
    return row["name"] != "stand" or row["planar_speed"] <= 0.10


def gate(report):
    """Require exact scenarios, seeds and trial counts; partial smoke reports cannot pass."""
    try:
        expected = scenarios(report["stage"])
        rows = report["scenarios"]
        if len(rows) != len(expected) or {r["name"] for r in rows} != set(expected):
            return False
        for row in rows:
            if (row["vx"], row["yaw"]) != expected[row["name"]]:
                return False
            if row["rollouts"] != 50 or row["seeds"] != list(SEEDS) or not passes_scenario(row):
                return False
        sequence = report["transition"]
        return (sequence["finite"] is True and sequence["rollouts"] == 50
                and sequence["seeds"] == list(SEEDS) and 0.95 <= sequence["success_rate"] <= 1.0
                and math.isfinite(sequence["success_rate"]))
    except (KeyError, TypeError, ValueError):
        return False


def next_stage(stage, report, used_iterations, cap=3000):
    if report.get("stage") != stage:
        raise ValueError("Report stage mismatch")
    if gate(report):
        sequence = stage_sequence(stage)
        index = sequence.index(stage)
        return sequence[index + 1] if index + 1 < len(sequence) else "complete"
    return "stop" if used_iterations >= cap else stage
