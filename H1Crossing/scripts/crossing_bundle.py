"""Read-only integrity/validation checks for an exported flat locomotion bundle."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_bundle(directory, gate, stages):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))
    contract = json.loads((directory / "contract.json").read_text(encoding="utf-8"))
    for name, key in (("policy.pt", "policy_sha256"), ("checkpoint.pt", "checkpoint_sha256"),
                      ("evaluation.json", "evaluation_sha256")):
        if sha256(directory / name) != manifest.get(key):
            raise ValueError(f"Bundle checksum mismatch: {name}")
    if (not gate(report) or report.get("checkpoint_sha256") != manifest["checkpoint_sha256"]
            or report.get("stage") != manifest.get("validated_stage")
            or contract != manifest.get("contract") or contract != report.get("contract")):
        raise ValueError("Bundle requires a passing report and matching policy contract")
    stage = manifest["validated_stage"]
    if list(stages[stage]) != manifest.get("validated_vx_range_mps") or manifest.get("vy") != 0:
        raise ValueError("Bundle velocity range differs from the validated command configuration")
    if (manifest.get("slew_limits") != {"vx_mps2": 2.0, "yaw_radps2": 1.0}
            or manifest.get("yaw_limit_radps") != {"speed_at_most_1.5": 0.5, "speed_above_1.5": 0.25}):
        raise ValueError("Unsupported bundle command limits")
    return manifest
