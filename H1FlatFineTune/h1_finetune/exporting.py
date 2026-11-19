"""CPU-only export, strictly gated by a full evaluation report for this checksum."""
import argparse
from pathlib import Path
import shutil
import torch
from .artifacts import ROOT, sha256, read_json, write_json, output_path
from .checkpoint import load_safe
from .control import STAGES
from .validation import gate


def export(checkpoint, report_path, destination):
    report = read_json(report_path)
    if not gate(report) or report.get("checkpoint_sha256") != sha256(checkpoint):
        raise ValueError("Export requires complete passing evaluation of this exact checkpoint checksum")
    data = load_safe(checkpoint)
    state = data.get("fine_tune")
    if not state or state["contract"] != report.get("contract"):
        raise ValueError("Export requires matching fine-tune contract in report and checkpoint")
    for name in ("env_config.json", "ppo_config.json", "provenance.json"):
        if not (Path(report_path).parent / name).is_file():
            raise ValueError(f"Evaluation bundle is missing {name}")
    destination = output_path(Path(destination) / "policy.pt").parent
    if any(destination.iterdir()):
        raise ValueError("Export destination must be empty")
    # Scalar Gaussian deterministic output is the raw actor MLP, without normalization.
    layers = []
    for ins, outs in ((69, 128), (128, 128), (128, 128), (128, 19)):
        layers.append(torch.nn.Linear(ins, outs))
        if outs != 19:
            layers.append(torch.nn.ELU())
    model = torch.nn.Sequential(*layers).eval()
    model.load_state_dict({k.removeprefix("mlp."): v for k, v in data["actor_state_dict"].items()
                           if k.startswith("mlp.")}, strict=True)
    jit = torch.jit.script(model)
    with torch.inference_mode():
        probe = torch.randn(10, 69)
        torch.testing.assert_close(jit(probe), model(probe))
    jit.save(str(destination / "policy.pt"))
    shutil.copy2(checkpoint, destination / "checkpoint.pt")
    shutil.copy2(report_path, destination / "evaluation.json")
    for name in ("env_config.json", "ppo_config.json", "provenance.json"):
        source = Path(report_path).parent / name
        shutil.copy2(source, destination / name)
    write_json(destination / "contract.json", state["contract"])
    write_json(destination / "manifest.json", {"schema_version": 1, "validated_stage": report["stage"],
               "validated_vx_range_mps": STAGES[report["stage"]], "vy": 0.0,
               "validation_scope": "Discrete tested speed milestones and stop/reverse sequence on plane; not crossing",
               "yaw_limit_radps": {"speed_at_most_1.5": 0.5, "speed_above_1.5": 0.25},
               "slew_limits": {"vx_mps2": 2.0, "yaw_radps2": 1.0},
               "checkpoint_sha256": sha256(checkpoint), "policy_sha256": sha256(destination / "policy.pt"),
               "evaluation_sha256": sha256(report_path), "contract": state["contract"]})
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(export(args.checkpoint, args.report, args.output))
