"""CPU-only checks. Never imports runtime/tasks, starts Kit, or executes training."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import subprocess
import unittest
import torch
from h1_finetune.artifacts import ROOT, sha256, read_json, write_json, provenance, BASE


def main():
    torch.set_num_threads(1)
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    baseline = read_json(ROOT / "reports/isolation_baseline.json")
    protected = {path: sha256(path) == checksum.lower() for path, checksum in baseline["protected_files"].items()}
    status = subprocess.run(["git", "-C", str(ROOT.parent / "IsaacLab"), "status", "--porcelain"],
                            text=True, capture_output=True, check=True).stdout.strip()
    copied_checkpoint = sha256(BASE) == sha256(baseline["base_checkpoint_source"]) == baseline["base_checkpoint_sha256"]
    report = {"scope": "CPU unit tests and syntax checks only; no simulator or training", "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
              "protected_files_unchanged": protected,
              "isaaclab_git_status_porcelain": status,
              "isaaclab_status_unchanged": status == baseline["isaaclab_git_status_porcelain"],
              "base_copy_hash_matches": copied_checkpoint,
              "not_run": ["runtime task contract", "32 env x 5 PPO iterations", "300-step headless play",
                          "100-step GUI play", "hardware joystick", "speed-envelope evaluation"],
              "provenance": provenance(42, BASE)}
    report["passed"] = (result.wasSuccessful() and all(protected.values()) and copied_checkpoint
                        and report["isaaclab_status_unchanged"])
    write_json(ROOT / "reports/code_checks.json", report)
    print(f"Code-only checks passed: {report['passed']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
