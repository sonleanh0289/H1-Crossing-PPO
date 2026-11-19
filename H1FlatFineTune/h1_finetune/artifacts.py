"""All generated artifacts are constrained to this project."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "checkpoints/base/checkpoint.pt"


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def output_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT) or path == ROOT:
        raise ValueError(f"Output must be inside {ROOT}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, data):
    output_path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_dir(label):
    path = ROOT / "logs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + label)
    path.mkdir(parents=True, exist_ok=False)
    return path


def source_hashes():
    return {str(p.relative_to(ROOT)): sha256(p) for folder in ("h1_finetune", "scripts", "tests")
            for p in sorted((ROOT / folder).rglob("*.py"))}


def provenance(seed, checkpoint):
    core = ROOT.parent / "IsaacLab"
    def git(*args):
        result = subprocess.run(["git", "-C", str(core), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
            "python": sys.version, "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
            "isaaclab_commit": git("rev-parse", "HEAD"), "isaaclab_status": git("status", "--porcelain"),
            "project_git_commit": None, "source_sha256": source_hashes(),
            "input_checkpoint": str(Path(checkpoint).resolve()), "input_checkpoint_sha256": sha256(checkpoint)}
