"""Compare already-created evaluation reports; never launches a simulator."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h1_finetune.artifacts import read_json, write_json
from h1_finetune.stability import compare_reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON file inside H1FlatFineTune")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new JSON file")
    result = compare_reports(read_json(args.baseline), read_json(args.candidate))
    write_json(args.output, result)
    print(f"Candidate locomotion passed: {result['candidate_locomotion_passed']}")
    for row in result["scenarios"]:
        change = row["metrics"]["torso_ang_xy_rms_radps"]["change_percent"]
        print(f"{row['name']}: torso angular RMS change={change}; survival={row['candidate_survival']}")
    print(result["note"])
    print(args.output.resolve())


if __name__ == "__main__":
    main()
