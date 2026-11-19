"""Pair a visible-robot audit with its render-hidden occlusion control."""
import argparse
from collections import Counter
import json
from pathlib import Path


def load(folder):
    return [json.loads(line) for line in (folder/'frames_seed_42.jsonl').read_text().splitlines()]


def roi_count(row):
    heads = row['filter_audits']
    counts = [h['truth_roi']['raw_count'] for h in heads if h and 'truth_roi' in h]
    return max(counts) if counts else None


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--visible',type=Path,required=True)
    p.add_argument('--hidden',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    a,b=load(args.visible),load(args.hidden)
    assert len(a)==len(b)==540
    mismatch=Counter()
    for x,y in zip(a,b):
        for key in ('sim_capture_s','diagnostic_truth','head_geometry'):
            if x.get(key)!=y.get(key):
                mismatch[key]+=1
    failures=[(x,y) for x,y in zip(a,b) if x['reason']=='no_lane_cluster']
    shortages=[(x,y) for x,y in failures if roi_count(x) is not None and roi_count(x)<2]
    recovered=[(x,y) for x,y in shortages if roi_count(y) is not None and roi_count(y)>=2]
    output=dict(frames_per_case=len(a),trajectory_or_mount_mismatches=dict(mismatch),
        visible_reasons=dict(Counter(r['reason'] for r in a)),
        hidden_reasons=dict(Counter(r['reason'] for r in b)),
        baseline_cluster_failures=len(failures), baseline_raw_roi_shortages=len(shortages),
        raw_roi_shortages_recovered_when_robot_hidden=len(recovered),
        baseline_cluster_failures_now_velocity_valid=sum(y['lane_speed_valid'] for _,y in failures),
        new_invalid_frames=sum(x['lane_speed_valid'] and not y['lane_speed_valid'] for x,y in zip(a,b)),
        baseline_failure_hidden_reasons=dict(Counter(y['reason'] for _,y in failures)),
        representative_recovered=[dict(time=x['sim_capture_s'],phase=x['command_phase'],
            visible_max_roi_points=roi_count(x),hidden_max_roi_points=roi_count(y),
            hidden_reason=y['reason']) for x,y in recovered[:10]],
        caveats=['Rendering visibility also affects camera occlusion; raw ROI recovery, not velocity-validity alone, isolates return availability.',
            'This is a whole-robot visibility control, not identification of a specific blocking link.',
            'Target ROI is geometric and may include clutter; it does not use object IDs.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
