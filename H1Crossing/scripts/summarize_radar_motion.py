"""Aggregate completed counterfactual screens and verify matched trajectories."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def stats(rows):
    valid = [r for r in rows if r['lane_speed_valid']]
    errors = [abs(r['error_lane_vx_mps']) for r in valid if r['error_lane_vx_mps'] is not None]
    speed = [r['gt_h1_speed_xy_mps'] for r in rows if r['gt_alignment_valid']]
    tilt = [r['diagnostic_truth']['torso_tilt_deg'] for r in rows if 'diagnostic_truth' in r]
    missing = [r for r in rows if not r['lane_speed_valid']]
    return dict(frames=len(rows), valid=len(valid), valid_pct=100*len(valid)/len(rows) if rows else None,
        mae_mps=float(np.mean(errors)) if errors else None,
        p95_abs_mps=float(np.percentile(errors,95)) if errors else None,
        error_frames=len(errors), invalid_reasons=dict(Counter(r['reason'] for r in missing)),
        mean_actual_h1_speed_mps=float(np.mean(speed)) if speed else None,
        max_actual_h1_speed_mps=max(speed) if speed else None,
        mean_torso_tilt_deg=float(np.mean(tilt)) if tilt else None,
        p95_torso_tilt_deg=float(np.percentile(tilt,95)) if tilt else None,
        invalid_center_outside_both_fov=sum(not any(h['center_in_fov'] for h in r['head_geometry'])
            for r in missing if 'head_geometry' in r),
        invalid_center_inside_some_fov=sum(any(h['center_in_fov'] for h in r['head_geometry'])
            for r in missing if 'head_geometry' in r))


def compare(base, rows):
    assert len(base) == len(rows) == 540, 'Expected 18 s at 30 Hz per case'
    deltas = {k: [] for k in ('car', 'root_pose', 'torso_pose', 'speed')}
    resets = passes = 0
    for a,b in zip(base,rows):
        assert abs(a['sim_capture_s']-b['sim_capture_s']) < 1e-6
        assert a['command_phase'] == b['command_phase']
        if 'diagnostic_truth' not in a or 'diagnostic_truth' not in b:
            continue
        x,y = a['diagnostic_truth'], b['diagnostic_truth']
        resets += x['reset'] != y['reset']
        passes += x['pass'] != y['pass']
        for k in deltas:
            deltas[k].append(float(np.max(np.abs(np.asarray(x[k])-np.asarray(y[k])))))
    maximum = {k:max(v) if v else None for k,v in deltas.items()}
    matched = resets == passes == 0 and all(v is not None and v <= 1e-5 for v in maximum.values())
    return dict(trajectory_matched=matched, max_absolute_deltas=maximum,
        reset_mismatches=resets, pass_mismatches=passes,
        both_valid=sum(a['lane_speed_valid'] and b['lane_speed_valid'] for a,b in zip(base,rows)),
        recovered=sum(not a['lane_speed_valid'] and b['lane_speed_valid'] for a,b in zip(base,rows)),
        lost=sum(a['lane_speed_valid'] and not b['lane_speed_valid'] for a,b in zip(base,rows)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report, data = {}, {}
    for folder in args.runs:
        summary = json.loads((folder/'summary.json').read_text())
        protocol = json.loads((folder/'protocol.json').read_text())
        assert summary['complete'] and summary['verification']['finite'] and summary['verification']['locomotion_unchanged']
        assert protocol['seeds'] == [42] and protocol['steps_per_seed'] == 180
        case = protocol['diagnostic_case']
        rows = [json.loads(line) for line in (folder/'frames_seed_42.jsonl').read_text().splitlines()]
        data[case] = rows
        report[case] = dict(source=folder.name, all_frames=stats(rows),
            timing_valid=stats([r for r in rows if r.get('timing_valid')]),
            by_actual_speed={name:stats([r for r in rows if r.get('gt_alignment_valid')
                and lo <= r['gt_h1_speed_xy_mps'] < hi])
                for name,lo,hi in (('under_0.15',0,.15),('0.15_to_1',.15,1),('1_to_2',1,2),('over_2',2,100))},
            by_command={name:stats([r for r in rows if r['command_phase'] == name])
                for name in ('stand','walk','run','reverse','turn_left','turn_right')},
            events=summary['seeds'][0]['events'], wall_seconds=summary['total_wall_seconds'],
            sensor=summary['sensor'])
    assert 'baseline' in data
    for case, rows in data.items():
        report[case]['comparison_to_baseline'] = compare(data['baseline'],rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:{'all':v['all_frames'],'matched':v['comparison_to_baseline'],
                        'events':v['events']} for k,v in report.items()},indent=2))


if __name__ == '__main__':
    main()
