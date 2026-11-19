"""Summarize the raw-return audit without modifying tracking or sensor data."""
import argparse
from collections import Counter
import json
from pathlib import Path


def head_cause(a):
    if a is None or 'truth_roi' not in a:
        return 'unscored'
    roi = a['truth_roi']
    if roi['raw_count'] < 2:
        return 'fewer_than_two_raw_roi_returns'
    for name,count in roi['stage_counts'].items():
        if count < 2:
            return 'roi_rejected_by_'+name
    point_ids = set(roi['point_ids'])
    target_groups = [g for g in a['clusters'] if point_ids.intersection(g['ids'])]
    if any(g['reason']=='candidate' for g in target_groups):
        return 'roi_candidate_present'
    if any(g['reason']=='oversized' for g in target_groups):
        return 'roi_merged_into_oversized_cluster'
    return 'roi_split_into_small_clusters'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    rows = [json.loads(l) for l in (args.run/'frames_seed_42.jsonl').read_text().splitlines()]
    old = [json.loads(l) for l in (args.baseline/'frames_seed_42.jsonl').read_text().splitlines()]
    assert len(rows)==len(old)==540
    compared = 0
    mismatch = Counter()
    for a,b in zip(rows,old):
        for k in ('sim_capture_s','reason','lane_speed_valid','diagnostic_truth'):
            if a.get(k)!=b.get(k):
                mismatch[k]+=1
        compared += 1
    missing = [r for r in rows if r['reason']=='no_lane_cluster']
    head_breakdown = Counter()
    rig_breakdown = Counter()
    examples = {}
    for r in missing:
        causes = [head_cause(a) for a in r['filter_audits']]
        head_breakdown.update(causes)
        # A failure with >=2 target-ROI points on either head takes precedence
        # over an empty/opposite-facing head. This is a diagnostic partition,
        # not a proof of physical point ownership or occlusion.
        priority = ('roi_merged_into_oversized_cluster','roi_split_into_small_clusters',
            'roi_candidate_present','roi_rejected_by_ego','roi_rejected_by_height',
            'roi_rejected_by_lane','roi_rejected_by_range','roi_rejected_by_finite',
            'fewer_than_two_raw_roi_returns','unscored')
        cause = next(c for c in priority if c in causes)
        rig_breakdown[cause] += 1
        if cause not in examples:
            examples[cause] = dict(sim_capture_s=r['sim_capture_s'],phase=r['command_phase'],
                truth=r.get('diagnostic_truth'),heads=r['filter_audits'])
    result = dict(frames=len(rows),baseline_reproduction_mismatches=dict(mismatch),
        rig_reasons=dict(Counter(r['reason'] for r in rows)),
        max_raw_roi_points_on_cluster_failures=dict(Counter(
            str(max(a['truth_roi']['raw_count'] for a in r['filter_audits'] if a and 'truth_roi' in a))
            if any(a and 'truth_roi' in a for a in r['filter_audits']) else 'unscored'
            for r in missing)),
        underlying_head_reasons=dict(Counter(' | '.join(
            a['original_result']['reason'] if a else 'missing' for a in r['filter_audits']) for r in missing)),
        missing_cluster_frames=len(missing), missing_cluster_head_causes=dict(head_breakdown),
        missing_cluster_rig_causes=dict(rig_breakdown),examples=examples,
        caveats=['Target ROI is an exposure-swept geometric box, not object-ID ground truth.',
                 'Fewer than two ROI returns does not by itself distinguish occlusion, scan sparsity, or absent target response.',
                 'Rig categories prioritize a head with surviving target-region points over an empty opposite head.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='examples'},indent=2))


if __name__ == '__main__':
    main()
