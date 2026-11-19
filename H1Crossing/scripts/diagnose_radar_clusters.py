"""Read-only filter audit of live radar returns, using the seed-42 motion screen.

The original association executes once and its result is returned unchanged.
All diagnostic truth is attached by an observer after sensor processing.
No production thresholds, sensor settings, or policy inputs are changed.
"""
import numpy as np


def inspect_filters(points, radial, position, rotation, ego_velocity, camera_center,
                    camera_age, vehicle_size, radar_cfg, *, lane_y, association_cfg,
                    ego_xy, timing_valid, now):
    from radar_processing import diagnose_returns
    points, radial = np.asarray(points), np.asarray(radial)
    world = points @ np.asarray(rotation).T + position
    ranges = np.linalg.norm(points, axis=1)
    masks = {
        'finite': np.isfinite(points).all(axis=1) & np.isfinite(radial),
        'range': (ranges >= radar_cfg.min_range_m) & (ranges <= radar_cfg.max_range_m),
        'lane': np.abs(world[:,1]-lane_y) <= vehicle_size[1]/2+radar_cfg.association_margin_m,
        'height': (world[:,2] > .25) & (world[:,2] < 2.),
        'ego': np.linalg.norm(world[:,:2]-ego_xy,axis=1) > association_cfg.ego_exclusion_radius_m,
    }
    keep = np.ones(len(points),dtype=bool)
    stages, stage_masks = {}, {}
    for name, mask in masks.items():
        keep &= mask
        stages[name] = int(keep.sum())
        stage_masks[name] = keep.copy()
    ids = np.flatnonzero(keep)
    ids = ids[np.argsort(world[ids,0])]
    groups = np.split(ids,np.flatnonzero(np.diff(world[ids,0]) > association_cfg.cluster_gap_m)+1)
    clusters = []
    for group in groups:
        if not len(group):
            continue
        span = np.ptp(world[group,:2],axis=0)
        reason = ('too_few_points' if len(group) < radar_cfg.min_returns else
                  'oversized' if np.any(span > np.asarray(vehicle_size[:2])+2*radar_cfg.association_margin_m)
                  else 'candidate')
        clusters.append(dict(ids=group.tolist(), count=len(group), span_xy=span.tolist(),
            min_xyz=world[group].min(axis=0).tolist(), max_xyz=world[group].max(axis=0).tolist(), reason=reason))
    guided = None
    if camera_center is not None and 0 <= camera_age <= radar_cfg.max_camera_age_s:
        guided = diagnose_returns(points[keep],radial[keep],position,rotation,ego_velocity,
            camera_center,camera_age,vehicle_size,radar_cfg,timing_valid=timing_valid)
    public = dict(now=now, total_returns=len(points), stage_counts=stages, clusters=clusters,
        camera_center=np.asarray(camera_center).tolist() if camera_center is not None else None,
        camera_age=camera_age, camera_guided_result=guided, lane_y=lane_y,
        ego_xy=np.asarray(ego_xy).tolist(), vehicle_size=list(vehicle_size),
        finite_points=[dict(id=int(i), world=world[i].tolist(), radial=float(radial[i]),
            gates={name:bool(mask[i]) for name,mask in masks.items()}) for i in np.flatnonzero(masks['finite'])])
    return public, world, stage_masks


def main():
    import argparse
    import sys
    flags = argparse.ArgumentParser(add_help=False)
    flags.add_argument('--hide_robot_for_audit',action='store_true')
    flags.add_argument('--audit_rays_per_degree',type=float)
    experiment, remaining = flags.parse_known_args()
    sys.argv = [sys.argv[0],*remaining]
    import evaluate_radar_tracking as screen
    if experiment.audit_rays_per_degree is not None:
        from dataclasses import replace
        original_load = screen.load_radar_config
        def load_config(path):
            return replace(original_load(path),rays_per_degree=experiment.audit_rays_per_degree)
        screen.load_radar_config = load_config
    from radar_observations import LaneRadarAssociation
    original_process = LaneRadarAssociation.process
    def process(self, *args, **kwargs):
        result = original_process(self,*args,**kwargs)
        self._filter_audit = inspect_filters(*args,**kwargs,lane_y=self.lane_y,association_cfg=self.cfg)
        self._filter_audit[0]['original_result'] = dict(result)
        return result
    LaneRadarAssociation.process = process
    original_observer = screen.RadarTruthObserver
    class AuditObserver(original_observer):
        def __init__(self, env):
            super().__init__(env)
            if experiment.hide_robot_for_audit:
                from pxr import UsdGeom
                from isaacsim.core.experimental.utils.stage import get_current_stage
                prim = get_current_stage(backend='usd').GetPrimAtPath('/World/envs/env_0/Robot')
                if not prim.IsValid():
                    raise RuntimeError('Robot prim missing for occlusion intervention')
                UsdGeom.Imageable(prim).MakeInvisible()

        def capture(self, now, camera_frames, vehicle_size):
            count = len(self.rows)
            result = super().capture(now,camera_frames,vehicle_size)
            if len(self.rows) == count:
                return result
            row = self.rows[-1]
            end = row.get('sensor_clock_reference_sim_s',-1)
            duration = (row.get('exposure_end_timestamp_ns',0)-row.get('exposure_start_timestamp_ns',0))*1e-9
            a = next((h for h in reversed(self.history) if abs(h['t']-(end-duration)) < 1e-5),None)
            b = next((h for h in reversed(self.history) if abs(h['t']-end) < 1e-5),None)
            row['filter_audits'] = []
            for sensor in self.env.radar.sensors:
                audit = getattr(sensor.association,'_filter_audit',None)
                if audit is None or abs(audit[0]['now']-end) > 1e-5:
                    row['filter_audits'].append(None)
                    continue
                public, world, stages = audit
                # Copy before attaching observer-only ground truth; no feedback
                # path from this data into native association or observations.
                public = dict(public)
                if a is not None and b is not None and a['reset']==b['reset'] and a['pass']==b['pass']:
                    center = np.asarray(a['car'])
                    # Broad vehicle ROI, not a claim that every point belongs to
                    # the car: include both exposure endpoints plus 0.20 m margin.
                    lo = np.minimum(a['car'],b['car'])
                    hi = np.maximum(a['car'],b['car'])
                    roi = ((world[:,0] >= lo[0]-vehicle_size[0]/2-.2)
                        & (world[:,0] <= hi[0]+vehicle_size[0]/2+.2)
                        & (np.abs(world[:,1]-center[1]) <= vehicle_size[1]/2+.2)
                        & (world[:,2] >= .15) & (world[:,2] <= 2.0))
                    public['truth_roi'] = dict(center_start=center.tolist(),
                        raw_count=int(roi.sum()), stage_counts={k:int((roi&m).sum()) for k,m in stages.items()},
                        point_ids=np.flatnonzero(roi).tolist(),
                        range_to_sensor=float(np.linalg.norm(center-np.asarray(sensor.latest['sensor_position_world']))))
                row['filter_audits'].append(public)
            return result
    screen.RadarTruthObserver = AuditObserver
    original_write = screen.write_json
    def write_json(path, data):
        if path.name == 'protocol.json':
            data['filter_audit'] = True
            data['robot_render_hidden_for_audit'] = experiment.hide_robot_for_audit
            data['audit_rays_per_degree_override'] = experiment.audit_rays_per_degree
        original_write(path,data)
    screen.write_json = write_json
    import diagnose_radar_motion
    diagnose_radar_motion.main()


if __name__ == '__main__':
    main()
