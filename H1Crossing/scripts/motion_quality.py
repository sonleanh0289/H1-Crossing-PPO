"""Read-only motion and sensor residual metrics. No policy inputs are modified."""
import numpy as np

def distribution(values):
    a=np.asarray(values,dtype=float)
    if not a.size:return {'samples':0}
    return {'samples':len(a),'mean':float(a.mean()),'rms':float(np.sqrt(np.mean(a*a))),
            'p95_abs':float(np.quantile(np.abs(a),.95)),'max_abs':float(np.abs(a).max()),
            'std':float(a.std())}

def speed_group(speed):
    return 'stationary_under_0.1' if speed<.1 else 'slow_0.1_to_1.5' if speed<1.5 else 'running_at_least_1.5'

def torso_sample(robot,index):
    """Isaac Lab body link pose/velocity; XYZW quaternions, world velocities."""
    def tensor(name):
        v=getattr(robot,name)
        return getattr(v,'torch',v)[:,index].detach().cpu().numpy()
    q=tensor('body_quat_w');v=tensor('body_lin_vel_w');w=tensor('body_ang_vel_w')
    x,y,z,t=q.T
    roll=np.arctan2(2*(t*x+y*z),1-2*(x*x+y*y))
    pitch=np.arcsin(np.clip(2*(t*y-z*x),-1,1))
    return [dict(torso_roll_deg=float(r*180/np.pi),torso_pitch_deg=float(p*180/np.pi),
                 torso_linear_velocity_world_mps=a.tolist(),torso_angular_velocity_world_radps=b.tolist())
            for r,p,a,b in zip(roll,pitch,v,w)]

def vibration_summary(rows):
    groups={};previous={}
    for row in rows:
        env,ep,state,terminal=row
        key=(env,ep);old=previous.get(key);previous[key]=state
        if 'torso_roll_deg' not in state or terminal or state['episode_elapsed_s']<.5:continue
        speed=np.linalg.norm(state['h1_world_velocity_xy_mps'])
        targets=[groups.setdefault(k,{}) for k in ('all_after_warmup',speed_group(speed))]
        vals={'roll_deg':state['torso_roll_deg'],'pitch_deg':state['torso_pitch_deg'],
              'roll_pitch_angular_speed_radps':float(np.linalg.norm(state['torso_angular_velocity_world_radps'][:2]))}
        if old is not None and 'torso_linear_velocity_world_mps' in old:
            dt=state['simulation_time_s']-old['simulation_time_s']
            if 0<dt<=.03:
                a=(np.array(state['torso_linear_velocity_world_mps'])-old['torso_linear_velocity_world_mps'])/dt
                vals.update(planar_acceleration_mps2=float(np.linalg.norm(a[:2])),vertical_acceleration_mps2=float(a[2]))
        for target in targets:
            for k,v in vals.items():target.setdefault(k,[]).append(v)
    return {'sample_hz':50,'warmup_excluded_s':.5,'terminal_samples_excluded':True,
            'scope':'Torso pose/angular speed and finite-difference kinematic acceleration; includes gait and commanded acceleration, not isolated high-frequency vibration or accelerometer specific force.',
            'groups':{g:{k:distribution(v) for k,v in vals.items()} for g,vals in groups.items()}}

def sensor_motion_summary(rows):
    groups={};previous={}
    definitions=[('camera_position','camera_position_valid','camera_position_world_xy_m','gt_car_position_world_xy_m'),
                 ('camera_velocity','camera_velocity_valid','camera_velocity_world_xy_mps','gt_car_velocity_world_xy_mps'),
                 ('radar_velocity','radar_velocity_valid','radar_velocity_world_xy_mps','gt_car_velocity_world_xy_mps'),
                 ('selected_velocity','selected_velocity_valid','selected_velocity_world_xy_mps','gt_car_velocity_world_xy_mps')]
    for r in rows:
        if 'h1_speed_mps' not in r:continue
        keys=['all',speed_group(r['h1_speed_mps'])]
        identity=(r['env_index'],r['episode_index'],r['vehicle_pass'],r.get('camera_track_id'))
        for name,flag,estimate,truth in definitions:
            valid=r[flag] and r.get(estimate) is not None
            error=np.array(r[estimate])-r[truth] if valid else None
            old=previous.get((r['env_index'],name))
            for k in keys:
                d=groups.setdefault(k,{}).setdefault(name,{'total':0,'valid':0,'x':[],'y':[],'norm':[],'residual_change':[]})
                d['total']+=1;d['valid']+=int(valid)
                if valid:
                    d['x'].append(error[0]);d['y'].append(error[1]);d['norm'].append(float(np.linalg.norm(error)))
                    if old and old[0]==identity and 0<r['simulation_time_s']-old[1]<=.101:
                        d['residual_change'].append(float(np.linalg.norm(error-old[2])))
            previous[(r['env_index'],name)]=(identity,r['simulation_time_s'],error) if valid else None
    result={}
    for g,sensors in groups.items():
        result[g]={name:{'total':d['total'],'valid':d['valid'],'invalid_rate':1-d['valid']/d['total'],
                        **{k:distribution(d[k]) for k in ['x','y','norm','residual_change']}} for name,d in sensors.items()}
    return {'sample_hz':10,'scope':'Current-world-truth residuals include latency, detector bias and tracking error; std is residual variability, not intrinsic sensor noise. Residual changes exclude invalid gaps, resets, vehicle respawns and camera ID changes. Invalid samples remain in coverage denominators.', 'groups':result}
