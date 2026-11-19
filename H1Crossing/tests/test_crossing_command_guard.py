from dataclasses import asdict
from pathlib import Path
from copy import deepcopy
import sys
import ast
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from behavior_control import BehaviorConfig, heading_command
from crossing_command_guard import CrossingCommandGuard
from camera_perception import Track, PerceptionConfig, Detection
from behavior_training import check_compatibility


def observation(relative=(-12., 2.), speed=2., car_speed=13.):
    obs = torch.zeros(1, 23)
    obs[0, :2] = torch.tensor(relative)/torch.tensor([50., 10.])
    obs[0, 4] = car_speed/15
    obs[0, 9] = speed/3
    obs[0, 11] = 1
    obs[0, 12] = speed/3
    obs[0, 6] = 1  # Invalid TTC is not a ten-second safety clearance.
    obs[0, 19:21] = 1
    return obs


def guard(count=1):
    return CrossingCommandGuard(count, 'cpu', BehaviorConfig(guard_enabled=1))


@pytest.mark.parametrize('missing', [19, 20])
def test_missing_sensor_blocks_forward_and_reverse_acceleration(missing):
    for v in (3., -1.):
        g=guard(); o=observation(); o[:,missing]=0
        result,info=g.apply(torch.tensor([[v,0.,.2]]),o,.02)
        assert not result.any() and info['blocked'].all()


def test_stopping_time_blocks_early_despite_invalid_policy_ttc():
    g=guard(); o=observation()
    result,info=g.apply(torch.tensor([[3.,0.,0.]]),o,.02)
    assert info['predicted_conflict'].all() and not result.any()
    assert info['guard_horizon_s'].item()>2.
    assert info['predicted_ttc_s'].item()<1.


def test_stationary_vehicle_in_crossing_path_is_not_clear():
    result,info=guard().apply(torch.tensor([[3.,0.,0.]]),observation((0.,2.),2.,0.),.02)
    assert info['predicted_conflict'].all() and not result.any()


def test_passed_vehicle_recovery_and_partial_reset():
    g=guard(2); o=observation((15.,2.)).repeat(2,1); proposal=torch.tensor([[3.,0.,0.]]).repeat(2,1)
    for _ in range(15):result,info=g.apply(proposal,o,.02)
    torch.testing.assert_close(result,proposal)
    g.reset(torch.tensor([0])); result,info=g.apply(proposal,o,.02)
    assert not result[0].any() and result[1,0]==3
    o[1,20]=0; result,_=g.apply(proposal,o,.02)
    assert not result[1].any()


def test_observation_not_mutated_and_no_gt_argument():
    o=observation(); before=o.clone()
    guard().apply(torch.tensor([[3.,0.,0.]]),o,.02)
    torch.testing.assert_close(o,before)


def test_guard_reset_after_inference_rollout_outside_inference_mode():
    g=guard(2)
    with torch.inference_mode():
        for _ in range(15):
            g.apply(torch.tensor([[3.,0.,0.]]).repeat(2,1), observation((15.,2.)).repeat(2,1), .02)
    # The final checkpoint reload calls env.reset outside InferenceMode.
    g.reset(torch.tensor([0]))
    assert g.clear_seconds[0] == 0 and g.clear_seconds[1] > 0


def measurement(x):
    return Detection(np.array([x,-1.75]),.01,(0,0,10,10),100)


def test_motion_gate_confirms_birth_and_rejects_jump_without_replacing_identity():
    t=Track(PerceptionConfig(track_quality='legacy_v1',motion_gate=1))
    for i in range(3):
        t.update(i/30,measurement(-10+i*.4))
        assert t.estimate(i/30)[1] == (i==2)
    identity=t.track_id; last=t.last_measurement
    # A one-frame 1.8 m jump: below legacy 2 m innovation threshold,
    # but exceeds measurement speed bound at 30 Hz.
    t.update(.1,measurement(-7.4))
    assert t.last_update_status=='measurement_motion_jump'
    assert t.last_measurement==last and t.track_id==identity
    t.update(4/30,measurement(-8.4))
    assert t.last_update_status=='accepted' and t.track_id==identity
    t.reset();assert t.last_detection_position is None and not t.estimate(.2)[1]


def test_guard_migration_allows_play_but_rejects_resume():
    old={'format':'h1-crossing-camera-radar-v1','settings':asdict(BehaviorConfig())}
    new=deepcopy(old);new['settings']['guard_enabled']=1
    with pytest.warns(UserWarning):check_compatibility(old,new,training=False)
    with pytest.raises(ValueError,match='resume'):check_compatibility(old,new,training=True)
    assert old['settings']['guard_enabled']==0


def test_real_environment_guard_integration():
    tree=ast.parse((ROOT/'scripts/behavior_environment.py').read_text())
    blocks=[n for n in ast.walk(tree) if isinstance(n,ast.If)
            and 'self.settings.guard_enabled' in ast.unparse(n.test)
            and 'self.perception' in ast.unparse(n.test)]
    assert len(blocks)==1
    o=observation();o[:,20]=0
    env=NS(perception=True,settings=BehaviorConfig(guard_enabled=1),command_guard=guard(),
           base=NS(step_dt=.02),goal_xy=torch.tensor([[0.,4.6]]))
    robot=NS(root_pos_w=NS(torch=torch.tensor([[0.,-3.,1.]])),root_quat_w=NS(torch=torch.tensor([[0.,0.,.70710678,.70710678]])))
    ctx=dict(self=env,robot=robot,manual_command=None,speed=torch.tensor([3.]),sensor_obs=o,
             active=torch.tensor([True]),guard_log={},torch=torch,heading_command=heading_command)
    exec(compile(ast.fix_missing_locations(ast.Module(body=blocks,type_ignores=[])),'guard_integration','exec'),ctx)
    assert not ctx['desired'].any() and ctx['guard_log']['missing_velocity'].all()
