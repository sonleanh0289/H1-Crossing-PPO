"""Physical integration probes run ONLY with --smoke_test, before PPO updates."""
import torch


@torch.no_grad()
def verify_physical_events(env):
    if env.num_envs != 4:
        raise ValueError("Physical smoke requires exactly four isolated environments")
    zero = torch.zeros((4, 1), device=env.device)
    env.reset()
    for _ in range(2):
        _, _, _, extra = env.step(zero)
        assert not extra["events"]["collision"].any(), "Ground contact leaked into vehicle filter"
    pose = env.robot.data.root_pose_w.torch.clone()
    pose[0, :2] = env.goal_xy[0]
    pose[1, 2] = env.origins[1, 2] + .2
    pose[3, 0] = env.origins[3, 0] + 60
    env.robot.write_root_pose_to_sim_index(root_pose=pose)
    env.elapsed[2] = env.settings.episode_seconds - .01
    obs, reward, done, extra = env.step(zero)
    assert done.tolist() == [1, 1, 1, 1], f"Missing terminal event: {done.tolist()}"
    assert extra["events"]["success"][0] and extra["reward_terms"]["goal"][0] == 1
    assert extra["reward_terms"]["safe_episode"][0] == 1
    assert extra["reward_terms"]["progress"][0] > 0  # test-only teleport reduces distance
    torch.testing.assert_close(reward, sum(extra["reward_terms"].values()))
    assert extra["events"]["fall"][1] and extra["reward_terms"]["safe_episode"][1] == 0
    assert extra["events"]["timeout"][2] and extra["reward_terms"]["safe_episode"][2] == 0
    assert extra["reward_terms"]["timeout"][2] == env.reward_cfg.timeout_penalty
    assert extra["reward_terms"]["timeout"][[0, 1, 3]].eq(0).all()
    assert extra["events"]["out_of_bounds"][3] and extra["reward_terms"]["safe_episode"][3] == 0
    assert "time_outs" not in extra, "Finite horizon must not bootstrap across safe timeout"
    assert torch.allclose(env.robot.data.root_pos_w.torch[:, :2]-env.origins[:, :2],
                          torch.tensor([[0., -4.6]], device=env.device).expand(4, 2), atol=1e-4)
    assert (env.elapsed == 0).all() and (env.tracker.stationary_elapsed_s == 0).all()
    torch.testing.assert_close(env.tracker.previous_xy, env.robot.data.root_pos_w.torch[:, :2])
    assert tuple(obs["behavior"].shape) == (4, 19)
    env.reset()
    env.traffic.x[0] = -3.
    env.traffic.speed[0] = 8.
    env.vehicle_y[0] = -4.6  # put actual car on the standing H1's path, test only
    env.place_vehicle()
    contact_force = 0.
    for step in range(20):
        obs, reward, done, extra = env.step(zero)
        contact_force = max(contact_force, float(extra["contact_force_peak_n"][0]))
        assert not extra["events"]["collision"][1:].any(), "Contact leaked across environments"
        if extra["events"]["collision"][0]:
            assert done[0] and extra["reward_terms"]["vehicle_collision"][0] == -2
            assert extra["reward_terms"]["safe_episode"][0] == 0 and extra["reward_terms"]["goal"][0] == 0
            assert contact_force > env.settings.contact_threshold_n
            # Earlier safe substeps may earn progress before contact is detected.
            torch.testing.assert_close(reward, sum(extra["reward_terms"].values()))
            assert env.vehicle_y[0] == -1.75 and env.elapsed[0] == 0
            env.reset()
            print("[Physical smoke] goal/fall/timeout/OOB/reset and real H1-car contact passed.", flush=True)
            return {"passed": True, "contact_force_peak_n": contact_force, "contact_behavior_step": step+1,
                    "checks": ["goal event +2 once", "fall blocks bonus", "timeout penalty without bonus", "OOB no bonus",
                               "progress included in total", "progress state reset",
                               "pre-reset terminal reward", "selective reset", "actual filtered contact -2",
                               "no ground/cross-env false collision", "finite-horizon no bootstrap"]}
    raise AssertionError("No real H1-car contact detected; refusing training smoke acceptance")
