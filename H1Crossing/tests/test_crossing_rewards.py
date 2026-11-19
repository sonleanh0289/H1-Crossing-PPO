"""CPU-only behavior reward checks; no physics, training or policy changes."""
from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from crossing_rewards import CrossingRewardConfig, CrossingRewardTracker, load_reward_config


class RewardTests(unittest.TestCase):
    def make(self, n=1, cfg=None):
        # Legacy event/stationary tests isolate those terms; progress tests opt in.
        tracker = CrossingRewardTracker(n, cfg or replace(CrossingRewardConfig(), progress_reward_per_m=0), dtype=torch.float64)
        position = torch.zeros((n, 2), dtype=torch.float64)
        goal = position + torch.tensor([0., 9.2], dtype=torch.float64)
        tracker.reset(position)
        return tracker, position, goal

    def step(self, tracker, position, goal, dt=.1, fallen=False, contact=False, timeout=False):
        def flag(value):
            return torch.as_tensor(value, dtype=torch.bool).expand(len(position)).clone()
        return tracker.step(position, goal, dt=dt, fallen=flag(fallen),
                            vehicle_contact=flag(contact), timeout=flag(timeout))

    def test_yaml_matches_documented_defaults(self):
        cfg = load_reward_config()
        self.assertEqual(cfg, CrossingRewardConfig())
        self.assertEqual((cfg.goal_reward, cfg.safe_episode_bonus, cfg.vehicle_collision_penalty), (1, 1, -2))

    def test_no_per_step_survival_reward(self):
        tracker, position, goal = self.make()
        for _ in range(20):
            result = self.step(tracker, position, goal)
            self.assertAlmostEqual(result["total"].item(), 0, places=12)
            self.assertEqual(result["safe_episode"].item(), 0)

    def test_goal_is_plus_two_once(self):
        tracker, position, goal = self.make()
        result = self.step(tracker, goal, goal)
        self.assertEqual(result["goal"].item(), 1)
        self.assertEqual(result["safe_episode"].item(), 1)
        self.assertEqual(result["total"].item(), 2)
        self.assertTrue(result["success"].item())
        self.assertTrue(result["terminated"].item())
        again = self.step(tracker, goal, goal)
        self.assertEqual(again["total"].item(), 0)
        self.assertFalse(again["terminated"].item())

    def test_goal_requires_xy_proximity_not_just_y(self):
        tracker, position, goal = self.make()
        position[:] = goal + torch.tensor([2., 0.])
        result = self.step(tracker, position, goal)
        self.assertFalse(result["success"].item())
        self.assertEqual(result["goal"].item(), 0)

    def test_collision_overrides_goal_and_safe_timeout(self):
        tracker, position, goal = self.make()
        result = self.step(tracker, goal, goal, contact=True, timeout=True)
        self.assertEqual(result["total"].item(), -2)
        self.assertEqual(result["goal"].item(), 0)
        self.assertEqual(result["safe_episode"].item(), 0)
        self.assertFalse(result["success"].item())
        self.assertFalse(result["truncated"].item())
        self.assertEqual(self.step(tracker, goal, goal, contact=True)["total"].item(), 0)

    def test_fall_has_no_unrequested_penalty_and_blocks_positive_rewards(self):
        tracker, position, goal = self.make()
        result = self.step(tracker, goal, goal, fallen=True, timeout=True)
        self.assertEqual(result["total"].item(), 0)
        self.assertTrue(result["terminated"].item())
        self.assertEqual(result["safe_episode"].item(), 0)
        self.assertEqual(result["goal"].item(), 0)

    def test_timeout_penalty_without_safe_bonus_once(self):
        tracker, position, goal = self.make()
        result = self.step(tracker, position, goal, timeout=True)
        self.assertEqual(result["total"].item(), -1)
        self.assertEqual(result["safe_episode"].item(), 0)
        self.assertEqual(result["timeout"].item(), -1)
        self.assertFalse(result["terminated"].item())
        self.assertTrue(result["truncated"].item())
        self.assertEqual(self.step(tracker, position, goal, timeout=True)["total"].item(), 0)

    def test_stationary_ramp_and_cap_exact_integrals(self):
        tracker, position, goal = self.make()
        totals = [self.step(tracker, position, goal, dt=1.)["stationary"].item() for _ in range(10)]
        expected = [0, 0, -.025, -.075, -.125, -.175, -.225, -.25, -.25, -.25]
        for observed, desired in zip(totals, expected):
            self.assertAlmostEqual(observed, desired, places=12)
        self.assertAlmostEqual(sum(totals), -1.375)

    def test_stationary_cost_independent_of_update_rate(self):
        costs = []
        for dt in (.02, .1, .25, 1.):
            tracker, position, goal = self.make()
            cost = sum(self.step(tracker, position, goal, dt=dt)["total"].item() for _ in range(round(10/dt)))
            costs.append(cost)
        for cost in costs:
            self.assertAlmostEqual(cost, -1.375, places=9)

    def test_terminal_events_not_multiplied_by_dt(self):
        for dt in (.02, .1, 1.):
            tracker, position, goal = self.make()
            self.assertEqual(self.step(tracker, goal, goal, dt=dt)["total"].item(), 2)

    def test_jitter_does_not_reset_timer_but_real_displacement_does(self):
        tracker, position, goal = self.make()
        for i in range(40):
            position[0, 0] = .03 if i % 2 else -.03
            result = self.step(tracker, position, goal)
        self.assertAlmostEqual(result["stationary_elapsed_s"].item(), 4.)
        self.assertLess(result["stationary"].item(), 0)
        position[0, 0] = .11
        result = self.step(tracker, position, goal)
        self.assertEqual(result["stationary_elapsed_s"].item(), 0)
        self.assertEqual(result["stationary"].item(), 0)

    def test_accumulated_slow_displacement_counts(self):
        tracker, position, goal = self.make()
        for i in range(11):
            position[0, 0] = (i+1)*.01
            self.step(tracker, position, goal)
        self.assertLessEqual(tracker.stationary_elapsed_s.item(), .11)

    def test_batch_selective_reset_and_terminal_latching(self):
        tracker, position, goal = self.make(3)
        self.step(tracker, position, goal, dt=3., contact=[True, False, False])
        tracker.reset(position, env_ids=torch.tensor([0, 2]))
        torch.testing.assert_close(tracker.stationary_elapsed_s, torch.tensor([0., 3., 0.], dtype=torch.float64))
        result = self.step(tracker, position, goal, contact=[True, False, False])
        self.assertEqual(result["vehicle_collision"].tolist(), [-2., 0., 0.])
        self.assertLess(result["stationary"][1].item(), 0)
        self.assertEqual(result["stationary"][2].item(), 0)

    def test_translation_invariance(self):
        tracker, position, goal = self.make(2)
        offset = torch.tensor([[100., -50.], [-70., 80.]], dtype=torch.float64)
        tracker.reset(position + offset)
        result = self.step(tracker, goal + offset, goal + offset)
        self.assertEqual(result["total"].tolist(), [2, 2])

    def test_reward_terms_sum_and_penalty_can_be_disabled(self):
        for key in ("stationary_penalty_growth_per_s2", "stationary_max_penalty_per_s"):
            tracker, position, goal = self.make(cfg=replace(CrossingRewardConfig(), **{key: 0}))
            result = self.step(tracker, position, goal, dt=20., timeout=True)
            self.assertEqual(result["stationary"].item(), 0)
            terms = sum(result[k] for k in ("goal", "safe_episode", "vehicle_collision", "fall", "stationary", "timeout", "progress"))
            torch.testing.assert_close(terms, result["total"])

    def test_invalid_config_and_unknown_yaml_keys(self):
        for kwargs in ({"goal_reward": math.nan}, {"safe_episode_bonus": True},
                       {"vehicle_collision_penalty": 2}, {"stationary_grace_s": -1}, {"goal_radius_m": 0},
                       {"timeout_penalty": 1}, {"progress_reward_per_m": -1}, {"progress_reward_per_m": math.inf}):
            with self.assertRaises(ValueError):
                CrossingRewardConfig(**kwargs)
        for text in ("typo_reward: 1", "[]", "goal_reward: bad"):
            with patch.object(Path, "read_text", return_value=text), self.assertRaises(ValueError):
                load_reward_config()


    def test_progress_closer_farther_and_no_proximity_farming(self):
        tracker, position, goal = self.make(cfg=CrossingRewardConfig())
        position[:, 1] = 1.
        self.assertAlmostEqual(self.step(tracker, position, goal)["progress"].item(), .5)
        self.assertEqual(self.step(tracker, position, goal)["progress"].item(), 0)
        position[:, 1] = 0.
        self.assertAlmostEqual(self.step(tracker, position, goal)["progress"].item(), -.5)

    def test_progress_timestep_independence_and_origin_offsets(self):
        for subdivisions in (1, 5, 50):
            tracker, position, goal = self.make(2, cfg=CrossingRewardConfig())
            offset = torch.tensor([[120., 0.], [-120., 120.]], dtype=torch.float64)
            position += offset
            goal += offset
            tracker.reset(position)
            reward = torch.zeros(2, dtype=torch.float64)
            for i in range(subdivisions):
                position[:, 1] = offset[:, 1] + (i+1)/subdivisions
                reward += self.step(tracker, position, goal, dt=1/subdivisions)["progress"]
            torch.testing.assert_close(reward, torch.tensor([.5, .5], dtype=torch.float64))

    def test_progress_partial_reset_has_no_cross_episode_jump(self):
        tracker, position, goal = self.make(2, cfg=CrossingRewardConfig())
        position[:, 1] = 1
        self.step(tracker, position, goal)
        position[0, 1] = 0
        tracker.reset(position, env_ids=torch.tensor([0]))
        result = self.step(tracker, position, goal)
        self.assertEqual(result["progress"].tolist(), [0., 0.])

    def test_progress_suppressed_on_failure_and_after_terminal(self):
        for failure in ("contact", "fallen"):
            tracker, position, goal = self.make(cfg=CrossingRewardConfig())
            position[:, 1] = 1.
            result = self.step(tracker, position, goal, **{failure: True})
            self.assertEqual(result["progress"].item(), 0)
            self.assertEqual(self.step(tracker, goal, goal)["total"].item(), 0)

    def test_progress_negative_on_failed_retreat(self):
        tracker, position, goal = self.make(cfg=CrossingRewardConfig())
        position[:, 1] = -1.
        result = self.step(tracker, position, goal, contact=True)
        self.assertEqual(result["progress"].item(), -.5)
        self.assertEqual(result["total"].item(), -2.5)

    def test_goal_timeout_precedence_and_total_includes_progress(self):
        tracker, position, goal = self.make(cfg=CrossingRewardConfig())
        result = self.step(tracker, goal, goal, timeout=True)
        self.assertEqual(result["timeout"].item(), 0)
        self.assertEqual(result["safe_episode"].item(), 1)
        self.assertAlmostEqual(result["progress"].item(), 4.6)
        self.assertAlmostEqual(result["total"].item(), 6.6)

    def test_timeout_penalty_and_progress_have_independent_coefficients(self):
        tracker, position, goal = self.make(cfg=replace(CrossingRewardConfig(), timeout_penalty=-3., progress_reward_per_m=2.))
        position[:, 1] = .5
        result = self.step(tracker, position, goal, timeout=True)
        self.assertEqual(result["safe_episode"].item(), 0)
        self.assertAlmostEqual(result["progress"].item(), 1)
        self.assertAlmostEqual(result["total"].item(), -2)


    def test_requires_reset_and_rejects_bad_inputs(self):
        tracker, position, goal = self.make()
        tracker.initialized.zero_()
        with self.assertRaises(RuntimeError):
            self.step(tracker, position, goal)
        tracker.reset(position)
        for dt in (0, -1, math.inf, math.nan, True):
            with self.assertRaises(ValueError):
                self.step(tracker, position, goal, dt=dt)
        position[0, 0] = math.nan
        with self.assertRaises(ValueError):
            self.step(tracker, position, goal)


if __name__ == "__main__":
    unittest.main()
