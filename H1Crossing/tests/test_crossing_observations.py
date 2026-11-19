"""CPU tests for raw behavior state, footprint TTC, origin/reset and group isolation."""
from pathlib import Path
import math
import sys
from types import SimpleNamespace as NS, ModuleType
import unittest
from unittest.mock import patch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from crossing_observations import (
    OBSERVATION_DIM, OBSERVATION_NAMES, planar_ttc, compute_crossing_observation,
    crossing_state, set_scripted_vehicle_velocity, add_crossing_observation_group,
)
from play_crossing_finetuned import parse_args


def xy(rows):
    return torch.tensor(rows, dtype=torch.float32)


class ObservationTests(unittest.TestCase):
    def test_approach_uses_front_extent_not_center_distance(self):
        ttc, valid = planar_ttc(xy([[-10, 0]]), xy([[2, 0]]))
        self.assertTrue(valid.item())
        self.assertAlmostEqual(ttc.item(), (10 - 2.1 - .35) / 2, places=5)

    def test_sidewalk_miss_receding_parallel_and_parked(self):
        pos = xy([[-20, 2.85], [-10, 0], [-10, 0], [-10, 3]])
        vel = xy([[8, 0], [-2, 0], [0, 0], [2, 0]])
        ttc, valid = planar_ttc(pos, vel)
        self.assertFalse(valid.any())
        torch.testing.assert_close(ttc, torch.full((4, 1), 10.))

    def test_overlap_and_touch_are_zero_even_when_receding(self):
        ttc, valid = planar_ttc(xy([[0, 0], [1, 0], [2.45, 0]]), xy([[0, 0], [8, 0], [0, 0]]))
        self.assertTrue(valid.all())
        torch.testing.assert_close(ttc, torch.zeros((3, 1)))

    def test_crossing_requires_overlapping_time_intervals(self):
        ttc, valid = planar_ttc(xy([[-10, 2], [-10, 5]]), xy([[4, -1], [4, -.5]]))
        self.assertEqual(valid[:, 0].tolist(), [True, False])
        self.assertAlmostEqual(ttc[0].item(), 1.8875, places=5)

    def test_horizon_and_near_zero_relative_speed(self):
        ttc, valid = planar_ttc(xy([[-22.45, 0], [-22.46, 0], [-10, 0]]), xy([[2, 0], [2, 0], [1e-9, 0]]))
        self.assertEqual(valid[:, 0].tolist(), [True, False, False])
        self.assertTrue(torch.isfinite(ttc).all())
        torch.testing.assert_close(ttc, torch.full((3, 1), 10.))

    def test_h1_velocity_is_in_ttc_and_no_z_or_joint_state(self):
        obs = compute_crossing_observation(xy([[0, 0]]), xy([[-2, 0]]),
                                           xy([[-10, 0]]), xy([[0, 0]]), xy([[0, 4.6]]))
        self.assertEqual(tuple(obs.shape), (1, OBSERVATION_DIM))
        self.assertEqual(OBSERVATION_DIM, 8)
        torch.testing.assert_close(obs[:, :6], xy([[-10, 0, 0, 4.6, 0, 0]]))
        self.assertAlmostEqual(obs[0, 6].item(), 3.775, places=5)
        self.assertEqual(obs[0, 7], 1)

    def test_same_velocity_not_imminent_collision(self):
        obs = compute_crossing_observation(xy([[0, 0]]), xy([[8, 0]]),
                                           xy([[-10, 0]]), xy([[8, 0]]), xy([[0, 4.6]]))
        self.assertEqual(obs[0, 6:].tolist(), [10, 0])

    def test_translation_invariance_and_batch_independence(self):
        p, v, car, cv, goal = xy([[0, -4.6], [0, -1.75]]), xy([[0, 0], [0, 0]]), xy([[-20, -1.75]] * 2), xy([[8, 0]] * 2), xy([[0, 4.6]] * 2)
        reference = compute_crossing_observation(p, v, car, cv, goal)
        origins = xy([[100, -50], [-20, 70]])
        translated = compute_crossing_observation(p + origins, v, car + origins, cv, goal + origins)
        torch.testing.assert_close(translated, reference)
        self.assertEqual(reference[:, 7].tolist(), [0, 1])
        for row in range(2):
            one = compute_crossing_observation(*(x[row:row+1] for x in (p, v, car, cv, goal)))
            torch.testing.assert_close(one, reference[row:row+1])

    def test_invalid_state_and_config_rejected(self):
        for bad in (xy([[math.nan, 0]]), xy([[math.inf, 0]]), torch.zeros(1, 3), torch.zeros(1, 2, dtype=torch.int64)):
            with self.assertRaises(ValueError):
                planar_ttc(bad, xy([[1, 0]]))
        for kwargs in ({"horizon_s": 0}, {"h1_radius_m": -1}, {"vehicle_size_xy": (0, 2)}):
            with self.assertRaises(ValueError):
                planar_ttc(xy([[0, 0]]), xy([[1, 0]]), **kwargs)

    def test_adapter_origin_and_scripted_velocity_after_teleport(self):
        origins = torch.tensor([[100., 50., 0.], [-40., -20., 0.]])
        robot_pos = origins + torch.tensor([0., -4.6, 1.2])
        car_pos = origins + torch.tensor([-20., -1.75, .68])
        robot = NS(data=NS(root_pos_w=NS(torch=robot_pos), root_lin_vel_w=NS(torch=torch.zeros(2, 3))))
        car = NS(data=NS(root_pos_w=NS(torch=car_pos), root_lin_vel_w=NS(torch=torch.full((2, 3), 9999.))))
        class Scene(dict):
            env_origins = origins
        env = NS(scene=Scene(robot=robot, vehicle=car))
        params = dict(goal_position_local=(0, 4.6, .017), vehicle_size_xy=(4.2, 1.8))
        self.assertTrue((crossing_state(env, **params)[:, 4:6] == 0).all())  # shape probe
        speed = xy([[8, 0], [12, 0]])
        set_scripted_vehicle_velocity(env, speed)
        speed.zero_()  # publisher owns a copy
        first = crossing_state(env, **params)
        torch.testing.assert_close(first[:, :4], xy([[-20, 2.85, 0, 9.2]] * 2))
        torch.testing.assert_close(first[:, 4:6], xy([[8, 0], [12, 0]]))
        car_pos[1, 0] = origins[1, 0] - 45  # only second env respawns
        set_scripted_vehicle_velocity(env, xy([[8, 0], [13, 0]]))
        second = crossing_state(env, **params)
        torch.testing.assert_close(second[0], first[0])
        self.assertEqual(second[1, 0].item(), -45)
        self.assertEqual(second[1, 4].item(), 13)

    def test_factory_preserves_policy_and_has_no_camera(self):
        module = ModuleType("isaaclab.managers")
        module.ObservationGroupCfg = NS
        module.ObservationTermCfg = NS
        policy = object()
        cfg = NS(observations=NS(policy=policy))
        with patch.dict(sys.modules, {"isaaclab.managers": module}):
            add_crossing_observation_group(cfg, NS(goal_position=(0, 4.6, .017), vehicle_size=(4.2, 1.8, .65)))
        self.assertIs(cfg.observations.policy, policy)
        self.assertFalse(cfg.observations.crossing.enable_corruption)
        self.assertIs(cfg.observations.crossing.state.func, crossing_state)
        self.assertEqual(list(vars(cfg.observations)), ["policy", "crossing"])
        self.assertEqual(OBSERVATION_NAMES[-2:], ("ttc_s", "ttc_valid"))

    def test_recording_is_opt_in(self):
        self.assertFalse(parse_args(["--headless"]).record_crossing_observations)
        self.assertTrue(parse_args(["--headless", "--record_crossing_observations"]).record_crossing_observations)


if __name__ == "__main__":
    unittest.main()
