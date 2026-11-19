"""Run with python -m unittest discover -s scripts -p test_windows_joystick.py."""
import unittest
from types import SimpleNamespace

from windows_joystick import CommandMapper, Sample, normalize_axis


class JoystickTests(unittest.TestCase):
    def setUp(self):
        self.state = SimpleNamespace(emergency_stop=False, reset_requested=False)
        self.mapper = CommandMapper(self.state)
        self.neutral = Sample(True, {"x": 0.0, "y": 0.0})
        self.forward = Sample(True, {"x": 0.0, "y": -1.0})

    def test_axis_range_and_dead_zone(self):
        self.assertEqual(normalize_axis(32767, 0, 65535, 0.05), 0.0)
        self.assertEqual(normalize_axis(0, 0, 65535, 0.05), -1.0)
        self.assertEqual(normalize_axis(70000, 0, 65535, 0.05), 1.0)

    def test_neutral_gate_forward_reverse_and_turn(self):
        self.assertEqual(self.mapper.advance(self.forward), (0, 0, 0))
        self.mapper.advance(self.neutral)
        self.assertEqual(self.mapper.advance(self.forward), (0.6, 0, 0))
        self.assertEqual(self.mapper.advance(Sample(True, {"x": -1, "y": 1})), (0, 0, 0.5))

    def test_stop_resume(self):
        self.mapper.advance(self.neutral)
        self.assertEqual(self.mapper.advance(Sample(True, self.forward.axes, 2)), (0, 0, 0))
        self.assertTrue(self.state.emergency_stop)
        self.mapper.advance(self.forward)
        self.assertEqual(self.mapper.advance(Sample(True, self.forward.axes, 1)), (0.6, 0, 0))

    def test_disconnect_clears_command_and_requires_neutral(self):
        self.mapper.advance(self.neutral)
        self.mapper.advance(self.forward)
        self.assertEqual(self.mapper.advance(Sample(False, {})), (0, 0, 0))
        self.assertEqual(self.mapper.advance(self.forward), (0, 0, 0))
        self.mapper.advance(self.neutral)
        self.assertEqual(self.mapper.advance(self.forward), (0.6, 0, 0))

    def test_held_reset_fires_once(self):
        held = Sample(True, self.neutral.axes, 4)
        self.mapper.advance(held)
        self.assertTrue(self.state.reset_requested)
        self.state.reset_requested = False
        self.mapper.reset()
        self.mapper.advance(held)
        self.assertFalse(self.state.reset_requested)
        self.mapper.advance(self.neutral)
        self.mapper.advance(held)
        self.assertTrue(self.state.reset_requested)


if __name__ == "__main__":
    unittest.main()
