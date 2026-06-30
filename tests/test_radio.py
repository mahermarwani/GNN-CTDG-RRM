import unittest

from tgnn_rrm.config import RadioConfig
from tgnn_rrm.radio import constraint_violations, link_rates, qos_violation_fraction


class RadioEquationTests(unittest.TestCase):
    def test_single_link_rate_without_interference(self):
        config = RadioConfig(num_rbs=1, rb_bandwidth_hz=1.0, noise_power_watt=1.0)
        gains = [[[3.0]]]
        allocation = [[1.0]]
        power = [[1.0]]

        self.assertAlmostEqual(link_rates(gains, allocation, power, config)[0], 2.0)

    def test_cochannel_interference_reduces_rate(self):
        config = RadioConfig(num_rbs=1, rb_bandwidth_hz=1.0, noise_power_watt=1.0)
        gains = [
            [[3.0], [1.0]],
            [[1.0], [3.0]],
        ]

        no_interference = link_rates(gains, [[1.0], [0.0]], [[1.0], [0.0]], config)[0]
        with_interference = link_rates(gains, [[1.0], [1.0]], [[1.0], [1.0]], config)[0]

        self.assertLess(with_interference, no_interference)

    def test_qos_violation_fraction(self):
        self.assertEqual(qos_violation_fraction([], 10.0), 0.0)
        self.assertAlmostEqual(qos_violation_fraction([5.0, 10.0, 15.0], 10.0), 1 / 3)


class ConstraintTests(unittest.TestCase):
    def test_valid_allocation_has_no_violations(self):
        config = RadioConfig(num_rbs=2, max_rbs_per_link=1)
        violations = constraint_violations(
            allocation=[[1.0, 0.0], [0.0, 1.0]],
            power=[[0.1, 0.0], [0.0, 0.2]],
            p_max=[0.1, 0.2],
            config=config,
        )

        self.assertEqual(violations.total, 0)

    def test_detects_constraint_violations(self):
        config = RadioConfig(num_rbs=2, max_rbs_per_link=1)
        violations = constraint_violations(
            allocation=[[1.0, 1.0], [0.0, 1.0]],
            power=[[0.1, 0.2], [0.1, -0.1]],
            p_max=[0.2, 0.2],
            config=config,
        )

        self.assertEqual(violations.rb_budget, 1)
        self.assertGreaterEqual(violations.power_budget, 1)
        self.assertEqual(violations.inactive_power, 1)
        self.assertEqual(violations.negative_power, 1)


if __name__ == "__main__":
    unittest.main()
