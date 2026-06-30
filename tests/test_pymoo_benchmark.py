import unittest

import torch

from tgnn_rrm.pymoo_benchmark import candidate_to_allocation_power


class PymooBenchmarkUtilityTests(unittest.TestCase):
    def test_candidate_decodes_to_feasible_allocation_and_power(self):
        candidate = [
            0.9,
            0.1,
            0.2,
            0.3,
            0.7,
            0.4,
            0.5,
            0.25,
            0.75,
            0.4,
            0.6,
            0.2,
        ]

        allocation, power = candidate_to_allocation_power(
            candidate=candidate,
            num_links=2,
            num_rbs=3,
            max_rbs_per_link=1,
            p_max=torch.tensor([0.1, 0.2]),
        )

        self.assertEqual(allocation.shape, (2, 3))
        self.assertEqual(power.shape, (2, 3))
        self.assertTrue(torch.equal(allocation.sum(dim=1), torch.ones(2)))
        self.assertTrue(torch.all(power >= 0.0))
        self.assertTrue(torch.all(power <= allocation * torch.tensor([[0.1], [0.2]])))
        self.assertTrue(torch.all(power.sum(dim=1) <= torch.tensor([0.1, 0.2])))

    def test_candidate_length_is_validated(self):
        with self.assertRaisesRegex(ValueError, "candidate length"):
            candidate_to_allocation_power(
                candidate=[0.0],
                num_links=1,
                num_rbs=1,
                max_rbs_per_link=1,
                p_max=0.1,
            )


if __name__ == "__main__":
    unittest.main()
