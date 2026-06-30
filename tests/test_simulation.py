import unittest

import torch

from tgnn_rrm.ctdg import EventType
from tgnn_rrm.simulation import DynamicD2DSimulator, DynamicNetworkConfig


def small_network_config(**overrides):
    values = {
        "num_entities": 4,
        "area_width_m": 10.0,
        "area_height_m": 10.0,
        "time_step_s": 0.1,
        "min_speed_mps": 0.0,
        "max_speed_mps": 0.0,
        "max_pair_distance_m": 100.0,
        "link_creation_probability": 1.0,
        "mean_link_duration_s": 100.0,
        "num_rbs": 3,
        "shadowing_std_db": 0.0,
        "fading_correlation": 0.0,
        "csi_error_std": 0.0,
        "csi_error_correlation": 0.0,
    }
    values.update(overrides)
    return DynamicNetworkConfig(**values)


class DynamicD2DSimulatorTests(unittest.TestCase):
    def test_step_returns_active_links_gains_and_ctdg_events(self):
        simulator = DynamicD2DSimulator(small_network_config(), seed=7)

        step = simulator.step()

        self.assertEqual(step.snapshot.time, 0.0)
        self.assertEqual(len(step.snapshot.active_links), 2)
        self.assertEqual(step.snapshot.gains.shape, (2, 2, 3))
        self.assertEqual(step.events.active_ids, step.snapshot.active_ids)
        self.assertTrue(all(event.event_type == EventType.ADD for event in step.events.events))

    def test_existing_links_emit_update_events_on_next_step(self):
        simulator = DynamicD2DSimulator(small_network_config(), seed=7)

        first = simulator.step()
        second = simulator.step()

        self.assertEqual(second.events.added_ids, ())
        self.assertEqual(second.events.updated_ids, first.snapshot.active_ids)
        self.assertTrue(all(event.event_type == EventType.UPDATE for event in second.events.events))

    def test_seeded_simulation_is_deterministic(self):
        first = DynamicD2DSimulator(small_network_config(), seed=11).step()
        second = DynamicD2DSimulator(small_network_config(), seed=11).step()

        self.assertEqual(first.snapshot.active_ids, second.snapshot.active_ids)
        self.assertTrue(torch.equal(first.snapshot.gains, second.snapshot.gains))

    def test_empty_network_has_empty_csi_tensor(self):
        simulator = DynamicD2DSimulator(small_network_config(num_entities=0), seed=3)

        step = simulator.step()

        self.assertEqual(step.snapshot.active_links, ())
        self.assertEqual(step.snapshot.gains.shape, (0, 0, 3))
        self.assertEqual(step.events.events, ())

    def test_fully_correlated_static_channel_is_stable(self):
        simulator = DynamicD2DSimulator(
            small_network_config(fading_correlation=1.0, csi_error_correlation=1.0),
            seed=5,
        )

        first = simulator.step()
        second = simulator.step()

        self.assertEqual(first.snapshot.active_ids, second.snapshot.active_ids)
        self.assertTrue(torch.equal(first.snapshot.gains, second.snapshot.gains))

    def test_rejects_invalid_channel_correlation(self):
        with self.assertRaisesRegex(ValueError, "fading_correlation"):
            DynamicD2DSimulator(small_network_config(fading_correlation=1.1), seed=1)


if __name__ == "__main__":
    unittest.main()
