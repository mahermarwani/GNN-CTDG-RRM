import unittest

import torch

from tgnn_rrm.simulation import DynamicD2DSimulator, DynamicNetworkConfig
from tgnn_rrm.tgnn import TGNNConfig, TGNNResourceAllocator


def simulator_config():
    return DynamicNetworkConfig(
        num_entities=4,
        area_width_m=10.0,
        area_height_m=10.0,
        min_speed_mps=0.0,
        max_speed_mps=0.0,
        max_pair_distance_m=100.0,
        link_creation_probability=1.0,
        mean_link_duration_s=100.0,
        num_rbs=3,
        shadowing_std_db=0.0,
        csi_error_std=0.0,
    )


class TGNNResourceAllocatorTests(unittest.TestCase):
    def test_processes_ctdg_batch_and_returns_resource_tensors(self):
        torch.manual_seed(1)
        simulator = DynamicD2DSimulator(simulator_config(), seed=7)
        step = simulator.step()
        model = TGNNResourceAllocator(TGNNConfig(num_rbs=3, memory_dim=8, message_dim=8, embedding_dim=8))

        output = model(step.events, p_max=torch.tensor([0.1, 0.2]))

        self.assertEqual(output.active_ids, step.snapshot.active_ids)
        self.assertEqual(output.embeddings.shape, (2, 8))
        self.assertEqual(output.rb_probabilities.shape, (2, 3))
        self.assertEqual(output.rb_allocation.shape, (2, 3))
        self.assertEqual(output.power.shape, (2, 3))
        self.assertTrue(torch.allclose(output.rb_probabilities.sum(dim=1), torch.ones(2)))
        self.assertTrue(torch.equal(output.rb_allocation.sum(dim=1), torch.ones(2)))
        self.assertTrue(torch.all(output.power >= 0.0))

    def test_model_state_persists_across_chronological_steps(self):
        torch.manual_seed(2)
        simulator = DynamicD2DSimulator(simulator_config(), seed=9)
        model = TGNNResourceAllocator(TGNNConfig(num_rbs=3, memory_dim=8, message_dim=8, embedding_dim=8))

        first = model(simulator.step().events)
        second = model(simulator.step().events)

        self.assertEqual(first.active_ids, second.active_ids)
        self.assertFalse(torch.equal(first.graph_memory, second.graph_memory))

    def test_reset_memory_clears_state(self):
        torch.manual_seed(3)
        simulator = DynamicD2DSimulator(simulator_config(), seed=4)
        model = TGNNResourceAllocator(TGNNConfig(num_rbs=3, memory_dim=8, message_dim=8, embedding_dim=8))

        model(simulator.step().events)
        self.assertGreater(len(model._node_memory), 0)

        model.reset_memory()

        self.assertEqual(len(model._node_memory), 0)
        self.assertTrue(torch.equal(model.graph_memory, torch.zeros_like(model.graph_memory)))


if __name__ == "__main__":
    unittest.main()
