import unittest

import torch

from tgnn_rrm.config import RadioConfig
from tgnn_rrm.objectives import RRMObjectiveConfig, torch_link_rates, unsupervised_rrm_loss
from tgnn_rrm.radio import link_rates
from tgnn_rrm.simulation import DynamicD2DSimulator, DynamicNetworkConfig
from tgnn_rrm.tgnn import TGNNConfig, TGNNResourceAllocator


class TorchRadioObjectiveTests(unittest.TestCase):
    def test_torch_rates_match_reference_radio_equation(self):
        config = RadioConfig(num_rbs=2, rb_bandwidth_hz=1.0, noise_power_watt=1.0)
        gains = torch.tensor(
            [
                [[3.0, 2.0], [1.0, 0.5]],
                [[0.25, 1.0], [4.0, 2.0]],
            ]
        )
        allocation = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        power = torch.tensor([[1.0, 0.0], [0.5, 1.0]])

        expected = link_rates(gains.tolist(), allocation.tolist(), power.tolist(), config)
        actual = torch_link_rates(gains, allocation, power, config)

        self.assertTrue(torch.allclose(actual, torch.tensor(expected), atol=1e-6))

    def test_unsupervised_loss_backpropagates_through_soft_allocation_and_power(self):
        config = RadioConfig(num_rbs=2, rb_bandwidth_hz=1.0, noise_power_watt=1.0, min_rate_bps=0.1)
        gains = torch.tensor([[[3.0, 2.0], [0.5, 0.25]], [[0.25, 0.5], [2.0, 3.0]]])
        allocation_logits = torch.randn((2, 2), requires_grad=True)
        power_logits = torch.randn((2, 2), requires_grad=True)
        allocation = torch.softmax(allocation_logits, dim=1)
        power = torch.sigmoid(power_logits) * 0.1

        result = unsupervised_rrm_loss(
            gains=gains,
            allocation=allocation,
            power=power,
            radio_config=config,
            p_max=0.1,
            objective_config=RRMObjectiveConfig(rate_scale_bps=1.0),
        )
        result.loss.backward()

        self.assertTrue(torch.isfinite(result.loss))
        self.assertIsNotNone(allocation_logits.grad)
        self.assertIsNotNone(power_logits.grad)


class MinimalTrainingStepTests(unittest.TestCase):
    def test_one_unsupervised_optimizer_step_updates_model_parameters(self):
        torch.manual_seed(5)
        simulator = DynamicD2DSimulator(
            DynamicNetworkConfig(
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
            ),
            seed=7,
        )
        step = simulator.step()
        model = TGNNResourceAllocator(TGNNConfig(num_rbs=3, memory_dim=8, message_dim=8, embedding_dim=8))
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        before = [parameter.detach().clone() for parameter in model.parameters()]

        output = model(step.events, p_max=0.1)
        result = unsupervised_rrm_loss(
            gains=step.snapshot.gains,
            allocation=output.rb_probabilities,
            power=output.power,
            radio_config=RadioConfig(num_rbs=3, min_rate_bps=0.0),
            p_max=0.1,
        )
        result.loss.backward()
        optimizer.step()

        after = list(model.parameters())
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, after)))


if __name__ == "__main__":
    unittest.main()
