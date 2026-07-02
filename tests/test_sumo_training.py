import math
import tempfile
import textwrap
import unittest
from pathlib import Path

import torch

from tgnn_rrm.config import RadioConfig
from tgnn_rrm.objectives import RRMObjectiveConfig
from tgnn_rrm.simulation import DynamicNetworkConfig
from tgnn_rrm.sumo import load_sumo_fcd_trace
from tgnn_rrm.sumo_training import train_sumo_unsupervised
from tgnn_rrm.tgnn import TGNNConfig, TGNNResourceAllocator


def write_trace(xml: str) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False)
    with handle:
        handle.write(textwrap.dedent(xml).strip())
    return Path(handle.name)


class SumoUnsupervisedTrainingTests(unittest.TestCase):
    def test_trains_and_evaluates_on_sumo_trace_steps(self):
        trace = load_sumo_fcd_trace(
            write_trace(
                """
                <fcd-export>
                  <timestep time="0.0">
                    <vehicle id="veh0" x="0.0" y="0.0" speed="0.0" angle="0.0" />
                    <person id="ped0" x="3.0" y="4.0" speed="0.0" angle="0.0" />
                  </timestep>
                  <timestep time="1.0">
                    <vehicle id="veh0" x="0.5" y="0.0" speed="0.0" angle="0.0" />
                    <person id="ped0" x="3.5" y="4.0" speed="0.0" angle="0.0" />
                  </timestep>
                  <timestep time="2.0">
                    <vehicle id="veh0" x="1.0" y="0.0" speed="0.0" angle="0.0" />
                    <person id="ped0" x="4.0" y="4.0" speed="0.0" angle="0.0" />
                  </timestep>
                </fcd-export>
                """
            )
        )
        streamed_metrics = []

        result = train_sumo_unsupervised(
            trace=trace,
            network_config=DynamicNetworkConfig(
                max_pair_distance_m=20.0,
                link_creation_probability=1.0,
                mean_link_duration_s=100.0,
                num_rbs=2,
                shadowing_std_db=0.0,
                fading_correlation=1.0,
                csi_error_std=0.0,
                csi_error_correlation=1.0,
            ),
            radio_config=RadioConfig(num_rbs=2, max_rbs_per_link=1, min_rate_bps=100.0),
            objective_config=RRMObjectiveConfig(rate_scale_bps=1_000.0),
            model_config=TGNNConfig(
                num_rbs=2,
                memory_dim=4,
                message_dim=4,
                embedding_dim=4,
                hidden_dim=8,
                attention_heads=1,
            ),
            epochs=2,
            train_steps=2,
            eval_steps=1,
            learning_rate=1e-3,
            optimizer_name="adam",
            grad_clip_norm=0.5,
            p_max_watt=0.0025,
            seed=7,
            device="auto",
            metrics_callback=streamed_metrics.append,
        )

        self.assertIsInstance(result.model, TGNNResourceAllocator)
        self.assertEqual(len(result.train_metrics), 2)
        self.assertEqual(len(result.eval_metrics), 2)
        self.assertEqual(result.train_metrics[0].phase, "train")
        self.assertEqual(result.eval_metrics[0].phase, "eval")
        self.assertGreater(result.train_metrics[0].avg_active_links, 0.0)
        self.assertTrue(math.isfinite(result.train_metrics[-1].avg_loss))
        self.assertTrue(math.isfinite(result.eval_metrics[-1].avg_mean_rate_bps))
        self.assertGreaterEqual(result.eval_metrics[-1].avg_qos_violation_fraction, 0.0)
        self.assertLessEqual(result.eval_metrics[-1].avg_qos_violation_fraction, 1.0)
        self.assertGreaterEqual(result.train_metrics[-1].elapsed_seconds, 0.0)
        self.assertGreaterEqual(result.train_metrics[-1].slots_per_second, 0.0)
        self.assertEqual(result.device.type, "cuda" if torch.cuda.is_available() else "cpu")
        self.assertEqual(
            [(metrics.epoch, metrics.phase) for metrics in streamed_metrics],
            [(1, "train"), (1, "eval"), (2, "train"), (2, "eval")],
        )

    def test_saves_checkpoint_when_requested(self):
        trace = load_sumo_fcd_trace(
            write_trace(
                """
                <fcd-export>
                  <timestep time="0.0">
                    <vehicle id="veh0" x="0.0" y="0.0" speed="0.0" angle="0.0" />
                    <person id="ped0" x="3.0" y="4.0" speed="0.0" angle="0.0" />
                  </timestep>
                </fcd-export>
                """
            )
        )
        checkpoint = Path(tempfile.NamedTemporaryFile(suffix=".pt", delete=False).name)
        checkpoint.unlink()

        result = train_sumo_unsupervised(
            trace=trace,
            network_config=DynamicNetworkConfig(
                max_pair_distance_m=20.0,
                link_creation_probability=1.0,
                mean_link_duration_s=100.0,
                num_rbs=2,
                shadowing_std_db=0.0,
                fading_correlation=1.0,
                csi_error_std=0.0,
                csi_error_correlation=1.0,
            ),
            radio_config=RadioConfig(num_rbs=2, max_rbs_per_link=1, min_rate_bps=100.0),
            model_config=TGNNConfig(
                num_rbs=2,
                memory_dim=4,
                message_dim=4,
                embedding_dim=4,
                hidden_dim=8,
                attention_heads=1,
            ),
            epochs=1,
            train_steps=1,
            eval_steps=0,
            checkpoint_path=checkpoint,
            device="cpu",
            seed=3,
        )

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.assertEqual(result.checkpoint_path, checkpoint)
        self.assertEqual(payload["device"], "cpu")
        self.assertIn("model_state_dict", payload)
        self.assertEqual(len(payload["train_metrics"]), 1)


if __name__ == "__main__":
    unittest.main()
