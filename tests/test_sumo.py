import tempfile
import textwrap
import unittest
from pathlib import Path

import torch

from tgnn_rrm.ctdg import EventType
from tgnn_rrm.simulation import DynamicNetworkConfig, DynamicNetworkStep
from tgnn_rrm.sumo import SumoD2DSimulator, load_sumo_fcd_trace


def write_trace(xml: str) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False)
    with handle:
        handle.write(textwrap.dedent(xml).strip())
    return Path(handle.name)


def sumo_config(**overrides):
    values = {
        "max_pair_distance_m": 20.0,
        "link_creation_probability": 1.0,
        "mean_link_duration_s": 100.0,
        "num_rbs": 2,
        "shadowing_std_db": 0.0,
        "fading_correlation": 1.0,
        "csi_error_std": 0.0,
        "csi_error_correlation": 1.0,
    }
    values.update(overrides)
    return DynamicNetworkConfig(**values)


class SumoFCDTraceTests(unittest.TestCase):
    def test_loads_vehicle_and_person_frames_with_stable_numeric_ids(self):
        path = write_trace(
            """
            <fcd-export>
              <timestep time="0.00">
                <vehicle id="veh0" x="10.0" y="20.0" speed="5.0" angle="90.0" />
                <person id="ped0" x="11.0" y="21.0" speed="1.5" angle="0.0" />
              </timestep>
              <timestep time="0.50">
                <vehicle id="veh0" x="12.5" y="20.0" speed="5.0" angle="90.0" />
              </timestep>
            </fcd-export>
            """
        )

        trace = load_sumo_fcd_trace(path)

        self.assertEqual(trace.entity_ids, ("ped0", "veh0"))
        self.assertEqual(len(trace.frames), 2)
        first_entities = {entity.external_id: entity for entity in trace.frames[0].entities}
        self.assertEqual(first_entities["ped0"].entity_id, 0)
        self.assertEqual(first_entities["ped0"].node_type, "person")
        self.assertEqual(first_entities["veh0"].entity_id, 1)
        self.assertTrue(torch.allclose(first_entities["veh0"].position_m, torch.tensor([10.0, 20.0])))
        self.assertTrue(torch.allclose(first_entities["veh0"].velocity_mps, torch.tensor([5.0, 0.0]), atol=1e-6))


class SumoD2DSimulatorTests(unittest.TestCase):
    def test_sumo_frames_produce_dynamic_network_steps_and_ctdg_events(self):
        path = write_trace(
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
            </fcd-export>
            """
        )
        simulator = SumoD2DSimulator(load_sumo_fcd_trace(path), config=sumo_config(), seed=7)

        first = simulator.step()
        second = simulator.step()

        self.assertIsInstance(first, DynamicNetworkStep)
        self.assertEqual(len(first.snapshot.entities), 2)
        self.assertEqual(len(first.snapshot.active_links), 1)
        self.assertEqual(first.snapshot.gains.shape, (1, 1, 2))
        self.assertEqual(first.events.added_ids, first.snapshot.active_ids)
        self.assertTrue(all(event.event_type == EventType.ADD for event in first.events.events))
        self.assertEqual(second.events.updated_ids, first.snapshot.active_ids)
        self.assertTrue(all(event.event_type == EventType.UPDATE for event in second.events.events))

    def test_run_stops_when_trace_is_exhausted(self):
        path = write_trace(
            """
            <fcd-export>
              <timestep time="0.0">
                <vehicle id="veh0" x="0.0" y="0.0" speed="0.0" angle="0.0" />
              </timestep>
            </fcd-export>
            """
        )
        simulator = SumoD2DSimulator(load_sumo_fcd_trace(path), config=sumo_config(), seed=1)

        self.assertEqual(len(simulator.run(5)), 1)


if __name__ == "__main__":
    unittest.main()
