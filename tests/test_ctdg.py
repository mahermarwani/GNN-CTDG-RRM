import unittest

import torch

from tgnn_rrm.ctdg import EventType, build_ctdg_events


class CTDGEventBuilderTests(unittest.TestCase):
    def test_initial_snapshot_emits_add_events_for_all_directed_pairs(self):
        gains = torch.ones((2, 2, 3))

        batch = build_ctdg_events(
            previous_active_ids=[],
            current_active_ids=[10, 20],
            previous_gains=None,
            current_gains=gains,
            time=1.5,
        )

        self.assertEqual(batch.time, 1.5)
        self.assertEqual(batch.active_ids, (10, 20))
        self.assertEqual(batch.added_ids, (10, 20))
        self.assertEqual(batch.updated_ids, ())
        self.assertEqual(batch.deleted_ids, ())
        self.assertEqual(len(batch.events), 2)
        self.assertTrue(all(event.event_type == EventType.ADD for event in batch.events))
        self.assertEqual({(event.source_id, event.destination_id) for event in batch.events}, {(10, 20), (20, 10)})

    def test_stable_snapshot_emits_update_events(self):
        gains = torch.ones((2, 2, 1))

        batch = build_ctdg_events(
            previous_active_ids=[10, 20],
            current_active_ids=[10, 20],
            previous_gains=gains,
            current_gains=gains,
            time=2.0,
        )

        self.assertEqual(batch.added_ids, ())
        self.assertEqual(batch.updated_ids, (10, 20))
        self.assertEqual(batch.deleted_ids, ())
        self.assertEqual(len(batch.events), 2)
        self.assertTrue(all(event.event_type == EventType.UPDATE for event in batch.events))

    def test_churn_classifies_added_updated_and_deleted_ids(self):
        previous_gains = torch.ones((2, 2, 2))
        current_gains = torch.ones((2, 2, 2))

        batch = build_ctdg_events(
            previous_active_ids=[1, 2],
            current_active_ids=[2, 3],
            previous_gains=previous_gains,
            current_gains=current_gains,
            time=3.0,
        )

        self.assertEqual(batch.added_ids, (3,))
        self.assertEqual(batch.updated_ids, (2,))
        self.assertEqual(batch.deleted_ids, (1,))
        self.assertEqual(
            {(event.source_id, event.destination_id, event.event_type) for event in batch.events},
            {
                (2, 3, EventType.UPDATE),
                (3, 2, EventType.ADD),
                (1, 2, EventType.DELETE),
            },
        )

    def test_features_are_directed_csi_slices(self):
        gains = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        )

        batch = build_ctdg_events(
            previous_active_ids=[],
            current_active_ids=[10, 20],
            previous_gains=None,
            current_gains=gains,
            time=0.0,
        )
        event = next(event for event in batch.events if event.source_id == 10 and event.destination_id == 20)

        self.assertTrue(torch.equal(event.source_features, torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.equal(event.destination_features, torch.tensor([7.0, 8.0])))
        self.assertTrue(torch.equal(event.edge_features, torch.tensor([3.0, 4.0])))

    def test_limits_events_to_strongest_interference_neighbors(self):
        gains = torch.zeros((4, 4, 1))
        gains[0, 0, 0] = 1.0
        gains[1, 1, 0] = 1.0
        gains[2, 2, 0] = 1.0
        gains[3, 3, 0] = 1.0
        gains[0, 1, 0] = 10.0
        gains[0, 2, 0] = 30.0
        gains[0, 3, 0] = 20.0

        batch = build_ctdg_events(
            previous_active_ids=[],
            current_active_ids=[10, 20, 30, 40],
            previous_gains=None,
            current_gains=gains,
            time=0.0,
            max_interference_neighbors=2,
        )

        self.assertEqual(len(batch.events), 8)
        self.assertEqual(
            {
                event.destination_id
                for event in batch.events
                if event.source_id == 10
            },
            {30, 40},
        )

    def test_isolated_active_link_emits_self_event_with_zero_edge_features(self):
        gains = torch.tensor([[[2.0, 3.0]]])

        batch = build_ctdg_events(
            previous_active_ids=[],
            current_active_ids=[7],
            previous_gains=None,
            current_gains=gains,
            time=4.0,
        )

        self.assertEqual(len(batch.events), 1)
        event = batch.events[0]
        self.assertEqual((event.source_id, event.destination_id), (7, 7))
        self.assertEqual(event.event_type, EventType.ADD)
        self.assertTrue(torch.equal(event.source_features, torch.tensor([2.0, 3.0])))
        self.assertTrue(torch.equal(event.edge_features, torch.zeros(2)))

    def test_deleted_isolated_link_emits_self_delete_from_previous_csi(self):
        previous_gains = torch.tensor([[[4.0, 5.0]]])
        current_gains = torch.empty((0, 0, 2))

        batch = build_ctdg_events(
            previous_active_ids=[7],
            current_active_ids=[],
            previous_gains=previous_gains,
            current_gains=current_gains,
            time=5.0,
        )

        self.assertEqual(batch.deleted_ids, (7,))
        self.assertEqual(len(batch.events), 1)
        event = batch.events[0]
        self.assertEqual((event.source_id, event.destination_id), (7, 7))
        self.assertEqual(event.event_type, EventType.DELETE)
        self.assertTrue(torch.equal(event.source_features, torch.tensor([4.0, 5.0])))
        self.assertTrue(torch.equal(event.edge_features, torch.zeros(2)))

    def test_input_tensors_keep_autograd_state(self):
        gains = torch.ones((1, 1, 2), requires_grad=True)

        batch = build_ctdg_events(
            previous_active_ids=[],
            current_active_ids=[1],
            previous_gains=None,
            current_gains=gains,
            time=0.0,
        )

        self.assertTrue(batch.events[0].source_features.requires_grad)


class CTDGValidationTests(unittest.TestCase):
    def test_rejects_duplicate_ids(self):
        gains = torch.ones((1, 1, 1))

        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_ctdg_events([1, 1], [1], gains, gains, time=0.0)

    def test_rejects_bad_tensor_rank(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            build_ctdg_events([], [1], None, torch.ones((1, 1)), time=0.0)

    def test_rejects_bad_tensor_shape(self):
        with self.assertRaisesRegex(ValueError, "link dimensions"):
            build_ctdg_events([], [1, 2], None, torch.ones((1, 1, 1)), time=0.0)

    def test_requires_previous_gains_for_deletions(self):
        with self.assertRaisesRegex(ValueError, "previous_gains"):
            build_ctdg_events([1], [], None, torch.empty((0, 0, 1)), time=0.0)


if __name__ == "__main__":
    unittest.main()
