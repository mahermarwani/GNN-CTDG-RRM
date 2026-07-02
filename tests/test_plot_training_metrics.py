import csv
import tempfile
import unittest
from pathlib import Path

from tgnn_rrm.plotting import plot_training_metrics


class PlotTrainingMetricsTests(unittest.TestCase):
    def test_saves_training_metric_plots_from_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            metrics_path = tmp_path / "metrics.csv"
            output_dir = tmp_path / "plots"
            with metrics_path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    (
                        "epoch",
                        "phase",
                        "avg_loss",
                        "avg_mean_rate_bps",
                        "avg_qos_penalty",
                        "avg_qos_violation_fraction",
                        "avg_active_links",
                        "slots",
                        "elapsed_seconds",
                        "slots_per_second",
                    )
                )
                writer.writerow((1, "train", 5.0, 100.0, 2.0, 0.4, 3.0, 10, 1.0, 10.0))
                writer.writerow((1, "eval", 6.0, 90.0, 3.0, 0.5, 4.0, 5, 0.5, 10.0))
                writer.writerow((2, "train", 4.0, 120.0, 1.0, 0.2, 3.5, 10, 1.2, 8.3))
                writer.writerow((2, "eval", 5.0, 95.0, 2.5, 0.4, 4.5, 5, 0.7, 7.1))

            output_paths = plot_training_metrics(metrics_path, output_dir)

            self.assertEqual(
                {path.name for path in output_paths},
                {
                    "loss.png",
                    "mean_rate_bps.png",
                    "qos_penalty.png",
                    "qos_violation_fraction.png",
                    "active_links.png",
                    "slots_per_second.png",
                },
            )
            self.assertTrue(all(path.exists() and path.stat().st_size > 0 for path in output_paths))

    def test_plots_legacy_metrics_without_qos_violation_fraction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            metrics_path = tmp_path / "legacy_metrics.csv"
            output_dir = tmp_path / "plots"
            with metrics_path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    (
                        "epoch",
                        "phase",
                        "avg_loss",
                        "avg_mean_rate_bps",
                        "avg_qos_penalty",
                        "avg_active_links",
                        "slots",
                        "elapsed_seconds",
                        "slots_per_second",
                    )
                )
                writer.writerow((1, "train", 5.0, 100.0, 2.0, 3.0, 10, 1.0, 10.0))
                writer.writerow((1, "eval", 6.0, 90.0, 3.0, 4.0, 5, 0.5, 10.0))

            output_paths = plot_training_metrics(metrics_path, output_dir)

            self.assertNotIn("qos_violation_fraction.png", {path.name for path in output_paths})
            self.assertTrue(all(path.exists() and path.stat().st_size > 0 for path in output_paths))


if __name__ == "__main__":
    unittest.main()
