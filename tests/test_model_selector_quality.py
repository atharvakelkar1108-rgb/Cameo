import unittest
from unittest.mock import patch

from utils.model_selector import select_model


class ModelQualityConstraintTests(unittest.TestCase):
    def setUp(self):
        self.profiles = {
            "baseline": {
                "model_family": "SmolVLM-256M",
                "compression": "FP32",
                "deployment": "PC / high-memory device",
                "footprint_mb": 800,
                "latency_ms": 500,
                "bleu4": 0.50,
                "display_name": "FP32",
                "input_resolution": 512,
            },
            "quantized": {
                "model_family": "SmolVLM-256M",
                "compression": "Dynamic INT8",
                "deployment": "CPU / edge-capable",
                "footprint_mb": 400,
                "latency_ms": 200,
                "bleu4": 0.44,
                "display_name": "INT8",
                "input_resolution": 384,
            },
            "pruned": {
                "model_family": "SmolVLM-256M",
                "compression": "L1 Unstructured Pruning",
                "deployment": "Experimental / ablation",
                "footprint_mb": 450,
                "latency_ms": 300,
                "bleu4": 0.48,
                "display_name": "Pruned",
                "input_resolution": 384,
            },
        }

    def test_quality_floor_filters_variants_below_baseline_retention(self):
        with patch(
            "utils.model_selector.load_model_profiles",
            return_value=self.profiles,
        ):
            selection = select_model(
                ram_available_mb=5000,
                cpu_cores=8,
                priority="balanced",
                min_quality_retention=0.95,
            )

        self.assertEqual(set(selection["candidates"]), {"baseline", "pruned"})
        self.assertNotEqual(selection["selected"], "quantized")
        self.assertGreaterEqual(
            selection["quality_retention_vs_baseline"], 0.95
        )

    def test_quality_floor_fails_closed_without_baseline_quality(self):
        profiles = {
            "quantized": self.profiles["quantized"],
        }
        with patch(
            "utils.model_selector.load_model_profiles",
            return_value=profiles,
        ):
            selection = select_model(
                ram_available_mb=5000,
                cpu_cores=8,
                min_quality_retention=0.9,
            )

        self.assertIsNone(selection["selected"])
        self.assertIn("No benchmarked model", selection["explanation"])

    def test_ram_failure_explains_compliant_profiles_and_budget(self):
        with patch(
            "utils.model_selector.load_model_profiles",
            return_value=self.profiles,
        ):
            selection = select_model(
                ram_available_mb=500,
                cpu_cores=4,
                priority="balanced",
                min_quality_retention=0.95,
            )

        self.assertIsNone(selection["selected"])
        self.assertEqual(selection["ram_budget_mb"], 375)
        self.assertEqual(
            set(selection["quality_compliant_profiles"]),
            {"baseline", "pruned"},
        )
        self.assertIn("safe RAM budget of 375 MB", selection["explanation"])
        self.assertIn("baseline: 800 MB runtime RAM", selection["explanation"])


if __name__ == "__main__":
    unittest.main()
