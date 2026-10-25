import os
import unittest
from unittest.mock import patch

import webapp.app as web


class FakeModel:
    def generate(self, **kwargs):
        return kwargs.get("input_ids")

    def to(self, device):
        return self

    def eval(self):
        return self


class FakeProcessor:
    def __call__(self, **kwargs):
        return {"input_ids": object()}


class ModelCacheTests(unittest.TestCase):
    def setUp(self):
        web.MODEL_CACHE.clear()
        web.PROCESSOR_CACHE.clear()

    def tearDown(self):
        web.MODEL_CACHE.clear()
        web.PROCESSOR_CACHE.clear()

    def test_cached_baseline_is_not_reloaded(self):
        fake = object()
        web.MODEL_CACHE["baseline"] = fake
        with patch(
            "transformers.AutoModelForMultimodalLM.from_pretrained",
            side_effect=AssertionError("model was reloaded"),
        ):
            self.assertIs(web.load_model("baseline"), fake)

    def test_cached_compressed_model_is_not_reloaded(self):
        fake = object()
        web.MODEL_CACHE["quantized"] = fake
        with patch.object(
            web.torch,
            "load",
            side_effect=AssertionError("model was reloaded"),
        ):
            self.assertIs(web.load_saved_model("quantized"), fake)

    def test_acquire_model_reports_cache_hit_on_second_call(self):
        model = FakeModel()
        processor = FakeProcessor()
        with patch.object(web, "load_model", return_value=model), \
             patch.object(web, "load_processor", return_value=processor), \
             patch.object(web, "warmup_inference") as warmup:
            missed = web.acquire_model("quantized")
            web.MODEL_CACHE["quantized"] = model
            web.PROCESSOR_CACHE["processor"] = processor
            hit = web.acquire_model("quantized")

        self.assertFalse(missed["cache_hit"])
        self.assertTrue(hit["cache_hit"])
        warmup.assert_called_once()

    def test_adaptive_selection_reuses_a_cached_candidate(self):
        web.MODEL_CACHE["quantized"] = object()
        selection = {
            "selected": "baseline",
            "priority_used": "balanced",
            "candidates": ["baseline", "quantized"],
            "explanation": "Chose baseline.",
            "profile": {"display_name": "SmolVLM FP32 (baseline)"},
        }
        quantized_profile = {
            "display_name": "SmolVLM INT8",
            "latency_ms": 400,
            "input_resolution": 384,
        }
        with patch.object(web, "get_system_stats", return_value={
            "available_ram_mb": 8192,
            "cpu_cores": 8,
            "cpu_percent": 10,
        }), patch.object(web, "select_model", return_value=selection), \
             patch.object(
                 web,
                 "load_model_profiles",
                 return_value={"quantized": quantized_profile},
             ):
            result = web.perform_selection("balanced")

        self.assertEqual(result["selected"], "quantized")
        self.assertTrue(result["reused_cached_model"])
        self.assertIn("already-loaded", result["explanation"])

    def test_warmup_skips_objects_without_generate(self):
        web.warmup_inference(object(), object())

    def test_preload_warms_int8_variants_and_baseline_by_default(self):
        with patch.dict(os.environ, {"CAMEO_PRELOAD_MODEL": ""}, clear=False), \
             patch.object(web, "load_processor") as load_processor, \
             patch.object(web, "acquire_model") as acquire:
            web.preload_default_models()

        load_processor.assert_called_once()
        self.assertEqual(
            [call.args[0] for call in acquire.call_args_list],
            ["quantized", "pruned", "optimized", "baseline"],
        )

    def test_pruned_model_is_converted_to_int8_on_first_load(self):
        fp32 = FakeModel()
        int8 = FakeModel()
        weights = web.RESULTS_DIR / "pruned_model.pt"
        with patch.object(type(weights), "exists", return_value=True), \
             patch.object(web.torch, "load", return_value=fp32), \
             patch.object(web, "apply_dynamic_int8", return_value=int8) as quant:
            loaded = web.load_saved_model("pruned")

        quant.assert_called_once_with(fp32)
        self.assertIs(loaded, int8)
        self.assertIs(web.MODEL_CACHE["pruned"], int8)

    def test_quantized_is_built_from_pretrained_when_weights_missing(self):
        fp32 = FakeModel()
        int8 = FakeModel()
        weights = web.RESULTS_DIR / "quantized_model.pt"
        with patch.object(type(weights), "exists", return_value=False), \
             patch.object(web, "load_pretrained_fp32_model", return_value=fp32), \
             patch.object(web, "apply_dynamic_int8", return_value=int8) as quant:
            loaded = web.load_saved_model("quantized")

        quant.assert_called_once_with(fp32)
        self.assertIs(loaded, int8)
        self.assertIs(web.MODEL_CACHE["quantized"], int8)

    def test_already_int8_model_is_not_requantized(self):
        already = FakeModel()
        already.modules = lambda: [already]
        with patch.object(web, "is_dynamic_int8_model", return_value=True), \
             patch.object(web.torch.quantization, "quantize_dynamic") as quant:
            loaded = web.apply_dynamic_int8(already)

        quant.assert_not_called()
        self.assertIs(loaded, already)

    def test_preload_continues_when_one_variant_is_missing(self):
        def acquire(key):
            if key == "pruned":
                raise FileNotFoundError("pruned_model.pt")
            return {"model": object()}

        with patch.dict(os.environ, {"CAMEO_PRELOAD_MODEL": ""}, clear=False), \
             patch.object(web, "load_processor"), \
             patch.object(web, "acquire_model", side_effect=acquire) as loader:
            web.preload_default_models()

        self.assertEqual(
            [call.args[0] for call in loader.call_args_list],
            ["quantized", "pruned", "optimized", "baseline"],
        )


if __name__ == "__main__":
    unittest.main()
