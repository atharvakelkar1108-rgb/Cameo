import unittest
from io import BytesIO
import time
from unittest.mock import patch

from PIL import Image
import torch

import webapp.app as web


class VideoApiTests(unittest.TestCase):
    @staticmethod
    def frame_data(color):
        image_data = BytesIO()
        Image.new("RGB", (32, 32), color).save(image_data, format="PNG")
        return image_data.getvalue()

    def test_video_start_rejects_invalid_energy_budget(self):
        response = web.app.test_client().post(
            "/api/video/start",
            json={"energy_budget_j_per_min": 0},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("energy budget", response.get_json()["error"].lower())

    def test_video_start_rejects_quality_retention_outside_unit_interval(self):
        response = web.app.test_client().post(
            "/api/video/start",
            json={"min_quality_retention": 1.1},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("quality retention", response.get_json()["error"].lower())

    def test_video_frame_rejects_invalid_audio_rms(self):
        client = web.app.test_client()
        started = client.post("/api/video/start", json={})
        session_id = started.get_json()["session_id"]
        response = client.post("/api/video/frame", data={
            "session_id": session_id,
            "audio_rms": "1.5",
            "image": (BytesIO(self.frame_data("black")), "frame.png"),
        })

        self.assertEqual(response.status_code, 400)
        self.assertIn("audio rms", response.get_json()["error"].lower())
        client.post("/api/video/stop", json={"session_id": session_id})

    def test_caption_returns_one_complete_sentence_with_bounded_generation(self):
        class FakeTokenizer:
            eos_token_id = 2
            pad_token_id = 0

        class FakeProcessor:
            tokenizer = FakeTokenizer()

            def apply_chat_template(self, messages, add_generation_prompt):
                self.prompt = messages[0]["content"][1]["text"]
                return "prompt"

            def __call__(self, **kwargs):
                return {"input_ids": torch.tensor([[1]])}

            def batch_decode(self, generated, skip_special_tokens):
                return ["A: A machine sorts green fruit. The frame is blurry."]

        class FakeModel:
            def generate(self, **kwargs):
                self.max_new_tokens = kwargs["max_new_tokens"]
                return torch.tensor([[1, 2, 3, 4]])

        processor = FakeProcessor()
        model = FakeModel()
        answer = web.generate_answer(
            model,
            processor,
            Image.new("RGB", (16, 16), "black"),
        )

        self.assertEqual(answer, "A machine sorts green fruit.")
        self.assertEqual(model.max_new_tokens, 48)
        self.assertIn("complete, concise sentence", processor.prompt)

    def test_caption_retries_numeric_answer_and_returns_caption_not_answer_label(self):
        class FakeTokenizer:
            eos_token_id = 2
            pad_token_id = 0

        class FakeProcessor:
            tokenizer = FakeTokenizer()

            def __init__(self):
                self.decoded = iter([
                    "The answer is: 18",
                    "A machine sorts green limes on a conveyor belt.",
                ])

            def apply_chat_template(self, messages, add_generation_prompt):
                return messages[0]["content"][1]["text"]

            def __call__(self, **kwargs):
                return {"input_ids": torch.tensor([[1]])}

            def batch_decode(self, generated, skip_special_tokens):
                return [next(self.decoded)]

        class FakeModel:
            def __init__(self):
                self.calls = 0

            def generate(self, **kwargs):
                self.calls += 1
                return torch.tensor([[1, 2, 3, 4]])

        processor = FakeProcessor()
        model = FakeModel()
        answer = web.generate_answer(
            model,
            processor,
            Image.new("RGB", (16, 16), "black"),
        )

        self.assertEqual(answer, "A machine sorts green limes on a conveyor belt.")
        self.assertEqual(model.calls, 2)

    def test_adaptive_video_frame_uses_resource_and_latency_selector(self):
        image_data = BytesIO()
        Image.new("RGB", (32, 32), "black").save(image_data, format="PNG")
        image_data.seek(0)
        client = web.app.test_client()
        started = client.post("/api/video/start", json={
            "variant": "quantized",
            "adaptive": True,
            "priority": "balanced",
            "task": "caption",
            "max_latency_ms": 500,
        })
        session_id = started.get_json()["session_id"]
        selection = {
            "selected": "quantized",
            "profile": {"input_resolution": 384},
            "explanation": "Fits current resources.",
        }
        stats = {"cpu_percent": 4.0, "process_rss_mb": 100.0}

        with patch.object(web, "get_system_stats", return_value=stats), \
             patch.object(
                 web, "perform_selection", return_value=selection
             ) as select, \
             patch.object(web, "load_result_info", return_value={
                 "mean_latency_ms": 250,
             }), \
             patch.object(web, "load_model", return_value=object()), \
             patch.object(web, "load_processor", return_value=object()), \
             patch.object(web, "generate_answer", return_value="caption"), \
             patch.object(web, "read_cpu_energy_joules", return_value=None):
            frame = client.post("/api/video/frame", data={
                "session_id": session_id,
                "image": (image_data, "frame.png"),
            })
            client.post("/api/video/stop", json={"session_id": session_id})

        self.assertEqual(frame.status_code, 200)
        self.assertEqual(frame.get_json()["model"], "quantized")
        select.assert_called_once_with(
            "accuracy",
            500.0,
            {},
            min_quality_retention=0.95,
        )

    def test_redundant_frames_do_not_invoke_the_vlm(self):
        image_data = BytesIO()
        Image.new("RGB", (32, 32), "black").save(image_data, format="PNG")
        frame_bytes = image_data.getvalue()
        white_data = BytesIO()
        Image.new("RGB", (32, 32), "white").save(white_data, format="PNG")
        white_bytes = white_data.getvalue()
        client = web.app.test_client()
        workspace = client.get("/")
        self.assertEqual(workspace.status_code, 200)
        self.assertIn(b"videoPreview", workspace.data)
        started = client.post("/api/video/start", json={
            "variant": "baseline",
            "adaptive": False,
            "priority": "balanced",
            "task": "caption",
            "single_result": True,
        })
        self.assertEqual(started.status_code, 200)
        session_id = started.get_json()["session_id"]
        stats = {"cpu_percent": 4.0, "process_rss_mb": 100.0}

        with patch.object(web, "get_system_stats", return_value=stats), \
             patch.object(web, "load_model", return_value=object()), \
             patch.object(web, "load_processor", return_value=object()), \
             patch.object(
                 web, "generate_answer", return_value="test caption"
             ) as infer, \
             patch.object(web, "read_cpu_energy_joules", return_value=None):
            first = client.post("/api/video/frame", data={
                "session_id": session_id,
                "timestamp_seconds": "0",
                "image": (BytesIO(frame_bytes), "frame.png"),
            })
            second = client.post("/api/video/frame", data={
                "session_id": session_id,
                "timestamp_seconds": "0.5",
                "image": (
                    BytesIO(white_bytes),
                    "frame.png",
                ),
            })
            summary_response = client.post(
                "/api/video/stop",
                json={"session_id": session_id},
            )

        self.assertFalse(first.get_json()["should_infer"])
        self.assertFalse(second.get_json()["should_infer"])
        self.assertEqual(infer.call_count, 1)
        metrics = summary_response.get_json()["metrics"]
        self.assertEqual(metrics["sampled_frames"], 2)
        self.assertEqual(metrics["vlm_invocations"], 1)
        self.assertEqual(metrics["single_result_frames_skipped"], 1)
        self.assertEqual(metrics["selected_model"], "baseline")
        self.assertEqual(metrics["model_size_mb"], 978.599)
        self.assertEqual(metrics["model_footprint_mb"], 1276.973)
        self.assertEqual(metrics["outputs"][0]["answer"], "test caption")
        self.assertEqual(metrics["outputs"][0]["timestamp_seconds"], 0.5)
        self.assertTrue(
            metrics["outputs"][0]["keyframe_data_url"].startswith(
                "data:image/jpeg;base64,"
            )
        )
        self.assertIsNone(metrics["measured_vlm_cpu_package_energy_j"])

    def test_video_api_recovers_skipped_frame_with_timestamped_evidence(self):
        client = web.app.test_client()
        started = client.post("/api/video/start", json={
            "variant": "baseline",
            "adaptive": False,
            "task": "caption",
        })
        session_id = started.get_json()["session_id"]
        session = web.VIDEO_SESSIONS[session_id]
        stats = {"cpu_percent": 4.0, "process_rss_mb": 100.0}

        with patch.object(web, "get_system_stats", return_value=stats), \
             patch.object(web, "load_result_info", return_value={}), \
             patch.object(web, "load_model", return_value=object()), \
             patch.object(web, "load_processor", return_value=object()), \
             patch.object(
                 web,
                 "generate_answer",
                 side_effect=["first caption", "recovered caption", "current caption"],
             ) as infer, \
             patch.object(web, "read_cpu_energy_joules", return_value=None):
            first = client.post("/api/video/frame", data={
                "session_id": session_id,
                "timestamp_seconds": "0",
                "image": (BytesIO(self.frame_data("black")), "frame.png"),
            })
            self.assertTrue(first.get_json()["should_infer"])

            session.last_inference_at = time.monotonic() - 1
            skipped = client.post("/api/video/frame", data={
                "session_id": session_id,
                "timestamp_seconds": "0.5",
                "image": (
                    BytesIO(self.frame_data((12, 12, 12))),
                    "frame.png",
                ),
            })
            self.assertFalse(skipped.get_json()["should_infer"])

            session.last_inference_at = time.monotonic() - 1
            current = client.post("/api/video/frame", data={
                "session_id": session_id,
                "timestamp_seconds": "1",
                "image": (BytesIO(self.frame_data("white")), "frame.png"),
            })
            self.assertEqual(current.status_code, 200)
            self.assertEqual(infer.call_count, 3)

        result = current.get_json()
        self.assertEqual(result["answer"], "current caption")
        self.assertEqual(len(result["recovered_outputs"]), 1)
        recovered = result["recovered_outputs"][0]
        self.assertEqual(recovered["answer"], "recovered caption")
        self.assertTrue(recovered["recovered"])
        self.assertEqual(recovered["timestamp_seconds"], 0.5)
        self.assertTrue(recovered["keyframe_data_url"].startswith("data:image"))
        stopped = client.post(
            "/api/video/stop", json={"session_id": session_id}
        )
        self.assertEqual(stopped.get_json()["metrics"]["recovered_frames"], 1)


if __name__ == "__main__":
    unittest.main()
