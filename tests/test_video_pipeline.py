import unittest

from PIL import Image

from utils.video_pipeline import VideoSession, energy_delta_joules


class VideoSessionTests(unittest.TestCase):
    def test_initial_scene_change_redundancy_cooldown_and_heartbeat(self):
        session = VideoSession(started_at=100.0)
        black = Image.new("RGB", (32, 32), "black")
        white = Image.new("RGB", (32, 32), "white")

        initial = session.analyze_frame(black, now=100.0)
        self.assertTrue(initial["should_infer"])
        self.assertEqual(initial["reason"], "initial_frame")
        session.record_inference(
            black, 100.0, 100.0, "baseline", "first", 2.0, 0.0
        )

        redundant = session.analyze_frame(black, now=100.5)
        self.assertFalse(redundant["should_infer"])
        self.assertEqual(redundant["reason"], "redundant")

        cooldown = session.analyze_frame(white, now=100.2)
        self.assertFalse(cooldown["should_infer"])
        self.assertEqual(cooldown["reason"], "cooldown")

        scene_change = session.analyze_frame(white, now=100.5)
        self.assertTrue(scene_change["should_infer"])
        self.assertEqual(scene_change["reason"], "scene_change")
        session.record_inference(
            white, 100.5, 80.0, "quantized", "second", 1.0, 0.5
        )

        heartbeat = session.analyze_frame(white, now=104.6)
        self.assertTrue(heartbeat["should_infer"])
        self.assertEqual(heartbeat["reason"], "heartbeat")

    def test_summary_reports_invocation_and_resource_savings(self):
        session = VideoSession(started_at=10.0)
        image = Image.new("RGB", (16, 16), "black")
        session.total_frames = 5
        session.record_resources(25.0, 400.0)
        session.record_resources(35.0, 500.0)
        session.record_inference(
            image, 11.0, 100.0, "baseline", "caption", 2.0, 0.0
        )
        session.record_inference(
            image, 12.0, 80.0, "quantized", "caption", 1.0, 1.0
        )

        metrics = session.summary(now=15.0)

        self.assertEqual(metrics["sampled_frames"], 5)
        self.assertEqual(metrics["vlm_invocations"], 2)
        self.assertEqual(metrics["invocation_reduction_pct"], 60.0)
        self.assertEqual(metrics["estimated_frame_by_frame_vlm_latency_ms"], 450.0)
        self.assertEqual(metrics["event_driven_vlm_latency_ms"], 180.0)
        self.assertEqual(metrics["mean_cpu_percent"], 30.0)
        self.assertEqual(metrics["mean_process_ram_mb"], 450.0)
        self.assertEqual(metrics["measured_vlm_cpu_package_energy_j"], 3.0)
        self.assertEqual(metrics["estimated_frame_by_frame_vlm_energy_j"], 7.5)
        self.assertEqual(session.energy_profiles(), {
            "baseline": 2.0,
            "quantized": 1.0,
        })

    def test_single_result_session_selects_one_representative_frame(self):
        session = VideoSession(settings={"single_result": True})
        black = Image.new("RGB", (32, 32), "black")
        white = Image.new("RGB", (32, 32), "white")

        first = session.analyze_frame(black, now=10.0, timestamp_seconds=0.0)
        self.assertFalse(first["should_infer"])

        next_scene = session.analyze_frame(
            white, now=11.0, timestamp_seconds=1.0
        )

        self.assertFalse(next_scene["should_infer"])
        self.assertEqual(next_scene["reason"], "candidate_frame")
        self.assertEqual(session.inference_count, 0)
        self.assertEqual(session.skipped_single_result, 1)
        self.assertEqual(session.best_frame.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(session.best_frame_timestamp_seconds, 1.0)

    def test_uncertain_event_requests_detailed_analysis(self):
        session = VideoSession()
        dark = Image.new("RGB", (32, 32), (0, 0, 0))
        slightly_lighter = Image.new("RGB", (32, 32), (16, 16, 16))
        session.record_inference(
            dark, 10.0, 100.0, "quantized", "A dark empty room.", None, 0.0
        )

        decision = session.analyze_frame(slightly_lighter, now=10.5)

        self.assertTrue(decision["should_infer"])
        self.assertTrue(decision["uncertain"])
        self.assertTrue(decision["detailed_analysis"])
        self.assertEqual(decision["reason"], "uncertain_event")

    def test_energy_debt_defers_low_value_but_not_important_event(self):
        session = VideoSession(settings={"energy_budget_j_per_min": 1.0})
        session.record_inference(
            Image.new("RGB", (32, 32), "black"),
            95.0,
            100.0,
            "quantized",
            "A still black frame.",
            4.0,
            0.0,
        )

        low_value = session.analyze_frame(
            Image.new("RGB", (32, 32), (27, 27, 27)),
            now=96.0,
        )
        important = session.analyze_frame(
            Image.new("RGB", (32, 32), "white"),
            now=96.1,
        )

        self.assertFalse(low_value["should_infer"])
        self.assertEqual(low_value["reason"], "energy_budget")
        self.assertTrue(important["should_infer"])
        self.assertTrue(important["important_event"])
        self.assertEqual(important["energy_debt_j"], 3.0)
        self.assertTrue(session.summary(now=96.1)["energy_controller_active"])

    def test_energy_debt_uses_a_rolling_one_minute_window(self):
        session = VideoSession(settings={"energy_budget_j_per_min": 2.0})
        session.energy_observations.extend([
            (10.0, 5.0),
            (55.0, 1.0),
        ])

        state = session.energy_state(now=70.0)

        self.assertEqual(state["used_last_minute_j"], 1.0)
        self.assertEqual(state["energy_debt_j"], 0.0)

    def test_caption_feedback_updates_event_threshold(self):
        session = VideoSession()
        image = Image.new("RGB", (16, 16), "black")
        session.record_inference(
            image, 1.0, 100.0, "quantized", "A green apple on a belt.", None, 0
        )
        session.gate_decisions.append({"event_value": 1.2})

        session.record_inference(
            image, 2.0, 100.0, "quantized", "A green apple on a belt.", None, 1
        )

        self.assertEqual(session.threshold_updates, 1)
        self.assertEqual(session.learned_event_threshold, 1.05)

    def test_energy_delta_requires_both_measurements(self):
        self.assertAlmostEqual(energy_delta_joules(1.2, 2.5), 1.3)
        self.assertIsNone(energy_delta_joules(None, 2.5))

    def test_skipped_frame_is_buffered_and_recovered_on_next_event(self):
        session = VideoSession(started_at=10.0)
        black = Image.new("RGB", (32, 32), "black")
        subtle = Image.new("RGB", (32, 32), (12, 12, 12))
        white = Image.new("RGB", (32, 32), "white")
        session.record_inference(
            black, 10.0, 100.0, "quantized", "A dark room.", None, 0.0
        )

        skipped = session.analyze_frame(
            subtle, now=10.5, timestamp_seconds=0.5
        )
        self.assertFalse(skipped["should_infer"])
        self.assertEqual(len(session.buffered_frames), 1)

        trigger = session.analyze_frame(
            white, now=11.0, timestamp_seconds=1.0
        )
        self.assertTrue(trigger["should_infer"])
        recovered = session.take_recovery_candidate(
            now=11.0, current_timestamp_seconds=1.0
        )
        self.assertIsNotNone(recovered)
        self.assertEqual(recovered["reason"], "missed_event_recovery")
        self.assertEqual(recovered["timestamp_seconds"], 0.5)

    def test_audio_confirms_a_low_magnitude_visual_event(self):
        session = VideoSession()
        dark = Image.new("RGB", (32, 32), "black")
        slight_change = Image.new("RGB", (32, 32), (12, 12, 12))
        session.record_inference(
            dark, 10.0, 100.0, "quantized", "A dark room.", None, 0.0
        )

        decision = session.analyze_frame(
            slight_change, now=10.5, audio_rms=0.1
        )

        self.assertTrue(decision["should_infer"])
        self.assertTrue(decision["audio_visual_verified"])
        self.assertEqual(decision["reason"], "audio_visual_event")

    def test_text_query_changes_event_relevance_and_threshold(self):
        session = VideoSession(settings={
            "question": "When does the green object appear?",
        })
        black = Image.new("RGB", (32, 32), "black")
        green = Image.new("RGB", (32, 32), (0, 180, 0))
        session.record_inference(
            black, 10.0, 100.0, "quantized", "An empty scene.", None, 0.0
        )

        decision = session.analyze_frame(green, now=10.5)

        self.assertGreater(decision["query_relevance"], 0.75)
        self.assertLess(decision["effective_threshold"], 1.0)


if __name__ == "__main__":
    unittest.main()
