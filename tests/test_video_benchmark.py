import unittest

from scripts.video_pipeline_benchmark import compare_outcomes, score_outputs


class VideoBenchmarkTests(unittest.TestCase):
    def test_caption_quality_and_annotated_missed_event_rate(self):
        scores = score_outputs(
            [
                {
                    "answer": "A red car moves",
                    "timestamp_seconds": 1.1,
                }
            ],
            ["A red car moves quickly"],
            [1.0, 2.0],
            0.2,
        )

        self.assertEqual(scores["caption_scored_outputs"], 1)
        self.assertEqual(scores["caption_token_f1"], 0.8889)
        self.assertEqual(scores["missed_events"], 1)
        self.assertEqual(scores["missed_event_rate"], 0.5)

    def test_missing_annotations_are_reported_as_unavailable(self):
        scores = score_outputs(
            [{"answer": "A scene.", "timestamp_seconds": 0.0}],
            [],
            [],
            0.5,
        )

        self.assertIsNone(scores["caption_token_f1"])
        self.assertIsNone(scores["missed_event_rate"])

    def test_comparison_reports_adaptive_minus_frame_by_frame_deltas(self):
        deltas = compare_outcomes(
            {
                "caption_token_f1": 0.8,
                "missed_event_rate": 0.2,
                "mean_inference_latency_ms": 100,
                "vlm_invocations": 2,
            },
            {
                "caption_token_f1": 0.9,
                "missed_event_rate": 0.0,
                "mean_inference_latency_ms": 200,
                "vlm_invocations": 10,
            },
        )

        self.assertEqual(
            deltas["caption_token_f1_delta_adaptive_minus_frame_by_frame"],
            -0.1,
        )
        self.assertEqual(
            deltas[
                "mean_inference_latency_ms_delta_adaptive_minus_frame_by_frame"
            ],
            -100,
        )
        self.assertEqual(deltas["mean_latency_reduction_pct"], 50.0)


if __name__ == "__main__":
    unittest.main()
