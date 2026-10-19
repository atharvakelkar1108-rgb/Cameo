"""Run adaptive-vs-frame-by-frame video evaluation from an annotated manifest.

Manifest structure:
{
  "videos": [{
    "id": "clip-01",
    "frames": [
      {"image": "frames/000.jpg", "timestamp_seconds": 0.0,
       "audio_rms": 0.0}
    ],
    "event_timestamps_seconds": [1.5, 4.0],
    "reference_captions": ["A person opens a door."]
  }]
}
"""

import argparse
from collections import Counter
import json
import math
import re
from pathlib import Path

from io import BytesIO


def caption_token_f1(caption, references):
    tokens = lambda text: re.findall(r"[a-z0-9]+", text.lower())
    candidate = Counter(tokens(caption))
    best_score = None
    for reference in references:
        reference_tokens = Counter(tokens(reference))
        overlap = sum((candidate & reference_tokens).values())
        precision = overlap / sum(candidate.values()) if candidate else 0.0
        recall = (
            overlap / sum(reference_tokens.values())
            if reference_tokens
            else 0.0
        )
        score = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        best_score = score if best_score is None else max(best_score, score)
    return best_score


def score_outputs(outputs, reference_captions, event_timestamps, tolerance):
    reference_captions = reference_captions or []
    caption_scores = [
        caption_token_f1(item["answer"], reference_captions)
        for item in outputs
        if item.get("answer") and reference_captions
    ]
    inferred_timestamps = [
        item["timestamp_seconds"]
        for item in outputs
        if item.get("timestamp_seconds") is not None
    ]
    missed_events = sum(
        not any(
            abs(inferred - expected) <= tolerance
            for inferred in inferred_timestamps
        )
        for expected in event_timestamps
    )
    return {
        "caption_token_f1": (
            round(sum(caption_scores) / len(caption_scores), 4)
            if caption_scores
            else None
        ),
        "caption_scored_outputs": len(caption_scores),
        "annotated_events": len(event_timestamps),
        "missed_events": missed_events,
        "missed_event_rate": (
            round(missed_events / len(event_timestamps), 4)
            if event_timestamps
            else None
        ),
        "event_timestamp_tolerance_seconds": tolerance,
    }

def compare_outcomes(adaptive, frame_by_frame):
    fields = (
        "caption_token_f1",
        "missed_event_rate",
        "mean_inference_latency_ms",
        "mean_cpu_percent",
        "mean_process_ram_mb",
        "measured_vlm_cpu_package_energy_j",
        "vlm_invocations",
    )
    deltas = {}
    for field in fields:
        adaptive_value = adaptive.get(field)
        baseline_value = frame_by_frame.get(field)
        deltas[f"{field}_delta_adaptive_minus_frame_by_frame"] = (
            round(adaptive_value - baseline_value, 4)
            if isinstance(adaptive_value, (int, float))
            and isinstance(baseline_value, (int, float))
            else None
        )
    baseline_latency = frame_by_frame.get("mean_inference_latency_ms")
    adaptive_latency = adaptive.get("mean_inference_latency_ms")
    deltas["mean_latency_reduction_pct"] = (
        round(
            100 * (baseline_latency - adaptive_latency) / baseline_latency,
            2,
        )
        if isinstance(baseline_latency, (int, float))
        and baseline_latency > 0
        and isinstance(adaptive_latency, (int, float))
        else None
    )
    return deltas


def run_session(client, video, mode, quality_retention, latency_target_ms):
    start_payload = {
        "adaptive": True,
        "variant": "quantized",
        "priority": "balanced",
        "task": "caption",
        "question": video.get("question", ""),
        "single_result": False,
        "min_quality_retention": quality_retention,
        "max_latency_ms": latency_target_ms,
        "benchmark_mode": mode,
        "audio_source": (
            "browser decoded file audio"
            if any(frame.get("audio_rms") is not None
                   for frame in video["frames"])
            else "unavailable"
        ),
        "energy_budget_j_per_min": video.get("energy_budget_j_per_min"),
    }
    started = client.post("/api/video/start", json=start_payload)
    if started.status_code != 200:
        raise RuntimeError(
            f"Could not start {mode} session: "
            f"{started.get_json().get('error', started.status_code)}"
        )
    session_id = started.get_json()["session_id"]
    try:
        for frame in video["frames"]:
            image_path = (video["_manifest_dir"] / frame["image"]).resolve()
            with image_path.open("rb") as image_file:
                fields = {
                    "session_id": session_id,
                    "timestamp_seconds": str(frame["timestamp_seconds"]),
                    "image": (
                        BytesIO(image_file.read()),
                        image_path.name,
                    ),
                }
            if frame.get("audio_rms") is not None:
                fields["audio_rms"] = str(frame["audio_rms"])
            response = client.post("/api/video/frame", data=fields)
            if response.status_code != 200:
                raise RuntimeError(
                    f"{mode} frame at {frame['timestamp_seconds']}s failed: "
                    f"{response.get_json().get('error', response.status_code)}"
                )
        stopped = client.post(
            "/api/video/stop", json={"session_id": session_id}
        )
        if stopped.status_code != 200:
            raise RuntimeError(
                f"Could not stop {mode} session: "
                f"{stopped.get_json().get('error', stopped.status_code)}"
            )
        return stopped.get_json()["metrics"]
    except Exception:
        client.post("/api/video/stop", json={"session_id": session_id})
        raise


def run_benchmark(manifest_path, quality_retention, latency_target_ms):
    from webapp.app import app

    if (
        not math.isfinite(quality_retention)
        or not 0 <= quality_retention <= 1
    ):
        raise ValueError("Quality retention must be between 0 and 1.")
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    videos = manifest.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError("The manifest must contain a non-empty 'videos' list.")
    tolerance = float(manifest.get("event_timestamp_tolerance_seconds", 0.5))
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError(
            "event_timestamp_tolerance_seconds must be non-negative."
        )
    report = {
        "manifest": str(manifest_path),
        "caption_metric": "mean best-reference unigram token F1",
        "event_timestamp_tolerance_seconds": tolerance,
        "videos": [],
    }
    with app.test_client() as client:
        for video in videos:
            if not video.get("frames"):
                raise ValueError(
                    f"Video {video.get('id', '<unknown>')} has no frames."
                )
            event_timestamps = video.get("event_timestamps_seconds", [])
            references = video.get("reference_captions", [])
            if not isinstance(event_timestamps, list) or not all(
                isinstance(value, (int, float))
                and math.isfinite(value)
                and value >= 0
                for value in event_timestamps
            ):
                raise ValueError(
                    "event_timestamps_seconds must contain non-negative numbers."
                )
            if not isinstance(references, list) or not all(
                isinstance(value, str) for value in references
            ):
                raise ValueError("reference_captions must be a list of strings.")
            for frame in video["frames"]:
                timestamp = frame.get("timestamp_seconds")
                if (
                    not isinstance(timestamp, (int, float))
                    or not math.isfinite(timestamp)
                    or timestamp < 0
                ):
                    raise ValueError(
                        "Each frame requires a non-negative numeric "
                        "timestamp_seconds value."
                    )
                audio_rms = frame.get("audio_rms")
                if audio_rms is not None and (
                    not isinstance(audio_rms, (int, float))
                    or not math.isfinite(audio_rms)
                    or not 0 <= audio_rms <= 1
                ):
                    raise ValueError("Frame audio_rms values must be 0 to 1.")
            video = {**video, "_manifest_dir": manifest_path.parent}
            outcomes = {}
            for mode in ("adaptive", "frame_by_frame"):
                metrics = run_session(
                    client,
                    video,
                    mode,
                    quality_retention,
                    latency_target_ms,
                )
                outputs = metrics.pop("outputs", [])
                outcomes[mode] = {
                    **metrics,
                    **score_outputs(
                        outputs,
                        references,
                        event_timestamps,
                        tolerance,
                    ),
                }
            report["videos"].append({
                "id": video.get("id", "unnamed"),
                "adaptive": outcomes["adaptive"],
                "frame_by_frame": outcomes["frame_by_frame"],
                "adaptive_vs_frame_by_frame": compare_outcomes(
                    outcomes["adaptive"], outcomes["frame_by_frame"]
                ),
            })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/video_pipeline_benchmark.json"),
    )
    parser.add_argument("--quality-retention", type=float, default=0.9)
    parser.add_argument("--latency-target-ms", type=float)
    args = parser.parse_args()
    if not 0 <= args.quality_retention <= 1:
        parser.error("--quality-retention must be between 0 and 1")
    report = run_benchmark(
        args.manifest,
        args.quality_retention,
        args.latency_target_ms,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    for video in report["videos"]:
        missed_rate = video["adaptive"]["missed_event_rate"]
        print(
            f"{video['id']}: adaptive missed-event rate="
            f"{missed_rate * 100:.1f}%"
            if missed_rate is not None
            else f"{video['id']}: adaptive missed-event rate=Unavailable"
        )
        print(
            f"caption token F1={video['adaptive']['caption_token_f1']}, "
            f"mean latency={video['adaptive']['mean_inference_latency_ms']} ms, "
            f"CPU={video['adaptive']['mean_cpu_percent']}%, "
            f"RAM={video['adaptive']['mean_process_ram_mb']} MB, "
            f"energy={video['adaptive']['measured_vlm_cpu_package_energy_j']} J"
        )
    print(f"Saved benchmark report to {args.output.resolve()}")


if __name__ == "__main__":
    main()
