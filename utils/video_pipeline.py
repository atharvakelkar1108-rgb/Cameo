"""Event detection and evaluation state for browser-decoded video frames."""

from dataclasses import dataclass, field
from collections import deque
from pathlib import Path
from statistics import mean
from time import monotonic
from typing import Optional

from PIL import Image, ImageChops, ImageStat


FRAME_CHANGE_THRESHOLD = 0.08
FRAME_CHANGED_PIXEL_THRESHOLD = 0.10
MIN_INFERENCE_INTERVAL_SECONDS = 0.35
MAX_INFERENCE_INTERVAL_SECONDS = 4.0
MAX_BUFFERED_FRAMES = 8
MIN_AUDIO_RMS_FOR_EVENT = 0.02
ENERGY_COUNTER_PATH = Path(
    "/sys/class/powercap/intel-rapl:0/energy_uj"
)
ENERGY_RANGE_PATH = Path(
    "/sys/class/powercap/intel-rapl:0/max_energy_range_uj"
)


def read_cpu_energy_joules() -> Optional[float]:
    """Read Intel RAPL package energy when the OS exposes its counter."""
    try:
        return int(ENERGY_COUNTER_PATH.read_text().strip()) / 1_000_000
    except (OSError, ValueError):
        return None


def energy_delta_joules(
    start_joules: Optional[float],
    end_joules: Optional[float],
) -> Optional[float]:
    """Return a package-energy delta, accounting for a RAPL counter wrap."""
    if start_joules is None or end_joules is None:
        return None
    if end_joules >= start_joules:
        return end_joules - start_joules

    try:
        range_joules = int(ENERGY_RANGE_PATH.read_text().strip()) / 1_000_000
    except (OSError, ValueError):
        return None
    return range_joules - start_joules + end_joules


@dataclass
class VideoSession:
    """Tracks event gating and evaluation metrics for one video/camera stream."""

    settings: dict = field(default_factory=dict)
    started_at: float = field(default_factory=monotonic)
    last_inference_at: Optional[float] = None
    reference_frame: Optional[Image.Image] = None
    best_frame: Optional[Image.Image] = None
    best_frame_timestamp_seconds: Optional[float] = None
    best_frame_score: float = -1.0
    best_frame_detailed: bool = False
    best_frame_audio_verified: bool = False
    learned_event_threshold: float = 1.0
    previous_caption: Optional[str] = None
    total_frames: int = 0
    inference_count: int = 0
    skipped_redundant: int = 0
    skipped_cooldown: int = 0
    skipped_single_result: int = 0
    inference_latencies_ms: list = field(default_factory=list)
    frame_processing_ms: list = field(default_factory=list)
    cpu_samples: list = field(default_factory=list)
    ram_samples_mb: list = field(default_factory=list)
    energy_samples_j: list = field(default_factory=list)
    energy_observations: list = field(default_factory=list)
    model_energy_samples_j: dict = field(default_factory=dict)
    model_counts: dict = field(default_factory=dict)
    outputs: list = field(default_factory=list)
    gate_decisions: list = field(default_factory=list)
    threshold_updates: int = 0
    buffered_frames: deque = field(
        default_factory=lambda: deque(maxlen=MAX_BUFFERED_FRAMES)
    )
    recovered_frame_count: int = 0
    audio_sample_count: int = 0
    audio_confirmed_count: int = 0
    query_relevant_count: int = 0
    last_query_alignment: Optional[float] = None

    @staticmethod
    def _small_grayscale(image: Image.Image) -> Image.Image:
        return image.convert("L").resize((64, 64), Image.Resampling.BILINEAR)

    def _query_relevance(
        self,
        image: Image.Image,
        event_value: float,
        changed_fraction: float,
    ) -> float:
        prompt = str(self.settings.get("question", "")).lower()
        if not prompt.strip():
            return 0.0

        relevance = min(event_value / 2.0, 1.0)
        if any(
            word in prompt
            for word in (
                "when", "happen", "change", "move", "moving", "walk",
                "run", "fall", "enter", "leave", "appear", "disappear",
                "action", "doing",
            )
        ):
            relevance = max(relevance, min(changed_fraction / 0.12, 1.0))

        color_words = {
            "green": lambda r, g, b: g > r * 1.15 and g > b * 1.05,
            "red": lambda r, g, b: r > g * 1.2 and r > b * 1.2,
            "blue": lambda r, g, b: b > r * 1.15 and b > g * 1.05,
            "yellow": lambda r, g, b: r > b * 1.2 and g > b * 1.2,
        }
        requested_colors = [
            predicate
            for color, predicate in color_words.items()
            if color in prompt
        ]
        if requested_colors:
            pixels = image.convert("RGB").resize(
                (24, 24), Image.Resampling.BILINEAR
            )
            matching = sum(
                any(
                    predicate(*pixels.getpixel((x, y)))
                    for predicate in requested_colors
                )
                for y in range(pixels.height)
                for x in range(pixels.width)
            )
            relevance = max(relevance, matching / (24 * 24))

        if self.last_query_alignment is not None:
            relevance = max(relevance, self.last_query_alignment)
        return min(relevance, 1.0)

    def _buffer_skipped_frame(
        self,
        image: Image.Image,
        timestamp_seconds: Optional[float],
        now: float,
        decision: dict,
        audio_rms: Optional[float],
    ) -> None:
        buffered_image = image.copy()
        buffered_image.thumbnail((640, 640), Image.Resampling.LANCZOS)
        self.buffered_frames.append({
            "image": buffered_image,
            "timestamp_seconds": timestamp_seconds,
            "seen_at": now,
            "event_value": decision["event_value"],
            "effective_threshold": decision["effective_threshold"],
            "query_relevance": decision["query_relevance"],
            "audio_rms": audio_rms,
            "audio_visual_verified": decision["audio_visual_verified"],
        })

    def take_recovery_candidate(
        self,
        now: Optional[float] = None,
        current_timestamp_seconds: Optional[float] = None,
    ) -> Optional[dict]:
        now = monotonic() if now is None else now
        eligible = []
        for index, item in enumerate(self.buffered_frames):
            if now - item["seen_at"] > MAX_INFERENCE_INTERVAL_SECONDS:
                continue
            if (
                current_timestamp_seconds is not None
                and item["timestamp_seconds"] is not None
                and abs(
                    current_timestamp_seconds - item["timestamp_seconds"]
                ) > MAX_INFERENCE_INTERVAL_SECONDS
            ):
                continue
            threshold = item["effective_threshold"]
            has_signal = (
                item["event_value"] >= threshold * 0.55
                or item["audio_visual_verified"]
                or (
                    item["query_relevance"] >= 0.75
                    and item["event_value"] >= threshold * 0.35
                )
            )
            if has_signal:
                score = (
                    item["event_value"]
                    + item["query_relevance"] * 0.4
                    + (0.5 if item["audio_visual_verified"] else 0.0)
                )
                eligible.append((score, index, item))
        if not eligible:
            self.buffered_frames.clear()
            return None

        _, selected_index, selected = max(eligible, key=lambda entry: entry[0])
        remaining = [
            item for index, item in enumerate(self.buffered_frames)
            if index != selected_index
        ]
        self.buffered_frames.clear()
        self.buffered_frames.extend(remaining)
        selected["reason"] = "missed_event_recovery"
        selected["uncertain"] = (
            selected["effective_threshold"] * 0.75
            <= selected["event_value"]
            <= selected["effective_threshold"] * 1.25
        )
        selected["detailed_analysis"] = selected["uncertain"]
        selected["important_event"] = selected["audio_visual_verified"]
        selected["expected_information_benefit"] = min(
            selected["event_value"] / 2.0, 1.0
        )
        selected["estimated_latency_cost_ms"] = (
            mean(self.inference_latencies_ms)
            if self.inference_latencies_ms
            else None
        )
        selected["estimated_energy_cost_j"] = (
            mean(self.energy_samples_j) if self.energy_samples_j else None
        )
        selected["benefit_cost_ratio"] = None
        return selected

    def analyze_frame(
        self,
        image: Image.Image,
        now: Optional[float] = None,
        timestamp_seconds: Optional[float] = None,
        audio_rms: Optional[float] = None,
    ) -> dict:
        now = monotonic() if now is None else now
        self.total_frames += 1
        if audio_rms is not None:
            self.audio_sample_count += 1

        if self.reference_frame is None:
            if self.settings.get("single_result"):
                self.reference_frame = self._small_grayscale(image)
                self.best_frame = image.copy()
                self.best_frame_timestamp_seconds = timestamp_seconds
                self.best_frame_score = 0.0
                self.best_frame_audio_verified = False
                return {
                    "should_infer": False,
                    "reason": "candidate_frame",
                    "mean_difference": 1.0,
                    "changed_fraction": 1.0,
                    "event_value": 0.0,
                    "uncertain": False,
                    "detailed_analysis": False,
                    "energy_debt_j": 0.0,
                    "effective_threshold": self.learned_event_threshold,
                }
            decision = {
                "should_infer": True,
                "reason": "initial_frame",
                "mean_difference": 1.0,
                "changed_fraction": 1.0,
                "event_value": 1.0,
                "uncertain": True,
                "detailed_analysis": True,
                "energy_debt_j": 0.0,
                "effective_threshold": self.learned_event_threshold,
            }
            self.gate_decisions.append(decision)
            return decision

        current = self._small_grayscale(image)
        previous = self._small_grayscale(self.reference_frame)
        difference = ImageChops.difference(current, previous)
        mean_difference = ImageStat.Stat(difference).mean[0] / 255
        changed_pixels = sum(
            count
            for value, count in enumerate(difference.histogram())
            if value >= 32
        )
        changed_fraction = changed_pixels / (64 * 64)
        event_value = max(
            mean_difference / FRAME_CHANGE_THRESHOLD,
            changed_fraction / FRAME_CHANGED_PIXEL_THRESHOLD,
        )
        query_relevance = self._query_relevance(
            image, event_value, changed_fraction
        )
        audio_confirmed = (
            audio_rms is not None and audio_rms >= MIN_AUDIO_RMS_FOR_EVENT
        )
        energy_state = self.energy_state(now)
        energy_debt = energy_state["energy_debt_j"]
        budget = self.settings.get("energy_budget_j_per_min")
        energy_pressure = (
            min(energy_debt / budget, 1.0)
            if budget and budget > 0 and energy_state["available"]
            else 0.0
        )
        effective_threshold = self.learned_event_threshold * (
            1 + energy_pressure
        )
        if self.settings.get("question", "").strip():
            effective_threshold *= 1 - 0.2 * query_relevance
        if audio_confirmed:
            effective_threshold *= 0.8
        audio_visual_verified = (
            audio_confirmed and event_value >= effective_threshold * 0.5
        )
        if self.settings.get("single_result"):
            score = event_value * (1 + query_relevance * 0.3)
            if audio_visual_verified:
                score += 0.5
            if (
                score > self.best_frame_score
                and (
                    mean_difference >= FRAME_CHANGE_THRESHOLD
                    or changed_fraction >= FRAME_CHANGED_PIXEL_THRESHOLD
                    or audio_visual_verified
                    or (
                        query_relevance >= 0.75
                        and event_value >= effective_threshold * 0.35
                    )
                )
            ):
                self.best_frame = image.copy()
                self.best_frame_timestamp_seconds = timestamp_seconds
                self.best_frame_score = score
                self.best_frame_detailed = 0.75 <= event_value <= 1.25
                self.best_frame_audio_verified = audio_visual_verified
            self.skipped_single_result += 1
            decision = {
                "should_infer": False,
                "reason": "candidate_frame",
                "mean_difference": mean_difference,
                "changed_fraction": changed_fraction,
                "event_value": event_value,
                "query_relevance": query_relevance,
                "audio_rms": audio_rms,
                "audio_visual_verified": audio_visual_verified,
                "uncertain": 0.75 <= event_value <= 1.25,
                "detailed_analysis": 0.75 <= event_value <= 1.25,
                "energy_debt_j": energy_debt,
                "effective_threshold": effective_threshold,
            }
            self.gate_decisions.append(decision)
            if audio_visual_verified:
                self.audio_confirmed_count += 1
            if query_relevance >= 0.75:
                self.query_relevant_count += 1
            return decision

        elapsed = now - self.last_inference_at
        uncertainty_margin = 0.25
        lower_bound = effective_threshold * (1 - uncertainty_margin)
        upper_bound = effective_threshold * (1 + uncertainty_margin)
        uncertain = lower_bound <= event_value <= upper_bound
        important_event = event_value >= max(2.0, effective_threshold * 2)
        important_event = important_event or audio_visual_verified
        heartbeat = elapsed >= MAX_INFERENCE_INTERVAL_SECONDS
        estimated_latency_ms = (
            mean(self.inference_latencies_ms)
            if self.inference_latencies_ms
            else None
        )
        expected_information_benefit = min(event_value / 2, 1.0)
        estimated_cost_j = energy_state["estimated_cost_j"]
        benefit_cost_ratio = (
            expected_information_benefit / estimated_cost_j
            if estimated_cost_j
            else (
                expected_information_benefit / (estimated_latency_ms / 1000)
                if estimated_latency_ms
                else None
            )
        )
        if self.settings.get("benchmark_mode") == "frame_by_frame":
            reason = "frame_by_frame_benchmark"
            should_infer = True
        elif heartbeat:
            reason = "heartbeat"
            should_infer = True
        elif audio_visual_verified:
            reason = "audio_visual_event"
            should_infer = True
        elif event_value < lower_bound:
            reason = (
                "energy_budget"
                if energy_pressure
                and event_value >= self.learned_event_threshold * (
                    1 - uncertainty_margin
                )
                else "redundant"
            )
            should_infer = False
        elif energy_pressure and not uncertain and not important_event:
            reason = "energy_budget"
            should_infer = False
        elif uncertain:
            reason = "uncertain_event"
            should_infer = True
        elif elapsed < MIN_INFERENCE_INTERVAL_SECONDS:
            reason = "cooldown"
            should_infer = False
        else:
            reason = "scene_change"
            should_infer = True

        if not should_infer:
            if reason == "cooldown":
                self.skipped_cooldown += 1
            else:
                self.skipped_redundant += 1
        decision = {
            "should_infer": should_infer,
            "reason": reason,
            "mean_difference": mean_difference,
            "changed_fraction": changed_fraction,
            "event_value": event_value,
            "query_relevance": query_relevance,
            "audio_rms": audio_rms,
            "audio_confirmed": audio_confirmed,
            "audio_visual_verified": audio_visual_verified,
            "uncertain": uncertain,
            "detailed_analysis": uncertain,
            "important_event": important_event,
            "energy_debt_j": energy_debt,
            "energy_pressure": energy_pressure,
            "energy_measurement_available": energy_state["available"],
            "effective_threshold": effective_threshold,
            "expected_information_benefit": expected_information_benefit,
            "benefit_cost_ratio": benefit_cost_ratio,
            "estimated_latency_cost_ms": estimated_latency_ms,
            "estimated_energy_cost_j": estimated_cost_j,
        }
        self.gate_decisions.append(decision)
        if audio_visual_verified:
            self.audio_confirmed_count += 1
        if query_relevance >= 0.75:
            self.query_relevant_count += 1
        if not should_infer:
            self._buffer_skipped_frame(
                image, timestamp_seconds, now, decision, audio_rms
            )
        return decision

    def record_resources(self, cpu_percent: float, process_ram_mb: float) -> None:
        self.cpu_samples.append(cpu_percent)
        self.ram_samples_mb.append(process_ram_mb)

    def energy_profiles(self) -> dict:
        return {
            model_key: mean(samples)
            for model_key, samples in self.model_energy_samples_j.items()
            if samples
        }

    @staticmethod
    def _caption_tokens(caption: Optional[str]) -> set:
        if not caption:
            return set()
        return {
            token.strip(".,!?;:()[]{}\"'").lower()
            for token in caption.split()
            if len(token.strip(".,!?;:()[]{}\"'")) > 2
        }

    def energy_state(self, now: Optional[float] = None) -> dict:
        now = monotonic() if now is None else now
        self.energy_observations = [
            (timestamp, joules)
            for timestamp, joules in self.energy_observations
            if now - timestamp < 60
        ]
        budget = self.settings.get("energy_budget_j_per_min")
        available = bool(self.energy_observations)
        used_joules = sum(joules for _, joules in self.energy_observations)
        return {
            "available": available,
            "budget_j_per_min": budget,
            "used_last_minute_j": used_joules if available else None,
            "energy_debt_j": (
                max(0.0, used_joules - budget)
                if available and budget is not None
                else 0.0
            ),
            "estimated_cost_j": (
                mean(self.energy_samples_j) if self.energy_samples_j else None
            ),
        }

    def record_inference(
        self,
        image: Image.Image,
        now: float,
        latency_ms: float,
        model_key: str,
        answer: str,
        energy_joules: Optional[float],
        timestamp_seconds: Optional[float],
        detailed: bool = False,
        evidence_data_url: Optional[str] = None,
        audio_visual_verified: bool = False,
        recovered: bool = False,
        reason: Optional[str] = None,
    ) -> None:
        self.reference_frame = self._small_grayscale(image)
        self.last_inference_at = now
        self.inference_count += 1
        self.inference_latencies_ms.append(latency_ms)
        self.model_counts[model_key] = self.model_counts.get(model_key, 0) + 1
        if energy_joules is not None:
            self.energy_samples_j.append(energy_joules)
            self.energy_observations.append((now, energy_joules))
            self.model_energy_samples_j.setdefault(model_key, []).append(
                energy_joules
            )
        caption_tokens = self._caption_tokens(answer)
        previous_tokens = self._caption_tokens(self.previous_caption)
        semantic_novelty = None
        if caption_tokens and previous_tokens:
            similarity = len(caption_tokens & previous_tokens) / max(
                len(caption_tokens | previous_tokens),
                1,
            )
            semantic_novelty = 1 - similarity
            event_value = (
                self.gate_decisions[-1].get("event_value", 1.0)
                if self.gate_decisions
                else 1.0
            )
            if (
                semantic_novelty < 0.2
                and event_value >= self.learned_event_threshold
            ):
                self.learned_event_threshold = min(
                    1.8, self.learned_event_threshold + 0.05
                )
                self.threshold_updates += 1
            elif (
                semantic_novelty > 0.55
                and event_value <= self.learned_event_threshold * 1.25
            ):
                self.learned_event_threshold = max(
                    0.65, self.learned_event_threshold - 0.05
                )
                self.threshold_updates += 1
        self.previous_caption = answer
        prompt_terms = {
            token.lower()
            for token in str(self.settings.get("question", "")).split()
            if len(token) > 3
        }
        caption_terms = self._caption_tokens(answer)
        if prompt_terms:
            self.last_query_alignment = len(
                prompt_terms & caption_terms
            ) / len(prompt_terms)
        self.outputs.append(
            {
                "answer": answer,
                "model": model_key,
                "timestamp_seconds": timestamp_seconds,
                "detailed": detailed,
                "semantic_novelty": semantic_novelty,
                "query_alignment": self.last_query_alignment,
                "audio_visual_verified": audio_visual_verified,
                "recovered": recovered,
                "reason": reason,
                "keyframe_data_url": evidence_data_url,
            }
        )

    def summary(self, now: Optional[float] = None) -> dict:
        now = monotonic() if now is None else now
        duration_seconds = max(now - self.started_at, 0.001)
        mean_inference_ms = (
            mean(self.inference_latencies_ms)
            if self.inference_latencies_ms
            else None
        )
        sorted_latencies = sorted(self.inference_latencies_ms)
        p95_index = max(0, int(0.95 * len(sorted_latencies) + 0.5) - 1)
        measured_energy = (
            sum(self.energy_samples_j) if self.energy_samples_j else None
        )
        energy_state = self.energy_state(now)
        observed_event_values = [
            item["event_value"]
            for item in self.gate_decisions
            if "event_value" in item
        ]
        information_benefits = [
            item["expected_information_benefit"]
            for item in self.gate_decisions
            if "expected_information_benefit" in item
        ]
        estimated_energy_costs = [
            item["estimated_energy_cost_j"]
            for item in self.gate_decisions
            if item.get("estimated_energy_cost_j") is not None
        ]
        estimated_latency_costs = [
            item["estimated_latency_cost_ms"]
            for item in self.gate_decisions
            if item.get("estimated_latency_cost_ms") is not None
        ]
        estimated_baseline_energy = (
            measured_energy / self.inference_count * self.total_frames
            if measured_energy is not None and self.inference_count
            else None
        )
        estimated_baseline_latency = (
            mean_inference_ms * self.total_frames
            if mean_inference_ms is not None
            else None
        )
        actual_inference_latency = sum(self.inference_latencies_ms)

        return {
            "sampled_frames": self.total_frames,
            "vlm_invocations": self.inference_count,
            "frames_skipped": self.total_frames - self.inference_count,
            "redundant_frames_skipped": self.skipped_redundant,
            "cooldown_frames_skipped": self.skipped_cooldown,
            "single_result_frames_skipped": self.skipped_single_result,
            "buffered_frames_pending_recovery": len(self.buffered_frames),
            "recovered_frames": self.recovered_frame_count,
            "audio_samples_received": self.audio_sample_count,
            "audio_measurement_source": self.settings.get(
                "audio_source", "unavailable"
            ),
            "min_quality_retention": self.settings.get(
                "min_quality_retention"
            ),
            "benchmark_mode": self.settings.get(
                "benchmark_mode", "adaptive"
            ),
            "audio_visual_events_verified": self.audio_confirmed_count,
            "audio_visual_verification_rate": (
                round(
                    self.audio_confirmed_count / self.total_frames,
                    4,
                )
                if self.total_frames
                else 0.0
            ),
            "query_relevant_frames": self.query_relevant_count,
            "mean_query_relevance": (
                round(
                    mean(
                        item["query_relevance"]
                        for item in self.gate_decisions
                        if item.get("query_relevance") is not None
                    ),
                    3,
                )
                if any(
                    item.get("query_relevance") is not None
                    for item in self.gate_decisions
                )
                else None
            ),
            "missed_event_rate": None,
            "missed_event_rate_note": (
                "Requires annotated ground-truth event timestamps; "
                "use scripts/video_pipeline_benchmark.py."
            ),
            "invocation_reduction_pct": round(
                100 * (self.total_frames - self.inference_count)
                / self.total_frames,
                2,
            )
            if self.total_frames
            else 0.0,
            "duration_seconds": round(duration_seconds, 3),
            "inference_frequency_hz": round(
                self.inference_count / duration_seconds, 3
            ),
            "mean_inference_latency_ms": (
                round(mean_inference_ms, 3)
                if mean_inference_ms is not None
                else None
            ),
            "mean_frame_cycle_ms": (
                round(mean(self.frame_processing_ms), 3)
                if self.frame_processing_ms
                else None
            ),
            "p95_inference_latency_ms": (
                round(sorted_latencies[p95_index], 3)
                if sorted_latencies
                else None
            ),
            "estimated_frame_by_frame_vlm_latency_ms": (
                round(estimated_baseline_latency, 3)
                if estimated_baseline_latency is not None
                else None
            ),
            "event_driven_vlm_latency_ms": round(actual_inference_latency, 3),
            "estimated_vlm_latency_saved_ms": (
                round(estimated_baseline_latency - actual_inference_latency, 3)
                if estimated_baseline_latency is not None
                else None
            ),
            "mean_cpu_percent": (
                round(mean(self.cpu_samples), 2) if self.cpu_samples else None
            ),
            "mean_process_ram_mb": (
                round(mean(self.ram_samples_mb), 2)
                if self.ram_samples_mb
                else None
            ),
            "measured_vlm_cpu_package_energy_j": (
                round(measured_energy, 6)
                if measured_energy is not None
                else None
            ),
            "estimated_frame_by_frame_vlm_energy_j": (
                round(estimated_baseline_energy, 6)
                if estimated_baseline_energy is not None
                else None
            ),
            "estimated_vlm_energy_saved_j": (
                round(estimated_baseline_energy - measured_energy, 6)
                if estimated_baseline_energy is not None
                else None
            ),
            "energy_measurement": (
                "Intel RAPL CPU package counter during VLM generation only"
                if measured_energy is not None
                else "Unavailable; no reliable CPU energy counter detected"
            ),
            "energy_budget_j_per_min": energy_state["budget_j_per_min"],
            "energy_used_last_minute_j": (
                round(energy_state["used_last_minute_j"], 6)
                if energy_state["used_last_minute_j"] is not None
                else None
            ),
            "energy_debt_j": round(energy_state["energy_debt_j"], 6),
            "energy_controller_active": (
                bool(energy_state["budget_j_per_min"])
                and energy_state["available"]
            ),
            "learned_event_threshold": round(self.learned_event_threshold, 3),
            "threshold_updates": self.threshold_updates,
            "mean_event_value": (
                round(mean(observed_event_values), 3)
                if observed_event_values
                else None
            ),
            "mean_expected_information_benefit": (
                round(mean(information_benefits), 3)
                if information_benefits
                else None
            ),
            "estimated_inference_energy_cost_j": (
                round(mean(estimated_energy_costs), 6)
                if estimated_energy_costs
                else None
            ),
            "estimated_inference_latency_cost_ms": (
                round(mean(estimated_latency_costs), 3)
                if estimated_latency_costs
                else None
            ),
            "detailed_inferences": sum(
                bool(item.get("detailed")) for item in self.outputs
            ),
            "model_counts": self.model_counts,
            "outputs": self.outputs,
        }
