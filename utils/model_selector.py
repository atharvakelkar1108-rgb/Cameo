"""
Resource-Aware Model Selection
==============================

Core adaptive-selection module for the project.

The selector considers:

    1. Available RAM
    2. CPU core count
    3. User priority
    4. Measured model latency
    5. Measured model footprint
    6. BLEU-4 quality
    7. Model size

The selector is RULE-BASED and explainable.

Supported model families:

    SmolVLM-256M
        - FP32
        - INT8
        - L1 Pruned
        - Pruned + INT8

    SmolVLM2-256M
        - FP32
        - INT8
        - Q4_K_M / GGUF

Important:
    The GGUF Q4_K_M model is a separate llama.cpp deployment
    backend. It should NOT be loaded with torch.load().
"""

import json
from pathlib import Path


# ============================================================
# Paths
# ============================================================

RESULTS_DIR = (
    Path(__file__).resolve().parent.parent
    / "results"
)


# ============================================================
# Model variants
# ============================================================

MODEL_VARIANTS = {

    # --------------------------------------------------------
    # Existing SmolVLM
    # --------------------------------------------------------

    "baseline": {

        "result_file": "baseline.json",

        "weights_file": None,

        "display_name":
            "SmolVLM FP32 (baseline)",

        "model_family":
            "SmolVLM-256M",

        "backend":
            "pytorch",

        "input_resolution":
            512,

        "compression":
            "FP32",

        "deployment":
            "PC / high-memory device",
    },


    "quantized": {

        "result_file":
            "quantized_dynamic.json",

        "weights_file":
            "quantized_model.pt",

        "display_name":
            "SmolVLM INT8",

        "model_family":
            "SmolVLM-256M",

        "backend":
            "pytorch",

        "input_resolution":
            384,

        "compression":
            "Dynamic INT8",

        "deployment":
            "CPU / edge-capable",
    },


    "pruned": {

        "result_file":
            "pruned.json",

        "weights_file":
            "pruned_model.pt",

        "display_name":
            "SmolVLM Pruned 30%",

        "model_family":
            "SmolVLM-256M",

        "backend":
            "pytorch",

        "input_resolution":
            384,

        "compression":
            "L1 Unstructured Pruning",

        "deployment":
            "Experimental / ablation",
    },


    "optimized": {

        "result_file":
            "optimized.json",

        "weights_file":
            "optimized_model.pt",

        "display_name":
            "SmolVLM INT8 + Pruned 30%",

        "model_family":
            "SmolVLM-256M",

        "backend":
            "pytorch",

        "input_resolution":
            256,

        "compression":
            "Pruning + INT8",

        "deployment":
            "Experimental / ablation",
    },


    # --------------------------------------------------------
    # SmolVLM2 FP32
    # --------------------------------------------------------

    "smolvlm2_baseline": {

        "result_file":
            "smolvlm2_baseline.json",

        "weights_file":
            None,

        "display_name":
            "SmolVLM2-256M FP32",

        "model_family":
            "SmolVLM2-256M",

        "backend":
            "pytorch",

        "input_resolution":
            512,

        "compression":
            "FP32",

        "deployment":
            "PC / high-memory device",
    },


    # --------------------------------------------------------
    # SmolVLM2 INT8
    # --------------------------------------------------------

    "smolvlm2_quantized": {

        "result_file":
            "smolvlm2_quantized.json",

        "weights_file":
            "smolvlm2_quantized_model.pt",

        "display_name":
            "SmolVLM2-256M INT8",

        "model_family":
            "SmolVLM2-256M",

        "backend":
            "pytorch",

        "input_resolution":
            384,

        "compression":
            "Dynamic INT8",

        "deployment":
            "CPU / edge-capable",
    },


    # --------------------------------------------------------
    # SmolVLM2 GGUF Q4_K_M
    # --------------------------------------------------------
    #
    # This is intentionally kept separate from the PyTorch
    # variants because it uses the llama.cpp backend.
    #
    # The result JSON will be created only after you perform
    # the actual GGUF benchmark.
    #

    "smolvlm2_q4": {

        "result_file":
            "smolvlm2_q4_k_m.json",

        "weights_file":
            None,

        "display_name":
            "SmolVLM2-256M Q4_K_M",

        "model_family":
            "SmolVLM2-256M",

        "backend":
            "llama.cpp",

        "input_resolution":
            384,

        "compression":
            "4-bit Q4_K_M",

        "deployment":
            "Raspberry Pi / edge device",
    },
}


# ============================================================
# RAM safety
# ============================================================

# Use 75% of currently available RAM as the safe model budget.
#
# This leaves room for:
#   - operating system
#   - Flask
#   - Python
#   - image processing
#   - request/response data
#
RAM_SAFETY_MARGIN = 0.75


# ============================================================
# Load model profiles
# ============================================================

def load_model_profiles() -> dict:

    profiles = {}

    for key, config in MODEL_VARIANTS.items():

        result_path = (
            RESULTS_DIR
            / config["result_file"]
        )

        # ----------------------------------------------------
        # Skip models that have not been benchmarked yet
        # ----------------------------------------------------

        if not result_path.exists():
            continue

        try:

            with open(
                result_path,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

        except (
            json.JSONDecodeError,
            OSError
        ):

            continue


        # ----------------------------------------------------
        # Prefer isolated footprint measurement
        # ----------------------------------------------------

        footprint = (
            data.get("isolated_footprint_mb")
            or
            data.get("model_footprint_mb")
        )


        # ----------------------------------------------------
        # Build profile
        # ----------------------------------------------------

        profiles[key] = {

            **config,

            "size_mb":
                data.get("size_mb"),

            "footprint_mb":
                footprint,

            "latency_ms":
                data.get("mean_latency_ms"),

            "p95_latency_ms":
                data.get("p95_latency_ms"),

            "bleu4":
                data.get("bleu4"),

            "sparsity_pct":
                data.get("sparsity_pct"),

            "n_eval_samples":
                data.get("n_eval_samples"),

        }


    return profiles


# ============================================================
# Helper functions
# ============================================================

def valid_number(value):

    return (
        value is not None
        and isinstance(value, (int, float))
        and value >= 0
    )


def fits_ram(profile, ram_budget_mb):

    footprint = profile.get(
        "footprint_mb"
    )

    if not valid_number(footprint):
        return False

    return footprint <= ram_budget_mb


# ============================================================
# Main selector
# ============================================================

def select_model(
    ram_available_mb: float,
    cpu_cores: int,
    priority: str = "auto",
    max_latency_ms: float = None,
    min_quality_retention: float = None,
    cached_models=None,
) -> dict:

    profiles = load_model_profiles()


    # --------------------------------------------------------
    # No results
    # --------------------------------------------------------

    if not profiles:

        return {

            "selected": None,

            "explanation":
                "No model results found yet. "
                "Run the model benchmarking scripts first.",

            "profiles_available":
                0,
        }

    if min_quality_retention is not None:
        if not 0.0 <= min_quality_retention <= 1.0:
            return {
                "selected": None,
                "explanation": "Quality retention must be between 0 and 1.",
                "profiles_available": len(profiles),
            }

        baseline_quality = {}
        for profile in profiles.values():
            if (
                profile.get("bleu4") is not None
                and profile["bleu4"] > 0
                and profile.get("model_family")
            ):
                family = profile["model_family"]
                if (
                    profile.get("compression") == "FP32"
                    or profile.get("deployment") == "PC / high-memory device"
                ):
                    baseline_quality[family] = profile["bleu4"]

        quality_filtered = {
            key: profile
            for key, profile in profiles.items()
            if (
                profile.get("bleu4") is not None
                and profile.get("model_family") in baseline_quality
                and profile["bleu4"]
                >= baseline_quality[profile["model_family"]]
                * min_quality_retention
            )
        }
        if not quality_filtered:
            return {
                "selected": None,
                "explanation": (
                    "No benchmarked model meets the requested BLEU-4 "
                    "retention relative to its FP32 family baseline."
                ),
                "profiles_available": len(profiles),
                "quality_retention_required": min_quality_retention,
            }
        profiles = quality_filtered

    # --------------------------------------------------------
    # RAM budget
    # --------------------------------------------------------

    ram_budget_mb = (
        ram_available_mb
        * RAM_SAFETY_MARGIN
    )


    # --------------------------------------------------------
    # Find models that fit
    # --------------------------------------------------------

    fitting = {

        key: profile

        for key, profile in profiles.items()

        if fits_ram(
            profile,
            ram_budget_mb
        )
    }


    fallback_used = False


    # --------------------------------------------------------
    # If nothing fits, use smallest footprint
    # --------------------------------------------------------

    if not fitting:

        if min_quality_retention is not None:
            required_models = ", ".join(
                (
                    f"{key}: {profile.get('footprint_mb', 'unknown')} MB "
                    f"runtime RAM, BLEU-4 "
                    f"{profile.get('bleu4', 'unknown')}"
                )
                for key, profile in profiles.items()
            )
            return {
                "selected": None,
                "explanation": (
                    f"No model meeting the requested BLEU-4 retention "
                    f"({min_quality_retention:.0%}) fits the current safe RAM "
                    f"budget of {ram_budget_mb:.0f} MB "
                    f"({ram_available_mb:.0f} MB available x "
                    f"{RAM_SAFETY_MARGIN:.0%}). Quality-compliant profiles: "
                    f"{required_models or 'none'}. Increase free RAM, choose "
                    "a manual model only if it fits your machine, or lower "
                    "the BLEU-4 retention control to permit a lower-quality "
                    "variant; selection was not silently downgraded."
                ),
                "profiles_available": len(profiles),
                "quality_retention_required": min_quality_retention,
                "ram_available_mb": round(ram_available_mb, 1),
                "ram_budget_mb": round(ram_budget_mb, 1),
                "quality_compliant_profiles": list(profiles),
            }

        valid_profiles = {

            key: profile

            for key, profile in profiles.items()

            if valid_number(
                profile.get("footprint_mb")
            )
        }

        if valid_profiles:

            smallest_key = min(
                valid_profiles,
                key=lambda k:
                    valid_profiles[k]["footprint_mb"]
            )

            fitting = {
                smallest_key:
                    valid_profiles[smallest_key]
            }

            fallback_used = True

        else:

            return {

                "selected": None,

                "explanation":
                    "No model has a valid RAM footprint measurement.",

                "profiles_available":
                    len(profiles),
            }


    # --------------------------------------------------------
    # Optional latency constraint
    # --------------------------------------------------------

    if max_latency_ms:

        latency_filtered = {

            key: profile

            for key, profile in fitting.items()

            if (
                valid_number(
                    profile.get("latency_ms")
                )
                and
                profile["latency_ms"]
                <= max_latency_ms
            )
        }

        if not latency_filtered:
            return {
                "selected": None,
                "explanation": (
                    f"No RAM-compatible, quality-compliant model meets the "
                    f"{max_latency_ms:g} ms latency limit."
                ),
                "profiles_available": len(profiles),
                "quality_retention_required": min_quality_retention,
            }
        fitting = latency_filtered


    # --------------------------------------------------------
    # Normalize priority
    # --------------------------------------------------------

    priority = (
        priority
        or "auto"
    ).lower()


    # --------------------------------------------------------
    # AUTO MODE
    # --------------------------------------------------------
    #
    # Very low RAM:
    #     memory-first
    #
    # Moderate RAM:
    #     balanced
    #
    # High RAM:
    #     balanced
    #
    # We deliberately do NOT blindly choose FP32.
    #

    if priority == "auto":

        if ram_available_mb < 1000:

            priority_used = "memory"

        elif ram_available_mb < 1800:

            priority_used = "fast"

        else:

            priority_used = "balanced"

    else:

        priority_used = priority


    # ========================================================
    # FAST
    # ========================================================

    if priority_used == "fast":

        candidates = {

            k: v

            for k, v in fitting.items()

            if valid_number(
                v.get("latency_ms")
            )
        }

        if candidates:

            chosen_key = min(
                candidates,
                key=lambda k:
                    candidates[k]["latency_ms"]
            )

            reason_metric = (
                "lowest measured inference latency"
            )

        else:

            chosen_key = next(
                iter(fitting)
            )

            reason_metric = (
                "available model with valid resources"
            )


    # ========================================================
    # MEMORY
    # ========================================================

    elif priority_used == "memory":

        candidates = {

            k: v

            for k, v in fitting.items()

            if valid_number(
                v.get("footprint_mb")
            )
        }

        chosen_key = min(
            candidates,
            key=lambda k:
                candidates[k]["footprint_mb"]
        )

        reason_metric = (
            "smallest measured runtime RAM footprint"
        )


    # ========================================================
    # ACCURACY
    # ========================================================

    elif priority_used == "accuracy":

        candidates = {

            k: v

            for k, v in fitting.items()

            if valid_number(
                v.get("bleu4")
            )
        }

        if candidates:

            chosen_key = max(
                candidates,
                key=lambda k:
                    candidates[k]["bleu4"]
            )

            reason_metric = (
                "highest measured BLEU-4"
            )

        else:

            chosen_key = next(
                iter(fitting)
            )

            reason_metric = (
                "available model with valid measurements"
            )


    # ========================================================
    # BALANCED
    # ========================================================

    else:

        candidates = {

            k: v

            for k, v in fitting.items()

            if (
                valid_number(
                    v.get("latency_ms")
                )
                and
                valid_number(
                    v.get("bleu4")
                )
            )
        }


        if not candidates:

            chosen_key = next(
                iter(fitting)
            )

            reason_metric = (
                "available model with valid measurements"
            )

        else:

            latencies = [
                v["latency_ms"]
                for v in candidates.values()
            ]

            bleus = [
                v["bleu4"]
                for v in candidates.values()
            ]


            lat_min = min(latencies)
            lat_max = max(latencies)

            bleu_min = min(bleus)
            bleu_max = max(bleus)


            def score(profile):

                if lat_max > lat_min:

                    latency_score = (
                        1
                        -
                        (
                            (
                                profile["latency_ms"]
                                -
                                lat_min
                            )
                            /
                            (
                                lat_max
                                -
                                lat_min
                            )
                        )
                    )

                else:

                    latency_score = 1.0


                if bleu_max > bleu_min:

                    accuracy_score = (
                        (
                            profile["bleu4"]
                            -
                            bleu_min
                        )
                        /
                        (
                            bleu_max
                            -
                            bleu_min
                        )
                    )

                else:

                    accuracy_score = 1.0


                # Balanced = equal importance
                # for speed and quality.

                return (
                    0.5 * latency_score
                    +
                    0.5 * accuracy_score
                )


            chosen_key = max(
                candidates,
                key=lambda k:
                    score(candidates[k])
            )

            reason_metric = (
                "balanced latency/accuracy score"
            )


    # --------------------------------------------------------
    # Final profile
    # --------------------------------------------------------

    chosen = fitting[chosen_key]


    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    explanation = (

        f"Selected "
        f"'{chosen['display_name']}' "
        f"({reason_metric}). "

        f"Available RAM is "
        f"{ram_available_mb / 1024:.1f} GB. "

        f"Safe model RAM budget is "
        f"{ram_budget_mb:.0f} MB. "

        f"Recommended input resolution: "
        f"{chosen['input_resolution']}x"
        f"{chosen['input_resolution']}."
    )


    # --------------------------------------------------------
    # Add compression/backend information
    # --------------------------------------------------------

    if chosen.get("compression"):

        explanation += (
            f" Compression: "
            f"{chosen['compression']}."
        )


    if chosen.get("backend"):

        explanation += (
            f" Backend: "
            f"{chosen['backend']}."
        )


    # --------------------------------------------------------
    # Fallback warning
    # --------------------------------------------------------

    if fallback_used:

        explanation += (

            " WARNING: no measured model fit "
            "within the RAM safety budget. "

            "The smallest available model "
            "was selected anyway; "
            "swapping or slowdown may occur."
        )


    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {

        "selected":
            chosen_key,

        "profile":
            chosen,

        "explanation":
            explanation,

        "ram_available_mb":
            ram_available_mb,

        "ram_budget_mb":
            round(
                ram_budget_mb,
                1
            ),

        "priority_used":
            priority_used,

        "fallback_used":
            fallback_used,

        "profiles_available":
            len(profiles),

        "candidates":
            list(fitting.keys()),
        "quality_retention_required": min_quality_retention,
        "quality_retention_vs_baseline": (
            round(
                chosen["bleu4"]
                / next(
                    profile["bleu4"]
                    for profile in load_model_profiles().values()
                    if (
                        profile.get("model_family")
                        == chosen.get("model_family")
                        and (
                            profile.get("compression") == "FP32"
                            or profile.get("deployment")
                            == "PC / high-memory device"
                        )
                    )
                ),
                4,
            )
            if min_quality_retention is not None
            and chosen.get("bleu4") is not None
            else None
        ),
    }


# ============================================================
# Manual test
# ============================================================

if __name__ == "__main__":

    print("=" * 65)

    print(
        "RESOURCE-AWARE MODEL SELECTOR TEST"
    )

    print("=" * 65)


    for ram in [
        512,
        1024,
        2048,
        4096
    ]:

        print(
            f"\nAvailable RAM: {ram} MB"
        )

        for mode in [
            "fast",
            "balanced",
            "accuracy",
            "memory",
            "auto"
        ]:

            result = select_model(
                ram_available_mb=ram,
                cpu_cores=4,
                priority=mode
            )

            selected = (
                result.get("selected")
            )

            if selected:

                name = (
                    result["profile"]
                    ["display_name"]
                )

                print(
                    f"  {mode:9s} -> "
                    f"{name}"
                )

            else:

                print(
                    f"  {mode:9s} -> "
                    "NONE"
                )