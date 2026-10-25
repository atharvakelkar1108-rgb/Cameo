# ============================================================
# Resource-Aware Adaptive Multimodal AI Web Application
# ============================================================

import os
import sys
import gc
import json
import math
import re
import time
import threading
import uuid
import base64
from io import BytesIO
from pathlib import Path

import torch
import psutil

from PIL import Image
from flask import Flask, render_template, request, jsonify


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

WEBAPP_DIR = BASE_DIR / "webapp"

RESULTS_DIR = BASE_DIR / "results"

sys.path.insert(
    0,
    str(BASE_DIR)
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from utils.model_selector import (
    load_model_profiles,
    select_model,
    valid_number,
)
from utils.video_pipeline import (
    FRAME_CHANGE_THRESHOLD,
    MAX_INFERENCE_INTERVAL_SECONDS,
    VideoSession,
    energy_delta_joules,
    read_cpu_energy_joules,
)


# ============================================================
# FLASK
# ============================================================

app = Flask(
    __name__,
    template_folder=str(
        WEBAPP_DIR / "templates"
    )
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = (
    "HuggingFaceTB/"
    "SmolVLM-256M-Instruct"
)

DEVICE = "cpu"


# ============================================================
# CPU CONFIGURATION
# ============================================================

try:

    torch.set_num_threads(8)

    torch.set_num_interop_threads(8)

except RuntimeError:

    pass


# ============================================================
# MODEL CACHE
# ============================================================
#
# Weights stay in process memory after the first load. Reloading
# SmolVLM from Hugging Face or a pickled .pt file is typically
# several seconds and dominates the first request. Later requests
# reuse MODEL_CACHE so measured latency is inference only.

MODEL_CACHE = {}

PROCESSOR_CACHE = {}

CACHE_LOCK = threading.RLock()

VIDEO_SESSIONS = {}
VIDEO_SESSIONS_LOCK = threading.Lock()


# ============================================================
# MODEL INFORMATION
# ============================================================

MODEL_INFO = {

    "baseline": {

        "display_name":
            "SmolVLM FP32 (baseline)",

        "resolution":
            512,

        "weights_file":
            None,

        "result_file":
            "baseline.json",
    },


    "quantized": {

        "display_name":
            "SmolVLM INT8",

        "resolution":
            384,

        "weights_file":
            "quantized_model.pt",

        "result_file":
            "quantized_dynamic.json",
    },


    "pruned": {

        "display_name":
            "SmolVLM Pruned 30%",

        "resolution":
            384,

        "weights_file":
            "pruned_model.pt",

        "result_file":
            "pruned.json",
    },


    "optimized": {

        "display_name":
            "SmolVLM INT8 + Pruned 30%",

        "resolution":
            256,

        "weights_file":
            "optimized_model.pt",

        "result_file":
            "optimized.json",
    },
}


# ============================================================
# LOAD PROCESSOR
# ============================================================

def load_processor():

    with CACHE_LOCK:

        if "processor" in PROCESSOR_CACHE:

            return PROCESSOR_CACHE["processor"]


        print()

        print(
            "Loading processor..."
        )


        from transformers import AutoProcessor


        processor = AutoProcessor.from_pretrained(
            MODEL_NAME
        )


        PROCESSOR_CACHE[
            "processor"
        ] = processor


        print(
            "Processor loaded."
        )


        return processor


# ============================================================
# LOAD BASELINE MODEL
# ============================================================

def load_baseline_model():

    with CACHE_LOCK:

        if "baseline" in MODEL_CACHE:

            return MODEL_CACHE[
                "baseline"
            ]


        print()

        print(
            "Loading baseline FP32 model..."
        )


        from transformers import (
            AutoModelForMultimodalLM
        )


        model = (
            AutoModelForMultimodalLM.from_pretrained(
                MODEL_NAME,

                dtype=torch.float32,

                _attn_implementation="eager"
            )
        )


        model.to(
            DEVICE
        )

        model.eval()


        MODEL_CACHE[
            "baseline"
        ] = model


        print(
            "Baseline model loaded."
        )


        return model


# ============================================================
# LOAD SAVED COMPRESSED MODEL
# ============================================================

def load_saved_model(
    model_key
):

    with CACHE_LOCK:

        if model_key in MODEL_CACHE:

            return MODEL_CACHE[
                model_key
            ]


        if model_key not in MODEL_INFO:

            raise ValueError(
                f"Unknown model variant: "
                f"{model_key}"
            )


        weights_file = (
            MODEL_INFO[
                model_key
            ]["weights_file"]
        )


        if weights_file is None:

            return load_baseline_model()


        weights_path = (
            RESULTS_DIR /
            weights_file
        )


        if not weights_path.exists():

            raise FileNotFoundError(
                "Saved model not found: "
                + str(weights_path)
            )


        print()

        print(
            "Loading saved model: "
            + weights_path.name
        )


        # --------------------------------------------------------
        # These .pt files contain complete model objects
        # --------------------------------------------------------

        model = torch.load(

            weights_path,

            weights_only=False,

            map_location=DEVICE
        )


        model.to(
            DEVICE
        )

        model.eval()


        MODEL_CACHE[
            model_key
        ] = model


        print(
            MODEL_INFO[
                model_key
            ]["display_name"]
            + " loaded."
        )


        return model


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    model_key
):

    if model_key == "baseline":

        return load_baseline_model()


    return load_saved_model(
        model_key
    )


def warmup_inference(model, processor):

    if (
        not hasattr(model, "generate")
        or not hasattr(processor, "__call__")
    ):
        return

    dummy = Image.new("RGB", (64, 64), (8, 8, 8))
    try:
        generate_answer(
            model,
            processor,
            dummy,
            task="caption",
        )
    except Exception as error:
        print("Model warmup skipped:", error)


def acquire_model(model_key):

    started = time.perf_counter()
    with CACHE_LOCK:
        cache_hit = (
            model_key in MODEL_CACHE
            and "processor" in PROCESSOR_CACHE
        )

    model = load_model(model_key)
    processor = load_processor()
    if not cache_hit:
        warmup_inference(model, processor)

    return {
        "model": model,
        "processor": processor,
        "cache_hit": cache_hit,
        "load_ms": round(
            (time.perf_counter() - started) * 1000,
            3,
        ),
    }


def models_to_preload():

    requested = os.environ.get(
        "CAMEO_PRELOAD_MODEL",
        "",
    ).strip()
    if requested:
        keys = [
            key.strip().lower()
            for key in requested.split(",")
            if key.strip()
        ]
        return [
            key
            for key in keys
            if key in MODEL_INFO
        ]

    return list(MODEL_INFO.keys())


def preload_default_models():

    keys = models_to_preload()
    if not keys:
        keys = list(MODEL_INFO.keys())

    print()
    print(
        "Preloading processor and all available models into memory..."
    )
    try:
        load_processor()
    except Exception as error:
        print("Startup processor preload failed:", error)
        return

    loaded = []
    for key in keys:
        try:
            acquire_model(key)
            loaded.append(key)
            print(
                "Cached: "
                + MODEL_INFO[key]["display_name"]
            )
        except FileNotFoundError as error:
            print(
                "Skipped "
                + key
                + " (weights not found): "
                + str(error)
            )
        except Exception as error:
            print(
                "Failed to preload "
                + key
                + ": "
                + str(error)
            )

    print()
    if loaded:
        print(
            "Ready. Cached models: "
            + ", ".join(loaded)
        )
    else:
        print(
            "No models were preloaded."
        )


# ============================================================
# LOAD BENCHMARK RESULT
# ============================================================

def load_result_info(
    model_key
):

    if model_key not in MODEL_INFO:

        return {}


    result_file = (
        MODEL_INFO[
            model_key
        ]["result_file"]
    )


    result_path = (
        RESULTS_DIR /
        result_file
    )


    if not result_path.exists():

        return {}


    try:

        with open(
            result_path,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)


    except Exception:

        return {}


# ============================================================
# SYSTEM INFORMATION
# ============================================================

def get_system_stats():

    memory = (
        psutil.virtual_memory()
    )


    process = psutil.Process(
        os.getpid()
    )


    # --------------------------------------------------------
    # CPU
    # --------------------------------------------------------

    cpu_percent = (
        psutil.cpu_percent(
            interval=None
        )
    )


    cpu_cores = (
        os.cpu_count()
        or 1
    )


    # --------------------------------------------------------
    # RAM
    # --------------------------------------------------------

    total_ram_mb = (
        memory.total /
        (1024 * 1024)
    )


    available_ram_mb = (
        memory.available /
        (1024 * 1024)
    )


    process_rss_mb = (
        process.memory_info().rss /
        (1024 * 1024)
    )


    ram_used_pct = (
        memory.percent
    )


    # --------------------------------------------------------
    # BATTERY
    # --------------------------------------------------------

    battery_data = None


    try:

        battery = (
            psutil.sensors_battery()
        )


        if battery is not None:

            battery_data = {

                "percent":
                    battery.percent,

                "plugged":
                    battery.power_plugged,
            }


    except Exception:

        battery_data = None


    # --------------------------------------------------------
    # TEMPERATURE
    # --------------------------------------------------------

    temperature_c = None


    try:

        temperatures = (
            psutil.sensors_temperatures()
        )


        if temperatures:

            values = []


            for entries in (
                temperatures.values()
            ):

                for entry in entries:

                    if (
                        entry.current
                        is not None
                    ):

                        values.append(
                            entry.current
                        )


            if values:

                temperature_c = round(

                    sum(values)
                    / len(values),

                    1
                )


    except Exception:

        temperature_c = None


    return {

        "cpu_percent":
            round(
                cpu_percent,
                1
            ),

        "cpu_cores":
            cpu_cores,

        "total_ram_mb":
            round(
                total_ram_mb,
                3
            ),

        "available_ram_mb":
            round(
                available_ram_mb,
                3
            ),

        "process_rss_mb":
            round(
                process_rss_mb,
                3
            ),

        "ram_used_pct":
            round(
                ram_used_pct,
                1
            ),

        "temperature_c":
            temperature_c,

        "battery":
            battery_data,

        "device":
            DEVICE,

        "cached_models":
            list(MODEL_CACHE.keys()),

        "model_ready":
            bool(MODEL_CACHE) and bool(PROCESSOR_CACHE),
    }


# ============================================================
# RESOURCE-AWARE MODEL SELECTION
# ============================================================

def perform_selection(
    priority,
    max_latency_ms=None,
    energy_profiles=None,
    min_quality_retention=None,
):

    stats = (
        get_system_stats()
    )


    available_ram = (
        stats[
            "available_ram_mb"
        ]
    )


    cpu_cores = (
        stats[
            "cpu_cores"
        ]
    )


    print()

    print(
        "=" * 60
    )

    print(
        "RESOURCE-AWARE MODEL SELECTION"
    )

    print(
        "=" * 60
    )


    print(
        f"Available RAM : "
        f"{available_ram:.1f} MB"
    )


    print(
        f"CPU cores     : "
        f"{cpu_cores}"
    )


    print(
        f"Priority      : "
        f"{priority}"
    )


    # --------------------------------------------------------
    # Existing project selector
    # --------------------------------------------------------

    selection_priority = priority
    if (
        stats["cpu_percent"] >= 85
        and available_ram >= 1000
        and priority in ("auto", "balanced")
    ):
        selection_priority = "fast"

    selection = select_model(

        ram_available_mb=
            available_ram,

        cpu_cores=
            cpu_cores,

        priority=
            selection_priority,

        max_latency_ms=
            max_latency_ms,

        min_quality_retention=
            min_quality_retention,
    )

    measured_energy = energy_profiles or {}
    if (
        priority in ("auto", "balanced")
        and len(measured_energy) >= 2
        and not (selection_priority == "fast" and max_latency_ms is None)
    ):
        profiles = load_model_profiles()
        eligible = [
            key
            for key in selection.get("candidates", [])
            if key in measured_energy and key in profiles
        ]
        if len(eligible) >= 2:
            energy_selected = min(
                eligible,
                key=lambda key: measured_energy[key],
            )
            selection["selected"] = energy_selected
            selection["profile"] = profiles[energy_selected]
            selection["explanation"] += (
                f" Selected for lowest measured CPU-package energy "
                f"({measured_energy[energy_selected]:.4f} J/inference)."
            )
            selection["energy_aware"] = True

    selected_key = selection.get("selected")
    if (
        selected_key
        and selection.get("priority_used") != "accuracy"
        and selected_key not in MODEL_CACHE
    ):
        cached_candidates = [
            key
            for key in selection.get("candidates", [])
            if key in MODEL_CACHE and key in MODEL_INFO
        ]
        if cached_candidates:
            profiles = load_model_profiles()
            reused = min(
                cached_candidates,
                key=lambda key: (
                    profiles[key]["latency_ms"]
                    if key in profiles and valid_number(
                        profiles[key].get("latency_ms")
                    )
                    else float("inf")
                ),
            )
            if reused in profiles:
                selection["selected"] = reused
                selection["profile"] = profiles[reused]
            else:
                selection["selected"] = reused
            selection["reused_cached_model"] = True
            selection["explanation"] = (
                str(selection.get("explanation") or "")
                + " Reused the already-loaded model to avoid reload latency."
            )


    print(
        f"Selected      : "
        f"{selection.get('selected')}"
    )


    print(
        f"Explanation   : "
        f"{selection.get('explanation')}"
    )


    print(
        "=" * 60
    )


    return selection


# ============================================================
# RESOURCE SIMULATION
# ============================================================

def perform_simulated_selection(

    ram_mb,

    cpu_cores,

    priority

):

    # --------------------------------------------------------
    # Use the SAME model selector as the real system.
    #
    # Only the resource values are simulated.
    # --------------------------------------------------------

    selection = select_model(

        ram_available_mb=
            ram_mb,

        cpu_cores=
            cpu_cores,

        priority=
            priority,

        max_latency_ms=
            None
    )


    return selection


# ============================================================
# PREPARE IMAGE
# ============================================================

def prepare_image(
    image,
    resolution
):

    image = (
        image.convert("RGB")
    )


    image = image.copy()


    image.thumbnail(

        (
            resolution,
            resolution
        ),

        Image.Resampling.LANCZOS
    )


    return image


# ============================================================
# GENERATE RESPONSE
# ============================================================

def generate_answer(

    model,

    processor,

    image,

    task="caption",

    question="",

    detailed=False,

):

    # --------------------------------------------------------
    # Caption
    # --------------------------------------------------------

    if task == "caption":

        if detailed:
            prompt = (
                "Carefully inspect the image for the main subject, its action, "
                "and relevant scene details. Give one complete, factual caption "
                "of at most 24 words. Do not answer a question or give a number. "
                "Return only the caption."
            )
        else:
            prompt = (
                "Describe the main subject in one complete, concise sentence "
                "of at most 18 words. Return only that sentence."
            )


    # --------------------------------------------------------
    # VQA
    # --------------------------------------------------------

    elif task == "vqa":

        if question.strip():

            prompt = (
                question.strip()
            )

        else:

            prompt = (
                "What is shown "
                "in this image?"
            )


    # --------------------------------------------------------
    # Default
    # --------------------------------------------------------

    else:

        prompt = (
            "Describe the main subject in one complete, concise sentence "
            "of at most 18 words. Return only that sentence."
        )


    def run_generation(generation_prompt: str, max_new_tokens: int) -> str:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": generation_prompt},
            ],
        }]
        text = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=text,
            images=[image],
            return_tensors="pt",
        )
        inputs = {
            key: value.to(DEVICE) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = generated_ids[:, input_length:]
        return processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )[0].strip()

    def clean_caption(raw_answer: str) -> str:
        answer = re.sub(
            r"^(?:(?:the\s+)?answer\s+is|answer|a)\s*:\s*",
            "",
            raw_answer.strip(),
            count=1,
            flags=re.IGNORECASE,
        )
        answer = re.sub(r"\s+", " ", answer).strip(" \t\r\n\"'")
        complete_sentence = re.search(r"^(.+?[.!?])(?:\s|$)", answer)
        if complete_sentence:
            answer = complete_sentence.group(1)
        elif answer and answer[-1] not in ".!?":
            answer += "."
        return answer

    answer = run_generation(
        prompt,
        max_new_tokens=64 if detailed else (
            48 if task == "caption" else 64
        ),
    )
    if task != "caption":
        return answer

    answer = clean_caption(answer)
    def is_caption(value: str) -> bool:
        return (
            len(re.findall(r"[A-Za-z]{2,}", value)) >= 3
            and not re.fullmatch(r"[\W\d_]+", value)
            and not re.match(
                r"^(?:the\s+)?answer\s+is\b",
                value,
                re.IGNORECASE,
            )
        )

    if not is_caption(answer):
        answer = clean_caption(run_generation(
            "Do not answer with a number or answer label. Describe the visible "
            "objects, scene, or action in one complete, factual sentence of "
            "at most 18 words. Return only the caption.",
            max_new_tokens=48,
        ))
        if not is_caption(answer):
            answer = (
                "The model could not produce a reliable caption for this frame."
            )
    return answer


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# FAVICON
# ============================================================

@app.route("/favicon.ico")
def favicon():

    return "", 204


# ============================================================
# SYSTEM STATS API
# ============================================================

@app.route(
    "/api/system_stats"
)
def system_stats():

    try:

        return jsonify(
            get_system_stats()
        )


    except Exception as e:

        print(
            "System stats error:",
            e
        )


        return jsonify({

            "error":
                str(e)

        }), 500


# ============================================================
# COMPARISON API
# ============================================================

@app.route(
    "/api/comparison"
)
def comparison():

    rows = []


    model_order = [

        "baseline",

        "quantized",

        "pruned",

        "optimized"

    ]


    for model_key in model_order:

        info = MODEL_INFO[
            model_key
        ]


        result = load_result_info(
            model_key
        )


        if not result:

            continue


        footprint = (

            result.get(
                "isolated_footprint_mb"
            )

            or

            result.get(
                "model_footprint_mb"
            )
        )


        row = {

            "model":
                info[
                    "display_name"
                ],

            "variant":
                model_key,

            "size_mb":
                result.get(
                    "size_mb"
                ),

            "footprint_mb":
                footprint,

            "latency_ms":
                result.get(
                    "mean_latency_ms"
                ),

            "p95_latency_ms":
                result.get(
                    "p95_latency_ms"
                ),

            "bleu4":
                result.get(
                    "bleu4"
                ),

            "sparsity_pct":
                result.get(
                    "sparsity_pct"
                ),
        }


        rows.append(
            row
        )


    return jsonify({

        "rows":
            rows

    })


# ============================================================
# RESOURCE SIMULATION API
# ============================================================

@app.route(
    "/api/simulate-selection",
    methods=["POST"]
)
def simulate_selection():

    try:

        data = (
            request.get_json(
                silent=True
            )
        )


        if not data:

            return jsonify({

                "success":
                    False,

                "error":
                    "No simulation data received."

            }), 400


        # ----------------------------------------------------
        # Read simulated values
        # ----------------------------------------------------

        ram_mb = float(

            data.get(
                "ram_mb",
                4096
            )

        )


        cpu_cores = int(

            data.get(
                "cpu_cores",
                8
            )

        )


        priority = (

            data.get(
                "priority",
                "balanced"
            )

            or

            "balanced"

        ).lower()


        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if ram_mb <= 0:

            return jsonify({

                "success":
                    False,

                "error":
                    "RAM must be greater than 0."

            }), 400


        if cpu_cores <= 0:

            return jsonify({

                "success":
                    False,

                "error":
                    "CPU cores must be greater than 0."

            }), 400


        valid_priorities = {

            "auto",

            "fast",

            "balanced",

            "accuracy",

            "memory"

        }


        if priority not in valid_priorities:

            priority = "balanced"


        # ----------------------------------------------------
        # Run selector
        # ----------------------------------------------------

        selection = (
            perform_simulated_selection(

                ram_mb,

                cpu_cores,

                priority
            )
        )


        selected_key = (
            selection.get(
                "selected"
            )
        )


        if selected_key not in MODEL_INFO:

            return jsonify({

                "success":
                    False,

                "error":
                    "Selector did not return "
                    "a valid model."

            }), 500


        # ----------------------------------------------------
        # Model information
        # ----------------------------------------------------

        model_info = (
            MODEL_INFO[
                selected_key
            ]
        )


        result_info = (
            load_result_info(
                selected_key
            )
        )


        footprint = (

            result_info.get(
                "isolated_footprint_mb"
            )

            or

            result_info.get(
                "model_footprint_mb"
            )
        )


        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return jsonify({

            "success":
                True,

            "simulated":
                True,

            "ram_mb":
                round(
                    ram_mb,
                    2
                ),

            "cpu_cores":
                cpu_cores,

            "priority":
                priority,

            "selected":
                selected_key,

            "display_name":
                model_info[
                    "display_name"
                ],

            "resolution":
                model_info[
                    "resolution"
                ],

            "size_mb":
                result_info.get(
                    "size_mb"
                ),

            "footprint_mb":
                footprint,

            "latency_ms":
                result_info.get(
                    "mean_latency_ms"
                ),

            "p95_latency_ms":
                result_info.get(
                    "p95_latency_ms"
                ),

            "bleu4":
                result_info.get(
                    "bleu4"
                ),

            "sparsity_pct":
                result_info.get(
                    "sparsity_pct"
                ),

            "explanation":
                selection.get(
                    "explanation"
                ),

            "ram_budget_mb":
                selection.get(
                    "ram_budget_mb"
                ),

            "profile":
                selection.get(
                    "profile",
                    {}
                )

        })


    except Exception as e:

        print()

        print(
            "=" * 60
        )

        print(
            "SIMULATION ERROR"
        )

        print(
            "=" * 60
        )

        print(
            str(e)
        )

        print(
            "=" * 60
        )


        return jsonify({

            "success":
                False,

            "error":
                str(e)

        }), 500


# ============================================================
# PREDICTION API
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def predict():

    request_start = (
        time.perf_counter()
    )


    try:

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        if (
            "image"
            not in request.files
        ):

            return jsonify({

                "error":
                    "No image was uploaded."

            }), 400


        image_file = (
            request.files[
                "image"
            ]
        )


        if not image_file.filename:

            return jsonify({

                "error":
                    "No image was selected."

            }), 400


        # ----------------------------------------------------
        # FORM VALUES
        # ----------------------------------------------------

        requested_variant = (

            request.form.get(
                "variant",
                "quantized"
            )

            or

            "quantized"

        ).lower()


        adaptive = (

            request.form.get(
                "adaptive",
                "false"
            )

            or

            "false"

        ).lower() == "true"


        priority = (

            request.form.get(
                "priority",
                "auto"
            )

            or

            "auto"

        ).lower()


        task = (

            request.form.get(
                "task",
                "caption"
            )

            or

            "caption"

        ).lower()


        question = (

            request.form.get(
                "question",
                ""
            )

            or

            ""
        )

        max_latency_value = request.form.get("max_latency_ms")
        try:
            max_latency_ms = (
                float(max_latency_value)
                if max_latency_value
                else None
            )
        except ValueError:
            return jsonify({
                "error": "Latency target must be a positive number."
            }), 400
        if max_latency_ms is not None and (
            max_latency_ms <= 0 or not math.isfinite(max_latency_ms)
        ):
            return jsonify({
                "error": "Latency target must be a positive number."
            }), 400


        valid_priorities = {

            "auto",

            "fast",

            "balanced",

            "accuracy",

            "memory"

        }


        if (
            priority
            not in valid_priorities
        ):

            priority = "auto"


        # ----------------------------------------------------
        # MODEL SELECTION
        # ----------------------------------------------------

        selector_result = None


        if adaptive:

            selector_result = (
                perform_selection(
                    priority,
                    max_latency_ms,
                )
            )


            selected_key = (
                selector_result.get(
                    "selected"
                )
            )


            if selected_key is None:

                return jsonify({

                    "error":
                        selector_result.get(

                            "explanation",

                            "No suitable model found."

                        )

                }), 500


            auto_selected = True


        else:

            selected_key = (
                requested_variant
            )

            auto_selected = False


        # ----------------------------------------------------
        # VALIDATE MODEL
        # ----------------------------------------------------

        if (
            selected_key
            not in MODEL_INFO
        ):

            return jsonify({

                "error":
                    "Invalid model variant: "
                    + str(
                        selected_key
                    )

            }), 400

        if max_latency_ms is not None:
            measured_latency = load_result_info(selected_key).get(
                "mean_latency_ms"
            )
            if not isinstance(measured_latency, (int, float)):
                return jsonify({
                    "error": (
                        f"No benchmark latency is available for {selected_key}; "
                        "cannot enforce the requested latency target."
                    )
                }), 422
            if measured_latency > max_latency_ms:
                return jsonify({
                    "error": (
                        f"No selected model meets the {max_latency_ms:g} ms "
                        f"latency target; {selected_key} is benchmarked at "
                        f"{measured_latency:g} ms."
                    )
                }), 422


        # ----------------------------------------------------
        # LOAD MODEL (cached after the first request)
        # ----------------------------------------------------

        runtime = acquire_model(
            selected_key
        )
        model = runtime["model"]
        processor = runtime["processor"]


        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        image = Image.open(
            image_file.stream
        ).convert("RGB")


        # ----------------------------------------------------
        # RESOLUTION
        # ----------------------------------------------------

        if selector_result:

            profile = (
                selector_result.get(
                    "profile",
                    {}
                )
            )


            resolution = profile.get(

                "input_resolution",

                MODEL_INFO[
                    selected_key
                ]["resolution"]

            )


        else:

            resolution = (
                MODEL_INFO[
                    selected_key
                ]["resolution"]
            )


        image = prepare_image(

            image,

            int(
                resolution
            )

        )


        # ----------------------------------------------------
        # INFERENCE
        # ----------------------------------------------------

        inference_start = (
            time.perf_counter()
        )


        answer = generate_answer(

            model,

            processor,

            image,

            task,

            question
        )


        latency_ms = (

            time.perf_counter()
            - inference_start

        ) * 1000


        # ----------------------------------------------------
        # BENCHMARK INFORMATION
        # ----------------------------------------------------

        result_info = (
            load_result_info(
                selected_key
            )
        )


        size_mb = (
            result_info.get(
                "size_mb"
            )
        )


        footprint_mb = (

            result_info.get(
                "isolated_footprint_mb"
            )

            or

            result_info.get(
                "model_footprint_mb"
            )
        )


        display_name = (
            MODEL_INFO[
                selected_key
            ]["display_name"]
        )


        # ----------------------------------------------------
        # SELECTOR INFORMATION
        # ----------------------------------------------------

        selector_payload = None


        if selector_result:

            selector_payload = {

                "selected":
                    selector_result.get(
                        "selected"
                    ),

                "explanation":
                    selector_result.get(
                        "explanation"
                    ),

                "priority_used":
                    selector_result.get(
                        "priority_used"
                    ),

                "ram_available_mb":
                    selector_result.get(
                        "ram_available_mb"
                    ),

                "ram_budget_mb":
                    selector_result.get(
                        "ram_budget_mb"
                    ),

                "fallback_used":
                    selector_result.get(
                        "fallback_used"
                    )
            }


        # ----------------------------------------------------
        # TOTAL REQUEST TIME
        # ----------------------------------------------------

        total_latency_ms = (

            time.perf_counter()
            - request_start

        ) * 1000


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        response = {

            "success":
                True,

            "answer":
                answer,

            "latency_ms":
                round(
                    latency_ms,
                    3
                ),

            "size_mb":
                size_mb,

            "footprint_mb":
                footprint_mb,

            "display_name":
                display_name,

            "model":
                selected_key,

            "variant":
                selected_key,

            "model_name":
                display_name,

            "auto_selected":
                auto_selected,

            "selector":
                selector_payload,

            "priority":
                priority,

            "task":
                task,

            "resolution":
                int(
                    resolution
                ),

            "total_latency_ms":
                round(
                    total_latency_ms,
                    3
                ),

            "model_load_ms":
                runtime["load_ms"],

            "cache_hit":
                runtime["cache_hit"],

            "device":
                DEVICE
        }


        return jsonify(
            response
        )


    except Exception as e:

        print()

        print(
            "=" * 60
        )

        print(
            "PREDICTION ERROR"
        )

        print(
            "=" * 60
        )

        print(
            str(e)
        )

        print(
            "=" * 60
        )


        return jsonify({

            "success":
                False,

            "error":
                str(e)

        }), 500


# ============================================================
# EVENT-DRIVEN VIDEO AND CAMERA API
# ============================================================

def encode_video_evidence(image):
    thumbnail = image.copy()
    thumbnail.thumbnail((480, 480), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    thumbnail.save(buffer, format="JPEG", quality=76)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def infer_video_candidate(
    video_session,
    image,
    timestamp_seconds,
    analysis,
    recovered=False,
):
    settings = video_session.settings
    selector_result = None
    accuracy_override = (
        analysis.get("uncertain", False)
        or analysis.get("audio_visual_verified", False)
        or analysis.get("query_relevance", 0.0) >= 0.75
    )
    if (
        settings["adaptive"]
        and video_session.pinned_model
        and video_session.pinned_model in MODEL_INFO
        and not accuracy_override
    ):
        selected_key = video_session.pinned_model
        selector_result = {
            "selected": selected_key,
            "profile": {},
            "explanation": (
                "Pinned the already-loaded model for this video session "
                "to avoid reload latency."
            ),
            "reused_cached_model": True,
        }
    elif settings["adaptive"]:
        uncertain_or_verified = (
            analysis.get("uncertain", False)
            or analysis.get("audio_visual_verified", False)
            or analysis.get("query_relevance", 0.0) >= 0.75
        )
        quality_retention = min(
            1.0,
            round(
                settings["min_quality_retention"]
                + (0.05 if uncertain_or_verified else 0.0),
                4,
            ),
        )
        selection_priority = (
            "accuracy" if uncertain_or_verified else settings["priority"]
        )
        selector_result = perform_selection(
            selection_priority,
            settings["max_latency_ms"],
            video_session.energy_profiles(),
            min_quality_retention=quality_retention,
        )
        selected_key = selector_result.get("selected")
        if selected_key is None:
            return {
                "error": selector_result.get(
                    "explanation", "No suitable model variant is available."
                )
            }, 422
    else:
        selected_key = settings["variant"]

    result_info = load_result_info(selected_key)
    measured_latency = result_info.get("mean_latency_ms")
    latency_target = settings["max_latency_ms"]
    if latency_target is not None and not isinstance(
        measured_latency, (int, float)
    ):
        return {
            "error": (
                f"No benchmark latency is available for {selected_key}; "
                "cannot enforce the requested latency target."
            )
        }, 422
    if latency_target is not None and measured_latency > latency_target:
        return {
            "error": (
                f"No selected model meets the {latency_target:g} ms latency "
                f"target; {selected_key} is benchmarked at "
                f"{measured_latency:g} ms."
            )
        }, 422

    stats = get_system_stats()
    video_session.record_resources(
        stats["cpu_percent"], stats["process_rss_mb"]
    )
    runtime = acquire_model(selected_key)
    model = runtime["model"]
    processor = runtime["processor"]
    profile = selector_result.get("profile", {}) if selector_result else {}
    resolution = profile.get(
        "input_resolution", MODEL_INFO[selected_key]["resolution"]
    )
    prepared_image = prepare_image(image, int(resolution))
    energy_start = read_cpu_energy_joules()
    inference_started = time.perf_counter()
    answer = generate_answer(
        model,
        processor,
        prepared_image,
        settings["task"],
        settings["question"],
        detailed=analysis.get("detailed_analysis", False),
    )
    latency_ms = (time.perf_counter() - inference_started) * 1000
    energy_joules = energy_delta_joules(
        energy_start, read_cpu_energy_joules()
    )
    post_inference_stats = get_system_stats()
    video_session.record_resources(
        post_inference_stats["cpu_percent"],
        post_inference_stats["process_rss_mb"],
    )
    evidence = encode_video_evidence(image)
    video_session.record_inference(
        image,
        time.monotonic(),
        latency_ms,
        selected_key,
        answer,
        energy_joules,
        timestamp_seconds,
        detailed=analysis.get("detailed_analysis", False),
        evidence_data_url=evidence,
        audio_visual_verified=analysis.get("audio_visual_verified", False),
        recovered=recovered,
        reason=analysis.get("reason"),
    )
    if recovered:
        video_session.recovered_frame_count += 1
    video_session.pinned_model = selected_key

    return {
        "answer": answer,
        "model": selected_key,
        "model_display": MODEL_INFO[selected_key]["display_name"],
        "latency_ms": round(latency_ms, 3),
        "energy_joules": energy_joules,
        "selector": selector_result,
        "timestamp_seconds": timestamp_seconds,
        "keyframe_data_url": evidence,
        "detailed_analysis": analysis.get("detailed_analysis", False),
        "audio_visual_verified": analysis.get(
            "audio_visual_verified", False
        ),
        "recovered": recovered,
        "reason": analysis.get("reason"),
        "query_relevance": analysis.get("query_relevance", 0.0),
    }, 200


@app.route("/api/video/start", methods=["POST"])
def start_video_session():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Expected a JSON configuration object."}), 400

    variant = str(payload.get("variant", "quantized")).lower()
    priority = str(payload.get("priority", "auto")).lower()
    task = str(payload.get("task", "caption")).lower()
    adaptive = payload.get("adaptive") is True
    max_latency_ms = payload.get("max_latency_ms")
    energy_budget = payload.get("energy_budget_j_per_min")
    min_quality_retention = payload.get("min_quality_retention", 0.9)
    benchmark_mode = payload.get("benchmark_mode", "adaptive")
    audio_source = payload.get("audio_source", "unavailable")
    if variant not in MODEL_INFO:
        return jsonify({"error": "Invalid model variant."}), 400
    if priority not in {"auto", "fast", "balanced", "accuracy", "memory"}:
        return jsonify({"error": "Invalid optimization priority."}), 400
    if task not in {"caption", "vqa"}:
        return jsonify({"error": "Task must be caption or vqa."}), 400
    if not isinstance(benchmark_mode, str) or benchmark_mode not in {
        "adaptive", "frame_by_frame"
    }:
        return jsonify({"error": "Invalid video benchmark mode."}), 400
    if not isinstance(audio_source, str) or audio_source not in {
        "unavailable", "browser decoded file audio", "browser microphone"
    }:
        return jsonify({"error": "Invalid audio measurement source."}), 400
    try:
        min_quality_retention = float(min_quality_retention)
    except (TypeError, ValueError):
        return jsonify({
            "error": "Minimum quality retention must be between 0 and 1."
        }), 400
    if (
        not math.isfinite(min_quality_retention)
        or not 0 <= min_quality_retention <= 1
    ):
        return jsonify({
            "error": "Minimum quality retention must be between 0 and 1."
        }), 400
    if max_latency_ms is not None:
        try:
            max_latency_ms = float(max_latency_ms)
        except (TypeError, ValueError):
            return jsonify({"error": "Latency target must be a positive number."}), 400
        if max_latency_ms <= 0 or not math.isfinite(max_latency_ms):
            return jsonify({"error": "Latency target must be a positive number."}), 400
    if energy_budget is not None:
        try:
            energy_budget = float(energy_budget)
        except (TypeError, ValueError):
            return jsonify({
                "error": "Energy budget must be a positive number of joules per minute."
            }), 400
        if energy_budget <= 0 or not math.isfinite(energy_budget):
            return jsonify({
                "error": "Energy budget must be a positive number of joules per minute."
            }), 400

    now = time.monotonic()
    with VIDEO_SESSIONS_LOCK:
        expired = [
            key
            for key, session in VIDEO_SESSIONS.items()
            if now - session.started_at > 3600
        ]
        for key in expired:
            del VIDEO_SESSIONS[key]
        if len(VIDEO_SESSIONS) >= 20:
            return jsonify({
                "error": "Video processing is busy; try again when a session finishes."
            }), 429

        session_id = uuid.uuid4().hex
        VIDEO_SESSIONS[session_id] = VideoSession(settings={
            "variant": variant,
            "priority": priority,
            "task": task,
            "question": str(payload.get("question", ""))[:500],
            "adaptive": adaptive,
            "max_latency_ms": max_latency_ms,
            "single_result": payload.get("single_result") is True,
            "energy_budget_j_per_min": energy_budget,
            "min_quality_retention": min_quality_retention,
            "benchmark_mode": benchmark_mode,
            "audio_source": audio_source,
        })

    return jsonify({
        "session_id": session_id,
        "sample_interval_ms": 500,
        "event_threshold": FRAME_CHANGE_THRESHOLD,
        "heartbeat_seconds": MAX_INFERENCE_INTERVAL_SECONDS,
    })


@app.route("/api/video/frame", methods=["POST"])
def process_video_frame():
    session_id = request.form.get("session_id", "")
    with VIDEO_SESSIONS_LOCK:
        video_session = VIDEO_SESSIONS.get(session_id)
    if video_session is None:
        return jsonify({"error": "Video session not found or expired."}), 404
    if "image" not in request.files:
        return jsonify({"error": "No video frame was uploaded."}), 400
    if request.content_length and request.content_length > 8 * 1024 * 1024:
        return jsonify({"error": "Video frame exceeds the 8 MB upload limit."}), 413

    try:
        image = Image.open(request.files["image"].stream).convert("RGB")
    except (OSError, Image.DecompressionBombError) as error:
        return jsonify({"error": f"Could not read video frame: {error}"}), 400

    timestamp_value = request.form.get("timestamp_seconds")
    try:
        timestamp_seconds = (
            float(timestamp_value) if timestamp_value is not None else None
        )
    except ValueError:
        return jsonify({"error": "Frame timestamp must be numeric."}), 400
    if timestamp_seconds is not None and (
        not math.isfinite(timestamp_seconds) or timestamp_seconds < 0
    ):
        return jsonify({"error": "Frame timestamp must be non-negative."}), 400
    audio_rms_value = request.form.get("audio_rms")
    try:
        audio_rms = (
            float(audio_rms_value) if audio_rms_value is not None else None
        )
    except ValueError:
        return jsonify({"error": "Audio RMS must be numeric."}), 400
    if audio_rms is not None and (
        not math.isfinite(audio_rms) or not 0 <= audio_rms <= 1
    ):
        return jsonify({"error": "Audio RMS must be between 0 and 1."}), 400

    frame_started = time.perf_counter()
    stats = get_system_stats()
    video_session.record_resources(
        stats["cpu_percent"],
        stats["process_rss_mb"],
    )
    analysis = video_session.analyze_frame(
        image,
        timestamp_seconds=timestamp_seconds,
        audio_rms=audio_rms,
    )
    if not analysis["should_infer"]:
        video_session.frame_processing_ms.append(
            (time.perf_counter() - frame_started) * 1000
        )
        return jsonify({
            "success": True,
            "should_infer": False,
            "reason": analysis["reason"],
            "mean_difference": round(analysis["mean_difference"], 4),
            "changed_fraction": round(analysis["changed_fraction"], 4),
            "event_value": round(analysis.get("event_value", 0.0), 3),
            "expected_information_benefit": round(
                analysis.get("expected_information_benefit", 0.0), 3
            ),
            "benefit_cost_ratio": analysis.get("benefit_cost_ratio"),
            "estimated_latency_cost_ms": analysis.get(
                "estimated_latency_cost_ms"
            ),
            "estimated_energy_cost_j": analysis.get(
                "estimated_energy_cost_j"
            ),
            "uncertain": analysis.get("uncertain", False),
            "query_relevance": round(
                analysis.get("query_relevance", 0.0), 3
            ),
            "audio_visual_verified": analysis.get(
                "audio_visual_verified", False
            ),
            "energy_debt_j": round(analysis.get("energy_debt_j", 0.0), 4),
            "effective_threshold": round(
                analysis.get("effective_threshold", 1.0), 3
            ),
            "sampled_frames": video_session.total_frames,
        })

    try:
        recovery = video_session.take_recovery_candidate(
            current_timestamp_seconds=timestamp_seconds
        )
        recovered_outputs = []
        if recovery:
            recovered_result, status = infer_video_candidate(
                video_session,
                recovery["image"],
                recovery["timestamp_seconds"],
                recovery,
                recovered=True,
            )
            if status != 200:
                return jsonify(recovered_result), status
            recovered_outputs.append(recovered_result)
        result, status = infer_video_candidate(
            video_session, image, timestamp_seconds, analysis
        )
        if status != 200:
            return jsonify(result), status
    except Exception as error:
        app.logger.exception("Video-frame inference failed")
        return jsonify({"error": str(error)}), 500

    video_session.frame_processing_ms.append(
        (time.perf_counter() - frame_started) * 1000
    )
    return jsonify({
        "success": True,
        "should_infer": True,
        "reason": analysis["reason"],
        "mean_difference": round(analysis["mean_difference"], 4),
        "changed_fraction": round(analysis["changed_fraction"], 4),
        **result,
        "answer": result["answer"],
        "recovered_outputs": recovered_outputs,
        "detailed_analysis": analysis.get("detailed_analysis", False),
        "uncertain_event": analysis.get("uncertain", False),
        "audio_visual_verified": analysis.get(
            "audio_visual_verified", False
        ),
        "query_relevance": round(
            analysis.get("query_relevance", 0.0), 3
        ),
        "event_value": round(analysis.get("event_value", 0.0), 3),
        "expected_information_benefit": round(
            analysis.get("expected_information_benefit", 0.0), 3
        ),
        "benefit_cost_ratio": analysis.get("benefit_cost_ratio"),
        "estimated_latency_cost_ms": analysis.get(
            "estimated_latency_cost_ms"
        ),
        "estimated_energy_cost_j": analysis.get(
            "estimated_energy_cost_j"
        ),
        "energy_debt_j": round(analysis.get("energy_debt_j", 0.0), 4),
        "sampled_frames": video_session.total_frames,
        "vlm_invocations": video_session.inference_count,
    })


@app.route("/api/video/stop", methods=["POST"])
def stop_video_session():
    payload = request.get_json(silent=True)
    session_id = payload.get("session_id") if isinstance(payload, dict) else None
    with VIDEO_SESSIONS_LOCK:
        video_session = VIDEO_SESSIONS.pop(session_id, None)
    if video_session is None:
        return jsonify({"error": "Video session not found or already stopped."}), 404

    if video_session.settings.get("single_result") and video_session.best_frame:
        settings = video_session.settings
        selector_result = None
        if settings["adaptive"]:
            stricter_quality = (
                video_session.best_frame_detailed
                or video_session.best_frame_audio_verified
            )
            selector_result = perform_selection(
                "accuracy" if stricter_quality else settings["priority"],
                settings["max_latency_ms"],
                video_session.energy_profiles(),
                min_quality_retention=min(
                    1.0,
                    round(
                        settings["min_quality_retention"]
                        + (0.05 if stricter_quality else 0.0),
                        4,
                    ),
                ),
            )
            model_key = selector_result.get("selected")
            if model_key is None:
                return jsonify({
                    "error": selector_result.get(
                        "explanation", "No suitable model variant is available."
                    )
                }), 422
        else:
            model_key = settings["variant"]

        if model_key not in MODEL_INFO:
            return jsonify({
                "error": (
                    f"The selected model variant {model_key!r} is not "
                    "supported by this web app."
                )
            }), 422

        result_info = load_result_info(model_key)
        measured_latency = result_info.get("mean_latency_ms")
        latency_target = settings["max_latency_ms"]
        if latency_target is not None and not isinstance(
            measured_latency, (int, float)
        ):
            return jsonify({
                "error": (
                    f"No benchmark latency is available for {model_key}; "
                    "cannot enforce the requested latency target."
                )
            }), 422
        if latency_target is not None and measured_latency > latency_target:
            return jsonify({
                "error": (
                    f"No selected model meets the {latency_target:g} ms latency "
                    f"target; {model_key} is benchmarked at "
                    f"{measured_latency:g} ms."
                )
            }), 422

        try:
            stats = get_system_stats()
            video_session.record_resources(
                stats["cpu_percent"],
                stats["process_rss_mb"],
            )
            runtime = acquire_model(model_key)
            model = runtime["model"]
            processor = runtime["processor"]
            profile = selector_result.get("profile", {}) if selector_result else {}
            resolution = profile.get(
                "input_resolution",
                MODEL_INFO[model_key]["resolution"],
            )
            prepared_image = prepare_image(video_session.best_frame, int(resolution))
            energy_start = read_cpu_energy_joules()
            inference_started = time.perf_counter()
            answer = generate_answer(
                model,
                processor,
                prepared_image,
                settings["task"],
                settings["question"],
                detailed=video_session.best_frame_detailed,
            )
            latency_ms = (time.perf_counter() - inference_started) * 1000
            energy_joules = energy_delta_joules(
                energy_start,
                read_cpu_energy_joules(),
            )
            stats = get_system_stats()
            video_session.record_resources(
                stats["cpu_percent"],
                stats["process_rss_mb"],
            )
            video_session.record_inference(
                video_session.best_frame,
                time.monotonic(),
                latency_ms,
                model_key,
                answer,
                energy_joules,
                video_session.best_frame_timestamp_seconds,
                detailed=video_session.best_frame_detailed,
                evidence_data_url=encode_video_evidence(
                    video_session.best_frame
                ),
                audio_visual_verified=(
                    video_session.best_frame_audio_verified
                ),
                reason="single_result_best_frame",
            )
        except Exception as error:
            app.logger.exception("Single-result video inference failed")
            return jsonify({"error": str(error)}), 500

    metrics = video_session.summary()
    if video_session.model_counts:
        model_key = max(
            video_session.model_counts,
            key=video_session.model_counts.get,
        )
        result_info = load_result_info(model_key)
        metrics.update({
            "selected_model": model_key,
            "selected_model_display": MODEL_INFO[model_key]["display_name"],
            "model_size_mb": result_info.get("size_mb"),
            "model_footprint_mb": (
                result_info.get("isolated_footprint_mb")
                or result_info.get("model_footprint_mb")
            ),
        })
    return jsonify({"success": True, "metrics": metrics})


# ============================================================
# HEALTH API
# ============================================================

@app.route(
    "/api/health"
)
def health():

    stats = (
        get_system_stats()
    )


    return jsonify({

        "status":
            "ok",

        "device":
            DEVICE,

        "cpu_cores":
            os.cpu_count()
            or 1,

        "available_ram_mb":
            stats[
                "available_ram_mb"
            ],

        "cached_models":
            list(
                MODEL_CACHE.keys()
            ),

        "model_ready":
            bool(MODEL_CACHE) and bool(PROCESSOR_CACHE),
    })


# ============================================================
# CLEAR CACHE
# ============================================================

@app.route(
    "/api/clear-cache",
    methods=["POST"]
)
def clear_cache():

    with CACHE_LOCK:
        MODEL_CACHE.clear()
        PROCESSOR_CACHE.clear()

    gc.collect()


    return jsonify({

        "success":
            True,

        "message":
            "Model cache cleared."

    })


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()

    print(
        "=" * 65
    )

    print(
        "RESOURCE-AWARE MULTIMODAL AI WEB APPLICATION"
    )

    print(
        "=" * 65
    )


    print(
        f"Model : {MODEL_NAME}"
    )


    print(
        f"Device: {DEVICE}"
    )


    print(
        f"CPU cores: "
        f"{os.cpu_count() or 1}"
    )


    print(

        "Available RAM: "

        f"{get_system_stats()['available_ram_mb']:.1f}"

        " MB"

    )


    print(
        "=" * 65
    )

    if os.environ.get("CAMEO_PRELOAD", "1") != "0":
        preload_default_models()

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=False,

        threaded=True
    )