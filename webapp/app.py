# ============================================================
# Resource-Aware Adaptive Multimodal AI Web Application
# ============================================================

import os
import sys
import gc
import json
import time
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

from utils.model_selector import select_model


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

MODEL_CACHE = {}

PROCESSOR_CACHE = {}


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
    }


# ============================================================
# RESOURCE-AWARE MODEL SELECTION
# ============================================================

def perform_selection(
    priority
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

    selection = select_model(

        ram_available_mb=
            available_ram,

        cpu_cores=
            cpu_cores,

        priority=
            priority,

        max_latency_ms=
            None
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

    question=""

):

    # --------------------------------------------------------
    # Caption
    # --------------------------------------------------------

    if task == "caption":

        prompt = (
            "Describe this image "
            "in one sentence."
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
            "Describe this image "
            "in one sentence."
        )


    # --------------------------------------------------------
    # Multimodal conversation
    # --------------------------------------------------------

    messages = [

        {

            "role":
                "user",

            "content": [

                {
                    "type":
                        "image"
                },

                {

                    "type":
                        "text",

                    "text":
                        prompt
                }

            ]
        }

    ]


    # --------------------------------------------------------
    # Create prompt
    # --------------------------------------------------------

    text = (
        processor.apply_chat_template(

            messages,

            add_generation_prompt=True
        )
    )


    # --------------------------------------------------------
    # Processor
    # --------------------------------------------------------

    inputs = processor(

        text=text,

        images=[image],

        return_tensors="pt"
    )


    # --------------------------------------------------------
    # Move tensors
    # --------------------------------------------------------

    inputs = {

        key: (

            value.to(DEVICE)

            if hasattr(
                value,
                "to"
            )

            else value

        )

        for key, value
        in inputs.items()
    }


    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    with torch.no_grad():

        generated_ids = (
            model.generate(

                **inputs,

                max_new_tokens=30,

                do_sample=False,

                use_cache=True,

                eos_token_id=(
                    processor
                    .tokenizer
                    .eos_token_id
                ),

                pad_token_id=(
                    processor
                    .tokenizer
                    .pad_token_id
                )
            )
        )


    # --------------------------------------------------------
    # Remove prompt
    # --------------------------------------------------------

    input_length = (
        inputs[
            "input_ids"
        ].shape[1]
    )


    generated_tokens = (

        generated_ids[
            :,
            input_length:
        ]

    )


    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    answer = (
        processor.batch_decode(

            generated_tokens,

            skip_special_tokens=True
        )[0]
    )


    return answer.strip()


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
                    priority
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


        # ----------------------------------------------------
        # LOAD MODEL
        # ----------------------------------------------------

        model = load_model(
            selected_key
        )


        processor = (
            load_processor()
        )


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
            )
    })


# ============================================================
# CLEAR CACHE
# ============================================================

@app.route(
    "/api/clear-cache",
    methods=["POST"]
)
def clear_cache():

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


    app.run(

        host="127.0.0.1",

        port=5000,

        debug=False,

        threaded=True
    )