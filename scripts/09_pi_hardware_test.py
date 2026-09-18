"""
Raspberry Pi hardware validation test for the Efficient Multimodal AI project.

This script is designed to run ON THE RASPBERRY PI.

It measures:
    - System RAM before model loading
    - Model loading time
    - RAM consumption after loading
    - Inference latency
    - Model output
    - System RAM after inference

Supported models:
    smolvlm
    smolvlm2

Usage:

    python3 scripts/09_pi_hardware_test.py --model smolvlm

    python3 scripts/09_pi_hardware_test.py --model smolvlm2

    python3 scripts/09_pi_hardware_test.py \
        --model smolvlm2 \
        --image test.jpg
"""

import sys
import time
import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM
)


# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))

from utils.resource_monitor import get_system_stats


# ---------------------------------------------------------
# Model configuration
# ---------------------------------------------------------

MODELS = {

    "smolvlm": {
        "name": "HuggingFaceTB/SmolVLM-256M-Instruct",
        "display_name": "SmolVLM-256M"
    },

    "smolvlm2": {
        "name": "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        "display_name": "SmolVLM2-256M"
    }
}


PROMPT_TEXT = (
    "Describe this image in one sentence."
)


# ---------------------------------------------------------
# Arguments
# ---------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Raspberry Pi hardware validation test"
)

parser.add_argument(
    "--model",
    choices=[
        "smolvlm",
        "smolvlm2"
    ],
    default="smolvlm2",
    help="Model to test"
)

parser.add_argument(
    "--image",
    default=None,
    help="Path to a real test image"
)

args = parser.parse_args()


# ---------------------------------------------------------
# Model information
# ---------------------------------------------------------

model_config = MODELS[args.model]

MODEL_NAME = model_config["name"]

DISPLAY_NAME = model_config["display_name"]


print("=" * 65)

print(
    "RASPBERRY PI HARDWARE VALIDATION TEST"
)

print("=" * 65)

print(
    f"\nModel: {DISPLAY_NAME}"
)

print(
    f"Hugging Face model: {MODEL_NAME}"
)


# ---------------------------------------------------------
# System stats before loading
# ---------------------------------------------------------

print(
    "\nSystem statistics BEFORE loading model:"
)

before = get_system_stats()

for key, value in before.items():

    print(
        f"  {key}: {value}"
    )


# ---------------------------------------------------------
# Load processor and model
# ---------------------------------------------------------

print(
    "\nLoading model..."
)

load_start = time.perf_counter()


try:

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        _attn_implementation="eager"
    )

    model.eval()

    load_time_s = (
        time.perf_counter() - load_start
    )

    print(
        f"\nModel loaded successfully."
    )

    print(
        f"Load time: {load_time_s:.2f} seconds"
    )


except (
    RuntimeError,
    MemoryError,
    OSError
) as e:

    print(
        "\n*** MODEL LOAD FAILED ***"
    )

    print(
        f"Error: {e}"
    )

    print(
        "\nThis is a valid hardware result:"
    )

    print(
        f"{DISPLAY_NAME} could not be loaded "
        "within the available RAM."
    )

    sys.exit(1)


# ---------------------------------------------------------
# System stats after loading
# ---------------------------------------------------------

print(
    "\nSystem statistics AFTER loading model:"
)

after_load = get_system_stats()

for key, value in after_load.items():

    print(
        f"  {key}: {value}"
    )


ram_consumed = (
    before["ram_available_mb"]
    -
    after_load["ram_available_mb"]
)


print(
    "\nEstimated RAM consumed by model:"
)

print(
    f"  {ram_consumed:.2f} MB"
)


# ---------------------------------------------------------
# Load image
# ---------------------------------------------------------

if args.image:

    image_path = Path(args.image)

    if not image_path.exists():

        print(
            f"\nImage not found: {image_path}"
        )

        sys.exit(1)

    image = Image.open(
        image_path
    ).convert("RGB")

    print(
        f"\nUsing image: {image_path}"
    )

else:

    image = Image.new(
        "RGB",
        (384, 384),
        color=(120, 130, 140)
    )

    print(
        "\nNo image supplied."
    )

    print(
        "Using generated 384x384 test image."
    )


# ---------------------------------------------------------
# Prepare multimodal input
# ---------------------------------------------------------

messages = [

    {
        "role": "user",

        "content": [

            {
                "type": "image"
            },

            {
                "type": "text",
                "text": PROMPT_TEXT
            }

        ]
    }

]


prompt = processor.apply_chat_template(
    messages,
    add_generation_prompt=True
)


inputs = processor(
    text=prompt,
    images=[image],
    return_tensors="pt"
)


# ---------------------------------------------------------
# Run inference
# ---------------------------------------------------------

print(
    "\nRunning inference..."
)

infer_start = time.perf_counter()


try:

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=30
        )

    infer_time_s = (
        time.perf_counter()
        -
        infer_start
    )

except (
    RuntimeError,
    MemoryError
) as e:

    print(
        "\n*** INFERENCE FAILED ***"
    )

    print(
        f"Error: {e}"
    )

    print(
        "\nThe model loaded successfully "
        "but could not complete inference."
    )

    sys.exit(1)


# ---------------------------------------------------------
# Decode output
# ---------------------------------------------------------

generated_text = processor.batch_decode(
    output,
    skip_special_tokens=True
)[0]


caption = (
    generated_text
    .split("Assistant:")[-1]
    .strip()
)


# ---------------------------------------------------------
# System stats after inference
# ---------------------------------------------------------

print(
    f"\nInference completed in "
    f"{infer_time_s:.2f} seconds"
)

print(
    f"Output: {caption}"
)


print(
    "\nSystem statistics AFTER inference:"
)

after_inference = get_system_stats()

for key, value in after_inference.items():

    print(
        f"  {key}: {value}"
    )


# ---------------------------------------------------------
# Final result
# ---------------------------------------------------------

print(
    "\n"
    + "=" * 65
)

print(
    "RESULT: SUCCESS"
)

print(
    "=" * 65
)

print(
    f"Model:             {DISPLAY_NAME}"
)

print(
    f"Load time:         {load_time_s:.2f} s"
)

print(
    f"Inference time:    {infer_time_s:.2f} s"
)

print(
    f"RAM consumed:      ~{ram_consumed:.2f} MB"
)

print(
    f"Device RAM:        "
    f"{before['ram_total_mb']:.0f} MB"
)

print(
    f"Available RAM:     "
    f"{before['ram_available_mb']:.0f} MB"
)

print(
    "=" * 65
)