"""
Week 4: Final Optimized Model
Pruning + Fine-tuning Recovery + Dynamic INT8 Quantization

Model: HuggingFaceTB/SmolVLM-256M-Instruct

Pipeline:
    Baseline
       ↓
    L1 Unstructured Pruning (30%)
       ↓
    Fine-tuning Recovery
       ↓
    Dynamic INT8 Quantization
       ↓
    Optimized Model

Run:
    python scripts/08_combine_optimized.py

Prerequisite:
    results/pruned_model.pt must already exist.
"""

import sys
import gc
from pathlib import Path

import torch
from transformers import AutoProcessor

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.metrics import (
    model_size_mb,
    measure_latency,
    measure_peak_memory,
    get_process_memory_mb,
    save_result,
)

from utils.data import get_caption_subset


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"

DEVICE = "cpu"

N_SAMPLES = 200

PROMPT_TEXT = "Describe this image in one sentence."

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ---------------------------------------------------------
# Caption Generation
# ---------------------------------------------------------

def generate_caption(processor, model, image, question=None):

    text = question if question else PROMPT_TEXT

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
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
    ).to(DEVICE)

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=30
        )

    generated_text = processor.batch_decode(
        output,
        skip_special_tokens=True
    )[0]

    return generated_text.split("Assistant:")[-1].strip()


# ---------------------------------------------------------
# Get Reference Caption
# ---------------------------------------------------------

def get_reference_caption(sample):

    """
    The local Flickr30K evaluation dataset stores
    multiple reference captions as a list.

    We use the first reference caption for BLEU evaluation.
    """

    captions = sample.get("caption", "")

    if isinstance(captions, list):

        if len(captions) > 0:
            return captions[0]

        return ""

    return captions


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("FINAL OPTIMIZED MODEL")
    print("=" * 60)

    # -----------------------------------------------------
    # Load pruned model
    # -----------------------------------------------------

    print("\nLoading pruned model from results/pruned_model.pt ...")

    pruned_model = torch.load(
        RESULTS_DIR / "pruned_model.pt",
        weights_only=False
    )

    pruned_model.eval()

    print("Pruned model loaded successfully.")

    # -----------------------------------------------------
    # Load processor
    # -----------------------------------------------------

    print("\nLoading processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    # -----------------------------------------------------
    # Dynamic INT8 Quantization
    # -----------------------------------------------------

    print("\nApplying dynamic INT8 quantization...")

    optimized_model = torch.quantization.quantize_dynamic(
        pruned_model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    del pruned_model

    gc.collect()

    print("Dynamic INT8 quantization completed.")

    # -----------------------------------------------------
    # Save optimized model
    # -----------------------------------------------------

    optimized_path = RESULTS_DIR / "optimized_model.pt"

    torch.save(
        optimized_model,
        optimized_path
    )

    print(f"\nOptimized model saved to:")
    print(optimized_path)

    # -----------------------------------------------------
    # Model size and memory
    # -----------------------------------------------------

    size_mb = model_size_mb(
        optimized_model
    )

    model_footprint_mb = get_process_memory_mb()

    print("\nOptimized model statistics:")
    print(f"Model size: {size_mb} MB")
    print(f"Memory footprint: {model_footprint_mb} MB")

    # -----------------------------------------------------
    # Load evaluation dataset
    # -----------------------------------------------------

    print("\nLoading evaluation dataset...")

    dataset = get_caption_subset(
        n_samples=N_SAMPLES
    )

    print(
        f"Loaded {len(dataset)} evaluation samples."
    )

    # -----------------------------------------------------
    # Latency benchmark
    # -----------------------------------------------------

    print("\nBenchmarking latency...")

    sample_image = dataset[0]["image"].convert("RGB")

    latency = measure_latency(
        lambda: generate_caption(
            processor,
            optimized_model,
            sample_image
        ),
        n_warmup=3,
        n_runs=15
    )

    # -----------------------------------------------------
    # Peak memory
    # -----------------------------------------------------

    print("\nMeasuring peak memory...")

    peak_mem = measure_peak_memory(
        lambda: generate_caption(
            processor,
            optimized_model,
            sample_image
        )
    )

    # -----------------------------------------------------
    # BLEU-4 evaluation
    # -----------------------------------------------------

    print("\nEvaluating BLEU-4...")

    from nltk.translate.bleu_score import (
        sentence_bleu,
        SmoothingFunction
    )

    smoothie = SmoothingFunction().method4

    scores = []

    for i, sample in enumerate(dataset):

        print(
            f"Evaluating image {i + 1}/{len(dataset)}..."
        )

        image = sample["image"].convert("RGB")

        # FIX:
        # caption is a list in the Flickr30K dataset
        reference = get_reference_caption(sample)

        prediction = generate_caption(
            processor,
            optimized_model,
            image
        )

        reference_tokens = reference.lower().split()

        prediction_tokens = prediction.lower().split()

        score = sentence_bleu(
            [reference_tokens],
            prediction_tokens,
            smoothing_function=smoothie
        )

        scores.append(score)

    bleu = (
        sum(scores) / len(scores)
        if scores
        else 0.0
    )

    # -----------------------------------------------------
    # Final results
    # -----------------------------------------------------

    result = {

        "model":
            "optimized_pruned30pct_int8_smolvlm256m",

        "size_mb":
            size_mb,

        "model_footprint_mb":
            model_footprint_mb,

        **latency,

        "peak_memory_mb":
            peak_mem,

        "bleu4":
            round(bleu, 4),

        "n_eval_samples":
            N_SAMPLES,

        "device":
            DEVICE,
    }

    # -----------------------------------------------------
    # Display results
    # -----------------------------------------------------

    print("\n")
    print("=" * 60)
    print("OPTIMIZED MODEL RESULTS")
    print("=" * 60)

    for key, value in result.items():

        print(f"{key}: {value}")

    print("=" * 60)

    # -----------------------------------------------------
    # Save result
    # -----------------------------------------------------

    save_result(
        "optimized",
        result
    )

    print("\nResults saved to:")
    print(
        RESULTS_DIR / "optimized.json"
    )


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()