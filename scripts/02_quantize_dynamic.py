"""
Week 2: Dynamic INT8 quantization of SmolVLM-256M-Instruct.

Run:
    python scripts/02_quantize_dynamic.py

Run 01_baseline.py first.
"""

import sys
import gc
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.metrics import (
    model_size_mb,
    measure_latency,
    measure_peak_memory,
    get_process_memory_mb,
    save_result,
)
from utils.data import get_caption_subset

print("PyTorch threads:", torch.get_num_threads())
print("PyTorch interop threads:", torch.get_num_interop_threads())

MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"
DEVICE = "cpu"

# Keep 200 for the final experiment
N_SAMPLES = 200

PROMPT_TEXT = "Describe this image in one sentence."

RESULTS_DIR = (
    Path(__file__).resolve().parent.parent / "results"
)


def generate_caption(processor, model, image, question: str = None):
    """
    Generate one caption for an image.
    """

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
        out = model.generate(
            **inputs,
            max_new_tokens=30
        )

    generated_text = processor.batch_decode(
        out,
        skip_special_tokens=True
    )[0]

    return generated_text.split("Assistant:")[-1].strip()


def evaluate_bleu(processor, model, dataset):
    """
    Evaluate BLEU-4 using all reference captions available
    for each Flickr30k image.

    Flickr30k provides multiple reference captions per image,
    so we compare the generated caption against all references.
    """

    smoothie = SmoothingFunction().method4

    scores = []

    print("\nEvaluating BLEU-4 ...")

    for i, sample in enumerate(dataset):

        print(
            f"Evaluating image {i + 1}/{N_SAMPLES}...",
            flush=True
        )

        image = sample["image"].convert("RGB")

        # Flickr30k contains 5 reference captions per image
        references = sample["caption"]

        pred = generate_caption(
            processor,
            model,
            image
        )

        # Convert every reference caption into tokens
        reference_tokens = [
            ref.lower().split()
            for ref in references
        ]

        # Generated caption tokens
        prediction_tokens = pred.lower().split()

        # BLEU using multiple reference captions
        score = sentence_bleu(
            reference_tokens,
            prediction_tokens,
            smoothing_function=smoothie
        )

        scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0


def main():

    print(f"Loading {MODEL_NAME} ...")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        _attn_implementation="eager"
    )

    model.eval()

    # --------------------------------------------------
    # Dynamic INT8 Quantization
    # --------------------------------------------------

    print("\nApplying dynamic quantization (INT8) ...")

    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    del model
    gc.collect()

    # --------------------------------------------------
    # Save Quantized Model
    # --------------------------------------------------

    output_path = RESULTS_DIR / "quantized_model.pt"

    torch.save(
        quantized_model,
        output_path
    )

    print(
        f"Quantized model saved to: {output_path}"
    )

    # --------------------------------------------------
    # Model Size / Memory
    # --------------------------------------------------

    size_mb = model_size_mb(
        quantized_model
    )

    model_footprint_mb = get_process_memory_mb()

    print(
        f"Footprint after quantizing: "
        f"{model_footprint_mb} MB"
    )

    # --------------------------------------------------
    # Load Evaluation Dataset
    # --------------------------------------------------

    dataset = get_caption_subset(
        n_samples=N_SAMPLES
    )

    print(
        f"Loaded {len(dataset)} evaluation samples."
    )

    # --------------------------------------------------
    # Latency Benchmark
    # --------------------------------------------------

    print("\nBenchmarking latency ...")

    sample_image = (
        dataset[0]["image"].convert("RGB")
    )

    latency = measure_latency(
        lambda: generate_caption(
            processor,
            quantized_model,
            sample_image
        ),
        n_warmup=3,
        n_runs=15
    )

    # --------------------------------------------------
    # Peak Memory
    # --------------------------------------------------

    print("Measuring peak memory ...")

    peak_mem = measure_peak_memory(
        lambda: generate_caption(
            processor,
            quantized_model,
            sample_image
        )
    )

    # --------------------------------------------------
    # BLEU-4 Evaluation
    # --------------------------------------------------

    bleu = evaluate_bleu(
        processor,
        quantized_model,
        dataset
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    result = {
        "model": "quantized_dynamic_int8_smolvlm256m",
        "size_mb": size_mb,
        "model_footprint_mb": model_footprint_mb,
        **latency,
        "peak_memory_mb": peak_mem,
        "bleu4": round(bleu, 4),
        "n_eval_samples": N_SAMPLES,
        "device": DEVICE,
    }

    print("\n" + "=" * 60)
    print("DYNAMIC INT8 QUANTIZATION RESULTS")
    print("=" * 60)

    for key, value in result.items():
        print(f"{key}: {value}")

    print("=" * 60)

    save_result(
        "quantized_dynamic",
        result
    )

    print(
        "\nResults saved to "
        f"{RESULTS_DIR / 'quantized_dynamic.json'}"
    )


if __name__ == "__main__":
    main()