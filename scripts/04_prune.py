"""
Week 3: Unstructured L1 Pruning + Fine-tuning Recovery
Model: HuggingFaceTB/SmolVLM-256M-Instruct

Run:
    python scripts/04_prune.py

Prerequisites:
    - 01_baseline.py completed
    - utils/data.py returns a Python list of samples
    - Evaluation set contains:
        image
        caption -> list of reference captions
"""

import sys
import gc
from pathlib import Path

import torch
import torch.nn.utils.prune as prune

from transformers import AutoProcessor, AutoModelForMultimodalLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# -------------------------------------------------------------------
# Project imports
# -------------------------------------------------------------------

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.metrics import (
    model_size_mb,
    measure_latency,
    measure_peak_memory,
    get_process_memory_mb,
    save_result,
)

from utils.data import get_caption_subset


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"

DEVICE = "cpu"

# Keep this SAME as baseline and quantization
N_SAMPLES = 200

PROMPT_TEXT = "Describe this image in one sentence."

# Percentage of Linear weights to prune
PRUNE_RATIO = 0.30

# Brief fine-tuning after pruning
FINETUNE_EPOCHS = 1
FINETUNE_LR = 1e-5

# Number of samples used for recovery fine-tuning
FINETUNE_SAMPLES = 50

RESULTS_DIR = (
    Path(__file__).resolve().parent.parent / "results"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# Caption generation
# -------------------------------------------------------------------

def generate_caption(
    processor,
    model,
    image,
    question: str = None
):
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
        output_ids = model.generate(
            **inputs,
            max_new_tokens=30
        )

    generated_text = processor.batch_decode(
        output_ids,
        skip_special_tokens=True
    )[0]

    # Handle different possible chat-template outputs
    if "Assistant:" in generated_text:
        generated_text = generated_text.split(
            "Assistant:"
        )[-1]

    return generated_text.strip()


# -------------------------------------------------------------------
# Reference caption handling
# -------------------------------------------------------------------

def get_reference_caption(sample):
    """
    Flickr30k provides 5 captions per image.

    Returns one caption for fine-tuning / compatibility.
    """

    caption = sample.get("caption", "")

    # Flickr30k format:
    # ["caption 1", "caption 2", ...]
    if isinstance(caption, list):

        if len(caption) > 0:
            return caption[0]

        return ""

    # Normal string caption
    if isinstance(caption, str):
        return caption

    return ""


def get_reference_captions(sample):
    """
    Returns ALL reference captions for BLEU evaluation.
    """

    caption = sample.get("caption", "")

    if isinstance(caption, list):
        return [
            str(c).lower().split()
            for c in caption
            if c
        ]

    if isinstance(caption, str):
        return [caption.lower().split()]

    return []


# -------------------------------------------------------------------
# L1 Unstructured Pruning
# -------------------------------------------------------------------

def apply_l1_unstructured_prune(model, ratio):
    """
    Apply L1 unstructured pruning to all Linear layers.

    Example:
        ratio = 0.30
        -> approximately 30% of weights in each Linear layer
           are masked/pruned.
    """

    pruned_layers = 0

    for name, module in model.named_modules():

        if isinstance(module, torch.nn.Linear):

            prune.l1_unstructured(
                module,
                name="weight",
                amount=ratio
            )

            pruned_layers += 1

    print(
        f"Pruning applied to {pruned_layers} Linear layers."
    )

    return model


# -------------------------------------------------------------------
# Remove pruning reparameterization
# -------------------------------------------------------------------

def finalize_pruning(model):
    """
    Permanently remove pruning masks.

    After this:
        weight_orig + weight_mask
    becomes:
        weight
    """

    finalized_layers = 0

    for name, module in model.named_modules():

        if (
            isinstance(module, torch.nn.Linear)
            and hasattr(module, "weight_mask")
        ):

            prune.remove(
                module,
                "weight"
            )

            finalized_layers += 1

    print(
        f"Finalized pruning for {finalized_layers} Linear layers."
    )

    return model


# -------------------------------------------------------------------
# Calculate sparsity
# -------------------------------------------------------------------

def compute_sparsity(model):
    """
    Calculate percentage of zero-valued parameters.
    """

    total = 0
    zeros = 0

    for _, param in model.named_parameters():

        total += param.numel()

        zeros += (
            param == 0
        ).sum().item()

    if total == 0:
        return 0.0

    return round(
        100.0 * zeros / total,
        2
    )


# -------------------------------------------------------------------
# Brief Fine-tuning
# -------------------------------------------------------------------

def finetune_briefly(
    processor,
    model,
    dataset,
    epochs=1,
    lr=1e-5
):
    """
    Brief fine-tuning after pruning to recover model quality.

    IMPORTANT:
    This is only a lightweight recovery phase.
    It is not intended to fully retrain the model.
    """

    print()
    print("=" * 60)
    print("Starting brief fine-tuning...")
    print("=" * 60)

    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr
    )

    for epoch in range(epochs):

        last_loss = None

        for i, sample in enumerate(dataset):

            image = sample["image"].convert("RGB")

            caption = get_reference_caption(
                sample
            )

            if not caption:
                continue

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": PROMPT_TEXT
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": caption
                        }
                    ],
                },
            ]

            prompt = processor.apply_chat_template(
                messages,
                add_generation_prompt=False
            )

            inputs = processor(
                text=prompt,
                images=[image],
                return_tensors="pt",
                padding=True
            ).to(DEVICE)

            optimizer.zero_grad()

            outputs = model(
                **inputs,
                labels=inputs["input_ids"]
            )

            loss = outputs.loss

            loss.backward()

            optimizer.step()

            last_loss = loss.item()

            print(
                f"  Epoch {epoch + 1}/{epochs} | "
                f"Sample {i + 1}/{len(dataset)} | "
                f"Loss: {last_loss:.4f}"
            )

        print(
            f"\nEpoch {epoch + 1}/{epochs} completed."
        )

        if last_loss is not None:
            print(
                f"Last loss = {last_loss:.4f}"
            )

    model.eval()

    del optimizer

    gc.collect()

    print("Fine-tuning completed.")

    return model


# -------------------------------------------------------------------
# BLEU Evaluation
# -------------------------------------------------------------------

def evaluate_bleu(
    processor,
    model,
    dataset
):
    """
    Evaluate generated captions using BLEU-4.

    Flickr30k has multiple reference captions,
    so all available captions are used as references.
    """

    smoothie = SmoothingFunction().method4

    scores = []

    total = len(dataset)

    print()
    print("=" * 60)
    print("Evaluating BLEU-4...")
    print("=" * 60)

    for i, sample in enumerate(dataset):

        print(
            f"Evaluating image {i + 1}/{total}..."
        )

        image = sample["image"].convert("RGB")

        references = get_reference_captions(
            sample
        )

        if not references:
            continue

        prediction = generate_caption(
            processor,
            model,
            image
        )

        prediction_tokens = (
            prediction.lower().split()
        )

        score = sentence_bleu(
            references,
            prediction_tokens,
            smoothing_function=smoothie
        )

        scores.append(score)

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():

    print("=" * 60)
    print("SmolVLM-256M L1 PRUNING EXPERIMENT")
    print("=" * 60)

    print(
        f"\nLoading {MODEL_NAME} ..."
    )

    # ---------------------------------------------------------------
    # Load processor
    # ---------------------------------------------------------------

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    # ---------------------------------------------------------------
    # Load baseline model
    # ---------------------------------------------------------------

    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        _attn_implementation="eager"
    )

    model.to(DEVICE)
    model.eval()

    print("Baseline model loaded.")

    # ---------------------------------------------------------------
    # Baseline size before pruning
    # ---------------------------------------------------------------

    baseline_size = model_size_mb(model)

    print(
        f"Baseline model size: "
        f"{baseline_size:.3f} MB"
    )

    # ---------------------------------------------------------------
    # Apply pruning
    # ---------------------------------------------------------------

    print()
    print(
        f"Applying L1 unstructured pruning "
        f"({PRUNE_RATIO * 100:.0f}% of Linear weights)..."
    )

    model = apply_l1_unstructured_prune(
        model,
        PRUNE_RATIO
    )

    # ---------------------------------------------------------------
    # Load evaluation dataset
    # ---------------------------------------------------------------

    dataset = get_caption_subset(
        n_samples=N_SAMPLES
    )

    print(
        f"\nLoaded {len(dataset)} evaluation samples."
    )

    # ---------------------------------------------------------------
    # Fine-tuning subset
    #
    # IMPORTANT:
    # dataset is a Python LIST, so:
    #
    # dataset[:50]
    #
    # NOT:
    #
    # dataset.select(...)
    # ---------------------------------------------------------------

    finetune_subset = dataset[
        :min(
            FINETUNE_SAMPLES,
            len(dataset)
        )
    ]

    print(
        f"Using {len(finetune_subset)} "
        f"samples for fine-tuning recovery."
    )

    # ---------------------------------------------------------------
    # Fine-tune
    # ---------------------------------------------------------------

    model = finetune_briefly(
        processor,
        model,
        finetune_subset,
        epochs=FINETUNE_EPOCHS,
        lr=FINETUNE_LR
    )

    # ---------------------------------------------------------------
    # Permanently finalize pruning
    # ---------------------------------------------------------------

    print()
    print(
        "Finalizing pruning masks..."
    )

    model = finalize_pruning(
        model
    )

    gc.collect()

    # ---------------------------------------------------------------
    # Calculate sparsity
    # ---------------------------------------------------------------

    sparsity_pct = compute_sparsity(
        model
    )

    print(
        f"\nFinal sparsity: "
        f"{sparsity_pct}%"
    )

    # ---------------------------------------------------------------
    # Save pruned model
    # ---------------------------------------------------------------

    model_path = (
        RESULTS_DIR /
        "pruned_model.pt"
    )

    torch.save(
        model,
        model_path
    )

    print(
        f"Pruned model saved to:\n"
        f"{model_path}"
    )

    # ---------------------------------------------------------------
    # Model size
    # ---------------------------------------------------------------

    size_mb = model_size_mb(
        model
    )

    # ---------------------------------------------------------------
    # Process memory
    # ---------------------------------------------------------------

    model_footprint_mb = (
        get_process_memory_mb()
    )

    print(
        f"\nModel footprint: "
        f"{model_footprint_mb:.3f} MB"
    )

    # ---------------------------------------------------------------
    # Latency benchmark
    # ---------------------------------------------------------------

    print()
    print("=" * 60)
    print("Benchmarking latency...")
    print("=" * 60)

    sample_image = (
        dataset[0]["image"]
        .convert("RGB")
    )

    latency = measure_latency(
        lambda: generate_caption(
            processor,
            model,
            sample_image
        ),
        n_warmup=3,
        n_runs=15
    )

    # ---------------------------------------------------------------
    # Peak memory
    # ---------------------------------------------------------------

    print(
        "\nMeasuring peak memory..."
    )

    peak_mem = measure_peak_memory(
        lambda: generate_caption(
            processor,
            model,
            sample_image
        )
    )

    # ---------------------------------------------------------------
    # BLEU evaluation
    # ---------------------------------------------------------------

    bleu = evaluate_bleu(
        processor,
        model,
        dataset
    )

    # ---------------------------------------------------------------
    # Final results
    # ---------------------------------------------------------------

    result = {
        "model":
            f"pruned_l1_"
            f"{int(PRUNE_RATIO * 100)}pct_"
            f"finetuned_smolvlm256m",

        "size_mb":
            round(size_mb, 3),

        "baseline_size_mb":
            round(baseline_size, 3),

        "model_footprint_mb":
            round(model_footprint_mb, 3),

        "sparsity_pct":
            sparsity_pct,

        **latency,

        "peak_memory_mb":
            round(peak_mem, 3),

        "bleu4":
            round(bleu, 4),

        "n_eval_samples":
            N_SAMPLES,

        "finetune_samples":
            len(finetune_subset),

        "device":
            DEVICE,
    }

    # ---------------------------------------------------------------
    # Print results
    # ---------------------------------------------------------------

    print()
    print("=" * 60)
    print("L1 PRUNING RESULTS")
    print("=" * 60)

    for key, value in result.items():
        print(
            f"{key}: {value}"
        )

    print("=" * 60)

    # ---------------------------------------------------------------
    # Save result JSON
    # ---------------------------------------------------------------

    save_result(
        "pruned",
        result
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        RESULTS_DIR /
        "pruned.json"
    )


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

if __name__ == "__main__":
    main()