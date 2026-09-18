"""
Week 1: Baseline — HuggingFaceTB/SmolVLM-256M-Instruct.
SmolVLM is purpose-built for edge/resource-constrained deployment, making it
a much better fit for this project than BLIP.

Run: python scripts/01_baseline.py
"""

import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForMultimodalLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.metrics import (
    model_size_mb, count_parameters, measure_latency, measure_peak_memory,
    get_process_memory_mb, save_result,
)
from utils.data import get_caption_subset

MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"
DEVICE = "cpu"
N_SAMPLES = 200
PROMPT_TEXT = "Describe this image in one sentence."


def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, _attn_implementation="eager"
    )
    model.to(DEVICE)
    model.eval()
    return processor, model


def generate_caption(processor, model, image: Image.Image, question: str = None) -> str:
    text = question if question else PROMPT_TEXT
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=30)
    generated_text = processor.batch_decode(out, skip_special_tokens=True)[0]
    return generated_text.split("Assistant:")[-1].strip()


def evaluate_bleu(processor, model, dataset) -> float:

    smoothie = SmoothingFunction().method4
    scores = []

    for i, sample in enumerate(dataset):

        if i >= N_SAMPLES:
            break

        print(f"Evaluating image {i + 1}/{N_SAMPLES}...", flush=True)

        image = sample["image"].convert("RGB")

        references = sample["caption"]

        pred = generate_caption(
            processor,
            model,
            image
        )

        reference_tokens = [
            ref.lower().split()
            for ref in references
        ]

        score = sentence_bleu(
            reference_tokens,
            pred.lower().split(),
            smoothing_function=smoothie
        )

        scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0


def main():
    print(f"Loading {MODEL_NAME} ...")
    processor, model = load_model()

    size_mb = model_size_mb(model)
    params = count_parameters(model)
    model_footprint_mb = get_process_memory_mb()
    print(f"Footprint after load: {model_footprint_mb} MB")

    dataset = get_caption_subset(n_samples=N_SAMPLES)
    sample_image = dataset[0]["image"].convert("RGB")

    latency = measure_latency(lambda: generate_caption(processor, model, sample_image), n_warmup=3, n_runs=15)
    peak_mem = measure_peak_memory(lambda: generate_caption(processor, model, sample_image))

    print("Evaluating BLEU-4 ...")
    bleu = evaluate_bleu(processor, model, dataset)

    result = {
        "model": "baseline_smolvlm256m_fp32", "size_mb": size_mb, **params,
        "model_footprint_mb": model_footprint_mb, **latency, "peak_memory_mb": peak_mem,
        "bleu4": round(bleu, 4), "n_eval_samples": N_SAMPLES, "device": DEVICE,
    }
    print(result)
    save_result("baseline", result)

    out_path = Path(__file__).resolve().parent.parent / "results" / "baseline_model.pt"
    torch.save(model.state_dict(), out_path)
    print(f"Baseline weights saved to {out_path}")


if __name__ == "__main__":
    main()
