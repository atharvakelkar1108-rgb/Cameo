"""
Shared measurement utilities used by every script in this project.
Keeping these in one place means every model (baseline, quantized, pruned,
ONNX) is measured the exact same way, so comparisons in the final report
are apples-to-apples.
"""

import os
import gc
import time
import json
from pathlib import Path

import torch
import psutil


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def model_size_mb(model, path="_tmp_model_size.pt"):
    """Save state_dict to disk and measure actual file size (most honest way
    to measure size, since in-memory dtype tricks can mislead)."""
    torch.save(model.state_dict(), path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    os.remove(path)
    return round(size_mb, 3)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


def measure_latency(fn, n_warmup=5, n_runs=30, *args, **kwargs):
    """
    fn: a zero/low-arg callable that runs one forward pass / one inference.
    Returns mean and p95 latency in milliseconds.
    """
    for _ in range(n_warmup):
        fn(*args, **kwargs)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        fn(*args, **kwargs)
        times.append((time.perf_counter() - start) * 1000)  # ms

    times.sort()
    mean_ms = sum(times) / len(times)
    p95_ms = times[int(0.95 * len(times)) - 1]
    return {"mean_latency_ms": round(mean_ms, 3), "p95_latency_ms": round(p95_ms, 3)}


def get_process_memory_mb():
    """
    Total resident memory (RSS) of the current process, right now, in MB.

    Call this AFTER the model is fully loaded (and after quantization/
    pruning is applied) to get the model's real memory footprint — this is
    the number that answers "will this fit in the device's RAM," which is
    what matters for a resource-constrained deployment claim.

    This is different from measure_peak_memory() below, which only
    captures the marginal memory used DURING one inference call — since
    the model is already loaded by that point, that delta is small and
    not what you want for a "does it fit" argument. Report get_process_memory_mb()
    as your headline memory number; keep measure_peak_memory() as a
    secondary "extra working memory per inference" number if you want it.
    """
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / (1024 * 1024), 3)


def measure_peak_memory(fn, *args, **kwargs):
    """
    Peak RSS (resident set size) memory during one call to fn, in MB.

    IMPORTANT: tracemalloc (used in an earlier version of this function)
    only tracks Python-level allocations — it does NOT see PyTorch's C++/
    Aten tensor memory, so it reported ~1-2MB for every model regardless
    of actual size. psutil's RSS reflects real OS-level process memory,
    which is what you want for a "does this fit on a resource-constrained
    device" claim.

    This is still an approximation: it measures the *process's* memory
    delta around one inference call, not a true peak sampled continuously
    during the call. For a more precise peak, sample memory_info().rss in
    a background thread every few ms while fn() runs — worth doing if you
    have time, but this single before/after delta is a reasonable and
    commonly-used simplification for a course project.
    """
    process = psutil.Process(os.getpid())
    gc.collect()
    mem_before = process.memory_info().rss

    fn(*args, **kwargs)

    mem_after = process.memory_info().rss
    delta_mb = (mem_after - mem_before) / (1024 * 1024)
    # Memory can dip below the "before" baseline if gc runs mid-call;
    # floor at 0 rather than reporting a nonsensical negative number.
    return round(max(delta_mb, 0.0), 3)


def save_result(name: str, result: dict):
    """Append/overwrite one model's results under results/<name>.json"""
    out_path = RESULTS_DIR / f"{name}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[saved] {out_path}")


def load_all_results():
    """Collect every results/*.json into one dict for the final comparison."""
    all_results = {}
    for f in RESULTS_DIR.glob("*.json"):
        with open(f) as fh:
            all_results[f.stem] = json.load(fh)
    return all_results
