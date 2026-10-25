# CAMEO: Compressed AI for Multimodal Edge Optimization

CAMEO is an image-captioning project that evaluates practical model
compression techniques for resource-constrained edge devices. It uses the
SmolVLM-256M-Instruct multimodal model and compares a baseline with dynamic
INT8 quantization, pruning, and a combined optimized variant.

The evaluation tracks model size, CPU latency, peak memory, and captioning
quality. A Flask web app provides an interactive image-captioning demo.

## Repository name

**CAMEO: Compressed AI for Multimodal Edge
Optimization**.

## Setup on Windows

Run these commands from the project directory in PowerShell:

```powershell
py -3 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, either run the command in a Command Prompt
with `.venv\\Scripts\\activate.bat`, or allow local scripts for your user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

The first model run downloads model files from Hugging Face. Dataset images
and evaluation metadata are local inputs and should not be committed.

## Run the pipeline

Run from the project root, with the virtual environment activated:

```powershell
python scripts/01_baseline.py
python scripts/02_quantize_dynamic.py
python scripts/04_prune.py
python scripts/06_benchmark.py
python scripts/07_resource_selector.py
python scripts/08_combine_optimized.py
python scripts/10_check_saved_models.py
```

The Raspberry Pi dependency set is intentionally smaller:

```powershell
python -m pip install -r requirements-pi.txt
```

For a Pi hardware check, run `scripts/09_pi_hardware_test.py` after copying
the required source and model artifacts to the device.

## Web app demo

Start the Flask app from the project root:

```powershell
python webapp/app.py
```

Open <http://localhost:5000>. On startup the app preloads the processor and
all four local variants: INT8, pruned converted to INT8, INT8+pruned, and
FP32 baseline. Compressed checkpoints are warmed first. If
`quantized_model.pt` is missing, the app builds dynamic INT8 from the
Hugging Face weights at startup. Later image and video requests reuse that
cache instead of calling `from_pretrained` / `torch.load` again. Adaptive
selection also prefers an already-loaded variant when it still fits the
resource budget, so swapping models does not reload weights on every frame.
Set `CAMEO_PRELOAD=0` to skip startup preload, or
`CAMEO_PRELOAD_MODEL=quantized,pruned` to warm only a subset. The app
supports baseline,
quantized, pruned, and combined optimized variants when their local artifacts
are available. In the AI Workspace, uploaded videos are decoded in the
browser and sampled at 2 frames per second; a lightweight image-difference
gate detects scene changes. Uploaded videos make one VLM call for a single
concise caption of the sampled frame with the strongest detected scene change;
subsequent sampled frames are event-checked without repeated VLM calls. Live
camera inference uses the same event gate and can continue producing captions.
Both modes can use adaptive model selection and an optional measured-latency
target. Live inference keeps a bounded buffer of skipped frames and replays a
high-value candidate when a later event triggers inference. Captions link to
their timestamps and keyframe thumbnails.

The event gate also estimates normalized visual-change value and treats
near-threshold frames as uncertain, requesting a more detailed caption.
Caption-word overlap feedback adjusts the event threshold within bounded
limits during a live session. An optional CPU-package energy budget in J/min
adjusts the event threshold when a reliable Intel RAPL reading is available;
important and uncertain events bypass energy-based deferral. Without a
supported energy meter the controller remains inactive and says so. Energy
debt sums observed VLM-generation package energy over a rolling 60-second
window; it excludes idle and browser-decoding energy. Auto model selection
applies a configurable BLEU-4 retention floor against the FP32 benchmark for
that model family. Uncertain, query-relevant, and audio-confirmed events
request a stricter accuracy-first choice within that quality floor.

Video queries use temporal/action terms and simple requested-color cues to
adjust frame relevance. This is a lightweight heuristic, not a text-image
embedding model. Uploaded-video audio is sampled in the browser only for files
up to 24 MB and 120 seconds; live inference optionally requests microphone
access. Audio RMS corroborates activity around a visual change but does not
recognize speech or audio-event semantics. If audio is unavailable, video
processing continues as visual-only and reports that source.

The completed video session reports sampled frames, VLM call reduction,
inference and frame-processing latency, CPU/RAM use, and a conventional
frame-by-frame comparison at the same 2 fps sampling cadence. CPU-package energy during VLM generation is reported only when a readable
Intel RAPL counter is available; frame decoding and event-gate energy are not
included. Otherwise the UI explicitly marks energy as unavailable. The
conventional latency/energy values are estimates based on the session's
measured per-inference average, not an additional frame-by-frame benchmark run.

## Annotated video benchmark

After generating model profiles (including BLEU-4), prepare a local JSON
manifest with sampled frame paths, event timestamps, and reference captions.
The benchmark runs the adaptive pipeline and a frame-by-frame VLM comparator
over the same samples:

```powershell
python scripts/video_pipeline_benchmark.py .\\data\\video_eval_manifest.json `
  --quality-retention 0.90 `
  --output .\\results\\video_pipeline_benchmark.json
```

Example manifest (frame paths are relative to the manifest):

```json
{
  "event_timestamp_tolerance_seconds": 0.5,
  "videos": [{
    "id": "clip-01",
    "frames": [
      {"image": "frames/clip-01-000.jpg", "timestamp_seconds": 0.0, "audio_rms": 0.01},
      {"image": "frames/clip-01-001.jpg", "timestamp_seconds": 0.5, "audio_rms": 0.08}
    ],
    "event_timestamps_seconds": [0.5],
    "reference_captions": ["A person opens a door."]
  }]
}
```

The report includes best-reference unigram token F1, annotated missed-event
rate (an event is detected when an inference is within the configured
timestamp tolerance), mean/p95 inference latency, CPU/RAM, inference frequency,
and measured CPU-package energy when RAPL is available. Token F1 is a lexical
overlap proxy, not a semantic or human quality judgment. Keep annotated media
and reference captions local. The frame-by-frame comparison can be very slow
because it deliberately invokes the VLM for every supplied frame.

## Troubleshooting adaptive selection and camera

The default 90% BLEU-4 retention control is a hard quality filter: it never
silently switches to a model below that benchmark. The selector also reserves
25% of currently available RAM for the operating system and application.
With the checked-in benchmark values, baseline and INT8 are the only models
currently above the 90% family-baseline BLEU-4 floor; their measured runtime
footprints are about 1,277 MB and 1,846 MB. Therefore, the safe RAM budget
must meet the footprint of at least one eligible model. Lowering the retention
control may admit lower-BLEU variants, but it does not make a model fit in less
RAM. Close memory-heavy applications, check that model artifacts and benchmark
profiles exist, or lower the floor only if the resulting quality tradeoff is
acceptable. The error now reports the current safe budget and eligible
profiles.

For live camera inference, allow camera access and wait until the preview shows
“Camera ready” before starting. Camera access requires `localhost` or HTTPS.
Microphone permission is optional; if the browser asks for both camera and
microphone at once, blocking the microphone still continues with visual-only
inference. CPU-only VLM inference can take several seconds per caption. The
camera keeps sampling during that wait and queues later frames so a scene
change is not dropped. The modal status reports skipped frames as well as
VLM events.

## Metrics and findings

The project records:

- model size and compression ratio
- CPU inference latency
- peak memory use
- captioning quality metrics such as BLEU and CIDEr

Unstructured pruning can increase sparsity without reducing the dense file
size. Also, PyTorch INT8 or sparse inference may be slower on some CPUs when
optimized kernels are unavailable. These are expected results to discuss;
ONNX Runtime or hardware-specific runtimes may provide better edge latency.

## Project structure

```text
.
├── data/                    # local datasets; ignored by Git
├── report/                  # setup, novelty, and progress documentation
├── results/                 # JSON/CSV summaries; model weights are ignored
├── scripts/                 # baseline, compression, benchmarking, and Pi tools
├── utils/                   # data, metrics, resource, and model-selection helpers
├── webapp/                 # Flask demo and HTML template
├── .gitignore
├── requirements.txt
└── requirements-pi.txt
```
