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

Open <http://localhost:5000>. The first request can be slow because the
processor and model are loaded and cached. The app supports baseline,
quantized, pruned, and combined optimized variants when their local artifacts
are available.

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

## Suggested GitHub publishing sequence

Install Git for Windows first if `git --version` is not recognized. Then run:

```powershell
git --version
git init
git branch -M main
git add .gitignore README.md requirements.txt requirements-pi.txt scripts utils webapp report results
git status
git commit -m "Initial CAMEO project"
git remote add origin https://github.com/<YOUR_USERNAME>/cameo-compressed-ai-for-multimodal-edge-optimization.git
git push -u origin main
```

Before `git commit`, inspect `git status` and confirm that no dataset,
personal image, secret, virtual environment, or model weight is listed. If a
large file was staged accidentally, remove it from the index without deleting
your local copy:

```powershell
git restore --staged path\\to\\file
```

Create the empty GitHub repository with the slug above before running
`git remote add`. Do not initialize it with another README, license, or
`.gitignore`, because those files already exist locally.

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
