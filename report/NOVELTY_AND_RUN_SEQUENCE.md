# Novelty: Resource-Aware Adaptive Multimodal AI

The project is no longer just "compress SmolVLM." It's a system that:
1. Produces multiple compressed variants of SmolVLM-256M-Instruct (baseline,
   quantized, pruned, and optimized = pruned+quantized combined)
2. Monitors the device's live CPU/RAM (and temperature/battery where available)
3. Uses a **Resource-Aware Model Selector** (`utils/model_selector.py`) to pick
   the best-fitting variant automatically, based on available RAM and a
   user-chosen priority (Fast / Balanced / Accuracy / Auto)
4. Also recommends a matching input image resolution (smaller resolution =
   less activation memory during inference — a second compression lever
   alongside the model itself)
5. Serves all of this through a web app with a live dashboard, an AI
   workspace (captioning + VQA + camera input), a performance comparison
   view, and device monitoring

## Full run sequence (in order)

```bash
pip install -r requirements.txt

# 1. Produce baseline + all three compressed variants
python scripts/01_baseline.py
python scripts/02_quantize_dynamic.py
python scripts/04_prune.py
python scripts/08_combine_optimized.py     # NEW: pruned + quantized combined

# 2. Get accurate isolated memory footprint for each (recommended — see README)
python scripts/07_measure_isolated.py --model baseline
python scripts/07_measure_isolated.py --model quantized
python scripts/07_measure_isolated.py --model pruned
# (optimized variant uses the in-process footprint from step 1's script for now —
#  extend 07_measure_isolated.py the same way if you want an isolated reading for it too)

# 3. Build the comparison table
python scripts/06_benchmark.py

# 4. Launch the full adaptive web app
cd webapp
python app.py
# open http://localhost:5000
```

## What the web app's tabs do

- **Dashboard** — live CPU/RAM, current model in use, its size, last
  latency, current optimization mode, compression % vs baseline
- **AI Workspace** — upload or capture (webcam) an image, choose
  Captioning or VQA (type a question), choose a priority mode, and either
  let the Resource-Aware Selector pick the model or override it manually.
  Shows the Adaptive Optimization Panel explaining *why* that model was
  picked.
- **Comparison** — Original → Quantized → Pruned → Optimized table, pulled
  live from `results/comparison.csv`
- **Device Monitoring** — live CPU/RAM bars, temperature (Pi only — shows
  N/A on most laptops, which is expected), battery (N/A unless you add a
  UPS HAT to the Pi), and last inference latency

## Honest scope notes for your report

- The selector is **rule-based**, not learned — this is a deliberate,
  defensible choice: transparent and explainable, no training data needed.
  Mention "a learned/bandit-style selector" as future work if you want to
  show awareness of the next step.
- Temperature/battery reporting depends entirely on the OS/hardware
  exposing sensors to `psutil` — expect `None`/`N/A` on your Windows dev
  machine, and real values once running directly on the Pi (temperature)
  or with a UPS HAT (battery). This is correct, expected behavior, not a
  bug.
- Camera input uses the **browser's own webcam** via `getUserMedia` — this
  works identically whether the Flask server runs on your laptop or on the
  Pi (you'd browse to `http://<pi-ip>:5000` from your phone/laptop and use
  its camera, since the Pi itself has no browser UI running headless).
