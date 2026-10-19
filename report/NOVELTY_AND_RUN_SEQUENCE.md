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
6. Applies an event-driven frame gate to uploaded videos and live browser
   camera streams; uploaded videos select the frame with the strongest
   detected scene change and make one VLM call, while live camera streams
   invoke it for significant scene changes or a periodic heartbeat
7. Measures inference frequency, latency, CPU/RAM use, and invocation
   reduction against a session-specific frame-by-frame estimate; measures
   CPU-package energy only when an Intel RAPL counter is exposed
8. Uses a counterfactual event-value gate: it scores frame changes, routes
   uncertain events to a detailed caption, and adjusts its threshold from
   caption novelty feedback during a live session
9. Accepts a rolling CPU-package energy budget and increases the event
   threshold when measured energy debt accumulates; uncertain and high-value
   events bypass that deferral
10. Buffers skipped frames and recovers a likely missed event when a later
    event or heartbeat permits another VLM call
11. Links video captions to timestamps and source keyframe thumbnails
12. Corroborates visual activity with browser-measured audio RMS where
    available; it does not claim speech recognition or semantic audio labels
13. Applies a BLEU-4 retention floor against the FP32 family baseline during
    adaptive model selection, promoting uncertain events to accuracy-first
    selection
14. Uses prompt temporal/action phrases and simple color cues to adjust event
    relevance without loading another embedding model
15. Includes an annotated adaptive-vs-frame-by-frame benchmark for caption
    overlap, missed-event rate, latency, resource use, and measurable energy

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

For annotated evaluation, create a local JSON manifest as described in
`README.md`, then run:

```bash
python scripts/video_pipeline_benchmark.py data/video_eval_manifest.json \
  --quality-retention 0.90 \
  --output results/video_pipeline_benchmark.json
```

This executes both adaptive inference and a real frame-by-frame VLM comparator
over the supplied samples, so the comparison can take substantially longer.

## What the web app's tabs do

- **Dashboard** — live CPU/RAM, current model in use, its size, last
  latency, current optimization mode, compression % vs baseline
- **AI Workspace** — upload images or videos, capture a webcam still, or
  process a live camera stream. Video is sampled at 2 fps and redundant
  frames are checked for changes; an uploaded clip selects its most eventful
  sampled frame and runs one VLM inference rather than separate descriptions
  for every event. Choose Captioning or VQA (type a question), set an optional
  latency target, and either let the Resource-Aware Selector pick the model
  or override it manually.
  An optional CPU energy budget (J/min) adjusts inference frequency only
  when Intel RAPL energy readings are available. Borderline events receive
  detailed analysis; important and uncertain events are protected from
  energy-budget deferral. Caption-word overlap provides bounded within-session
  threshold feedback. Live streams buffer skipped frames for recovery.
  Captions carry timestamps and keyframe evidence. Adaptive model selection
  excludes variants below the configured BLEU-4 retention floor for their
  FP32 family baseline. Temporal/action phrases and requested-color cues
  influence query-aware frame relevance.
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
- Energy-based model selection is enabled only after comparable per-model
  CPU-package energy readings are available from Intel RAPL. Other machines
  show energy as unavailable rather than substituting battery percentage or
  an unverified power estimate. Session frame-by-frame latency/energy values
  are extrapolations from measured inference averages; they are not a second
  live baseline execution. Energy measurements cover VLM generation only,
  not browser decoding, frame-gate work, or idle energy.
- The energy-debt controller is deliberately meter-gated. Its budget is in
  CPU-package joules per rolling minute and it is inactive without reliable
  Intel RAPL readings; a user-entered budget does not cause battery percentage
  or CPU utilization to be misrepresented as energy. The rolling debt sums
  observed VLM-generation energy only, not idle or browser-decoding energy.
- Camera input uses the **browser's own webcam** via `getUserMedia` — this
  works identically whether the Flask server runs on your laptop or on the
  Pi (you'd browse to `http://<pi-ip>:5000` from your phone/laptop and use
  its camera, since the Pi itself has no browser UI running headless).
- Audio corroboration uses browser-side RMS activity near each frame
  timestamp. Uploaded audio decoding is limited to files up to 24 MB and
  120 seconds; live inference may request microphone permission. RMS does not
  identify speech or semantic sound events.
- Query-aware gating is a heuristic using temporal/action language,
  requested-color pixel ratios, and prior caption/query overlap. It is not a
  text-image embedding and cannot guarantee prompt-specific event detection.
- BLEU-4 retention is a benchmark proxy, not a per-video accuracy guarantee.
  Adaptive auto selection fails closed when quality profiles are missing or
  no quality-compliant model fits the RAM budget.
- The benchmark caption score is unigram token F1 against the best reference.
  Missed-event rate requires annotated event timestamps. Frame-by-frame mode
  invokes the VLM for every supplied frame and may be substantially slower.
