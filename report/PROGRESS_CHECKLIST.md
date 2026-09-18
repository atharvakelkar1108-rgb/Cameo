# Progress Checklist (4 weeks)

## Week 1 — Baseline
- [ ] Install requirements, verify `01_baseline.py` runs end-to-end
- [ ] Confirm baseline size / latency / memory / BLEU-4 numbers look sane
- [ ] Read up on quantization vs pruning vs distillation (for your report's "related work")

## Week 2 — Quantization
- [ ] Run `02_quantize_dynamic.py`, compare vs baseline
- [ ] (Stretch) implement static PTQ with calibration (`03_quantize_static.py` — not yet written, build from `02` + `torch.quantization.prepare/convert`)
- [ ] Export to ONNX (`05_export_onnx.py` — not yet written, use `torch.onnx.export`)

## Week 3 — Pruning
- [ ] Run `04_prune.py`, check accuracy drop before/after fine-tune
- [ ] If time allows: swap unstructured pruning for `torch-pruning`'s structured pruner for a "real" model that's genuinely smaller/faster, not just sparse

## Week 4 — Deployment + writeup
- [ ] Deploy best compressed model on target device (Pi/Jetson) or simulated CPU-constrained env
- [ ] Run `06_benchmark.py` to generate `results/comparison.csv` and `results/comparison_plot.png`
- [ ] Write report: motivation, related work, method, results, conclusion
- [ ] Prepare slides + demo video/live demo

## Not yet built (build these when you get there)
- `03_quantize_static.py` — static PTQ with calibration pass (uses same
  structure as `02_quantize_dynamic.py`, but needs `model.qconfig`,
  `torch.quantization.prepare`, a calibration loop over ~100 samples, then
  `torch.quantization.convert`)
- `05_export_onnx.py` — `torch.onnx.export(model, dummy_input, "model.onnx")`
  then benchmark with `onnxruntime.InferenceSession`
