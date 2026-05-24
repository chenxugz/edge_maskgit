# Known issues / TODO

## 1. Pre-quantized VQGAN conv (int8) NaNs in XNNPACK  — open

**Symptom.** When the VQGAN decoder conv weights are stored **pre-quantized** as
per-output-channel int8 in the GGUF and fed to XNNPACK's
`xnn_define_convolution_2d` as a `qd8-f32-qc8w` convolution (dynamic int8 input +
`qcint8` filter), the decoded image is **all NaN** from the first conv.

**What works (so we have a correct path).** The *exact same* int8 conv via the
**on-load** path — where the runtime reads F32 conv weights and quantizes them to
int8 in C++ at subgraph-build time — produces a correct image (`cosine 0.99939`
vs the PyTorch oracle, ~0.78 s decode). The transformer's **FC** pre-quant path
(`qc8w`/`qc4w` read from the GGUF) also works perfectly.

**Why it's puzzling.** We verified the two paths feed XNNPACK *byte-identical*
inputs:
- Parsed the q8 GGUF and confirmed the stored conv int8 bytes **and** per-channel
  scales match a fresh re-quantization exactly (`int8 match: True`, `scale match: True`).
- All non-conv F32 tensors (codebook, GroupNorm affines, biases) are byte-identical
  between the F32 and q8 GGUFs.
- `dims`, `channel_dim`, and the `xnn_define_channelwise_quantized_tensor_value`
  call are identical between on-load and pre-quant.
- Copying the mmap'd int8/scale into owned buffers (ruling out mmap/lifetime
  aliasing) did **not** fix it.

So the difference is solely the weight *data source*, with provably identical
bytes — yet on-load works and pre-quant NaNs.

**Current workaround.** Quantized GGUFs keep VQGAN **conv weights F32** and
quantize them **on load** (proven, fast). Only the transformer **FC** weights are
stored pre-quantized (int8/int4). This is why `q4` is ~207 MB rather than the
~125 MB ideal — the F32 conv weights are the remaining bulk. The pre-quant conv
code path in `src/mg-xnn-vqgan.cpp` throws if reached.

**Repro.** Re-enable conv quantization in `tools/convert_to_gguf.py` (the
`is_conv` branch, currently a no-op), regenerate `maskgit-256-q8.gguf`, and run
`./bazel-bin/verify-xnn-vqgan models/maskgit-256-q8.gguf reference/export` →
`nan=196608/196608`.

**Hypotheses to chase.**
- XNNPACK's `qd8-f32-qc8w` igemm/conv microkernel may require a specific weight
  *packing* or alignment that the on-load path happens to satisfy (it allocates
  fresh buffers) but the GGUF-resident layout does not — e.g. a row-padding or
  per-channel-block packing expectation for conv (vs FC) filters.
- The dynamic-input quantization (`num_nonbatch_dims=3`) interacting with a
  particular intermediate scale; instrument by making `conv_in`'s output an
  external output and inspecting it directly.
- Try `xnn_create_runtime` with `XNN_FLAG_*` weight-cache options, or define the
  conv filter via the v2/v3 channelwise API.

**Payoff when fixed.** `q4` model file drops ~207 MB → ~125 MB; lower on-device
RSS for the VQGAN stage.
