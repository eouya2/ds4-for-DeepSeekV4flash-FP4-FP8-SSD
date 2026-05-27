# ds4 for DeepSeek V4 Flash FP4/FP8 - SSD Sidecar

This is a DeepSeek V4 Flash-specific `ds4` fork for running the FP4/FP8 SSD
sidecar Flash-MoE package published by Anemll.

The current implementation targets Apple Metal and has been validated on macOS.

This project is based on DwarfStar 4 / `ds4`, the original DeepSeek V4
Flash-specific inference engine. The work here keeps the normal ds4 runtime
surface and adds native support for Anemll's dense GGUF + SSD sidecar MoE
layout.

- Model package: `anemll/DeepSeek-V4-Flash-FP4-FP8-SSD`
- Base model: `deepseek-ai/DeepSeek-V4-Flash`
- Engine base: DwarfStar 4 / `ds4`
- Reference runtime: `Anemll/anemll-flash-llama.cpp`

The SSD package is not a single dense GGUF. It is a dense GGUF plus a sidecar
MoE expert bank:

```text
DeepSeek-V4-Flash-FP4-FP8-SSD/
  dense/
    model-dense.gguf              # dense/shared tensors
    flashmoe-package.json
  sidecar/
    manifest.json                 # sidecar metadata
    layer_000.bin ... layer_042.bin
```

The dense GGUF stores dense/shared tensors mostly as `F8_E4M3_B128` FP8. The
routed MoE experts are stored outside the dense GGUF as `MXFP4` sidecar tensors.
This fork keeps that layout and runs it directly; it does not rewrite, prune, or
re-quantize the model.

## Download the Model

Use the modern `hf` CLI, not `huggingface-cli`:

```bash
hf download anemll/DeepSeek-V4-Flash-FP4-FP8-SSD \
  --local-dir /path/to/DeepSeek-V4-Flash-FP4-FP8-SSD
```

Example local path:

```bash
hf download anemll/DeepSeek-V4-Flash-FP4-FP8-SSD \
  --local-dir ./models/DeepSeek-V4-Flash-FP4-FP8-SSD
```

Expected size is roughly:

```text
dense/model-dense.gguf  ~8.4 GiB
sidecar/                ~137 GiB
whole package           ~145 GiB
```

## Build

On macOS with Metal:

```bash
make
```

Useful binaries:

```text
./ds4          CLI / interactive runner
./ds4-server   OpenAI/Anthropic-compatible local server
./ds4-bench    prompt/generation throughput benchmark
```

## Run

You can pass the SSD package directory directly:

```bash
./ds4 \
  -m /path/to/DeepSeek-V4-Flash-FP4-FP8-SSD \
  --ctx 512 \
  --nothink \
  --temp 0 \
  -n 64 \
  -p "Hello!"
```

The runner resolves this automatically:

```text
-m /path/to/DeepSeek-V4-Flash-FP4-FP8-SSD
=> dense/model-dense.gguf
=> sidecar/manifest.json + layer_*.bin
```

Inspect the package:

```bash
./ds4 --inspect -m /path/to/DeepSeek-V4-Flash-FP4-FP8-SSD
```

Expected summary:

```text
experts: count=256 used=6
ssd sidecar: .../sidecar/manifest.json
entries: total=129 mxfp4=129
coverage: layers_with_gate_up_down=43/43
tensor types include: bf16, f32, i32, f8_e4m3_b128
```

## Reference Runtime

The reference implementation for this model layout is:

```text
https://github.com/Anemll/anemll-flash-llama.cpp
```

Reference command:

```bash
/path/to/anemll-flash-llama.cpp/build/bin/llama-cli \
  -m /path/to/DeepSeek-V4-Flash-FP4-FP8-SSD/dense/model-dense.gguf \
  --moe-sidecar /path/to/DeepSeek-V4-Flash-FP4-FP8-SSD/sidecar \
  --moe-mode slot-bank \
  --moe-slot-bank 6 \
  --ctx-size 512 \
  --n-predict 64 \
  --temp 0 \
  --reasoning off \
  --no-warmup \
  --simple-io \
  --no-display-prompt \
  --single-turn \
  -p "Hello!"
```

This repo includes a small comparison helper:

```bash
python3 speed-bench/compare_ssd_runtime.py \
  --model /path/to/DeepSeek-V4-Flash-FP4-FP8-SSD \
  --reference-bin /path/to/anemll-flash-llama.cpp/build/bin/llama-cli
```

Local development result on an Apple M5 Max with 128 GB unified memory
(`ctx=512`, `n=64`):

```text
case   ds4 prefill   ref prefill   ds4 gen   ref gen   total ratio
short  7.29 t/s      5.30 t/s      9.09      5.00      1.75x
code   6.82 t/s      4.80 t/s      8.42      4.70      1.74x
ko     11.13 t/s     5.00 t/s      8.93      5.70      1.71x
```

These are single-run local numbers and should be treated as smoke/engineering
comparisons, not a formal benchmark.

## What Was Implemented

### SSD package loading

`ds4`, `ds4-server`, and `ds4-bench` now accept the SSD package directory. The
engine resolves:

```text
<package>/dense/model-dense.gguf
<package>/sidecar/manifest.json
```

You can also pass an explicit sidecar path:

```bash
./ds4 -m /path/to/dense/model-dense.gguf --moe-sidecar /path/to/sidecar
```

### Dense FP8 support

The dense GGUF uses `F8_E4M3_B128` tensors. This fork adds native Metal matmul
and attention-output paths for that tensor type.

### MXFP4 sidecar MoE support

The routed experts live in sidecar layer banks as `MXFP4`. The engine parses the
sidecar manifest, binds the routed `ffn_gate_exps`, `ffn_up_exps`, and
`ffn_down_exps` tensors, and routes MoE execution through native Metal kernels.

### Slot-bank execution

The full sidecar is much larger than the dense GGUF, so the Metal path does not
try to make the GPU randomly page through every expert tensor directly. Instead:

1. The router selects top-k experts.
2. The runtime finds the experts needed for the current token/prefill chunk.
3. Needed experts are installed into a resident per-layer slot-bank.
4. Selected expert ids are remapped to slot ids.
5. The existing routed MoE graph runs over the compact resident bank.

Sidecar misses are loaded with bounded parallel `pread`, which is important for
SSD-backed Flash-MoE performance.

Useful knobs:

```bash
--moe-slot-bank N                     # default 16 for SSD sidecar mode
DS4_METAL_DISABLE_MXFP4_SLOT_BANK=1   # debug: disable slot-bank path
DS4_METAL_MXFP4_SERIAL_PREAD=1        # debug: force serial sidecar pread
DS4_METAL_MXFP4_PREAD_WORKERS=N       # default 4, capped at 16
DS4_METAL_MXFP4_SLOT_PROFILE=1        # print slot-bank install stats
```

### Existing ds4 features kept

The SSD path is integrated into the normal ds4 graph path. Existing CLI/server
features remain available, including:

- interactive CLI
- `ds4-server`
- `ds4-bench`
- disk/memory KV cache machinery
- directional steering options
- optional MTP flags and code path

MTP support was preserved in the code path, but local MTP execution was not
verified because the MTP GGUF was not present during testing.

## Validation Performed

Commands run locally:

```bash
make ds4
make ds4-server ds4-bench
make ds4_test && ./ds4_test --metal-kernels

./ds4 --inspect -m /path/to/DeepSeek-V4-Flash-FP4-FP8-SSD

./ds4 \
  -m /path/to/DeepSeek-V4-Flash-FP4-FP8-SSD \
  --ctx 512 --nothink --temp 0 -n 24 -sys "" \
  -p "Hello!"

python3 speed-bench/compare_ssd_runtime.py \
  --min-prefill-ratio 1.0 \
  --min-generation-ratio 1.0 \
  --min-total-ratio 1.0
```

Longer smoke on the same Apple M5 Max 128 GB machine:

```text
8192 prompt tokens:
prefill 11.86 t/s, generation 6.75 t/s
```

`32768` context allocation plus short generation also succeeded locally.

## Limitations

- The fast SSD path is currently validated on Apple Metal.
- CUDA/NVIDIA support for this SSD MXFP4 sidecar path is not complete; CUDA stubs
  exist to keep the build/API surface intact.
- AMD is not supported by this fork.
- MTP options are preserved, but real MTP execution still needs a local MTP GGUF
  smoke test.
- This is not a generic GGUF runner. It is specialized for DeepSeek V4 Flash and
  this Flash-MoE SSD package layout.
- The model is mixed quantization, not "all 4-bit":
  - dense/shared tensors: FP8 `F8_E4M3_B128`
  - routed MoE experts: FP4/MXFP4 sidecar
  - some metadata/norm/router tensors: BF16/F32/I32
- No model quality benchmark was performed here. The validation is loading,
  execution, kernel smoke, long-context smoke, and speed comparison.

## Notes for GitHub

Do not commit the model package or generated GGUF/sidecar files to this repo.
Keep this repository as the runtime source tree and download the model with
`hf download` as shown above.
