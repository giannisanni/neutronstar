---
base_model: tencent/Hy3
base_model_relation: quantized
tags:
- gguf
- hy3
- moe
- ds4
- ssd-streaming
---

# Hy3 (295B) GGUF for ds4/NeutronStar (SSD streaming, CUDA)

Mixed-precision GGUF of [tencent/Hy3](https://huggingface.co/tencent/Hy3)
(295B total / 21B active MoE), quantized from the original BF16 checkpoint
for the [NeutronStar](https://github.com/giannisanni/neutronstar) `hy3`
branch: a CUDA port of [ds4](https://github.com/antirez/ds4) that streams
routed experts from disk so the model runs on GPUs that cannot hold it.

Reference machine: RTX 4060 Ti 16GB, 32GB RAM, one NVMe. Per token only 8
of 192 experts per layer are read (~3GB/token at this quant); attention,
shared experts, and the router stay resident.

## Recipe

The layout targets ds4's streaming expert cache: routed experts must be
uniform fixed-size slabs, and everything that makes decisions stays high
precision. Same design as the GLM-5.2 ds4 build, including the MTP layer
riding at Q2_K because importance matrices never cover the draft layer
(imatrix generation runs normal forwards, which skip it).

| Tensors | Type | Why |
|---|---|---|
| routed experts, layers 1-79 (gate/up/down) | IQ2_XXS (imatrix) | streamed from disk per token; uniform slabs |
| routed experts, layer 80 (MTP) | Q2_K | no imatrix coverage exists for the draft layer |
| attention q/k/v/output, all layers | Q8_0 | resident, paid once |
| shared expert + dense layer 0 FFN | Q8_0 | resident |
| nextn.eh_proj (MTP glue) | Q8_0 | tiny, no imatrix coverage |
| token embeddings, output head | Q8_0 | ds4 embed kernel contract |
| router (ffn_gate_inp), expert bias, all norms | F32 | decision makers stay exact |

imatrix: the 125-chunk general-purpose matrix published with
[YanissAmz/Hy3-295B-A21B-GGUF](https://huggingface.co/YanissAmz/Hy3-295B-A21B-GGUF).

Architecture string is `hy-v3` (matching the reference GGUFs; llama.cpp
PR 25395 patched from `hy_v3` before conversion).

## Usage

```sh
git clone -b hy3 https://github.com/giannisanni/neutronstar
cd neutronstar && make cuda CUDA_ARCH=sm_89
./ds4 -m Hy3-ds4-IQ2XXS-AttnQ8.gguf --cuda --ssd-streaming \
  --ctx 4096 --tokens 200 --temp 0 -p "Hello"
```

MTP speculative decoding is not wired for Hy3 yet; blk.80 is present in
the file so it can be enabled later without requantizing.

Built unattended on a RunPod CPU pod from the BF16 master; build log
included in this repo as build-log.txt.
