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

Mixed-precision GGUFs of [tencent/Hy3](https://huggingface.co/tencent/Hy3)
(295B total / 21B active MoE, Apache 2.0) built for the
[NeutronStar](https://github.com/giannisanni/neutronstar) `hy3` branch: a
CUDA port of [ds4](https://github.com/antirez/ds4) that streams routed
experts from disk, so the model runs on GPUs that cannot hold it.

Reference machine: RTX 4060 Ti 16GB, Ryzen 9900X, 32GB RAM, one Gen4 NVMe.
Per token only 8 of 192 experts per layer are read (~3GB/token at this
quant); attention, shared experts, and the router stay resident. Measured:
~1.8 t/s generation with a 16GB host expert cache (68% hit rate),
interactive chat with KV retained across turns.

## Which file to download

| File | Quantized from | For |
|---|---|---|
| `Hy3-ds4-IQ2XXS-AttnQ8-fromBF16.gguf` | the original BF16 checkpoint | **recommended** for 16GB cards: canonical single-step quantization |
| `Hy3-ds4-IQ2XXS-AttnQ4-fromBF16.gguf` | the original BF16 checkpoint | 8GB cards: resident set (attention, shared expert, dense FFN, embeddings) at Q4_K instead of Q8_0, output head Q6_K; routed experts identical IQ2_XXS |
| `Hy3-ds4-IQ2XXS-AttnQ8.gguf` | the IQ4-UD edition of [YanissAmz/Hy3-295B-A21B-GGUF](https://huggingface.co/YanissAmz/Hy3-295B-A21B-GGUF) | the original build; a requantization of an existing quant, kept for reproducibility |

The two AttnQ8 files run identically in ds4 (same shapes, same slab sizes,
same speed). The difference is quality margin: the older file went IQ4/IQ3 -> IQ2_XXS,
so its routed experts carry a second round of quantization noise. At a
2-bit target the extra loss is small (the 2-bit noise dominates), but the
fromBF16 build removes it entirely. The tensors that were already Q8_0 in
the source (attention, shared expert, dense FFN, embeddings, output head)
passed through essentially lossless either way. The requant is kept for
reproducibility and because early benchmarks were run on it.

The AttnQ4 file trades resident-set precision for VRAM: the streamed
experts (the bulk of compute) are bit-identical to the AttnQ8 build, only
the always-resident tensors drop to Q4_K, roughly halving the VRAM
footprint so the model fits 8GB cards. Expect a modest quality dip on the
attention path; the router and norms stay F32 in all files.

## Recipe (AttnQ8 files; AttnQ4 swaps the Q8_0 rows for Q4_K, output head Q6_K)

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
cd neutronstar && make ds4
./ds4 -m Hy3-ds4-IQ2XXS-AttnQ8-fromBF16.gguf --cuda --ssd-streaming \
  --ssd-streaming-cache-experts 64 --ctx 4096 --nothink
```

No prompt drops you into interactive chat (KV retained across turns).
Useful knobs: `DS4_CUDA_HOST_EXPERT_CACHE_GB=16` (host expert cache, the
main speed lever; scale to your free RAM) and
`DS4_CUDA_PARALLEL_FETCH_THREADS=16`.

MTP speculative decoding is not wired for Hy3 (measurements on GLM show
the accept loop cannot pay while expert streaming dominates eval cost);
blk.80 is present in both files so it can be enabled later without
requantizing.

The fromBF16 files were built unattended on RunPod pods straight from
the BF16 master; build logs are in this repo as build-log-attnq8.txt and
build-log-attnq4.txt.
