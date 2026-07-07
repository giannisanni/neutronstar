# NeutronStar — GLM-5.2 (743B) on a single consumer GPU

This is a fork of [antirez/ds4](https://github.com/antirez/ds4) (DwarfStar), collapsed
further. The `glm-local` branch you are reading adds a **CUDA port for GLM-5.2**, a stack of
SSD expert-streaming optimizations, and the **first MTP speculative-decoding
implementation for GLM 5.2 on any backend**.

The point of all of it: run a 743B-parameter MoE on hardware that has no business
running it. Reference machine: RTX 4060 Ti 16GB, 30GB DDR5, Ryzen 9900X, one
Gen4 NVMe. The model file is 196.6 GiB; routed experts are read from disk on
every token while ~20 GiB of attention/shared weights stay resident.

Current state on that machine: **~0.40 tokens/s generation, ~0.35 prefill**.
The campaign arc was 0.05 → 0.40 t/s on identical hardware, all software.
HuggingFace tells you a 4060 Ti cannot run this model. HuggingFace is wrong,
just slowly.

## The model

Grab the matching quant (custom merge, uniform-slab routed experts, MTP layer
included in the main file):

**[huggingface.co/giannisan/GLM-5.2-ds4-gguf](https://huggingface.co/giannisan/GLM-5.2-ds4-gguf)**

Recipe in that card. Short version: all routed experts uniform IQ2_XXS (the
streaming cache uses fixed-size slabs and the dp4a kernels decode IQ2_XXS
directly), everything that makes decisions stays at Q8_0/F32, and blk.78 (the
MTP draft layer) rides along at Q2_K inside the same file.

## What this branch adds over upstream

### GLM-5.2 CUDA port
Upstream runs GLM-5.2 on Metal. This branch makes the whole GLM path work on
CUDA: MLA attention, DSA sparse indexer, compact KV, indexed prefill, and the
routed-MoE kernels, including new IQ2_XXS dp4a down-projection kernels (upstream
had Q2_K down only).

### SSD streaming optimizations (the 0.05 → 0.40 arc)
- **Parallel fetch backfill** for expert-cache misses: 0.6 → 1.75 GB/s effective
  disk feed (measured at ~89% of the PCIe link ceiling).
- **io_uring + O_DIRECT fetch engine** (QD 64, `DS4_CUDA_FETCH_URING=1`), with a
  buffered-mode escape hatch (`DS4_CUDA_FETCH_BUFFERED=1`) for models that fit
  mostly in page cache.
- **Aligned buffer recycling pool** (kills per-fetch mmap churn).
- **Host expert cache with LFU eviction**: expert popularity is concentrated
  enough that the hottest ~4% of experts serve ~30% of lookups, so a 7 GiB RAM
  cache removes ~30% of all disk traffic (`DS4_CUDA_HOST_EXPERT_CACHE_GB`).
- **Cross-layer expert prefetch** (Fate-style router lookahead, 74% prediction
  accuracy; throughput-neutral while the drive is saturated, armed for faster
  disks, `DS4_GLM_EXPERT_PREFETCH=1`).

### MTP speculative decoding for GLM 5.2 (first implementation anywhere)
GLM-5.2 ships a draft head (blk.78) that no backend had wired up. This branch:
- binds it from the main gguf (no separate draft file, pass the same path to
  `--mtp`),
- runs it as a single-token predictor: measured **95% next-token hit rate** at
  temp 0,
- chains it recursively: d2 hits 61% conditional; depth 2 is the useful maximum
  (`DS4_GLM_MTP_DEPTH`),
- includes an accept loop (`DS4_GLM_MTP_ACCEPT=1`) with 2-token batch
  verification. The batch MoE kernels address whole expert tensors through
  model views, which under streaming OOM'd 30GB hosts;
  `DS4_GLM_INDEXED_PER_EXPERT_FFN=1` reroutes small indexed batches through
  the decode expert cache (only the selected experts load, per token), which
  makes the accept loop run within a decode-sized memory budget. Probe mode
  (`DS4_MTP_PROBE=1`) works everywhere.

  Status (measured, 30GB host): the loop runs correctly end to end
  (`batches=16 accepted=3 tokens=20`, byte-identical output) but is a net
  slowdown: per-token verify loads cost ~2 evals per 2-token batch, and
  2/(1+p) evals/token cannot beat plain decode for any acceptance p. The
  profitable version needs union expert loads across the verify rows (load
  each selected expert once, run both rows against it). Also open: d2
  acceptance measures 19% through the indexed-attention verify vs 61% in
  probe mode against full-attention decode; near-tie argmax flips between
  the two attention paths are the suspect.

### Latent CUDA-streaming bugs fixed along the way
Nobody had run interactive GLM sessions on CUDA streaming before, and it showed:
- the split batch-attention fast path hard-required the f16 compact cache
  (Apple-only) and silently killed any indexed batch on CUDA (f32 cache),
- every model-span install released the entire CUDA weight cache, forcing
  multi-GiB rebuilds per step in batch paths (installs are now skipped while the
  static decode map is live),
- the uniform-Q2_K expert-cache gate would overrun IQ2_XXS-sized cache slabs
  with a mixed-quant model (now slab-budget checked),
- session resume (chat turn 2+) routed into the batch prefill and OOM'd 30GB
  hosts (now token-major for small suffixes).

Plus `DS4_CUDA_ARENA_VRAM_RESERVE_GB` to keep VRAM headroom for batch kernels.

## Quick start

```sh
git clone -b glm-local https://github.com/giannisanni/neutronstar
cd ds4 && make cuda CUDA_ARCH=sm_89

M=GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf

# one-shot
DS4_GLM_CUDA_UNSAFE=1 DS4_CUDA_HOST_EXPERT_CACHE_GB=7 DS4_CUDA_PARALLEL_FETCH_THREADS=16 \
./ds4 -m $M --cuda --ssd-streaming --ssd-streaming-cache-experts 64 \
  --ctx 4096 --tokens 400 --nothink -p "Tell me something surprising about Suriname."

# interactive chat (Ollama-style): drop -p
# MTP probe telemetry: add --mtp $M with DS4_MTP_PROBE=1 DS4_MTP_STREAMING_UNSAFE=1
```

Memory sizing on a 30GB host: the resident weights want ~9-13 GiB VRAM plus
~8 GiB pinned host RAM; give the expert cache whatever is left minus a few GiB
of headroom (7 GiB cache is the knife-edge, 5 is comfortable).

## Environment knobs added by this branch

| Env | Default | What it does |
|---|---|---|
| `DS4_CUDA_HOST_EXPERT_CACHE_GB` | 0 | LFU host RAM cache for routed experts |
| `DS4_CUDA_PARALLEL_FETCH_THREADS` | 16 | expert fetch worker threads |
| `DS4_CUDA_FETCH_URING` / `DS4_CUDA_FETCH_QD` | on / 64 | io_uring O_DIRECT fetch engine |
| `DS4_CUDA_FETCH_BUFFERED` | 0 | page-cache reads (models ≲ RAM × small multiple) |
| `DS4_GLM_EXPERT_PREFETCH` | 0 | cross-layer router-lookahead prefetch |
| `DS4_GLM_MTP_DEPTH` | 4 | draft chain depth (2 is the useful max) |
| `DS4_GLM_MTP_ACCEPT` | 0 | experimental speculative accept loop (needs `--mtp` + `--temp 0`) |
| `DS4_GLM_INDEXED_PER_EXPERT_FFN` | 0 | small indexed batches load selected experts per token instead of whole tensors |
| `DS4_MTP_PROBE` | 0 | draft hit-rate telemetry |
| `DS4_CUDA_ARENA_VRAM_RESERVE_GB` | 0 | VRAM headroom the weight arena must not eat |
| `DS4_GLM_SYNC_TRACE` | 0 | session prefill branch tracing |

## Roadmap on this branch

Gen5 NVMe (drive-limited today: the engine runs at ~89% of the link ceiling),
second 16GB GPU (two-worker split removes the pinned-arena tax), per-expert
batch loads (unlocks the MTP accept loop), and GPU-initiated NVMe reads
(BaM-style; simulator already passing, design in `docs/gpu-nvme-design.md`).
MTP notes in `docs/glm-mtp-port.md`.

## Relation to upstream

Everything here builds on [antirez/ds4](https://github.com/antirez/ds4), which
in turn stands on llama.cpp/GGML. The `flash-local` branch carries the subset of
this work that applies to DeepSeek V4 Flash. The original upstream README is
preserved as `README.upstream.md`.
