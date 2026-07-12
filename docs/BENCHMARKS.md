# Benchmarking NeutronStar

Numbers in the README come from the commands below. If you file a
performance issue, please include the exact command, the full stderr, and
the hardware facts from step 0 — "it's slow on my machine" is not
reproducible; this is.

## 0. Report your hardware honestly

The token clock is `bytes-per-token / disk-bandwidth`, so the disk facts
matter more than the GPU:

```sh
# which device holds the model, and its link speed
df <model.gguf>
lsblk -o NAME,MODEL,TRAN,SIZE
cat /sys/class/nvme/nvme0/device/current_link_width   # want 4, not 1
cat /sys/class/nvme/nvme0/device/current_link_speed
free -g        # host expert cache steals from free RAM
nvidia-smi --query-gpu=name,memory.total --format=csv
```

A model file on a SATA SSD or a x1-linked NVMe will be several times
slower with identical software. Measured here: moving the same file from
SATA to Gen4 x4 NVMe was 5.1x by itself.

## 1. Generation (decode) benchmark

```sh
DS4_CUDA_HOST_EXPERT_CACHE_GB=16 \
DS4_CUDA_PARALLEL_FETCH_THREADS=16 \
DS4_GLM_EXPERT_PREFETCH=1 \
./ds4 -m Hy3-ds4-IQ2XXS-AttnQ8.gguf --cuda --ssd-streaming \
  --ssd-streaming-cache-experts 64 --ctx 4096 --temp 0 --raw \
  -n 48 -p "The three most important considerations when designing a database index are"
```

Read the `generation:` line at the end and the last
`host expert cache: N% hit rate` line. Run at least twice: run-to-run
variance across prompts is roughly ±0.15 t/s, so never conclude anything
from a single run. Reference (RTX 4060 Ti 16GB, Gen4 x4 NVMe, 16GB cache):
~2.1 t/s at ~89% hit rate; 20GB cache: ~90% and a shade faster.

The host cache is the main lever. Scale
`DS4_CUDA_HOST_EXPERT_CACHE_GB` to your free RAM and expect hit rate,
not GPU speed, to decide your tokens per second.

## 2. Long-prompt (prefill) benchmark

```sh
# ~1900-token prompt file
DS4_CUDA_HOST_EXPERT_CACHE_GB=16 \
DS4_CUDA_PARALLEL_FETCH_THREADS=16 \
DS4_GLM_EXPERT_PREFETCH=1 \
./ds4 -m Hy3-ds4-IQ2XXS-AttnQ8.gguf --cuda --ssd-streaming \
  --ssd-streaming-cache-experts 64 --ctx 4096 --temp 0 --raw \
  -n 4 --prompt-file longprompt.txt
```

Read the `prefill:` line. Reference: 6.1 t/s at the default 256-token
chunk (`DS4_HY3_PREFILL_CHUNK`); chunk scaling measured 64/128/256 =
2.77/4.17/6.06 t/s. `DS4_HY3_DISABLE_BATCH_PREFILL=1` gives the
token-major baseline (0.62 t/s on the reference box).

## 3. Warm vs cold starts

`DS4_CUDA_HOST_CACHE_STATE=/path/file` persists the cache index across
runs; the second run should open at roughly steady-state hit rate (84% on
the first 2048 lookups vs ~59% cold, reference box; full 16GB re-read in
~13s). Delete the state file to measure cold behavior.

## 4. Disk-side sanity

While a benchmark runs, sample the drive (no iostat needed):

```sh
python3 - <<'EOF'
import time
def snap():
    for l in open("/proc/diskstats"):
        f = l.split()
        if f[2] == "nvme0n1":
            return time.time(), int(f[5]) * 512, int(f[12])
prev = snap()
for _ in range(60):
    time.sleep(1); cur = snap()
    mb = (cur[1] - prev[1]) / 1e6
    busy = (cur[2] - prev[2]) / ((cur[0] - prev[0]) * 1000) * 100
    if mb > 50: print(f"{mb:8.0f} MB/s busy={busy:5.1f}%")
    prev = cur
EOF
```

With prefetch on, expect the disk ~60-70% busy during decode. If you see
~40%, prefetch is off; if you see ~100% at low MB/s, your disk or link is
the bottleneck (see step 0).

## Rules this repo benchmarks by

- Same prompt, same flags, same file placement across A/B runs.
- Cold-start numbers and warm numbers are different numbers; say which.
- One run proves nothing; report two or more.
- Correctness first: batch and token-major paths are argmax-equivalent on
  the top tokens but not bit-identical (activation quantization differs
  per kernel path), so compare speeds, not transcripts.
