# GPU-initiated NVMe expert streaming: design and runbook

Goal: CUDA-side NVMe reads for routed experts, CPU out of the storage path,
building on the gpu-nvme-direct PoC (BaM-style, consumer hardware). Tiered so
every step ships value alone.

## Why this composes with what exists

- The expert-fetch seam is one function boundary: `cuda_fetch_execute()`
  consumes an array of `cuda_fetch_job` (offset, bytes, destination). Any
  backend that fills the destinations satisfies the engine.
- The DS4BNDL1 expert bundle is already the right on-disk layout: one
  contiguous record per expert, 4096-aligned, offsets in a header. A raw
  block device is just a big file to its reader: byte offset = LBA * 512.
  Deploying to a raw namespace is `dd` + a verify pass, no format change.
- The router runs on the GPU. Today the host reads back selected ids
  (a sync) to submit I/O. GPU-initiated I/O closes that loop on-device:
  router kernel -> SQ entry -> doorbell, no host round-trip. The cross-layer
  prefetch predictor becomes a fully GPU-side pipeline.

## Tiers

**Tier 0 (works today, no kmod):** open the raw block device with O_DIRECT,
point the existing io_uring engine + bundle reader at it. No filesystem
overhead, validates the raw layout end to end. Fallback if BaM stalls.

**Tier 1 (kmod, no P2P):** queues + data buffers in pinned host RAM
(`gpunvme_dma_alloc_host`), GPU builds commands and rings doorbells via BAR0
MMIO, NVMe DMAs into pinned host, existing H2D upload stage unchanged.
Wins: no syscalls, no io-wq, no CPU wakeups, GPU-autonomous reads.
PoC measured 3.35 GB/s sustained on a Gen3 x4 link this way.

**Tier 2 (kmod + nvidia_p2p on open modules):** data buffers in VRAM
(`gpunvme_load_layer_vram` / `gpunvme_dma_alloc_gpu`). NVMe DMAs experts
directly into the compact decode buffers. Host RAM untouched (matters:
30GB box, 9GB already pinned by the GLM arena), upload stage deleted.
Substrate prerequisites already met: NVIDIA open kernel modules 580.82,
BAR1 = 16GB (ReBAR active), kernel 6.17 with headers.

## Integration sketch

New backend in ds4_cuda.cu behind `DS4_CUDA_GPUNVME=1`:

```
cuda_fetch_execute(jobs, njobs)
  ├── host cache pre-pass (unchanged)
  ├── if gpunvme ready and job->fd is the raw bundle device:
  │     Tier 1: gpunvme_read_blocks(ctrl, lba(job->offset), nblocks,
  │              pinned_buf) from a persistent CUDA kernel or host-launch;
  │              then existing upload + host-cache donation
  │     Tier 2: gpunvme_load_layer_vram(...) straight into job->dst
  └── else: io_uring path (unchanged)
```

LBA math: `lba = (bundle_base + job->offset) / 512`, lengths already
4096-aligned by the bundle writer. MDTS chunking handled by the PoC's
block_io layer.

Later (the real unlock): move submission into the decode graph itself so
the router's selected ids never leave the GPU. The prefetch predictor's
next-layer ids feed the same path one layer early.

## Safety rails (non-negotiable)

- Target device: the Samsung 9100 PRO ONLY, identified by /dev/disk/by-id,
  never by bare /dev/nvmeXnY. The PoC's setup_vfio.sh already refuses the
  boot drive; we wrap it with an allowlist of the 9100's serial.
- The root TEAM NVMe and the Kingston never get unbound, ever.
- Models needed during experiments live on Kingston (247GB free).
- vfio unbind makes the drive invisible to Linux until teardown.sh; plan
  disk usage around experiment windows.
- IOMMU: PoC wants it off or passthrough. Check /proc/cmdline; changing it
  means a GRUB/kernelstub edit + reboot window on the prod box: announce
  before doing.
- Secure Boot: absent on substrate (Pop!_OS), unsigned kmod loads fine.

## Monday runbook (ordered)

1. Install 9100 PRO in M2_1. Boot, confirm root drive untouched.
2. Filesystem-first benchmarks (safe wins, no kmod):
   fio at the expert pattern (3.2MB random reads, QD 16/32/64),
   GLM re-bench + prefetch A/B (expect prefetch to pay now),
   io_uring QD sweep, Flash re-download + re-bench with fetch fix.
3. Decide the disk split: models on the 9100 for daily use vs dedicating
   it (or a partition-sized namespace, if the drive supports multiple
   namespaces) to raw experiments.
4. Separate window, explicit go-ahead: build + load kmod (already
   compiles against 6.17 headers per PoC), setup_vfio.sh against the
   9100 by-id, run PoC hardware milestone tests, then first
   gpunvme_read_blocks of a real DS4BNDL1 record into pinned memory.
5. Tier 1 seam into cuda_fetch_execute, A/B against io_uring on the same
   drive/bundle.
6. Tier 2 only after Tier 1 is stable.

## Open questions

- 9100 PRO multi-namespace support (would let one drive serve both fs
  models and a raw experiment namespace; check `nvme id-ctrl` nn field).
- 580.82 open-module compatibility with the PoC's nvidia_p2p usage
  (author ran 590.48 patched; Tier 1 needs none of it).
- 5060 Ti as the I/O GPU: it gets the CPU-attached Gen5 x8 slot; the
  4060 Ti moves behind the chipset. If the 9100 is CPU-attached (M2_1)
  and the 4060 Ti is chipset-attached, Tier 2 P2P crosses the chipset
  uplink. Topology to verify with the PoC's pcie topology tool.
