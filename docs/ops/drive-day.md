# Drive-day runbook — Samsung 9100 PRO (Gen5) install

State before install (captured 2026-07-06):
- Model: ~/ds4-model/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf (197G)
- Lives on: nvme0n1p3 = ROOT drive, 94% full, 29G free, shared with OS + prod containers
- RAM: 30GB total. Root TEAM NVMe at x4 (reseated, holding).
- Baseline GLM: ~0.40 t/s generation, disk-bound (~89% of this drive ceiling).

Goal: move the model onto a dedicated Gen5 x4 drive with no OS/container contention.

---

## 0. Before powering off (capture a real before-number)
    GLM_VERBOSE=1 glm-chat --tokens 40 -p "Explain MoE routing in two sentences."
Write down the prefill + generation t/s. This is the A/B before.

## 1. Physical install
- [ ] Power down, unplug PSU.
- [ ] Seat 9100 PRO in M2_1 (the CPU-direct slot, Gen5 x4). NOT a chipset slot.
- [ ] Same session: unplug the front-panel HDD-activity LED header (the blinking-LED
      you asked about, it is NZXT H510 HDD activity, harmless but you wanted it gone).
- [ ] Reseat check: firm click, retention screw in. Marginal contact = link trains low.

## 2. First boot checks (do NOT skip, link trains every boot)
    # must be 4, and speed should now be Gen5 (~32 GT/s) on the new drive
    for d in /sys/class/nvme/nvme*/device/current_link_width; do echo "$d = $(cat $d)"; done
    for d in /sys/class/nvme/nvme*/device/current_link_speed; do echo "$d = $(cat $d)"; done
    # kernel bandwidth complaints?
    sudo dmesg | grep -i "link is capped\|reduced\|degraded\|PCIe.*x1\|PCIe.*x2" | tail
If new drive shows x1/x2 or Gen4 speed: reseat before doing anything else. Do not benchmark a degraded link, the whole point is bandwidth.

## 3. Move the model onto the new drive
    # identify the new device (the one that is NOT nvme0)
    lsblk -d -o NAME,SIZE,MODEL | grep -i samsung
    # format + mount (adjust nvmeXn1 to the new device)
    sudo mkfs.ext4 -L glmfast /dev/nvme1n1
    sudo mkdir -p /mnt/glmfast && sudo mount /dev/nvme1n1 /mnt/glmfast
    sudo chown gianni:gianni /mnt/glmfast
    # copy model, verify, then repoint
    rsync -ah --progress ~/ds4-model/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf /mnt/glmfast/
    sha256sum /mnt/glmfast/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
    # expected: a49de64c5020...b07fea0 (matches the HF card)
Then edit ~/bin/glm-chat: change M=~/ds4-model/... to M=/mnt/glmfast/...
Only after the new run is verified: delete the old copy to free root.
    # rm ~/ds4-model/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf   # AFTER verify
Add to /etc/fstab so it remounts on boot (use blkid UUID, not /dev name).

## 4. Validate raw drive (expert read pattern)
Experts are ~large chunk random reads. Simulate:
    fio --name=expert --filename=/mnt/glmfast/fiotest --size=8G --rw=randread \
        --bs=1M --iodepth=32 --numjobs=4 --direct=1 --runtime=30 --time_based --group_reporting
    rm /mnt/glmfast/fiotest
Note the read bandwidth. Gen5 x4 should be multiples of the old root drive.

## 5. GLM A/B (the payoff number)
    GLM_VERBOSE=1 glm-chat --tokens 40 -p "Explain MoE routing in two sentences."
Compare generation t/s to step 0. This is the drive win, dedicated Gen5 + no contention.

    # MTP draft still healthy on new drive?
    DS4_MTP_PROBE=1 GLM_VERBOSE=1 glm-chat --tokens 40 -p "Count to twenty."
Expect d1 ~95%, d2 lower. Probe measures draft accuracy, not speed.

## 6. What this drive does and does NOT unlock (read this)
- DOES: raise base generation t/s (streaming is disk-bound). Frees 197G off root.
- Does NOT by itself unlock the MTP accept loop. That path (DS4_GLM_MTP_ACCEPT=1)
  batch-verifies 2 tokens, which reads WHOLE expert tensors and OOMs under 30GB RAM.
  The Gen5 drive makes reads faster but does not shrink that memory spike.
  Accept loop needs EITHER: +RAM (so the batch does not OOM), OR the per-expert
  batch-load code change (verify path loads experts per-token instead of whole
  tensors). Do not expect the 2x from MTP on drive-day alone.
- The accept-loop CODE is already correct and committed (DS4_GLM_MTP_ACCEPT=1),
  reviewed 2026-07-06. It is flip-the-switch once memory allows, not a rewrite.

## 7. Order of the roadmap after this
1. Gen5 drive (this): base t/s up, root freed. <- you are here
2. +RAM to ~64GB: unlocks accept loop (flip DS4_GLM_MTP_ACCEPT=1) -> the MTP 2x.
3. 2nd GPU: more resident non-routed weight, less streaming.

## UPDATE 2026-07-06: per-expert batch loads LANDED (accept loop may not need +RAM)

New env: DS4_GLM_INDEXED_PER_EXPERT_FFN=1
Routes the MTP verify batch (and any small indexed prefill chunk) through the
decode expert cache: only the ~8 selected experts per token per layer load,
instead of whole ~2.4GB/layer expert tensors. This removes the memory spike
that OOMd the accept loop on 30GB.

Smoke test (works TODAY, before any hardware). BOTH extra flags matter:
--mtp binds the draft head (blk.78); --temp 0 selects the argmax generate
path, which is where the accept loop lives (default temp 1.0 diverts to the
session engine, which only has the Phase A probe, no accept).
    M=~/ds4-model/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
    DS4_GLM_MTP_ACCEPT=1 DS4_GLM_INDEXED_PER_EXPERT_FFN=1 \
    GLM_DEBUG=1 glm-chat --mtp $M --temp 0 --tokens 40 -p "Count from one to twenty in words."
Look for: "using GPU graph generation" (right path), then
"mtp accept summary: batches=N accepted=M" with N>0 and no OOM.
Known gap: interactive chat (no -p) runs the session engine, which has no
accept loop yet; porting it there is the follow-up work.
Compare generation t/s vs plain glm-chat. Accept helps when d1 hits (~95%),
so repetitive/structured prompts show it best.

VERDICT (measured 2026-07-06, run 3): the loop RUNS end to end on 30GB,
correct output, no OOM: batches=16 accepted=3 (19%) tokens=20 evals=33.
BUT it is net-SLOWER than plain decode (0.19 vs 0.32 t/s): per-token verify
loads make a 2-token batch cost ~2 evals, and 2/(1+p) evals/token cannot
beat 1.0 for any acceptance p. Do NOT enable it for daily use yet.
CORRECTION (same night): union loads do NOT make it profitable either.
Expert selections barely overlap across rows (~10-20%), so a 2-token batch
costs ~1.85 evals even with union loads = 1.15 evals/token at the probe d2
rate. Speculative decoding has almost nothing to amortize on a
disk-streaming MoE; every speculated token drags its own ~5GB off disk.
Park MTP accept entirely; revisit only if compute share ever dominates
(post-drive measurement will tell). The drive + RAM + 2nd GPU stack is
the real speedup path.
Second open issue: d2 acceptance 19% via indexed verify vs 61% in probe
(indexed vs full attention argmax flips suspected). Both issues documented
in the README on github (giannisanni/neutronstar, commit cbe1a54).

## UPDATE 2026-07-07: long-prompt prefill FIXED and 21x faster (commit 1d8a26c)

Batch prefill was producing fluent garbage on GLM (uninitialized IQ2 dequant
LUTs in the CUDA expert-tile kernels, only correct for n_embd<=4096 models).
Fixed. Measured after fix: 600-token prompt prefills at 6.5 t/s vs 0.30
token-major (21x), correct retrieval from the prefilled context.
- Long prompts (>64 tokens) now batch automatically, nothing to enable.
- A full 4k-context paste is ~10 min today; scales with the Gen5 drive
  (expect roughly 2-4x more from the 9100 PRO since the chunk read is
  sequential, the drive pattern batch prefill loves).
- This bug also corrupted the MTP verify logits (19% vs 61% d2), so the
  accept-loop acceptance number needs remeasuring IF that path is ever
  revisited; the structural evals-per-token argument still stands.
- Drive-day A/B: add a 600-token prompt prefill test to step 5:
    time ./ds4 ... --tokens 5 --temp 0 --nothink -p "$(cat /tmp/longprompt_q.txt)"
