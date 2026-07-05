# GLM-5.2 MTP port: implementation map

Goal: --mtp speculative decoding for GLM-5.2 on CUDA. No reference exists on
any backend (Metal marks it TODO); this is a first implementation. Expected
value ~1.5-1.8x effective decode, multiplicative with all disk/cache work.

## What upstream already did (verified in tree)

- Shape system knows `glm-dsa.nextn_predict_layers = 1` (DS4_N_NEXTN_PREDICT).
- Loader binds and validates blk.78 fully: a complete GLM layer
  (MLA attention + DSA indexer + 256-expert MoE at Q2_K + shared expert)
  PLUS the nextn glue on ds4_layer_weights: nextn_eh_proj (Q8_0,
  [2*6144, 6144]), nextn_enorm, nextn_hnorm, nextn_shared_head_norm
  (ds4.c ~4007 validation, ~4777 binding). weights.layer[78] is populated.
- The MTP accept loop is family-generic (ds4.c ~41315): drafts[16],
  DS4_MTP_STRICT, margin env, batch verify + accept. Reusable unchanged
  if the GLM eval path supplies drafts.

## The blockers to remove (all found)

1. ds4.c ~38806: `--mtp is not supported for GLM 5.2 yet` guard in engine
   open.
2. ds4.c ~35165: `ds4_engine_mtp_draft_tokens` returns 0 for GLM_DSA
   unconditionally.
3. ds4.c ~41200 (GLM eval branch): `(void)probe_mtp; return 0;` — the GLM
   token eval ignores drafting entirely. This is where the draft forward
   integrates.

## Architecture of the draft (DeepSeek MTP recipe, GLM flavor)

```
h_final = hidden after layer 77 (pre output-norm)
e = enorm(embed(current_token))          # rms_norm, kernels exist
h = hnorm(h_final)                        # rms_norm
x = eh_proj(concat(e, h))                 # q8_0 matmul 12288 -> 6144
    (equivalently: W[:, :6144] @ e + W[:, 6144:] @ h, two matmuls, no
     concat kernel needed if a matmul-with-view is easier)
x = GLM_layer_78(x)                       # full layer: MLA attn + DSA
                                          #   indexer + MoE; needs its OWN
                                          #   KV + indexer cache slot
logits = lm_head(shared_head_norm(x))    # shared embd/head with main model
draft = argmax(logits)
```

Differences from Flash's ds4_mtp_weights: GLM has fused eh_proj
(Flash: separate e_proj/h_proj); no hc_head_* (n_hc==1); block weights come
from the MAIN gguf (weights.layer[78]) instead of a separate --mtp file.

## Implementation steps

1. **Plumbing (small, self-contained):**
   - Engine open: for GLM, accept --mtp <any path or same file>; skip
     model_open of mtp_model; set a `mtp_glm_view` mode: e->mtp_ready=true.
   - Fix ds4_engine_mtp_draft_tokens to not zero GLM.
   - Guard instead on what is actually unsupported: distributed + GLM MTP.
2. **Graph state:** extend ds4_glm_gpu_graph with the draft layer's caches:
   one extra compact DSA KV row-space + indexer cache slot (the main alloc
   sizes kv_layers=78 for layers 0..77; the draft needs its own, sized like
   any other layer, plus scratch for e/h/x vectors). Alloc in the same
   DS4_GLM_GRAPH_ALLOC_TENSOR block; free alongside.
3. **Draft forward:** new function glm_graph_mtp_draft_one(g, model,
   weights, token, pos, h_final, logits_out). Reuses: embed q8_0 kernel,
   rms_norm, matmul_q8_0 (eh_proj split as two matmuls on weight column
   halves to avoid a concat), the existing per-layer attention+FFN
   machinery pointed at weights->layer[78] with the draft cache indices,
   output head matmul. Called from the GLM eval branch (~41200) when
   probe_mtp: fills s->mtp_logits, s->mtp_draft_token, s->mtp_draft_valid.
4. **Verify:** the generic accept loop calls the family batch eval for
   verification; confirm the GLM batch prefill path accepts the verify
   batch shape (it should: it is the prompt-prefill path; our PR #497
   analog for GLM = make sure single-token verify batches still load
   selected experts; watch for the same n_tokens==1 gate bug we fixed on
   main, ds4.c metal_graph_cuda_stream_prefill_batch_selected_load).
5. **Validation:** temp 0, DS4_MTP_STRICT=1: output must be token-identical
   with and without --mtp. Then throughput A/B at margin defaults, plus
   DS4_MTP_PROBE=1 hit-rate telemetry (existing counters print draft hit
   rates). MTP draft layer experts are Q2_K: they stream through the same
   expert path; check host-cache keying treats layer 78 offsets normally.

## Files

- ds4.c: engine open (38891-38907 region), mtp_draft_tokens (35165),
  GLM eval branch (41200 region), glm graph struct/alloc/free, the new
  draft forward.
- ds4_cuda.cu: nothing new expected; all kernels exist (embed q8_0,
  rms_norm, q8_0 matmuls, routed MoE with Q2_K, indexer, attention).

## Gotchas from the Flash MTP campaign (issue #495 / PR #497)

- Margin-mode verify encodes with n_tokens==1 skipped selected-expert
  loads on main; the fix was the n_tokens==0 gate. GLM's equivalent path
  needs the same audit before trusting verify results.
- Strict mode is the correctness oracle; margin mode is the speed mode.
- MTP+streaming needed DS4_MTP_STREAMING_UNSAFE on main before the fix;
  GLM may need an equivalent escape hatch during bring-up.
