#!/bin/bash
# Hy3 BF16 -> ds4-native IQ2_XXS quant, RunPod CPU pod job.
# Runs unattended; expects env: HF_TOKEN (write), RUNPOD_API_KEY, RUNPOD_POD_ID.
# Volume mounted at /vol (1.1TB). Image: python:3.11-bookworm.
# Cost profile: 16 vCPU cpu3c ~$0.48/hr + volume ~$0.075/hr, ~5-6h total.
set -uo pipefail
VOL=/vol
LOG="$VOL/job.log"
# RECIPE selects the resident-set quant (experts always IQ2_XXS, streamed from SSD):
#   attnq8: attn/shexp/dense/embd Q8_0 (16GB cards)   attnq4: Q4_K, output Q6_K (8GB cards)
RECIPE="${RECIPE:-attnq8}"
case "$RECIPE" in
  attnq8) RES_T=q8_0; OUT_T=q8_0; EXPECT_EMBD=Q8_0; OUTFILE=Hy3-ds4-IQ2XXS-AttnQ8-fromBF16.gguf ;;
  attnq4) RES_T=q4_K; OUT_T=q6_K; EXPECT_EMBD=Q4_K; OUTFILE=Hy3-ds4-IQ2XXS-AttnQ4-fromBF16.gguf ;;
  *) echo "unknown RECIPE=$RECIPE"; exit 1 ;;
esac
mark() { echo "=== $1 $(date -u +%H:%M:%S) ===" | tee -a "$LOG"; }
die() {
  echo "FATAL: $1" | tee -a "$LOG"
  # ship the log so failures are diagnosable (local volume dies with the pod)
  hf upload giannisan/Hy3-ds4-gguf "$LOG" "build-log-failed-$RECIPE.txt" >/dev/null 2>&1 || true
  terminate 1
}
terminate() {
  echo "JOB_EXIT rc=$1 $(date -u)" | tee -a "$LOG"
  # self-terminate the pod so billing stops even if nobody is watching
  # (REST API; the legacy graphql?api_key= endpoint 403s for new rpa_ keys)
  curl -s --max-time 30 -X DELETE "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" >/dev/null
  exit "$1"
}

mark "PHASE 0: deps"
# hf first so die() can ship the log from any later phase
pip install -q --no-cache-dir "huggingface_hub[hf_transfer]" || { echo "FATAL: pip hf" | tee -a "$LOG"; terminate 1; }
apt-get update -qq && apt-get install -y -qq cmake g++ git curl aria2 >/dev/null 2>&1 || die "apt"
pip install -q --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu || die "torch cpu"
pip install -q --no-cache-dir gguf safetensors "transformers>=4.45" sentencepiece pyyaml || die "pip deps"
export HF_HUB_ENABLE_HF_TRANSFER=1

# Resume support: a balance-stop keeps /vol but restarts this script from
# scratch. Detect completed phases by their volume artifacts. The q8_0
# intermediate counts as complete only when its header parses AND it is
# past the truncation floor (convert writes it incrementally).
Q8="$VOL/hy3-q8_0.gguf"
q8_complete() {
  [ -f "$Q8" ] || return 1
  [ "$(stat -c%s "$Q8" 2>/dev/null || echo 0)" -gt 300000000000 ] || return 1
  python - "$Q8" <<'PY' >/dev/null 2>&1
import sys
from gguf import GGUFReader
GGUFReader(sys.argv[1])
PY
}

mark "PHASE 1: network speed gate"
if [ -f "$Q8" ] || [ -d "$VOL/bf16" ]; then
  echo "resume detected: volume has prior progress, skipping speed gate" | tee -a "$LOG"
else
  # pull 1GB of a known-fast HF file; require >=45MB/s or bail out cheap
  T0=$(date +%s)
  curl -sL --max-time 120 -r 0-1073741823 -o /dev/null \
    "https://huggingface.co/tencent/Hy3/resolve/main/model-00001-of-00099.safetensors" || die "speed test fetch"
  T1=$(date +%s); MBPS=$(( 1024 / (T1 - T0 + 1) ))
  echo "network: ~${MBPS} MB/s" | tee -a "$LOG"
  [ "$MBPS" -lt 45 ] && die "network too slow (${MBPS} MB/s), aborting to save credit"
fi

mark "PHASE 2: llama.cpp (PR 25395, arch string patched to hy-v3)"
cd /root
git clone --depth 1 -q https://github.com/ggml-org/llama.cpp.git || die "clone"
cd llama.cpp
git fetch --depth 1 -q origin pull/25395/head:hy3-pr && git checkout -q hy3-pr || die "pr fetch"
# ds4 + the reference GGUFs use "hy-v3"; the PR says "hy_v3". Patch BOTH sides.
grep -rl '"hy_v3"' src/llama-arch.cpp gguf-py/gguf/constants.py | while read -r f; do
  sed -i 's/"hy_v3"/"hy-v3"/g' "$f"; done
grep -q '"hy-v3"' src/llama-arch.cpp || die "arch patch failed (llama-arch)"
grep -q '"hy-v3"' gguf-py/gguf/constants.py || die "arch patch failed (gguf-py)"
cmake -B build -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF >/dev/null 2>&1 || die "cmake"
cmake --build build --target llama-quantize -j"$(nproc)" >/dev/null 2>&1 || die "build quantize"

mark "PHASE 3: download BF16 (598GB) + imatrix"
if q8_complete; then
  echo "resume: q8_0 intermediate complete, skipping bf16 download" | tee -a "$LOG"
  rm -rf "$VOL/bf16"
else
  # hf download resumes: existing complete files are skipped
  hf download tencent/Hy3 --local-dir "$VOL/bf16" >>"$LOG" 2>&1 || die "bf16 download"
fi
hf download YanissAmz/Hy3-295B-A21B-GGUF Hy3.imatrix.gguf --local-dir "$VOL" >>"$LOG" 2>&1 || die "imatrix download"
df -h "$VOL" | tee -a "$LOG"

mark "PHASE 4: convert BF16 -> Q8_0 GGUF"
if q8_complete; then
  echo "resume: skipping convert" | tee -a "$LOG"
else
  rm -f "$Q8"
  python convert_hf_to_gguf.py "$VOL/bf16" --outtype q8_0 \
    --outfile "$Q8" >>"$LOG" 2>&1 || die "convert"
fi
rm -rf "$VOL/bf16"
df -h "$VOL" | tee -a "$LOG"

mark "PHASE 5: quantize to ds4 recipe (uniform IQ2_XXS experts, $RES_T rest, recipe=$RECIPE)"
out_complete() {
  [ -f "$VOL/$OUTFILE" ] || return 1
  [ "$(stat -c%s "$VOL/$OUTFILE" 2>/dev/null || echo 0)" -gt 70000000000 ] || return 1
  python - "$VOL/$OUTFILE" <<'PY' >/dev/null 2>&1
import sys
from gguf import GGUFReader
GGUFReader(sys.argv[1])
PY
}
if out_complete; then
  echo "resume: quantized output present, skipping quantize" | tee -a "$LOG"
else
rm -f "$VOL/$OUTFILE"
# NOTE: --tensor-type is unanchored regex_search; keep patterns anchored.
# NOTE: NO pipes on this command (head/grep SIGPIPE killed earlier runs).
./build/bin/llama-quantize --allow-requantize --imatrix "$VOL/Hy3.imatrix.gguf" \
  --tensor-type 'blk\.([0-9]|[1-7][0-9])\.ffn_(gate|up|down)_exps\.weight=iq2_xxs' \
  --tensor-type 'blk\.80\.ffn_(gate|up|down)_exps\.weight=q2_k' \
  --tensor-type 'blk\.80\.nextn\.eh_proj\.weight=q8_0' \
  --tensor-type "attn_(q|k|v|output)\\.weight=$RES_T" \
  --tensor-type "ffn_(gate|up|down)_shexp\\.weight=$RES_T" \
  --tensor-type "blk\\.0\\.ffn_(gate|up|down)\\.weight=$RES_T" \
  --token-embedding-type "$RES_T" --output-tensor-type "$OUT_T" \
  "$Q8" "$VOL/$OUTFILE" iq2_xxs "$(nproc)" \
  >>"$LOG" 2>&1 || die "quantize"
fi

mark "PHASE 6: verify output header"
EXPECT_EMBD="$EXPECT_EMBD" python - "$VOL/$OUTFILE" <<'PY' >>"$LOG" 2>&1 || die "header verify"
import os, sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
f = r.fields["general.architecture"]
arch = bytes(f.parts[f.data[0]]).decode()
assert arch == "hy-v3", f"arch={arch}"
q = {}
for t in r.tensors:
    q.setdefault(str(t.tensor_type).split(".")[-1], 0)
    q[str(t.tensor_type).split(".")[-1]] += 1
print("quant mix:", q)
exps = [t for t in r.tensors if "ffn_gate_exps" in t.name]
assert all("IQ2_XXS" in str(t.tensor_type) for t in exps), "experts not uniform IQ2_XXS"
emb = [t for t in r.tensors if t.name == "token_embd.weight"][0]
want = os.environ["EXPECT_EMBD"]
assert str(emb.tensor_type).endswith(want), f"embd {emb.tensor_type}, want {want}"
print("HEADER_OK")
PY

mark "PHASE 7: upload to HF"
[ -n "${CARD_B64:-}" ] && echo "$CARD_B64" | base64 -d > /tmp/README.md && \
  hf upload giannisan/Hy3-ds4-gguf /tmp/README.md README.md >>"$LOG" 2>&1
hf upload giannisan/Hy3-ds4-gguf "$VOL/$OUTFILE" "$OUTFILE" >>"$LOG" 2>&1 || die "hf upload"
hf upload giannisan/Hy3-ds4-gguf "$VOL/job.log" "build-log-$RECIPE.txt" >>"$LOG" 2>&1

mark "ALL DONE — self-terminating"
terminate 0
