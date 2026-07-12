#!/bin/bash
# Launch the Hy3 BF16 quant job on a RunPod CPU pod. Run from the Mac.
# Prereqs: runpodctl configured; gh authed; HF write token available.
# Budget: pod ~$0.48/hr + 1.1TB volume ~$0.075/hr, ~5-6h => ~$3.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
KEY=$(python3 -c "import re;print(re.search(r\"=\s*'([A-Za-z0-9_-]{20,})'\",open('$HOME/.runpod/config.toml').read()).group(1))")
# NOTE: || true so a flaky ssh doesn't silently kill the script under set -e
HF_TOKEN=${HF_TOKEN:-$(ssh -o ConnectTimeout=15 gianni@substrate 'cat ~/.cache/huggingface/token' 2>/dev/null || true)}
[ -z "$HF_TOKEN" ] && { echo "no HF token (ssh to substrate failed?)"; exit 1; }
# REST API: the legacy graphql?api_key= endpoint 403s for new rpa_ keys
rest() { curl -s --max-time 30 -X "$1" "https://rest.runpod.io/v1/$2" \
         -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
         ${3:+-d "$3"}; }

echo "== 1. job script: fetched from the public repo (committed + pushed) =="
RAW_URL="https://raw.githubusercontent.com/giannisanni/neutronstar/hy3/docs/ops/runpod-bf16/pod_job.sh"
# refuse to launch if the local script differs from what the pod will fetch
if ! curl -sL --max-time 30 "$RAW_URL" | diff -q - "$HERE/pod_job.sh" >/dev/null; then
  echo "local pod_job.sh differs from $RAW_URL; commit + push first"; exit 1
fi
echo "job: $RAW_URL"

# No network volume: RunPod requires a $5 minimum balance to create one, and
# the job is self-contained anyway. A pod-local volume (dies with the pod, no
# resume; worst case rerun ~$4) carries the 1.1TB working set instead.
echo "== 2. create CPU pod (cpu3c, 16 vCPU / 32GB, 1.1TB local volume) =="
runpodctl create pod \
  --computeType CPU --secureCloud \
  --name hy3-bf16-quant \
  --imageName python:3.11-bookworm \
  --vcpu 16 --mem 32 \
  --containerDiskSize 30 \
  --volumeSize 1100 --volumePath /vol \
  --cost 0.80 \
  --env "HF_TOKEN=$HF_TOKEN" \
  --env "RUNPOD_API_KEY=$KEY" \
  --env "JOB_URL=$RAW_URL" \
  --env "CARD_B64=$(base64 < "$HERE/hf-model-card.md" | tr -d '\n')" \
  --args 'bash -c "curl -sL $JOB_URL -o /job.sh && bash /job.sh"'

cat <<EOF

Launched. The job self-terminates the pod when done (or on any failure);
the local volume dies with it, so nothing bills afterward.
Watch:   runpodctl pod list
Result:  https://huggingface.co/giannisan/Hy3-ds4-gguf  (file + build-log.txt)
EOF
