#!/bin/bash
# Launch the Hy3 BF16 quant job on a RunPod CPU pod. Run from the Mac.
# Prereqs: runpodctl configured; gh authed; HF write token available.
# Budget: pod ~$0.48/hr + 1.1TB volume ~$0.075/hr, ~5-6h => ~$3.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
KEY=$(python3 -c "import re;print(re.search(r\"=\s*'([A-Za-z0-9_-]{20,})'\",open('$HOME/.runpod/config.toml').read()).group(1))")
HF_TOKEN=${HF_TOKEN:-$(ssh gianni@substrate 'cat ~/.cache/huggingface/token' 2>/dev/null)}
[ -z "$HF_TOKEN" ] && { echo "no HF token"; exit 1; }
gql() { curl -s --max-time 30 "https://api.runpod.io/graphql?api_key=$KEY" \
        -H 'Content-Type: application/json' -d "{\"query\":\"$1\"}"; }

echo "== 1. host the job script as a secret gist (no secrets inside it) =="
GIST_URL=$(gh gist create --secret "$HERE/pod_job.sh" 2>/dev/null | tail -1)
RAW_URL="${GIST_URL/gist.github.com/gist.githubusercontent.com}/raw/pod_job.sh"
echo "gist: $GIST_URL"

echo "== 2. create 1.1TB network volume (>1TB tier = \$0.05/GB/mo) =="
for DC in EU-RO-1 CA-MTL-1 EUR-IS-1 US-KS-2; do
  R=$(gql "mutation { createNetworkVolume(input: {name: \\\"hy3-quant\\\", size: 1100, dataCenterId: \\\"$DC\\\"}) { id dataCenterId } }")
  VOLID=$(echo "$R" | python3 -c "import json,sys;d=json.load(sys.stdin);print((d.get('data') or {}).get('createNetworkVolume',{}).get('id') or '')" 2>/dev/null)
  [ -n "$VOLID" ] && { echo "volume $VOLID in $DC"; break; }
  echo "  $DC: no ($(echo "$R" | head -c 120))"
done
[ -z "${VOLID:-}" ] && { echo "no volume created"; exit 1; }

echo "== 3. create CPU pod (cpu3c, 16 vCPU / 32GB) =="
runpodctl create pod \
  --computeType CPU --secureCloud \
  --name hy3-bf16-quant \
  --imageName python:3.11-bookworm \
  --vcpu 16 --mem 32 \
  --containerDiskSize 30 \
  --networkVolumeId "$VOLID" --volumePath /vol \
  --dataCenterId "$DC" \
  --cost 0.60 \
  --env "HF_TOKEN=$HF_TOKEN" \
  --env "RUNPOD_API_KEY=$KEY" \
  --env "JOB_URL=$RAW_URL" \
  --env "CARD_B64=$(base64 < "$HERE/hf-model-card.md" | tr -d '\n')" \
  --args 'bash -c "curl -sL $JOB_URL -o /job.sh && bash /job.sh"'

cat <<EOF

Launched. The job self-terminates the pod when done (or on any failure).
Watch:   runpodctl pod list
Result:  https://huggingface.co/giannisan/Hy3-ds4-gguf  (file + build-log.txt)
AFTER SUCCESS, delete the volume (it bills until deleted!):
  curl -s "https://api.runpod.io/graphql?api_key=\$KEY" -H 'Content-Type: application/json' \\
    -d '{"query":"mutation { deleteNetworkVolume(input: {id: \\"$VOLID\\"}) }"}'
Also delete the gist: gh gist delete ${GIST_URL##*/}
EOF
