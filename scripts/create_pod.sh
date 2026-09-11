#!/usr/bin/env bash
# Provision the sweep pod. Reads the manifest for infra config, picks the first GPU in
# gpu_preference_order that has stock in the right datacenter, and passes RUNPOD_API_KEY
# into the pod env (read from ~/.runpod/config.toml) so the pod can self-terminate.
#
#   ./scripts/create_pod.sh [manifest]
#
# The API key is read from disk and handed to runpodctl; it is never printed.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="${1:-spec/runs/lr-sweep.json}"
cd "$REPO"

cfg=$(python3 - "$MANIFEST" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
i = m["infra"]
print(i["datacenter"], i["network_volume_id"], i["volume_mount_path"],
      i["template_id"], i["min_cuda_version"], "|".join(i["gpu_preference_order"]))
PY
)
read -r DC VOL MOUNT TEMPLATE CUDA GPUS <<<"$cfg"

KEY=$(grep -m1 -iE '^[[:space:]]*apikey' "$HOME/.runpod/config.toml" 2>/dev/null | sed -E 's/.*=[[:space:]]*"?([^"]+)"?.*/\1/' || true)
if [ -z "$KEY" ]; then
  echo "WARNING: no apiKey found in ~/.runpod/config.toml." >&2
  echo "         The pod will not be able to self-terminate; you must delete it by hand." >&2
  read -r -p "Continue anyway? [y/N] " a; [ "$a" = y ] || exit 1
fi

echo "Checking stock in $DC for: ${GPUS//|/, }"
STOCK=$(runpodctl gpu list -o json)

IFS='|' read -ra ORDER <<<"$GPUS"
for gpu in "${ORDER[@]}"; do
  have=$(echo "$STOCK" | python3 -c "
import json,sys
d=json.load(sys.stdin); g=d if isinstance(d,list) else d.get('gpus',d.get('data',[]))
name,dc='$gpu','$DC'
for x in g:
    if (x.get('displayName') or '') == name or (x.get('gpuId') or '') == name:
        for a in x.get('dataCenterAvailability') or []:
            if a.get('dataCenterId')==dc and str(a.get('stockStatus')).lower() not in ('none','null',''):
                print(a.get('stockStatus')); break
")
  if [ -n "$have" ]; then
    echo "-> $gpu has stock ($have) in $DC; provisioning"
    set +e
    if [ -n "$KEY" ]; then
      runpodctl pod create --name "chagas-sweep" --template-id "$TEMPLATE" \
        --gpu-id "$gpu" --data-center-ids "$DC" --network-volume-id "$VOL" \
        --volume-mount-path "$MOUNT" --min-cuda-version "$CUDA" \
        --env "{\"RUNPOD_API_KEY\":\"$KEY\"}" --wait
    else
      runpodctl pod create --name "chagas-sweep" --template-id "$TEMPLATE" \
        --gpu-id "$gpu" --data-center-ids "$DC" --network-volume-id "$VOL" \
        --volume-mount-path "$MOUNT" --min-cuda-version "$CUDA" --wait
    fi
    rc=$?; set -e
    [ $rc -eq 0 ] && { echo "Pod created on $gpu."; exit 0; }
    echo "Provisioning $gpu failed (low-stock listings are racy); trying next in the order." >&2
  else
    echo "-> $gpu: no stock in $DC"
  fi
done

echo "FAILED: no GPU in the manifest preference order could be provisioned in $DC." >&2
echo "Per failure_policy.gpu_out_of_stock: halt and report. Do not substitute another GPU." >&2
exit 1
