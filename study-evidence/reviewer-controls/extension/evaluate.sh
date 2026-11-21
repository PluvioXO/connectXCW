#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "%s\n" "$rc" > evaluate-exit-code' EXIT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
py=${1:?Provide the Python executable with PyTorch}
date -u +%FT%TZ > evaluate-started-utc
mkdir matches
match() {
    local name=$1 mode=$2 checkpoint=$3
    until [[ -s "$checkpoint" ]]; do
        [[ ! -e train-exit-code ]] || { echo "Missing weights for $name"; exit 1; }
        sleep 5
    done
    nice -n 12 "$py" -u -m scripts.q_controls matches --pack openings.json \
        --output "matches/$name.json" --mode "$mode" --checkpoint "$checkpoint" \
        > "matches/$name.log" 2>&1
}
pids=()
for seed in 81 82 83 84 85; do
    (
        one="runs/n1-$seed/training"
        three="runs/n3-$seed/training"
        match "initial-$seed" initial "$one/move-000000.pt"
        match "lookahead-$seed" lookahead "$one/move-080000.pt"
        match "guarded-$seed" guarded "$one/move-080000.pt"
        match "n1-$seed" horizon "$one/move-080000.pt"
        match "n3-$seed" horizon "$three/move-080000.pt"
    ) &
    pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
date -u +%FT%TZ > evaluate-ended-utc
exit "$rc"
