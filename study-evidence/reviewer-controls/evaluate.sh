#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "%s\n" "$rc" > evaluate-exit-code' EXIT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
py=${1:?Provide the Python executable with PyTorch}
data=study-evidence/reviewer-controls
old=study-evidence/tactical-followup
date -u +%FT%TZ > evaluate-started-utc
mkdir matches
pids=()
match() {
    local name=$1
    shift
    nice -n 12 "$py" -u -m scripts.q_controls matches --pack "$data/openings.json" \
        --output "matches/$name.json" "$@" > "matches/$name.log" 2>&1
}
match heuristic --mode heuristic &
pids+=("$!")
for seed in 61 62 63 64 65; do
    (
        path="$old/runs/plain-$seed/training"
        match "lookahead-$seed" --mode lookahead --checkpoint "$path/move-080000.pt"
        match "initial-$seed" --mode initial --checkpoint "$path/move-000000.pt"
        match "guarded-$seed" --mode guarded --checkpoint "$path/move-080000.pt"
    ) &
    pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
((rc == 0)) || exit "$rc"

pids=()
for seed in 71 72 73 74 75; do
    (
        for horizon in 1 3; do
            run="n$horizon-$seed"
            path="runs/$run/training/move-080000.pt"
            until [[ -s "$path" ]]; do
                [[ ! -f train-exit-code ]] || { echo "Missing final weights for $run"; exit 1; }
                sleep 10
            done
            match "$run" --mode horizon --checkpoint "$path"
        done
    ) &
    pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
date -u +%FT%TZ > evaluate-ended-utc
exit "$rc"
