#!/usr/bin/env bash
set -euo pipefail
trap 'rc=$?; printf "%s\n" "$rc" > train-exit-code' EXIT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
py=${1:?Provide the Python executable with CUDA-enabled PyTorch}
date -u +%FT%TZ > train-started-utc
mkdir runs
pids=()
for spec in "0 71 1 3" "1 72 3 1" "2 73 1 3" "3 74 3 1" "4 75 1 3"; do
    read -r gpu seed first second <<< "$spec"
    (
        for horizon in "$first" "$second"; do
            run="runs/n$horizon-$seed"
            mkdir "$run"
            date -u +%FT%TZ > "$run/started-utc"
            CUDA_VISIBLE_DEVICES="$gpu" nice -n 8 "$py" -u -m scripts.tactical_dqn \
                --variant plain --n-step "$horizon" --moves 80000 --decay-moves 64000 \
                --seed "$seed" --output "$run/training" --device cuda > "$run/train.log" 2>&1
            date -u +%FT%TZ > "$run/ended-utc"
        done
    ) &
    pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
date -u +%FT%TZ > train-ended-utc
exit "$rc"
