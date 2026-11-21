set -euo pipefail
trap 'rc=$?; printf "%s\n" "$rc" > evaluate-exit-code' EXIT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
py=${1:-python}
pids=()
mkdir evaluation
for run in dqn-51 ddqn-51 dqn-52 ddqn-52 dqn-53 ddqn-53 tactics-only; do
  (
    args=()
    if [[ "$run" != tactics-only ]]; then
      checkpoint="runs/$run/training/game-005000.pt"
      until [[ -f "$checkpoint" ]]; do
        if [[ -f train-exit-code ]]; then
          echo "Missing checkpoint: $checkpoint"
          exit 1
        fi
        sleep 8
      done
      args=(--training "runs/$run/training")
    fi
    nice -n 10 "$py" -u -m scripts.dqn_study "${args[@]}" --excluded-pilot-openings excluded-pilot-openings.json --output "evaluation/$run"
  ) > "evaluation/$run.log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
date -u +%FT%TZ > evaluate-ended-utc
exit "$rc"
