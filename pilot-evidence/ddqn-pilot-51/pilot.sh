set -euo pipefail
trap 'rc=$?; printf "%s\n" "$rc" > exit-code' EXIT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
py=${1:-python}
date -u +%FT%TZ > started-utc
CUDA_VISIBLE_DEVICES=5 nice -n 5 "$py" -u -m scripts.double_dqn train --method ddqn --games 5000 --decay-games 4000 --save-every 1000 --seed 51 --output training --device cuda
date -u +%FT%TZ > training-ended-utc
mkdir evaluation
pids=()
for game in 000000 001000 002000 003000 004000 005000; do
  (
    CUDA_VISIBLE_DEVICES= nice -n 10 "$py" -m scripts.double_dqn evaluate --checkpoint "training/game-$game.pt" --opponent random --pairs 32 --seed 80421 --output "evaluation/random-$game.json"
    CUDA_VISIBLE_DEVICES= nice -n 10 "$py" -m scripts.double_dqn evaluate --checkpoint "training/game-$game.pt" --opponent minimax --pairs 32 --seed 80421 --output "evaluation/minimax-$game.json"
  ) > "evaluation/game-$game.log" 2>&1 &
  pids+=("$!")
done
eval_rc=0
for pid in "${pids[@]}"; do
  wait "$pid" || eval_rc=1
done
date -u +%FT%TZ > ended-utc
exit "$eval_rc"
