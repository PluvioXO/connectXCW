set -euo pipefail
trap 'rc=$?; printf "%s\n" "$rc" > train-exit-code' EXIT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
py=${1:-python}
date -u +%FT%TZ > train-started-utc
pids=()
for spec in "1 dqn 51" "2 dqn 52" "3 ddqn 52" "4 dqn 53" "5 ddqn 53"; do
  read -r gpu method seed <<< "$spec"
  run="runs/$method-$seed"
  mkdir "$run"
  (
    date -u +%FT%TZ > "$run/started-utc"
    CUDA_VISIBLE_DEVICES="$gpu" nice -n 5 "$py" -u -m scripts.double_dqn train --method "$method" --games 5000 --decay-games 4000 --save-every 1000 --seed "$seed" --output "$run/training" --device cuda
    date -u +%FT%TZ > "$run/ended-utc"
  ) > "$run/train.log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
date -u +%FT%TZ > train-ended-utc
exit "$rc"
