set -euo pipefail
trap 'rc=$?; printf "%s\n" "$rc" > train-exit-code' EXIT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
py=${1:-python}
date -u +%FT%TZ > train-started-utc
mkdir runs
pids=()
for spec in "1 61 plain tactical" "2 62 tactical plain" "3 63 plain tactical" "4 64 tactical plain" "5 65 plain tactical"; do
  read -r gpu seed first second <<< "$spec"
  (
    for variant in "$first" "$second"; do
      run="runs/$variant-$seed"
      mkdir "$run"
      date -u +%FT%TZ > "$run/started-utc"
      CUDA_VISIBLE_DEVICES="$gpu" nice -n 8 "$py" -u -m scripts.tactical_dqn --variant "$variant" --moves 80000 --decay-moves 64000 --save-every 20000 --seed "$seed" --output "$run/training" --device cuda > "$run/train.log" 2>&1
      date -u +%FT%TZ > "$run/ended-utc"
    done
  ) &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
date -u +%FT%TZ > train-ended-utc
exit "$rc"
