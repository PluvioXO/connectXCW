set -euo pipefail
trap 'rc=$?; printf "%s\n" "$rc" > evaluate-exit-code' EXIT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
py=${1:-python}
mkdir evaluation
date -u +%FT%TZ > evaluate-started-utc
pids=()
for seed in 61 62 63 64 65; do
  for variant in plain tactical; do
    run="$variant-$seed"
    (
      for step in 0 20000 40000 60000 80000; do
        printf -v name 'move-%06d.pt' "$step"
        until [[ -f "runs/$run/training/$name" ]]; do
          if [[ -f train-exit-code ]]; then
            echo "Missing checkpoint runs/$run/training/$name"
            exit 1
          fi
          sleep 10
        done
        nice -n 12 "$py" -u -m scripts.tactical_study evaluate --training "runs/$run/training" --testpack testpack.json --output "evaluation/$run/step-$step" --step "$step"
      done
    ) > "evaluation/$run.log" 2>&1 &
    pids+=("$!")
  done
done
nice -n 12 "$py" -u -m scripts.tactical_study evaluate --testpack testpack.json --output evaluation/tactics-only > evaluation/tactics-only.log 2>&1 &
pids+=("$!")
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
date -u +%FT%TZ > evaluate-ended-utc
exit "$rc"
