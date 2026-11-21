set -euo pipefail
trap 'rc=$?; printf "%s\n" "$rc" > lookahead-exit-code' EXIT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
py=${1:-python}
mkdir lookahead
date -u +%FT%TZ > lookahead-started-utc
pids=()
for seed in 61 62 63 64 65; do
  for variant in plain tactical; do
    run="$variant-$seed"
    (
      until [[ -f "runs/$run/training/move-080000.pt" ]]; do
        if [[ -f train-exit-code ]]; then echo "Missing final weights for $run"; exit 1; fi
        sleep 10
      done
      nice -n 12 "$py" -u -m scripts.learned_lookahead evaluate --training "runs/$run/training" --pack lookahead-pack.json --output "lookahead/$run"
    ) > "lookahead/$run.log" 2>&1 &
    pids+=("$!")
  done
done
nice -n 12 "$py" -u -m scripts.learned_lookahead evaluate --pack lookahead-pack.json --output lookahead/tactics-only > lookahead/tactics-only.log 2>&1 &
pids+=("$!")
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
date -u +%FT%TZ > lookahead-ended-utc
exit "$rc"
