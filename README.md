# Connect X training and model source

Connect Four self-play training, Kaggle submission export, and comparisons between a pretrained reference model and an experimental fine-tune.

This project builds on [Eugene Peng's connect4-alphazero](https://github.com/eugeneyp/connect4-alphazero), upstream commit `a936aadd74a6a6dd9035e576b89aeb5a3ea7ef73`. The reference weights were downloaded from that project; they were not trained from scratch here. See [NOTICE](NOTICE) and the retained [MIT license](LICENSE).

## Included checkpoints

| File | Origin and status |
|---|---|
| `models/large.pt` | Upstream pretrained large model. Its submitted agent reached a user-reported Kaggle rating of **1154**. |
| `models/iteration2.pt` | Experimental fine-tune from this work: learning rate `8e-5`, six epochs. Its Kaggle rating is unmeasured. |

Both checkpoints contain a five-block, 128-filter policy/value network in the repository's native PyTorch checkpoint format. [Model hashes](models/manifest.json) identify the files.

Iteration 2 used **448 completed self-play games**, generated with **512 MCTS simulations per move**, yielding **25,952 samples including horizontal reflections**. The actual seven compressed training shards are committed under [`data/iteration2`](data/iteration2), with a [manifest](data/iteration2/manifest.json). Generation stopped at its time limit; incomplete games were excluded.

Fine-tuning freezes BatchNorm statistics, mixes deeper-search policies with the reference policy, and blends final game outcomes with reference value predictions. These choices and the other learning rates are recorded in [`reports/iteration2.json`](reports/iteration2.json).

## Results

The actual submission archives played **48 games across 24 distinct openings**, with both player assignments and unchanged per-move time budgets:

| Result | Games |
|---|---:|
| Iteration 2 wins | 21 |
| Reference wins | 18 |
| Draws | 9 |

Iteration 2 scored **53.125%**, counting draws as half a win. A paired-opening bootstrap interval of **44.8–62.5%** includes 50%, so this does not establish a reliable improvement. No agent errors occurred, and both stayed below 1.69 seconds per move. Raw games, timing records, and archive hashes are in [`reports/headtohead`](reports/headtohead).

An earlier independent comparison at fixed simulation counts produced 22 wins, 21 losses, and 5 draws for iteration 2. A larger confirmation exceeded its evaluation time limit and was not scored. Local comparisons do not establish a Kaggle rating gain.

## Setup and training

Use Python 3.11 or newer. Fine-tuning requires a CUDA GPU.

```bash
pip install '.[dev,kaggle]'
CUDA_VISIBLE_DEVICES=0 python iteration2.py train \
  --data data/iteration2 --output checkpoints/iteration2 \
  --seed 202 --lr 0.00008
```

This repeats the selected fine-tuning procedure from the saved data, starting at `models/large.pt`. GPU kernels and software versions can affect exact results. The provided checkpoint preserves the model that was evaluated.

`iteration2.py generate` produces new self-play data; `scripts/train.py` retains the upstream full-training workflow. `finetune_20min.py` contains the earlier bounded experiment and paired fixed-simulation evaluation.

## Build Kaggle submissions

```bash
python scripts/kaggle_submit.py --checkpoint models/large.pt \
  --output submissions/reference --tar --time-budget 1.9
python scripts/kaggle_submit.py --checkpoint models/iteration2.pt \
  --output submissions/iteration2 --tar --time-budget 1.9
```

Upload the generated `submission.tar.gz` unchanged. Each archive contains `main.py` and `weights.npz`; the agent needs NumPy and the Python standard library. Export removes negligible subnormal weights to avoid slow CPU arithmetic. The agent handles immediate tactical moves and reserves time before returning.

## Benchmark the actual archives

```bash
for worker in 0 1 2 3; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python archive_benchmark.py \
    --reference submissions/reference/submission.tar.gz \
    --candidate submissions/iteration2/submission.tar.gz \
    --output experiments/headtohead/worker_$worker.json \
    --worker "$worker" --workers 4 --pairs 24 &
done
wait
```

Use separate physical CPU cores for the workers when comparing wall-clock performance. Results are saved after each game. `verify_submission.py` checks exported predictions against PyTorch and plays both sides against Kaggle's built-in random and negamax agents.

## Tests

```bash
python -m pytest tests/test_board.py tests/test_model.py tests/test_mcts.py \
  tests/test_agents.py tests/test_numpy_export.py tests/test_iteration2.py \
  tests/test_archive_benchmark.py -q
```

The upstream `tests/test_training.py` also contains training and benchmarking integration tests that take longer to run.
