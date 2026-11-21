# Connect X experiment guide

Run the commands from the repository or the extracted code and evidence archive.
The study replay commands use raw checkpoints and traces from the full archive.
The [coursework guide](../coursework/README.md) gives the final million-game
self-play summary and the ten-seed Double DQN results.

## Persistent self-play history

`python -m scripts.selfplay_run` starts from `models/large.pt`, generates 256 games per iteration
with 512 MCTS simulations per move, and trains for four epochs on the latest
16 saved 64-game shards. It reuses the conservative policy/value targets and
frozen BatchNorm statistics from `iteration2.py`. Each candidate plays 128 games
against its retained best model, using 64 openings with both seats and equal
512-simulation budgets. Promotion requires score >=55% and a paired-bootstrap
95% lower bound >50%. This gate is local evidence, not a Kaggle score guarantee.

Initially Weatherwax ran two independent branches on physical GPUs 4 and 5, using seeds
4101/5101 and learning rates 3e-5/8e-5, each bounded to 100 iterations (25,600
self-play games). Their root is `/mnt/fast0/mn866-connectx-training`, with outputs
`runs/selfplay_gpu4_20261115` and `runs/selfplay_gpu5_20261115`. Each saves a plan,
status, JSONL metrics, replay shards, epoch losses, candidates, paired arenas and
`best.pt`. The original model files remain unchanged. Jobs and source hashes are
recorded in `runs/jobs_20261115.json` and `source_manifest.json`.

The runner requires a new output directory unless `--resume` is passed. Recovery
restores the last committed champion, reuses completed self-play shards, and
replays the incomplete candidate's training/evaluation. It preserves the
iteration limit; partial games are restarted. In this original mode each
candidate starts from its branch's retained best weights with fresh Adam state.
A separate Weatherwax watchdog checks exited jobs every minute.

On 16 November the two experiments were consolidated into one continuation of
GPU 5's 8e-5 branch at `runs/selfplay_shared_20261116`. The old run directories
remain archived. The continuation used `--gpus 4,5 --resume` with
`CUDA_VISIBLE_DEVICES=4,5`: each generation worker produces alternate 64-game
shards from the same retained champion. All four shards feed one learner and
the rolling replay buffer. The short four-epoch optimization runs on GPU 4;
two workers split the 128-game candidate arena across GPUs 4 and 5. The
256-game iteration size, 512-search budget and promotion criteria stay fixed.
The continuation retains GPU 5's game/iteration history and its 100-iteration
limit. `gpu_transition.json` records the migration and checkpoint/source hashes.

On 17 November the ConnectX continuation moved to
`runs/selfplay_sixgpu_20261117` on physical GPUs 0–5. Six workers produce one
64-game shard each (384 games per iteration); all feed the same learner at
8e-5. The rolling replay remains capped at 16 shards. The 64 paired evaluation
openings are split 11/11/11/11/10/10 across six workers, retaining 128 games and
the existing promotion rule. Optimization uses one GPU. Generation and arena
evaluation use the full allocation. The original ConnectX runs are archived.

The 17 November throughput update removes redundant PUCT/terminal calculations
and encodes leaf boards as one NumPy batch. Six GPUs still serve the same
champion, 512-simulation budget and promotion rule. Search workers use one Torch
CPU thread; learning uses its existing settings. Worker logs record the search
source hash, games/minute, searched positions, simulations and network batch
counts; arena reports include timing and search statistics for both models.

On Weatherwax, interleaved 64-position tests measured median search latency of
8.93 s before versus 6.11 s after (1.46× throughput). A separate 16-game test at
512 simulations/move took 247.65 s versus 202.52 s (1.22× games/minute), producing
identical states, policies and outcomes. These are short tests alongside the
active run; they do not establish sustained six-GPU throughput or stronger play.
The workload still spends substantial time in CPU tree search, and workers can
finish their game batches at different times. GPU utilization alone is not a
strength metric. Evidence and rollback source copies are on Weatherwax under
`reports/throughput_20261117`.

### Native search and CUDA graphs

Weatherwax workers now default to a compiled C++ tree kernel plus CUDA graph
replay for inference. Build the optional kernel before launching workers:

```bash
python -m src.mcts.build_native
```

The compiler is the installed `g++`; no Python dependency was added. Compilation
disables fast math and floating-point contraction to preserve Python's PUCT
arithmetic. Neural weights, FP32 inference, exploration RNG, 512 simulations,
replay capacity, learner settings and the promotion gate stay unchanged. The
Kaggle submission/export path continues to use its existing standalone agent.

Graph replay follows PyTorch's [CUDA graph capture guidance](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs):
warm up on a side stream, retain input/output buffers and copy new inputs before
replay. Each search object caches a graph for each batch size it encounters;
production generation is bounded to 64 simultaneous games. The 64-game GPU
pilot reserved about 2.5 GB, within Weatherwax's 8 GB devices.

Native/Python visit distributions matched exactly in CPU and GPU checks. The
16-game reference batch produced the same 926 states/policies/outcomes
byte-for-byte in 15.65 s, versus the earlier Python run's 202.52 s. These are
short trials with different background loads, not a measured sustained speedup
for the whole training loop. A full 64-game production-shaped pilot completed
in 51.27 s. Evidence, test results, deployment hashes and live GPU samples are
under `reports/native_20261117` on Weatherwax. Check worker-start records for
`native_mcts`, `cuda_graphs` and `native_binary_sha256` to verify activation.

For a deliberate Python-backend comparison or rollback, launch with both
`CONNECTX_NATIVE_MCTS=0` and `CONNECTX_CUDA_GRAPHS=0`. To validate the optional
backend, run `python -m pytest tests/test_native_mcts.py -q`; its CUDA test needs
a GPU. The compiler-free inference/export path does not load the native library.

The initial 100-iteration limit was reached at 04:22 GMT on 17 November, after
35,072 saved self-play games in this lineage. At 07:13 GMT the existing run was
resumed from iteration 101 with a new limit of 1,000 iterations. Only the run
limit changed; champion, replay, learner settings and six-GPU allocation were
preserved. The extension and startup verification are recorded on Weatherwax
under `reports/extension_20261117`.

### Persistent learner, 17 November

From iteration 183, the six-GPU run keeps a separate `learner.pt`. Every cycle
continues its weights, Adam moments, update count and Torch shuffle RNG, even
when the candidate fails promotion. Self-play uses the latest learner; `best.pt`
is a separate champion and only changes after the existing promotion gate passes.
Recovery restores both from the last committed iteration. The learner was seeded
from a four-round pilot trained from the retained champion with clean Adam state.

The policy target is now 100% MCTS visit probabilities. The value target is the
terminal outcome from each recorded player's perspective: win +1, loss -1,
draw 0, with no intermediate rewards or discount. The loss is policy
cross-entropy plus value MSE. Teacher predictions no longer dilute either target.
BatchNorm running statistics remain frozen; Adam uses learning rate `8e-5`,
weight decay `1e-5` and gradient clipping at 1.0.

Replay holds 64 saved shards, or 4,096 games including their mirrored positions,
with one training epoch per cycle. This processes approximately as many samples
per cycle as the previous 16-shard, four-epoch setting, with a broader replay
window. Six GPUs generate 384 new games and share the arena each cycle; the
short optimization phase runs on GPU 0. Arena workers now partition one common
set of 64 distinct openings, removing duplicate openings between workers.

The controlled pilot compared four training rounds and 128 arena games per
setting, with common held-out games and arena openings:

| Protocol | Score vs frozen champion | Paired 95% interval | Held-out policy CE | Held-out value MSE |
|---|---:|---:|---:|---:|
| Original reset/blended targets, `8e-5` | 48.83% | 44.14–53.52% | 1.02995 | 0.06125 |
| Persistent/raw targets, `3e-5` | 51.17% | 46.88–55.47% | 1.02724 | 0.05407 |
| Persistent/raw targets, `8e-5` (selected) | 53.13% | 48.05–58.20% | 1.02531 | 0.05241 |

None passed the promotion gate. These short trials support the configuration
choice but do not establish a stronger champion or a Kaggle improvement.
Pilot evidence, recovery tests, source backups and deployment hashes are on
Weatherwax under `reports/persistent_20261117`. Historical blended-target loss
and new raw-target loss are different objectives and should not be compared
as one continuous loss curve.

For a new persistent run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m scripts.selfplay_run \
  --output runs/new_persistent --seed 5101 --lr 0.00008 --iterations 1000 \
  --gpus 0,1,2,3,4,5 --persistent-learner --epochs 1 --replay-shards 64
```

Add `--resume` with the same settings to recover it. The earlier Weatherwax continuation retained its game history when the
iteration limit changed. The
watchdog restarts exited jobs; it does not detect a live but stalled process or
restart itself after a server reboot.

## Separate Double DQN implementation

The Q-network, zero-sum TD update, replay and training loop are implemented
directly in PyTorch in `src/training/double_dqn.py` and `scripts/double_dqn.py`.
They reuse the inherited Connect Four environment and use its depth-two minimax
agent as an evaluation opponent. See [the evaluation plan](../coursework/evaluation-plan.md).

Run from the source root after installing the project dependencies:

```sh
python -m scripts.double_dqn train --method ddqn --games 5000 --decay-games 4000 \
  --seed 51 --output experiments/ddqn-51 --device cuda
python -m scripts.double_dqn evaluate --checkpoint experiments/ddqn-51/game-000000.pt \
  --opponent minimax --pairs 64 --seed 80421 --output experiments/ddqn-51/before.json
python -m scripts.double_dqn evaluate --checkpoint experiments/ddqn-51/game-005000.pt \
  --opponent minimax --pairs 64 --seed 80421 --output experiments/ddqn-51/after.json
python -m pytest tests/test_double_dqn.py -q
```

Use `--method dqn` with a separate output directory for the matched control.
Training saves an initial checkpoint, periodic and final inference snapshots,
and JSONL progress. The Weatherwax development pilot used the settings above
with snapshots every 1,000 games and 32 opening pairs per opponent. See
`pilot-evidence/ddqn-pilot-51/` in the coursework source ZIP for the original
checkpoints, match traces and runner. To replay the traces and regenerate the
summary and plot from the source root:

```sh
PYTHONPATH=. python pilot-evidence/ddqn-pilot-51/reproduce_summary.py \
  pilot-evidence/ddqn-pilot-51 output-figures
```

Create the output directory before running the replay script.

## Matched Q-learning and tactical study

The archive's `study-evidence/dqn-study/` contains all six runs, fixed-opening
evaluations and a reproducible analysis. The existing seed-51 Double DQN
checkpoint is reused. Each other run uses the same training command above
with `--save-every 1000` and its recorded method and seed. The study evaluates
checkpoints on two disjoint sets of openings and saves complete game traces.

To repeat one checkpoint series from the archived weights, run from the archive
root with the project dependencies installed:

```sh
python -m scripts.dqn_study \
  --training study-evidence/dqn-study/runs/ddqn-51/training \
  --excluded-pilot-openings study-evidence/dqn-study/excluded-pilot-openings.json \
  --output experiments/study-ddqn-51
```

Omit `--training` to repeat the tactics-only control. To replay all archived
games and regenerate the study figure, create an output directory and run:

```sh
PYTHONPATH=. python study-evidence/dqn-study/reproduce_summary.py \
  study-evidence/dqn-study output-figures
```

See [the two demo replays](../coursework/demo-replay.md) for exact positions and
match histories to use in the agent-performance video.

## Move-matched tactical follow-up

The follow-up uses Double DQN with a partially supervised loss on exact
immediate wins, draws and opponent winning replies. The control and treatment
use fresh paired initialisations, 80,000 moves and 19,745 updates. Source paths
are recorded in the coursework archive validation. To run one seed with the
same settings:

~~~sh
python -m scripts.tactical_dqn --variant tactical --seed 61 \
  --moves 80000 --decay-moves 64000 --save-every 20000 \
  --device cuda --output experiments/tactical-61
~~~

Use `--variant plain` for the matched control. The frozen evaluation data,
checkpoints, run logs and complete traces are under
`study-evidence/tactical-followup/` in the code and evidence archive. From the
archive root, create an output directory and replay the evidence:

~~~sh
PYTHONPATH=. python study-evidence/tactical-followup/reproduce_summary.py \
  study-evidence/tactical-followup output-figures
~~~

The budget, opening lists, and primary comparison are in the frozen pack.
`scripts/verify_endgame_positions.py` independently re-solves the labelled
endgames using a column grid. `scripts/oracle_reference.py` reproduces the
inherited model's fixed-search comparison. `scripts/tactical_study.py` and
`scripts/learned_lookahead.py` regenerate the match and oracle traces from the
archived checkpoints. The shell launchers record the Weatherwax job mapping
and accept a Python executable as their first argument.

To render the captioned agent demonstration, install Pillow and FFmpeg and
run from the source root:

~~~sh
python -m scripts.render_agent_demo --output coursework/agent-performance-demo.mp4
~~~
