# Connect Four coursework draft

The draft uses the supplied Bath RL project template and an unchanged
`rl_project.sty`. The build command below produces `build/report.pdf` from
the repository. The local `report.pdf` is the compiled draft, and the local
`connectx-coursework-draft.zip` contains LaTeX, figures, compact evidence and
validation. The revised report has
**6 main-text pages** and **30 pages total**, within the seven-page main limit.
The template specifies US Letter and a 139.7 mm text column.

The main paper centres the independently implemented Q-learning experiments. The inherited search-agent training and all prior diagnostic figures remain in the appendices. This is still a draft:
member details, personal reflections, individual GenAI disclosures and
the final submission checks remain to be completed. Section 3.2 documents the separate
Double DQN implementation. The [evaluation plan](evaluation-plan.md) records
the controlled comparison and submission checklist.

## Updated evidence

Training completed at **08:17 GMT, 21 November 2026** after **1,000,000 self-play games**. Panel A covers candidates **0–2430** through raw iteration 2,613. The persistent phase recorded **30** promotions and ended with candidate **2353** as champion.

Benchmark snapshot **21 November 2026, 05:59 GMT**.

| New candidate | W/D/L | Score | Paired 95% interval |
|---|---|---|---|
| 2223 | 69/13/46 | 59.0% | 52.3–65.6% |
| 2309 | 67/13/48 | 57.4% | 51.6–63.3% |
| 2353 | 69/16/43 | 60.2% | 54.7–66.0% |

All 29 checkpoints faced the same frozen original using identical
agents, 1.9-second move budgets and 64 shared openings in both seats.
Candidate 1667 has the highest score at 62.1% (56.2--68.4%).
All 3,712 games were independently replayed. Training continued during
the CPU tests, and its CPU affinity was restored afterward.
The weighted absolute-error trend is +0.31 percentage points per 100 candidates (95% interval +0.08 to +0.52).

At 15:20 GMT on 18 November, training resumed after raw iteration 1,000 with the saved learner
and Adam state. Only the run limit changed, to 2,000 raw iterations. The first
resumed update preserved the optimiser step count, and all six GPU workers and
the existing watchdog were verified. Panel A includes the resumed run; its
checkpoint and optimiser continuity were checked. Continuation records are in
`training-evidence/connectx-extension-20261118-2000/`.

The draft also records the team-reported first-place Kaggle ranking on
19 November 2026. The exact score, leaderboard and submission identity are
pending confirmation and marked explicitly in the paper. This result is separate
from the local benchmark percentages above.

## Double DQN pilot

The first run used seed 51 and 5,000 self-play games on Weatherwax. Six
checkpoints were evaluated against uniform random play and depth-two minimax
on the same 32 openings in both seats. The report and `figures/ddqn-pilot.pdf`
show the development curves. `evidence/ddqn-pilot-summary.json` records
scores, paired intervals and checkpoint hashes. The later matched study uses independent additional seeds.

## Matched Q-learning study

DQN and Double DQN used three matched initialisations, seeds 51–53, for
5,000 games each. Curves use 32 openings, with a separate 64-opening final
evaluation. Both sets exclude the pilot boards. The final comparison includes
greedy networks with and without a tactical filter, before and after training,
and a tactics-only control. All 11,008 game traces were replayed.
The source ZIP includes the training and evaluation launchers, checkpoints,
configuration, results, and the replay script under
`study-evidence/dqn-study`.

## Move-matched tactics and lookahead follow-up

Five paired initialisations compare plain and tactically supervised Double DQN
with 80,000 moves and 19,745 updates in each run. The primary test uses 64
new openings. An independently verified 288-position endgame set distinguishes
winning moves, defensive choices and positions that require multi-step reasoning.
A separate 24-opening experiment compares guarded greedy Q-values with
two-ply search using initial and trained weights.

The code and evidence archive contains the frozen test packs, 50 checkpoints,
training logs, match and position traces and the replay script under
`study-evidence/tactical-followup/`. The replay verified 10,400 primary-study
and 2,496 lookahead match traces. Report section 4 and Appendix J summarise
the results. See the [captioned agent demo](agent-performance-demo.mp4)
and [group speaker plan](presentation-plan.md).

## Matched horizon, search and defence controls

Each contrast now uses ten seed pairs. Five additional seed blocks on
Weatherwax used one plain one-step run, one tactical one-step run and one
plain up-to-three-step run per seed. The shared plain control keeps the
addition to 15 runs. Each run uses 80,000 moves and 19,745 updates. We
verified 13,056 match traces. Rates below show mean ±
sample SD across training seeds on the same frozen test sets.

- **Matched search.** On 128 opening pairs against depth-three minimax,
  initial Q scored 16.4 ± 4.5% and trained Q scored
  30.4 ± 1.7% in the same tree. The fixed heuristic scored
  39.5%.
- **Defence.** On 1,797 common early positions, plain networks exposed a
  winning reply at 38.3 ± 1.6% and tactical networks at
  42.9 ± 1.4%.
- **Horizon.** Against depth-two minimax on the same 128 opening pairs,
  one step scored 10.9 ± 0.9% and up to three steps scored
  9.7 ± 0.9%.

The [matched controls summary](evidence/q-controls-summary.json) gives
the seed-level results, paired intervals, realised horizons and the new
seed cohort on its own. Appendix K starts on page 25.
The source archive contains checkpoints, traces, launchers and checksums
under `study-evidence/reviewer-controls/`. The extension plan and its raw
outputs are under `extension/` there. To reproduce the summary from an
extracted code archive, run this command at the repository root with the
project dependencies installed.

```sh
PYTHONPATH=. python study-evidence/reviewer-controls/reproduce_summary.py --output reproduced-controls.json
```

## Figures and analysis

Figures 2 and 3 are arranged side by side. Run `make_inline_figures.py` with
the same plotting environment to regenerate the compact PDF and PNG versions
from the frozen summaries.

The continued-training figure shows changing-champion arenas and fixed-original
benchmarks. Both top axes show cumulative completed self-play games, starting
at 66,944 for candidate 0. Full cycles add 384 games and the last cycle adds
320 games. Panel (b) fits a line by minimizing weighted mean absolute error.
Weights are inverses of the checkpoint score standard errors. Joint opening-pair
bootstrap resampling keeps the weights fixed and preserves correlations between
checkpoint scores. The slope and interval are in `evidence/benchmark-linear-fit.json`.
The appendix also includes the following:

- Appendix A (pages 9–13): game/value conventions, network equations,
  search, reflection, replay, gradients, Adam and paired evaluation.
- Appendix B (pages 14–15): scoped literature review.
- Appendices C–F: configuration, benchmark counts, training losses and remaining work.
- Appendix G (pages 19–22): W/D/L and player assignment, opening-level scores,
  paired checkpoint differences, and held-out pilot losses.

`evidence/results.json` contains plotted summaries, pair scores and source hashes.
`evidence/benchmark-details.json` contains both player outcomes and hashes of
232 worker files. Bootstrap diagnostics use 50,000 opening-ID resamples with
seed 20261118. Intervals are pointwise and do not adjust for repeated selection.
The pilot remains one continuation per setting on 384 shared withheld games.

`validation.json` records data checks, page counts, source hashes and visual review.
Hashes for earlier runs use the source paths recorded at the time. The full
checkpoints and traces are in the local code and evidence archive, the workspace
evidence directories and on Weatherwax.

## Build

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build report.tex
```

The included vector PDFs allow compilation without Python or Figma. BibTeX uses
`references.bib` and `unsrtnat`; the style requires `datetime` and `fmtcount`.
To regenerate all plots, activate the Connect Four environment and run this
command from `coursework/`.

```sh
python make_figures.py
```

## Diagram sources

Both diagrams use subtle grey rounded backgrounds; plots remain white.
The vector PDF/SVG/PNG assets are in `figures/`, with equation sources in
`figures/network-math/` and `figures/training-math/`.
Editable Figma designs:
[inference](https://www.figma.com/design/hPdudsaS8YffnWtcW7PVI1?node-id=2-16) and
[persistent training](https://www.figma.com/design/hPdudsaS8YffnWtcW7PVI1?node-id=2-110).
Both diagrams are 139.7 mm wide and 69.85 mm high before captions.

To regenerate the training diagram's equation assets with XeLaTeX, dvisvgm and STIX:

```sh
python3 figures/network-math/render.py figures/training-math/training-math.tex
```

The renderer's default invocation regenerates the inference equations.
